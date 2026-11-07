"""Train the asli detector.

  python train.py [--epochs 14] [--out ../models/asli.pt]

Training generators: MMS-TTS (VITS) and Microsoft neural voices (edge-tts).
Google TTS and XTTS-v2 voice clones are NEVER used here: they are the held-out test.
Batches are balanced real/fake and, within fake, balanced across the two generators.
Each crop is used either clean or through its pre-computed lossy channel (phone,
Opus, MP3, AAC, noise), plus on-the-fly time shift and light noise.
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asli.model import AsliNet  # noqa: E402
from common import DATA  # noqa: E402

TRAIN_GENS = {"real", "mms", "edge"}


def load_split(split):
    parts = []
    for group in ("real", "fake"):
        d = os.path.join(DATA, "crops", group, split)
        if not os.path.exists(os.path.join(d, "index.csv")):
            continue
        idx = list(csv.DictReader(open(os.path.join(d, "index.csv"))))
        arrs = {v: np.load(os.path.join(d, f"{v}.npy"), mmap_mode="r")
                for v in ("clean", "codec") if os.path.exists(os.path.join(d, f"{v}.npy"))}
        for r in idx:
            if r["gen"] in TRAIN_GENS:
                parts.append((arrs, int(r["row"]), int(r["label"]), r["gen"], r["lang"]))
    return parts


def batches(items, bs, rng, steps):
    real = [x for x in items if x[2] == 0]
    fake_by_gen = {}
    for x in items:
        if x[2] == 1:
            fake_by_gen.setdefault(x[3], []).append(x)
    gens = sorted(fake_by_gen)
    for _ in range(steps):
        pick = [rng.choice(real) for _ in range(bs // 2)]
        pick += [rng.choice(fake_by_gen[gens[i % len(gens)]]) for i in range(bs - bs // 2)]
        wav = np.empty((bs, 64000), np.float32)
        y = np.empty(bs, np.float32)
        for i, (arrs, row, label, gen, lang) in enumerate(pick):
            v = "codec" if ("codec" in arrs and rng.random() < 0.5) else "clean"
            x = arrs[v][row].astype(np.float32) / 32767
            x = np.roll(x, rng.randrange(-8000, 8000))
            if rng.random() < 0.15:  # light broadband noise, 10-35 dB SNR
                snr = rng.uniform(10, 35)
                n = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(64000).astype(np.float32)
                x = x + n * np.sqrt((x ** 2).mean() / 10 ** (snr / 10))
            wav[i] = x * rng.uniform(0.3, 1.0)
            y[i] = label
        yield torch.from_numpy(wav), torch.from_numpy(y)


def prefetch(gen, n=4):
    """Build batches in a background thread; any error there is re-raised here instead of hanging."""
    q = queue.Queue(n)

    def run():
        try:
            for item in gen:
                q.put(item)
            q.put(None)
        except BaseException as e:  # noqa: BLE001
            q.put(e)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def eer(scores, labels):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(roc_auc_score(labels, scores)), float(thr[i])


@torch.no_grad()
def score(net, items, variant, dev, bs=128):
    net.eval()
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = np.stack([(c[0][variant] if variant in c[0] else c[0]["clean"])[c[1]] for c in chunk]).astype(np.float32) / 32767
        out.append(torch.sigmoid(net(torch.from_numpy(x).to(dev))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "asli.pt"))
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    train, val = load_split("train"), load_split("dev")
    val_full = val
    val = random.Random(1).sample(val, min(2500, len(val)))  # per-epoch model selection on a fixed subset
    print(f"train {len(train)} (fake {sum(x[2] for x in train)}) | dev {len(val)} (fake {sum(x[2] for x in val)})",
          flush=True)
    net = AsliNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    best, log = 9.0, []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for step, (x, y) in enumerate(prefetch(batches(train, args.bs, rng, args.steps))):
            x, y = x.to(dev), y.to(dev)
            loss = F.binary_cross_entropy_with_logits(net(x, specaug=True), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if step % 100 == 0:
                print(f"  step {step} loss {tot / (step + 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        labels = np.array([c[2] for c in val])
        res = {}
        for v in ("clean", "codec"):
            res[v] = eer(score(net, val, v, dev), labels)[:2]
        m = (res["clean"][0] + res["codec"][0]) / 2
        log.append({"epoch": ep + 1, "loss": tot / args.steps, "dev_eer_clean": res["clean"][0],
                    "dev_eer_codec": res["codec"][0], "dev_auc_clean": res["clean"][1]})
        print(f"ep {ep + 1:2d} loss {tot / args.steps:.4f} dev EER clean {res['clean'][0]:.4f} "
              f"codec {res['codec'][0]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if m < best:
            best = m
            torch.save(net.state_dict(), args.out)
    json.dump(log, open(args.out.replace(".pt", ".trainlog.json"), "w"), indent=1)
    # operating threshold = equal-error point on dev (clean + lossy channels pooled)
    net.load_state_dict(torch.load(args.out, map_location=dev))
    labels = np.array([c[2] for c in val_full])
    s = np.concatenate([score(net, val_full, "clean", dev), score(net, val_full, "codec", dev)])
    lab = np.concatenate([labels, labels])
    e, auc, thr = eer(s, lab)
    json.dump({"threshold": thr, "dev_eer_pooled": e, "dev_auroc_pooled": auc},
              open(args.out.replace(".pt", ".threshold.json"), "w"), indent=1)
    print("BEST dev mean EER", best, "| pooled dev EER", e, "threshold", thr, "->", args.out, flush=True)


if __name__ == "__main__":
    main()
