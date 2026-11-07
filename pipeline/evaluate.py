"""Evaluate a detector on the asli test set -> bench/out/<name>.json

  python evaluate.py asli                       # our model (models/asli.pt)
  python evaluate.py hf:<model_id> [--max N]    # an open Hugging Face detector (baseline)

Real negatives: FLEURS test speech (10 training languages + Odia/Assamese never seen).
Fakes: MMS-TTS and Microsoft neural voices (training generators, unseen speakers/sentences),
Google TTS and XTTS-v2 voice clones (generators NEVER seen in training).
Channels: clean, phone (G.711 8 kHz), opus (12 kbps, WhatsApp-like), mp3 (32 kbps), noise (10 dB).
For every slice: EER, AUROC, and TPR / FPR at the operating threshold chosen on dev.
Shortcut test: false-alarm rate on REAL test speech passed through the exact MP3 encoder
the Microsoft voices come out of (`edgemp3`).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import DATA  # noqa: E402

CONDS = ["clean", "phone", "opus", "mp3", "noise"]
OUT = os.path.join(os.path.dirname(__file__), "..", "bench", "out")
SEEN_GENS, UNSEEN_GENS = ["mms", "edge"], ["gtts", "xtts"]


def load_test():
    rows = []
    for group in ("real", "fake"):
        d = os.path.join(DATA, "crops", group, "test")
        idx = list(csv.DictReader(open(os.path.join(d, "index.csv"))))
        arrs = {c: np.load(os.path.join(d, f"{c}.npy"), mmap_mode="r") for c in CONDS + ["edgemp3"]}
        for r in idx:
            rows.append((arrs, int(r["row"]), int(r["label"]), r["gen"], r["lang"]))
    return rows


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def metrics(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return {"n": int(len(y)), "flag_rate": float((s >= thr).mean())}
    return {"n": int(len(y)), "eer": round(eer(y, s), 4), "auroc": round(float(roc_auc_score(y, s)), 4),
            "tpr": round(float((s[y == 1] >= thr).mean()), 4), "fpr": round(float((s[y == 0] >= thr).mean()), 4)}


class Ours:
    def __init__(self, path):
        from asli.model import AsliNet
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.net = AsliNet().to(self.dev).eval()
        self.net.load_state_dict(torch.load(path, map_location=self.dev))

    @torch.no_grad()
    def __call__(self, wavs):
        return torch.sigmoid(self.net(torch.from_numpy(wavs).to(self.dev))).float().cpu().numpy()


class HF:
    """Open-source audio deepfake classifiers (wav2vec2-style) from the HF hub."""

    def __init__(self, mid):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.fe = AutoFeatureExtractor.from_pretrained(mid)
        self.m = AutoModelForAudioClassification.from_pretrained(mid).eval()
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.m.to(self.dev)
        labels = {i: l.lower() for i, l in self.m.config.id2label.items()}
        fake = [i for i, l in labels.items() if any(k in l for k in ("fake", "spoof", "synthetic", "ai", "generated"))]
        self.fake_idx = fake[0] if fake else 1
        self.labels = labels

    @torch.no_grad()
    def __call__(self, wavs):
        x = self.fe(list(wavs), sampling_rate=16000, return_tensors="pt", padding=True)
        logits = self.m(**{k: v.to(self.dev) for k, v in x.items()}).logits
        return torch.softmax(logits.float(), -1)[:, self.fake_idx].cpu().numpy()


def run(det, rows, cond, bs):
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        x = np.stack([r[0][cond][r[1]] for r in chunk]).astype(np.float32) / 32767
        out.append(det(x))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("--max", type=int, default=0, help="subsample per class (for slow baselines)")
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    rows = load_test()
    if args.max:
        rng = np.random.default_rng(0)
        keep = []
        for lab in (0, 1):
            for g in sorted({r[3] for r in rows if r[2] == lab}):
                sub = [r for r in rows if r[2] == lab and r[3] == g]
                k = min(len(sub), args.max if lab == 1 else args.max * 2)
                keep += [sub[j] for j in rng.choice(len(sub), k, replace=False)]
        rows = keep
    if args.which == "asli":
        det, name, bs = Ours(os.path.join(os.path.dirname(__file__), "..", "models", "asli.pt")), "asli", 128
        thr = args.thr if args.thr is not None else json.load(open(os.path.join(os.path.dirname(__file__), "..", "models",
                                                                                 "asli.threshold.json")))["threshold"]
    else:
        mid = args.which[3:]
        det, name, bs = HF(mid), "baseline_" + mid.replace("/", "__"), 16
        thr = 0.5
    y = np.array([r[2] for r in rows])
    gens = np.array([r[3] for r in rows])
    langs = np.array([r[4] for r in rows])
    res = {"model": args.which, "threshold": thr, "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
           "conditions": {}}
    if isinstance(det, HF):
        res["label_map"] = det.labels
        res["params"] = f"{sum(p.numel() for p in det.m.parameters()) / 1e6:.0f}M"
    for cond in CONDS:
        s = run(det, rows, cond, bs)
        c = {"all": metrics(y, s, thr), "by_generator": {}, "by_language": {}}
        for g in sorted(set(gens[y == 1])):
            m = (y == 0) | (gens == g)
            c["by_generator"][g] = metrics(y[m], s[m], thr)
        for l in sorted(set(langs)):
            m = langs == l
            c["by_language"][l] = metrics(y[m], s[m], thr)
        seen = (y == 0) & ~np.isin(langs, ["or_in", "as_in"]) | np.isin(gens, SEEN_GENS)
        c["seen_generators"] = metrics(y[(y == 0) | np.isin(gens, SEEN_GENS)], s[(y == 0) | np.isin(gens, SEEN_GENS)], thr)
        c["unseen_generators"] = metrics(y[(y == 0) | np.isin(gens, UNSEEN_GENS)], s[(y == 0) | np.isin(gens, UNSEEN_GENS)], thr)
        ul = np.isin(langs, ["or_in", "as_in"])
        c["unseen_languages"] = metrics(y[ul], s[ul], thr)
        res["conditions"][cond] = c
        a = c["all"]
        print(f"{name:40s} {cond:6s} EER {a.get('eer')} AUROC {a.get('auroc')} TPR {a.get('tpr')} FPR {a.get('fpr')} | "
              f"unseen gens EER {c['unseen_generators'].get('eer')} | unseen langs EER {c['unseen_languages'].get('eer')}",
              flush=True)
    s_real_mp3 = run(det, [r for r in rows if r[2] == 0], "edgemp3", bs)
    s_real = run(det, [r for r in rows if r[2] == 0], "clean", bs)
    res["shortcut_test"] = {"real_clean_fpr": float((s_real >= thr).mean()),
                            "real_through_edge_mp3_fpr": float((s_real_mp3 >= thr).mean())}
    print("shortcut test:", res["shortcut_test"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
