"""Assemble docs/data/report.json and demo sample clips for the asli website.

Samples: real FLEURS test speech (CC-BY 4.0) and MMS-TTS synthetic speech (CC-BY-NC 4.0)
from the held-out test split. Voice clones and commercial-TTS audio are not published.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import DATA, LANGS, HELDOUT_LANGS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "bench", "out")
DOCS = os.path.join(ROOT, "docs")
GEN_LABEL = {"mms": "Meta MMS-TTS", "edge": "Microsoft neural voices", "gtts": "Google TTS", "xtts": "XTTS-v2 voice clone"}


def samples():
    os.makedirs(os.path.join(DOCS, "samples"), exist_ok=True)
    out = []
    for group, gen in (("real", "real"), ("fake", "mms")):
        d = os.path.join(DATA, "crops", group, "test")
        idx = [r for r in csv.DictReader(open(os.path.join(d, "index.csv"))) if r["gen"] == gen]
        arr = np.load(os.path.join(d, "clean.npy"), mmap_mode="r")
        for lang in ("hi_in", "ta_in", "bn_in"):
            r = next(x for x in idx if x["lang"] == lang)
            fn = f"{gen}_{lang}.wav"
            sf.write(os.path.join(DOCS, "samples", fn), arr[int(r["row"])].astype(np.float32) / 32767, 16000, subtype="PCM_16")
            out.append({"file": fn, "label": LANGS[lang][0], "real": group == "real"})
    json.dump(out, open(os.path.join(DOCS, "samples", "samples.json"), "w"))
    return out


def main():
    ours = json.load(open(os.path.join(OUT, "asli.json")))
    thr = json.load(open(os.path.join(ROOT, "models", "asli.threshold.json")))
    C = ours["conditions"]
    channels = {k: {"eer": v["all"]["eer"], "auroc": v["all"]["auroc"]} for k, v in C.items()}
    gens = {g: {"label": GEN_LABEL[g], "eer": v["eer"], "auroc": v["auroc"], "seen": g in ("mms", "edge")}
            for g, v in C["clean"]["by_generator"].items()}
    langs = {}
    for l, v in C["clean"]["by_language"].items():
        langs[l] = {"name": LANGS[l][0], "n": v["n"], "eer": v.get("eer"), "flag_rate": v.get("flag_rate"),
                    "unseen": l in HELDOUT_LANGS}
    baselines = [{"name": "asli (ours)", "params": "0.6M", "clean": C["clean"]["all"]["auroc"], "phone": C["phone"]["all"]["auroc"],
                  "opus": C["opus"]["all"]["auroc"], "unseen": C["clean"]["unseen_generators"]["auroc"],
                  "clones": C["clean"]["by_generator"]["xtts"]["auroc"], "us": True}]
    for f in sorted(glob.glob(os.path.join(OUT, "baseline_*.json"))):
        b = json.load(open(f))
        bc = b["conditions"]
        baselines.append({"name": b["model"][3:], "params": b.get("params", ""), "clean": bc["clean"]["all"]["auroc"],
                          "phone": bc["phone"]["all"]["auroc"], "opus": bc["opus"]["all"]["auroc"],
                          "unseen": bc["clean"]["unseen_generators"]["auroc"], "clones": bc["clean"]["by_generator"]["xtts"]["auroc"]})
    n_real, n_fake = ours["n_real"], ours["n_fake"]
    train_counts = {}
    for g in ("real", "fake"):
        for s in ("train", "dev", "test"):
            p = os.path.join(DATA, "crops", g, s, "index.csv")
            if os.path.exists(p):
                train_counts[(g, s)] = sum(1 for _ in open(p)) - 1
    total = sum(train_counts.values())
    rep = {
        "kpis": [
            {"v": f"{C['clean']['all']['eer'] * 100:.1f}%", "l": "equal error rate on speakers it never heard (clean)"},
            {"v": f"{C['phone']['all']['eer'] * 100:.1f}%", "l": "through an 8 kHz phone line"},
            {"v": f"{C['clean']['by_generator']['gtts']['auroc'] * 100:.0f}%", "l": "AUROC on Google TTS, an engine it never saw"},
            {"v": f"{C['clean']['by_generator']['xtts']['auroc'] * 100:.0f}%", "l": "AUROC on XTTS-v2 voice clones it never saw: not solved"},
        ],
        "channels": channels, "generators": gens, "languages": langs, "baselines": baselines,
        "shortcut": ours["shortcut_test"], "threshold": thr["threshold"],
        "lang_sub": "equal error rate per language on clean audio; dashed = languages never used in training (only MMS fakes exist for them, so a flag rate is shown when one class is missing)",
        "eval_note": (f"Test set: {n_real:,} real clips from FLEURS speakers never seen in training and {n_fake:,} synthetic clips. "
                      "Equal error rate = the point where false alarms equal misses. Operating threshold picked on the dev set."),
        "data": [{"v": f"{train_counts.get(('real', 'train'), 0):,}", "l": "real training clips (FLEURS, 10 languages)"},
                 {"v": f"{train_counts.get(('fake', 'train'), 0):,}", "l": "synthetic training clips (2 engines)"},
                 {"v": f"{n_real + n_fake:,}", "l": "test clips × 5 channels"},
                 {"v": "4", "l": "speech engines (2 held out)"}],
        "samples": samples(),
    }
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    json.dump(rep, open(os.path.join(DOCS, "data", "report.json"), "w"), indent=1)
    print(json.dumps(rep["kpis"], indent=1))


if __name__ == "__main__":
    main()
