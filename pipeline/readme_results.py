"""Fill the <!-- RESULTS --> block of README.md from docs/data/report.json."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = json.load(open(os.path.join(ROOT, "docs", "data", "report.json")))
pct = lambda x: "–" if x is None else f"{x * 100:.1f}%"
CH = {"clean": "clean", "phone": "phone line (G.711, 8 kHz)", "opus": "WhatsApp-like Opus 12 kbps", "mp3": "MP3 32 kbps",
      "noise": "pink noise, 10 dB SNR"}


def main():
    ch = "\n".join(f"| {CH[k]} | {pct(v['eer'])} | {pct(v['auroc'])} |" for k, v in R["channels"].items())
    gens = "\n".join(f"| {g['label']} | {'trained on' if g['seen'] else '**never seen**'} | {pct(g['eer'])} | {pct(g['auroc'])} |"
                     for g in R["generators"].values())
    langs = " · ".join(f"{l['name']} {pct(l['eer']) if l['eer'] is not None else 'flag ' + pct(l['flag_rate'])}"
                       + (" (unseen)" if l["unseen"] else "") for l in R["languages"].values())
    base = "\n".join(f"| {'**' + b['name'] + '**' if b.get('us') else b['name']} | {b['params']} | {pct(b['clean'])} | "
                     f"{pct(b['phone'])} | {pct(b['opus'])} | {pct(b['unseen'])} |" for b in R["baselines"])
    others = [b for b in R["baselines"] if not b.get("us")]
    baseline_note = ""
    if others:
        lo = min(others, key=lambda b: b["phone"]); hi = max(others, key=lambda b: b["clean"])
        baseline_note = (f"The open detectors (trained mostly on English anti-spoofing data) range from AUROC {pct(lo['clean'])} to "
                         f"{pct(hi['clean'])} on clean Indian-language speech and fall to {pct(lo['phone'])} over a phone line, i.e. worse than "
                         f"a coin flip. On the XTTS voice clones every detector here, asli included, is weak: asli "
                         f"{pct(R['generators']['xtts']['auroc'])} AUROC.")
    S = R["shortcut"]
    res = f"""### Unseen speakers, all synthetic engines pooled

| channel | equal error rate | AUROC |
|---|---|---|
{ch}

### By speech engine (clean audio)

| engine | in training? | EER | AUROC |
|---|---|---|---|
{gens}

### By language (clean, EER)

{langs}

### Against open-source detectors (same test set, AUROC)

| detector | params | clean | phone | Opus | unseen engines |
|---|---|---|---|---|---|
{base}

{baseline_note}

### Shortcut test

False alarms on real test speech: **{pct(S['real_clean_fpr'])}** as recorded, **{pct(S['real_through_edge_mp3_fpr'])}** after passing
through the exact MP3 encoder one training engine uses. If the model had learned “MP3 artefacts = fake”, the second number would jump."""
    p = os.path.join(ROOT, "README.md")
    s = open(p).read()
    if "<!-- RESULTS -->" in s:
        open(p, "w").write(s.replace("<!-- RESULTS -->", res))
        print("README filled")


if __name__ == "__main__":
    main()
