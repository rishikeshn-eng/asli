"""asli command line.

  asli voice_note.ogg           # is this voice real or synthetic?
  asli --json call.mp3
  asli *.wav                    # several files

Exit code: 0 real, 1 uncertain, 2 synthetic.
"""
import argparse
import json
import sys

RED, YEL, GRN, DIM, B, R = "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def bar(p, width=30):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="asli", description="Detect AI-generated (cloned / TTS) speech in Indian languages.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from .detect import Detector
    det = Detector()
    color = sys.stdout.isatty() and not a.json
    c = (lambda code, s: f"{code}{s}{R}") if color else (lambda code, s: s)
    worst = 0
    out = {}
    for f in a.files:
        r = det.analyse(f)
        out[f] = r
        worst = max(worst, {"real": 0, "uncertain": 1, "synthetic": 2}[r["verdict"]])
        if a.json:
            continue
        col = {"real": GRN, "uncertain": YEL, "synthetic": RED}[r["verdict"]]
        print(f"{c(B, f)}  {r['duration_s']}s")
        print(f"  {c(col, r['verdict'].upper())}  P(synthetic) = {r['p_synthetic']:.2f}  {c(DIM, 'threshold %.2f' % r['threshold'])}")
        for w in r["windows"]:
            print(f"  {c(DIM, '%5.1fs' % w['start_s'])} {c(RED if w['p'] >= r['threshold'] else GRN, bar(w['p']))} {w['p']:.2f}")
    if a.json:
        print(json.dumps(out, indent=1))
    return worst


if __name__ == "__main__":
    sys.exit(main())
