"""Turn raw real/fake audio into fixed 4-second training/evaluation crops.

  python prepare.py real        # all FLEURS clips
  python prepare.py fake        # all synthesised clips present on disk

Every clip, real or fake, goes through the identical chain:
  load -> 16 kHz mono -> trim leading/trailing silence -> RMS-normalise -> 4 s crop
so silence length, loudness and sample rate carry no label information.

Shortcut control (codec history). The edge-tts voices arrive as 48 kbps MP3, so a
detector could learn "MP3 artefacts = fake". For training/dev, 45% of REAL clips are
passed through the identical MP3 encode first, making codec history independent of
the label. On the test set we additionally store an `edgemp3` condition so we can
measure the false-alarm rate on real speech that went through that exact encoder.

Outputs, per group (real|fake) and split, in $ASLI_DATA/crops/<group>/<split>/:
  clean.npy        int16 [N, 64000]
  codec.npy        (train/dev) the same crop through one random lossy channel
  phone/opus/mp3/noise/edgemp3.npy   (test) fixed evaluation channels
  index.csv        one row per crop
"""
import csv
import glob
import os
import random
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

from common import DATA, FLEURS, LANGS, splits_for, read_tsv

SR, N = 16000, 64000
TEST_CONDS = ["phone", "opus", "mp3", "noise", "edgemp3"]


# ---------------------------------------------------------------- signal utils
def load16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim(x, top_db=35.0):
    hop, win = 160, 400
    if len(x) < win * 2:
        return x
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    keep = np.where(20 * np.log10(rms / rms.max()) > -top_db)[0]
    if len(keep) == 0:
        return x
    a = max(0, keep[0] * hop - 1600)
    b = min(len(x), keep[-1] * hop + win + 1600)
    return x[a:b]


def rms_norm(x, target_db=-24.0):
    r = np.sqrt((x ** 2).mean() + 1e-12)
    y = x * (10 ** (target_db / 20) / r)
    pk = np.abs(y).max()
    return y / pk * 0.99 if pk > 0.99 else y


def crop(x, rng):
    if len(x) >= N:
        s = rng.randrange(0, len(x) - N + 1)
        return x[s:s + N]
    reps = int(np.ceil(N / len(x)))  # tile short clips with a 10 ms crossfade
    fade = 160
    out = x.copy()
    for _ in range(reps):
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        out = np.concatenate([out[:-fade], out[-fade:] * (1 - ramp) + x[:fade] * ramp, x[fade:]])
    return out[:N]


def ffmpeg_roundtrip(x, enc_args, sr_in=SR, fixed=True):
    """Encode float audio with ffmpeg and decode back to 16 kHz float."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr_in), "-ac", "1", "-i", "pipe:0",
                          *enc_args, "pipe:1"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                          "pipe:1"], input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32767
    return fit(y) if fixed else y


def fit(y):
    return y[:N] if len(y) >= N else np.pad(y, (0, N - len(y)))


_SOS_PHONE = butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")


def phone(x):
    """Narrowband telephone: 8 kHz, 300-3400 Hz, G.711 mu-law 8-bit companding."""
    y = resample_poly(x, 1, 2)
    y = sosfilt(_SOS_PHONE, y)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
    c = np.round((c + 1) / 2 * 255) / 255 * 2 - 1           # 8-bit quantisation
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    return fit(resample_poly(y, 2, 1).astype(np.float32))


def pink_noise(n, rng):
    w = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    p = np.fft.irfft(f, n)
    return (p / (np.sqrt((p ** 2).mean()) + 1e-9)).astype(np.float32)


def add_noise(x, snr_db, rng):
    n = pink_noise(len(x), rng)
    px = (x ** 2).mean() + 1e-12
    return np.clip(x + n * np.sqrt(px / 10 ** (snr_db / 10)), -1, 1).astype(np.float32)


EDGE_MP3 = ["-ar", "24000", "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3"]


def random_channel(x, rng):
    k = rng.choice(["opus", "mp3", "aac", "phone", "phone_opus", "noise"])
    if k == "opus":
        br = rng.choice(["8k", "12k", "16k", "24k"])
        return ffmpeg_roundtrip(x, ["-c:a", "libopus", "-b:a", br, "-f", "ogg"]), f"opus{br}"
    if k == "mp3":
        br = rng.choice(["24k", "32k", "48k", "64k"])
        return ffmpeg_roundtrip(x, ["-c:a", "libmp3lame", "-b:a", br, "-f", "mp3"]), f"mp3{br}"
    if k == "aac":
        br = rng.choice(["24k", "32k", "48k"])
        return ffmpeg_roundtrip(x, ["-c:a", "aac", "-b:a", br, "-f", "adts"]), f"aac{br}"
    if k == "phone":
        return phone(x), "phone"
    if k == "phone_opus":
        return ffmpeg_roundtrip(phone(x), ["-c:a", "libopus", "-b:a", "12k", "-f", "ogg"]), "phone+opus12k"
    snr = rng.uniform(5, 25)
    return add_noise(x, snr, rng), f"noise{snr:.0f}dB"


def test_condition(x, cond, rng):
    if cond == "phone":
        return phone(x)
    if cond == "opus":  # WhatsApp-like voice note
        return ffmpeg_roundtrip(x, ["-c:a", "libopus", "-b:a", "12k", "-f", "ogg"])
    if cond == "mp3":
        return ffmpeg_roundtrip(x, ["-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"])
    if cond == "noise":
        return add_noise(x, 10.0, rng)
    if cond == "edgemp3":
        return ffmpeg_roundtrip(x, EDGE_MP3)
    raise ValueError(cond)


# ---------------------------------------------------------------- per-clip worker
def work(item):
    key, path, label, split = item["key"], item["path"], item["label"], item["split"]
    rng = random.Random(key)
    try:
        x = rms_norm(trim(load16k(path)))
        if len(x) < 8000:
            return None
        out = {}
        base = "none"
        if label == 0 and split in ("train", "dev") and rng.random() < 0.45:
            # same encoder settings as the edge-tts service output (48 kbps MP3 @ 24 kHz)
            x = rms_norm(ffmpeg_roundtrip(x[: N * 3], EDGE_MP3, fixed=False)[: min(len(x), N * 3)])
            base = "edgemp3"
        c = crop(x, rng).astype(np.float32)
        out["clean"] = c
        if split in ("train", "dev"):
            out["codec"], desc = random_channel(c, rng)
        else:
            desc = ""
            for cond in TEST_CONDS:
                out[cond] = test_condition(c, cond, rng)
        return key, {k: (np.clip(v, -1, 1) * 32767).astype(np.int16) for k, v in out.items()}, base, desc
    except Exception as e:  # corrupt file etc.
        print("skip", key, e, flush=True)
        return None


# ---------------------------------------------------------------- manifests
def real_items():
    items = []
    for lang in LANGS:
        for split in splits_for(lang):
            for r in read_tsv(lang, split):
                p = os.path.join(FLEURS, lang, split, r["file"])
                if os.path.exists(p):
                    items.append({"key": f"real/{lang}/{split}/{r['file']}", "path": p, "label": 0, "gen": "real",
                                  "lang": lang, "split": split, "gender": r["gender"], "text": r["text"]})
    return items


def fake_items():
    items = []
    for p in sorted(glob.glob(os.path.join(DATA, "fake", "*", "*", "*", "*.wav"))):
        gen, lang, split, f = p.split(os.sep)[-4:]
        items.append({"key": f"{gen}/{lang}/{split}/{f[:-4]}", "path": p, "label": 1, "gen": gen, "lang": lang,
                      "split": split, "gender": "", "text": ""})
    return items


def _base_of(it):
    """Re-derive the deterministic codec-history decision for an already written row."""
    rng = random.Random(it["key"])
    return "edgemp3" if it["label"] == 0 and it["split"] in ("train", "dev") and rng.random() < 0.45 else "none"


def main(group, splits=("train", "dev", "test")):
    from concurrent.futures import ProcessPoolExecutor
    items = real_items() if group == "real" else fake_items()
    for split in splits:
        sub = [it for it in items if it["split"] == split]
        if not sub:
            continue
        d = os.path.join(DATA, "crops", group, split)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "index.csv")):
            print("skip finished", group, split, flush=True)
            continue
        variants = ["clean", "codec"] if split != "test" else ["clean"] + TEST_CONDS
        paths = {v: os.path.join(d, f"{v}.npy") for v in variants}
        resume = all(os.path.exists(p) and np.load(p, mmap_mode="r").shape == (len(sub), N) for p in paths.values())
        mm = {v: np.lib.format.open_memmap(p, "r+" if resume else "w+", np.int16, (len(sub), N)) for v, p in paths.items()}
        rows, todo = [], list(range(len(sub)))
        if resume:  # rows that already hold audio are kept; row i == position in `sub`
            done = np.zeros(len(sub), bool)
            for s in range(0, len(sub), 2000):
                done[s:s + 2000] = np.any(mm["clean"][s:s + 2000] != 0, axis=1)
            for i in np.where(done)[0]:
                it = sub[i]
                rows.append([int(i), it["key"], it["label"], it["gen"], it["lang"], split, it["gender"], _base_of(it), ""])
            todo = [int(i) for i in np.where(~done)[0]]
            print(f"resume {group} {split}: {len(rows)} done, {len(todo)} to go", flush=True)
        with ProcessPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
            for n, (i, res) in enumerate(zip(todo, pool.map(work, [sub[i] for i in todo], chunksize=8))):
                it = sub[i]
                if res is None:
                    continue
                _, arrs, base, desc = res
                for v in variants:
                    mm[v][i] = arrs[v]          # row i == position in `sub`; skipped rows stay unused
                rows.append([i, it["key"], it["label"], it["gen"], it["lang"], split, it["gender"], base, desc])
                if n % 2000 == 0:
                    print(group, split, n, "/", len(todo), flush=True)
        rows.sort(key=lambda r: r[0])
        for v in variants:
            mm[v].flush()
        with open(os.path.join(d, "index.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "key", "label", "gen", "lang", "split", "gender", "base", "channel"])
            w.writerows(rows)
        print("DONE", group, split, len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("train", "dev", "test"))
