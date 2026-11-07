"""Run the asli detector on an audio file or array.

    from asli.detect import Detector
    r = Detector().analyse("voice_note.ogg")
    r["verdict"], r["p_synthetic"], r["windows"]

Audio is decoded with ffmpeg (any format WhatsApp / phones produce), converted to
16 kHz mono, silence-trimmed and loudness-normalised exactly like the training data,
then scored in 4-second windows with a 2-second hop.
"""
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "data", "asli.onnx")
SR, WIN, HOP = 16000, 64000, 32000


def decode(path):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32767


def trim(x, top_db=35.0):
    hop, win = 160, 400
    if len(x) < win * 2:
        return x
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    keep = np.where(20 * np.log10(rms / rms.max()) > -top_db)[0]
    if len(keep) == 0:
        return x
    return x[max(0, keep[0] * hop - 1600): min(len(x), keep[-1] * hop + win + 1600)]


def normalise(x, target_db=-24.0):
    y = x * (10 ** (target_db / 20) / np.sqrt((x ** 2).mean() + 1e-12))
    pk = np.abs(y).max()
    return y / pk * 0.99 if pk > 0.99 else y


def windows(x):
    if len(x) < WIN:
        reps = int(np.ceil(WIN / max(len(x), 1)))
        return np.tile(x, reps)[None, :WIN], [0.0]
    starts = list(range(0, len(x) - WIN + 1, HOP))
    if starts[-1] + WIN < len(x):
        starts.append(len(x) - WIN)
    return np.stack([x[s:s + WIN] for s in starts]), [s / SR for s in starts]


class Detector:
    def __init__(self, model=MODEL):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        meta = json.load(open(model.replace(".onnx", ".json")))
        self.threshold = meta["threshold"]
        self.meta = meta

    def analyse(self, audio, sr=SR):
        if isinstance(audio, str):
            x = decode(audio)
        else:
            x = np.asarray(audio, dtype=np.float32)
            if sr != SR:
                from scipy.signal import resample_poly
                g = np.gcd(sr, SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        dur = len(x) / SR
        x = normalise(trim(x))
        w, starts = windows(x)
        p = self.sess.run(None, {"wav": w.astype(np.float32)})[0].reshape(-1)
        mean = float(p.mean())
        verdict = "synthetic" if mean >= self.threshold else "real"
        if abs(mean - self.threshold) < 0.1:
            verdict = "uncertain"
        return {"p_synthetic": mean, "max_window": float(p.max()), "threshold": self.threshold,
                "verdict": verdict, "duration_s": round(dur, 2),
                "windows": [{"start_s": round(s, 2), "p": float(v)} for s, v in zip(starts, p)]}
