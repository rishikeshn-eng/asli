"""HELD-OUT generator: XTTS-v2 zero-shot *voice cloning* (Hindi), test split only.

This is the realistic threat: a scammer takes a few seconds of someone's real
voice (a WhatsApp voice note, a reel) and makes it say anything. For every
Hindi test sentence we clone a *different* real FLEURS test speaker from a 4-9 s
reference clip. The detector never sees XTTS audio during training.
Run inside .venv-xtts:  COQUI_TOS_AGREED=1 python gen_xtts.py
(XTTS-v2 weights are under the Coqui Public Model License: non-commercial research use.)
"""
import os
import random
import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS
from common import FLEURS, read_tsv, sentences, fake_path

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

rows = read_tsv("hi_in", "test")
refs = []
for r in rows:
    p = os.path.join(FLEURS, "hi_in", "test", r["file"])
    if os.path.exists(p):
        d = sf.info(p).duration
        if 4.0 <= d <= 9.0:
            refs.append((p, r["text"]))

rng = random.Random(11)
n = 0
CAP = int(os.environ.get("XTTS_CAP", "80"))
for sid, text in sentences("hi_in", "test"):
    if n >= CAP:
        break
    out = fake_path("xtts", "hi_in", "test", sid)
    if os.path.exists(out):
        continue
    # never clone the speaker from a clip of the same sentence
    ref = rng.choice([p for p, t in refs if t.strip() != text])
    if any(ch.isdigit() for ch in text) or len(text) > 150:
        continue  # XTTS's Hindi normaliser cannot spell digits; its hard limit is ~150 characters
    try:
        wav = tts.tts(text=text, speaker_wav=ref, language="hi")
    except Exception as e:  # keep going on any per-sentence failure
        print("skip", sid, type(e).__name__, flush=True)
        continue
    wav = np.asarray(wav, dtype=np.float32)
    # XTTS outputs 24 kHz; resample to 16 kHz like every other source
    import torchaudio.functional as AF
    w16 = AF.resample(torch.from_numpy(wav), 24000, 16000).numpy()
    sf.write(out, w16 / max(1e-6, np.abs(w16).max()) * 0.9, 16000, subtype="PCM_16")
    n += 1
    if n % 20 == 0:
        print("[xtts]", n, flush=True)
print("XTTS_DONE", n, flush=True)
