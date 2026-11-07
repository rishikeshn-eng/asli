"""Synthesise every sentence with Meta's MMS-TTS (VITS) voice for that language.

Usage: python gen_mms.py hi_in bn_in ...
Prosody is varied per sentence (speaking rate, noise scales) so the detector
does not learn one fixed VITS rendering.
"""
import sys
import os
import random
import warnings
import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
from common import LANGS, splits_for, sentences, fake_path

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("THREADS", "4")))

# Evaluation splits first; training fakes are capped because the neural (edge) voices
# already cover every training sentence. Caps: CAP_TRAIN / CAP_DEV / CAP_TEST sentences.
CAPS = {"test": int(os.environ.get("CAP_TEST", "220")), "dev": int(os.environ.get("CAP_DEV", "100")),
        "train": int(os.environ.get("CAP_TRAIN", "600"))}
DEVICE = os.environ.get("DEVICE", "cpu")

for lang in sys.argv[1:]:
    code = LANGS[lang][1]
    tok = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{code}")
    model = VitsModel.from_pretrained(f"facebook/mms-tts-{code}").eval().to(DEVICE)
    done = 0
    for split in [s for s in ("test", "dev", "train") if s in splits_for(lang)]:
        sents = sentences(lang, split)
        random.Random(lang + split).shuffle(sents)
        for sid, text in sents[:CAPS[split]]:
            path = fake_path("mms", lang, split, sid)
            if os.path.exists(path):
                continue
            rng = random.Random(sid)
            torch.manual_seed(rng.randrange(1 << 30))
            model.speaking_rate = rng.uniform(0.9, 1.12)
            model.noise_scale = rng.uniform(0.55, 0.75)
            model.noise_scale_duration = rng.uniform(0.7, 0.9)
            ids = tok(text, return_tensors="pt")
            if ids.input_ids.shape[1] < 2:
                continue
            with torch.no_grad():
                wav = model(**{k: v.to(DEVICE) for k, v in ids.items()}).waveform[0].float().cpu().numpy()
            sf.write(path, (wav / max(1e-6, np.abs(wav).max()) * 0.9).astype(np.float32), 16000, subtype="PCM_16")
            done += 1
    print(f"[mms] {lang} done (+{done})", flush=True)
print("MMS_DONE", flush=True)
