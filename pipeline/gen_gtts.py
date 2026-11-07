"""HELD-OUT generator: Google Translate TTS (gTTS). Test split only, never trained on.
Usage: python gen_gtts.py [per_lang_cap]
"""
import os
import random
import subprocess
import sys
import time
from gtts import gTTS
from common import LANGS, TRAIN_LANGS, sentences, fake_path

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
for lang in TRAIN_LANGS:
    code = LANGS[lang][3]
    sents = sentences(lang, "test")
    random.Random(7).shuffle(sents)
    ok = 0
    for sid, text in sents[:CAP]:
        wav = fake_path("gtts", lang, "test", sid)
        if os.path.exists(wav):
            ok += 1
            continue
        for attempt in range(5):
            try:
                mp3 = wav[:-4] + ".mp3"
                gTTS(text, lang=code).save(mp3)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", wav],
                               check=True)
                os.remove(mp3)
                ok += 1
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        time.sleep(0.4)
    print(f"[gtts] {lang} {ok}", flush=True)
print("GTTS_DONE", flush=True)
