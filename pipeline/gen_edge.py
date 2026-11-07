"""Synthesise sentences with Microsoft's neural Indian voices (edge-tts).

These are commercial-grade neural voices of the kind used for scam robocalls and
dubbed social video. The service returns 24 kHz MP3; we decode to 16 kHz WAV and
record the codec history so the training pipeline can neutralise it.
Usage: python gen_edge.py [langs...]
"""
import asyncio
import os
import random
import subprocess
import sys
import edge_tts
from common import LANGS, TRAIN_LANGS, splits_for, sentences, fake_path

CONC = int(os.environ.get("CONC", "6"))


async def synth(sem, voice, text, mp3, wav):
    async with sem:
        for attempt in range(6):
            try:
                rate = random.choice(["-8%", "-4%", "+0%", "+4%", "+8%"])
                # the service occasionally never answers; without a timeout one stuck call blocks a whole language
                await asyncio.wait_for(edge_tts.Communicate(text, voice, rate=rate).save(mp3), 60)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", wav],
                               check=True)
                os.remove(mp3)
                return True
            except Exception as e:  # network hiccups / throttling
                await asyncio.sleep(2 ** attempt + random.random())
        print("FAILED", voice, wav, flush=True)
        return False


async def main(langs):
    sem = asyncio.Semaphore(CONC)
    for lang in langs:
        voices = LANGS[lang][2]
        if not voices:
            continue
        tasks = []
        for split in splits_for(lang):
            for i, (sid, text) in enumerate(sentences(lang, split)):
                voice = voices[i % 2]  # alternate female / male
                wav = fake_path("edge", lang, split, sid)
                if os.path.exists(wav):
                    continue
                tasks.append(synth(sem, voice, text, wav[:-4] + ".mp3", wav))
        ok = sum(await asyncio.gather(*tasks))
        print(f"[edge] {lang} +{ok}/{len(tasks)}", flush=True)
    print("EDGE_DONE", flush=True)


asyncio.run(main(sys.argv[1:] or TRAIN_LANGS))
