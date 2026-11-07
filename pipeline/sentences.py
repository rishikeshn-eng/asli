"""Collect the unique FLEURS sentences per language/split -> sentences.jsonl.

Every fake clip is synthesised from a sentence that a real FLEURS speaker also
read, so real and fake audio share the same linguistic content: the detector
cannot win by recognising *what* is said, only *how* it was produced.
"""
import json
import os
from common import DATA, LANGS, splits_for, read_tsv

out = os.path.join(DATA, "sentences.jsonl")
n = 0
with open(out, "w", encoding="utf-8") as f:
    for lang in LANGS:
        for split in splits_for(lang):
            seen = {}
            for r in read_tsv(lang, split):
                t = r["text"].strip()
                if t and t not in seen:
                    seen[t] = len(seen)
            for t, i in seen.items():
                f.write(json.dumps({"lang": lang, "split": split, "sid": f"{lang}-{split}-{i:04d}", "text": t},
                                   ensure_ascii=False) + "\n")
                n += 1
print("sentences:", n, "->", out)
