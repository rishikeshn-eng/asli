"""Shared paths and language tables for the asli data pipeline."""
import os
import json
import csv

DATA = os.environ.get("ASLI_DATA", os.path.expanduser("~/Downloads/claude/rawdata2/asli"))
FLEURS = os.environ.get("ASLI_FLEURS", os.path.expanduser("~/Downloads/claude/rawdata2/fleurs"))

# FLEURS code -> (display name, MMS-TTS code, edge-tts voices, gTTS code)
LANGS = {
    "hi_in": ("Hindi", "hin", ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"], "hi"),
    "bn_in": ("Bengali", "ben", ["bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"], "bn"),
    "ta_in": ("Tamil", "tam", ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"], "ta"),
    "te_in": ("Telugu", "tel", ["te-IN-ShrutiNeural", "te-IN-MohanNeural"], "te"),
    "mr_in": ("Marathi", "mar", ["mr-IN-AarohiNeural", "mr-IN-ManoharNeural"], "mr"),
    "gu_in": ("Gujarati", "guj", ["gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural"], "gu"),
    "kn_in": ("Kannada", "kan", ["kn-IN-SapnaNeural", "kn-IN-GaganNeural"], "kn"),
    "ml_in": ("Malayalam", "mal", ["ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"], "ml"),
    "pa_in": ("Punjabi", "pan", [], "pa"),
    "ur_pk": ("Urdu", "urd-script_arabic", ["ur-IN-GulNeural", "ur-IN-SalmanNeural"], "ur"),
    # held-out languages: never used for training, test split only
    "or_in": ("Odia", "ory", [], None),
    "as_in": ("Assamese", "asm", [], None),
}
TRAIN_LANGS = [l for l in LANGS if l not in ("or_in", "as_in")]
HELDOUT_LANGS = ["or_in", "as_in"]


def splits_for(lang):
    return ["test"] if lang in HELDOUT_LANGS else ["train", "dev", "test"]


def read_tsv(lang, split):
    path = os.path.join(DATA, "tsv", f"{lang}.{split}.tsv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"fid": r[0], "file": r[1], "text": r[2], "gender": r[6] if len(r) > 6 else ""})
    return rows


def sentences(lang, split):
    """Unique sentences in a split, in a stable order, as (sid, text)."""
    path = os.path.join(DATA, "sentences.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["lang"] == lang and d["split"] == split:
                out.append((d["sid"], d["text"]))
    return out


def fake_path(gen, lang, split, sid, ext="wav"):
    d = os.path.join(DATA, "fake", gen, lang, split)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{sid}.{ext}")
