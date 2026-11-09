# asli · असली

**“Beta, it's me. I'm in trouble.” Is that voice real?**

[**Try it in your browser →**](https://rishikeshn-eng.github.io/asli/) Upload a voice note or record yourself. The detector runs locally; nothing is uploaded.

---

Voice clones made from a few seconds of someone's WhatsApp voice note now drive “digital arrest” and family-emergency scams in India.
The audio-deepfake detectors you can download are trained mostly on English studio speech, and scam audio arrives through 8 kHz
phone lines and Opus-compressed voice notes, which destroy the high-frequency artefacts those detectors rely on.

**asli** is a small (589k-parameter, 3.4 MB) synthetic-speech detector trained on **real speech in ten Indian languages** against
several speech engines, and tested on what matters: **speakers, engines and languages it never saw**, through **phone and WhatsApp
channels**.

**The headline result, both halves of it:** asli generalises to a text-to-speech engine it never saw (Google TTS, AUROC 0.99) and
holds up over phone and WhatsApp channels, but it **does not** catch zero-shot voice clones from a model it never saw (XTTS-v2,
AUROC 0.66). Cloning is the open problem, and this repo measures it rather than hiding it inside a pooled average.

```bash
$ asli suspicious_call.ogg
suspicious_call.ogg  9.4s
  SYNTHETIC  P(synthetic) = 0.94  threshold 0.50
    0.0s ████████████████████████████·· 0.93
    2.0s █████████████████████████████· 0.96
    ...
```

## Results

### Unseen speakers, all synthetic engines pooled

| channel | equal error rate | AUROC |
|---|---|---|
| clean | 2.8% | 99.4% |
| phone line (G.711, 8 kHz) | 2.5% | 99.5% |
| WhatsApp-like Opus 12 kbps | 2.4% | 99.4% |
| MP3 32 kbps | 2.3% | 99.5% |
| pink noise, 10 dB SNR | 10.1% | 96.5% |

### By speech engine (clean audio)

| engine | in training? | EER | AUROC |
|---|---|---|---|
| Microsoft neural voices | trained on | 0.3% | 100.0% |
| Google TTS | **never seen** | 3.7% | 99.1% |
| Meta MMS-TTS | trained on | 1.2% | 99.9% |
| XTTS-v2 voice clone | **never seen** | 38.5% | 65.9% |

### By language (clean, EER)

Assamese 0.0% (unseen) · Bengali 5.4% · Gujarati 1.2% · Hindi 7.6% · Kannada 0.3% · Malayalam 0.4% · Marathi 2.2% · Odia 1.2% (unseen) · Punjabi 0.6% · Tamil 2.5% · Telugu 5.5% · Urdu 0.0%

### Against open-source detectors (same test set, AUROC)

| detector | params | clean | phone | Opus | unseen engines | voice clones (XTTS) |
|---|---|---|---|---|---|---|
| **asli (ours)** | 0.6M | 99.4% | 99.5% | 99.4% | 97.6% | 65.9% |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 316M | 98.3% | 96.4% | 97.4% | 96.4% | 81.9% |
| MelodyMachine/Deepfake-audio-detection-V2 | 95M | 41.3% | 23.7% | 41.1% | 37.8% | 39.7% |
| mo-thecreator/Deepfake-audio-detection | 95M | 81.1% | 61.5% | 79.7% | 80.5% | 44.4% |

The two 95M wav2vec2 detectors fail on Indian-language speech: AUROC 41% and 81% clean, falling to 24% and 62% over a phone line (24% means worse than a coin flip). The 316M XLS-R detector is strong: asli matches or beats it on text-to-speech engines and every channel at about 1/500th of the size, but **XLS-R is clearly better on the unseen voice clones (AUROC 82% vs asli's 66%)**. Large self-supervised speech models generalise better to new cloning systems, which is the obvious direction for a v2.

### Shortcut test

False alarms on real test speech: **1.4%** as recorded, **1.2%** after passing
through the exact MP3 encoder one training engine uses. If the model had learned “MP3 artefacts = fake”, the second number would jump.

## Data

| | |
|---|---|
| Real speech | Google **FLEURS**: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu (train/dev/test, speaker-disjoint) + Odia and Assamese (test only) |
| Training fakes | **Meta MMS-TTS** (VITS, one voice per language, prosody randomised) and **Microsoft neural voices** (two per language) |
| Held-out fakes | **Google TTS** (10 languages) and **XTTS-v2 zero-shot voice cloning** (Hindi, each sentence cloned from a *different* real test speaker) |
| Pairing | every fake reads a sentence a real FLEURS speaker also read, so content carries no label |

## Built so it can't cheat

1. **Identical preprocessing.** Every clip is converted to 16 kHz mono, silence-trimmed, RMS-normalised and cut to 4 s.
2. **Codec history neutralised.** The Microsoft voices arrive as 48 kbps MP3, so 45% of real training clips pass through the
   identical encoder. We then measure false alarms on real test speech that went through that encoder (the shortcut test).
3. **Channel robustness.** Each training crop is used clean or through a random lossy channel: Opus 8–24 kbps, MP3, AAC,
   G.711 μ-law telephone (8 kHz, 300–3400 Hz), phone+Opus, or noise. SpecAugment and per-band normalisation remove level and fixed EQ.
4. **Hard test.** Unseen speakers; two unseen engines; two unseen languages; five channels.

## Model

Raw waveform → log-mel (STFT implemented as a fixed Conv1d, so the whole pipeline exports to ONNX and runs identically in the browser)
→ per-band mean/variance normalisation → 5-stage residual CNN with squeeze-excitation → attentive statistics pooling → P(synthetic).
Long recordings are scored in 4 s windows with a 2 s hop.

## Install & use

```bash
pip install -e .            # numpy, scipy, onnxruntime; ffmpeg for decoding
asli voice_note.ogg
asli --json *.wav
```

```python
from asli import Detector
Detector().analyse("call.mp3")      # {'verdict': 'synthetic', 'p_synthetic': 0.94, 'windows': [...]}
```

Tests: `pip install -e ".[dev]" && pytest` runs the detector tests, including the six demo clips, with no PyTorch.
Add the training extra (`pip install -e ".[dev,train]"`) to also run the model-definition tests.

## Reproduce

```
pipeline/sentences.py   pipeline/gen_mms.py   pipeline/gen_edge.py   pipeline/gen_gtts.py   pipeline/gen_xtts.py
pipeline/prepare.py real|fake      crops + channel variants (resumable)
pipeline/train.py                  balanced real/fake batches, dev-based model selection
pipeline/evaluate.py asli          full breakdown → bench/out/asli.json
pipeline/evaluate.py hf:<model>    same test set for open-source baselines
pipeline/export.py                 ONNX + parity check
```

## Limitations

- **A score, not proof.** Newer cloning systems, studio-quality clones and heavy post-processing can evade any detector. If a call asks
  for money or OTPs, hang up and call the person back on a number you know.
- Real speech comes from one source (FLEURS read speech). Spontaneous conversation, children and very noisy environments are
  under-represented, so false alarms on such audio may be higher than measured.
- **It does not reliably detect voice clones from models it hasn't seen** (XTTS-v2: AUROC 0.66). Training on clone audio is the obvious
  next step, but then a different cloning model must be held out to make a fair claim.
- Voice-clone testing covers one cloning model (XTTS-v2), 68 clips, in one language (Hindi).
- XTTS-v2 audio was generated under the Coqui Public Model License for non-commercial evaluation only and is not distributed.

## Credits

FLEURS (CC-BY 4.0) · Meta MMS-TTS (CC-BY-NC 4.0) · Microsoft Edge neural voices (via `edge-tts`) · Google TTS (via `gTTS`) ·
Coqui XTTS-v2 (CPML, evaluation only). Code MIT.
