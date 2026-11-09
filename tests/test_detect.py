"""Detector tests: need only the base install (numpy, scipy, onnxruntime), no torch."""
import os

import numpy as np
import pytest
from scipy.io import wavfile

from asli import detect

HERE = os.path.dirname(__file__)
ONNX = os.path.join(HERE, "..", "asli", "data", "asli.onnx")
SAMPLES = os.path.join(HERE, "..", "docs", "samples")
needs_model = pytest.mark.skipif(not os.path.exists(ONNX), reason="model not exported yet")


def test_windows_cover_long_and_short_audio():
    w, starts = detect.windows(np.zeros(16000 * 9, np.float32))
    assert w.shape[1] == 64000 and starts[0] == 0.0 and starts[-1] == 5.0
    w, starts = detect.windows(np.ones(16000, np.float32))
    assert w.shape == (1, 64000)


def test_trim_and_normalise():
    x = np.concatenate([np.zeros(16000), 0.3 * np.sin(np.arange(32000) / 5), np.zeros(16000)]).astype(np.float32)
    y = detect.normalise(detect.trim(x))
    assert len(y) < len(x)
    assert abs(20 * np.log10(np.sqrt((y ** 2).mean())) + 24) < 0.5


@needs_model
def test_detector_runs_on_array():
    r = detect.Detector().analyse(np.random.default_rng(0).standard_normal(16000 * 5).astype(np.float32) * 0.05)
    assert 0.0 <= r["p_synthetic"] <= 1.0
    assert r["verdict"] in ("real", "synthetic", "uncertain")
    assert len(r["windows"]) >= 1


@needs_model
@pytest.mark.parametrize("name,verdict", [
    ("real_hi_in.wav", "real"), ("real_ta_in.wav", "real"), ("real_bn_in.wav", "real"),
    ("mms_hi_in.wav", "synthetic"), ("mms_ta_in.wav", "synthetic"), ("mms_bn_in.wav", "synthetic"),
])
def test_demo_samples_are_classified_correctly(name, verdict):
    """The six clips shipped with the demo site (real FLEURS speech and MMS-TTS speech)."""
    sr, x = wavfile.read(os.path.join(SAMPLES, name))
    assert detect.Detector().analyse(x.astype(np.float32) / 32768.0, sr)["verdict"] == verdict
