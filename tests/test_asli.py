import os

import numpy as np
import pytest
import torch

from asli.model import AsliNet, LogMel, mel_filterbank, N_FFT, HOP, WIN
from asli import detect

ONNX = os.path.join(os.path.dirname(__file__), "..", "asli", "data", "asli.onnx")


def test_conv_stft_matches_torch_stft():
    x = torch.randn(1, 16000)
    lm = LogMel()
    spec = torch.nn.functional.conv1d(x.unsqueeze(1), lm.basis, stride=HOP)
    power = spec[:, :257] ** 2 + spec[:, 257:] ** 2
    ref = torch.stft(x, N_FFT, HOP, WIN, torch.hann_window(WIN, periodic=True), center=False,
                     return_complex=True).abs() ** 2
    n = min(power.shape[-1], ref.shape[-1])
    assert float((power[..., :n] - ref[..., :n]).abs().max() / ref.abs().max()) < 1e-4


def test_mel_filterbank_shape_and_coverage():
    fb = mel_filterbank()
    assert fb.shape == (80, 257)
    assert (fb.sum(1) > 0).all()


def test_model_shape_and_level_invariance():
    net = AsliNet().eval()
    x = torch.randn(3, 64000) * 0.1
    with torch.no_grad():
        a = net(x)
        b = net(x * 0.25)  # per-band mean/variance normalisation removes gain
    assert a.shape == (3,)
    assert torch.allclose(a, b, atol=1e-3)


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


@pytest.mark.skipif(not os.path.exists(ONNX), reason="model not exported yet")
def test_detector_runs_on_array():
    d = detect.Detector()
    r = d.analyse(np.random.default_rng(0).standard_normal(16000 * 5).astype(np.float32) * 0.05)
    assert 0.0 <= r["p_synthetic"] <= 1.0
    assert r["verdict"] in ("real", "synthetic", "uncertain")
    assert len(r["windows"]) >= 1
