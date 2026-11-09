"""Model-definition tests: need PyTorch (`pip install -e ".[train]"`); skipped cleanly without it."""
import pytest

torch = pytest.importorskip("torch")

from asli.model import AsliNet, LogMel, mel_filterbank, N_FFT, HOP, WIN  # noqa: E402


def test_conv_stft_matches_torch_stft():
    x = torch.randn(1, 16000)
    lm = LogMel()
    spec = torch.nn.functional.conv1d(x.unsqueeze(1), lm.basis, stride=HOP)
    power = spec[:, :257] ** 2 + spec[:, 257:] ** 2
    ref = torch.stft(x, N_FFT, HOP, WIN, torch.hann_window(WIN, periodic=True), center=False,
                     return_complex=True).abs() ** 2
    n = min(power.shape[-1], ref.shape[-1])
    assert float((power[..., :n] - ref[..., :n]).abs().max() / ref.abs().max()) < 1e-4


def test_training_front_end_matches_export_front_end():
    """Training uses torch.stft (fast on GPU); ONNX export uses the Conv1d path. They must agree."""
    x = torch.randn(2, 64000)
    lm = LogMel()
    a = lm(x)
    spec = torch.nn.functional.conv1d(x.unsqueeze(1), lm.basis, stride=HOP)
    p = spec[:, :257] ** 2 + spec[:, 257:] ** 2
    b = torch.log(torch.matmul(p.transpose(1, 2), lm.fb) + 1e-6).transpose(1, 2)
    assert float((a - b).abs().max()) < 1e-3


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
