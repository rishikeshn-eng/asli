"""asli detector: raw 16 kHz waveform -> P(synthetic speech).

The front end (STFT -> mel -> log) is written with ordinary Conv1d/MatMul ops using
fixed, non-trainable weights, so the *entire* pipeline exports to ONNX and runs
bit-for-bit the same in Python, the CLI and the browser.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT, WIN, HOP, N_MELS = 512, 400, 160, 80


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=20.0, fmax=8000.0):
    """HTK-style triangular mel filterbank, shape [n_mels, n_fft//2+1]."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(c - lo, 1e-9)
        down = (hi - bins) / max(hi - c, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


class LogMel(nn.Module):
    """STFT as a strided Conv1d with a fixed windowed-DFT basis."""

    def __init__(self):
        super().__init__()
        n_bins = N_FFT // 2 + 1
        win = np.zeros(N_FFT, dtype=np.float64)
        pad = (N_FFT - WIN) // 2
        win[pad:pad + WIN] = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(WIN) / WIN)  # periodic Hann
        n = np.arange(N_FFT)
        k = np.arange(n_bins)[:, None]
        cos = np.cos(2 * math.pi * k * n / N_FFT) * win
        sin = -np.sin(2 * math.pi * k * n / N_FFT) * win
        basis = np.concatenate([cos, sin], 0).astype(np.float32)[:, None, :]  # [2*bins, 1, N_FFT]
        self.register_buffer("basis", torch.from_numpy(basis))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank().T.copy()))  # [bins, mels]
        self.n_bins = n_bins

    def forward(self, wav):  # wav [B, T]
        if torch.onnx.is_in_onnx_export():
            # ONNX path: plain Conv1d, so the exported graph runs anywhere (incl. onnxruntime-web)
            spec = F.conv1d(wav.unsqueeze(1), self.basis, stride=HOP)   # [B, 2*bins, frames]
            re, im = spec[:, : self.n_bins], spec[:, self.n_bins:]
            power = re * re + im * im                                   # [B, bins, frames]
        else:
            # identical maths (tests: < 1e-4 rel. error), ~60x faster on GPU
            win = torch.zeros(N_FFT, device=wav.device, dtype=wav.dtype)
            pad = (N_FFT - WIN) // 2
            win[pad:pad + WIN] = torch.hann_window(WIN, periodic=True, device=wav.device, dtype=wav.dtype)
            power = torch.stft(wav, N_FFT, HOP, N_FFT, win, center=False, return_complex=True).abs() ** 2
        mel = torch.matmul(power.transpose(1, 2), self.fb)              # [B, frames, mels]
        return torch.log(mel + 1e-6).transpose(1, 2)                    # [B, mels, frames]


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, c // r), nn.Linear(c // r, c)

    def forward(self, x):
        s = x.mean((2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.se = SE(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.se(self.b2(self.c2(y)))
        return F.relu(y + self.skip(x))


class AsliNet(nn.Module):
    def __init__(self, widths=(32, 48, 64, 96, 128)):
        super().__init__()
        self.front = LogMel()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(Block(cin, w, 1 if i == 0 else 2))
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.att = nn.Sequential(nn.Conv1d(cin, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
        self.head = nn.Sequential(nn.Linear(cin * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def features(self, wav):
        m = self.front(wav)
        # per-utterance, per-band mean/variance normalisation: removes level and fixed channel EQ
        m = (m - m.mean(2, keepdim=True)) / (m.std(2, keepdim=True) + 1e-5)
        return m

    def forward(self, wav, specaug=False):
        m = self.features(wav)
        if specaug and self.training:
            m = spec_augment(m)
        x = self.blocks(self.stem(m.unsqueeze(1)))       # [B, C, F', T']
        x = x.mean(2)                                    # pool frequency -> [B, C, T']
        a = torch.softmax(self.att(x), dim=2)            # attention over time
        mu = (x * a).sum(2)
        sd = torch.sqrt(((x - mu.unsqueeze(2)) ** 2 * a).sum(2).clamp_min(1e-6))
        return self.head(torch.cat([mu, sd], 1)).squeeze(1)  # logit: >0 means synthetic


def spec_augment(m, n_freq=2, f_w=10, n_time=2, t_w=40):
    """Vectorised SpecAugment: random frequency and time bands zeroed per example (no Python loop over the batch)."""
    B, Fq, T = m.shape
    dev = m.device
    keep = torch.ones(B, Fq, T, device=dev)
    fr = torch.arange(Fq, device=dev)[None, :]
    tr = torch.arange(T, device=dev)[None, :]
    for _ in range(n_freq):
        w = torch.randint(0, f_w + 1, (B, 1), device=dev)
        s0 = (torch.rand(B, 1, device=dev) * (Fq - w)).long()
        keep = keep * (~((fr >= s0) & (fr < s0 + w))).float()[:, :, None]
    for _ in range(n_time):
        w = torch.randint(0, t_w + 1, (B, 1), device=dev)
        s0 = (torch.rand(B, 1, device=dev) * (T - w)).long()
        keep = keep * (~((tr >= s0) & (tr < s0 + w))).float()[:, None, :]
    return m * keep


class Exportable(nn.Module):
    """ONNX wrapper: returns (P(synthetic), log-mel for visualisation)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, wav):
        return torch.sigmoid(self.net(wav)), self.net.front(wav)


if __name__ == "__main__":
    net = AsliNet()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "trainable params")
    print(net(torch.randn(2, 64000)).shape)
