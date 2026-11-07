"""Export the trained detector to ONNX for the Python package and the browser demo.

Writes asli/data/asli.onnx (+ .json with threshold and provenance) and copies it to
docs/model/. Verifies ONNX output against PyTorch on real test crops.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asli.model import AsliNet, Exportable  # noqa: E402
from common import DATA  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "asli", "data", "asli.onnx")


def main():
    net = AsliNet().eval()
    net.load_state_dict(torch.load(os.path.join(ROOT, "models", "asli.pt"), map_location="cpu"))
    ex = Exportable(net).eval()
    dummy = torch.zeros(2, 64000)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.onnx.export(ex, dummy, OUT, input_names=["wav"], output_names=["p_synthetic", "logmel"],
                      dynamic_axes={"wav": {0: "b"}, "p_synthetic": {0: "b"}, "logmel": {0: "b"}},
                      opset_version=17, dynamo=False)
    thr = json.load(open(os.path.join(ROOT, "models", "asli.threshold.json")))
    meta = {**thr, "sample_rate": 16000, "window_samples": 64000, "hop_samples": 32000,
            "trained_on": "FLEURS (10 Indian languages) vs MMS-TTS + Microsoft neural voices",
            "params": sum(p.numel() for p in net.parameters() if p.requires_grad)}
    json.dump(meta, open(OUT.replace(".onnx", ".json"), "w"), indent=1)
    # parity check on real test audio
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    x = np.load(os.path.join(DATA, "crops", "fake", "test", "clean.npy"), mmap_mode="r")[:16].astype(np.float32) / 32767
    a = sess.run(None, {"wav": x})[0].reshape(-1)
    with torch.no_grad():
        b = torch.sigmoid(net(torch.from_numpy(x))).numpy()
    print("ONNX vs torch max abs diff:", float(np.abs(a - b).max()), "| size KB", os.path.getsize(OUT) // 1024)
    docs = os.path.join(ROOT, "docs", "model")
    os.makedirs(docs, exist_ok=True)
    shutil.copy(OUT, docs)
    shutil.copy(OUT.replace(".onnx", ".json"), docs)


if __name__ == "__main__":
    main()
