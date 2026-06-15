"""Regression test for the cog-pose adapter producer (cog_calibrate.py).

Uses the in-repo `pose_v1.safetensors` (skips if absent). Verifies the produced adapter:
  - has the exact keys/shapes the Rust `cog-pose-estimation --adapter` loader expects,
  - reduces calibration fit error,
  - actually changes inference output,
  - is tiny.
Run: python test_cog_calibration.py   (or via pytest)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cog_calibrate as C  # noqa: E402

BASE = HERE / "../../v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors"


def test_cog_adapter_producer():
    if not BASE.exists():
        print(f"(skip — {BASE} not present)")
        return
    from safetensors.torch import load_file

    rng = np.random.default_rng(0)
    n = 120
    X = rng.standard_normal((n, 56, 20)).astype("float32")
    Y = (0.5 + 0.1 * X[:, :34, 0].reshape(n, 34)).clip(0, 1).astype("float32")

    with tempfile.TemporaryDirectory() as d:
        calib = os.path.join(d, "calib.npz")
        adapter = os.path.join(d, "room.safetensors")
        np.savez(calib, X=X, Y=Y)

        net0 = C.CogPose()
        C.load_base(net0, str(BASE))
        net0.eval()
        with torch.no_grad():
            base_err = F.smooth_l1_loss(net0(torch.tensor(X)), torch.tensor(Y)).item()

        _, nparam, _ = C.fit(str(BASE), calib, adapter, rank=4, iters=400)
        t = load_file(adapter)

        # exact Rust loader contract: a:[in,r], b:[r,out]
        assert tuple(t["fc1.a"].shape) == (128, 4)
        assert tuple(t["fc1.b"].shape) == (4, 256)
        assert tuple(t["fc2.a"].shape) == (256, 4)
        assert tuple(t["fc2.b"].shape) == (4, 34)

        net = C.CogPose()
        C.load_base(net, str(BASE))
        net.add_lora(4)
        with torch.no_grad():
            net.fc1_lora[0].copy_(t["fc1.a"]); net.fc1_lora[1].copy_(t["fc1.b"] / (16 / 4))
            net.fc2_lora[0].copy_(t["fc2.a"]); net.fc2_lora[1].copy_(t["fc2.b"] / (16 / 4))
        net.eval()
        with torch.no_grad():
            cal_err = F.smooth_l1_loss(net(torch.tensor(X)), torch.tensor(Y)).item()
            changed = (net0(torch.tensor(X[:8])) - net(torch.tensor(X[:8]))).abs().sum().item()

        assert cal_err < base_err, f"calibration did not reduce error ({base_err} -> {cal_err})"
        assert changed > 1e-3, "adapter inert"
        assert nparam < 5000, f"adapter unexpectedly large ({nparam} params)"


if __name__ == "__main__":
    test_cog_adapter_producer()
    print("PASS: cog adapter producer (Rust-loadable format, reduces error, active)")
