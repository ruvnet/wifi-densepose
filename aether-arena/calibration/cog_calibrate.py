"""Per-room calibration producer for the cog-pose-estimation **conv+MLP** model
(`pose_v1.safetensors`, 56 subcarriers x 20 frames). Companion to `calibrate.py`
(which targets the MM-Fi *transformer* model) — different model, different adapter
key layout, NOT interchangeable (ADR-150 §3.5).

Fits a rank-r LoRA on the pose head (fc1, fc2) from a short labeled in-room capture and
writes a **safetensors** adapter with keys `fc1.a`/`fc1.b`/`fc2.a`/`fc2.b` (scale baked
into `b`) — exactly what `cog-pose-estimation run --adapter <file>` consumes.

    python cog_calibrate.py --base pose_v1.safetensors --data calib.npz --out room.safetensors

`calib.npz`: `X` [N,56,20] CSI window + `Y` [N,17,2] (or [N,34]) keypoints in [0,1].
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CogPose(nn.Module):
    """Mirrors cog-pose-estimation's PoseNet (Candle) exactly — same safetensors keys."""

    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleDict({
            "c1": nn.Conv1d(56, 64, 3, padding=1, dilation=1),
            "c2": nn.Conv1d(64, 128, 3, padding=2, dilation=2),
            "c3": nn.Conv1d(128, 128, 3, padding=4, dilation=4),
        })
        self.head = nn.ModuleDict({"fc1": nn.Linear(128, 256), "fc2": nn.Linear(256, 34)})
        self.fc1_lora = None
        self.fc2_lora = None

    def _lora(self, slot, x, y):
        if slot is None:
            return y
        a, b = slot
        return y + (x @ a) @ b

    def forward(self, x):                       # x: [B, 56, 20]
        h = F.relu(self.enc["c1"](x))
        h = F.relu(self.enc["c2"](h))
        h = F.relu(self.enc["c3"](h))
        h = h.mean(2)                            # [B, 128]
        z1 = self.head["fc1"](h)
        z1 = self._lora(self.fc1_lora, h, z1)
        h1 = F.relu(z1)
        z2 = self.head["fc2"](h1)
        z2 = self._lora(self.fc2_lora, h1, z2)
        return torch.sigmoid(z2)                 # [B, 34]

    def add_lora(self, r=4):
        self.fc1_lora = (nn.Parameter(torch.randn(128, r) * 0.02), nn.Parameter(torch.zeros(r, 256)))
        self.fc2_lora = (nn.Parameter(torch.randn(256, r) * 0.02), nn.Parameter(torch.zeros(r, 34)))
        for p in (*self.fc1_lora, *self.fc2_lora):
            self.register_parameter(f"lora_{id(p)}", p)
        return self


def load_base(net: CogPose, path: str):
    from safetensors.torch import load_file
    sd = load_file(path)
    # remap "enc.c1.weight" -> module dict keys
    mapped = {}
    for k, v in sd.items():
        mapped[k.replace("enc.", "enc.").replace("head.", "head.")] = v
    net.load_state_dict(mapped, strict=False)
    return net


def fit(base: str, data: str, out: str, rank: int = 4, iters: int = 400, lr: float = 1e-3):
    z = np.load(data)
    X = torch.tensor(z["X"].astype(np.float32))          # [N,56,20]
    Y = torch.tensor(z["Y"].reshape(len(z["Y"]), 34).astype(np.float32))
    n = len(X)
    net = CogPose()
    load_base(net, base)
    net.add_lora(rank)
    for p in net.parameters():
        p.requires_grad = False
    lora = [*net.fc1_lora, *net.fc2_lora]
    for p in lora:
        p.requires_grad = True
    opt = torch.optim.AdamW(lora, lr=lr, weight_decay=0.0)
    lossf = nn.SmoothL1Loss(beta=0.1)
    bs = min(64, n)
    net.train()
    for _ in range(iters):
        bi = torch.randint(0, n, (bs,))
        opt.zero_grad()
        lossf(net(X[bi]), Y[bi]).backward()
        opt.step()

    alpha = 16.0
    scale = alpha / rank
    a1, b1 = net.fc1_lora
    a2, b2 = net.fc2_lora
    tensors = {
        "fc1.a": a1.detach().contiguous(),
        "fc1.b": (b1.detach() * scale).contiguous(),    # bake scale into b
        "fc2.a": a2.detach().contiguous(),
        "fc2.b": (b2.detach() * scale).contiguous(),
    }
    from safetensors.torch import save_file
    save_file(tensors, out)
    return out, sum(p.numel() for p in lora), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    out, np_, n = fit(a.base, a.data, a.out, a.rank, a.iters)
    print(f"saved {out} | {np_} LoRA params from {n} samples "
          f"(keys fc1.a/fc1.b/fc2.a/fc2.b — load with cog-pose-estimation run --adapter)")
