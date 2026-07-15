"""WiFi-CSI pose model + LoRA adapter for the RuView calibration service.

Architecture matches the published flagship checkpoint
[`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose)
(`pose_mmfi_best.pt`): transformer encoder + temporal attention pooling + skeleton-graph head.

The calibration service freezes this base and fits a tiny per-room **LoRA adapter** (rank 8 on the
input projection + pose head ≈ 11 KB) from ~100–200 labeled in-room samples. Empirically that lifts
cross-subject 64→72% and cross-environment 11→73% (ADR-150 §3.3–3.6).
"""
import numpy as np
import torch
import torch.nn as nn

# COCO-17 skeleton edges for the graph-refinement head.
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
_A = np.eye(17, dtype=np.float32)
for _i, _j in EDGES:
    _A[_i, _j] = _A[_j, _i] = 1.0
_A = _A / _A.sum(1, keepdims=True)


class LoRA(nn.Module):
    """Low-rank adapter wrapping a frozen Linear: y = W·x + (x·A·B)·(alpha/r)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(base.in_features, r))
        self.B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.normal_(self.A, std=0.02)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


class GR(nn.Module):
    """Skeleton-graph refinement: nudges joints toward anatomically consistent positions."""

    def __init__(self, d=256, h=96):
        super().__init__()
        self.je = nn.Parameter(torch.randn(17, 32) * 0.02)
        self.inp = nn.Linear(d + 34, h)
        self.g1 = nn.Linear(h, h)
        self.g2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 2)
        self.register_buffer("A", torch.tensor(_A))

    def forward(self, z, kp0):
        B = z.shape[0]
        f = torch.relu(self.inp(torch.cat(
            [z.unsqueeze(1).expand(-1, 17, -1), self.je.unsqueeze(0).expand(B, -1, -1), kp0], -1)))
        f = torch.relu(self.g1(torch.einsum('ij,bjh->bih', self.A, f)))
        f = torch.relu(self.g2(torch.einsum('ij,bjh->bih', self.A, f)))
        return kp0 + 0.3 * torch.tanh(self.out(f))


class PoseNet(nn.Module):
    """Flagship pose model. Input [B,3,114,10] CSI amplitude (per-sample standardized) -> [B,34]."""

    def __init__(self, na=3, nsc=114, nt=10, d=256, L=4, H=8):
        super().__init__()
        self.proj = nn.Linear(na * nsc, d)
        self.pos = nn.Parameter(torch.randn(1, nt, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, H, d * 2, dropout=0.2, batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(enc, L)
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 34))
        self.gr = GR(d)
        self.na, self.nsc, self.nt = na, nsc, nt

    def forward(self, x):
        B = x.shape[0]
        t = x.permute(0, 3, 1, 2).reshape(B, self.nt, self.na * self.nsc)
        h = self.tf(self.proj(t) + self.pos)
        w = torch.softmax(self.att(h), 1)
        z = (h * w).sum(1)
        kp0 = torch.sigmoid(self.head(z)).reshape(B, 17, 2)
        return self.gr(z, kp0).reshape(B, 34)

    def add_lora(self, r=8, alpha=16):
        """Wrap the input projection + pose head with LoRA adapters (the ~11 KB calibration set)."""
        self.proj = LoRA(self.proj, r, alpha)
        self.head[0] = LoRA(self.head[0], r, alpha)
        self.head[3] = LoRA(self.head[3], r, alpha)
        return self

    def lora_state(self) -> dict:
        """Extract just the LoRA A/B tensors (the per-room adapter to save)."""
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()
                if k.endswith(".A") or k.endswith(".B")}

    def load_lora(self, adapter: dict):
        sd = self.state_dict()
        for k, v in adapter.items():
            sd[k] = torch.tensor(v)
        self.load_state_dict(sd)
        return self


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample standardization used in training/inference."""
    return (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)
