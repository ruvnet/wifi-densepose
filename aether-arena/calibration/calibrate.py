"""RuView per-room calibration — fit a ~11 KB LoRA adapter from a short labeled in-room capture.

    python calibrate.py --base pose_mmfi_best.pt --data room_calib.npz --out room_A.adapter.npz

`room_calib.npz` must contain `X` [N,3,114,10] CSI amplitude and `Y` [N,17,2] (or [N,34]) keypoints
in [0,1] — the labeled calibration samples from the deployment room (~100–200 recommended; ≥20).
Outputs a tiny adapter (.npz, ~11 KB) that, loaded over the shared base at inference, recovers
SOTA-level pose for that room/person (ADR-150 §3.5–3.6).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import PoseNet, standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base checkpoint (pose_mmfi_best.pt)")
    ap.add_argument("--data", required=True, help="labeled calibration .npz with X and Y")
    ap.add_argument("--out", required=True, help="output adapter .npz")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    z = np.load(a.data)
    X = torch.tensor(z["X"].astype(np.float32))
    Y = torch.tensor(z["Y"].reshape(len(z["Y"]), 34).astype(np.float32))
    n = len(X)
    if n < 20:
        print(f"WARNING: only {n} calibration samples — below ~20 the adapter may underperform "
              f"zero-shot (ADR-150 §3.5). Recommend ~100–200.")
    dev = a.device

    net = PoseNet().to(dev)
    net.load_state_dict(torch.load(a.base, map_location=dev), strict=False)
    net.add_lora(r=a.rank).to(dev)
    for k, p in net.named_parameters():
        p.requires_grad = k.endswith(".A") or k.endswith(".B")
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)

    Xs = standardize(X.to(dev))
    Yt = Y.to(dev)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    lossf = nn.SmoothL1Loss(beta=0.1)
    bs = min(128, n)
    net.train()
    for it in range(a.iters):
        bi = torch.randint(0, n, (bs,), device=dev)
        xb = Xs[bi]
        # light augmentation (subcarrier dropout + noise) — matches training-time regularization
        m = (torch.rand(xb.shape[0], xb.shape[1], 1, 1, device=dev) > 0.15).float()
        xb = xb * m + 0.03 * torch.randn_like(xb) * torch.rand(xb.shape[0], 1, 1, 1, device=dev)
        opt.zero_grad()
        lossf(net(xb), Yt[bi]).backward()
        opt.step()

    adapter = net.lora_state()
    nbytes = sum(v.astype(np.float16).nbytes for v in adapter.values())
    np.savez(a.out, **{k: v.astype(np.float16) for k, v in adapter.items()},
             _meta=np.array([a.rank, n, n_tr], dtype=np.int64))
    print(f"saved {a.out} | rank {a.rank} | {n_tr:,} params | ~{nbytes/1024:.1f} KB fp16 | "
          f"from {n} labeled samples")


if __name__ == "__main__":
    main()
