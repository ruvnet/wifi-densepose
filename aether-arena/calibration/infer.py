"""Run calibrated WiFi-CSI pose inference: shared base + a per-room LoRA adapter.

    python infer.py --base pose_mmfi_best.pt --adapter room_A.adapter.npz --data frames.npz

`frames.npz` contains `X` [N,3,114,10] CSI amplitude. Prints/saves [N,17,2] keypoints in [0,1].
Omit --adapter to run the uncalibrated (zero-shot) base. With a room adapter, expect SOTA-level
accuracy in that room/person; without one, zero-shot degrades in unseen rooms (ADR-150 §3.6).
"""
import argparse
import numpy as np
import torch

from model import PoseNet, standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="per-room .adapter.npz (omit for zero-shot)")
    ap.add_argument("--data", required=True, help=".npz with X [N,3,114,10]")
    ap.add_argument("--out", default=None, help="optional .npy to save [N,17,2] keypoints")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device

    net = PoseNet().to(dev)
    net.load_state_dict(torch.load(a.base, map_location=dev), strict=False)
    if a.adapter:
        net.add_lora(r=a.rank).to(dev)
        z = np.load(a.adapter)
        net.load_lora({k: z[k].astype(np.float32) for k in z.files if k.endswith(".A") or k.endswith(".B")})
    net.eval()

    X = torch.tensor(np.load(a.data)["X"].astype(np.float32)).to(dev)
    Xs = standardize(X)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 4096):
            out.append(net(Xs[i:i + 4096]).cpu().numpy())
    kp = np.concatenate(out).reshape(-1, 17, 2)
    print(f"inferred {len(kp)} frames | adapter={'yes' if a.adapter else 'NONE (zero-shot)'}")
    if a.out:
        np.save(a.out, kp)
        print(f"saved keypoints -> {a.out}")


if __name__ == "__main__":
    main()
