#!/usr/bin/env python3
"""Train a CSI->pose model on the camera-supervised dataset (ADR-079/180).

Input  : 410-d CSI vector (4 global feats + 6 per-node + 400 signal-field).
Target : 17 COCO keypoints (x,y), normalized 0..1 from the camera (ground truth).
Reports HONEST held-out PCK@k + MPJPE on a chronological val split (the last
20% of the session — never trained on), so the number is not leaked.

Usage (ruvultra venv):
    python wiflow_train.py --data ~/wiflow-room/dataset.jsonl --out ~/wiflow-room/model.pt
"""
import argparse, json, math, os, sys
import numpy as np
import torch, torch.nn as nn


def load(path):
    X, Y, V = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            X.append(r["csi"])                       # 410
            kp = r["kps"]                            # 17 x [x,y,vis]
            Y.append([c for k in kp for c in (k[0], k[1])])   # 34
            V.append([k[2] for k in kp])             # 17 visibilities
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(V, np.float32)


class Net(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, dout), nn.Sigmoid())   # coords in 0..1
    def forward(self, x): return self.net(x)


def pck(pred, gt, vis, thr):
    # pred/gt: [N,34] -> [N,17,2]; PCK@thr in normalized image units, visible kps only
    p = pred.reshape(-1, 17, 2); g = gt.reshape(-1, 17, 2)
    d = np.linalg.norm(p - g, axis=2)             # [N,17]
    m = vis > 0.5
    return float((d[m] < thr).mean()) if m.any() else 0.0, float(d[m].mean()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/wiflow-room/model.pt"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    X, Y, V = load(args.data)
    n = len(X)
    print(f"[train] {n} samples, X={X.shape} Y={Y.shape}")
    if n < 200:
        print("[train] too few samples"); sys.exit(2)

    # chronological split (NOT shuffled) so val is a held-out time segment -> honest
    cut = int(n * 0.8)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-6           # standardize on train only
    Xn = (X - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xn[:cut]).to(dev); Ytr = torch.tensor(Y[:cut]).to(dev)
    Xva = torch.tensor(Xn[cut:]).to(dev); Yva = Y[cut:]; Vva = V[cut:]

    # mean-pose baseline (predict the train-mean pose for everything) — the bar to beat
    mean_pose = Y[:cut].mean(0)
    base_pck, base_mpjpe = pck(np.tile(mean_pose, (len(Yva), 1)), Yva, Vva, 0.10)

    net = Net(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best = (1e9, None)
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), args.bs):
            idx = perm[i:i+args.bs]
            opt.zero_grad(); out = net(Xtr[idx]); loss = lossf(out, Ytr[idx]); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad(): pv = net(Xva).cpu().numpy()
            p10, mpj = pck(pv, Yva, Vva, 0.10); p05, _ = pck(pv, Yva, Vva, 0.05)
            vloss = float(((pv - Yva) ** 2).mean())
            print(f"[train] ep{ep+1:3d} val_mse={vloss:.4f} PCK@0.10={p10*100:.1f}% PCK@0.05={p05*100:.1f}% MPJPE={mpj:.4f}")
            if vloss < best[0]: best = (vloss, {"sd": net.state_dict(), "p10": p10, "p05": p05, "mpj": mpj})

    torch.save({"model": best[1]["sd"], "mu": mu, "sd": sd, "din": X.shape[1]}, args.out)
    print("\n==================== HONEST RESULT (held-out 20%, never trained) ====================")
    print(f"  MEAN-POSE BASELINE : PCK@0.10 = {base_pck*100:.1f}%  MPJPE = {base_mpjpe:.4f}  (the bar to beat)")
    print(f"  CSI->POSE MODEL    : PCK@0.10 = {best[1]['p10']*100:.1f}%  PCK@0.05 = {best[1]['p05']*100:.1f}%  MPJPE = {best[1]['mpj']:.4f}")
    delta = (best[1]['p10'] - base_pck) * 100
    print(f"  VERDICT: model {'BEATS' if delta>1 else 'does NOT beat'} mean-pose baseline by {delta:+.1f} pp "
          f"-> {'real CSI->pose signal' if delta>1 else 'NO usable CSI->pose signal (honest negative)'}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
