"""ADR-152 SS2.2 measurement (b): WiFlow-STD fine-tuned on our fresh ESP32 paired dataset.

Dataset: ~/wiflow-std-bench/paired-20260610.jsonl -- 2,046 paired windows collected
2026-06-10 22:10-22:40 (ONE subject, ONE room, ONE ESP32 node, varied poses).
Per record: csi = flat float32 list, csi_shape, kp = 17 COCO [x, y] normalized [0,1]
camera coords, conf (MediaPipe mean confidence, all > 0.5 in this set), ts_start/ts_end.
Aligner: scripts/align-ground-truth.js, non-overlapping 20-frame windows (~0.42 s each).

Dataset findings (MEASURED on this file, 2026-06-10):
  - csi_shape is HETEROGENEOUS, not uniformly [70, 20]: 1,347x [70,20], 284x [134,20],
    243x [26,20], 130x [12,20], 42x [20,20]. The ESP32 stream emits mixed frame types
    and the aligner stamps each window's subcarrier count from frame[0]
    (extractCsiMatrix: nSc = window[0].subcarriers), zero-padding/truncating the rest.
    Even native-70 windows contain ~20.4% internally zero-padded short frames
    (subcarriers 40..69 all-zero for those frames).
  - LAYOUT BUG: the aligner fills matrix[f * nSc + s] (frame-major) but declares
    shape [nSc, nFrames]. The true layout is (frame, subcarrier); we reshape
    (nFrames, nSc) and transpose. Confirmed by coherent per-frame zero-tails.
  - Handling here (primary suite, "all2046"): every frame's subcarrier axis is
    linearly resampled to 70 bins (np.interp over a normalized index domain;
    identity for native-70 frames) so the pre-registered n=2,046 and split sizes
    hold. Secondary suite ("native70") restricts to the 1,347 native [70,20]
    windows (temporal 70/15/15 of those) as a homogeneity robustness check.

Pre-registered protocol (followed exactly):
  1. TEMPORAL split (records are time-sorted; asserted): first 70% train (1,432),
     next 15% val (307), last 15% test (307). No shuffling across time. Seed 42
     for everything else.
  2. Model: upstream WiFlow-STD trunk (WiFlowPoseModel) with a learned 1x1 Conv1d
     projection 70->540 prepended, and K=17 via the parameter-free adaptive pool
     (AdaptiveAvgPool2d((17, 1)) instead of (15, 1)) -- pretrained weights load
     for any K. CSI normalization: divide by the TRAIN-split 99th-percentile
     amplitude, clip to [0, 1] (documented in output JSON).
  3. Three runs, <=60 epochs, early-stop patience 8 on val MPJPE, batch 32,
     AdamW, fp32 (no autocast):
       (i)   pretrained-init: trunk init from upstream/test/best_pose_model.pth
             (the measurement-(a) retrained checkpoint, ~96% PCK@20 on WiFlow data;
             key remap att.->attention. / final_conv.->decoder. applied defensively
             as in eval_repro.py -- a no-op for this checkpoint, which already uses
             the new names). Discriminative lr: adapter 1e-4, trunk 1e-5.
       (ii)  scratch: same architecture, random init, all params lr 1e-4.
       (iii) frozen-trunk: pretrained trunk frozen (requires_grad=False AND held in
             .eval() so BatchNorm running stats cannot drift -- pure transfer probe);
             only the 70->540 adapter trains, lr 1e-4.
  4. Metrics on the temporal TEST split: torso-normalized PCK@10/20/30/40/50 and
     MPJPE. Upstream utils/metrics.py calculate_pck(use_torso_norm=True) hardcodes
     NECK_IDX/PELVIS_IDX = 2, 12 -- a 15-keypoint convention that is WRONG for our
     17 COCO keypoints (2 = right_eye, 12 = right_hip). We therefore reimplement the
     identical math (per-frame norm distance, clamp min 0.01, mean over all
     keypoints x frames) with torso = ||l_shoulder(5) - l_hip(11)||.
     Also reported: prediction std across test frames (constant-pose detector;
     must be > 0) and the mean-pose-predictor baseline (train-split mean pose
     evaluated on test -- the honesty bar).

Usage (on ruvultra):
  nice -n 10 nohup ~/wiflow-std-bench/venv/bin/python train_measb.py > train_measb.log 2>&1 &

NOTE: deployed to ruvultra as a standalone single file, so it deliberately
inlines its helpers. The reference implementations (upstream import shim,
np.load mmap patch, key-remap loader, canonical evaluate loop) live in
benchmarks/wiflow-std/_bench_common.py — keep copies in sync.
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

BENCH = os.path.expanduser("~/wiflow-std-bench")
UPSTREAM = os.path.join(BENCH, "upstream")
MEASB = os.path.join(BENCH, "measb")
DATA = os.path.join(BENCH, "paired-20260610.jsonl")
CHECKPOINT = os.path.join(UPSTREAM, "test", "best_pose_model.pth")

sys.path.insert(0, UPSTREAM)

# Upstream defect (1): models/__init__.py imports a name tcn.py does not define.
# Register a stub package so the broken __init__ never executes (as eval_repro.py).
import types  # noqa: E402

_models_pkg = types.ModuleType("models")
_models_pkg.__path__ = [os.path.join(UPSTREAM, "models")]
sys.modules["models"] = _models_pkg

from models.pose_model import WiFlowPoseModel  # noqa: E402

SEED = 42
K = 17
N_SUBC = 70
TRUNK_IN = 540
BATCH = 32          # <= 64 per protocol (GPU shared with the efficiency sweep)
MAX_EPOCHS = 60
PATIENCE = 8
LR_ADAPTER = 1e-4
LR_TRUNK_FT = 1e-5  # 10x lower for the pretrained trunk vs the fresh adapter
L_SHOULDER, L_HIP = 5, 11
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resample_subcarriers(frame_major, n_out=N_SUBC):
    """(nFrames, nSc) -> (nFrames, n_out) by per-frame linear interpolation.

    Identity for nSc == n_out. Normalized index domain [0, 1] on both sides.
    """
    nf, nsc = frame_major.shape
    if nsc == n_out:
        return frame_major
    xi = np.linspace(0.0, 1.0, nsc)
    xo = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(xo, xi, frame_major[f]) for f in range(nf)]).astype(np.float32)


def load_dataset():
    csi, kps, confs, ts, native70 = [], [], [], [], []
    shape_counts = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            nsc, nf = r["csi_shape"]
            shape_counts[f"{nsc}x{nf}"] = shape_counts.get(f"{nsc}x{nf}", 0) + 1
            assert nf == 20, r["csi_shape"]
            # Aligner layout bug: data is frame-major despite the declared
            # [nSc, nFrames] shape -- reshape (nFrames, nSc), then resample the
            # subcarrier axis to 70 and transpose to (70 subcarriers, 20 frames).
            fm = np.asarray(r["csi"], dtype=np.float32).reshape(nf, nsc)
            csi.append(resample_subcarriers(fm).T)
            kp = np.asarray(r["kp"], dtype=np.float32)
            assert kp.shape == (K, 2), kp.shape
            kps.append(kp)
            confs.append(r["conf"])
            ts.append(r["ts_start"])
            native70.append(nsc == N_SUBC)
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)), "records not time-sorted"
    return (np.stack(csi), np.stack(kps), np.asarray(confs, dtype=np.float32),
            np.asarray(native70), shape_counts, ts[0], ts[-1])


def temporal_split(n):
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


class AdaptedWiFlow(nn.Module):
    """1x1 Conv1d adapter 70->540 + upstream WiFlow-STD trunk with K=17 pool head."""

    def __init__(self, k=K, dropout=0.5):
        super().__init__()
        self.adapter = nn.Conv1d(N_SUBC, TRUNK_IN, kernel_size=1)
        nn.init.kaiming_normal_(self.adapter.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.adapter.bias, 0)
        self.trunk = WiFlowPoseModel(dropout=dropout)
        # K=17 via the parameter-free adaptive pool: decoder emits [B, 2, 15, 20]
        # spatial maps; pooling H->17 instead of 15 yields [B, 17, 2] with no new
        # parameters, so the pretrained state_dict loads strict=True for any K.
        self.trunk.avg_pool = nn.AdaptiveAvgPool2d((k, 1))

    def forward(self, x):
        return self.trunk(self.adapter(x))


def load_pretrained_trunk(trunk, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Defensive remap as in eval_repro.py (no-op for the retrained checkpoint).
    renames = {"att.": "attention.", "final_conv.": "decoder."}
    state = {next((new + k[len(old):] for old, new in renames.items()
                   if k.startswith(old)), k): v
             for k, v in state.items()}
    trunk.load_state_dict(state, strict=True)


def pck_torso(pred, target, thresholds=THRESHOLDS):
    """Upstream calculate_pck math, torso = l_shoulder(5)<->l_hip(11) for 17-kp COCO."""
    norm = torch.sqrt(((target[:, L_SHOULDER] - target[:, L_HIP]) ** 2).sum(dim=1))
    norm = torch.clamp(norm, min=0.01)
    dist = torch.sqrt(((pred - target) ** 2).sum(dim=2)) / norm.unsqueeze(1)
    return {f"pck@{int(t * 100)}": (dist <= t).float().mean().item() for t in thresholds}


def mpjpe(pred, target):
    return torch.sqrt(((pred - target) ** 2).sum(dim=2)).mean().item()


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def eval_preds(pred, target):
    out = pck_torso(pred, target)
    out["mpjpe"] = mpjpe(pred, target)
    # Constant-pose detector: std across test frames per coordinate, mean over
    # the 17x2 coordinates. 0.0 == degenerate constant predictor.
    out["pred_std"] = pred.std(dim=0).mean().item()
    return out


def train_run(name, x_tr, y_tr, x_va, y_va, device, pretrained, freeze_trunk,
              lr_trunk):
    set_seed(SEED)
    model = AdaptedWiFlow().to(device)
    if pretrained:
        load_pretrained_trunk(model.trunk, CHECKPOINT)
    if freeze_trunk:
        for p in model.trunk.parameters():
            p.requires_grad = False
        groups = [{"params": model.adapter.parameters(), "lr": LR_ADAPTER}]
    else:
        groups = [{"params": model.adapter.parameters(), "lr": LR_ADAPTER},
                  {"params": model.trunk.parameters(), "lr": lr_trunk}]
    opt = torch.optim.AdamW(groups)
    loss_fn = nn.MSELoss()

    n = len(x_tr)
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        if freeze_trunk:
            model.trunk.eval()  # keep BatchNorm running stats fixed: pure transfer
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        val_mpjpe = mpjpe(predict(model, x_va), y_va)
        history.append({"epoch": epoch, "train_mse": ep_loss / n, "val_mpjpe": val_mpjpe})
        marker = ""
        if val_mpjpe < best_val:
            best_val, best_epoch, bad = val_mpjpe, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            bad += 1
        print(f"[{name}] epoch {epoch:02d} train_mse {ep_loss / n:.6f} "
              f"val_mpjpe {val_mpjpe:.5f}{marker}", flush=True)
        if bad >= PATIENCE:
            print(f"[{name}] early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MEASB, f"{name}_best.pth"))
    return model, {"best_epoch": best_epoch, "best_val_mpjpe": best_val,
                   "epochs_run": len(history), "wall_seconds": round(time.time() - t0, 1),
                   "history": history}


def run_suite(tag, csi, kps, device):
    """Temporal 70/15/15 split, mean-pose baseline, three training runs."""
    n = len(csi)
    tr, va, te = temporal_split(n)
    print(f"=== suite {tag}: n={n} train={tr.stop} val={va.stop - va.start} "
          f"test={te.stop - te.start} ===", flush=True)

    # CSI normalization constant from TRAIN split only.
    train_p99 = float(np.percentile(csi[tr], 99))
    train_max = float(csi[tr].max())
    print(f"[{tag}] train p99={train_p99:.3f} max={train_max:.3f} -> /p99, clip [0,1]",
          flush=True)
    csi_n = np.clip(csi / train_p99, 0.0, 1.0).astype(np.float32)

    x = torch.from_numpy(csi_n).to(device)
    y = torch.from_numpy(kps).to(device)
    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]
    x_te, y_te = x[te], y[te]

    suite = {
        "n_windows": n,
        "split": {"n_train": int(tr.stop), "n_val": int(va.stop - va.start),
                  "n_test": int(te.stop - te.start)},
        "csi_norm": {"method": "divide by train-split p99 amplitude, clip [0,1]",
                     "train_p99": train_p99, "train_max": train_max},
        "runs": {},
    }

    # Honesty bar: mean-pose predictor fit on TRAIN, evaluated on TEST.
    mean_pose = y_tr.mean(dim=0, keepdim=True).expand(len(y_te), -1, -1)
    suite["mean_pose_baseline"] = eval_preds(mean_pose, y_te)
    suite["mean_pose_baseline"]["note"] = "train-split mean pose; pred_std 0 by construction"
    print(f"[{tag}] mean-pose baseline:", json.dumps(suite["mean_pose_baseline"]),
          flush=True)

    configs = [
        ("pretrained", dict(pretrained=True, freeze_trunk=False, lr_trunk=LR_TRUNK_FT)),
        ("scratch", dict(pretrained=False, freeze_trunk=False, lr_trunk=LR_ADAPTER)),
        ("frozen_trunk", dict(pretrained=True, freeze_trunk=True, lr_trunk=0.0)),
    ]
    for name, cfg in configs:
        print(f"=== run: {tag}/{name} {cfg} ===", flush=True)
        model, train_info = train_run(f"{tag}_{name}", x_tr, y_tr, x_va, y_va,
                                      device, **cfg)
        test_metrics = eval_preds(predict(model, x_te), y_te)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        suite["runs"][name] = {"config": cfg, "trainable_params": n_trainable,
                               "train": {k: v for k, v in train_info.items()
                                         if k != "history"},
                               "history": train_info["history"],
                               "test": test_metrics}
        print(f"[{tag}/{name}] TEST:", json.dumps(test_metrics), flush=True)
    return suite


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, torch {torch.__version__}", flush=True)
    set_seed(SEED)

    csi, kps, confs, native70, shape_counts, ts_first, ts_last = load_dataset()
    print(f"shape distribution: {shape_counts}", flush=True)

    results = {
        "protocol": {
            "dataset": DATA, "n_windows": len(csi),
            "ts_first": ts_first, "ts_last": ts_last,
            "conf_mean": float(confs.mean()), "conf_min": float(confs.min()),
            "csi_shape_distribution": shape_counts,
            "csi_layout_note": "aligner stores frame-major data under a transposed "
                               "[nSc, nFrames] shape label; corrected on load",
            "csi_resample": "per-frame linear interp of subcarrier axis to 70 bins "
                            "(identity for native-70 frames); native-70 windows still "
                            "contain ~20.4% internally zero-padded short frames",
            "split": "temporal 70/15/15 (no shuffle across time)",
            "model": "1x1 Conv1d 70->540 adapter + WiFlowPoseModel trunk, "
                     "AdaptiveAvgPool2d((17,1)) head (parameter-free K=17)",
            "checkpoint": CHECKPOINT,
            "checkpoint_note": "measurement-(a) retrained checkpoint (~96% PCK@20 on "
                               "WiFlow data); att./final_conv. remap applied "
                               "defensively (no-op, already new-style keys)",
            "optimizer": f"AdamW, adapter lr {LR_ADAPTER}, fine-tuned trunk lr "
                         f"{LR_TRUNK_FT} (10x lower), scratch all {LR_ADAPTER}",
            "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "precision": "fp32", "seed": SEED,
            "pck": "torso-normalized, torso = ||l_shoulder(5) - l_hip(11)||, "
                   "clamp min 0.01, mean over keypoints x frames "
                   "(upstream math; upstream 2/12 indices are a 15-kp convention)",
        },
        # Primary: all 2,046 windows (pre-registered n), subcarrier axis resampled.
        "all2046": None,
        # Secondary robustness check: the 1,347 native [70,20] windows only.
        "native70": None,
    }

    results["all2046"] = run_suite("all2046", csi, kps, device)
    results["native70"] = run_suite("native70", csi[native70], kps[native70], device)

    out = os.path.join(MEASB, "measurement_b.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
