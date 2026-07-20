"""ADR-152 §2.2 measurement (a): reproduce WiFlow-STD (DY2434) published test metrics.

Runs the released pretrained checkpoint (upstream/best_pose_model.pth) against the
released Kaggle dataset (kaka2434/wiflow-dataset) using the upstream code path:
identical dataset class, identical file-level 70/15/15 split at seed 42, identical
PCK/MPJPE implementations (utils/metrics.py).

Published claims (README, "Setting 1 random split"):
  PCK@20 97.25% | PCK@30 98.63% | PCK@40 99.16% | PCK@50 99.48% | MPJPE 0.007 m

Usage:
  .venv/Scripts/python.exe eval_repro.py --data-dir <dir containing csi_windows.npy>
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from _bench_common import (UPSTREAM, evaluate, import_upstream,
                           load_remapped_state, set_seed)

import_upstream()  # sys.path + models stub + >1GB np.load mmap patch

from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders  # noqa: E402
from models.pose_model import WiFlowPoseModel  # noqa: E402


def find_data_dir(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "csi_windows.npy" in filenames:
            return dirpath
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing csi_windows.npy (searched recursively)")
    parser.add_argument("--checkpoint", default=os.path.join(UPSTREAM, "best_pose_model.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "results", "repro_a.json"))
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, "csi_windows.npy")):
        located = find_data_dir(data_dir)
        if located is None:
            sys.exit(f"csi_windows.npy not found under {data_dir}")
        data_dir = located
    print(f"data dir: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, torch {torch.__version__}")

    set_seed(42)

    dataset = PreprocessedCSIKeypointsDataset(
        data_dir=data_dir, keypoint_scale=1000.0, enable_temporal_clean=True)

    # split must match upstream: file-level shuffle at random_seed=42, 70/15/15
    _train_loader, _val_loader, test_loader = create_preprocessed_train_val_test_loaders(
        dataset=dataset, batch_size=args.batch_size, num_workers=0, random_seed=42)

    model = WiFlowPoseModel(dropout=0.5).to(device)
    # released checkpoint predates the published code: modules were renamed
    # att -> attention, final_conv -> decoder (param count identical, 2.23M)
    state = load_remapped_state(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {args.checkpoint} ({n_params/1e6:.2f}M params)")

    # upstream also evaluates with drop_last=True; we report the full test set
    # (drop_last=False) and the drop_last variant for exact comparability
    results = {"published": {"pck@20": 0.9725, "pck@30": 0.9863, "pck@40": 0.9916,
                             "pck@50": 0.9948, "mpjpe": 0.007},
               "params_millions": n_params / 1e6,
               "data_dir": data_dir,
               "device": str(device)}

    print("=== test set (full, drop_last=False) ===")
    results["test_full"] = evaluate(model, test_loader, device=device)
    print(json.dumps(results["test_full"], indent=2))

    test_loader_dl = DataLoader(test_loader.dataset, batch_size=args.batch_size,
                                shuffle=False, drop_last=True)
    print("=== test set (drop_last=True, as upstream train.py) ===")
    results["test_drop_last"] = evaluate(model, test_loader_dl, device=device)
    print(json.dumps(results["test_drop_last"], indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
