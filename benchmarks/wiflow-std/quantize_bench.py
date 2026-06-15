"""ADR-152 "optimize beyond SOTA": edge-optimization benchmark for the
retrained WiFlow-STD checkpoint (results/retrained_best_pose_model.pth,
~96% PCK@20, fp32 params 2,225,042).

Measures, for fp32 / fp16 / dynamic-int8 torch variants:
  (a) serialized state_dict size on disk,
  (b) CPU inference latency per window at batch 1 and batch 64
      (median of repeated runs, this Windows box),
  (c) accuracy (PCK@20/50 + MPJPE, upstream metrics) on a corruption-free
      random subset of the seed-42 file-level 70/15/15 test split
      (same split as eval_repro.py; corrupted windows 487-499 excluded via
      results/nan_windows_mask.npy | results/big_windows_mask.npy).

Also verifies the paper's "~2.2 MB int8" size claim: reports which layer
types torch dynamic quantization actually converts (the model contains NO
nn.Linear -- it is Conv1d/Conv2d/BatchNorm only) and the real on-disk size.

Usage:
  .venv/Scripts/python.exe quantize_bench.py \
      --data-dir C:/Users/ruv/.cache/kagglehub/datasets/kaka2434/wiflow-dataset/versions/1/preprocessed_csi_data \
      [--subset 10000] [--skip-accuracy]

Writes/merges into results/edge_optimization.json under key "torch".
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _bench_common import HERE, RESULTS, evaluate, import_upstream, load_wiflow_model

import_upstream()  # sys.path + models stub + >1GB np.load mmap patch

from dataset import (  # noqa: E402
    PreprocessedCSIKeypointsDataset,
    create_preprocessed_train_val_test_loaders,
)

CHECKPOINT = os.path.join(RESULTS, "retrained_best_pose_model.pth")


def load_fp32_model():
    # legacy upstream key remap inside is a harmless no-op on this checkpoint
    return load_wiflow_model(CHECKPOINT)


def state_dict_size_bytes(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


def bench_latency(model, batch_size, n_runs, dtype=torch.float32):
    gen = torch.Generator().manual_seed(123)
    x = torch.rand(batch_size, 540, 20, generator=gen).to(dtype)
    with torch.no_grad():
        for _ in range(max(5, n_runs // 10)):  # warmup
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {
        "batch_size": batch_size,
        "runs": n_runs,
        "median_ms_per_batch": med * 1e3,
        "median_ms_per_window": med * 1e3 / batch_size,
        "windows_per_second": batch_size / med,
    }


def build_test_subset(data_dir, subset_size, batch_size=64):
    """Seed-42 file-level 70/15/15 test split (exactly as eval_repro.py),
    minus corrupted windows, then a seed-42 random subset."""
    dataset = PreprocessedCSIKeypointsDataset(
        data_dir=data_dir, keypoint_scale=1000.0, enable_temporal_clean=True)
    _tr, _va, test_loader = create_preprocessed_train_val_test_loaders(
        dataset=dataset, batch_size=batch_size, num_workers=0, random_seed=42)
    test_indices = np.asarray(test_loader.dataset.indices)

    corrupted = (np.load(os.path.join(RESULTS, "nan_windows_mask.npy"))
                 | np.load(os.path.join(RESULTS, "big_windows_mask.npy")))
    clean = test_indices[~corrupted[test_indices]]
    print(f"test split: {len(test_indices)} windows, "
          f"{len(test_indices) - len(clean)} corrupted excluded, "
          f"{len(clean)} clean")

    if subset_size and subset_size < len(clean):
        rng = np.random.default_rng(42)
        clean = np.sort(rng.choice(clean, size=subset_size, replace=False))
    subset = torch.utils.data.Subset(dataset, clean.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    return loader, len(clean)


def quantize_int8_dynamic(fp32_model):
    """torch.ao.quantization.quantize_dynamic on Linear/Conv where supported.
    Returns (model, report) where report documents what actually quantized."""
    qmodel = torch.ao.quantization.quantize_dynamic(
        fp32_model, {nn.Linear, nn.Conv1d, nn.Conv2d}, dtype=torch.qint8)

    quantized, total_params, quant_params = [], 0, 0
    for name, mod in qmodel.named_modules():
        cls = type(mod).__module__ + "." + type(mod).__name__
        if "quantized" in cls:
            w = mod.weight() if callable(getattr(mod, "weight", None)) else None
            numel = w.numel() if w is not None else 0
            quant_params += numel
            quantized.append({"module": name, "class": cls, "params": numel})
    for p in fp32_model.parameters():
        total_params += p.numel()

    n_linear = sum(isinstance(m, nn.Linear) for m in fp32_model.modules())
    n_conv1d = sum(isinstance(m, nn.Conv1d) for m in fp32_model.modules())
    n_conv2d = sum(isinstance(m, nn.Conv2d) for m in fp32_model.modules())
    report = {
        "eligible_module_counts": {
            "nn.Linear": n_linear, "nn.Conv1d": n_conv1d, "nn.Conv2d": n_conv2d},
        "modules_actually_quantized": quantized,
        "n_modules_quantized": len(quantized),
        "params_total": total_params,
        "params_quantized": quant_params,
        "params_quantized_fraction": quant_params / total_params,
    }
    return qmodel, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "kaka2434",
        "wiflow-dataset", "versions", "1", "preprocessed_csi_data"))
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--runs-b1", type=int, default=100)
    parser.add_argument("--runs-b64", type=int, default=30)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--out", default=os.path.join(RESULTS, "edge_optimization.json"))
    args = parser.parse_args()

    torch.manual_seed(42)
    results = {
        "env": {
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "num_threads": torch.get_num_threads(),
            "checkpoint": os.path.relpath(CHECKPOINT, HERE),
        },
        "variants": {},
    }

    # ---- build variants ---------------------------------------------------
    fp32 = load_fp32_model()
    n_params = sum(p.numel() for p in fp32.parameters())
    results["env"]["params"] = n_params
    print(f"fp32 model: {n_params:,} params")

    fp16 = load_fp32_model().half()

    int8, q_report = quantize_int8_dynamic(load_fp32_model())
    results["int8_dynamic_quant_report"] = q_report
    print(f"int8 dynamic: {q_report['n_modules_quantized']} modules quantized, "
          f"{q_report['params_quantized_fraction']*100:.1f}% of params")

    variants = {
        "fp32": (fp32, torch.float32, "retrained_fp32_resaved.pth"),
        "fp16": (fp16, torch.float16, "retrained_fp16.pth"),
        "int8_dynamic": (int8, torch.float32, "retrained_int8_dynamic.pth"),
    }

    # ---- (a) size + (b) latency -------------------------------------------
    for name, (model, dtype, fname) in variants.items():
        path = os.path.join(RESULTS, fname)
        size = state_dict_size_bytes(model, path)
        print(f"\n=== {name}: {size/1e6:.3f} MB on disk ({fname}) ===")
        lat1 = bench_latency(model, 1, args.runs_b1, dtype)
        lat64 = bench_latency(model, 64, args.runs_b64, dtype)
        print(f"  batch 1:  {lat1['median_ms_per_window']:.2f} ms/window "
              f"({lat1['windows_per_second']:.0f}/s)")
        print(f"  batch 64: {lat64['median_ms_per_window']:.3f} ms/window "
              f"({lat64['windows_per_second']:.0f}/s)")
        results["variants"][name] = {
            "file": fname,
            "size_bytes": size,
            "size_mb": size / 1e6,
            "latency_batch1": lat1,
            "latency_batch64": lat64,
        }

    # ---- (c) accuracy ------------------------------------------------------
    if not args.skip_accuracy:
        loader, n_clean = build_test_subset(args.data_dir, args.subset)
        results["accuracy_subset"] = {
            "description": "seed-42 file-level 70/15/15 test split, corrupted "
                           "windows (files 487-499) excluded, seed-42 random "
                           "subset",
            "subset_size": min(args.subset, n_clean) if args.subset else n_clean,
            "clean_test_total": n_clean,
        }
        for name, (model, dtype, _f) in variants.items():
            print(f"\n=== accuracy: {name} ===")
            results["variants"][name]["accuracy"] = evaluate(
                model, loader, dtype=dtype, label=name)
            print(json.dumps(results["variants"][name]["accuracy"], indent=2))

    # ---- merge into edge_optimization.json ---------------------------------
    merged = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            merged = json.load(f)
    merged["torch"] = results
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
