"""ADR-152 edge optimization follow-up: ONNX Runtime STATIC post-training
quantization (calibration-based QDQ) of the retrained WiFlow-STD model, to
improve on the dynamic-int8 result (2.44 MB, PCK@20 96.52%, 6.5 ms/win b1).

Static PTQ pre-computes activation ranges from calibration data, so inference
uses QLinearConv/QDQ kernels instead of dynamic ConvInteger -- typically both
faster and (with good calibration) closer to fp32 accuracy.

Method:
  - Calibration set: corruption-free windows drawn ONLY from the seed-42
    file-level TRAINING split (same split as eval_repro.py; corrupted windows
    excluded via results/nan_windows_mask.npy | big_windows_mask.npy), chosen
    with np.random.default_rng(42). Never test windows.
  - quantize_static, QuantFormat.QDQ, per-channel int8 weights, int8
    activations; calibration methods MinMax / Entropy / Percentile(99.99);
    scopes "all" (ORT default op set) vs "conv" (op_types_to_quantize=
    ["Conv"] -- leaves the attention path, which exports as Einsum/Softmax
    and elementwise ops, in fp32).
  - Model is pre-processed first (quant_pre_process: symbolic shape
    inference + ORT graph optimization, folds BatchNormalization into Conv).
  - Accuracy: identical protocol to eval_ort_accuracy.py -- the 10,000-window
    seed-42 subset of the corruption-free test split (PCK@20/50, MPJPE).
  - Latency: median ms/window at batch 1 (100 runs) and batch 64 (30 runs),
    3 interleaved repetitions across all variants (fp32 and dynamic-int8
    sessions included as same-session reference points).

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe static_ptq_bench.py \
      [--data-dir <preprocessed_csi_data>] [--subset 10000]
      [--calib-minmax 1000] [--calib-hist 512] [--skip-accuracy]

Writes/merges into results/edge_optimization.json under key "onnx_static_ptq".
"""

import argparse
import collections
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _bench_common import RESULTS  # noqa: E402
# quantize_bench sets up upstream imports + the np.load mmap patch
# (both via _bench_common.import_upstream)
from quantize_bench import build_test_subset  # noqa: E402
import quantize_bench as qb  # noqa: E402
from eval_ort_accuracy import evaluate_ort  # noqa: E402

FP32_ONNX = os.path.join(RESULTS, "retrained_fp32_dynamic.onnx")
DYN_INT8_ONNX = os.path.join(RESULTS, "retrained_int8_ort_dynamic.onnx")
PREPROC_ONNX = os.path.join(RESULTS, "retrained_fp32_preproc.onnx")


# ---------------------------------------------------------------------------
# calibration data: corruption-free TRAINING-split windows only
# ---------------------------------------------------------------------------

def build_calibration_windows(data_dir, n_windows):
    """Seed-42 file-level 70/15/15 TRAIN split (exactly as eval_repro.py),
    minus corrupted windows, then a seed-42 random draw of n_windows."""
    dataset = qb.PreprocessedCSIKeypointsDataset(
        data_dir=data_dir, keypoint_scale=1000.0, enable_temporal_clean=True)
    train_loader, _va, _te = qb.create_preprocessed_train_val_test_loaders(
        dataset=dataset, batch_size=64, num_workers=0, random_seed=42)
    train_indices = np.asarray(train_loader.dataset.indices)

    corrupted = (np.load(os.path.join(RESULTS, "nan_windows_mask.npy"))
                 | np.load(os.path.join(RESULTS, "big_windows_mask.npy")))
    clean = train_indices[~corrupted[train_indices]]
    print(f"train split: {len(train_indices)} windows, "
          f"{len(train_indices) - len(clean)} corrupted excluded, "
          f"{len(clean)} clean")

    rng = np.random.default_rng(42)
    sel = np.sort(rng.choice(clean, size=n_windows, replace=False))
    xs = np.stack([dataset[int(i)][0].numpy() for i in sel]).astype(np.float32)
    print(f"calibration tensor: {xs.shape} from {n_windows} clean TRAIN windows")
    return xs


def make_reader(windows, batch_size=64):
    from onnxruntime.quantization import CalibrationDataReader

    class WindowReader(CalibrationDataReader):
        def __init__(self):
            self._batches = [windows[i:i + batch_size]
                             for i in range(0, len(windows), batch_size)]
            self._it = iter(self._batches)

        def get_next(self):
            b = next(self._it, None)
            return None if b is None else {"input": b}

        def rewind(self):
            self._it = iter(self._batches)

        def __len__(self):
            return len(self._batches)

    return WindowReader()


# ---------------------------------------------------------------------------
# quantization variants
# ---------------------------------------------------------------------------

def preprocess_model():
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(FP32_ONNX, PREPROC_ONNX)
    return PREPROC_ONNX


def quantize_variant(src, dst, method, scope, calib_windows):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    # NB: do NOT pass CalibMaxIntermediateOutputs -- in ORT 1.26 the MinMax
    # calibrater clears its buffer every N batches and then raises
    # "No data is collected" if the batch count is divisible by N.
    extra = {}
    if method == "percentile":
        extra["CalibPercentile"] = 99.99
    op_types = ["Conv"] if scope == "conv" else None

    t0 = time.time()
    quantize_static(
        src, dst, make_reader(calib_windows),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=methods[method],
        extra_options=extra,
    )
    secs = time.time() - t0

    import onnx
    ops = collections.Counter(n.op_type for n in onnx.load(dst).graph.node)
    return {
        "file": os.path.basename(dst),
        "size_bytes": os.path.getsize(dst),
        "size_mb": os.path.getsize(dst) / 1e6,
        "calibration": {"method": method,
                        "windows": int(len(calib_windows)),
                        "percentile": extra.get("CalibPercentile"),
                        "seconds": secs},
        "scope": scope,
        "per_channel": True,
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "node_counts": {k: v for k, v in sorted(ops.items())},
    }


# ---------------------------------------------------------------------------
# latency (3 interleaved reps, like the latency_controlled_rerun)
# ---------------------------------------------------------------------------

def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def bench_ort(sess, batch, n_runs):
    rng = np.random.default_rng(123)
    x = rng.random((batch, 540, 20), dtype=np.float32)
    inp = sess.get_inputs()[0].name
    for _ in range(max(5, n_runs // 10)):
        sess.run(None, {inp: x})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3 / batch  # ms/window


def interleaved_latency(sessions, reps=3, runs_b1=100, runs_b64=30):
    lat = {name: {"batch1_reps": [], "batch64_reps": []} for name in sessions}
    for rep in range(reps):
        for name, sess in sessions.items():
            lat[name]["batch1_reps"].append(bench_ort(sess, 1, runs_b1))
            lat[name]["batch64_reps"].append(bench_ort(sess, 64, runs_b64))
            print(f"  rep {rep + 1}/{reps} {name}: "
                  f"b1={lat[name]['batch1_reps'][-1]:.2f} "
                  f"b64={lat[name]['batch64_reps'][-1]:.3f} ms/win", flush=True)
    for name in lat:
        lat[name]["batch1_ms_per_window_median"] = statistics.median(
            lat[name]["batch1_reps"])
        lat[name]["batch64_ms_per_window_median"] = statistics.median(
            lat[name]["batch64_reps"])
    return lat


# ---------------------------------------------------------------------------

def main():
    import onnxruntime
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "kaka2434",
        "wiflow-dataset", "versions", "1", "preprocessed_csi_data"))
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--calib-minmax", type=int, default=1000)
    parser.add_argument("--calib-hist", type=int, default=512,
                        help="calibration windows for Entropy/Percentile "
                             "(histogram calibraters hold all intermediate "
                             "activations in RAM)")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--methods", default="minmax,entropy,percentile",
                        help="comma list of calibration methods to (re)run; "
                             "results merge into existing onnx_static_ptq")
    parser.add_argument("--out", default=os.path.join(RESULTS, "edge_optimization.json"))
    args = parser.parse_args()

    results = {
        "env": {
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "source_model": os.path.basename(FP32_ONNX),
        },
        "variants": {},
    }

    # ---- calibration data (TRAIN split only) -------------------------------
    calib_mm = build_calibration_windows(args.data_dir, args.calib_minmax)
    calib_hist = calib_mm[:args.calib_hist]

    # ---- preprocess + quantize ---------------------------------------------
    print("\n=== quant_pre_process (shape inference + graph optimization) ===")
    src = preprocess_model()
    results["env"]["preprocessed_model"] = {
        "file": os.path.basename(src),
        "size_mb": os.path.getsize(src) / 1e6,
    }

    matrix = [(m, s) for m in args.methods.split(",")
              for s in ("all", "conv")]
    for method, scope in matrix:
        name = f"{method}_{scope}"
        dst = os.path.join(RESULTS, f"retrained_int8_static_{name}.onnx")
        calib = calib_mm if method == "minmax" else calib_hist
        print(f"\n=== quantize_static: {name} "
              f"({len(calib)} calib windows) ===", flush=True)
        try:
            results["variants"][name] = quantize_variant(
                src, dst, method, scope, calib)
            print(f"  {results['variants'][name]['size_mb']:.3f} MB")
        except Exception as e:  # noqa: BLE001
            results["variants"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {e}")

    # ---- fixture parity (sanity, batch 2) ----------------------------------
    fixture = np.load(os.path.join(RESULTS, "parity_fixture.npz"))
    fx, fy = fixture["input"], fixture["output"]
    sessions = {}
    for name, info in results["variants"].items():
        if "error" in info:
            continue
        path = os.path.join(RESULTS, info["file"])
        try:
            sess = ort_session(path)
            yq = sess.run(None, {sess.get_inputs()[0].name: fx})[0]
            info["max_abs_diff_vs_fp32_fixture"] = float(np.abs(yq - fy).max())
            sessions[name] = sess
        except Exception as e:  # noqa: BLE001
            info["run_error"] = f"{type(e).__name__}: {e}"
    print("\nfixture max-abs-diff vs fp32:",
          {n: round(results["variants"][n].get("max_abs_diff_vs_fp32_fixture",
                                               float("nan")), 5)
           for n in results["variants"]})

    # ---- latency: 3 interleaved reps incl. fp32 + dynamic-int8 reference ----
    print("\n=== latency (3 interleaved reps) ===")
    lat_sessions = {"onnx_fp32": ort_session(FP32_ONNX),
                    "onnx_int8_ort_dynamic": ort_session(DYN_INT8_ONNX)}
    lat_sessions.update(sessions)
    results["latency"] = {
        "note": "3 interleaved repetitions per variant, median ms/window; "
                "onnx_fp32 / onnx_int8_ort_dynamic are same-session references",
        **interleaved_latency(lat_sessions),
    }

    # ---- accuracy on the standard 10k corruption-free test subset ----------
    if not args.skip_accuracy:
        loader, n_clean = build_test_subset(args.data_dir, args.subset)
        results["accuracy_subset"] = {
            "description": "seed-42 file-level 70/15/15 test split, corrupted "
                           "windows excluded, seed-42 random subset (same as "
                           "quantize_bench/eval_ort_accuracy)",
            "subset_size": min(args.subset, n_clean) if args.subset else n_clean,
        }
        for name, sess in sessions.items():
            print(f"\n=== accuracy: {name} ===")
            results["variants"][name]["accuracy"] = evaluate_ort(
                sess, loader, name)
            print(json.dumps(results["variants"][name]["accuracy"], indent=2))

    # ---- merge into edge_optimization.json ----------------------------------
    merged = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            merged = json.load(f)
    prev = merged.get("onnx_static_ptq")
    if prev:  # nested merge so partial --methods reruns don't clobber
        prev["env"] = results["env"]
        prev["variants"].update(results["variants"])
        prev.setdefault("latency", {}).update(results["latency"])
        if "accuracy_subset" in results:
            prev["accuracy_subset"] = results["accuracy_subset"]
    else:
        merged["onnx_static_ptq"] = results
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
