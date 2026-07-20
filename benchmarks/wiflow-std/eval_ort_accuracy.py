"""ADR-152 edge optimization: accuracy of the ONNX fp32 and ORT-dynamic-int8
models on the same corruption-free 10k test subset used by quantize_bench.py.

The torch dynamic-int8 path quantizes nothing (no nn.Linear in the model), so
the only real int8 datapoint for the paper's "~2.2 MB int8" claim is the
onnxruntime dynamically quantized model -- this script measures what that
quantization costs in PCK/MPJPE.

Usage:
  .venv/Scripts/python.exe eval_ort_accuracy.py \
      --data-dir <preprocessed_csi_data> [--subset 10000]

Writes/merges into results/edge_optimization.json under key "onnx_accuracy".
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _bench_common import RESULTS, evaluate  # noqa: E402
from quantize_bench import build_test_subset  # noqa: E402  (sets up upstream imports)


def evaluate_ort(sess, loader, label):
    """ORT-session evaluation via the canonical _bench_common.evaluate loop."""
    return evaluate(sess, loader, label=label)


def main():
    import onnxruntime as ort
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "kaka2434",
        "wiflow-dataset", "versions", "1", "preprocessed_csi_data"))
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--out", default=os.path.join(RESULTS, "edge_optimization.json"))
    args = parser.parse_args()

    loader, _n_clean = build_test_subset(args.data_dir, args.subset)
    results = {}
    for label, fname in (("onnx_fp32", "retrained_fp32_dynamic.onnx"),
                         ("onnx_int8_ort_dynamic", "retrained_int8_ort_dynamic.onnx")):
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            results[label] = {"error": f"{fname} not found; run onnx_bench.py first"}
            continue
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"=== accuracy: {label} ({fname}) ===")
        results[label] = evaluate_ort(sess, loader, label)
        print(json.dumps(results[label], indent=2))

    merged = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            merged = json.load(f)
    merged["onnx_accuracy"] = results
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
