"""ADR-152 efficiency-sweep follow-up: edge pipeline for the TINY compact
WiFlow-STD variant (56,290 params, results/tiny_best.pth, trained overnight
2026-06-10/11 -- see RESULTS.md "Efficiency sweep").

Headline question: what does the smallest deployable WiFlow-class model look
like (KB + ms + PCK)? Reuses the onnx_bench.py / static_ptq_bench.py
machinery on the tiny checkpoint:

  1. Load tiny_best.pth with remote/sweep/model_compact.py
     (depthwise TCN groups, input_pw_groups=4, conv [2,4,8,16], attn groups 2).
  2. Export ONNX: dynamic batch, opset 17, TorchScript exporter (dynamo=False)
     -- same recipe that worked for the full model; verified at batch 1/2/64.
     One forced deviation: tiny's stride schedule [2,1,1,1] leaves final_width
     16, and the TorchScript exporter cannot export AdaptiveAvgPool2d((15,1))
     when 15 is not a factor of the input height (the full model never hit
     this -- its width was exactly 15). The adaptive pool over a fixed-size
     feature map is a fixed linear map, so the export wrapper replaces it with
     an exact matmul equivalent (PyTorch adaptive-pool bin semantics:
     bin i averages rows floor(i*H/K)..ceil((i+1)*H/K)); the W axis (20->1,
     a factor) becomes mean(-1). Exactness is proven by the parity check
     below, which compares against the ORIGINAL torch model with the real
     AdaptiveAvgPool2d.
  3. Torch-vs-ORT parity on the stored fixture input
     (results/parity_fixture.npz, batch 2, seed 42 -- same 540x20 input layout;
     reference output recomputed with the tiny torch model). PASS < 1e-4.
  4. Static QDQ conv-only int8 (quant_pre_process + quantize_static,
     per-channel QInt8 weights+activations, Percentile(99.99) calibration on
     512 corruption-free TRAIN-split windows -- the winning recipe and
     calibration count from static_ptq_bench.py. 512, not "about 500":
     ORT 1.26's histogram collector np.asarray()'s the per-batch maxima, so
     the calibration count must be a multiple of the batch size 64 or the
     ragged last batch crashes it).
  5. Disk size + CPU latency b1/b64 (3 interleaved reps, median ms/window)
     for tiny fp32 + tiny int8, with the full-model ONNX fp32 + static-int8
     sessions interleaved as same-session references.
  6. Accuracy (PCK@20/50 + MPJPE) on the identical 10k-window seed-42
     corruption-free test subset for tiny fp32 + tiny int8.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe tiny_edge_bench.py \
      [--data-dir <preprocessed_csi_data>] [--subset 10000] [--calib 512]
  (--calib must be a multiple of 64; see step 4 above)

Writes/merges into results/edge_optimization.json under key "tiny_variant".
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "remote", "sweep"))

# quantize_bench sets up upstream imports + the np.load mmap patch
from quantize_bench import build_test_subset  # noqa: E402
from eval_ort_accuracy import evaluate_ort  # noqa: E402
from static_ptq_bench import (  # noqa: E402
    build_calibration_windows,
    interleaved_latency,
    make_reader,
    ort_session,
)
from model_compact import CompactWiFlowPoseModel, describe  # noqa: E402

TINY_CKPT = os.path.join(RESULTS, "tiny_best.pth")
TINY_FP32_ONNX = os.path.join(RESULTS, "tiny_fp32_dynamic.onnx")
TINY_PREPROC_ONNX = os.path.join(RESULTS, "tiny_fp32_preproc.onnx")
TINY_INT8_ONNX = os.path.join(RESULTS, "tiny_int8_static_percentile_conv.onnx")
FULL_FP32_ONNX = os.path.join(RESULTS, "retrained_fp32_dynamic.onnx")
FULL_INT8_ONNX = os.path.join(RESULTS, "retrained_int8_static_percentile_conv.onnx")

# Exact tiny config from remote/sweep/run_sweep.py VARIANTS (measured 56,290
# params, clean-test PCK@20 94.11% -- results/efficiency_sweep.jsonl).
TINY = dict(tcn=[68, 56, 44, 32], conv=[2, 4, 8, 16], attn_groups=2,
            groups_mode="depthwise", input_pw_groups=4)


def load_tiny_model():
    model = CompactWiFlowPoseModel(
        tcn_channels=TINY["tcn"], conv_channels=TINY["conv"],
        attn_groups=TINY["attn_groups"], groups_mode=TINY["groups_mode"],
        input_pw_groups=TINY["input_pw_groups"], dropout=0.5)
    state = torch.load(TINY_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def adaptive_pool_matrix(h_in, h_out):
    """Exact AdaptiveAvgPool1d as a (h_out, h_in) averaging matrix, using
    PyTorch's bin rule: bin i covers rows floor(i*h_in/h_out) ..
    ceil((i+1)*h_in/h_out)."""
    w = torch.zeros(h_out, h_in)
    for i in range(h_out):
        s = (i * h_in) // h_out
        e = -((-(i + 1) * h_in) // h_out)  # ceil division
        w[i, s:e] = 1.0 / (e - s)
    return w


class ExportWrapper(torch.nn.Module):
    """CompactWiFlowPoseModel forward with the AdaptiveAvgPool2d((K,1))
    replaced by an exact fixed linear map (mean over the factor W axis, then
    a constant averaging matmul over the non-factor H axis) so the
    TorchScript ONNX exporter accepts it. Bit-equivalent up to float
    round-off; proven by the parity check against the original model."""

    def __init__(self, m, num_keypoints=15):
        super().__init__()
        self.m = m
        self.register_buffer(
            "pool_w_t", adaptive_pool_matrix(m.final_width, num_keypoints).t())

    def forward(self, x):
        m = self.m
        x = m.tcn(x)
        x = x.transpose(1, 2).unsqueeze(1)
        x = m.up(x)
        for block in m.residual_blocks:
            x = block(x)
        x = x.permute(0, 1, 3, 2)
        x = m.attention(x)
        x = m.decoder(x)                  # [B, 2, H=final_width, T=20]
        x = x.mean(-1)                    # W-axis pool (20 -> 1, a factor)
        x = x.matmul(self.pool_w_t)       # exact adaptive H pool: [B, 2, K]
        return x.transpose(1, 2)          # [B, K, 2]


def export_onnx(model):
    """Dynamic-batch TorchScript export (the recipe that worked for the full
    model in onnx_bench.py), verified at batch 1/2/64. Uses ExportWrapper
    (see docstring) because final_width 16 is not a multiple of 15."""
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(2, 540, 20)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (x,), TINY_FP32_ONNX, opset_version=17,
            input_names=["input"], output_names=["output"], dynamo=False,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    sess = ort_session(TINY_FP32_ONNX)
    inp = sess.get_inputs()[0].name
    for b in (1, 2, 64):
        y = sess.run(None, {inp: np.zeros((b, 540, 20), dtype=np.float32)})[0]
        assert y.shape == (b, 15, 2), y.shape
    return {
        "mode": "dynamic-batch", "exporter": "torchscript", "opset": 17,
        "file": os.path.basename(TINY_FP32_ONNX),
        "size_bytes": os.path.getsize(TINY_FP32_ONNX),
        "size_mb": os.path.getsize(TINY_FP32_ONNX) / 1e6,
        "verified_batches": [1, 2, 64],
        "note": "AdaptiveAvgPool2d((15,1)) replaced at export by an exact "
                "mean(-1) + constant averaging matmul (final_width 16 is not "
                "a multiple of 15, which the TorchScript exporter rejects); "
                "exactness proven by the parity check vs the original torch "
                "model",
    }


def quantize_tiny(calib_windows):
    """quant_pre_process + static QDQ conv-only Percentile(99.99) int8 --
    the winning recipe from static_ptq_bench.py."""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    quant_pre_process(TINY_FP32_ONNX, TINY_PREPROC_ONNX)
    t0 = time.time()
    quantize_static(
        TINY_PREPROC_ONNX, TINY_INT8_ONNX, make_reader(calib_windows),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.99},
    )
    return {
        "file": os.path.basename(TINY_INT8_ONNX),
        "size_bytes": os.path.getsize(TINY_INT8_ONNX),
        "size_mb": os.path.getsize(TINY_INT8_ONNX) / 1e6,
        "calibration": {"method": "percentile", "percentile": 99.99,
                        "windows": int(len(calib_windows)),
                        "scope": "conv-only TRAIN-split corruption-free",
                        "seconds": time.time() - t0},
        "per_channel": True,
        "activation_type": "QInt8",
        "weight_type": "QInt8",
    }


def main():
    import onnxruntime
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "kaka2434",
        "wiflow-dataset", "versions", "1", "preprocessed_csi_data"))
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--calib", type=int, default=512,
                        help="calibration windows; must be a multiple of the "
                             "64-window calibration batch (ORT histogram "
                             "collector rejects ragged batches)")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--out", default=os.path.join(RESULTS, "edge_optimization.json"))
    args = parser.parse_args()

    if args.calib % 64 != 0:
        parser.error(
            f"--calib must be a multiple of 64 (got {args.calib}): ORT 1.26's "
            f"histogram calibration collector np.asarray()'s the per-batch "
            f"maxima and crashes on a ragged final batch (calibration batch "
            f"size is 64)")

    model = load_tiny_model()
    info = describe(model)
    print(f"tiny model: {info['params']:,} params, tcn_groups={info['tcn_groups_per_block']}, "
          f"strides={info['conv_strides']}, final_width={info['final_width']}")
    assert info["params"] == 56290, info["params"]

    results = {
        "env": {
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
            "num_threads": torch.get_num_threads(),
            "checkpoint": os.path.relpath(TINY_CKPT, HERE),
            "checkpoint_size_bytes": os.path.getsize(TINY_CKPT),
            "params": info["params"],
            "variant_config": TINY,
        },
    }

    # ---- export + parity ----------------------------------------------------
    print("\n=== ONNX export (dynamic batch, opset 17, torchscript) ===")
    results["export"] = export_onnx(model)
    print(f"  {results['export']['size_mb']:.3f} MB, batches {results['export']['verified_batches']} OK")

    fixture = np.load(os.path.join(RESULTS, "parity_fixture.npz"))
    fx = fixture["input"]  # (2, 540, 20), seed 42 -- same input layout as full model
    sess_fp32 = ort_session(TINY_FP32_ONNX)
    y_ort = sess_fp32.run(None, {sess_fp32.get_inputs()[0].name: fx})[0]
    with torch.no_grad():
        y_torch = model(torch.from_numpy(fx)).numpy()
    results["parity"] = {
        "fixture": "results/parity_fixture.npz input (batch 2, seed 42); "
                   "reference output recomputed with the tiny torch model",
        "max_abs_diff_vs_torch": float(np.abs(y_ort - y_torch).max()),
        "pass_lt_1e-4": bool(np.abs(y_ort - y_torch).max() < 1e-4),
    }
    print("parity:", json.dumps(results["parity"], indent=2))
    assert results["parity"]["pass_lt_1e-4"], "torch-vs-ORT parity FAILED"

    # ---- static PTQ int8 ------------------------------------------------------
    print(f"\n=== static QDQ int8 (Percentile conv-only, {args.calib} calib windows) ===")
    calib = build_calibration_windows(args.data_dir, args.calib)
    results["int8_static_percentile_conv"] = quantize_tiny(calib)
    print(f"  {results['int8_static_percentile_conv']['size_mb']:.3f} MB")
    sess_int8 = ort_session(TINY_INT8_ONNX)
    yq = sess_int8.run(None, {sess_int8.get_inputs()[0].name: fx})[0]
    results["int8_static_percentile_conv"]["max_abs_diff_vs_fp32_fixture"] = float(
        np.abs(yq - y_torch).max())

    # ---- latency (3 interleaved reps, full-model sessions as references) -----
    print("\n=== latency (3 interleaved reps) ===")
    lat_sessions = {
        "tiny_onnx_fp32": sess_fp32,
        "tiny_onnx_int8_static_percentile_conv": sess_int8,
        "full_onnx_fp32_reference": ort_session(FULL_FP32_ONNX),
        "full_onnx_int8_static_percentile_conv_reference": ort_session(FULL_INT8_ONNX),
    }
    results["latency"] = {
        "note": "3 interleaved repetitions per variant, median ms/window; "
                "full-model sessions are same-session references",
        **interleaved_latency(lat_sessions),
    }

    # ---- accuracy on the standard 10k corruption-free test subset ------------
    if not args.skip_accuracy:
        loader, n_clean = build_test_subset(args.data_dir, args.subset)
        results["accuracy_subset"] = {
            "description": "seed-42 file-level 70/15/15 test split, corrupted "
                           "windows excluded, seed-42 random subset (same as "
                           "quantize_bench/eval_ort_accuracy/static_ptq_bench)",
            "subset_size": min(args.subset, n_clean) if args.subset else n_clean,
        }
        results["accuracy"] = {}
        for name, sess in (("tiny_onnx_fp32", sess_fp32),
                           ("tiny_onnx_int8_static_percentile_conv", sess_int8)):
            print(f"\n=== accuracy: {name} ===")
            results["accuracy"][name] = evaluate_ort(sess, loader, name)
            print(json.dumps(results["accuracy"][name], indent=2))

    # ---- merge into edge_optimization.json -----------------------------------
    merged = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            merged = json.load(f)
    merged["tiny_variant"] = results
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
