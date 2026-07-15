"""ADR-152 edge optimization: ONNX export + onnxruntime CPU benchmark for the
retrained WiFlow-STD checkpoint.

- Exports fp32 to ONNX. The axial attention reshapes with python ints taken
  from tensor.size() (view(N*W, C, H)), so a traced graph bakes the batch
  size; we first try a dynamic-batch export and verify it actually works at
  batch sizes 1/2/64 -- if not, we fall back to fixed-batch exports.
- Verifies output parity vs torch on the stored fixture
  (results/parity_fixture.npz, batch 2, seed 42): max abs diff < 1e-4.
- Measures onnxruntime CPU latency at batch 1 and 64 (median of N runs).
- Supplementary: onnxruntime dynamic int8 quantization of the exported model
  (weight size datapoint for the paper's "~2.2 MB int8" claim).

Usage:
  .venv/Scripts/python.exe onnx_bench.py

Writes/merges into results/edge_optimization.json under key "onnx".
"""

import json
import os
import platform
import statistics
import time
import traceback

import numpy as np
import torch

from _bench_common import RESULTS, import_upstream, load_wiflow_model

import_upstream()  # sys.path + models stub + >1GB np.load mmap patch

CHECKPOINT = os.path.join(RESULTS, "retrained_best_pose_model.pth")
OUT_JSON = os.path.join(RESULTS, "edge_optimization.json")


def load_fp32_model():
    return load_wiflow_model(CHECKPOINT)


def try_export(model, path, batch, dynamic, opset=17):
    """Returns (ok, exporter_used, error)."""
    x = torch.rand(batch, 540, 20)
    attempts = []
    if dynamic:
        attempts.append(("dynamo", dict(dynamo=True,
                                        dynamic_shapes={"x": {0: "batch"}})))
        attempts.append(("torchscript", dict(dynamo=False,
                                             dynamic_axes={"input": {0: "batch"},
                                                           "output": {0: "batch"}})))
    else:
        attempts.append(("torchscript", dict(dynamo=False)))
        attempts.append(("dynamo", dict(dynamo=True)))
    last_err = None
    for name, kw in attempts:
        try:
            with torch.no_grad():
                torch.onnx.export(model, (x,), path, opset_version=opset,
                                  input_names=["input"], output_names=["output"],
                                  **kw)
            return True, name, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{name}: {type(e).__name__}: {e}"
            traceback.print_exc()
    return False, None, last_err


def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def ort_run(sess, x):
    inp = sess.get_inputs()[0].name
    return sess.run(None, {inp: x})[0]


def bench_ort(sess, batch, n_runs):
    rng = np.random.default_rng(123)
    x = rng.random((batch, 540, 20), dtype=np.float32)
    for _ in range(max(5, n_runs // 10)):
        ort_run(sess, x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ort_run(sess, x)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {
        "batch_size": batch,
        "runs": n_runs,
        "median_ms_per_batch": med * 1e3,
        "median_ms_per_window": med * 1e3 / batch,
        "windows_per_second": batch / med,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ONNX export + onnxruntime CPU benchmark for the "
                    "retrained WiFlow-STD checkpoint (no options; see "
                    "module docstring). NB: the published "
                    "retrained_fp32_dynamic.onnx came from the TorchScript "
                    "exporter; on newer torch the dynamo attempt may succeed "
                    "first and produce a different (external-data) artifact.")
    parser.parse_args()

    import onnxruntime
    model = load_fp32_model()
    results = {
        "env": {
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
        },
    }

    fixture = np.load(os.path.join(RESULTS, "parity_fixture.npz"))
    fx, fy = fixture["input"], fixture["output"]  # (2,540,20) -> (2,15,2)

    # ---- export: dynamic batch first, fall back to fixed --------------------
    dyn_path = os.path.join(RESULTS, "retrained_fp32_dynamic.onnx")
    ok, exporter, err = try_export(model, dyn_path, batch=2, dynamic=True)
    dynamic_works = False
    if ok:
        # verify the dynamic graph really runs at other batch sizes
        try:
            sess = ort_session(dyn_path)
            for b in (1, 2, 64):
                y = ort_run(sess, np.zeros((b, 540, 20), dtype=np.float32))
                assert y.shape == (b, 15, 2), y.shape
            dynamic_works = True
        except Exception as e:  # noqa: BLE001
            print(f"dynamic-batch model does not generalize: {e}")

    sessions = {}
    if dynamic_works:
        results["export"] = {"mode": "dynamic-batch", "exporter": exporter,
                             "file": os.path.basename(dyn_path),
                             "size_mb": os.path.getsize(dyn_path) / 1e6}
        sess = ort_session(dyn_path)
        sessions = {1: sess, 2: sess, 64: sess}
        print(f"dynamic-batch export OK via {exporter}")
    else:
        results["export"] = {"mode": "fixed-batch", "fallback_reason": err,
                             "files": {}}
        for b in (1, 2, 64):
            p = os.path.join(RESULTS, f"retrained_fp32_b{b}.onnx")
            ok, exporter, err = try_export(model, p, batch=b, dynamic=False)
            if not ok:
                results["export"]["files"][str(b)] = {"error": err}
                print(f"EXPORT FAILED at batch {b}: {err}")
                continue
            results["export"]["files"][str(b)] = {
                "exporter": exporter, "file": os.path.basename(p),
                "size_mb": os.path.getsize(p) / 1e6}
            sessions[b] = ort_session(p)
            print(f"fixed-batch {b} export OK via {exporter}")

    # ---- parity vs torch on the fixture -------------------------------------
    if 2 in sessions:
        y_ort = ort_run(sessions[2], fx)
        with torch.no_grad():
            y_torch = model(torch.from_numpy(fx)).numpy()
        results["parity"] = {
            "fixture": "results/parity_fixture.npz (batch 2, seed 42)",
            "max_abs_diff_vs_stored_fixture": float(np.abs(y_ort - fy).max()),
            "max_abs_diff_vs_torch_now": float(np.abs(y_ort - y_torch).max()),
            "pass_lt_1e-4": bool(np.abs(y_ort - y_torch).max() < 1e-4),
        }
        print("parity:", json.dumps(results["parity"], indent=2))

    # ---- latency -------------------------------------------------------------
    results["latency"] = {}
    if 1 in sessions:
        results["latency"]["batch1"] = bench_ort(sessions[1], 1, 100)
        print(f"ORT batch 1:  {results['latency']['batch1']['median_ms_per_window']:.2f} ms/window")
    if 64 in sessions:
        results["latency"]["batch64"] = bench_ort(sessions[64], 64, 30)
        print(f"ORT batch 64: {results['latency']['batch64']['median_ms_per_window']:.3f} ms/window")

    # ---- supplementary: ORT dynamic int8 (size datapoint for the 2.2MB claim)
    src = (dyn_path if dynamic_works
           else os.path.join(RESULTS, "retrained_fp32_b1.onnx"))
    if os.path.exists(src):
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            q_path = os.path.join(RESULTS, "retrained_int8_ort_dynamic.onnx")
            quantize_dynamic(src, q_path, weight_type=QuantType.QInt8)
            entry = {"file": os.path.basename(q_path),
                     "size_mb": os.path.getsize(q_path) / 1e6}
            try:
                qs = ort_session(q_path)
                yq = ort_run(qs, fx[:1] if not dynamic_works else fx)
                ref = fy[:1] if not dynamic_works else fy
                entry["runs"] = True
                entry["max_abs_diff_vs_fp32_fixture"] = float(np.abs(yq - ref).max())
            except Exception as e:  # noqa: BLE001
                entry["runs"] = False
                entry["run_error"] = f"{type(e).__name__}: {e}"
            results["ort_int8_dynamic_supplementary"] = entry
            print("ORT int8:", json.dumps(entry, indent=2))
        except Exception as e:  # noqa: BLE001
            results["ort_int8_dynamic_supplementary"] = {
                "error": f"{type(e).__name__}: {e}"}

    merged = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            merged = json.load(f)
    merged["onnx"] = results
    with open(OUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
