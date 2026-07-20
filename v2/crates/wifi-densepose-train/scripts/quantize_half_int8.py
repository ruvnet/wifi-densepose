#!/usr/bin/env python3
"""ADR-175: int8 quantization of the WiFlow-STD "half" pose model + MEASURED accuracy/size trade-off.

Sub-deliverable 8.2 of the benchmark/optimization milestone. Quantizes the 843,834-param
"half" WiFlow-STD pose model to int8 (QAT primary, static-PTQ fallback) and MEASURES the
accuracy delta against the fp32 baseline under ONE locked PCK normalization.

LOCKED NORMALIZATION (ADR-173): torso-diameter PCK — neck(idx 2)->pelvis(idx 12) distance,
exactly the default `use_torso_norm=True` path of upstream `utils/metrics.calculate_pck`,
which is the standard MM-Fi/GraphPose-Fi convention. The SAME `calculate_pck` /
`calculate_mpjpe` from the upstream harness scores BOTH fp32 and int8 so the comparison is
metric-locked. The test split is the seed-42 file-level 70/15/15 test partition (54,000
windows full / 52,560 NaN-free) produced by the SAME loader that produced half_best.pth.

int8 backend: FX graph-mode quantization, fbgemm engine (server x86 int8). Quantized int8
kernels execute on CPU, so int8 eval is CPU; an fp32-CPU baseline is also measured so the
accuracy delta is device-matched (CPU fp32 vs CPU int8), and an fp32-GPU number is reported
for continuity with the sweep's recorded numbers.

REPRODUCE (exact command run for ADR-175, run date 2026-06-15, on host ruvultra / RTX 5080):
  ssh ruvultra 'cd ~/wiflow-std-bench && source venv/bin/activate && \
    python ~/quantize_half_int8.py --mode both --qat-epochs 3 2>&1'

  (the script lives in-repo at v2/crates/wifi-densepose-train/scripts/quantize_half_int8.py;
   it was scp'd to ~/quantize_half_int8.py on ruvultra and invoked as above. It is read-only
   to everything under ~/wiflow-std-bench except that it WRITES its int8 artifacts + a JSON
   results file into ~/wiflow-std-bench/sweep/int8/ — it never modifies half_best.pth or any
   upstream file.)

Everything this script prints to stdout is MEASURED. Nothing is estimated.
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

BENCH = os.path.expanduser('~/wiflow-std-bench')
SWEEP = os.path.join(BENCH, 'sweep')
OUTDIR = os.path.join(SWEEP, 'int8')
sys.path.insert(0, os.path.join(BENCH, 'upstream'))
sys.path.insert(0, SWEEP)

from dataset import (PreprocessedCSIKeypointsDataset,  # noqa: E402
                     create_preprocessed_train_val_test_loaders)
from losses.pose_loss import PoseLoss                  # noqa: E402
from utils.metrics import calculate_pck, calculate_mpjpe  # noqa: E402  LOCKED metric (torso norm)
from model_compact import CompactWiFlowPoseModel, describe  # noqa: E402

# half variant config — IDENTICAL to sweep/run_sweep.py VARIANTS[0] that produced half_best.pth
HALF = dict(tcn=[270, 220, 170, 120], conv=[4, 8, 16, 32], attn_groups=4,
            groups_mode='gcd20', input_pw_groups=1)
HALF_CKPT = os.path.join(SWEEP, 'half_best.pth')
CORRUPT_FILE_START = 487   # files 487-499 were zero-filled by clean_nan.py (same as sweep)
SEED = 42
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)   # PCK@10..50


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_half(dropout=0.5):
    return CompactWiFlowPoseModel(
        tcn_channels=HALF['tcn'], conv_channels=HALF['conv'],
        attn_groups=HALF['attn_groups'], groups_mode=HALF['groups_mode'],
        input_pw_groups=HALF['input_pw_groups'], dropout=dropout)


@torch.no_grad()
def evaluate(model, loader, device):
    """MEASURED PCK@10..50 + MPJPE under the LOCKED torso-diameter normalization."""
    model.eval()
    totals = {t: 0.0 for t in THRESHOLDS}
    total_mpe, n = 0.0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        bs = by.size(0)
        total_mpe += calculate_mpjpe(out, by) * bs
        pck = calculate_pck(out, by, thresholds=list(totals))  # use_torso_norm=True default
        for t in totals:
            totals[t] += pck[t] * bs
        n += bs
    return {'samples': n, 'mpjpe': total_mpe / n,
            **{f'pck@{int(t * 100)}': totals[t] / n for t in totals}}


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def state_dict_size_mb(model, path):
    """On-disk size of the *quantized* checkpoint (int8 weights are packed by fbgemm)."""
    torch.save(model.state_dict(), path)
    return file_size_mb(path)


def loaders():
    set_seed(SEED)
    data_dir = os.path.join(BENCH, 'preprocessed_csi_data')
    dataset = PreprocessedCSIKeypointsDataset(data_dir=data_dir, keypoint_scale=1000.0,
                                              enable_temporal_clean=True)
    train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
        dataset=dataset, batch_size=64, num_workers=2, random_seed=SEED)
    return dataset, train_loader, val_loader, test_loader


def clean_loader_from(dataset, test_loader, bs=256):
    w2f = dataset.window_to_file
    clean_idx = [i for i in test_loader.dataset.indices if w2f[i] < CORRUPT_FILE_START]
    return DataLoader(Subset(dataset, clean_idx), batch_size=bs, shuffle=False, num_workers=2)


def eval_loaders(dataset, test_loader, bs=256):
    full = DataLoader(test_loader.dataset, batch_size=bs, shuffle=False, num_workers=2)
    clean = clean_loader_from(dataset, test_loader, bs=bs)
    return full, clean


# --------------------------------------------------------------- int8 paths (FX graph mode)
def ptq_static(fp32_model, train_loader, calib_batches=64):
    """Static post-training quantization, FX graph mode, fbgemm. CPU int8."""
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    torch.backends.quantized.engine = 'fbgemm'
    m = copy.deepcopy(fp32_model).cpu().eval()
    qconfig = get_default_qconfig('fbgemm')
    qmap = QConfigMapping().set_global(qconfig)
    example = torch.randn(1, 540, 20)
    prepared = prepare_fx(m, qmap, example_inputs=(example,))
    prepared.eval()
    with torch.no_grad():
        for i, (bx, _) in enumerate(train_loader):
            prepared(bx.cpu())
            if i + 1 >= calib_batches:
                break
    return convert_fx(prepared)


def qat(fp32_model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """Quantization-aware training, FX graph mode, fbgemm. Fine-tune fake-quant from fp32, convert. CPU int8."""
    from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    torch.backends.quantized.engine = 'fbgemm'
    set_seed(SEED)
    m = copy.deepcopy(fp32_model).to(device).train()
    qconfig = get_default_qat_qconfig('fbgemm')
    qmap = QConfigMapping().set_global(qconfig)
    example = torch.randn(1, 540, 20).to(device)
    prepared = prepare_qat_fx(m, qmap, example_inputs=(example,))
    prepared.to(device)

    criterion = PoseLoss(position_weight=1.0, bone_weight=0.2, loss_type='smooth_l1')
    opt = torch.optim.AdamW(prepared.parameters(), lr=lr, weight_decay=5e-5, betas=(0.9, 0.999))

    best_val = float('inf')
    best_state = None
    for ep in range(1, epochs + 1):
        prepared.train()
        t0 = time.time()
        ep_loss, nb = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            out = prepared(bx)
            loss, _ = criterion(out, by)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        # eval the fake-quant model on GPU (proxy for int8) to pick the best epoch
        prepared.eval()
        v = evaluate(prepared, val_loader, device)
        print(f"[qat] epoch {ep}/{epochs} train_loss={ep_loss / max(nb,1):.5f} "
              f"val_mpjpe(fakequant)={v['mpjpe']:.5f} val_pck20={v['pck@20']*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v['mpjpe'] < best_val:
            best_val = v['mpjpe']
            best_state = copy.deepcopy(prepared.state_dict())
    if best_state is not None:
        prepared.load_state_dict(best_state)
    prepared.cpu().eval()
    return convert_fx(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['ptq', 'qat', 'both'], default='both')
    ap.add_argument('--qat-epochs', type=int, default=3)
    ap.add_argument('--calib-batches', type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    print(f"torch {torch.__version__} | cuda {torch.cuda.get_device_name(0)} | "
          f"quantized.engine candidates {torch.backends.quantized.supported_engines}", flush=True)

    dataset, train_loader, val_loader, test_loader = loaders()
    test_full, test_clean = eval_loaders(dataset, test_loader)

    # ---------- fp32 baseline (loads half_best.pth strict; same arch as sweep) ----------
    fp32 = build_half().eval()
    state = torch.load(HALF_CKPT, map_location='cpu', weights_only=True)
    fp32.load_state_dict(state, strict=True)
    fp32_size = file_size_mb(HALF_CKPT)
    params = describe(fp32)['params']
    print(f"\n=== fp32 baseline: half_best.pth | params={params:,} | "
          f"on-disk={fp32_size:.3f} MB ===", flush=True)

    results = {
        'host': os.uname().nodename, 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'locked_normalization': 'torso-diameter (neck idx2 -> pelvis idx12), '
                                'upstream calculate_pck use_torso_norm=True (ADR-173 standard)',
        'checkpoint': HALF_CKPT, 'params': params, 'fp32_size_mb': fp32_size,
        'test_split': 'seed-42 file-level 70/15/15 test (full 54000 / clean 52560)',
        'fp32': {}, 'int8': {},
    }

    fp32_gpu = build_half().to(cuda).eval()
    fp32_gpu.load_state_dict(state, strict=True)
    print('[fp32/gpu] full ...', flush=True)
    results['fp32']['gpu_full'] = evaluate(fp32_gpu, test_full, cuda)
    print(json.dumps(results['fp32']['gpu_full']), flush=True)
    print('[fp32/gpu] clean ...', flush=True)
    results['fp32']['gpu_clean'] = evaluate(fp32_gpu, test_clean, cuda)
    print(json.dumps(results['fp32']['gpu_clean']), flush=True)

    print('[fp32/cpu] full (device-matched ref for int8) ...', flush=True)
    results['fp32']['cpu_full'] = evaluate(fp32.to(cpu), test_full, cpu)
    print(json.dumps(results['fp32']['cpu_full']), flush=True)
    print('[fp32/cpu] clean ...', flush=True)
    results['fp32']['cpu_clean'] = evaluate(fp32.to(cpu), test_clean, cpu)
    print(json.dumps(results['fp32']['cpu_clean']), flush=True)

    # ---------- int8 ----------
    def measure_int8(label, qmodel):
        path = os.path.join(OUTDIR, f'half_int8_{label}.pth')
        size = state_dict_size_mb(qmodel, path)
        print(f"[int8/{label}] on-disk={size:.3f} MB | full ...", flush=True)
        full = evaluate(qmodel, test_full, cpu)
        print(json.dumps(full), flush=True)
        print(f"[int8/{label}] clean ...", flush=True)
        clean = evaluate(qmodel, test_clean, cpu)
        print(json.dumps(clean), flush=True)
        results['int8'][label] = {'size_mb': size, 'checkpoint': path,
                                  'cpu_full': full, 'cpu_clean': clean}

    if args.mode in ('ptq', 'both'):
        print("\n=== int8 PTQ (static, FX, fbgemm) ===", flush=True)
        qp = ptq_static(fp32.to(cpu).eval(), train_loader, calib_batches=args.calib_batches)
        measure_int8('ptq_static', qp)

    if args.mode in ('qat', 'both'):
        print(f"\n=== int8 QAT (FX, fbgemm, {args.qat_epochs} epochs from half_best) ===", flush=True)
        qq = qat(fp32, train_loader, val_loader, cuda, epochs=args.qat_epochs)
        measure_int8('qat', qq)

    out = os.path.join(OUTDIR, 'int8_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote', out, flush=True)

    # ---------- comparison table (MEASURED) ----------
    print("\n================= MEASURED COMPARISON (clean test subset, torso-PCK) =================", flush=True)
    base = results['fp32']['cpu_clean']
    print(f"{'model':16s} {'size_MB':>8s} {'pck@20':>8s} {'pck@50':>8s} {'mpjpe':>9s}", flush=True)
    print(f"{'fp32 (cpu)':16s} {fp32_size:8.3f} {base['pck@20']*100:7.2f}% {base['pck@50']*100:7.2f}% {base['mpjpe']:9.6f}", flush=True)
    for label, r in results['int8'].items():
        c = r['cpu_clean']
        d20 = (c['pck@20'] - base['pck@20']) * 100
        d50 = (c['pck@50'] - base['pck@50']) * 100
        print(f"{'int8 '+label:16s} {r['size_mb']:8.3f} {c['pck@20']*100:7.2f}% {c['pck@50']*100:7.2f}% {c['mpjpe']:9.6f}  "
              f"(d_pck20={d20:+.2f}pp d_pck50={d50:+.2f}pp size={fp32_size/r['size_mb']:.2f}x smaller)", flush=True)


if __name__ == '__main__':
    main()
