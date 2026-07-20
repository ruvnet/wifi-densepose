"""WiFlow-STD compact-variant efficiency sweep (ADR-152) — sequential overnight runner.

Trains compact variants of the upstream WiFlow-STD architecture on the same
data/split as the full-size reference retraining (seed 42, file-level 70/15/15,
upstream dataset.py) and evaluates PCK@10..50 + MPJPE on the full test split and
the corruption-free test subset (file indices < 487).

Training mirrors upstream run.py/train.py defaults except:
- fp32 only (no fp16 autocast / GradScaler — avoids the BN-poisoning trap
  documented in RESULTS.md defect 5; data on disk is already cleaned).
- batch 64 (kept modest: another GPU job may share the 16 GB card tonight).
- scheduler + early stopping keyed on val MPJPE (upstream early-stops on val MPE
  with patience 5; same here).

Usage:
  venv/bin/python sweep/run_sweep.py --dry-run    # param counts only
  nohup venv/bin/python sweep/run_sweep.py > sweep/sweep.log 2>&1 &

Idempotent: variants already present in sweep/results.jsonl are skipped.

NOTE: deployed to ruvultra (~/wiflow-std-bench/sweep) as a standalone file, so
it deliberately inlines its helpers. The reference implementations (upstream
import shim, >1GB np.load mmap patch, key-remap loader, canonical evaluate
loop) live in benchmarks/wiflow-std/_bench_common.py — keep copies in sync.
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
from torch.utils.data import DataLoader, Subset

# csi_windows.npy is ~13 GB; mmap large arrays instead of eagerly loading
# ~15 GB into RAM (same patch as _bench_common._np_load_mmap).
_np_load = np.load


def _np_load_mmap(path, *a, **kw):
    if (isinstance(path, str) and path.endswith('.npy')
            and os.path.getsize(path) > 1 << 30 and 'mmap_mode' not in kw):
        kw['mmap_mode'] = 'r'
    return _np_load(path, *a, **kw)


np.load = _np_load_mmap

BENCH = os.path.expanduser('~/wiflow-std-bench')
SWEEP = os.path.join(BENCH, 'sweep')
sys.path.insert(0, os.path.join(BENCH, 'upstream'))
sys.path.insert(0, SWEEP)

from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders  # noqa: E402
from losses.pose_loss import PoseLoss          # noqa: E402
from utils.metrics import calculate_pck, calculate_mpjpe  # noqa: E402
from model_compact import CompactWiFlowPoseModel, describe  # noqa: E402

VARIANTS = [
    # name, tcn_channels, conv_channels, attn_groups, groups_mode, input_pw_groups
    dict(name='half',    tcn=[270, 220, 170, 120], conv=[4, 8, 16, 32], attn_groups=4,
         groups_mode='gcd20', input_pw_groups=1),
    dict(name='quarter', tcn=[135, 110, 85, 60],   conv=[2, 4, 8, 16],  attn_groups=2,
         groups_mode='gcd20', input_pw_groups=1),
    dict(name='tiny',    tcn=[68, 56, 44, 32],     conv=[2, 4, 8, 16],  attn_groups=2,
         groups_mode='depthwise', input_pw_groups=4),
]

BATCH = 64
EPOCHS = 50
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 5e-5
SEED = 42
CORRUPT_FILE_START = 487  # files 487-499 were zero-filled by clean_nan.py


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(v, dropout=0.5):
    return CompactWiFlowPoseModel(
        tcn_channels=v['tcn'], conv_channels=v['conv'], attn_groups=v['attn_groups'],
        groups_mode=v['groups_mode'], input_pw_groups=v['input_pw_groups'],
        dropout=dropout)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {t: 0.0 for t in (0.1, 0.2, 0.3, 0.4, 0.5)}
    total_mpe, n = 0.0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        bs = by.size(0)
        total_mpe += calculate_mpjpe(out, by) * bs
        pck = calculate_pck(out, by, thresholds=list(totals))
        for t in totals:
            totals[t] += pck[t] * bs
        n += bs
    return {'samples': n, 'mpjpe': total_mpe / n,
            **{f'pck@{int(t * 100)}': totals[t] / n for t in totals}}


def train_variant(v, dataset, device):
    set_seed(SEED)
    train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
        dataset=dataset, batch_size=BATCH, num_workers=2, random_seed=SEED)

    set_seed(SEED)  # re-seed after split so init is split-independent
    model = build_model(v).to(device)
    info = describe(model)
    print(f"[{v['name']}] params={info['params']:,} tcn_groups={info['tcn_groups_per_block']} "
          f"conv_strides={info['conv_strides']} final_width={info['final_width']}", flush=True)

    criterion = PoseLoss(position_weight=1.0, bone_weight=0.2, loss_type='smooth_l1')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=LR / 1000,
        cooldown=1, threshold=1e-4)

    best_val_mpe = float('inf')
    best_val_pck20 = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    t0 = time.time()
    error = None
    epochs_run = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        te = time.time()
        for i, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss, _parts = criterion(out, by)
            if not torch.isfinite(loss):
                error = f'non-finite loss at epoch {epoch} step {i}'
                break
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
            if epoch == 1 and i % 500 == 0:
                print(f"[{v['name']}] e1 step {i}/{len(train_loader)} loss={loss.item():.5f}",
                      flush=True)
        if error:
            break
        epochs_run = epoch

        val = evaluate(model, val_loader, device)
        scheduler.step(val['mpjpe'])
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{v['name']}] epoch {epoch}/{EPOCHS} train_loss={ep_loss / max(nb, 1):.5f} "
              f"val_mpjpe={val['mpjpe']:.5f} val_pck20={val['pck@20'] * 100:.2f}% "
              f"lr={lr_now:.2e} ({time.time() - te:.0f}s)", flush=True)

        if val['mpjpe'] < best_val_mpe:
            best_val_mpe = val['mpjpe']
            best_val_pck20 = val['pck@20']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{v['name']}] early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    train_seconds = time.time() - t0
    result = {
        'variant': v['name'], 'params': info['params'],
        'tcn_channels': v['tcn'], 'conv_channels': v['conv'],
        'attn_groups': v['attn_groups'], 'groups_mode': v['groups_mode'],
        'input_pw_groups': v['input_pw_groups'],
        'tcn_groups_per_block': info['tcn_groups_per_block'],
        'conv_strides': info['conv_strides'], 'final_width': info['final_width'],
        'batch_size': BATCH, 'max_epochs': EPOCHS, 'patience': PATIENCE,
        'lr': LR, 'weight_decay': WEIGHT_DECAY, 'seed': SEED, 'precision': 'fp32',
        'epochs_run': epochs_run, 'best_epoch': best_epoch,
        'best_val_mpjpe': best_val_mpe if best_state else None,
        'best_val_pck20': best_val_pck20 if best_state else None,
        'train_seconds': round(train_seconds, 1),
        'torch': torch.__version__, 'error': error,
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    if best_state is not None:
        ckpt = os.path.join(SWEEP, f"{v['name']}_best.pth")
        torch.save(best_state, ckpt)
        result['checkpoint'] = ckpt
        model.load_state_dict(best_state)

        eval_loader = DataLoader(test_loader.dataset, batch_size=256, shuffle=False,
                                 num_workers=2)
        result['test_full'] = evaluate(model, eval_loader, device)

        w2f = dataset.window_to_file
        clean_idx = [i for i in test_loader.dataset.indices if w2f[i] < CORRUPT_FILE_START]
        clean_loader = DataLoader(Subset(dataset, clean_idx), batch_size=256,
                                  shuffle=False, num_workers=2)
        result['test_clean'] = evaluate(model, clean_loader, device)
        print(f"[{v['name']}] TEST clean: pck20={result['test_clean']['pck@20'] * 100:.2f}% "
              f"mpjpe={result['test_clean']['mpjpe']:.5f} | full: "
              f"pck20={result['test_full']['pck@20'] * 100:.2f}%", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='print param counts and exit')
    args = ap.parse_args()

    if args.dry_run:
        for v in VARIANTS:
            m = build_model(v)
            info = describe(m)
            x = torch.randn(2, 540, 20)
            m.eval()
            y = m(x)
            print(f"{v['name']:8s} params={info['params']:>9,} "
                  f"tcn={v['tcn']} conv={v['conv']} attn_g={v['attn_groups']} "
                  f"mode={v['groups_mode']} pw_g={v['input_pw_groups']} "
                  f"tcn_groups={info['tcn_groups_per_block']} strides={info['conv_strides']} "
                  f"W'={info['final_width']} out={tuple(y.shape)}")
        return

    results_path = os.path.join(SWEEP, 'results.jsonl')
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['variant'])
                except Exception:
                    pass

    device = torch.device('cuda')
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    data_dir = os.path.join(BENCH, 'preprocessed_csi_data')
    dataset = PreprocessedCSIKeypointsDataset(data_dir=data_dir, keypoint_scale=1000.0,
                                              enable_temporal_clean=True)

    for v in VARIANTS:
        if v['name'] in done:
            print(f"[{v['name']}] already in results.jsonl — skipping", flush=True)
            continue
        print(f"\n===== variant: {v['name']} =====", flush=True)
        try:
            result = train_variant(v, dataset, device)
        except Exception as e:  # record and move on to next variant
            import traceback
            traceback.print_exc()
            result = {'variant': v['name'], 'error': repr(e),
                      'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        with open(results_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
            f.flush()
    print('\nSWEEP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
