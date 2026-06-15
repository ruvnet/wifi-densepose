"""Evaluate the retrained WiFlow-STD checkpoint (ADR-152 §2.2a fallback).

Scores the model produced by run.py (train_output/best_pose_model.pth or similar)
on the seed-42 test split: full test set AND NaN-free subset (excluding windows
that were zero-filled by clean_nan.py — file indices 487-499).

NOTE: deployed to ruvultra (~/wiflow-std-bench) as a standalone single file,
so it deliberately inlines its helpers. The reference implementations (upstream
import shim, >1GB np.load mmap patch, key-remap loader, canonical evaluate
loop) live in benchmarks/wiflow-std/_bench_common.py — keep copies in sync.
"""
import json, os, random, sys

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

sys.path.insert(0, os.path.expanduser('~/wiflow-std-bench/upstream'))
from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders
from models.pose_model import WiFlowPoseModel
from utils.metrics import calculate_pck, calculate_mpjpe


def find_checkpoint():
    cands = []
    for root, _, files in os.walk(os.path.expanduser('~/wiflow-std-bench/train_output')):
        for f in files:
            if f.endswith('.pth'):
                cands.append(os.path.join(root, f))
    # also upstream/test default output dir
    for root, _, files in os.walk(os.path.expanduser('~/wiflow-std-bench/upstream')):
        for f in files:
            if f.endswith('.pth') and 'best' in f and 'cross_dataset' not in root:
                p = os.path.join(root, f)
                if os.path.getmtime(p) > os.path.getmtime(os.path.expanduser('~/wiflow-std-bench/train.log')) - 86400 * 2:
                    cands.append(p)
    cands = [c for c in cands if not c.endswith('upstream/best_pose_model.pth')]
    if not cands:
        sys.exit('no retrained checkpoint found')
    return max(cands, key=os.path.getmtime)


def evaluate(model, loader, device):
    model.eval()
    totals = {t: 0.0 for t in (0.1, 0.2, 0.3, 0.4, 0.5)}
    total_mpe, n = 0.0, 0
    with torch.no_grad():
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
            **{f'pck@{int(t*100)}': totals[t] / n for t in totals}}


random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

d = os.path.expanduser('~/wiflow-std-bench/preprocessed_csi_data')
dataset = PreprocessedCSIKeypointsDataset(data_dir=d, keypoint_scale=1000.0,
                                          enable_temporal_clean=True)
_, _, test_loader = create_preprocessed_train_val_test_loaders(
    dataset=dataset, batch_size=256, num_workers=2, random_seed=42)

device = torch.device('cuda')
ckpt = find_checkpoint()
print('checkpoint:', ckpt)
model = WiFlowPoseModel(dropout=0.5).to(device)
state = torch.load(ckpt, map_location=device, weights_only=True)
renames = {'att.': 'attention.', 'final_conv.': 'decoder.'}
state = {next((new + k[len(old):] for old, new in renames.items()
               if k.startswith(old)), k): v for k, v in state.items()}
model.load_state_dict(state, strict=True)

results = {'checkpoint': ckpt}
print('=== full test set ===')
results['test_full'] = evaluate(model, test_loader, device)
print(json.dumps(results['test_full'], indent=2))

# NaN-free subset: exclude windows from corrupted files 487-499
test_subset = test_loader.dataset            # Subset(dataset, test_indices)
w2f = dataset.window_to_file
clean_idx = [i for i in test_subset.indices if w2f[i] < 487]
print(f'=== NaN-free test subset ({len(clean_idx)} of {len(test_subset.indices)}) ===')
clean_loader = DataLoader(Subset(dataset, clean_idx), batch_size=256, shuffle=False)
results['test_clean'] = evaluate(model, clean_loader, device)
print(json.dumps(results['test_clean'], indent=2))

out = os.path.expanduser('~/wiflow-std-bench/eval_retrained.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print('wrote', out)
