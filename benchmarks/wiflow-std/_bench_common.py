"""Shared infrastructure for the LOCAL wiflow-std benchmark scripts (ADR-152).

This module is the single canonical implementation of the helpers that were
previously copy-pasted across eval_repro.py / quantize_bench.py /
onnx_bench.py / eval_ort_accuracy.py / export_to_safetensors.py:

  - ``import_upstream()``  -- sys.path setup + the models-package stub that
    works around the upstream import bug, plus the >1GB np.load mmap patch
  - ``install_np_load_mmap_patch()`` -- the mmap patch on its own
  - ``remap_legacy_keys()`` / ``load_remapped_state()`` -- checkpoint
    key remap for the pre-rename released checkpoint
  - ``load_wiflow_model()`` -- WiFlowPoseModel from a checkpoint, eval mode
  - ``set_seed()`` -- mirrors upstream run.py seeding exactly
  - ``evaluate()`` -- THE canonical batch-weighted PCK/MPJPE evaluation loop
    (thresholds 0.1-0.5, upstream utils/metrics.py math); accepts either a
    torch nn.Module or an onnxruntime InferenceSession

The scripts under remote/ deploy to ruvultra as standalone single files and
therefore intentionally inline private copies of these helpers; when editing
them, treat this module as the reference implementation and keep the copies
in sync.
"""

import os
import random
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
RESULTS = os.path.join(HERE, "results")

DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)

# ---------------------------------------------------------------------------
# >1GB np.load mmap patch
# ---------------------------------------------------------------------------

# csi_windows.npy is ~13 GB; mmap large arrays instead of loading into RAM
# (loading it eagerly needs ~15 GB).
_np_load = np.load


def _np_load_mmap(path, *a, **kw):
    if (isinstance(path, str) and path.endswith(".npy")
            and os.path.getsize(path) > 1 << 30 and "mmap_mode" not in kw):
        kw["mmap_mode"] = "r"
    return _np_load(path, *a, **kw)


def install_np_load_mmap_patch():
    """Globally patch np.load so .npy files >1GB are mmap'd read-only.

    Idempotent. Patching the numpy module attribute is equivalent to the
    historical ``upstream_dataset.np.load = _np_load_mmap`` (dataset.np IS
    the numpy module), but works regardless of import order.
    """
    np.load = _np_load_mmap


# ---------------------------------------------------------------------------
# upstream import shim
# ---------------------------------------------------------------------------

def import_upstream(mmap_patch=True):
    """Make the upstream WiFlow-STD clone importable; returns its path.

    Upstream bug: models/__init__.py imports TemporalConvNet, which
    models/tcn.py does not define -- the package fails to import as
    published. Register a stub package so the broken __init__ never
    executes; submodules (models.pose_model etc.) still resolve via
    __path__. Idempotent.
    """
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    if "models" not in sys.modules:
        _models_pkg = types.ModuleType("models")
        _models_pkg.__path__ = [os.path.join(UPSTREAM, "models")]
        sys.modules["models"] = _models_pkg
    if mmap_patch:
        install_np_load_mmap_patch()
    return UPSTREAM


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------

# The released checkpoint predates the published code: modules were renamed
# att -> attention, final_conv -> decoder (param count identical, 2.23M).
LEGACY_RENAMES = {"att.": "attention.", "final_conv.": "decoder."}


def remap_legacy_keys(state):
    """Remap pre-rename state_dict keys; no-op for already-new-style keys."""
    return {next((new + k[len(old):] for old, new in LEGACY_RENAMES.items()
                  if k.startswith(old)), k): v
            for k, v in state.items()}


def load_remapped_state(path, map_location="cpu"):
    """torch.load (weights_only) + legacy key remap."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    return remap_legacy_keys(state)


def load_wiflow_model(checkpoint, map_location="cpu", dropout=0.5):
    """Full-size WiFlowPoseModel from a checkpoint, strict load, eval mode."""
    import_upstream()
    from models.pose_model import WiFlowPoseModel
    model = WiFlowPoseModel(dropout=dropout)
    model.load_state_dict(load_remapped_state(checkpoint, map_location),
                          strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    # mirror upstream run.py exactly
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# THE canonical evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device=None, dtype=None, label="",
             thresholds=DEFAULT_THRESHOLDS, progress_every=50):
    """Batch-weighted PCK/MPJPE over a DataLoader (upstream metrics math).

    ``model`` may be a torch nn.Module (optionally evaluated on ``device``
    with inputs cast to ``dtype``) or an onnxruntime InferenceSession.
    Per-threshold PCK values are independent in upstream calculate_pck, so
    evaluating a superset of thresholds never changes any individual value.

    Returns {"samples", "mpjpe", "pck@10".."pck@50", "wall_seconds"}.
    """
    import_upstream()
    from utils.metrics import calculate_mpjpe, calculate_pck

    is_ort = hasattr(model, "get_inputs")  # onnxruntime InferenceSession
    if is_ort:
        inp = model.get_inputs()[0].name

        def forward(bx):
            return torch.from_numpy(model.run(None, {inp: bx.numpy()})[0])
    else:
        model.eval()

        def forward(bx):
            if device is not None:
                bx = bx.to(device)
            if dtype is not None:
                bx = bx.to(dtype)
            return model(bx).float()

    thresholds = list(thresholds)
    totals = {t: 0.0 for t in thresholds}
    total_mpe, n = 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (bx, by) in enumerate(loader):
            out = forward(bx)
            if device is not None and not is_ort:
                by = by.to(device)
            mpe = calculate_mpjpe(out, by)
            pck = calculate_pck(out, by, thresholds=thresholds)
            bs = by.size(0)
            total_mpe += mpe * bs
            for t in totals:
                totals[t] += pck[t] * bs
            n += bs
            if batch_idx % progress_every == 0:
                tag = f"[{label}] " if label else ""
                pck20 = totals.get(0.2)
                pck20_str = f"pck20={pck20 / n:.4f} " if pck20 is not None else ""
                print(f"  {tag}batch {batch_idx}: n={n} {pck20_str}"
                      f"mpjpe={total_mpe / n:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
    return {
        "samples": n,
        "mpjpe": total_mpe / n,
        **{f"pck@{int(t * 100)}": totals[t] / n for t in thresholds},
        "wall_seconds": time.time() - t0,
    }
