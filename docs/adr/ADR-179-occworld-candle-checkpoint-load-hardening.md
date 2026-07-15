# ADR-179: `wifi-densepose-occworld-candle` Checkpoint-Load Hardening

| Field | Value |
|-------|-------|
| **Status** | Accepted — 1 HIGH + 2 LOW bugs fixed + pinned (MEASURED on Windows) |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **OCCWORLD-DTYPE** |
| **Reviews** | `wifi-densepose-occworld-candle` (Candle occupancy-world model) |
| **Milestone** | #9 (ungated-crate security sweep) — crate 4 of 4 — **CLOSES the milestone** |

## Context

`wifi-densepose-occworld-candle` is a Candle-based occupancy-world model
(VQ-VAE + transformer over occupancy tokens). The real risk surface for an ML
crate is degenerate-input / malformed-weights handling: a `#[forbid(unsafe_code)]`
crate can still **panic** (a DoS, and under WASM an abort) when a tensor op hits an
inconsistent shape. The crate **builds and tests on Windows**, so all findings are
MEASURED.

## Decision

Fix the three reachable bugs, each pinned by a fails-on-old test; attest the rest
clean with evidence.

### Findings fixed (all MEASURED)

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | **HIGH** | `model.rs:95` (`Dtype::I32 => Some(DType::I64)`) | **Crash on any int32-tensor checkpoint.** An I32 byte buffer (4 B/elem) is handed to `from_raw_buffer(.., I64, shape, ..)`; candle derives `elem_count = data.len()/8`, **halving** the count while keeping the original shape → a tensor that claims 2× its storage. Reading it **panics** with a slice-OOB (`range end index 6 out of range for slice of length 3`) inside candle-core. A checkpoint with any int32 tensor (index/buffer tensors are common in PyTorch exports) → **DoS on load**. | Map `I32 → DType::I32`, `I16 → DType::I16` (both first-class candle dtypes). Pinned by `int32_tensor_loads_with_consistent_shape_and_values` (panics on old, passes on new). |
| 2 | LOW | `inference.rs::predict` | Frame/batch dims weren't validated (only H/W/D were): `f_in > num_frames*2` over-indexes the temporal embedding → a cryptic candle `InvalidIndex` *error* (not a panic — candle bounds-checks); zero frame/batch feeds a zero-element tensor. | Boundary guard rejects zero / over-capacity frame+batch with a clear `ShapeMismatch`. 5 pins. |
| 3 | LOW | `vqvae.rs:141` (`z.elem_count() / last`) | **Divide-by-zero panic** in public `VQCodebook::encode` on a rank-0 / empty-last-dim tensor (`last == 0`). | Fail-closed guard returns a clear error. Pinned by `encode_rejects_scalar_without_panicking`. |

The HIGH finding is the notable one: the crate's own dtype mapping **defeated**
the upstream `safetensors::validate()` byte-length guarantee by misdeclaring the
dtype — the one place malformed/widened weights could reach a panicking candle op.

### Dimensions confirmed clean (with evidence)

- **Panic surface** — grep for `unwrap()/expect()/panic!/unreachable!` across `src/`
  → **zero in production paths**; all ops use `?`/`map_err`; the `last().unwrap_or(&0)`
  is now guarded. `as` casts operate only on config-bounded/internal values.
- **NaN-state-poisoning (the named class) — N/A.** The engine is **stateless between
  `predict` calls** (no persistent world-model buffer to latch into), and input is
  `u8` class indices (non-finite input structurally impossible). NaN weights flow to
  `argmax` (deterministic, bounded to a valid class index) — no panic, no persistence.
- **Unbounded alloc / shape-data mismatch from malformed weights** — defended upstream
  by `safetensors::validate()` (overflow-checked `nelements*dtype.size()` vs declared
  byte range + contiguous-offset + buffer-length checks), rejected before reaching
  candle. Finding #1 was the one place the crate defeated that guarantee.
- **Model/path loading** — `load`/`load_safetensors` check `path.exists()` → typed
  `CheckpointNotFound`; corrupt bytes → `CheckpointParse` (pinned). No path-traversal
  surface (caller-supplied path, opened read-only, never joined with untrusted segments).
- **Secrets** — grep clean (only `token_h`/`token_w` config fields match `token`).
- **Determinism** — the crate's central honesty claim, verified by the pre-existing
  `tests/predict_honesty.rs` (3 tests, still pass).
- `unsafe_code = "forbid"` in the manifest.

## Validation

- `cargo test -p wifi-densepose-occworld-candle --no-default-features` → **31/31**
  (lib 17, checkpoint_loading 4, input_validation 5, predict_honesty 3, doctests 2),
  0 failed.
- `cargo test --workspace --no-default-features` → 0 failed across every crate (a lone
  `wifi-densepose-desktop --test api_integration` "Access is denied (os error 5)" was a
  Windows file-lock/AV flake — re-ran isolated 21/21, unrelated).
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash `f8e76f21…46f7a`
  unchanged (occworld off the signal proof path).

## Consequences

### Positive
- A checkpoint-load DoS (the int32 dtype-widening panic) and two degenerate-input
  panics are closed in the world-model crate, each pinned. **Milestone #9 (all 4
  ungated crates) is complete.**

### Negative / Neutral
- None. Guards reject only malformed/degenerate inputs.

## Links
- ADR-176 / ADR-177 / ADR-178 — sibling Milestone-#9 reviews (ruview-swarm, nvsim, desktop)
