# ADR-185: Python P6 SOTA bindings — AETHER, MERIDIAN, and MAT via PyO3 extras

| Field | Value |
|-------|-------|
| **Status** | Proposed — **P1–P4 implemented & tested** (commits `d060998e3`, `189ac9dfb`, `1c9727f9c`, `0f405213d`) + **leaf-crate hoists done** (`a47bb71b2`/`7ed57f041`/`99fea9df9`); **not yet Accepted** (§6.6 CI gate PARTIAL, §6.7 accuracy bars OPEN — see §13) |
| **Date** | 2026-07-21 (impl status recorded 2026-07-21) |
| **Deciders** | ruv |
| **Codename** | **PIP-TRINITY** — three SOTA subsystems join the `wifi_densepose` wheel |
| **Relates to** | [ADR-117](ADR-117-pip-wifi-densepose-modernization.md) (PIP-PHOENIX — the PyO3 wheel this extends), [ADR-024](ADR-024-contrastive-csi-embedding-model.md) (AETHER contrastive embeddings), [ADR-027](ADR-027-cross-environment-domain-generalization.md) (MERIDIAN domain generalization), [ADR-152](ADR-152-wifi-pose-sota-2026.md) (WiFlow-STD ~96% PCK@20 SOTA bar) |
| **Tracking issue** | TBD — file under RuView issue tracker |

---

## 1. Context

### 1.1 Where ADR-117 stopped

ADR-117 (PIP-PHOENIX) shipped the `wifi-densepose` v2.x PyPI wheel as a PyO3 +
maturin compiled extension (`wifi_densepose._native`) with a pure-Python facade.
The bound surface today (`python/src/bindings/*.rs`, `python/src/lib.rs`):

| Bound today | Crate | Kind |
|---|---|---|
| `CsiFrame`, `Keypoint`, `KeypointType`, `BoundingBox`, `PersonPose`, `PoseEstimate` | `wifi-densepose-core` | P2 core types |
| 4-stage vitals (`BreathingExtractor`, `HeartRateExtractor`, `VitalEstimate`, `VitalReading`, `VitalStatus`) | `wifi-densepose-vitals` | P3 DSP |
| `BfldFrame`, `BfldReport`, `BfldKind` + `PrivacyClass` gate | `wifi-densepose-bfld` | P3.5 / ADR-118 |
| `SensingClient` (WS), `RuViewMqttClient` (MQTT), HA helpers | pure-Python `wifi_densepose.client` | P4 `[client]` extra |

ADR-117's own phase ledger (§6, "P6+ — Deferred") explicitly parked three
higher-value subsystems as post-v2.0.0 work:

> - [ ] `wifi-densepose-nn` bindings … · `wifi-densepose-ruvector` bindings …
> - [ ] MQTT/Matter integration helpers …

and ADR-117 §5.1 deferred `wifi-densepose-mat` (depends on nn) and the RuVector
tier for wheel-size reasons. The three SOTA subsystems that a Python researcher
most wants — re-identification embeddings, cross-environment transfer, and the
disaster-triage tool — are precisely the ones still unreachable from
`pip install wifi-densepose`.

### 1.2 The three subsystems already exist and are tested in Rust

None of this is new research. Each subsystem is a shipped, tested Rust module:

| Subsystem | ADR | Rust location (verified HEAD) | Nature |
|---|---|---|---|
| **AETHER** — contrastive CSI embedding / re-identification | ADR-024 | `wifi-densepose-sensing-server/src/embedding.rs` (`EmbeddingExtractor`, `ProjectionHead`, `CsiAugmenter`, `AetherConfig`, `aether_loss`, `info_nce_loss`, `alignment_metric`, `uniformity_metric`) | Pure-sync DSP + linear algebra; 128-dim L2-normalized embeddings |
| **MERIDIAN** — cross-environment domain generalization | ADR-027 | `wifi-densepose-train` (`domain::{DomainFactorizer, DomainClassifier, GradientReversalLayer, AdversarialSchedule}`, `geometry::{GeometryEncoder, FourierPositionalEncoding, FilmLayer, MeridianGeometryConfig}`, `rapid_adapt::{RapidAdaptation, AdaptationLoss}`, `virtual_aug::VirtualDomainAugmentor`, `eval::CrossDomainEvaluator`) + `wifi-densepose-signal::hardware_norm::{HardwareNormalizer, HardwareType, CanonicalCsiFrame}` | Inference/adaptation path is pure-Rust and **un-gated**; only `model`/`trainer`/`losses` need `tch-backend` (libtorch) |
| **MAT** — Mass Casualty Assessment Tool | (root CLAUDE.md crate table) | `wifi-densepose-mat` (`DisasterResponse`, `DisasterConfig`, `DetectionPipeline`, `EnsembleClassifier`, `TriageCalculator`, `TriageStatus`, `Survivor`, `VitalSignsReading`) | Cargo-feature-gated (`mat`); sync ingest (`push_csi_data`) + async scan loop (`start_scanning`, tokio) |

### 1.3 Why now, and why gated extras

Two forces make P6 timely: (a) the v2.0.0 wheel is stable and its abi3-py310
build matrix is proven, so adding modules is incremental; (b) integrators reading
the ADR-115/ADR-117 notes are asking for Python access to re-identification and
cross-room transfer specifically.

But pulling all three into the **default** wheel would break ADR-117 §5.4's
**≤ 5 MB per-platform wheel budget** and its "no heavy system deps" invariant:

- MAT is already cargo-`mat`-gated upstream *because* it drags in the ML/detection
  stack; the default wheel must not carry it.
- MERIDIAN's training path (`model`/`trainer`/`losses`) is `tch-backend`-gated and
  would pull libtorch (30 MB+), the exact wheel-size risk ADR-117 §5.1 flagged.

So P6 mirrors the existing `[client]` extra pattern (ADR-117 §5.6): each subsystem
becomes an **optional pip extra**, and the compiled surface is **feature-gated in
`wifi-densepose-py`'s `Cargo.toml`** so the default wheel stays lean.

### 1.4 What this ADR is *not*

- Not a port of the Rust subsystems to Python — the Rust workspace stays
  authoritative and unmodified, exactly as ADR-117 §1.3 established.
- Not the `wifi-densepose-nn` / libtorch binding (still deferred; MERIDIAN binds
  only the un-gated inference/adaptation path, not `tch-backend` training).
- Not a change to the default wheel's contents, size budget, or abi3 base.

---

## 2. Gap analysis

| Capability | Rust crate(s) | pip v2.x status | Gap severity |
|---|---|---|---|
| Extract a 128-dim re-ID embedding from a CSI window | `sensing-server::embedding` (AETHER) | Not present | **High** |
| Compare two CSI observations by learned similarity (same room? same person?) | AETHER `EmbeddingExtractor` + cosine | Not present | **High** |
| Hardware-invariant CSI normalization (ESP32 / Intel 5300 / Atheros → canonical 56) | `signal::hardware_norm` (MERIDIAN) | Not present | **High** |
| Geometry-conditioned zero-shot deployment (AP positions → FiLM) | `train::geometry` (MERIDIAN) | Not present | **Medium** |
| 10-second unlabeled few-shot room adaptation | `train::rapid_adapt` (MERIDIAN) | Not present | **Medium** |
| Cross-domain evaluation protocol (in/cross/few-shot MPJPE) | `train::eval` (MERIDIAN) | Not present | **Medium** |
| Disaster-survivor detection + START triage from CSI | `wifi-densepose-mat` | Not present | **Medium** (specialist audience) |

---

## 3. Decision

Adopt **three new optional pip extras**, each binding one SOTA subsystem into the
existing `wifi_densepose` wheel as a dedicated Python submodule, gated behind a
matching Cargo feature so the default wheel is unchanged:

```
pip install wifi-densepose              # unchanged: core + vitals + bfld (≤5 MB)
pip install wifi-densepose[aether]      # + wifi_densepose.aether
pip install wifi-densepose[meridian]    # + wifi_densepose.meridian
pip install wifi-densepose[mat]         # + wifi_densepose.mat   (mirrors upstream `mat` cargo feature)
pip install wifi-densepose[sota]        # convenience: aether + meridian + mat
```

This path is called **PIP-TRINITY**. It reuses ADR-117's established idiom
end-to-end: `#[pyclass]` newtype wrappers holding an `inner` Rust value, `#[new]`
constructors, `#[getter]` accessors, `__repr__`, a per-module `register(m)` fn,
and — critically — **GIL release via `py.allow_threads(|| …)` on every
compute-heavy call**, exactly as `bindings/vitals.rs:229` and `:293` already do.

### 3.1 Feature gating in `wifi-densepose-py`

New Cargo features and optional path-deps in `python/Cargo.toml`; each binding
module is `#[cfg(feature = "…")]`-compiled and conditionally `register()`ed in
`src/lib.rs`, so a default build links none of the three:

```toml
[features]
default = []
aether   = ["dep:wifi-densepose-sensing-server"]
meridian = ["dep:wifi-densepose-train", "dep:wifi-densepose-signal"]
mat      = ["dep:wifi-densepose-mat"]              # upstream `mat` feature flows through
sota     = ["aether", "meridian", "mat"]

[dependencies]
wifi-densepose-sensing-server = { version = "0.3.0", path = "../v2/crates/wifi-densepose-sensing-server", optional = true, default-features = false }
wifi-densepose-train          = { version = "0.3.0", path = "../v2/crates/wifi-densepose-train", optional = true, default-features = false }  # NO tch-backend
wifi-densepose-signal         = { version = "0.3.0", path = "../v2/crates/wifi-densepose-signal", optional = true }
wifi-densepose-mat            = { version = "0.3.0", path = "../v2/crates/wifi-densepose-mat", optional = true, default-features = false }
```

`[project.optional-dependencies]` in `pyproject.toml` gains `aether`, `meridian`,
`mat`, and `sota` keys mirroring the existing `client`/`dev` extras. Because each
extra changes the compiled surface, extras map to **cibuildwheel feature-flag
builds**, not pure-Python markers — the publish workflow (ADR-117 §5.4) gains a
build axis for the `[sota]` wheel variant.

### 3.2 Binding surface — AETHER (`wifi_densepose.aether`)

Backing crate: `wifi-densepose-sensing-server::embedding` (ADR-024 §2.6). The
crate is Axum/tokio-based, so we depend on it `default-features = false` and bind
**only the sync `embedding` types** — never the server/runtime. If the embedding
module cannot be reached without a tokio dependency (Open Question §11.1), the
fallback is to hoist `embedding.rs` into a leaf crate; that is a Rust-side
refactor, not a Python API change.

| Python symbol | Wraps | Signature (Python) |
|---|---|---|
| `AetherConfig` | `AetherConfig` | `AetherConfig(d_model=64, d_proj=128, temperature=0.07, vicreg_alpha=1.0, vicreg_beta=25.0, vicreg_gamma=1.0)` — frozen, `__repr__` |
| `CsiAugmenter` | `CsiAugmenter` | `CsiAugmenter(seed)`; `.augment(window: list[list[float]]) -> list[list[float]]` |
| `EmbeddingExtractor` | `EmbeddingExtractor` | `.embed(csi_features: list[list[float]]) -> list[float]` (128-dim, L2-normed); `.forward_dual(...) -> tuple[PoseEstimate, list[float]]` |
| `aether_loss(...)` | `aether_loss` | returns `AetherLossComponents(total, info_nce, variance, covariance)` — frozen dataclass-like |
| `cosine_similarity(a, b)` | thin helper | `float`; convenience for re-ID scoring (not a re-impl — calls the same dot product) |
| `alignment_metric`, `uniformity_metric` | same | `float` |

GIL strategy: `embed`, `forward_dual`, `augment`, and `aether_loss` wrap their
Rust call in `py.allow_threads(|| …)` — these are pure-sync matrix ops that touch
no Python objects, matching the vitals precedent. A single-frame `embed()` is
sub-millisecond (ADR-024 §2.8 target <1 ms FP32), but batch/augment calls exceed
the 0.5 ms GIL-release threshold ADR-117 §P3 set.

`.pyi` stubs: add `wifi_densepose/aether.pyi` declaring the five classes/functions
with precise numeric types; extend the top-level `wifi_densepose/__init__.pyi`
with a `TYPE_CHECKING`-guarded re-export so `mypy --strict` sees them only when
the extra is installed.

### 3.3 Binding surface — MERIDIAN (`wifi_densepose.meridian`)

Backing crates: `wifi-densepose-train` (inference/adaptation path, **no
`tch-backend`**) + `wifi-densepose-signal::hardware_norm`. The `model`/`trainer`/
`losses` modules are libtorch-gated and are **out of scope** — Python gets the
domain-generalization *inference and calibration* surface, not the training loop.

| Python symbol | Wraps | Signature (Python) |
|---|---|---|
| `HardwareType` | `HardwareType` | `#[pyclass(eq, eq_int, hash, frozen)]` enum: `Esp32S3 / Intel5300 / Atheros / Generic`; `HardwareType.detect(subcarrier_count) -> HardwareType` |
| `HardwareNormalizer` | `HardwareNormalizer` | `.normalize(frame: CsiFrame, hw: HardwareType) -> CanonicalCsiFrame` |
| `CanonicalCsiFrame` | `CanonicalCsiFrame` | frozen; `.amplitudes`, `.phases`, `.hardware_type` getters |
| `GeometryEncoder` | `GeometryEncoder` | `GeometryEncoder(MeridianGeometryConfig)`; `.encode(ap_positions: list[tuple[float,float,float]]) -> list[float]` (64-dim, permutation-invariant) |
| `MeridianGeometryConfig` | `MeridianGeometryConfig` | frozen config |
| `RapidAdaptation` | `RapidAdaptation` | `.calibrate(csi_windows: list[list[list[float]]]) -> AdaptationResult` (10-sec unlabeled few-shot) |
| `AdaptationResult` | `AdaptationResult` | frozen result: `.frames_used`, `.converged`, `.loss` |
| `CrossDomainEvaluator` | `CrossDomainEvaluator` | `.evaluate(...) -> dict[str, float]` (in/cross/few-shot MPJPE, domain-gap ratio) |

GIL strategy: `normalize`, `encode`, `calibrate`, and `evaluate` are wrapped in
`py.allow_threads`. `normalize` targets <50 µs/frame (ADR-027 §4.1) and `encode`
<100 µs (§4.3), but `calibrate` runs contrastive test-time training over 200
frames and is the primary GIL-release beneficiary.

`.pyi` stubs: `wifi_densepose/meridian.pyi`. `DomainFactorizer` /
`GradientReversalLayer` / `VirtualDomainAugmentor` are **training-time only** and
are *not* bound in P6 (they need the tch training loop) — Open Question §11.2
records this boundary.

### 3.4 Binding surface — MAT (`wifi_densepose.mat`)

Backing crate: `wifi-densepose-mat`, bound behind the `[mat]` extra so the
disaster/ML stack never enters the default wheel — mirroring the upstream `mat`
cargo feature exactly. `DisasterResponse::start_scanning` is async (tokio); rather
than bind an event loop, P6 binds the **sync ingest + query surface** and a
single-shot `scan_once()` helper (a sync wrapper over one `scan_cycle`, added
Rust-side if needed — see §11.3).

| Python symbol | Wraps | Signature (Python) |
|---|---|---|
| `DisasterType` | `DisasterType` | `#[pyclass(eq, eq_int, hash, frozen)]` enum: `Earthquake / BuildingCollapse / Avalanche / Flood / Mine / Unknown` |
| `TriageStatus` | `TriageStatus` | frozen enum (START protocol classes) |
| `DisasterConfig` | `DisasterConfig` | builder-style kwargs: `DisasterConfig(disaster_type, sensitivity=0.8, confidence_threshold=0.5, max_depth=5.0)` |
| `DisasterResponse` | `DisasterResponse` | `.push_csi_data(amplitudes, phases)`; `.scan_once()`; `.survivors() -> list[Survivor]`; `.survivors_by_triage(status) -> list[Survivor]` |
| `Survivor` | `Survivor` | frozen: `.id`, `.triage_status`, `.location`, `.vital_signs` getters |
| `VitalSignsReading` | `VitalSignsReading` | frozen: breathing / heartbeat / movement fields |

GIL strategy: `push_csi_data` and `scan_once` wrap the detection-pipeline call in
`py.allow_threads` — the ensemble classifier + localization are the compute-heavy
part and touch no Python state.

`.pyi` stubs: `wifi_densepose/mat.pyi`.

---

## 4. Benchmarking & the measured-vs-claimed parity requirement

A binding that "runs without crashing" is worthless if it silently regresses
accuracy versus the native Rust call. The point of P6 is to prove the Python
surface reproduces the Rust subsystem **bit-for-bit**, then to hold each binding
to the *same* published SOTA bar its ADR already claims.

### 4.1 Parity harness (bit-for-bit, mandatory)

Each subsystem ships a golden-vector parity test. A committed input fixture is
run through **both** a tiny native-Rust reference binary (in
`v2/crates/wifi-densepose-py/tests/golden/`) and the Python binding; the two
outputs must hash-match under SHA-256 (the ADR-028 / ADR-117 §5.7 witness scheme):

- `aether`: identical 128-dim embedding bytes for a fixed CSI window + fixed seed.
- `meridian`: identical `CanonicalCsiFrame` bytes for a fixed ESP32 (64-sub) and
  Intel-5300 (30-sub) frame; identical 64-dim geometry vector for fixed AP set.
- `mat`: identical triage classification + survivor count for a fixed CSI stream.

A mismatch is a **release blocker**, not a warning. This is the "MEASURED, not
CLAIMED" gate the project holds itself to.

**Scope, stated honestly:** parity proves the **strongest claim available today** —
the Python binding is bit-identical to native Rust for the bound surface. It is
**not** accuracy validation. The bound AETHER surface moreover ships *untrained*
(random-init weights; a `load_weights` API exists since `65da488ad` but no trained
checkpoint exists to load — §6.7.2, §13.c.a), so byte-equality here says nothing
about the SOTA accuracy bars in §4.3; those remain OPEN (§6.7, §13.c).

### 4.2 pytest-benchmark micro-benchmarks

Following the existing `python/bench/test_bench_vitals.py` pattern (skipped by
default via `addopts`; run with `pytest python/bench/ --benchmark-only`):

- `python/bench/test_bench_aether.py` — steady-state `embed()` per-window cost;
  assert < 2 ms (ADR-024 §2.8 FP32 target < 1 ms with headroom) and that batched
  `embed()` scales linearly (no accidental O(n²)).
- `python/bench/test_bench_meridian.py` — `normalize()` < 200 µs/frame,
  `encode()` < 200 µs (ADR-027 §4.1/§4.3 targets ×2 headroom).
- `python/bench/test_bench_mat.py` — `scan_once()` per-cycle cost bounded by the
  configured scan interval.

### 4.3 SOTA accuracy bar the binding must reproduce (not merely run)

The parity harness (§4.1) guarantees the Python path is byte-identical to Rust, so
these published numbers are the bar the *binding output* is validated against on a
committed labeled fixture — a regression in any is a binding bug:

| Metric | Bar | Source |
|---|---|---|
| WiFlow-STD pose accuracy | **~96% PCK@20** (MEASURED-EQUIVALENT) | ADR-152 §2.2 |
| Room identification (k-NN on `env_fingerprint`) | **> 95%** | ADR-024 §2.8 |
| Person re-ID mAP | **> 80%** (WhoFi bar 95.5% on NTU-Fi) | ADR-024 §2.8, §1.5 |
| Anomaly detection F1 | **> 0.90** | ADR-024 §2.8 |
| INT8 rank correlation vs FP32 (Spearman) | **> 0.95** | ADR-024 §2.8 |
| Cross-domain MPJPE improvement | **> 20%** vs non-adversarial | ADR-027 §4.2 |
| Domain-gap ratio (cross/in-domain) | **< 1.5** | ADR-027 §4.6 |
| Few-shot MPJPE after 10-sec calibration | within **15%** of in-domain | ADR-027 §4.5 |

---

## 5. Phase ledger

```
P1  ──►  P2  ──►  P3  ──►  P4
aether   meridian  mat      docs +
bindings bindings  behind   examples
                   extra
```

> **Implementation note (2026-07-21):** P1–P4 were built against the **real Rust
> code at HEAD**, not this ADR's proposed surface. Where §3's proposed API named
> functions/fields that do not exist in the crates (e.g. `aether_loss`/VICReg
> components/`alignment_metric`/`forward_dual`, `RapidAdaptation.calibrate`,
> `AdaptationResult.converged`), the coder **did not fabricate them** — the real
> API was bound and the deviation documented in each module header and commit body.
> Treat §3 as the original proposal and the commit messages as the authoritative
> record of what shipped.

### P1 — AETHER bindings (`[aether]` extra) — **DONE** (`d060998e3`; leaf-crate hoist `a47bb71b2`)

- [x] `aether` Cargo feature + gated optional `wifi-densepose-sensing-server` dep;
  default build links **0** sensing-server refs (base wheel stays lean).
- [x] `python/src/bindings/aether.rs` — `AetherConfig` (→ real `EmbeddingConfig`),
  `CsiAugmenter.augment_pair`, `EmbeddingExtractor.embed` (128-dim L2-normed,
  GIL-released), `info_nce_loss`, `cosine_similarity`. **Not bound** (absent in
  `embedding.rs` at HEAD, a Rust-side gap, not fabricated): `aether_loss`/VICReg
  components, `alignment_metric`, `uniformity_metric`, `forward_dual`, `vicreg_*`.
- [x] `#[cfg(feature = "aether")]` gate + facade + `aether.pyi` + `[aether]` extra.
- [x] `python/tests/golden/aether_embedding.sha256` parity fixture:
  `tests/aether_parity.rs` locks the native reference; `tests/test_aether.py`
  asserts identical SHA-256 of the LE-f32 bytes.
- [x] **Verified:** `cargo test --features aether --test aether_parity` → 2/2;
  `pytest tests/test_aether.py` → 9/9.
- [x] **Leaf-crate hoist (`a47bb71b2`):** `embedding.rs` moved into a new
  `wifi-densepose-aether` crate. Measured stripped wheel **~361 KB → ~312 KB** (was
  already ~14× under the 5 MB budget — see §13.a; the hoist's value is build-time
  71 s → 12 s + dep-graph hygiene, not size). No regression: `aether_parity` 2/2,
  `pytest` 9/9, sensing-server 217+388 tests 0 failed, new `wifi-densepose-aether`
  crate 96 passed.

### P2 — MERIDIAN bindings (`[meridian]` extra) — **DONE** (`189ac9dfb`)

- [x] `meridian` feature + gated optional `wifi-densepose-train` (**no `tch-backend`
  — libtorch avoided, confirmed**) + `wifi-densepose-signal` deps.
- [x] `python/src/bindings/meridian.rs` — `HardwareType`/`HardwareNormalizer`/
  `CanonicalCsiFrame` (real API: `normalize(amplitude, phase, hw)` over f64 →
  `Result`; singular `amplitude`/`phase` fields), `MeridianGeometryConfig`/
  `GeometryEncoder` (64-dim, permutation-invariant), `RapidAdaptation`
  (**real API: `push_frame` + `adapt()`**, not the ADR's `calibrate`) →
  `AdaptationResult` (`lora_weights`/`final_loss`/`frames_used`/
  `adaptation_epochs`; **no `converged`**), `CrossDomainEvaluator` + `mpjpe`. All
  compute paths GIL-released. Training-time types (`DomainFactorizer`, GRL,
  `VirtualDomainAugmentor`) correctly left out of P6 scope.
- [x] Gate + facade + `meridian.pyi` + `[meridian]` extra; default dep graph has 0
  train/signal/sensing-server refs.
- [x] `tests/golden/meridian_output.sha256` parity fixture (esp32 + intel canonical
  frames + 64-dim geometry vector + rapid-adapt LoRA weights).
- [x] **Verified:** `cargo test --features meridian --test meridian_parity` → 2/2;
  `pytest tests/test_meridian.py` → 13/13.

### P3 — MAT bindings behind `[mat]` extra — **DONE** (`1c9727f9c`)

- [x] `mat` feature + gated optional `wifi-densepose-mat` dep. **§11.3 resolved: no
  Rust change needed** — the public async `start_scanning()` already runs exactly
  one `scan_cycle` when `continuous_monitoring == false`; the binding forces that
  flag off and drives one cycle on a private current-thread tokio runtime.
- [x] `python/src/bindings/mat.rs` — `DisasterType` (**9 variants at HEAD**, not the
  6 the ADR listed), `TriageStatus` (5, START), `DisasterConfig`,
  `DisasterResponse` (`initialize_event`/`add_zone`/`push_csi_data`/`scan_once`/
  `survivors`/`survivors_by_triage` — `initialize_event`+`add_zone` are **required
  additions** the ADR surface omitted), `Survivor` (`latest_vitals`, since real
  `vital_signs` is a history), `VitalSignsReading`, `ScanZone.rectangle`/`.circle`.
  `push_csi_data`+`scan_once` GIL-released.
- [x] Gate + facade + `mat.pyi` + `[mat]` **and** `[sota]` (superset) extras.
- [x] `tests/golden/mat_result.sha256` parity fixture over a canonical
  `count=<K>;triage_priorities=<sorted>` string (UUIDs/timestamps excluded as
  non-deterministic). **Honest scope: proves binding==native path, NOT live
  detection accuracy** — the synthetic stream yields 1 survivor, triage Delayed.
- [x] **Verified:** `cargo test --features mat --test mat_parity` → 2/2;
  `pytest tests/test_mat.py` → 7/7.

### P4 — Docs, examples, and benchmark suite — **DONE** (`0f405213d`)

- [x] `python/bench/test_bench_{aether,meridian,mat}.py` (pytest-benchmark, §4.2).
  Measured on a `--release --features sota` wheel: AETHER `embed()` ~150 µs
  (target <2 ms), batch 1/8/64 = 140/1091/8509 µs (linear); MERIDIAN `normalize()`
  ~2.2 µs (target <200 µs), `encode()` ~6.9 µs; MAT ingest+`scan_once()` ~40 ms /
  256-frame (< 500 ms). All pass.
- [x] `python/examples/{reid_from_csi,cross_room_calibrate,mat_triage}.py` — typed,
  runnable, `mypy --strict` clean; README SOTA extras table.
- [~] Parity harness wiring into CI as a **release-blocking gate** — golden gates
  are green locally (`cargo test --features sota` → 6/6; 3/3 SHA gates), but the CI
  **wiring** is not done (§6.6 PARTIAL — see §13.b).
- [ ] Update ADR-117 §6 "P6+ Deferred" to point at this ADR — still open.

### P5 — New required follow-ups (blocking Accepted)

See §13. In short: (a) three leaf-crate hoists — **DONE** (`a47bb71b2`/`7ed57f041`/
`99fea9df9`; only MAT was a real budget fix, AETHER was a false alarm), (b) wire the
parity harness into CI as an actual release gate — **still open**, (c) source/generate
labeled fixtures to validate the SOTA accuracy bars (§4.3) for real — **still open**.

### P6+ — Deferred (unchanged from ADR-117)

- [ ] `wifi-densepose-nn` / libtorch bindings (MERIDIAN training loop,
  `DomainFactorizer`, GRL) — still blocked on the libtorch wheel-size question.
- [ ] `wifi-densepose-ruvector` RuVector attention bindings.
- [ ] Matter integration helpers.

---

## 6. Acceptance criteria

Status recorded from the P4 self-verification run (`0f405213d`), reference machine
per ADR-117 §10. **7 of 9 met; 2 remain** — the ADR is therefore **not** Accepted.

- [x] **§6.1** `pip install wifi-densepose` (no extras) → default wheel **279 KB**
  (≤ 5 MB); `build_features()` carries no `p6-*` feature — base wheel byte-for-byte
  unaffected by P6. **PASS**
- [x] **§6.2** `pytest python/tests/test_aether.py -q` — **9/9**, incl. a real
  128-dim `embed()` round-trip asserting L2-norm ≈ 1.0 and byte-identity to the
  golden Rust reference. **PASS**
- [x] **§6.3** `pytest python/tests/test_meridian.py -q` — **13/13**, incl.
  ESP32 (64-sub) **and** Intel-5300 (30-sub) canonicalization hash-matching native
  Rust. **PASS**
- [x] **§6.4** `pytest python/tests/test_mat.py -q` — **7/7**, incl. a fixed CSI
  stream whose triage classification matches native `DisasterResponse` exactly.
  **PASS**
- [x] **§6.5** `pytest python/bench/ --benchmark-only` — all targets met (AETHER
  `embed()` ~150 µs < 2 ms; MERIDIAN `normalize()` ~2.2 µs, `encode()` ~6.9 µs
  < 200 µs; MAT `scan_once()` ~40 ms < 500 ms). **PASS**
- [~] **§6.6** Parity harness (§4.1): all three golden-vector SHA-256 gates green
  (`cargo test --features sota` → 6/6). **But CI wiring** as a release-blocking
  gate is **not done** (out of `python/` scope). **PARTIAL — see §13.b.**
- [ ] **§6.7** SOTA-bar reproduction (§4.3): **definitively OPEN** — cannot be
  closed transitively via the parity harness. Investigated; three concrete reasons:
  1. **The native SOTA numbers aren't reproduced by any committed, runnable-today
     test.** ADR-152 ~96% PCK@20 is a frozen result in
     `benchmarks/wiflow-std/results/eval_retrained.json` that points at an **external
     checkpoint** (`/home/ruvultra/wiflow-std-bench/upstream/test/best_pose_model.pth`,
     not in the repo); the only relevant test `test_wiflow_std_parity.rs` is
     `#![cfg(feature = "tch-backend")]` **and** `#[ignore]`d (needs gitignored
     fixtures + LibTorch). ADR-027's `eval.rs::CrossDomainEvaluator` tests are pure
     unit-math on hand-coded 2–3-element vectors, not dataset accuracy. ADR-024's
     only accuracy-ish test asserts Spearman > 0.90 on **synthetic random**
     embeddings (not real CSI, not the published > 0.95 bar); no room-ID / mAP /
     anomaly-F1 test exists at all.
  2. **The bound AETHER surface ships untrained.** `EmbeddingExtractor`/
     `ProjectionHead` default to random Xavier init (`Linear::with_seed(…,
     2024/2025)`, `embedding.rs:97–98`). A weight-loading API now **does** exist
     (`load_weights`/`save_weights`, `65da488ad` — see §13.c.a), so the earlier
     "no loading path" blocker is removed; but **no trained checkpoint exists** to
     load, so the binding still produces untrained embeddings and cannot validate
     mAP > 80% or any trained-model bar today.
  3. **No committed labeled CSI input/pose-pair data exists** to reuse (MM-Fi/NTU-Fi
     appear only as config-default subcarrier counts / external paths;
     `benchmarks/wiflow-std/results/*.npy` are corruption masks + result summaries,
     not labeled fixtures).
  The §4.1 parity harness proves the **strongest claim available today** — the
  Python binding is bit-identical to native Rust for the bound (untrained) surface.
  That is **not** accuracy validation. **See §13.c.**
- [x] **§6.8** `.pyi` stubs present for all three modules; `mypy --strict` passes on
  the three examples. **PASS**
- [x] **§6.9** `python -c "import wifi_densepose.aether"` (etc.) on the base wheel
  raises a clear `ImportError` naming the missing extra. **PASS**

No regression: 76 pre-existing tests pass on the default wheel. The two unmet
criteria (§6.6 CI wiring, §6.7 accuracy) plus the wheel-size hoists (§13.a) are the
gate to Accepted.

---

## 7. Consequences

### 7.1 Positive

- **Closes the ADR-117 P6 gap**: the three most-requested SOTA subsystems become
  scriptable from Python without touching the Rust workspace.
- **Default wheel stays lean**: feature-gated extras preserve ADR-117 §5.4's ≤ 5 MB
  budget and "no heavy system deps" invariant; MAT's ML stack and MERIDIAN's
  libtorch path never enter the base wheel.
- **Reuses the proven idiom**: no new binding machinery — same `#[pyclass]` +
  `py.allow_threads` + `register()` pattern already shipping in `bindings/vitals.rs`.
- **Prove-everything alignment**: the parity harness makes "the Python binding
  equals the Rust core" a *measured, hash-verified* claim, not an assertion —
  matching the project's MEASURED-vs-CLAIMED discipline.
- **Upstream consistency**: `[mat]` pip extra mirrors the `mat` cargo feature, so
  the Python packaging story matches the Rust one exactly.

### 7.2 Negative

- **cibuildwheel matrix grows**: `[sota]` is a distinct compiled variant, adding a
  build axis (and CI time) beyond ADR-117's 5-wheel abi3 matrix.
- **AETHER's backing crate is server-shaped**: depending on
  `wifi-densepose-sensing-server` (Axum/tokio) risks pulling a runtime into an
  extension module; may force a Rust-side refactor to hoist `embedding.rs` into a
  leaf crate (§11.1).
- **MERIDIAN surface is partial**: training-time types (`DomainFactorizer`, GRL,
  `VirtualDomainAugmentor`) stay unbound until the deferred libtorch tier, so the
  Python API is inference/adaptation-only — potential user confusion (mitigated by
  docs + `.pyi` omissions).
- **Golden fixtures are maintenance surface**: any intentional numeric change in a
  Rust subsystem requires regenerating and re-witnessing its golden vector.

### 7.3 Neutral

- The `[sota]` convenience extra is purely additive; users who want one subsystem
  install one extra.
- No change to the v2.0.0 semver line; extras ship additively as v2.x.y.

---

## 8. Alternatives considered

### Alt-A: Fold all three into the default wheel

Rejected — breaks ADR-117 §5.4's ≤ 5 MB budget, drags MAT's ML stack and (via
MERIDIAN training) libtorch into every install, and contradicts the upstream
`mat` cargo-feature gating.

### Alt-B: Separate PyPI packages (`wifi-densepose-aether`, etc.)

Rejected for the SOTA trio — three packages fragment the import namespace and
duplicate the abi3/cibuildwheel setup. (This remains the right call for the
libtorch `nn` tier per ADR-117 Open Q §11.2, which is genuinely heavy.) Extras of
one wheel keep `wifi_densepose.*` coherent.

### Alt-C: Pure-Python reimplementation of the three subsystems

Rejected explicitly — this is the exact drift ADR-117 §8 Alt-C was created to
exit. A Python reimplementation would immediately begin diverging from the Rust
SOTA and could not pass the §4.1 bit-for-bit parity gate.

### Alt-D: REST/WS client to a running sensing-server for AETHER

Rejected as the primary path — provides zero offline embedding utility and cannot
host the parity harness over local Rust code (same reasoning as ADR-117 §8 Alt-B).
The pure-Python client layer (`[client]`) remains available for streaming.

---

## 9. Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| `wifi-densepose-sensing-server` pulls tokio into the extension module | ~~High~~ **Not realized** | ~~High~~ **Low** | **Measured, not realized:** the stripped `[aether]` wheel was **~361 KB** (14× under budget) even before the hoist — linker DCE (`--gc-sections`) strips the server's unreached Axum/tokio/worldgraph code because the binding reaches only pure-compute symbols. Hoist (`a47bb71b2`) still done for build-time / dep-graph hygiene, not budget. See §11.1, §13.a |
| MERIDIAN accidentally links `tch-backend` (libtorch) via a default feature | Medium | High | Explicit `default-features = false` on `wifi-densepose-train`; CI `auditwheel`/`ldd` check that no libtorch symbol is present in the `[meridian]` wheel |
| `[sota]` build axis blows up cibuildwheel time | Medium | Medium | Build `[sota]` variant only on tagged releases, not every PR |
| Golden vectors drift when a Rust subsystem changes intentionally | Medium | Low | Documented regeneration step + ADR-028 witness re-sign; parity mismatch is a loud release blocker, never silent |
| MAT async-only surface has no clean sync entry point | Medium | Medium | Add sync `scan_once()` wrapper Rust-side (§11.3) before binding |
| Users install base wheel and expect `wifi_densepose.aether` | Low | Low | Clear `ImportError` naming the missing extra (acceptance criterion §6) |

---

## 10. Compatibility

- No change to the default wheel, its abi3-py310 base, or its size budget.
- Extras ship additively on the existing v2.x line; no semver break.
- `[mat]` pip extra ↔ `mat` cargo feature parity is preserved by construction.
- `.pyi` stubs are gated so `mypy --strict` only sees a subsystem when its extra
  is installed.

---

## 11. Open questions

1. **AETHER crate shape** — **RESOLVED (`a47bb71b2`).** The original worry that
   linking `wifi-densepose-sensing-server` would bloat the wheel was **never
   measured** — it reasoned from the dependency tree (server has non-optional
   tokio/Axum ⇒ wheel must be huge). The stripped-release measurement disproves it:
   `[aether]` was **369,782 B (~361 KB)** *before* the hoist — already ~14× under
   the 5 MB budget — and **319,719 B (~312 KB)** after. Linker dead-code elimination
   (`--gc-sections` on the pyo3 cdylib) already strips the server's unreached
   Axum/tokio/worldgraph/ruvector paths because the binding reaches only
   pure-compute symbols. The hoist into `wifi-densepose-aether` was still done — its
   real payoff is **build-time** (`[aether]` alone 71 s → 12 s), **dep-graph
   hygiene** (`python/Cargo.lock` −1238 lines), and **removing latent risk** (a
   future change that makes server code reachable would then genuinely bloat the
   wheel). **Convention note:** measure the stripped release wheel size before
   assuming a dependency-tree risk requires a hoist — linker DCE handles pure-Rust
   unreached code, but native/FFI-bundled deps (e.g. `ort`/ONNX Runtime, see §13.a
   MAT) are *not* stripped and are the real size-risk category.

2. **MERIDIAN training-time types**: `DomainFactorizer`, `GradientReversalLayer`,
   and `VirtualDomainAugmentor` are meaningful only with the tch training loop.
   Confirm they stay unbound in P6 and move with the deferred libtorch tier.
   *Tentative: yes — P6 is inference/adaptation only.*

3. **MAT sync entry point**: `DisasterResponse::start_scanning` is an async tokio
   loop. Does a sync single-cycle `scan_once()` already exist, or must it be added
   Rust-side? *Tentative: add a thin sync `scan_once()` wrapping one `scan_cycle`;
   do not bind an event loop into the extension.*

4. **`[sota]` wheel vs per-extra wheels**: cibuildwheel builds one binary per
   feature-set. Do we publish one `[sota]` wheel and let pip select, or per-extra
   wheels? This affects the number of build variants. *Tentative: single `[sota]`
   superset wheel on tagged releases; base wheel stays feature-free.*

5. **INT8 embedding path in Python**: ADR-024 §2.8 sets an INT8 rank-correlation
   bar. Do we expose the INT8 quantized `embed()` in P6, or FP32 only first?
   *Tentative: FP32 in P6; INT8 follows once the Rust quantized path is stable.*

---

## 12. References

### Internal ADRs
- **ADR-117**: pip modernization via PyO3 + maturin — the wheel this ADR extends;
  §5.1/§5.4/§5.6 (extras + wheel budget), §6 "P6+ Deferred".
- **ADR-024**: Project AETHER — contrastive CSI embedding; §2.6 module surface,
  §2.8 performance/accuracy targets.
- **ADR-027**: Project MERIDIAN — cross-environment domain generalization; §4
  phase acceptance criteria, §4.6 evaluation protocol.
- **ADR-152**: WiFi-Pose SOTA 2026 — WiFlow-STD ~96% PCK@20 MEASURED-EQUIVALENT bar.
- **ADR-028**: ESP32 capability audit / witness scheme — the SHA-256 parity gate
  the §4.1 golden harness reuses.

### Rust source (verified HEAD)
- `v2/crates/wifi-densepose-sensing-server/src/embedding.rs` — AETHER.
- `v2/crates/wifi-densepose-train/src/{domain,geometry,rapid_adapt,virtual_aug,eval}.rs` — MERIDIAN.
- `v2/crates/wifi-densepose-signal/src/hardware_norm.rs` — MERIDIAN HardwareNormalizer.
- `v2/crates/wifi-densepose-mat/src/lib.rs` — MAT.
- `python/src/bindings/vitals.rs` — the `py.allow_threads` GIL-release precedent.
- `python/bench/test_bench_vitals.py` — the pytest-benchmark pattern P4 follows.

---

## 13. Open follow-ups (blocking Accepted)

P1–P4 are real, well-tested progress: **32/32 binding tests** (aether 9, meridian
13, mat 7, + 3 smoke) and **6/6 native parity tests** all pass, verified on the
reference machine. The three leaf-crate hoists (§13.a) are now **done**. Two items
still gate Accepted: **§13.b** (wire the parity harness into CI as a release gate)
and **§13.c** (the SOTA accuracy gap — bindings are structurally *untrained*, and no
eval harness or labeled data exists yet; genuine long-term work, not a quick fix).

### 13.a — Leaf-crate hoists (all three DONE) — one real fix, one minor, one false alarm

All three extras' backing crates carry heavy declared deps, so the hoist was applied
to each. But **measuring the stripped release wheel** (not reasoning from the
dependency tree) showed the wheel-size story differs sharply per extra. Linker
dead-code elimination (`--gc-sections` on the pyo3 cdylib) strips **pure-Rust
unreached** code, so a heavy declared dep tree does **not** imply a big wheel;
**native/FFI-bundled** deps (`ort`/ONNX Runtime's native library) are the exception
— DCE cannot strip them, and those are the real size risk.

| Extra | Commit | Wheel size (stripped) | Verdict |
|---|---|---|---|
| `[aether]` | `a47bb71b2` | **~361 KB → ~312 KB** | **False alarm.** Never breached the 5 MB budget — DCE already stripped the sensing-server's unreached Axum/tokio/worldgraph/ruvector code. Hoist justified by build-time (71 s → 12 s), dep-graph hygiene (`Cargo.lock` −1238 lines), and latent-risk removal — **not** budget. |
| `[mat]` | `7ed57f041` | **8.4 MB → 2.0 MB** | **Real, measured regression.** `wifi-densepose-nn` bundles `ort`/ONNX Runtime, a **native** library DCE does **not** strip → genuine breach. Fix necessary and correctly characterized. |
| `[meridian]` | `99fea9df9` | **1.8 MB → 1.7 MB** | **Real but minor.** Measured from the start; a dead dep removed. Already under budget; small win. `libtorch` correctly avoided throughout (`tch` optional, off). |

These were changes **inside** the upstream `v2/` crates (owned by other agents this
session); the default wheel was unaffected throughout because every extra is
feature-gated off. All three hoists are now landed — the remaining Accepted blockers
are §13.b (CI gate) and §13.c (accuracy fixtures), **not** wheel size.

### 13.b — Wire the parity harness into CI as a real release gate (§6.6)

The three golden-vector SHA-256 gates pass locally (`cargo test --features sota` →
6/6) but are not yet wired into a CI workflow that **blocks release** on mismatch.
Add a job to the ADR-117 §5.4 publish pipeline that runs the native `*_parity.rs`
references + the `pytest` binding checks and fails the release on any divergence.

### 13.c — Close the SOTA accuracy gap (§4.3, §6.7) — genuine long-term work

This is the most important honesty gap and it is **more fundamental than missing
labeled data** (see §6.7 for the three findings). The parity harness proves the
Python binding is **byte-identical to the native Rust path** for the bound, **but
untrained**, surface — it does **not** prove the cited SOTA numbers (ADR-152
~96% PCK@20; ADR-024 room-ID > 95% / re-ID mAP > 80% / anomaly F1 > 0.90; ADR-027
cross-domain MPJPE + 20% / domain-gap < 1.5). Those bars remain **CLAIMED, not
MEASURED** by this work.

Closing it requires three steps, in dependency order:

- **(a) Add trained-weight loading to the AETHER/pose bindings — DONE (`65da488ad`).**
  `EmbeddingExtractor` gained `save_weights(path)` / `load_weights(path)` /
  `param_count` on both the native crate and the Python binding (GIL-released,
  `ValueError`/never-panics on bad input), removing the "structurally untrained, no
  loading path" blocker: a real checkpoint can now be loaded whenever one exists.
  Default construction is unchanged (still random `with_seed` init, clearly labeled
  untrained) — purely additive. **Format tradeoff:** rather than pull in
  `safetensors`/`serde`/`bincode`, the on-disk format is raw little-endian `f32`
  with a 12-byte header (8-byte magic `AETHERW1` + `u32` param count), reusing the
  pre-existing `flatten_weights`/`unflatten_weights` — this deliberately preserves
  `wifi-densepose-aether`'s zero-dependency std-only leaf-crate property from the
  §13.a hoist. **Verified:** `cargo test -p wifi-densepose-aether` 98/98; parity 3/3
  incl. the new cross-language golden `aether_weights_parity.rs` (native Rust and the
  Python binding load the same weight file and produce a byte-identical embedding
  SHA-256, and the loaded weights demonstrably move the output off the random-init
  baseline — not a silent no-op); `pytest test_aether.py` 13/13 (up from 9).
  **This does NOT close §6.7** — it is the *capability* to load weights, not trained
  weights; (b) and (c) below remain, and no SOTA number is validated yet.
- **(b) Commit or source a small labeled CSI fixture** (input CSI + ground-truth
  pose/identity/room labels) — **still OPEN.** Genuine **data-acquisition scope**.
- **(c) Build a real eval harness** computing PCK / mAP / room-ID / anomaly-F1 /
  Spearman on (a)+(b) and asserting the published bars — **still OPEN.**

With (a) landed, the remaining work is (b) and (c): genuine research /
data-acquisition scope beyond one session. This is now purely a data-availability +
missing-eval-infra problem, **not** a binding defect. Status stays **Proposed**
until (b)–(c) land and §4.3 is run for real.
