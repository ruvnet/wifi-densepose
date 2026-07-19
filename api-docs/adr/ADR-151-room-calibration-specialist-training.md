# ADR-151: RuView Per-Room Calibration & Specialized Model Training System

| Field | Value |
|-------|-------|
| **Status** | Accepted — Stages 1–5 implemented (statistical specialists); HF-backbone distillation pending |
| **Date** | 2026-06-09 |
| **Deciders** | ruv |
| **Codebase target** | New `wifi-densepose-calibration` crate (orchestration); `wifi-densepose-train` (`rapid_adapt.rs`, `signal_features.rs`, `trainer.rs`); `wifi-densepose-ruvector` (RVF specialist storage); `wifi-densepose-signal/ruvsense/*` (feature extractors); `wifi-densepose-cli` (`enroll`, `train-room`, `room-status` subcommands) |
| **Relates to** | ADR-135 (Empty-Room Baseline Calibration), ADR-030 (Persistent Field Model), ADR-134 (CIR), ADR-024 (Contrastive CSI Embedding / AETHER), ADR-027 (Cross-Environment Domain Generalization / MERIDIAN), ADR-070 (Self-Supervised Pretraining), ADR-105 (Federated CSI Training), ADR-149 (AetherArena / Hugging Face), ADR-150 (RF Foundation Encoder) |

---

## 1. Context

### 1.1 The thesis — teach the room before you teach the model

RuView's deployment frontier is not a better generic model. ADR-150 documents the wall directly: an MM-Fi pose head scores **81.63% torso-PCK@20 in-domain but ~11.6% leakage-free cross-subject**, and bigger capacity *hurts* cross-subject (transformer 24.8% < conv 27.3%). A single oversized model that "understands the world" overfits the rooms and bodies it has seen. The lever is the opposite of scale: **a small model that understands *one* room and *one* person**, calibrated in minutes, run locally, and specialised per biological signal.

This positions RuView between the two incumbents in ambient sensing:

- **Wearables** — high fidelity, but people forget to wear them, and they only measure the wearer.
- **Cameras** — powerful, but invasive, store identifiable video, and fail in the dark / under covers.

RuView sits in the middle: it learns the *space*, learns the *person*, and tracks biological rhythm (breathing, heartbeat, restlessness, posture, presence) without seeing skin or storing video. Heartbeat and breathing are not visual problems — they are tiny, repeating disturbances in the RF field. Capturing them well is a *calibration* problem, not a *model-size* problem.

### 1.2 What already exists (and what is missing)

The pieces of a calibration→training pipeline exist as disconnected modules. There is no system that runs them end to end and emits a per-room model bank.

| Capability | Status today | Gap |
|------------|--------------|-----|
| Empty-room baseline (environmental fingerprint) | ADR-135 `BaselineCalibration` (Proposed): per-subcarrier amplitude + circular-phase stats, `ruvcal` NVS namespace | Captures the *room*, but there is no step that captures *guided human anchors* on top of it |
| Field eigenstructure | ADR-030 `field_model.rs` (SVD room eigenmodes) | Consumes calibration; not wired to a training trigger |
| Shared invariant backbone | ADR-150 RF Foundation Encoder (pose-preserving, subject/room/device-invariant) | Defined as a *foundation* embedding; nothing distills it into per-room specialists |
| Few-shot adaptation | `train/src/rapid_adapt.rs` — test-time training → LoRA weight deltas (MERIDIAN P5) | Produces a *single* pose-adaptation delta, not a bank of per-modality specialists |
| Feature extractors | `ruvsense/{bvp,longitudinal,intention,gesture,pose_tracker,adversarial}.rs`, `train/src/signal_features.rs` | Each emits a signal; none is packaged as a labelled training source for enrollment |
| Small-model storage | `wifi-densepose-ruvector` (RVF cognitive containers, HNSW, sketch) | No schema for "a bank of specialist models scoped to a room_id" |
| HF publishing | ADR-149 AetherArena (Hugging Face Space + signed scorer), `sensing-server` `from_pretrained` path | Publishes/評価s a *global* model; no notion of a published *base* + private *local* heads |

**The missing system is the connective tissue**: a guided enrollment protocol, a feature-extraction-to-label bridge, a specialist-bank trainer that reuses the frozen HF backbone, and a runtime that fuses the specialists with confidence gating. This ADR defines that system.

### 1.3 The four-step user model (and where each step lands)

The system is deliberately presented to operators as four plain steps. Each maps to existing or new code:

1. **Capture a quiet baseline** — no people, just room/router/reflections/noise/drift → the *environmental fingerprint*. → **Reuse ADR-135** `BaselineCalibration` + **ADR-030** field eigenmodes. No new capture code; the calibration crate calls it.
2. **Capture guided samples** — stand, sit, lie down, slow vs normal breathing, small movement, sleep posture. Clean anchors, not hours of data. → **NEW** `EnrollmentProtocol` (Section 2.2).
3. **Extract the useful signal** — CSI phase, amplitude, Doppler shift, micro-motion, periodicity, variance, timing. → **Reuse** `signal_features.rs` + ruvsense extractors, packaged as labelled `AnchorFeature` records (Section 2.3).
4. **Compress patterns into small ruVector models** — *specialised* per signal: breathing, heartbeat, sleep restlessness, posture, presence, anomaly. → **NEW** `SpecialistBank` trained via `rapid_adapt` LoRA heads over the frozen ADR-150 backbone, stored as RVF (Section 2.4).

---

## 2. Decision

**Build the RuView Per-Room Calibration & Specialized Model Training System: a four-stage, local-first pipeline (`baseline → enroll → extract → train`) that produces a versioned *bank of small specialised ruVector models* scoped to one `room_id`, each a lightweight head distilled/adapted from the frozen, Hugging-Face-published RF Foundation Encoder (ADR-150).** Big model understands the world; small ruVector models understand *your room*.

Two invariants govern every design choice below:

> **(A) Specialisation over scale.** One small model per biological signal, not one large model for all of them. Each specialist is faster, cheaper, more private, and — because it is calibrated to the room's actual fingerprint — often *more accurate* than a general model.
>
> **(B) Local-first, base-shared.** The frozen room/subject/device-invariant backbone is the only artifact published to Hugging Face. Per-room baselines and per-specialist heads never leave the device unless the operator opts into federation (ADR-105).

### 2.1 System architecture

```
                       HUGGING FACE HUB (public, room-agnostic)
                       ┌───────────────────────────────────────┐
                       │  RF Foundation Encoder (ADR-150)       │
                       │  pose-preserving · subject/room/device │
                       │  -invariant · frozen · safetensors     │
                       └───────────────┬───────────────────────┘
                                       │  from_pretrained() once, cached on device
                                       ▼
  STAGE 1 baseline        STAGE 2 enroll        STAGE 3 extract         STAGE 4 train (per room_id)
  ┌──────────────┐        ┌──────────────┐      ┌────────────────┐      ┌─────────────────────────┐
  │ ADR-135      │        │ Enrollment   │      │ signal_features│      │ SpecialistBank          │
  │ Baseline-    │──fp──► │ Protocol     │─clip►│ + ruvsense     │─AF──►│  frozen backbone        │
  │ Calibration  │        │ guided       │      │ extractors     │      │   │  ┌────────────────┐  │
  │ (env finger- │        │ anchors:     │      │ → AnchorFeature│      │   ├─►│ breathing head │  │
  │  print)      │        │ stand/sit/   │      │ (phase, amp,   │      │   ├─►│ heartbeat head │  │
  │ ADR-030      │        │ lie/breathe/ │      │  doppler,      │      │   ├─►│ restless head  │  │
  │ field eigen  │        │ move/sleep   │      │  micromotion,  │      │   ├─►│ posture head   │  │
  └──────────────┘        └──────────────┘      │  periodicity,  │      │   ├─►│ presence head  │  │
        │                                        │  variance,     │      │   └─►│ anomaly head   │  │
        │  baseline drift > τ → invalidate bank  │  timing)       │      │     (LoRA / ruVector    │
        └───────────────────────────────────────┴────────────────┴──────┤      small models)      │
                                                                          └───────────┬─────────────┘
                                                                                      │ RVF container
                                                                                      ▼
                                                              RUNTIME: Mixture-of-Specialists
                                                              each head emits {value, confidence};
                                                              coherence_gate (ADR-135) + anomaly
                                                              head veto → fused RoomState
```

The shared backbone is loaded **once per device** and frozen. Every specialist is a small head over its embedding — so the marginal cost of a sixth specialist is kilobytes of LoRA weights, not another full model.

### 2.2 Stage 2 — the guided enrollment protocol (NEW)

`EnrollmentProtocol` is a CLI-driven state machine that walks the operator through a fixed sequence of labelled **anchors**. The design rule from the user vision is explicit: *clean anchors, not hours of data.* Each anchor is a short (default 20 s @ 20 Hz = 400 frames) labelled clip captured against the already-recorded baseline.

| Anchor | Label | Duration | Primary signal taught | Feature emphasis |
|--------|-------|----------|-----------------------|------------------|
| `empty` | presence=0 | (reuse ADR-135 baseline) | absence reference | amplitude variance floor |
| `stand_still` | posture=standing, presence=1 | 20 s | static human load | amplitude mean shift, eigenmode delta |
| `sit` | posture=sitting | 20 s | lower static load | amplitude profile |
| `lie_down` | posture=lying | 20 s | sleep-position load | amplitude profile, low Doppler |
| `breathe_slow` | resp≈0.1–0.15 Hz | 30 s | slow respiration | periodicity, micro-Doppler |
| `breathe_normal` | resp≈0.2–0.3 Hz | 30 s | normal respiration | periodicity, BVP phase |
| `small_move` | motion=1 | 20 s | limb micro-motion | Doppler spread, variance |
| `sleep_posture` | posture=lying, restless=0 | 30 s | quiescent sleep baseline | long-window variance, timing |

The protocol is **adaptive**: an anchor is only accepted when its captured features pass a quality gate (coherence ≥ threshold from `coherence_gate.rs`, sufficient SNR vs baseline, no saturation). A failed anchor is re-prompted rather than silently kept — bad anchors poison small models far more than large ones. Total guided enrollment is ~4 minutes of wall-clock, producing 8 clean anchors. This is intentionally far below the "hours of data" that a from-scratch model needs, because the backbone already carries world knowledge; enrollment only teaches *this* room's offsets.

Anchors are persisted as an append-only `EnrollmentSession` (event-sourced, per CLAUDE.md state rules) under `room_id`, so re-enrollment is incremental and auditable.

### 2.3 Stage 3 — feature extraction to labelled records (REUSE + bridge)

Each accepted anchor clip is run through the existing extractor stack, baseline-subtracted per ADR-135, and packaged into an `AnchorFeature` record. No new DSP is invented — this stage is a *bridge*, not a new algorithm.

| Feature group | Source module | Used by specialists |
|---------------|---------------|---------------------|
| CSI amplitude mean/variance | ADR-135 baseline subtraction + `signal_features.rs` | presence, posture |
| CSI phase (sanitised, LO-aligned) | `phase_sanitizer` → `phase_align` | posture, heartbeat |
| Doppler shift / micro-Doppler | `ruvsense/bvp.rs`, `breathing` path | breathing, small-move |
| Micro-motion / intention lead | `ruvsense/intention.rs` | restlessness, anomaly |
| Periodicity / spectral peaks | `bvp.rs` autocorrelation + FFT | breathing, heartbeat |
| Long-window variance / drift | `ruvsense/longitudinal.rs` (Welford) | restlessness, presence |
| Timing / inter-frame epoch | `c6_timesync` epoch, frame Δt | all (rhythm alignment) |
| Field eigenmode coefficients | ADR-030 `field_model.rs` | posture, presence |

`AnchorFeature` = `{ room_id, anchor_label, t_epoch_us, embedding: [f32; D] (backbone output), aux: { resp_hz?, doppler_spread, variance, periodicity_score, eigen_coeffs } }`. The backbone embedding is the *shared* representation; `aux` carries the cheap hand-features that let small heads specialise without re-learning DSP.

### 2.4 Stage 4 — the specialist bank (NEW, the core contribution)

A **`SpecialistBank`** is a versioned collection of small models scoped to one `room_id`, persisted as a single RVF cognitive container (`wifi-densepose-ruvector`). Each specialist is a *head* over the frozen backbone embedding, trained from the labelled `AnchorFeature` records via the existing `rapid_adapt.rs` LoRA machinery (test-time/few-shot training, contrastive + entropy losses), **not** a from-scratch network.

| Specialist | Model type | Params (typ.) | Label source | Output |
|------------|-----------|---------------|--------------|--------|
| **breathing** | 1-D temporal head + periodicity regressor | ~8 KB LoRA + aux | `breathe_slow`/`breathe_normal` | resp rate (Hz) + confidence |
| **heartbeat** | narrowband phase head (harmonic-aware) | ~12 KB | quiescent anchors + periodicity | HR (bpm) + confidence |
| **sleep restlessness** | variance/drift classifier | ~4 KB | `sleep_posture` vs `small_move` | restlessness score [0,1] |
| **posture** | k-way prototype classifier (HNSW NN) | prototypes only | `stand/sit/lie` anchors | posture class + margin |
| **presence** | binary energy/eigenmode gate | ~2 KB | `empty` vs occupied anchors | presence prob |
| **anomaly** | one-class / physically-impossible detector (`adversarial.rs`) | ~6 KB | baseline + all anchors (novelty) | anomaly score + veto flag |

Design properties that follow from invariant (A):

- **Independently versioned & swappable.** Re-enrolling breathing does not retrain posture. A specialist carries its own `{trained_at, anchor_set_hash, baseline_hash, backbone_rev}`.
- **HNSW prototype storage for the classifiers.** Posture and presence are nearest-prototype lookups in the RVF index — no inference engine, microsecond latency, and new postures are added by inserting a prototype, not retraining.
- **SONA online adaptation.** Each specialist may carry a SONA/MicroLoRA online-adaptation slot (`ruvllm_sona_*` / `microlora` primitives) so it tracks slow drift (furniture moved, seasonal RF change) between full re-enrollments, gated by ADR-135 baseline drift.
- **Teacher–student distillation (optional, offline).** Where a labelled public corpus exists (MM-Fi, Wi-Pose), the ADR-150 backbone acts as teacher to pre-shape a head before per-room fine-tuning, improving cold-start. The *teacher* is global/HF; the *student head* is local.

**Invalidation contract.** The bank stores the `baseline_id` (the baseline UUID) it was trained against. **As implemented**, the runtime marks the bank `STALE` whenever the *current* baseline id differs from the trained one — a conservative trigger that catches re-calibration (room rearranged, AP moved, band changed) because any of those produces a new baseline. A finer **drift-threshold** trigger (mark STALE when ADR-135's per-subcarrier deviation exceeds τ *without* a full re-baseline) is a planned refinement (P6). Either way the runtime prompts re-enrollment rather than emitting silently wrong vitals — the calibration analogue of the #954 `DEGRADED` honesty rule: never report confident numbers from an invalid model.

### 2.5 Runtime — mixture of specialists with confidence gating

At inference, the frozen backbone embeds each CSI window once; every specialist consumes that shared embedding and emits `{value, confidence}`. Fusion rules:

- The **anomaly** specialist holds a **veto**: a high anomaly score (physically-impossible signal per `adversarial.rs`, or a coherence-gate `Reject`) suppresses positive vitals/posture output and raises a flag, rather than propagating a hallucinated reading.
- **presence=0** short-circuits breathing/heartbeat/posture to `null` (you cannot have a respiration rate in an empty room).
- Each emitted reading is tagged with the specialist's confidence and the `baseline_hash`/`backbone_rev` provenance, so downstream consumers (sensing-server, MQTT, Home Assistant) can gate on quality — consistent with ADR-135 coherence-gate semantics.

### 2.6 Crate & module layout

New bounded-context crate `wifi-densepose-calibration` (orchestration only; files < 500 lines, typed public APIs, event-sourced sessions — per CLAUDE.md):

```
wifi-densepose-calibration/
  src/
    lib.rs                 # public API: CalibrationSystem facade
    enrollment.rs          # EnrollmentProtocol state machine (Stage 2)
    anchor.rs              # Anchor, EnrollmentSession (event-sourced)
    extract.rs             # AnchorFeature bridge over signal_features + ruvsense (Stage 3)
    specialist.rs          # Specialist trait, SpecialistKind enum
    bank.rs                # SpecialistBank (RVF container, versioning, invalidation)
    runtime.rs             # MixtureOfSpecialists fusion + veto (Stage 5)
    backbone.rs            # frozen ADR-150 encoder loader (hf_hub from_pretrained, cached)
    error.rs
```

Dependencies (no duplication — orchestrates existing crates): `wifi-densepose-signal` (ruvsense extractors, ADR-135 baseline), `wifi-densepose-train` (`rapid_adapt`, `signal_features`, `trainer`), `wifi-densepose-ruvector` (RVF, HNSW), `wifi-densepose-nn` (backbone inference). The `wifi-densepose-cli` gains `enroll`, `train-room`, and `room-status` subcommands, sequenced after the existing ADR-135 `calibrate`.

### 2.7 CLI flow (operator-facing)

```bash
# Stage 1 — environmental fingerprint (ADR-135, existing)
wifi-densepose calibrate --room living-room --duration 60s     # empty room

# Stage 2+3 — guided enrollment (NEW); prompts through 8 anchors, ~4 min
wifi-densepose enroll --room living-room
#   → "Stand still in view of the sensor…"  [✓ anchor accepted: coherence 0.91]
#   → "Sit down…"                            [✗ low SNR, retrying]
#   ...

# Stage 4 — train the specialist bank (NEW); reuses cached HF backbone
wifi-densepose train-room --room living-room \
    --specialists breathing,heartbeat,restlessness,posture,presence,anomaly

# Status / invalidation
wifi-densepose room-status --room living-room
#   baseline: fresh (drift 0.04 < 0.20) · backbone: rf-foundation@1.2.0
#   breathing  ✓ trained 2026-06-09  conf p50 0.88
#   heartbeat  ✓ trained 2026-06-09  conf p50 0.71
#   posture    ✓ 3 prototypes (stand/sit/lie)
#   anomaly    ✓  · presence ✓  · restlessness ✓
```

---

## 3. Consequences

### 3.1 Positive

- **Fidelity through specialisation.** Six small calibrated heads beat one oversized general model on the cross-room/cross-subject frontier that ADR-150 quantified — and each runs in microseconds-to-milliseconds, on-device.
- **Privacy by construction.** Only the room-agnostic backbone is public (HF). The environmental fingerprint and the person-specific heads stay local; no video, no skin, no cloud round-trip. This is the core differentiator vs cameras and the convenience differentiator vs wearables.
- **Minutes, not hours.** Because the backbone carries world knowledge, ~4 minutes of clean anchors calibrates a room. Re-enrollment is incremental.
- **Honest degradation.** The `baseline_hash` invalidation + anomaly veto mean an out-of-calibration room reports `STALE`/flagged rather than confidently wrong — the same honesty principle as the firmware `DEGRADED` flag.
- **Composable & cheap to extend.** A new biological signal = a new small head over the same embedding, not a new model.

### 3.2 Negative / risks

- **Backbone dependency.** Every specialist rides on ADR-150's encoder; its quality and revision compatibility (`backbone_rev`) are a single point of leverage. Mitigation: pin `backbone_rev` in each specialist; distillation cold-start reduces sensitivity.
- **Enrollment burden.** 4 minutes is small but non-zero, and anchor quality depends on the operator following prompts. Mitigation: adaptive re-prompting + quality gates; ship sane defaults so a partial bank (presence+posture) works after just the static anchors.
- **Heartbeat is hard.** Sub-mm chest displacement at HR frequencies is near the ESP32-S3 noise floor; the heartbeat specialist will have lower and more variable confidence than breathing. The confidence-gated runtime surfaces this rather than faking it.
- **Per-room storage proliferation.** A bank per room per person; needs a clear RVF lifecycle (list/prune/export) — handled by `bank.rs` versioning and the `room-status` CLI.

### 3.3 Alternatives considered

| Alternative | Verdict | Reason |
|-------------|---------|--------|
| One large general model for all signals | **Rejected** | The ADR-150 evidence: scale overfits rooms/subjects and collapses cross-domain; also slower, costlier, less private. Directly contradicts invariant (A). |
| Cloud training of per-room models | **Rejected** | Violates invariant (B): would ship raw CSI of a person's home/sleep to a server. Local-first is the privacy promise. Federation (ADR-105) is the *opt-in* path for shared improvement, exchanging gradients/deltas, never raw CSI. |
| Skip the backbone; train each specialist from scratch | **Rejected** | Reintroduces the "hours of data" requirement the user vision explicitly rejects, and loses cross-room priors. |
| Fold this into ADR-135 | **Rejected** | ADR-135 is *room* calibration (no humans). This ADR is *human-anchor* enrollment + model training on top of it. Distinct lifecycles, distinct invalidation; kept as separate bounded contexts. |

---

## 4. Implementation phases

| Phase | Scope | Exit criterion | Status |
|-------|-------|----------------|--------|
| **P1** | Scaffold `wifi-densepose-calibration` crate; `AnchorFeature` schema; (backbone via `hf_hub` deferred) | Crate + schema; unit tests | ✅ Done (crate + Stage-1 baseline via `calibrate`/`calibrate-serve`; HF backbone deferred) |
| **P2** | `EnrollmentProtocol` + `anchor.rs` (event-sourced sessions) + CLI `enroll` with quality gates | 8-anchor enrollment; bad anchors re-prompt | ✅ Done (`anchor.rs`, `enrollment.rs`, CLI `enroll`) |
| **P3** | `extract.rs` bridge → labelled records; baseline subtraction (ADR-135) | `AnchorFeature` records persisted per `room_id` | ✅ Done (`extract.rs`; autocorr periodicity + variance/motion) |
| **P4** | `SpecialistBank` + presence/posture (prototype) + breathing (periodicity); persistence + versioning | `train-room` produces a bank; `room-status` reads it back | ✅ Done (`specialist.rs`, `bank.rs`, CLI `train-room`/`room-status`; JSON persistence — RVF/HNSW = future) |
| **P5** | heartbeat + restlessness + anomaly specialists; `runtime.rs` mixture + veto + confidence gating | End-to-end RoomState on hardware; anomaly veto verified | ✅ Done (`runtime.rs`, CLI `room-watch`; breathing read live on COM8 ESP32) |
| **P6** | Baseline-drift `STALE` invalidation; SONA online adaptation; optional ADR-105 federation; HF teacher–student distillation | Drift marks bank STALE; AetherArena entry | ◐ Partial (STALE done; SONA/federation/HF-backbone = follow-ups) |

**Current status (2026-06-10):** Stages 1–5 implemented with *statistical* specialists (threshold/prototype/autocorrelation). 55 tests (35 unit incl. multistatic + 1 full-loop integration + 19 CLI), all passing under qemu-aarch64. **Validation scope is precise:** baseline capture + HTTP API + auth are proven on real CSI (Pi-5 nexmon, 6,813 frames; and an ESP32-S3). The complete `baseline → enroll → train-room → infer` loop is now **proven in-process** on deterministic synthetic CSI (`tests/full_loop.rs`: clean baseline with zero motion flags, 8/8 anchors through the quality gate, 6 specialists trained, JSON bank round-trip, trained-bank inference 18±2 BPM positive / absent negative / foreign-baseline STALE; seed-robust). The one live runtime signal (breathing ~16–31 BPM via `room-watch`) used the *stateless* breathing head, **not** a trained bank; the clean empty-room loop has **not** yet run on-target — the remaining gap is strictly the hardware session (empty room + operator anchors). The four behavioral findings from the full-loop test (z-band squeeze, variance-only presence, ungated hz embedding, heart-band lag-floor leakage) are FIXED and regression-guarded — see the integration doc §7. SOTA-intake decisions affecting this system (geometry conditioning, checkerboard alignment) are recorded in ADR-152. Open refinements: `--source-format adr018v6` (drive from the Pi's own nexmon), phase-based breathing carrier, RVF/HNSW storage, and the ADR-150 frozen HF backbone the specialists would distill from.

Validation per CLAUDE.md: `cargo test --workspace --no-default-features` green; hardware verification on the ESP32-S3 (currently COM8) before any release; witness bundle regenerated if the proof surface changes.

---

## 6. Review notes

### 6.1 Correctness + security review (2026-06-14)

Beyond-SOTA correctness+security review of `wifi-densepose-calibration` (this
ADR's pipeline), un-covered by the ADR-154–159 sweep.

**Finding (FIXED) — NaN-poisoning of the feature path (numerical / fail-closed).**
`Features::from_series` — the carrier for both live inference and training-anchor
extraction — computed `mean`/`variance`/`motion` over the raw scalar series with
no non-finite guard. A single `NaN`/`±inf` sample (corrupt CSI frame) yielded
`mean=NaN, variance=NaN` and an all-`NaN` prototype embedding. Persisted into a
`PresenceSpecialist::threshold`/`empty_mean` at train time, the `NaN` **silently
disabled presence detection** for the bank's lifetime (every `>` / `|·|`
comparison against `NaN` is false → always reads *absent*, confidence 0), with no
error — and an asymmetry against the rigorously NaN-guarded `geometry_embedding`.
Fixed at the production boundary: non-finite samples are dropped (a corrupt frame
counts as no frame), an all-non-finite series degrades to `Features::ZERO` like
the empty series. Value-identical for all-finite input (full-loop + extract tests
unchanged); pinned by `non_finite_samples_do_not_poison_features` and
`all_non_finite_series_is_zero` (both fail on the old code).

**Clean dimensions (evidence, no invented issues).**
- *File/path handling:* the crate performs **zero** file/path I/O (no
  `std::fs`/`Path`/`File`/`read`/`write` in `src/`; only in-memory `serde_json`).
  Path-traversal / unbounded-read / artifact-path handling live entirely in the
  `wifi-densepose-cli` consumer (`room.rs`), outside this crate's boundary.
- *Untrusted-load:* `SpecialistBank::from_json` shape-validates via serde
  (malformed → `CalibrationError::Serde`); banks are local-first (invariant B),
  never network-received. A well-formed bank with adversarial numerics is trusted
  as-is — acceptable under the local-first threat model; a validate-on-load
  defense-in-depth pass is a possible future hardening, not a present bug.
- *Receipt/hash integrity:* the crate emits no hash/receipt/witness/signature, so
  the unframed-concatenation bug class (cf. the engine `witness_of` fix) is
  structurally absent.
- *Other numerical paths:* `geometry_embedding` sanitizes every input and sweeps
  to finite; presence/restlessness/anomaly divisions are `.max(1e-3)`-guarded;
  `autocorr_dominant` guards `r0`, short signals, and empty bands; `train` rejects
  empty anchors; anomaly requires ≥2 anchors.

De-magicked the bare specialist threshold literals (breathing/heartbeat default
min-scores, anomaly outlier-spread multiple + label cutoff) into named documented
consts, value-identical, pinned by const-equality tests. Tests
**58→62 unit + 1 integration, 0 failed**; Python deterministic proof unchanged
(off the signal proof path).

---

## 5. Summary

> Big models understand the world. Small ruVector models understand *your room*.

ADR-151 makes that operational: a local-first `baseline → enroll → extract → train` pipeline that turns ~4 minutes of clean human anchors — layered on ADR-135's empty-room fingerprint and ADR-150's Hugging-Face-published invariant backbone — into a versioned bank of tiny, specialised, privacy-preserving models for breathing, heartbeat, restlessness, posture, presence, and anomaly. Specialisation over scale; local heads over a shared base; honest `STALE` degradation over confident error.
