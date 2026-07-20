# ADR-158: MAT / World-Model Cluster — Beyond-SOTA Sweep, Anti-"AI-Slop" Hardening

- **Status**: accepted
- **Date**: 2026-06-11
- **Deciders**: ruv
- **Tags**: mat, life-safety, localization, triage, worldmodel, worldgraph, geo, engine, prove-everything

## Context

This ADR records the beyond-SOTA sweep over the MAT / world-model cluster
(`wifi-densepose-mat`, `-worldmodel`, `-worldgraph`, `-geo`, `-engine`), executed
under the project's **prove-everything / anti-"AI-slop"** directive: every stub is
either implemented with real logic or replaced by an honest typed error; no
fake/always-empty/random outputs; tests pass on real behaviour; results are graded
**MEASURED** (reproduced here with the command recorded), **CLAIMED**,
**DATA-GATED** (real code path present, needs hardware/data we lack), or
**NO-ACTION** (already-SOTA — cited as a positive).

The Mass Casualty Assessment Tool touches life-safety. A triage metric that is
disconnected from the decision it gates, or a survivor count that inflates, is the
worst class of slop: it produces confident, wrong rescue prioritisation. An audit
against live code found six concrete defects, four of which were silent
correctness bugs (not missing features) in the triage → gate → record path and in
the localization/dedup path.

Grading vocabulary follows ADR-152 (F-evidence grades) and the sweep convention:
- **MEASURED** — reproduced in this worktree, command recorded below.
- **DATA-GATED** — real code path implemented; returns a typed error / honest
  provenance flag where hardware or labelled data is genuinely absent.
- **NO-ACTION (already-SOTA)** — audited, found correct, cited as a positive.
- **ACCEPTED-FUTURE** — deliberately deferred, nothing dropped.

## Graded SOTA Landscape

| Capability | Grade | Note |
|------------|-------|------|
| RF-through-rubble survivor detection | **DATA-GATED** | Real detection + triage + localization code paths run end-to-end on real CSI bytes; field detection *accuracy* is unproven without instrumented rubble trials and is **not fabricated** here. |
| OccWorld occupancy architecture (`-worldmodel`) | **NO-ACTION (current)** | `occupancy.rs` voxel mapping is clamp-proven bounds-safe; converts WorldGraph person positions to a 200×200×16 grid with no out-of-bounds path. |
| WorldGraph provenance / privacy / pruning (`-worldgraph`) | **NO-ACTION (already-SOTA)** | `graph.rs` implements append-with-provenance (`DerivedFrom`), deterministic LRU pruning, and a privacy rollup (`PrivacyLimitedBy`). Cited as a positive; no changes needed. |
| Point-cloud parser bounds-safety (`-pointcloud`) | **NO-ACTION (already-SOTA)** | Another agent's crate; cited only — its parser is bounds-checked. Out of scope for this ADR's edits. |
| Learned multi-person counter | **DATA-GATED** | Deferred; requires labelled multi-occupant CSI. The zone+vitals-signature dedup (below) is the honest non-learned stand-in. |
| RF point-cloud generation | **ACCEPTED-FUTURE** | Not dropped; tracked as future work. |

## Decision — Fixes Landed (MEASURED)

### §1 Unify the two divergent triage engines (CRITICAL)

**Was:** `EnsembleClassifier::determine_triage` (ensemble gate) and
`TriageCalculator::calculate` (survivor record) were two different START-protocol
approximations with different rate bands and movement handling. The pipeline
gated on the ensemble's confidence (`lib.rs:489`), discarded the ensemble triage
(`lib.rs:524`, `_ensemble`), and recomputed via `TriageCalculator` in
`Survivor::new` (`survivor.rs:194`). A survivor could be admitted at one priority
and recorded at another.

**Now:** `determine_triage` delegates to `TriageCalculator` — the **single source
of truth** used by both the gate and the survivor record. The only ensemble-
specific behaviour retained is the confidence gate (low confidence → `Unknown`,
except `Immediate`, which is never suppressed — a missed survivor in distress is
costlier than a false positive). Rate bands follow START (<10 / >30 bpm →
Immediate).

**Failing-on-old test:** `detection::ensemble::tests::test_divergent_boundary_28bpm_tremor_gate_equals_survivor`
— 28 bpm Normal + Tremor. Old gate → Delayed, old survivor record → Immediate
(divergent). Unified result: gate == survivor == **Immediate**. Companion tests
(`test_no_vitals_is_unknown_canonical`, `test_normal_breathing_no_movement_is_immediate_canonical`,
the updated `integration_adr001::test_ensemble_classifier_triage_logic`) assert
gate-vs-record equality on every boundary.

### §2 Real RSSI/ToA localization + kill count-inflation (HIGH)

**Was:** `fusion.rs:79 simulate_rssi_measurements` always returned `vec![]`, so
every survivor got `location: None`, so spatial dedup (`disaster_event.rs:285`,
which only fired on `Some` location) was disabled. One trapped person re-detected
across N scan cycles became **N survivors** — a fabricated mass-casualty count.

**Now, two real mechanisms:**
1. **Real RSSI source:** `SensorPosition` gains an optional `last_rssi`
   (populated by the hardware layer from actual signal-strength readings).
   `collect_rssi_measurements` reads only real per-sensor RSSI and feeds the
   existing triangulator; it **never fabricates** a value. With `< min_sensors`
   real readings, `estimate_position` returns `None` (honest).
2. **Zone + vitals-signature dedup:** when no usable location exists,
   `record_detection` matches an existing *active, un-located* survivor in the
   same zone whose latest vital signature (breathing presence + START rate band,
   heartbeat presence, movement class) is compatible — collapsing repeat
   detections of one person while keeping genuinely distinct survivors separate.

**MEASURED:** `test_identical_vitals_no_location_dedup_to_one` — 3× identical-vitals
/ `None`-location → **1 survivor** (old code: 3). `test_distinct_vitals_no_location_stay_separate`
keeps two distinct survivors at 2 (no under-count). `test_estimate_position_uses_real_rssi`
yields a position from 3 real-RSSI sensors; `test_estimate_position_none_without_real_rssi`
yields `None` (no fabrication).

### §3 Real ESP32/UDP/PCAP CSI ingest; honest typed errors elsewhere (HIGH)

**Was:** `hardware_adapter.rs read_esp32_csi` / `read_udp_csi` / `read_pcap_csi`
returned "not yet implemented" — even though `csi_receiver.rs` already contained a
working `CsiParser` (ESP32 CSV, JSON, Intel5300/Atheros/Nexmon byte decoders) and a
real `PcapCsiReader`.

**Now:**
- **UDP** — binds, receives one datagram, parses (auto-detect) → `CsiReadings`.
  End-to-end test sends a real JSON datagram on the wire.
- **PCAP** — `load` + `read_next` + parse. End-to-end test writes a real
  little-endian `.pcap` with one record and reads it back.
- **ESP32** — parses `CSI_DATA` CSV via the real parser. Live serial byte I/O is
  behind an optional `serial` cargo feature (native `serialport` kept off the
  default / aarch64 appliance build); with the feature off, live reads return a
  typed `UnsupportedAdapter` while the byte parser still works.
- **Intel 5300 / Atheros / PicoScenes** — return typed
  `AdapterError::HardwareUnavailable` / `UnsupportedAdapter` (no device, no
  driver, or no validatable format here). **Never fake CSI.** New error variants
  added to make the gating typed rather than a `String` "Hardware" soup.

**MEASURED:** `test_esp32_bytes_parse_end_to_end`, `test_udp_read_end_to_end`,
`test_pcap_read_end_to_end`, `test_intel_and_atheros_are_honestly_unavailable`.

### §4 Real parabolic peak interpolation in `find_dominant_frequency` (MED)

**Was:** `breathing.rs:243` comment claimed interpolation but returned the bin
center, capping breathing-rate resolution at ±half a bin.

**Now:** 3-point parabolic (quadratic) peak interpolation,
`δ = 0.5·(yL − yR)/(yL − 2y0 + yR)`, clamped to `[-0.5, 0.5]`, with an edge
fallback to bin center.

**MEASURED:** `test_find_dominant_frequency_parabolic_interpolation` — for a
parabola-shaped peak at true bin 10.4 the recovery is exact (δ = 0.4); the test
asserts the result lands within half a bin of truth and strictly beats the
old bin-center estimate.

### §5 GDOP honesty (LOW)

**Was:** `triangulation.rs:248 estimate_gdop` returned an ad-hoc average-pair-angle
factor *labelled* GDOP (the same defect class ADR-156 §2.3 fixed elsewhere).

**Now:** real, dimensionless **GDOP = √(trace((HᵀH)⁻¹))** from the range-measurement
Jacobian `H` (unit target→sensor bearings), returning `None` for singular
(collinear) geometry, which the caller treats as factor 1.0 (no fabrication).

**MEASURED:** `test_gdop_is_real_dilution` — a well-spread array gives a lower GDOP
than a near-collinear one, cross-checked against the closed form;
`test_gdop_singular_collinear_is_none` confirms singular geometry returns `None`.

### §6 OccWorld trajectory-prior consumer honesty (fail-safe)

**Finding:** `wifi-densepose-mat` does **not** consume OccWorld trajectory priors
and has no `-worldmodel`/`-worldgraph`/occworld dependency (grep-verified: zero
hits across `crates/wifi-densepose-mat/`). There is therefore no random-derived
prior being consumed. **No code change** is warranted; the fail-safe (ignore
priors until a typed `weights_complete`/`stubbed` flag exists) is already the
status quo by absence. Recorded here so a future consumer wires the flag rather
than re-introducing the risk.

## Negative Results (Confirmed — NO-ACTION)

These were audited and found genuinely correct; they are cited as positives, not
edited:

- **`worldgraph` provenance / privacy / pruning** (`graph.rs`) — append-with-
  provenance (`add_semantic_state` + `DerivedFrom`), deterministic LRU pruning
  (`prune_semantic_states`, with `prune_is_deterministic_for_equal_timestamps`),
  and a privacy rollup (`apply_privacy_mode` → `PrivacyLimitedBy`). Already-SOTA.
- **`worldmodel` occupancy clamp** (`occupancy.rs:74–125`) — `to_voxel_xy` /
  `to_voxel_z` `.clamp()` voxel indices into `[0, GRID-1]`; the flat index is
  always in-bounds. No out-of-bounds / fabrication path.
- **`pointcloud` parser bounds-safety** — another agent's crate; cited only, its
  parser is bounds-checked.

## Deferred Backlog (Nothing Dropped)

- **Learned multi-person counter** — DATA-GATED on labelled multi-occupant CSI.
  The zone+vitals-signature dedup (§2) is the honest non-learned stand-in until
  then.
- **RF point-cloud generation** — ACCEPTED-FUTURE.
- **PicoScenes container decode** — DATA-GATED; needs matching NIC/plugin to
  validate against. Returns `UnsupportedAdapter` today.
- **Intel 5300 / Atheros live capture** — DATA-GATED on patched drivers; byte
  parsers exist and are exercised on supplied bytes.

## Consequences

- Triage is now a single auditable function; gate and survivor record can never
  diverge.
- Survivor counts cannot inflate from repeat detection of one un-located person.
- The CSI ingest layer either produces real data or fails with a typed error that
  names *why* — no path silently substitutes simulated/fabricated CSI.
- `SensorPosition` grows an optional `last_rssi` field (serde-`default`, non-
  breaking for deserialisation; 7 constructors updated).
- A new optional `serial` feature isolates the native `serialport` dependency from
  the default / appliance builds.

## Reproduction (MEASURED)

```bash
cd v2
# MAT — default features (181 unit + 6 + 3[3 ignored] integration)
cargo test -p wifi-densepose-mat
# MAT — all features (same counts; exercises ruvector + api + serde paths)
cargo test -p wifi-densepose-mat --all-features
# MAT — serial feature compiles (native serialport path)
cargo check -p wifi-densepose-mat --features serial
# Sibling crates (cited NO-ACTION; confirmed green)
cargo test -p wifi-densepose-worldmodel   # 12 + 1
cargo test -p wifi-densepose-worldgraph   # 9
cargo test -p wifi-densepose-geo          # 9 + 8
cargo test -p wifi-densepose-engine       # 27
```

Result at time of writing: MAT **181 passed; 0 failed** (default and all-features);
worldmodel **13**, worldgraph **9**, geo **17**, engine **27** — all 0 failed.
