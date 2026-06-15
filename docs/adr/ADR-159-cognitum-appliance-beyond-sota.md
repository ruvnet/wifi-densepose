# ADR-159: Cognitum Appliance Cluster — Beyond-SOTA Sweep, Anti-"AI-Slop" Hardening

- **Status**: accepted
- **Date**: 2026-06-11
- **Deciders**: ruv
- **Tags**: cognitum, cogs, person-count, pose-estimation, ha-matter, drone-swarm, remote-id, manifest, prove-everything

## Context

This ADR records the beyond-SOTA sweep over the Cognitum appliance cluster
(`cog-person-count`, `cog-pose-estimation`, `cog-ha-matter`, `ruview-swarm`),
executed under the project's **prove-everything / anti-"AI-slop"** directive: the
claim surface every cog presents (manifests, descriptions, runtime events,
broadcast fields) must match what the code and the shipped weights actually do.

### Headline — the "never identified anyone" accusation is REFUTED

A read-only audit raised the worst-class accusation: that these cogs are slop that
"never identified anyone." That accusation is **refuted by byte-level evidence**:

- `cog-pose-estimation` and `cog-person-count` ship **real, trained Candle models**
  (`pose_v1.safetensors`, `count_v1.safetensors`), not placeholders. The forward
  passes (`PoseNet`, `CountNet`) mirror the training scripts exactly and run on
  real CSI bytes.
- The artifacts are **SHA-pinned and Ed25519-signed**: the on-disk
  `manifests/x86_64/manifest.json` carries a real `binary_sha256`
  (`051614ce…388b3` for person-count, `a434739a…71fa` for pose), a real
  `weights_sha256`, and a `binary_signature` over `sig_algo: Ed25519`.
- The manifests are **brutally honest about accuracy**: person-count's
  `build_metadata` ships `training_class1_accuracy = 0.343` and a candid
  `training_caveat`; pose ships `training_pck20 = 3.0` / `training_pck50 = 18.5`.
  Nothing is inflated. That honesty *is* the anti-slop win — the models are weak
  in the field, and the manifests say so.

So the cogs **do** run real trained inference and **do** disclose how weak it is.
What the audit correctly found were not fabrications but **claim-surface
overclaims** — four places where the surface said more than the weights deliver.
This ADR tightens those four (A1–A4) and cites the already-correct subsystems as
NO-ACTION positives.

Grading vocabulary follows ADR-152 / ADR-158:
- **MEASURED** — reproduced in this worktree, command + failing-on-old test recorded.
- **DATA-GATED** — real code path present; honestly flagged where data/hardware is absent.
- **NO-ACTION (already-SOTA)** — audited, found correct, cited as a positive.
- **ACCEPTED-FUTURE** — deliberately deferred, nothing dropped.

## Graded SOTA Landscape

| Capability | Grade | Note |
|------------|-------|------|
| CSI person counting (`cog-person-count`) | **DATA-GATED** | Real Candle count head + Bayesian fusion; weights trained only on classes 0/1 (presence). Multi-occupant accuracy is genuinely unproven and is **not fabricated** — counts above the trained range are now flagged `low_confidence` and clamped. |
| CSI pose estimation (`cog-pose-estimation`) | **DATA-GATED** | Real Candle encoder + 17-keypoint head; field accuracy honestly weak (PCK@50 = 18.5%, disclosed in the manifest). The default-install gate bug (A1) is fixed so it actually emits frames. |
| Signed cog manifests (Ed25519 + SHA-256) | **NO-ACTION (already-SOTA)** | On-disk manifests are real, signed, SHA-pinned, and honest about accuracy. The CLI now emits them verbatim (A4). |
| HA bridge (`cog-ha-matter`) MQTT + witness | **NO-ACTION (already-SOTA)** | Real Ed25519 hash-chain witness, mDNS, embedded broker. Matter commissioning is honestly deferred to v0.8 (TLS off, LAN-only) — description softened to stop claiming Matter (honest-absence). |
| Drone-swarm MARL (`ruview-swarm`) | **DATA-GATED / honest** | `candle_ppo.rs` is real autodiff PPO; it is **untrained at runtime** (random init) by design — the swarm must be trained before deploy, which the code does not hide. |
| ASTM F3411 Remote ID | **MEASURED (A3)** | Basic ID message is real; the Location/Vector message is honestly *not* implemented (NED metres are no longer mislabelled as WGS84 lat/lon). |

## Decision — Fixes Landed (MEASURED)

### §A1 Pose runtime emitted ZERO frames under default config (HIGH)

**Overclaim (silent correctness bug):** `inference.rs` hardcoded
`confidence: 0.185` for every inference, `config.rs default_min_confidence()`
returned `0.3`, and `runtime.rs` gated emission on `confidence >= min_confidence`.
A default install therefore **never emitted a single `pose.frame`** while
`health` reported healthy — the cog *claimed* to be a running pose estimator but
silently produced nothing.

**Real fix:** `pose_v1` has **no confidence head** (the head emits 34 keypoint
coordinates only), so a real per-frame confidence is genuinely unavailable. We
took the disclosed "ok" path rather than silently lowering the threshold:
- Introduced `inference::MODEL_TYPICAL_CONFIDENCE = 0.185` (the validation PCK@50)
  as the single published per-frame confidence, used by both `infer()` and the
  config default.
- Pinned `default_min_confidence()` to `MODEL_TYPICAL_CONFIDENCE` so a default
  install clears its own gate and emits.
- Documented the trade-off in the config field doc, the JSON schema
  (`default` 0.3 → 0.185, with a description), **and** added a `run.started`
  warning in `main.rs` that fires when an operator raises `min_confidence` above
  the model's typical confidence — so a deliberately-high threshold is loud, not
  silent.

**Failing-on-old test:** `cog_pose_estimation` smoke
`default_config_emits_frames_with_real_model` — parses a default config and
asserts `min_confidence <= MODEL_TYPICAL_CONFIDENCE` (and, with the real model
loaded, that `infer().confidence >= min_confidence`). **Proven to fail** on the
old `default_min_confidence()=0.3`:
`default min_confidence 0.3 exceeds model typical confidence 0.185 — a default
install would emit zero pose.frame events`.

**Grade: MEASURED.**

### §A2 8-class count head on a 2-class-trained model (MEDIUM)

**Overclaim:** `inference.rs COUNT_CLASSES = 8` with argmax over {0..7}, but
`count_train_results.json` has support only for classes 0 and 1 (`per_class_accuracy`
keys `"0"`/`"1"`). The model is a **presence detector**, not a calibrated
multi-occupant counter; an argmax on classes 2..=7 is out-of-distribution, yet the
cog would emit it as a confident headcount. The Cargo.toml billed it as a
"learned multi-person counter."

**Real fix (no network change — DATA-GATED, accuracy not fabricated):**
- Added `inference::MAX_TRAINED_CLASS = 1`, plus `CountPrediction::is_low_confidence()`
  (argmax beyond the trained ceiling) and `clamped_count()` (report clamped to the
  trained range, raw argmax kept for audit).
- `person.count` events now carry `low_confidence` + `raw_count`, and downgrade to
  `level: "warn"` when out-of-distribution; the reported `count` is clamped so we
  never emit a fabricated headcount the weights can't back.
- `run.started` discloses `count_max_trained_class` and `count_classes`.
- Cargo.toml description changed from "learned multi-person counter" to
  "presence detector + (data-gated) person count".

**Failing-on-old test:** `cog_person_count` smoke
`untrained_class_argmax_is_flagged_low_confidence` — a prediction whose argmax is
class 5 is asserted `is_low_confidence() == true` and `clamped_count() ==
MAX_TRAINED_CLASS`; a class-1 prediction is asserted *not* flagged. Fails on old
code (no such methods/flag existed).

**Grade: MEASURED (mechanism); multi-occupant accuracy DATA-GATED.**

### §A3 Remote ID broadcast NED metres as WGS84 lat/lon (MEDIUM — safety/compliance)

**Overclaim (compliance hazard):** `security/remote_id.rs update()` stored
`state.position.x/.y` (NED **metres**) into `drone_lat`/`drone_lon`, so the Remote
ID broadcast would carry physically-impossible coordinates (e.g. "latitude =
37.5 m"). The module doc claimed a "Basic ID + Location/Vector message," but only
`encode_basic_id()` exists.

**Real fix (honest naming — never broadcast impossible coordinates):**
- Renamed `drone_lat`/`drone_lon` → `drone_north_m`/`drone_east_m` (NED metres
  relative to the operator/takeoff datum), with field docs stating they are *not*
  geodetic. `operator_lat`/`operator_lon` remain true WGS84 (from the operator's
  GNSS).
- Corrected the module doc to claim **Basic ID only**; the Location/Vector encoder
  is explicitly deferred until a datum-anchored NED→WGS84 transform lands
  (ACCEPTED-FUTURE), rather than removing a real feature.

**Failing-on-old test:** `security::remote_id::tests::test_ned_offset_stored_as_metres_not_latlon`
— a 37.5 m north / −12.0 m east NED offset is asserted to land in
`drone_north_m`/`drone_east_m`; the operator's real WGS84 fix stays in range. Fails
on old code, where these values were stored into `drone_lat`/`drone_lon`.

**Grade: MEASURED.**

### §A4 Hollow CLI manifest (LOW)

**Overclaim:** `cog-person-count main.rs cmd_manifest` emitted a null skeleton
(`binary_sha256: null`, no training metadata), making the CLI look unsigned even
though the **real signed manifest** existed at
`cog/artifacts/manifests/x86_64/manifest.json`.

**Real fix:** new `cog_person_count::manifest` module `include_str!`-embeds the
real signed manifests (x86_64 + arm), selected by build target arch.
`cmd_manifest` now parses-then-emits the embedded signed manifest — exactly the
pattern `cog-pose-estimation`'s `manifest_roundtrips` test demonstrates. The CLI
now reports the real `binary_sha256`, `weights_sha256`, Ed25519 signature, and
honest `build_metadata` (`training_class1_accuracy = 0.343`).

**Failing-on-old test:** `manifest::tests::embedded_manifest_has_non_null_binary_sha256`
asserts a 64-hex-char `binary_sha256`; companions assert the embedded manifest is
signed (`sig_algo == Ed25519`) and `id == COG_ID`. End-to-end verified:
`cog-person-count manifest` prints `binary_sha256:
051614ce6ba63df704fae848a67ad095df4bb88862fdff05ef3c0419cc8388b3`.

**Grade: MEASURED.**

### §A5 cog-ha-matter description claimed Matter before it exists (LOW — honest-labeling)

**Overclaim:** the Cargo.toml description said "Home Assistant + Matter
integration," but Matter commissioning is deferred to v0.8 (`TlsConfig::Off`,
LAN-only, asserted by `runtime.rs tls_defaults_to_off_for_v1_lan_only`).

**Real fix (no code change):** softened the description to "Home Assistant (MQTT)
integration … LAN-only (no TLS); Matter Bridge commissioning is deferred to v0.8
and not yet implemented." Mirrors ADR-158 §6 honest-absence: state what isn't
there rather than implying it is.

**Grade: MEASURED (label).**

## Negative Results (Confirmed — NO-ACTION positives)

Audited and found genuinely correct; cited as positives, not edited:

- **`cog-ha-matter` witness chain** (`witness.rs` / `witness_signing.rs`) — real
  Ed25519 hash-chained witness log. Already-SOTA.
- **`cog-person-count` fusion** (`fusion.rs`) — real Bayesian product-of-experts
  multi-node fusion (Stoer-Wagner-bounded clip), not a heuristic. Already-SOTA.
- **`ruview-swarm` PPO** (`marl/candle_ppo.rs`) — real Candle autodiff PPO with a
  genuine policy-gradient update; its `randn` uses (init, action sampling,
  exploration) are all legitimate, not fake-output substitutes. Untrained at
  runtime by design (the swarm must be trained before deploy), which the code
  does not hide. Already-SOTA / honest.

## Deferred Backlog (Nothing Dropped)

- **Multi-occupant count accuracy** — DATA-GATED on labelled multi-occupant CSI.
  The `low_confidence` flag + clamp (§A2) is the honest stand-in until then.
- **Remote ID Location/Vector message** — ACCEPTED-FUTURE; requires a
  datum-anchored local-tangent-plane NED→WGS84 transform with an operator datum.
  Basic ID ships today.
- **Matter Bridge commissioning** — ACCEPTED-FUTURE (v0.8); LAN-only MQTT ships today.
- **Criterion benches** for cog inference latency and `mesh_guard` — ACCEPTED-FUTURE
  (cold-start timings are recorded in the manifests' `build_metadata`, not yet a
  regression bench).
- **`wasm-edge` skill accuracy** — unvalidated; **now honestly labelled, not
  claimed** (done in ADR-160: medical/affect/security/exotic claim surfaces
  disclaimed, renamed, and feature-gated; per-skill accuracy remains DATA-GATED).

## Consequences

- A default pose-estimation install now actually emits `pose.frame` events;
  raising the threshold above the model's reach is a loud `run.started` warning,
  not a silent dropout.
- A person-count reading on an untrained class is flagged `low_confidence`,
  clamped, and downgraded to `warn` — no fabricated headcounts.
- The Remote ID broadcast can never carry physically-impossible coordinates; NED
  metres live in honestly-named metre fields.
- `cog-person-count manifest` now reports the real signed manifest instead of a
  hollow null skeleton.
- No cog Cargo.toml description claims a capability (multi-person counting, Matter)
  the code/weights don't yet deliver.

## Reproduction (MEASURED)

```bash
cd v2
cargo test -p cog-person-count -p cog-pose-estimation -p cog-ha-matter -p ruview-swarm \
  --no-default-features
# ruview-swarm train path compiles (PPO autodiff)
cargo check -p ruview-swarm --features train
# A4 end-to-end — real signed manifest, non-null binary_sha256
cargo run -q -p cog-person-count --no-default-features -- manifest
```

Result at time of writing (all 0 failed):
- `cog-person-count` — **19 passed** (lib 10 incl. 3 manifest; smoke 9)
- `cog-pose-estimation` — **8 passed** (smoke)
- `cog-ha-matter` — **64 passed** (unchanged; description-only edit)
- `ruview-swarm` — **117 passed** (default features); `--features train` compiles clean.

Scope was limited to the four named crates. NO-ACTION positives (witness chain,
fusion, PPO + randn audit) were verified by inspection and left untouched.
