# OpenISAC Adversarial Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OpenISAC/X310 integration fail closed: publish versioned RF observations only, pair raw/metadata by frame ID, bound UDP reassembly, reject stale or invalid input, and stop advertising unverified human-sensing capabilities.

**Architecture:** The Python bridge is the OpenISAC protocol boundary. It reassembles bounded sender-scoped chunks, decodes raw and metadata independently, pairs them by frame ID, and emits a versioned `RfObservation` envelope. The Rust server is the RuView trust boundary: it validates the envelope and monotonic sequence, stores it as an observation, reports freshness and ingest counters, and never derives presence, count, pose, or vitals from `rf-direct` data.

**Tech Stack:** Python 3, NumPy, pytest, Rust, Tokio, Axum, Serde, Clap, Docker Compose.

---

### Task 1: Bound OpenISAC UDP Reassembly

**Files:**
- Modify: `tests/test_openisac_to_ruview_bridge.py`
- Modify: `scripts/openisac_to_ruview_bridge.py`

- [x] **Step 1: Write failing tests**

Add tests that construct `FrameAssembler(max_chunks=4, max_payload_bytes=16, max_partial_frames=2, partial_ttl_seconds=1.0)` and prove that it:

```python
assert assembler.add_datagram(packet_with_total_chunks(5), sender=("127.0.0.1", 1), now=0.0) is None
assert assembler.stats.rejected_datagrams == 1
assert assembler.add_datagram(oversized_two_chunk_payload, sender=("127.0.0.1", 1), now=0.0) is None
assert assembler.partial_frame_count <= 2
assembler.expire(now=2.0)
assert assembler.partial_frame_count == 0
```

Also prove identical frame IDs from two senders do not share chunks, duplicate chunks are counted, and malformed chunk declarations are rejected without allocating an attacker-controlled list.

- [x] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_openisac_to_ruview_bridge.py -q --no-cov -p no:cacheprovider
```

Expected: new tests fail because the constructor has no limits, sender is not part of the key, and TTL/counters do not exist.

- [x] **Step 3: Implement the bounded assembler**

Add explicit defaults:

```python
DEFAULT_MAX_CHUNKS = 4096
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PARTIAL_FRAMES = 32
DEFAULT_PARTIAL_TTL_SECONDS = 2.0
```

Store sparse chunks in `dict[int, bytes]`, track accumulated bytes and `updated_at`, key partials by `(sender, frame_id, is_metadata)`, evict expired/oldest entries before accepting a new partial, and expose accepted/rejected/duplicate/expired/evicted counters.

- [x] **Step 4: Verify GREEN**

Run the Python test command again and require all tests to pass.

### Task 2: Pair Raw And Metadata Into A Versioned Observation

**Files:**
- Modify: `tests/test_openisac_to_ruview_bridge.py`
- Modify: `scripts/openisac_to_ruview_bridge.py`

- [x] **Step 1: Write failing semantic tests**

Add tests proving:

```python
static = summarize_range_doppler(static_zero_doppler_frame, ...)
assert "motion_energy" not in static
assert "targets" not in static
assert static["rd_diagnostics"]["peaks"][0]["kind"] == "unclassified_peak"

pairer.add(raw_frame, now=0.0) is None
observation = pairer.add(metadata_frame, now=0.1)
assert observation["schema"] == "ruview.rf_observation"
assert observation["protocol_version"] == 1
assert observation["sequence"] == observation["frame_id"]
assert observation["freshness"] == "fresh"
assert observation["observation"]["cfar"]["candidate_clusters"]
assert "presence" not in observation
assert "estimated_persons" not in observation
```

Add timeout, out-of-order, duplicate, and metadata-missing tests. A timed-out raw frame must increment `pair_timeouts` and never be forwarded.

- [x] **Step 2: Verify RED**

Run the targeted pytest command and confirm the failures are caused by the current separate-forwarding behavior and inference-shaped fields.

- [x] **Step 3: Implement observation pairing**

Make raw summaries diagnostic-only (`range_profile`, SNR, shape, `unclassified_peak` values). Make metadata summaries CFAR-observation-only. Add a bounded `FramePairer` keyed by `(sender, frame_id)` with a two-second TTL and 32-pair capacity. When both halves exist, emit:

```json
{
  "schema": "ruview.rf_observation",
  "protocol_version": 1,
  "source": "openisac-rd",
  "frame_id": 42,
  "sequence": 42,
  "source_timestamp_ns": null,
  "received_at_ns": 0,
  "config_hash": "sha256:<hex>",
  "freshness": "fresh",
  "observation": {
    "range_doppler": {},
    "cfar": {"candidate_clusters": []},
    "micro_doppler": null
  }
}
```

The live bridge records raw payloads as before but sends/records JSON only after pairing. Replay accepts raw/metadata filename pairs and also fails closed on a missing half.

- [x] **Step 4: Verify GREEN**

Run pytest and `python -m py_compile scripts/openisac_to_ruview_bridge.py`.

### Task 3: Validate RF Observations In Rust Without Human Inference

**Files:**
- Modify: `v2/crates/wifi-densepose-sensing-server/src/main.rs`

- [x] **Step 1: Write failing Rust tests**

Add pure unit tests for `parse_rf_observation`, `accept_rf_sequence`, `source_capabilities`, and RF timeout behavior. Cover wrong schema/version, non-finite numeric values, empty config hash, duplicate and out-of-order sequences, a valid paired observation, and `rf-direct:offline` after timeout. Assert the RF capability manifest has observation transport enabled and pose/presence/count/vitals disabled.

- [x] **Step 2: Verify RED**

Run:

```powershell
cargo test -p wifi-densepose-sensing-server --no-default-features rf_observation --manifest-path v2\Cargo.toml
```

Expected: compile/test failure because the versioned observation contract and helpers do not exist.

- [x] **Step 3: Implement the Rust trust boundary**

Replace the permissive `RfDirectFrame` with a typed, versioned `RfObservation` and validation method. Add RF receive time, last accepted sequence, accepted/invalid/duplicate/out-of-order/stale counters, and a five-second freshness timeout to state. `rf_direct_udp_receiver_task` binds to `127.0.0.1` by default, accepts only fresh monotonically increasing version-1 observations, and creates `SensingUpdate` with:

```rust
classification: ClassificationInfo {
    motion_level: "unverified".into(),
    presence: false,
    confidence: 0.0,
},
vital_signs: None,
persons: None,
estimated_persons: None,
rf_observation: Some(observation),
```

Do not call pose derivation, person scoring, vital smoothing, or MQTT inference for RF observations. Stale broadcasts change freshness to `stale`, clear candidate clusters, and report `rf-direct:offline`.

- [x] **Step 4: Verify GREEN**

Run the targeted Rust tests and then the complete sensing-server crate tests.

### Task 4: Reject Unknown Sources And Report Real Capabilities

**Files:**
- Modify: `v2/crates/wifi-densepose-sensing-server/src/main.rs`
- Modify: `ui/services/sensing.service.js`

- [x] **Step 1: Write failing tests**

Add unit tests that parse the CLI source into an enum and reject `rf-direc`, plus endpoint/helper tests showing `rf-direct` has no pose capability, stale sources are not ready, pose current returns no generated skeleton, pose confidence is absent when no detections exist, and stream FPS is absent when it was not measured.

- [x] **Step 2: Verify RED**

Run the sensing-server crate tests and confirm the tests fail against the current wildcard simulated fallback and hard-coded endpoint values.

- [x] **Step 3: Implement fail-closed source/capability behavior**

Validate source names before state creation; only `auto`, `wifi`, `esp32`, `usrp`, `rf-direct`, `rf`, `simulated`, and `simulate` are accepted. Replace the wildcard simulated spawn with explicit variants. `/api/v1/info` returns a source capability manifest. `/api/v1/pose/current` never calls procedural pose generation for `rf-direct`. `/api/v1/pose/stats`, `/api/v1/stream/status`, `/health/health`, and `/health/metrics` return measured values or `null`/`unavailable`, never fixed demo values.

Update the UI source mapper to treat only the explicit `usrp`, `rf-direct`, and `rf` source labels as live; remove the broad `startsWith('rf-')` acceptance.

- [x] **Step 4: Verify GREEN**

Run the Rust crate tests and the available UI tests or syntax check.

### Task 5: Secure Defaults And Documentation

**Files:**
- Modify: `docker/docker-compose.yml`
- Modify: `docker/docker-entrypoint.sh`
- Modify: `scripts/openisac_to_ruview_bridge.py`
- Modify: `docs/integrations/x310-rf-direct.md`
- Modify: `CHANGELOG.md`

- [x] **Step 1: Write/configure verification assertions**

Extend Python parser tests to assert `--openisac-host` defaults to `127.0.0.1`. Add a source-level test that Compose does not publish `5010/udp` or `5020/udp` by default.

- [x] **Step 2: Verify RED**

Run pytest; expect failures because the bridge binds all interfaces and Compose publishes both ports.

- [x] **Step 3: Apply secure defaults**

Default the bridge and Rust RF receiver to loopback. Remove `5010/udp` and `5020/udp` from Compose `ports`. Document that remote UDP is intentionally unsupported until authenticated transport is implemented; do not provide an unauthenticated override. Update the integration guide and changelog with the observation-only limitation and schema.

- [x] **Step 4: Verify GREEN**

Run pytest, Python compilation, Rust format check, sensing-server tests, and Compose config validation if Docker is available.

### Task 6: Final Audit And Additional Findings Report

**Files:**
- Modify: `docs/superpowers/plans/2026-07-20-openisac-adversarial-hardening.md`
- Create only if new issues are found: `docs/integrations/ruview-openisac-follow-up-findings.md`

- [x] **Step 1: Re-read the adversarial report**

Map every P0 item to a code change or a documented intentional limitation. Do not claim research validation for motion, presence, range, count, pose, or vitals.

- [x] **Step 2: Audit adjacent code**

Search the changed paths for silent simulation fallbacks, `estimated_persons = 1`, hard-coded health/FPS/confidence values, unrestricted UDP binds, unbounded allocation, and RF-derived procedural pose.

- [x] **Step 3: Write a separate report only for genuinely new findings**

If the audit finds issues not already covered by `ruview-openisac-adversarial-review.md`, record severity, evidence, impact, and recommended fix in `docs/integrations/ruview-openisac-follow-up-findings.md`. If no new issues exist, do not create an empty report.

- [x] **Step 4: Run final verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_openisac_to_ruview_bridge.py -q --no-cov -p no:cacheprovider
.\.venv\Scripts\python.exe -m py_compile scripts/openisac_to_ruview_bridge.py
cargo fmt --manifest-path v2\Cargo.toml --all -- --check
cargo test -p wifi-densepose-sensing-server --no-default-features --manifest-path v2\Cargo.toml
```

Then inspect `git diff --check` and `git diff --stat`. Record any environment-only failure exactly rather than treating it as a pass.
