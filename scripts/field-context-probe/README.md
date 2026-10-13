# RuField to WorldGraph integrity probe

This standalone Rust executable calls the actual pinned `rufield-core`,
`rufield-provenance`, `rufield-fusion`, and `wifi-densepose-worldgraph` crates.
It demonstrates evidence handling with deterministic **SYNTHETIC** fixtures.
It does not train an activity classifier or establish sensing accuracy.

## Run

Requires Rust 1.90.0 or newer. Dependencies are pinned to full Git revisions in
`Cargo.toml`; `Cargo.lock` locks their resolved transitive dependencies.

```sh
cd scripts/field-context-probe
cargo test --locked
cargo run --locked --quiet
cargo run --release --locked --quiet -- --benchmark
# Optional bounded format-only inspection, no calibration or fusion:
cargo run --release --locked --quiet -- --inspect-jsonl /path/to/capture.csi.jsonl
```

No network, file writes, hardware access, person tracking, or external release
occurs when the built executable runs. Initial Cargo dependency resolution needs
network access. The public fixed signing seed is exclusively a test fixture.

## Actual pipeline

1. Construct a signed RuField `FieldEvent` with `synthetic: true`. Its clock,
   amplitude vector and presence feature are explicitly authored test values.
2. Verify its signature with `rufield_provenance::verify_event`, then ingest it
   through the actual `RuFieldFusion` simulation trust policy and a tiny explicit
   presence rule. The output score is a heuristic, not a calibrated probability.
3. A small bridge checks fresh, unique root evidence and inference expiry before
   calling `WorldGraph::add_semantic_state`. Original event IDs become mandatory
   provenance handles and `DerivedFrom` edges to concrete event nodes.
4. Serialize and restore the actual WorldGraph. Assert preserved node counts,
   edge counts and ancestry.

The bridge has a fixed capacity of 32 root events. The synthetic bridge accepts no external input. The separate optional replay inspector reads a regular UTF-8 file of at most 1 MiB and 10,000 nonempty lines. It calls actual `CsiReplayAdapter` parsing, but never calibration, event generation or graph insertion. Every malformed JSON line is rejected, including a truncated tail which the native adapter deliberately tolerates. Parsed frame count must also match nonempty line count, preventing schema-invalid final objects from being silently discarded. Rejection exits with status 2; accepted format exits with status 0. Format acceptance establishes no labels, empty-room baseline or sensing accuracy.
The executable prints a JSON receipt; all rejected fixtures leave the WorldGraph
unchanged. Nothing is automatically promoted.

## Native guarantees versus bridge policy

| Check | Responsible component |
| --- | --- |
| Exact event replay and nonmonotonic timestamp rejection | Actual RuField trust verifier, including simulation mode |
| Rejection of synthetic input under production policy | Actual RuField production verifier |
| Signature corruption | Actual RuField signature verification invoked explicitly by bridge |
| Wall clock freshness for this simulation | Bridge; native simulation deliberately skips wall clock freshness |
| Repeated content hash under a different event ID | Bridge |
| Derived evidence feeding back as independent root evidence | Bridge conservatively rejects every nonempty ancestry declaration |
| Expired inference, missing ancestry, duplicate belief | Bridge |
| Typed belief and source ancestry persistence | Actual WorldGraph |
| General cycle prevention | Not supplied by native WorldGraph; a separate boundary test demonstrates that native self edges are accepted |

This is deliberately conservative. It is not a complete general purpose graph
cycle detector or a production authorization service. Production use requires
independently enrolled sensor keys, appropriate privacy policy, fresh calibrated
measurements and application specific lineage rules.

## Why no ESP-Fi samples enter this probe

The available activity recordings do not establish calibrated empty room
baselines, synchronized wall clock times, persistent physical tracks, or complete
world topology. Fabricating those fields would create false evidence. The
synthetic probe therefore validates software integrity only. The independent
activity training harness continues to evaluate its own actual dataset.

## Pinned source

* RuField: `b74bc035e4a3d4023665aa5479b41e938facd113`
* WorldGraph: `9b1c79c836cdacfb7b44f058c593157bac4c1dab`

Acceptance: all six tests pass; the receipt reports synthetic evidence,
uncalibrated confidence, complete root ancestry, rejected duplicate/stale/derived
fixtures, successful graph roundtrip and `promotionAuthorized: false`.

## Measured overhead

The release benchmark uses 100 warmups and 1,000 alternating pairs on the same
signed synthetic fixture. Both paths construct a fresh engine, parse the rule,
verify the signature, clone the fixture, ingest and infer. The bridge additionally
checks its guards, constructs WorldGraph and publishes ancestry. Signing, sensing,
training, calibration and persistent-stream behavior are outside the measurement.
The JSON receipt reports both paths and the distribution of paired differences;
its target is added p95 below 0.1 ms. This is a software integrity cost measurement,
not predictive improvement or a production throughput guarantee.

The optional recording inspector also applies stricter policy than the native
adapter: required numeric timestamp and amplitude array, finite nonnegative
values, nanoseconds below u64 maximum, amplitudes within f32 range, 1–2,048
channels, fixed channel width and strictly increasing integer nanosecond times.
These format/representation guards do not certify capture provenance or labels.

## Active evidence replay and bounded request proposals

```sh
cargo run --release --locked --quiet -- --active-loop
cargo run --release --locked --quiet -- --active-loop --ranking ranking.json
cat ranking.json | cargo run --release --locked --quiet -- --active-loop --ranking -
cargo run --release --locked --quiet -- --active-benchmark
```

`--active-loop` runs an authored **SYNTHETIC** sequence: explicit empty,
occupied, contradictory sources causing unknown, agreeing occupied sources,
and expired observations returning to unknown. Missing updates or motionlessness
never create empty evidence. Each accepted root is signed and verified with the
actual pinned RuField implementation; actual WorldGraph evidence nodes and
`DerivedFrom` relationships record the supporting roots. A wrapper recomputes
expiry using a monotonic authored clock. Persisted WorldGraph states are history,
not an occupancy authorization interface: callers must consult the wrapper's
current snapshot, never treat an old graph node as still valid.

Only the latest fresh root per logical source participates. Root IDs and hashes
are immutable and unique within a run; nonempty dependency declarations are
rejected instead of treating derived feedback as independent evidence. Source
timestamps must increase, and the 32-root or 32-belief capacity limit latches a
fail-closed unknown state. Capacity exhaustion is an intentional sticky safety
mutation; other rejected ingestion leaves both evidence and the accepted clock
unchanged. Independent sources are assumed by the fixture;
this does not establish physical statistical independence. Wifi, radar and door
are logical source names using signed WiFi feature fixtures, not production
radar or door adapters. Presence values, thresholds and costs are authored.
Native fusion scores are reported as uncalibrated heuristics, not probabilities.

The optional KGE sidecar input has this exact schema:

```json
{"schemaVersion":1,"evidence":"SYNTHETIC","candidates":[{"sensorId":"radar","score":2.3},{"sensorId":"door","score":1.1}]}
```

Input is limited to 4,096 bytes and one to three unique allowlisted sensor IDs
(`wifi`, `radar`, `door`). Unknown fields, duplicate object keys, repeated sources,
nonfinite scores and nonregular file inputs are rejected. On Linux x86_64 and
AArch64 the file reader opens with `O_NOFOLLOW | O_NONBLOCK`, then validates the
opened descriptor as a bounded regular file. Unsupported platforms reject file
paths; bounded stdin remains portable. Higher scores only
change proposal ordering, with lexical source order breaking ties. They cannot
modify evidence, belief state or fusion confidence and cannot trigger hardware
or external actions. This is a sidecar contract, not a claim that a KGE model was
trained. Rank scores can be outside zero to one and are never probabilities.

The initial proposal reservation budget is 2 packets, 2,200 microseconds of
airtime and 220 microjoules. Authored per-request costs are respectively:
WiFi `(2, 1000, 100)`, radar `(1, 2000, 200)`, and door `(1, 100, 10)`.
Every reservation must fit all three remaining limits atomically; an unaffordable
candidate can be skipped for an affordable suggestion. Reservations are for
proposals only; no packets are transmitted and no energy use is measured.

The active benchmark reports three rounds of 1,000 alternating baseline/active
pairs after 100 warmups per round, including p50 and p95 for each path and their
paired difference. The baseline verifies, ingests and fuses one signed root.
The active path additionally checks root policy, selects current sources,
fuses selected evidence into an engine, publishes WorldGraph ancestry and creates a
receipt. Both construct fresh engines. Signing, physical sensing, ranking file
I/O and persistent-stream throughput are excluded. Release mode is required for
reported comparisons. Timing establishes software cost only, not sensing uplift.
