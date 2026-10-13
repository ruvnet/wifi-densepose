# ADR 421: Bounded active evidence loop

Status: Research implementation. Physical qualification and production integration pending.

Date: 2026 10 13 UTC

## Context

Fast activity inference does not establish continuous occupancy. The available
ESP-Fi recordings do not supply synchronized room topology or independently
annotated occupancy transitions. Motion evidence fading cannot prove an empty
room. ADR 346 protects existing firmware semantics; this experiment does not
replace its deployed presence gate. ADR 382 remains the separate geometry export
contract. No geometry, person identity or pose is inferred by this loop.

## Decision and boundaries

Extend the standalone field context probe with an explicit research mode using
the actual pinned RuField and WorldGraph dependencies. Keep production binaries,
firmware, published model weights and existing inference defaults unchanged.

RuView observations carry source, capture time, quality, calibration and root
provenance. RuField verifies the event and provides fusion primitives. A bounded
application policy owns clock monotonicity, expiry, source dependence and
contradiction handling. WorldGraph stores supporting roots and semantic states.
KGE may rank measurement proposals through ADR 422. It has no evidence or device
authority. The predictive encoder in ADR 423 is an independent experiment until
its downstream utility is established.

The executable fixture uses public test keys and authored observations. It is
always labelled SYNTHETIC. Its score is not a calibrated probability. Actual
production admission still requires independently enrolled sensor keys and
durable, integrity protected replay watermarks. Serializing WorldGraph alone
does not serialize an entire trust verifier or make a restart safe.

## State transition specification

1. Validate bounded input, source identity, root uniqueness, signature, declared
   ancestry, capture time and monotonic application time before accepting evidence.
2. Keep at most one current observation per source. Conservatively reject derived
   evidence as a new root. Multiple graph paths do not constitute multiple sensors.
3. Expire roots according to an explicit lifetime. Preserve historical provenance
   without allowing historical graph nodes to sustain a current decision.
4. Compute empty, occupied or unknown. Conflicting eligible evidence and missing
   eligible evidence produce unknown. Absence of motion alone is not emptiness.
5. If additional evidence is useful, filter ranked candidates through the sensor
   allowlist, availability and packet, airtime and energy budgets.
6. Emit a proposal and its supporting context. Do not perform hardware actions,
   autonomously promote models, publish personal data or silently expand budgets.

An observation rejection must not mutate evidence or graph state. Budget
rejection must not consume budget. Clock rollback must not revive evidence.
Bounded capacity fails closed instead of evicting replay protection silently.

## Alternatives

Keeping the activity classifier alone is simplest but cannot establish the
requested temporal evidence contract. Letting a language model infer room state
has poorly bounded cost and no adequate sensor evidence boundary. A full
production service would add enrollment, durable replay state and device control
before there is field evidence of value. The isolated deterministic loop permits
rollback by removing the experiment and minimizes those additional authorities.

## Acceptance and evidence

Unit tests must cover duplicates, source dependence, contradiction, expiry,
derived roots, invalid ranking, clock rollback, capacity and budget exhaustion.
An integrated run must consume real KGE output through the bounded Rust reader.
Release benchmarks record warmup, repeated samples, baseline scope and p50/p95;
software timing on synthetic observations is not sensing accuracy.

Field acceptance is separate: a frozen occupancy baseline versus memory and
adaptive selection on fresh rooms and sessions, at equal packets, airtime,
energy and available links. Target at least 20% lower false occupied duration per
verified empty room hour, a room clustered interval supporting improvement and
recall noninferiority within one percentage point. Unknown counts as a miss when
occupied. Require the upper confidence bound on additional unknown duration to
be at most one percentage point separately in empty and occupied intervals;
otherwise selective abstention could manufacture the false alarm improvement.
Recall intervals must not be conditioned on nonzero false alarm counts.
Report coverage and acquisition/window/detection delay separately.
Replay may compare only actions for which measurements were actually recorded.

## Sources and implementation

* `scripts/field-context-probe/README.md`
* `docs/adr/ADR-346-fail-closed-edge-occupancy-evidence.md`
* `docs/adr/ADR-382-spatial-evidence-rf-export.md`
* https://github.com/ruvnet/rufield
* https://github.com/ruvnet/worldgraph

See `docs/active-evidence-experiment.md` for exact measured receipts and limits.
