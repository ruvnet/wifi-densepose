# ADR-331: Consumer NLOS evidence levels, privacy/security, benchmarks, and optional MetaHarness governance

| Field | Decision |
|---|---|
| **Status** | Accepted; software gates are implemented independently of live-hardware evidence, research promotion remains blocked until the preregistered capture passes |
| **Date** | 2026-08-22 |
| **Owners** | RuView Labs maintainers, security/privacy reviewers, benchmark owners |
| **Scope** | Claim taxonomy, acceptance records, benchmark governance, contributor harness, release/promotion and rollback |
| **Depends on** | ADR-166, ADR-168, ADR-282, ADR-295, ADR-299, ADR-303 through ADR-305, ADR-318/319, ADR-328 through ADR-330 |
| **Implementation** | `harness/ruview`, `docs/security/consumer-nlos-threat-model.md`, `docs/research/consumer-nlos-acceptance-protocol.md` |

## Context

Consumer NLOS is unusually easy to overstate. A point-cloud animation can look
plausible even when driven by replay, stale state, calibration leakage, direct
line of sight, or a reflectivity regime unlike the deployment target. Native and
web builds can validate contracts without measuring one multipath photon. CSI
and LiDAR may also share temporal or labeling leakage, producing an apparent
fusion gain that contains no independent RF information.

The capability reveals presence and trajectories outside direct view, which
raises privacy and misuse risks despite not producing a conventional photograph.
It also has no present safety case for autonomous actuation. The project needs a
promotion gate that separates source integrity, software validation, controlled
laboratory evidence, field generalization, and production evidence.

The existing RuView contributor harness is dependency-free at runtime and uses
MetaHarness/Flywheel/Darwin only as development aids. NLOS verification should
reuse that posture: helpful, deterministic, and removable, never a requirement
for sensing, fusion, memory, routing, MCP, or normal runtime behavior.

## Decision

The `@ruvnet/ruview` packed-size ceiling increases from 160 KiB to 220 KiB to
carry the bounded NLOS contract verifier, research-gate evaluator, and operator
skill. The existing no-source-map, no-runtime-dependency, tarball smoke, and
claim-honesty gates remain mandatory. This is a reviewed capability budget, not
an unbounded exemption; the current dry-run package is approximately 203 KiB.

### 1. Keep three orthogonal labels

Every NLOS result records:

1. **Source provenance**: exactly one logical value for live hardware, replay,
   or synthetic input; unknown is never coerced to live. The acceptance record
   encodes these as `LIVE_HARDWARE`/`REPLAY`/`SYNTHETIC`, while the
   `ruview.nlos.track.v1` wire contract uses `live`/`replay`/`synthetic`.
2. **Claim tag**: `MEASURED` with a named reproducer/manifest, `CLAIMED` with a
   primary external source, or `SYNTHETIC` for simulation/fixtures. A build is
   “validated software,” not a measured sensing result.
3. **ADR-282 evidence level**:

| Level | Meaning for NLOS |
|---|---|
| L0 | simulation or generated transient/track fixtures only |
| L1 | captured replay; deterministic pipeline behavior, not a fresh live result |
| L2 | controlled laboratory capture with external ground truth and frozen protocol |
| L3 | held-out room plus target/subject validation with leakage-resistant splits |
| L4 | multi-site field pilot under approved privacy/safety operations |
| L5 | production operational evidence, incident monitoring, drift and rollback history |

The upstream paper is cited as `CLAIMED` in RuView until reproduced. The first
RuView live acceptance can reach L2 only. No amount of L0/L1 replay volume
upgrades a capability to L2.

This ADR-282 maturity level is not the similarly named wire
`evidenceLevel` in `ruview.nlos.track.v1`. V1 accepts
`l0_synthetic`/`l1_measured`/`l2_calibrated`; `l3_corroborated` is reserved for
a future contract that can retain authenticated dual-modality lineage. These
values describe one envelope's source/calibration ceiling. They never self-promote
research maturity: a synthetic envelope is ADR L0, a captured replay remains
ADR L1 regardless of its historical wire label, and a live
`l2_calibrated` envelope reaches ADR L2 only through this
frozen external-ground-truth witness protocol. The acceptance JSON therefore
uses separate `claim_tag: MEASURED` and ADR maturity `evidence_level: L2`.

### 2. Separate software, research, and release gates

| Gate | What can pass it | What it proves | What it does not prove |
|---|---|---|---|
| Software | Rust/Swift/TypeScript unit, property, contract, build, replay and security tests | implementations compile and enforce declared invariants | photon capture, NLOS accuracy, physical iPhone support |
| Research | preregistered `LIVE_HARDWARE` capture with external ground truth | objective reproduction and fusion endpoints for named strata | field generalization, safety, identity, production readiness |
| Release/promotion | software + security + privacy + required evidence/certificate + human review | capability may be exposed at its exact evidence level | authority to actuate or claim a higher level |

An unavailable Xcode/hardware toolchain is an explicit `SKIPPED`, not a pass or
failure. A present partial/incompatible surface fails shallow discovery. If no
toolchain executes, the verifier reports `NO_BUILD_TOOLCHAINS_AVAILABLE`, never
a pass. Available-build success is only a subset of Gate A and is forbidden in
a hardware or release claim.

### 3. Preregister the reproduction and fusion endpoint

Before opening the test partition, freeze:

1. upstream commit and firmware/configuration;
2. sensor/CSI identity, calibration and coordinate transforms;
3. scene/target strata, exclusion rules and direct-line-of-sight checks;
4. randomized sequence order and grouped split manifest;
5. background capture and OOD/freshness thresholds;
6. track initialization, lost-track definition and maximum association gap;
7. primary metrics and bootstrap confidence interval procedure; and
8. all model/weight/threshold versions for LiDAR-only and fused arms.

The hard gate requested for this program is:

1. **Reproduction:** LiDAR-only produces at least 27 accepted end-to-end track
   updates per second on live hardware, the objective definition of roughly
   30 fps.
2. **Fusion:** over at least 100 paired live sequences, fusion achieves

\[
G_e = \frac{E_L-E_F}{E_L} \ge 0.25
\quad\text{or}\quad
G_\ell = \frac{\ell_L-\ell_F}{\ell_L} \ge 0.25,
\]

where \(E\) is the preregistered mean target-position error and \(\ell\) is the
preregistered lost-track rate. Both metrics are reported even if only one is
the success endpoint. Zero denominators do not count as improvement.

The protocol also reports confidence intervals, p95 position error, time to
first lock, false tracks in an empty hidden volume, update/latency distribution,
frame loss, calibration/OOD rejection, and performance per reflectivity, range,
motion, relay surface, room, and RF geometry. These secondary metrics prevent a
single successful aggregate from hiding unacceptable behavior.

### 4. Use a bounded, reviewable acceptance record

`ruview.nlos.acceptance.v1` is repository-contained JSON that references, but
does not embed, sensitive captures. Required fields include:

1. exact schema, `LIVE_HARDWARE` provenance, `MEASURED` claim tag, L2 evidence
   level, and `EXTERNAL` ground truth;
2. protocol-frozen-before-capture flag;
3. enrolled external ST VL53L8-series sensor-model label for v1, verified raw/CNH transient kind, full
   upstream SHA-1, and SHA-256 digests for protocol, firmware/API, combined and
   CSI capture manifests, scoped enrolled identities, calibration and analysis;
4. zero synthetic frames and zero replay frames in the scored set;
5. independently verified CSI, a CSI-only ablation, at least one CSI source and
   at least 100 paired sequences;
6. raw aggregate counts/durations/sums from which each arm's update rate, mean
   position error, lost-track rate, offered optical rate and fused optical-frame
   loss are recomputed; full paired-sequence position coverage and shared
   endpoint denominators are mandatory; and
7. preregistered bootstrap seed digest, at least 10,000 resamples,
   multiplicity-adjusted intervals, witness review, LOS exclusion, and passed
   privacy/security review.

The advisory verifier validates an exact-key record, bounds, provenance,
digests, arithmetic, and thresholds; unknown fields fail so secrets or raw data
cannot silently hitchhike in the acceptance artifact. It cannot independently
prove that a digest corresponds to an honest physical experiment. Human reviewers inspect the immutable manifest,
external ground-truth synchronization, raw-capture access controls, exclusions,
and analysis reproducer before promotion.

### 5. Extend, but do not require, the RuView contributor harness

The dependency-free `@ruvnet/ruview` CLI/MCP registry adds governed advisory
surfaces. MCP/static inspection remains read-only. Local `--run-builds` is an
explicit execution mode, must target a trusted checkout, uses an allowlisted
child environment and redacted bounded tails, and is not a sandbox:

| Surface | Behavior |
|---|---|
| `ruview nlos plan` / `ruview_nlos_plan` | returns four staged phases, measurement invariant, expected surfaces and exit gates |
| `ruview nlos verify` | shallowly discovers expected Rust/native/web manifests/contracts, can explicitly run available local builds in a trusted checkout, and evaluates an optional live acceptance record |
| `ruview_nlos_verify` | read-only MCP inspection of the auto-detected repository and optional repository-confined evidence; rejects repository selection and build execution |
| `consumer-nlos` skill | contributor playbook for measurement boundary, reproduction, fusion, evidence, security, and iOS limitations |

The verifier discovers `v2/crates/ruview-nlos`, `ui/ios-nlos`, and `ui/mobile`.
If an optional surface is absent, it reports `ABSENT`. Once its feature marker
exists, missing, escaping, oversized or incompatible required artifacts are
`MALFORMED` and fail. String/manifest discovery does not replace contract tests.
Evidence files are regular, bounded, repository-confined JSON; path escape,
symlink escape, oversize, malformed JSON, replay, or synthetic input fails.

MetaHarness, Ruflo, Darwin, and Flywheel remain contributor tooling. Direct
crate/app builds without the harness are the evidence that runtime does not
require it; the verifier's status field is not proof by itself. Harness
proposals cannot modify sensing evidence, promote a
model, publish a package, merge code, or authorize hardware. Promotion retains
human review.

For future harness-policy evolution, a proposal is retained only if a frozen
holdout shows more than 2 percent quality lift, contributor cost regression is
below 1 percent, p95 latency regression is below 5 percent, security/legacy
tests do not regress, provenance is verified, and a human approves it. These
thresholds govern contributor tooling only; they do not replace the 25 percent
sensor-fusion endpoint.

### 6. Apply privacy-by-default controls

1. A named controller/operator records purpose, lawful/organizational basis,
   approved spaces, experiment window, access list, retention, and deletion.
2. Visible notices and an active sensing indicator are required. Participants
   can pause/withdraw where applicable. No hidden deployment is allowed.
3. Raw transient, CSI, ground-truth video/tag data, and joined trajectories are
   P0/P1 research data: encrypted locally, access logged, minimized, separated
   by tenant/experiment, and deleted on the frozen schedule. They are never
   committed to Git or emitted in harness logs.
4. Default outputs are bounded session-scoped tracks with covariance,
   provenance and expiry. No face image is needed, but “camera-free” is not
   “privacy-free.” No biometric identity or cross-session re-identification is
   claimed or enabled by default.
5. Dataset/model publication needs a separate disclosure/re-identification
   review, consent/license check, and removal request path.
6. Safety and high-consequence use are out of scope. Tracks are advisory and
   never direct actuation authority.

### 7. Make security a release blocker

The threat model in `docs/security/consumer-nlos-threat-model.md` is mandatory.
Release requires no confirmed high/critical finding across:

1. untrusted transient/track/parser boundaries and numeric/resource exhaustion;
2. sensor, session, calibration and capture identity/provenance;
3. replay/stale/future/duplicate and cross-tenant/world-frame attacks;
4. transport authorization, short-lived web tickets, TLS/origin and secret
   handling;
5. firmware/upstream/npm/Cargo/Swift supply chain and license review;
6. raw research-data retention, logs, fixtures, telemetry and repository policy;
7. denial/degradation behavior and lack of direct actuation.

Findings are confirmed with focused tests before remediation claims. Automated
scanner output alone is not a confirmed vulnerability or a cleared release.

## Performance and optimization governance

Performance reports name hardware, OS/toolchain, configuration, capture digest,
commit, warmup, sample count, and statistics. Report sensor capture rate,
accepted frame rate, track update rate, sensor-to-track latency, service-to-view
latency, CPU, peak memory, queue drops, and output quality separately.

Optimization sequence:

1. profile parser, normalization, rendering/likelihood, resampling, temporal
   join, serialization, and UI independently;
2. add scalar golden vectors and property tests before vectorization/caching;
3. bound buffers/particles/history and prefer newest-frame backpressure;
4. compare LiDAR-only and fused quality after every material optimization; and
5. roll back any change that violates numerical tolerance, security, privacy,
   freshness, or the preregistered endpoint.

Configured 30 Hz, display 30/60 fps, and replay throughput are not equivalent
to live end-to-end tracking. Reports must name which one was measured.

## Alternatives considered

### Allow synthetic evidence to pass when hardware is unavailable

Rejected. Synthetic and replay are essential for software QA but cannot measure
the required optical path or independent RF gain.

### Treat the published MIT result as RuView's baseline evidence

Rejected. It is a primary `CLAIMED` reference with different hardware/scenes and
does not validate RuView's integration, privacy, security, or fusion.

### Make MetaHarness a required runtime coordinator

Rejected. Core sensing, fusion, memory, routing, MCP and UI must operate without
development orchestration. An advisory tool cannot be a physical trust anchor.

### Store all raw captures indefinitely for reproducibility

Rejected. Reproducibility uses controlled access, immutable digests, frozen
manifests and a retention schedule. Indefinite person/space/RF data creates
disproportionate risk.

### Promote on the 25 percent point estimate alone

Rejected. Pairing, confidence intervals, strata, secondary harms, calibration,
provenance, security, privacy and human review remain required.

## Consequences

### Positive

1. Reviewers can distinguish compile/replay success from physical NLOS evidence.
2. The 30 fps and 25 percent claims become executable, preregistered gates.
3. Privacy and security are part of experiment design, not a post hoc checklist.
4. The optional harness makes missing toolchains/surfaces explicit without
   weakening the independent runtime.

### Costs and limitations

1. Controlled capture, external ground truth, review and data governance take
   more time than a visual demo.
2. The acceptance JSON verifies integrity/arithmetic, not physical truth by
   itself; witness review remains necessary.
3. L2 success does not establish L3–L5 generalization or safety.
4. A negative fusion result blocks promotion even if each modality is
   individually interesting.

## Rollout and rollback

| Stage | Published capability | Required evidence | Rollback |
|---|---|---|---|
| G0 | development preview | L0 synthetic software fixtures, explicitly labeled | disable flag/view/tool; retain tests |
| G1 | captured replay demo | L1 capture manifest and privacy approval | revoke fixture/access; return to synthetic |
| G2 | controlled research result | L2 live protocol + acceptance + security/privacy review | invalidate certificate; remove live label; optical-only or unknown |
| G3+ | held-out/pilot/production | level-specific ADR-282 artifacts and monitoring | level downgrade, stop capture, delete per schedule, incident review |

Rollback preserves audit/witness records and never rewrites provenance. It may
disable NLOS/fusion independently while CSI and the rest of RuView continue.

## Objective acceptance mapping

| ID | Requirement | Evidence |
|---|---|---|
| NLOS-331-01 | Claim tag, provenance and L0–L5 level never alias | type/state tests, UI label tests, claim checker |
| NLOS-331-02 | Synthetic/replay cannot pass research | `evaluateResearchEvidence` negative tests and CLI nonzero with `--require-research-pass` |
| NLOS-331-03 | Reproduction is roughly 30 fps | live accepted update rate >=27 Hz, external-ground-truth manifest |
| NLOS-331-04 | Fusion has objective value | `MEASURED` in the capture-manifest witness report under the frozen protocol over >=100 paired sequences; >=25% mean-error **or** lost-track reduction with adjusted interval excluding zero, shared pairing digest, and all frozen guardrails passing |
| NLOS-331-05 | Harness remains optional | core/app tests without harness dependencies; absent-surface test |
| NLOS-331-06 | Present malformed surface/evidence fails closed | path/bounds/schema/digest/provenance and partial-surface tests |
| NLOS-331-07 | Privacy controls cover collection through deletion | approved experiment record, access/retention/deletion audit |
| NLOS-331-08 | No unaccepted high/critical security finding at release | threat-model review, focused regression tests and dependency/secret scans; `NLOS-SEC-EX-001` was closed by lockfile remediation without an exception, while future findings still require correction or an exact signed, unexpired exception record |
| NLOS-331-09 | Optimization does not trade away quality or freshness | golden/reference equivalence and named benchmark report |
| NLOS-331-10 | No hardware/App Store claim from CI alone | PR/release claim check and explicit physical-device witness field |
| NLOS-331-11 | Optional harness-policy evolution cannot self-promote or regress its frozen gates | `MEASURED` frozen held-out/anchor report proving >2% quality lift, <1% cost regression, <5% p95 latency regression, unchanged security/legacy gates, verified provenance and human approval; otherwise discard proposal |

## References

1. ADR-282: mandatory L0–L5 evidence ladder.
2. ADR-295: source provenance state machine.
3. ADR-303/304/305: ground-truth synchronization, evidence engine, authenticated sensor identity.
4. ADR-318/319: capability certificates and witness chain.
5. MIT, [Consumer NLOS project and 30 Hz demonstration](https://cornar.media.mit.edu/).
6. Somasundaram et al., [Nature paper](https://doi.org/10.1038/s41586-026-10502-x).
7. STMicroelectronics, [VL53L8CH raw histogram interface](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html).
