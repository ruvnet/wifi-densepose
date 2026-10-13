# Active evidence, KGE proposals and predictive CSI experiment

Research implementation, measured on October 13, 2026 UTC. No field accuracy or
production readiness claim. Keep the existing sensing runtime and model defaults.

## Implemented outcome

The actual pinned RuField and WorldGraph libraries now run an authored sequence
that transitions through empty, occupied, unknown, occupied and unknown. A
conflicting observation creates uncertainty. Expired observations cannot keep an
occupancy belief alive. Historical graph nodes remain provenance, not a current
occupancy API. Repeated, derived, invalid, stale or future evidence is rejected.

The separately trained actual `@ruvector/kge` package emits bounded ranking JSON.
Rust accepts only three registered logical sensor names and finite scores. It
reserves an authored packet, airtime and energy budget for proposals, never
actuates. An integrated verifier compares the default and KGE replays and requires
identical evidence and belief fields. Only the proposal can differ.

The logical radar and door sources currently use signed WiFi feature fixtures to
exercise the pinned rule. They are not implemented physical radar or door
drivers. All replay observations, costs, relevance labels and thresholds are
SYNTHETIC. KGE scores and fusion heuristics are not calibrated probabilities.

## Decisions

* [ADR 421: evidence lifecycle](adr/ADR-421-active-evidence-loop.md)
* [ADR 422: KGE proposal authority](adr/ADR-422-kge-measurement-proposals.md)
* [ADR 423: predictive representation experiment](adr/ADR-423-predictive-csi-representation-experiment.md)

## Measured results

| Experiment | Result | Decision |
| --- | --- | --- |
| Rust active loop, three rounds of 1,000 alternating pairs | Median 0.0771 to 0.0853 ms; p95 0.1187 to 0.1377 ms | Retain bounded research implementation |
| Added Rust cost over signed single root fusion | Paired extra p95 0.0412 to 0.0576 ms | Small measured software overhead, no sensing benefit claim |
| Actual KGE, six authored held out target links | Hits@1 6/6; frequency and shuffled graph controls 2/6 each | Retain synthetic integration test, not real utility evidence |
| KGE reference ranking, three rounds of 1,000 calls | Median 0.0345 to 0.0349 ms; p95 0.0506 to 0.0577 ms | Retain original default |
| Cached KGE validation experiment | Fails lower median and p95 in every round gate | Reject optimization; preserve initial failure receipt |
| Predictive CSI, fixed five epochs on original training data only | Latent MSE 0.9740 versus global constant 0.9413 | Reject utility/promotion claim |
| Predictive frozen encoder timing | Median 0.919 to 1.175 ms; p95 1.326 to 1.478 ms | Timing only; not a sensing model improvement |

Rust and Python ran on the existing ruOS Linux x86_64 AMD EPYC environment.
Rust 1.90.0 used the release profile and exact existing lockfile. Python used
Torch 2.11.0+cpu and NumPy 2.4.4 with one compute thread. Node KGE ran separately
on local Linux x64 Node 24.19.0, native backend. Do not add these timings to claim
a measured end to end latency across machines. Rust excludes signing, acquisition,
ranking transport and hardware. KGE excludes training and loading. Encoder timing
includes input validation, log normalization, token encoding and pooling but
excludes acquisition and file loading. This is not ESP32 timing or peak RAM.

The predictive experiment fit 280 recordings and evaluated 139 inner development
recordings from the original training partition. The remaining 418 training
recordings were unused. No original validation or final test arrays were decoded.
These development groups have been reused in previous research and are not a new
independent confirmation set. Training took 7.34 seconds in the recorded scope.
The same final teacher targets and masks were used for every loss comparator:
untrained 1.1399, global constant 0.9413, per position constant 1.0487, trained
0.9740. Effective recording rank was 1.4055 and triggered the collapse warning.
Reloaded embeddings were bit identical. A loss improvement against an untrained
network is insufficient when a constant predictor performs better.

No candidate from this experiment replaces the activity classifier, the published
eight feature encoder or any production model. No paid compute or hardware
operation was performed. Existing ruOS CPU resources were used.

## Reproduce the software acceptance test

From the repository root, with Node, Python and Rust 1.90.0 installed:

```sh
cd scripts/active-kge
npm ci --ignore-scripts --no-audit --no-fund
npm test
cd ../..
cargo +1.90.0 test --locked --manifest-path scripts/field-context-probe/Cargo.toml
node scripts/active-kge/cli.mjs --train /tmp/ruview-kge-new --context context_1_8 > /tmp/ruview-ranking.json
cargo +1.90.0 run --release --locked --manifest-path scripts/field-context-probe/Cargo.toml -- --active-loop > /tmp/ruview-reference.json
cargo +1.90.0 run --release --locked --manifest-path scripts/field-context-probe/Cargo.toml -- --active-loop --ranking /tmp/ruview-ranking.json > /tmp/ruview-candidate.json
python scripts/verify-active-replay.py /tmp/ruview-reference.json /tmp/ruview-candidate.json
```

Use a new output directory for each KGE training run. The Linux file reader
supports x86_64 and AArch64; other platforms must explicitly supply ranking JSON
on stdin using `--ranking -`. Built replay executables perform no network or
hardware activity. Initial npm/Cargo installation requires package access.

Python training requires `scripts/requirements-csi-training.txt`. Run:

```sh
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s scripts/tests -p 'test_csi_training*.py' -q
python scripts/pretrain-csi-predictive.py --synthetic --output /tmp/ruview-predictive-new --epochs 5 --max-seconds 120
```

For the public run, replace `--synthetic` with `--manifest` pointing to the
existing ADR 420 verified prepared manifest, with five sample temporal pooling.
Do not reinterpret different input axes or change splits to obtain a favorable
result. See [predictive harness contract](../scripts/csi_training/PREDICTIVE.md).

## Validation and security

The completed focused suites passed 41 Python tests, 14 Rust tests and 5 KGE
tests. The integrated replay verifier passed. An independent review found no
remaining source blocker after fixes. The first remote Python run lacked the
unchanged collector helper; restoring that test dependency produced the passing
run. A KGE test initially depended on the current directory; its CLI path now
resolves relative to the test file and passes from the repository root.

| Boundary | Check and residual limit |
| --- | --- |
| Observation admission | Signature, unique root/hash, monotonic source and application clocks, explicit expiry, derived root rejection; public fixture keys are not production enrollment |
| Rejected observation | Does not advance the accepted clock or mutate the graph; exhausted capacity intentionally sets a sticky safety state |
| Ranking file | Open once with nofollow and nonblocking flags, validate the opened regular descriptor, bound bytes, reject duplicate JSON keys, unknown fields and nonfinite scores |
| KGE artifact | Bound file reads and native dimensions; exact authored vocabulary, relations, indexed triples, training splits and training receipt binding; checksums are not signer authentication |
| Predictive artifact | Bound descriptor reads, strict JSON, exact finite tensor shape/dtype checks, safetensors, source hash before/after and reload parity; local artifact trust still required |
| Authority | No hardware, network endpoint or executable supplied by ranking; proposals cannot modify roots or promote models |
| Evaluation | Unknown counts as an occupied miss; separate unknown coverage limits prevent abstention from manufacturing gains; bootstrap by room |

Scoped npm audit of the exact KGE lockfile returned zero reported vulnerabilities
on October 13, 2026 UTC. This is not a security audit of all RuView dependencies,
the native binary internals or external device firmware. Rust pins and its lockfile
were unchanged. No fresh Rust advisory database audit was performed.

## Field measurement contract

`python scripts/evaluate-active-evidence.py /path/to/aligned-intervals.json`
computes aggregate duration metrics and seeded room clustered percentile
intervals. It accepts schema version 1, declared evidence, and sessions with
room/session IDs, matching baseline/candidate packets, airtime, energy and sensor
availability, and contiguous aligned intervals with independently supplied empty
or occupied truth plus empty/occupied/unknown predictions. Input is a bounded
regular JSON file. See `test_csi_training_active_evaluation.py` for the minimal
synthetic input generator.

The numerical gate requires at least 20% lower false occupied duration, a 95%
room clustered interval supporting reduction, recall delta lower bound at least
minus one percentage point, and unknown duration delta upper bound at most one
percentage point separately for empty and occupied intervals. Zero baseline
false alarms or inadequate bootstrap exposure produce inconclusive results.
Two rooms are a computational minimum, not a claim of sufficient statistical
power. Pilot data must size a separate confirmation study.

Labels, actual costs, fresh splits and capture completeness require independent
verification. Consequently even a numerical pass always returns
`fieldAcceptance: not_established` and `promotionAuthorized: false`. The paired
transition gate requires empty-to-occupied and occupied-to-empty events in at
least two rooms. Room-clustered upper confidence bounds must show no increase in
interval-aligned delay, unresolved transitions or false-occupied episodes per
empty hour. Unknown predictions extend delay; a transition that never resolves
within its truth run is retained with that run's full duration. `overallGate`
passes only when both duration and transition gates pass. Stationary occupant
sensitivity and sub-interval acquisition latency still require field analysis.
Replay cannot compare an action for which no eligible measurement was captured.

## Receipts and rollback

Receipts contain aggregate results and hashes, not captures or learned tensors:

* `validation/active-loop-benchmark-20261013.json`
* `validation/active-kge-field-replay-20261013.json`
* `validation/active-kge-20261013.json`
* `validation/active-kge-cache-rejected-20261013.json`
* `validation/active-predictive-public-20261013.json`
* `validation/active-evidence-verification-20261013.json`

Rollback is removal of the isolated experiment and its CI additions. Existing
production paths and published weights are unchanged. The next useful investment
is independently annotated continuous capture, followed by frozen room/session
confirmation and an inductive sensor utility experiment, not promotion of the
rejected predictive checkpoint.
