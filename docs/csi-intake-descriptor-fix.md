# Capture intake descriptor repair

Decision: ACCEPT the bounded intake repair for draft review. No model or production deployment changes.

The existing loader checked path metadata and later reopened the path. Two synthetic regressions reproduce on commit `721a094f09b97422e32555fc8444cdaa17c5fd3a`: replacing a capture with a symlink after inventory was accepted, and files growing after inventory could exceed the aggregate byte budget while satisfying their expected hashes. Hash verification alone does not enforce file type or resource bounds. A substituted FIFO could also block the former normal open.

The loader now opens with `O_NOFOLLOW | O_NONBLOCK`, checks the opened descriptor with `fstat`, and consumes that same descriptor. It rejects nonregular files and enforces cumulative actual bytes before parsing each line. Manifest reads are bounded to the observed size plus one and reject length changes. Every descriptor closes on success or failure. No new dependency or model behavior is introduced.

Seven synthetic regressions cover symlink and FIFO substitution, actual aggregate growth, descriptor binding after pathname replacement, manifest growth, manifest bounds, and valid reload. The prior loader failed the first two reproductions. All 65 training tests passed on ruOS in 9.780 seconds. The preexisting probe magnitude warning remains unrelated. Valid baseline and candidate partitions have identical features, timestamps, nodes, groups, sessions, inventory and manifest digest.

[Benchmark receipt](benchmarks/intake-descriptor-20261013.json): 1,000 alternating paired loads after 20 warmups, three synthetic files and nine records. Full intake p50 is 0.180943 ms baseline and 0.203307 ms candidate; p95 is 0.218539 ms and 0.241545 ms. The measured security cost is about 0.023 ms at p95 (10.5%) on this tiny warm cache workload. Python traced allocation peaks are 15,118 and 16,670 bytes. These are local shared CPU measurements, not inference latency, large capture scaling or field accuracy. No training, final test decoding or paid compute occurred. Incremental infrastructure cost was not metered.

An intermediate implementation eagerly allocated about 1 MB when reading a tiny manifest. The observed size bound removed that avoidable allocation before final measurement. This was a resource correction, not a tuning of acceptance thresholds.

Reproduce the regression tests and trusted baseline comparison:

```sh
PYTHONPATH=scripts python -m unittest discover -s scripts/tests -p test_csi_training_intake_races.py -v
git show 721a094f09b97422e32555fc8444cdaa17c5fd3a:scripts/csi_training/data.py > /tmp/ruview-intake-baseline.py
PYTHONPATH=scripts python scripts/benchmark-csi-intake.py --baseline /tmp/ruview-intake-baseline.py
```

The benchmark executes only the exact hash pinned baseline source after checking it; it rejects arbitrary supplied Python. The required historical commit must be present in the checkout. Synthetic files are generated in a temporary directory and removed afterward.

Residual boundaries: parent directories remain trusted, with no filesystem confinement claim. Concurrent writes to an already open capture inode rely on the pinned content hash and read budgets; the loader does not create an immutable filesystem snapshot. Metadata inventory now requires read permission because it opens descriptors. Platforms without `O_NOFOLLOW` and `O_NONBLOCK` fail closed, matching the experimental artifact readers. This is a scoped source and regression review, not a repository wide security certification.

Rollback is reverting this scoped patch; no state migration is needed. Next priority remains useful capture/data integration. Acceptance: the seven regression tests pass and a valid pinned dataset retains exact partition parity. Missing physical sensing evidence is unchanged.
