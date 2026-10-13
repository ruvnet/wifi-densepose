# Selective public CSI intake

## Scope and frozen acceptance

2026-10-13: implement a research intake and format diagnostic for
[Zenodo 18924941](https://zenodo.org/records/18924941), version 2 published
2026-06-25. Baseline PR2256 head `606de9f64f931b62bb5f0daed457c2d2a602b797`,
tree `7efe6c1ac475598ba428fad9b782dd8eb4ff41b7`, has no intake for this source.
SciPy `whosmat` on the first acquired member rejects MATLAB v7.3; this source
requires HDF5. That is a format gap, not a model performance baseline.

Before diagnostic payload measurement: freeze the three capture1 beacon members
selected in `zenodo_sample.SESSIONS`, first 32 frames only, native antenna and
subcarrier axes, finite complex values, no external HDF storage or plugins,
maximum 4 MiB compressed member and decompression chunk, 12 MiB aggregate HTTP
range budget. Acceptance is successful bounded intake and correct format/quality
reporting plus all adversarial tests. Reject malformed inputs; do not adjust
signal quality thresholds after looking at results. No accuracy threshold,
training, preprocessing fit, probe, inference optimization, or model selection.
Use existing CPU resources, no training, at most 30 minutes active work. Rollback
removes these two modules, tests, optional requirements and CI install line.

Source files at diagnostic freeze:

* `zenodo_sample.py`: `6ea89c1e1a2d254665d1d44f04270893acabba3e3ce6924745cc7c66efcf7009`
* `zenodo_mat.py`: `0fb071712ab357f08667248eaaceddb28b8c66e6da28db67231a0f87b281d52d`

## Data meaning and exposure

The source README describes Nexmon complex CSI, with empty, sitting and walking
recordings. The selected configuration flags explicitly identify presence and
movement; activity numbers are not used to infer occupancy semantics. This
diagnostic does not predict occupancy. The author describes collection in 2025
centered on one office. Do not treat three recordings as independent room,
participant or device generalization. Participant identity, device aliases and
stable physical group identity remain unverified.

All selected members and the source revision are now exposed for exploratory
development. Never present them as fresh confirmation or reopen a previously
held final test. Any future scientific split requires reviewed source aliases,
room/person/session/device provenance and complete exposure history. There is no
training adapter or automatic connection to the ADR425 advisory preflight.

The catalog declares CC-BY-4.0 while its README declares CC-BY-SA-4.0. Preserve
both notices and source attribution. The discrepancy remains unresolved; raw
samples stay private, and no dataset or model redistribution is authorized here.
This is not a license compatibility decision.

## Reproduce

The downloader uses the standard library and fetches only six explicitly named
members, writing flat filenames into a new private directory. It never extracts
archive paths. The optional diagnostic requires NumPy and pinned h5py:

```sh
python -m pip install -r scripts/requirements-zenodo-intake.txt
PYTHONPATH=scripts python -m unittest discover -s scripts/tests -p test_csi_training_zenodo.py -v
PYTHONPATH=scripts timeout 150s python -m csi_training.zenodo_sample --output /private/zenodo-new
(ulimit -v 1048576; OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts timeout 30s \
  python -m csi_training.zenodo_mat /private/zenodo-new/empty.mat \
  --sha256 561f2dd6ac9d08597365d90b08da829bdce99cba003261986a946e41fd0ba46b \
  --all-frames)
```

Use a trusted private parent directory. Download failures retain a failed status
and partial files; reruns require a new directory. Local SHA256 pins are observed
content identities, not signatures from the dataset publisher. The server
supplied no ETag during acquisition. Exact HTTPS ranges and ZIP CRC do not verify
the full 16,125,799,937 byte archive or guarantee a single immutable snapshot.
The receipt says both explicitly. Transfer took 22 requests and 5,670,108 bytes
in 16.844 seconds on existing ruOS CPU infrastructure. This is a single intake
observation, not a paired speed benchmark or inference improvement.

## Security boundary

HTTP rejects redirects, ignored or incorrect ranges, encoding changes and ETag
changes; enforces byte, request and between-request time budgets. A socket timeout
is not a hard total wall clock deadline; use the executor timeout as the outer
limit. ZIP checks member uniqueness, compression method, encryption, symlinks,
expanded size and CRC. Full archive parsing is bounded by the range read budget.

The diagnostic reads a bounded regular descriptor with nofollow/nonblock,
checks its SHA256, and opens HDF from those bytes. It rejects soft/external links,
virtual/external storage, unexpected axes/dtypes, oversized chunks and filters
other than builtin deflate/shuffle/Fletcher32 before reading the selected window.
The default inspects only the first 32 frames; `--all-frames` scans every CSI
frame in 32-frame slices. Decoding every CSI value does not validate source
provenance, labels, other HDF objects or signal usefulness. Native HDF5 parsing
is not sandboxed by Python metadata checks. Run hostile inputs in an externally
bounded process. No network or sensor actions occur during diagnostics. No claim
of a complete dependency vulnerability audit is made.

## Result

ACCEPT for draft research intake review. All three first windows contain finite
values; zero amplitude fractions are 0.071686, 0.093506 and 0.086975. These are
reported diagnostics, not a quality threshold or evidence of class separation.
Seven offline adversarial tests pass on ruOS in 0.124 seconds. Independent source
review found no high or critical blocker within the private research scope;
outer time/memory limits are required for native parsing. No new paid resources
were used; incremental existing infrastructure cost is unmetered. Full raw
receipts and acquired samples remain private. Sanitized aggregates are in
[the diagnostic receipt](validation/zenodo-intake-20261013.json).

Next: resolve the license discrepancy and establish stable physical provenance
before any training or independent evaluation. Main risk is mistaking these
exposed, correlated office recordings for independent confirmation.

## Frozen whole-recording diagnostic followup

Frozen before decoding beyond the first 32 frames. Baseline is draft PR 2256
commit `47589e00f636971ec7e0ca2483d4809f9ebcf775`, whose diagnostic reads only the
first 32 frames. Candidate source SHA256 is
`b2d53f65ac865860baf30c2ac9cb844b6f0be50346db2f9354c88754451c62f2`;
synthetic evaluator SHA256 is
`3e0e03eac570f93eb150dd8d8e51df23a68e09c43732c0608408a129d67064cb`.
The three already exposed MAT identities remain `561f2dd6ac9d...`,
`4db0d302c6b6...`, and `7c1d78deb3fd...`. No new split exists and no labels are
read. This is a capture integrity and coverage diagnostic, not a model
experiment.

The single candidate scans every native frame in fixed slices of at most 32,
reports finite complex count, zero amplitude by antenna, per-frame nonzero
coverage and amplitude range, and never emits an occupancy decision. Acceptance
requires exact input hashes, successful complete scans for all three recordings,
all finite decoded values, no full decoded CSI array, synthetic tail corruption
detection, all focused tests, at most 30 seconds and 1 GiB virtual memory per
sample, and `training_eligible=false` plus `promotionAuthorized=false`. No
quality threshold is selected after seeing the data. Full-scan measurements are
descriptive and cannot authorize training. Rollback restores the first-window
diagnostic and its prior tests.

The frozen candidate was accepted for draft review. It reproduced every prior
first-window aggregate, then decoded all 596/597/597 frames in 19 slices each:
1,832,960 complex values were finite, with no all-zero frame. Whole-recording
zero amplitude fractions were 0.077581 empty, 0.093155 sitting and 0.084006
walking. Minimum per-frame nonzero coverage was 0.251953, 0.879883 and 0.851562,
respectively. The low empty-sample minimum is a capture-quality observation to
investigate, not evidence for or against occupancy. The three scans plus baseline
reproduction took 0.221 seconds and process peak RSS was 56,528 KiB under the
frozen outer limits on existing ruOS CPU infrastructure. This was one run, not a
latency optimization benchmark. Eight focused adversarial tests pass in 0.151
seconds after the review fixes. Sanitized aggregates are in
[`validation/zenodo-full-quality-20261013.json`](validation/zenodo-full-quality-20261013.json);
raw captures and the detailed private receipt remain untracked.

## Locate the worst coverage frame

The diagnostic now reports `worst_frame_index_zero_based`,
`worst_frame_tie_count`, and `worst_frame_nonzero_fraction_by_antenna`.
Indices refer to the native CSI frame axis, not timestamps. Ties select the
earliest frame; the antenna vector belongs to that frame and retains native
antenna order. Both minima and tie counts cover only `inspected_frames`:
the default still checks the first 32 frames, while `--all-frames` checks all.
No capture-quality threshold, occupancy classification or training gate is added.

Frozen baseline: `34eb522ec81c944b12988dd7422795a45f41058c`.
Acceptance required exact preservation of every old report field in both modes,
agreement with an independent frame-by-frame nonzero-count oracle, adversarial
tests for chunk-boundary ties, antenna order and unread tails, and the existing
30-second/1-GiB verification limits. Source, evaluator and input identities are
in [the frozen protocol](validation/zenodo-location-protocol-20261013.json).
The same three already exposed recordings were used; there is no new split,
training, model selection or final-test access. The implementation retains
32-frame reads and at most one eight-element antenna vector, with no extra
dataset read. Existing file/hash/HDF validation is unchanged.

ACCEPT for draft diagnostic review: ten focused tests pass. Every old field and
the prior whole-scan receipt reproduced exactly. The independent oracle agrees
in both modes for all three samples. The empty-sample minimum is at frame 69
(zero-based), with native antenna nonzero fractions 0.246094, 0.261719, 0.257812
and 0.242188. This localizes the previously reported drop across all four
antennas; its physical cause remains unknown. The sitting and walking minima
are at frames 350 and 396. These filenames are source labels, not predictions.
Verification took 1.650 seconds with peak RSS 57,700 KiB on existing ruOS CPU;
this includes baseline, candidate and oracle work and is not a speed comparison.
See [the aggregate receipt](validation/zenodo-location-20261013.json).

Reproduce using the diagnostic and test commands above. Security review confirms
no new input, network, file or model authority; output is bounded to three extra
fields and contains no raw signal values. Raw receipts remain private. Existing
infrastructure cost is unmetered; no paid resource or training was started.
Rollback restores the diagnostic and test from the baseline commit.
Next: correlate frame 69 with acquisition metadata before assigning a cause;
license and physical provenance still block model use.
