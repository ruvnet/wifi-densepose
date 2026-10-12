# Public amplitude model optimization

This experiment trains a separate seven-class activity model from public
ESP-Fi HAR amplitudes. It does not replace the eight-feature Hugging Face
encoder, estimate pose or establish real hardware performance.

## Reproduce

Use Python 3.12, a fresh virtual environment and a data directory outside Git.
Install the CPU PyTorch wheel explicitly before the remaining dependencies.
Linux also needs the system libarchive shared library to read the RAR archive.

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/requirements-espfi-training.txt
PYTHONPATH=scripts python -m unittest discover -s scripts/tests -p 'test_espfi*.py' -v

# This is a public 645 MB download, pinned to exact upstream bytes.
curl --fail --location --output /private/espfi/source.rar \
  https://media.githubusercontent.com/media/AutoSmartGroup/ESP-Fi-HAR/c8bf6aa0d680fa02695c85a82b05e79b3ebb10c4/ESP-Fi_dataset.rar
PYTHONPATH=scripts python -m csi_training.espfi \
  --archive /private/espfi/source.rar --output /private/espfi/prepared
python scripts/train-espfi.py \
  --manifest /private/espfi/prepared/manifest.json \
  --output /private/espfi/run-001 --epochs 30 --seed 42 --max-seconds 600
```

Intake verifies the full archive SHA256 before decoding. It reads bounded MAT
members directly without extracting archive paths. It preserves 52 amplitude
channels and averages groups of five temporal samples into 190 time points.
Source timestamps are unavailable, so the preprocessing makes no calibrated
sampling frequency claim. Data are licensed CC BY 4.0; keep the recorded source
and Wen et al. citation with derived work.

## Frozen evaluation rules

Training uses rooms 1 and 2 with participants 1 through 6. Validation uses
room 3 with participant 7. Test uses room 4 with participant 8. Other
combinations are excluded to avoid sharing either room or participant across
partitions. Duplicate signals are quarantined, with counts in provenance.
The small final test measures one particular joint room/person transfer.

The source README and benchmark code disagree on activity names. Class labels
remain `activity_1` through `activity_7`; do not call any output a fall detector.

The declared search compares a statistics MLP with temporal networks of widths
16 and 32. Training statistics fit the scaler. Validation macro F1 selects the
checkpoint and architecture. The final test is opened after selection, with a
fixed training-majority control. Do not use that score to tune a later run and
continue calling it untouched.

## Artifacts and acceptance

The bundle contains safetensors, preprocessing, metrics and lifecycle status.
It records implementation/source hashes, per-class scores, confusion matrix,
parameter count and artifact bytes. Inference folding must preserve logits
within tolerance; exported reload is checked against the selected model.

```python
# PYTHONPATH=scripts; input shape N,52,190 after the same temporal pooling.
from csi_training.espfi_train import AmplitudeBundle
model = AmplitudeBundle('/private/espfi/run-001')
logits = model.predict(amplitudes)
```

CPU timings are batch one and exclude preprocessing and I/O. They are not
ESP32 timings or end-to-end response times. Failed and corrupted bundles are
rejected. `promotionAuthorized` remains false. Acceptance requires passing
contract tests, exact source verification, disjoint groups, successful bundle
reload and a recorded final evaluation. Production qualification additionally
requires resolved labels and independent field hardware data.

See [ADR-420](adr/ADR-420-public-amplitude-model-optimization.md).

## Measured public data pilot, October 12, 2026

The pinned archive contained two exact duplicate groups comprising four
recordings. All four were quarantined, leaving 837 training recordings and
70 each for validation and test under the declared joint split. Another
1,260 source recordings fall outside the split intersections, with one of
these also in the duplicate quarantine. No signal arrays are stored in Git.

| Candidate | Parameters before inference folding | Validation macro F1 |
| --- | ---: | ---: |
| Statistics MLP | 5,255 | 0.327905 |
| Temporal width 16 | 1,399 | 0.072092 |
| Temporal width 32 | 4,871 | 0.324024 |

The validation-selected statistics model achieved 22/70 correct (31.43%) and
0.288679 macro F1 on the final test. The fixed majority control achieved
10/70 (14.29%) and 0.035714 macro F1. Each test error changes accuracy by
1.43 percentage points. This is inadequate for deployment, and neither
room-wide nor population-wide performance can be inferred from one person
in one room.

The selected safetensors file is 21,316 bytes, SHA256
`42fad3cfb9bf532643c2bead4b4c5e51bd446c88e0c18686ad9f7df80d5cf781`.
The run used seed 42, a 30-epoch ceiling, early stopping patience 5, CPU one
thread, torch 2.11.0+cpu and NumPy 2.4.4. Its measured model-only latency
was p50 0.09645 ms and p95 0.15732 ms; preprocessing and I/O were excluded.
This timing is an observation on a shared x86_64 host, not a device SLA.
The three-candidate training loop took 5.75 seconds, excluding intake.

Decision: retain as a research candidate. The next accuracy experiment needs
domain-generalization work and an independent confirmation set. Do not tune
against this opened final test or replace the deployed eight-feature encoder.

## Inference optimization of the frozen candidate

The statistics model permits an algebraic optimization: channel scaling of
mean, standard deviation and absolute temporal differences can be folded into
the first linear layer. This avoids normalizing every amplitude value before
statistics extraction. No retraining or test-based selection is involved.

```sh
python scripts/train-espfi.py \
  --manifest /private/espfi/prepared/manifest.json \
  --optimize-bundle /private/espfi/run-001 \
  --output /private/espfi/run-001-folded
```

The transformed bundle records parent digests and validation-only parity;
it does not copy the parent's test score as a newly measured result. All 70
validation predictions remained identical, with maximum logit difference
0.000004054 and exact reload parity.

A paired alternating benchmark over 1,000 repetitions measured p50 bundle
prediction latency of 0.171623 ms before and 0.144312 ms after folding, a
15.9% reduction (1.189x speedup). p95 was 0.233717 and 0.204566 ms. This
includes input checks and normalization but excludes temporal pooling and I/O.
An earlier 300-pair measurement also improved p50 by 15.2%, but worsened p95
from 0.316924 to 0.349389 ms. Shared-host timing noise prevents a stable tail
latency claim. Both observations are retained; no hardware SLA is implied.

## Development-only domain generalization

The follow-up search uses a separate loader that does not deserialize test
arrays. The old final test has already been opened and cannot be reused as an
independent acceptance set. The existing validation set is also development
evidence, including when a new candidate outperforms the original model there.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/improve-espfi.py \
  --manifest /private/espfi/prepared/manifest.json \
  --baseline /private/espfi/run-001 \
  --output /private/espfi/generalization-001 --max-seconds 600
```

The frozen grid comprises raw statistics, relative temporal features, relative
spectral features and a combined family, each at ridge alpha 0.1, 1, 10 and 100.
Relative features log amplitudes, remove per-channel temporal means, and
normalize by each recording's motion RMS. Frequency bands are in normalized
sample units; no physical sampling rate is inferred.

Two folds use only original training data. Each fits on participants 1 through
4 in one room and evaluates participants 5 and 6 in the other room, then swaps
room direction. Scalers fit only fold training data. Mean fold macro F1 selects
the family and alpha, with feature count and declaration order breaking ties.
Only then is the model fit to the full original training partition and compared
with the frozen baseline on the existing 70-record development partition.

Every candidate score is retained, including losing candidates. Artifacts
include exact source and model digests, safe reload checks, latency scope and
explicit nonpromotion status. No claim of improved final-test performance
follows from this development search.

### Measured development result

The train-only search selected relative spectral features with ridge alpha 10.
The mean inner-fold macro F1 was 0.297443. Best scores by family were 0.200165
for raw statistics, 0.214085 for relative temporal features and 0.278602 for
the combined family. All 16 configurations are retained in the receipt.

On the existing 70-record development partition, macro F1 increased from
0.327905 to 0.366065, an absolute gain of 0.038160 and relative gain of 11.64%.
Accuracy remained 29/70 (41.43%). The three previously zero-F1 classes gained
nonzero scores, while classes 1, 2 and 3 regressed. This is a balanced-metric
tradeoff, not uniform improvement. No new final-test metric was computed.

The classifier has 3,283 fitted parameters and 936 additional preprocessing
scalars. Its first float64 safetensors artifact is 34,016 bytes, larger than
the original float32 MLP despite fewer model parameters. The 16-configuration
search took 0.81 seconds on this CPU, excluding intake. Feature extraction,
scaling and classification measured p50 0.129795 ms and p95 0.180376 ms over
200 calls; this unpaired timing alone is not proof of a speedup.

### Frozen float32 export

After statistical selection was complete, the fitted tensors and extracted
feature vector were converted to float32. Feature extraction itself retains
float64 arithmetic. The exported file is 17,140 bytes, 49.6% smaller than the
float64 candidate and 19.6% smaller than the original 21,316-byte MLP.
All 70 development predictions remain identical; maximum logit difference is
6.663e-7 and exported reload is exact. These are precision checks, not new
accuracy evidence.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/improve-espfi.py \
  --manifest /private/espfi/prepared/manifest.json \
  --baseline /private/espfi/run-001 \
  --optimized-baseline /private/espfi/run-001-folded \
  --optimize-bundle /private/espfi/generalization-001 \
  --output /private/espfi/generalization-001-float32
```

The paired benchmark rotates call order over 1,000 repetitions. It includes
complete public inference APIs, input/output checks and preprocessing, but
excludes acquisition, temporal pooling and I/O. PyTorch and BLAS use one thread.

| Bundle | p50 ms | p95 ms |
| --- | ---: | ---: |
| Original MLP | 0.196385 | 0.284381 |
| Previously folded MLP | 0.164547 | 0.256346 |
| Spectral ridge float64 | 0.178709 | 0.258502 |
| Spectral ridge float32 | 0.150447 | 0.224474 |

Median latency is 8.57% lower than the strongest previous folded baseline in
this experiment. Shared-host timing is not a device SLA. Load the candidate
with `csi_training.espfi_generalize.GeneralizationBundle`, which includes
the required feature transform, scaler and classifier. Never feed it the
old encoder's eight engineered features.

Acceptance remains a new untouched room and participant, resolved semantic
labels, and target-device measurement before any production promotion.
