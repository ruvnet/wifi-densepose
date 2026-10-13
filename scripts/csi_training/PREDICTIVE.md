# Experimental masked latent CSI prediction

This small CPU harness is inspired by the masked latent prediction idea associated
with CSI-JEPA and AM-FM. It is **not a reproduction** of either system. It makes
no occupancy, pose, activity-accuracy, robustness or SOTA claim. No downloads,
remote weights, publishing or automatic promotion occur.

```sh
python scripts/pretrain-csi-predictive.py --synthetic --output /tmp/csi-predictive-run --epochs 5 --max-seconds 120
# Optional already-authorized local ESP-Fi prepared manifest:
python scripts/pretrain-csi-predictive.py --manifest /path/prepared/manifest.json --output /tmp/csi-predictive-public --epochs 5 --max-seconds 120
python -m unittest discover -s scripts/tests -p test_csi_training_predictive.py
```

Dependencies are the existing Torch, NumPy and safetensors training environment.
The existing eight-summary-feature dataset and encoder are incompatible with this
experiment's explicit amplitude axes and are not silently converted.

## Fixed model and evidence boundary

Input is nonnegative amplitude `[recording, 52 carrier indices, 190 pooled sample
indices]`. A patch spans four carriers by 19 samples, giving 130 tokens. Carrier
order and sample index do not imply calibrated physical coordinates or time.
A 32-dimensional, one-layer, four-head transformer encodes patches. The online
model predicts an EMA teacher's stopped latent targets on 52 masked patches per
recording. Target patch values are replaced by a mask token before context
attention. Normalization uses per-carrier log-amplitude statistics fitted only
on the training split, never per-recording normalization that could reveal
masked values. An additional sample-variance penalty discourages collapse.

This is a baseline prototype: the variance penalty is not a guarantee against
collapse. Reports include sample standard deviation, effective rank of pooled
recording embeddings, and a warning. Low latent error alone does not establish
a useful representation.

Public intake reuses the existing hash/provenance-verified ESP-Fi adapter's
train-only partition path. Original validation and test array payloads are
never decoded. Inner training uses room 1 / participants 1–4; inner heldout uses
room 2 / participants 5–6. Other original training recordings are excluded.
These rows have previously participated in earlier training-fold research, so
this is **adaptive research**, not a new independent confirmation population.
No activity label becomes an occupancy label. Within this run heldout signals
never fit normalization, gradients, EMA state, checkpoints or hyperparameters.
The synthetic mode is explicitly labeled SYNTHETIC.

Epoch count and all hyperparameters are fixed before evaluation. After the final
step, heldout latent MSE is measured for the trained predictor, its untrained
initial copy a global constant and a per-token positional constant estimated only on training
teacher targets. All four
use the same final teacher targets and same evaluation masks. Teacher coordinates
move during training, so this comparison is a representation diagnostic, not
an external task metric. No success threshold or promotion is inferred from it.

The per-token baseline controls positional priors: beating a global constant
alone does not establish learned signal structure. The mask-conditioned variance
penalty can also reflect mask noise; unmasked heldout diagnostics are reported.

Frozen encoder inference is timed in three rounds of 100 rotating batch-one
calls after ten warmups, including input checks, log normalization, encoding and
pooling, excluding capture and file IO. These CPU timings are not occupancy
accuracy evidence.

The time budget is checked between optimization/evaluation steps; it cannot
preempt a running Torch operator. Incomplete runs have failed status. CPU seed42,
deterministic Torch operations and fixed masking seeds are recorded in the
contract. Exact replay still depends on the software/platform environment.

## Frozen downstream interface

`FrozenEncoder(directory).encode(amplitudes)` returns `[N,32]` embeddings using
a bounded, hash-checked safetensors encoder and train normalization. Artifact JSON rejects duplicate keys and nonfinite numbers. Reads use no-follow,
nonblocking open and validate the actual file descriptor as a bounded regular
file; unsupported platforms fail explicitly. Source hashes are captured before
training and rechecked before completion. Hashes are
integrity checks, not signatures. Encoder parameters are frozen. The exported
artifact contains the online encoder, not a resumable teacher/optimizer checkpoint.

`bundle.new_adapter(classes)` creates an **untrained** linear head. Its forward
method accepts amplitudes and updates only that head if the caller trains it.
It does not infer task semantics. A supervised study needs real task labels,
fresh grouped confirmation data and appropriate baseline comparisons. Existing
heldout diagnostics cannot authorize occupancy deployment.
