# Predictive transfer experiment, 2026-10-13

Decision: reject the candidate for promotion. Recording diversity improved, but useful activity prediction did not beat the controls. A reduction in collapse is not evidence of sensing accuracy.

ADR 424 froze the protocol before the single public campaign. Its SHA256 is `609430cefbfdea4d8c5c70a7a016987475e9751ad418f730acd0db0f1b3acee2`. The aggregate receipt is [predictive-transfer-20261013.json](benchmarks/predictive-transfer-20261013.json). It binds source hashes, data revision, manifest, initialization, mask order, artifacts, metrics, and all acceptance checks. No captures, record embeddings, or weights are committed.

The public campaign used 280 fitting recordings from room 1 participants 1 through 4 and 139 development recordings from room 2 participants 5 and 6, drawn exclusively from the original training partition. This development split has been reused: results are exploratory, not independent confirmation. Original validation and final test payloads were not decoded by this campaign. Both arms used five epochs, seed 42, batch eight, and identical masks and ordering. The original baseline encoder and normalization reproduced exactly against its pinned prior artifact before candidate evaluation.

| Frozen representation | Macro F1 | Accuracy | Dimensions |
|---|---:|---:|---:|
| Original encoder | 0.03502 | 0.13669 | 32 |
| Candidate | 0.05163 | 0.15108 | 32 |
| Untrained encoder | 0.07121 | 0.15827 | 32 |
| Raw statistics | 0.19939 | 0.25180 | 156 |
| Training majority | 0.03436 | 0.13669 | 0 |

All learned probes used training only scaling, an intercept, fixed ridge alpha 1, and seven numeric activity classes. Raw statistics use more features and fitted scalars (1,411 versus 295). These are activity metrics, not occupancy or motionless person detection metrics.

The candidate's recording effective rank rose from 1.40553 to 4.01552, passing the rank threshold of two. It introduced no new zero F1 class, but five classes still had zero F1. Its macro F1 failed the required two percentage point margin over the strongest comparator; accuracy also failed the one percentage point tolerance. The combined research gate is false.

Three paired CPU timing rounds alternated model calls on identical inputs, with 100 measurements per round. Candidate p95 was 1.64015, 1.40467, and 1.43937 ms; baseline p95 was 1.72908, 1.65279, and 1.48452 ms. All passed the 20% regression budget. These timings include input checks, normalization, and encoding, but exclude acquisition and IO. Shared host noise prevents a speedup claim. The campaign took 20.2021 seconds on ruOS, Torch 2.11.0+cpu and NumPy 2.4.4, one compute thread.

Validation: all 50 `test_csi_training*.py` tests passed in 11.848 seconds on ruOS. The new tests check exact baseline reproduction, development mutation independence, frozen probe math, bounds, safe artifact reload, and failure states. Independent source review found no blocker. A NumPy warning about casting the float32 magnitude bound occurred in validation; finite float32 values remain below that bound and nonfinite values are rejected. It does not change the reported results.

Security: artifacts require bounded regular files, strict schema and tensor validation, safe tensor decoding, finite values, and complete status. The public runner requires the pinned baseline artifact. Source identity is checked before and after the campaign. No artifact authorizes promotion. This review is scoped to this experimental surface, not a repository wide security certification.

Reproduction on the prepared public manifest:

```sh
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/pretrain-csi-transfer.py \
  --manifest /path/to/prepared/manifest.json \
  --baseline-artifact /path/to/prior/model.safetensors \
  --output /path/to/new-run --max-seconds 120
```

Next decision: prioritize room and device generalization, collection diversity, and comparison against a strong statistics model before another large encoder campaign. Do not tune repeatedly on these 139 recordings. A future frozen candidate must beat all controls on independent rooms and participants, retain calibration and unknown behavior, and meet the inference budget. Physical motionless occupancy still requires suitable independently labeled measurements. WorldGraph, RuField and KGE cannot manufacture missing evidence.
