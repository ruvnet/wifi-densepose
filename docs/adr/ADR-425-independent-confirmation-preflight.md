# ADR 425: Independent confirmation preflight

Status: accepted for experimental metadata validation only.

## Decision and rationale

ADR 424 improved representation rank while failing useful activity prediction. Further tuning on reused development records cannot establish independent generalization. Add a small standard library preflight before a future independent experiment. It does not modify current training, decode captures, invoke sensors, evaluate models, or authorize promotion.

Inputs are a versioned manifest of opaque record identifiers, canonical signal digests, source revisions, split assignments, room, participant, session, device and numeric activity; plus a versioned ledger of previously exposed sources. Output is a deterministic aggregate report with input digests, class and group counts, reasons, and `promotionAuthorized=false`. No record identifiers appear in output. Direct API callers must supply bounded Python data; CLI intake reads bounded regular JSON files with duplicate key and nonfinite constant rejection. Symlinks and FIFOs are rejected. Input files remain private and untracked.

All three partitions must have disjoint rooms, participants and sessions. Device separation is required only for the explicit `new_room_person_device` claim. Group identifiers must refer to globally stable physical groups across datasets; adding dataset prefixes to the same physical room would evade the check and is prohibited by the intake contract. Canonical signal digests must be generated from the same documented signal representation, before window augmentation, by trusted intake. The gate rejects duplicate digests even when record IDs differ. It cannot identify near duplicates or malicious relabeling.

All seven numeric activity classes must occur in training and development. Confirmation requires at least ten recordings per class and at least two rooms and two participants overall. This is a minimum coverage screen, not a statistical power calculation. It does not prove each class covers every room or participant. Correlated trials are not independent participants. A future frozen protocol must set scientifically appropriate group coverage and power before collection.

Any source revision in the supplied exposure ledger is barred from confirmation. The previously inspected ESP-Fi HAR revision is also embedded in the validator so an empty ledger cannot erase that known exposure. Exclusion is deliberately conservative at source revision granularity. A renamed source, changed revision, omitted history or fabricated metadata can evade this unsigned local check. The experiment owner must review source aliases and the complete exposure history. Passing metadata is never a confirmation result.

The manifest binds a protocol SHA256 but does not verify that a corresponding protocol was registered before observation. A reviewer must verify protocol timing, capture provenance, consent and license suitability, model and baseline freezing, and independence before an actual confirmation run. This is a preflight advisory CLI, not an enforced gate in every training entrypoint.

## Control flow and alternatives

Validate exact schemas and bounds; collect global record and content sets; aggregate groups and class counts by partition; compare every partition pair; reject exposed confirmation sources and insufficient coverage; emit only aggregate evidence. Malformed or unsafe input exits 1, a valid but blocked plan exits 2, and a metadata pass exits 0. Every path keeps promotion disabled. No state or ledger is modified, so reruns are idempotent.

Automatic retraining was rejected because it would consume compute without fixing the evidence limitation. A signed acquisition registry would authenticate more of the chain but requires real capture infrastructure and key ownership that this change does not supply. The standard library preflight is reversible and inexpensive; adding its module has no production inference cost.

## Acceptance trace

The synthetic tests cover a valid plan, all leakage axes, renamed duplicate recordings, known and caller declared source exposure, missing classes, insufficient room diversity, device claim scoping, malformed schema, bounded reads, duplicate JSON keys, nonfinite JSON, symlinks, FIFOs, and CLI exit codes. A pass means those software invariants hold on supplied metadata only. Tests are included by the existing `test_csi_training*.py` CI glob.

## Use and rollback

```sh
PYTHONPATH=scripts python -m csi_training.confirmation \
  --manifest /private/confirmation-plan.json \
  --ledger /private/exposure-ledger.json
PYTHONPATH=scripts python -m unittest discover -s scripts/tests \
  -p test_csi_training_confirmation.py -v
```

The exact schema and a generated synthetic example are in `scripts/tests/test_csi_training_confirmation.py:fixture`. Never substitute that fixture for public or physical evidence. Remove the module and test to roll back; no model, production service or database changes exist.

Next action: assemble a reviewed inventory of fresh rooms, participants and device captures, then freeze the actual scientific comparison. Main risk: an incomplete exposure ledger or inconsistent physical identifiers. Acceptance: a fresh plan passes, and injecting one known source, shared room, participant, session, device under its applicable claim, or duplicate signal causes a nonzero exit before any training starts.
