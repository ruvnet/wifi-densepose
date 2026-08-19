---
name: verify
description: Prove a RuView result is real — run the deterministic SHA-256 proof and the witness bundle (ADR-028), and lint any claim for MEASURED-vs-CLAIMED honesty.
---

# verify

The "prove everything" skill. Nothing ships as validated without this.

## Deterministic proof (Trust Kill Switch)

`ruview_verify` runs `archive/v1/data/proof/verify.py`: it feeds a reference signal
through the production pipeline and hashes the output against
`expected_features.sha256`. Must print **VERDICT: PASS**. If numpy/scipy changed the
hash, regenerate with `verify.py --generate-hash` then re-verify.

## Witness bundle (ADR-028)

For a release-grade attestation:

```
bash scripts/generate-witness-bundle.sh
cd dist/witness-bundle-ADR028-*/ && bash VERIFY.sh   # must be 7/7 PASS
```

Contains the Rust test log, the proof + expected hash, firmware SHA-256 manifest, and
crate versions — a recipient can re-verify with one command.

## Claim honesty

Run `ruview_claim_check {text}` on any report, README section, PR body, or model card
before quoting accuracy. It flags:
- untagged accuracy numbers (must be MEASURED / CLAIMED / SYNTHETIC),
- MEASURED claims with no reproducer cited,
- the retracted "100%/perfect accuracy" framing.

## Firmware-specific

A firmware fix is **not** "hardware-validated" without a captured boot log on real
silicon (e.g. the `v0.8.1-esp32` rev-v0.2 validation: `running headless so CSI
captures (#1000)` + `CSI filter upgraded to MGMT+DATA` + a no-false-detect mmwave
probe). Do not merge or release on a build-passes signal alone.
