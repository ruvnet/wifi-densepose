# Tick 12 — 2026-05-22 06:08 UTC

**Thread:** R3 (cross-room re-ID)
**Verdict:** Cross-room re-ID is **technically feasible** (MERIDIAN closes the env-shift gap) and **ethically constrained** (4 additional privacy constraints beyond R14 baseline).

## What shipped

- `examples/research-sota/r3_crossroom_reid.py` — pure-numpy simulation of person + environment + noise decomposition with 4 K-NN configurations.
- `examples/research-sota/r3_reid_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R3-crossroom-reid.md` — synthesis of AETHER (ADR-024) + MERIDIAN (ADR-027) + privacy framing + physics-informed extension path.

## Headline numbers

| Configuration | 1-shot accuracy |
|---|---:|
| Within-room (matches AETHER ~95%) | **100%** |
| Cross-room, raw cosine K-NN | 70% |
| Cross-room, MERIDIAN 100% env removal | 100% |
| Cross-room, MERIDIAN 70% env removal (realistic) | 100% |
| Chance | 10% |

The 30 pp gap from within-room to raw cross-room is exactly the angular contribution of the env-shift that cosine similarity can't normalise away. MERIDIAN-style per-room centroid subtraction recovers it — even at 70% effectiveness (realistic for limited labelled examples).

## Privacy constraints surfaced

R14 baseline (opt-in default, on-device data, one-tap override) + **4 new constraints specific to re-ID**:

1. No cross-installation linkage (each install = isolated embedding space)
2. Embedding storage requires explicit opt-in (biometric-class consent)
3. Cryptographically verifiable forgetting (not just unlabelled storage)
4. No re-ID across legal entities (hard-walled inter-org boundaries)

These rule out: cross-building tracking, mass surveillance, long-term unlabelled storage, third-party data sharing. They allow: per-installation personalisation, household anomaly detection, multi-person pose association in the same room.

## Why R3 matters as a synthesis

R3 closes the loop on the empathic-appliance vision from R14: re-ID is **the** primitive that makes per-occupant features possible (V1 stress-responsive lighting needs to know it's "this person", not "any person"). Without R3, R14's verticals can't ship; with R3 + its privacy constraints, they can.

It also identifies the **next research lever**: physics-informed env_sig prediction from R6's forward operator + a room map → zero-shot transfer without labelled examples in the new room.

## Composes cleanly

- **R5/R6**: person + env decomposition lives in the embedding space; physics-informed env prediction is the unbuilt sophistication.
- **R7**: mincut multi-link consistency = defence against re-ID spoofing.
- **R9**: RSSI K-NN showed env-locality dominance for the K-NN primitive; CSI is harder but the same decomposition works.
- **R14**: the four R3 privacy constraints extend R14's framework to biometric-class data.

## Honest scope landed

- Additive decomposition is a first-order model; real CSI env effects are multiplicative in subcarrier domain
- The 70% raw-cosine K-NN number depends on env / person scale ratio (here ~4.7×)
- Adversarial scenarios not simulated; R7 mincut would weigh in

## Coordination

`ticks/tick-12.md`. No PROGRESS.md edit. Branch `research/sota-r3-crossroom-reid`.

## Remaining threads

R4 (federated learning), R15 (RF biometric across rooms — now partly subsumed by R3).

~5.8h to cron stop. 12 threads landed (2 negative results, 1 synthesis).
