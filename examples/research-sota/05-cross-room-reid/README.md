# 05 — Cross-room person re-identification (R3 arc, 3 ticks)

Whether the same person can be identified across two different rooms with WiFi CSI. Three-tick arc: R3 baseline → R3.1 (architecture error negative) → R3.2 (corrected architecture, structurally validated).

## Scripts (in arc order)

| Script | Thread | Headline | State |
|---|---|---|---|
| `r3_crossroom_reid.py` | R3 | MERIDIAN env-centroid subtraction recovers cross-room re-ID to 100% (synthetic 128-dim embedding setup) | POSITIVE |
| `r3_1_physics_informed_env.py` | R3.1 | **Physics-informed env subtraction at raw-CSI level FAILS** (10% = chance). Even labelled MERIDIAN at raw CSI = 10%. Identifies architecture error: position-variance dominates at raw level. | **NEGATIVE (architecture-error)** |
| `r3_2_embedding_physics_env.py` | R3.2 | Embedding-level physics-informed env: **20% = matches labelled oracle with zero labels**. Architecture correct; per-subject signal weak in synthetic AETHER stand-in. | **STRUCTURALLY VALIDATED** |

## The three-kind-of-negative pattern

R3.1 was the loop's first **architecture-error negative**:

| Kind | Example | Path forward |
|---|---|---|
| Missing-tool (revisitable) | R12 → R12 PABS | Tool arrives later; approach works |
| Physics-floor (permanent) | R13 contactless BP | Hard wall; no tool changes this |
| **Architecture-error (correctable)** | **R3.1 (here)** | **Wrong application level; corrected sketch explicit** |

## Corrected architecture (R3.2)

```
raw CSI → AETHER embedding (position-invariant, ADR-024)
              → physics-informed env subtraction (uses R6.1 forward operator)
                  → cosine K-NN against per-installation gallery
```

The physics-informed env prediction must operate at **embedding level**, not raw level. AETHER does position-invariance; predicted-env removes the remaining room-shift component.

## Privacy constraints (R14 + R15 + ADR-106/107)

Cross-room re-ID is a powerful biometric. The loop's privacy framework enforces:

1. **No cross-installation linkage** — each install has its own embedding space
2. **Storage requires explicit opt-in** (biometric-class consent)
3. **Cryptographically verifiable forgetting** of raw primitives
4. **No re-ID across legal entities** (hard-walled)

ADR-107 Layer 5 (per-installation embedding-space rotation key) enforces these technically.

## See also

- Research notes: `docs/research/sota-2026-05-22/R3-*.md`, `R3_1-*.md`, `R3_2-*.md`
- ADRs: ADR-024 (AETHER), ADR-027 (MERIDIAN), ADR-106 (DP+isolation), ADR-107 (cross-install)
- Composes with: `01-physics-floor/` (R6.1 forward operator)
