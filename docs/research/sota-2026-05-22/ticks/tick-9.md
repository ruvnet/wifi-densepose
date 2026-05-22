# Tick 9 — 2026-05-22 05:34 UTC

**Thread:** R1 (ToA multistatic CRLB)
**Verdict:** Quantitative precision floor for WiFi multistatic localisation. Phase ranging beats ToA ranging by **238×** at WiFi bandwidths — but only after solving the integer-ambiguity (cycle-slip) problem.

## What shipped

- `examples/research-sota/r1_toa_crlb.py` — pure-numpy CRLB grid over bandwidth/SNR + phase-noise-vs-precision grid + 4-anchor multistatic geometric dilution.
- `examples/research-sota/r1_toa_crlb_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R1-toa-crlb.md` — research note with the math, the headline numbers, the integer-ambiguity catch, ADR-029 architectural implication.

## Headline numbers

**20 MHz HT20 channel, 20 dB SNR (ESP32-S3 typical):**

| Method | Single-shot | 100x averaged |
|---|---:|---:|
| ToA CRLB | 0.413 m | 0.041 m |
| Phase (single-subcarrier, 5° noise) | **1.73 mm** | 0.17 mm |
| **Phase advantage** | 238× | 240× |

**4-anchor multistatic 5×5 m room, GDOP 1.5:**

| Method | Position precision |
|---|---:|
| ToA | 25.3 cm |
| Phase (ambiguity-resolved) | 1.06 mm |

## Why this matters for the loop

1. **Bounds what's physically possible** for any WiFi-localisation feature. 25 cm position precision via ToA-only is the room-pose-quality floor; 1 mm via phase is RTK-quality but ambiguity-resolution-bound.
2. **Strongest architectural lever for ADR-029**: explicit ToA-then-phase pipeline (≤2× from CRLB by Kay's theory) probably outperforms the current learning-based attention. Provable optimality vs flexibility tradeoff.
3. **Composes cleanly with R6**: spatial envelope (R6) × ranging precision (R1) = full multistatic geometry budget. They are independent and additive.
4. **Closes a gap R10 created**: foliage drops SNR, which directly worsens ToA CRLB. A 50 m foliage link at 5 dB SNR → ~1 m ToA precision. The 100 m sparse-foliage number from R10 is **not** the same as 100 m localisable.

## Honest scope landed

- CRLB is a lower bound; real estimators sit 1-2× above it
- 5° phase noise assumes `phase_align.rs` is applied; raw ESP32 is 60-180°
- Multipath degrades CRLB by 2-5× even with MUSIC super-resolution
- Cycle-slip is unsolved at the WiFi bandwidth level without multi-subcarrier wide-lane unwrap

## Coordination

`ticks/tick-9.md`. No PROGRESS.md edit. Branch `research/sota-r1-toa-crlb`.

## Remaining threads

R2 (subsumed by R6+R12), R3 (cross-room re-ID), R4 (federated learning), R11 (through-bulkhead maritime), R13 (contactless BP), R15 (RF biometric).

~6.4h to cron stop. 9 threads landed.
