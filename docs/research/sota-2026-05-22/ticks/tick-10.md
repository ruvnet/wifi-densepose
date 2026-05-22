# Tick 10 — 2026-05-22 05:46 UTC

**Thread:** R11 (maritime / through-bulkhead sensing)
**Verdict:** Physics scrutiny re-frames "through-bulkhead" to "through-seam" — the romantic submarine-radar vision is impossible at WiFi bands; the actual product category is **gasket-leakage sensing**.

## What shipped

- `examples/research-sota/r11_maritime_propagation.py` — pure-numpy skin-depth + lossy-dielectric saltwater + slot-diffraction physics for 7 maritime scenarios.
- `examples/research-sota/r11_maritime_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R11-maritime-sensing.md` — research note with the physics, verdicts table, feasible/infeasible verticals, honest scope, composition with prior threads.

## Headline (verdict table)

| Scenario | Verdict | Margin |
|---|---:|---:|
| Man-overboard surface @ 200 m | ✅ | +25 dB |
| Through 10 mm closed steel door | ❌ | -938 dB |
| Through cabin door **2 mm seam** | ✅ | **+31 dB** |
| Through cabin door **5 mm seam** | ✅ | +39 dB |
| Container w/ 30 mm vent slot | ✅ | +45 dB |
| Submarine 30 mm pressure hull | ❌ | -929 dB |
| Head 30 cm underwater | ❌ | -231 dB |

Key physics: steel skin depth = **3.25 µm at 2.4 GHz** (impassable). Saltwater = **853 dB/m**. The loophole is **slot diffraction** through gasket seams.

## Feasible verticals catalogued

1. Man-overboard surface detection (200 m range)
2. Through-seam crew vitals (lone-watch monitoring without compromise)
3. Container tamper detection (cargo security)
4. Hatch-seal integrity audit (predictive maintenance)
5. Engine room thermal-anomaly detection (via condensation envelope)

## What this matters for the loop

R11 is the first thread that **explicitly debunks** a romantic 10-20y framing. The "through-bulkhead" terminology used in the original PROGRESS.md is physically wrong; the actual category is "through-seam". Replacing one vision with a more honest one is the kind of progress this loop is meant to surface.

Composes cleanly:
- R6 Fresnel envelope + slot diffraction = narrower composite envelope
- R10 link-budget primitives reused unmodified for air-side maritime
- R7 multi-link consistency essential for adversarial-resistant maritime
- R14 privacy framework transfers directly to crew-cabin monitoring

## Honest scope landed

- Best-case ignores vessel vibration, engine ignition noise, salt-spray, multipath
- Vibration (5-30 Hz) is **in-band** with R10's gait frequencies — maritime gait-classification harder than land
- No GPS in steel compartments — alternative positioning needed

## Coordination

`ticks/tick-10.md`. No PROGRESS.md edit. Branch `research/sota-r11-maritime`.

## Remaining threads

R3 (cross-room re-ID), R4 (federated), R13 (contactless BP — likely negative-result candidate), R15 (RF biometric).

~6.3h to cron stop. 10 threads landed.
