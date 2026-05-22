# Tick 5 — 2026-05-22 03:45 UTC

**Thread:** R12 (RF weather mapping — structural drift from passive ambient WiFi)
**Verdict:** Negative-ish result with a clearly-actionable revision path. **Honest progress.**

## What shipped

- `examples/research-sota/r12_rf_weather_eigenshift.py` — pure-NumPy demo that tests "can SVD-eigenvalue drift detect a synthetic structural perturbation?"
- `examples/research-sota/r12_rf_weather_results.json` — full numbers.
- `docs/research/sota-2026-05-22/R12-rf-weather-mapping.md` — research note covering: 10-year vision, first-experiment method, **negative result**, why it failed, three concrete revisions for next attempts (PABS / per-subcarrier residuals / multi-day baseline), what still holds, vertical applications.

## Headline numbers

| | Cosine distance from baseline |
|---|---|
| Control (no perturbation) | 0.00035 |
| With 15% attenuation on 3 top-saliency subcarriers | 0.00024 |
| Signal / natural-drift ratio | **0.69×** |

The synthetic perturbation produced a *smaller* spectral distance than natural temporal drift from operator movement. The top-K SVD-spectrum distance approach is too coarse.

## Why this is still useful

1. **Saves anyone going down this path** the time of trying naive SVD-distance — the data tells us it's the wrong primitive.
2. **Identifies the right primitives:** principal angles between subspaces (PABS), per-subcarrier residual analysis, multi-day baselines.
3. **Cross-validates R5:** task-specific saliency (count) ≠ task-specific saliency (structure detection). Same model, same data — different relevant features. Publishable distinction.
4. **Confirms R12 is CSI-only:** RSSI is the trace of the CSI covariance matrix; if top-10 SVD can't see this perturbation, RSSI definitely can't. Bounds R8's commercial-enablement story to counting only.

## What's queued for later ticks

- Implement PABS-based change detection.
- Per-subcarrier residual time-series analysis.
- Acquire (or simulate) multi-day data with a known structural change.

## Coordination note

This tick wrote NOTHING to `PROGRESS.md` to avoid races with the horizon-tracker agent (which is on the `feat/ruview-mcp-m*` track and editing PROGRESS.md concurrently). The `ticks/tick-N.md` convention used here means each cron-driven tick is fully self-contained — the final 08:00 ET summary script will consolidate them.
