# PROOF — reproduce every claim, or find the one we can't yet

This project (RuView / wifi-densepose) has been publicly called "AI slop" and
"fake." This document is the answer: **a skeptic can clone the repo, run one
script, and have every headline claim either verified on their own machine or
shown — explicitly — as "CLAIMED, not yet reproduced (here's exactly what it
needs)."** Nothing below is asserted without a command you can run.

```bash
git clone https://github.com/ruvnet/RuView && cd RuView
bash scripts/prove.sh          # core gate + the anti-slop assertion tests
bash scripts/prove.sh --full   # also attempt the feature-gated subset
```

`prove.sh` exits 0 only if every **non-gated** claim passes. Gated claims never
fail the run; they print the prerequisite (a GPU, a dataset, real hardware, a
trained checkpoint) so you can reproduce them yourself.

## Grading

- **MEASURED** — reproduced on our hardware, with the exact command recorded, and
  pinned by a test that *fails on the pre-fix code*. `prove.sh` re-runs these.
- **CLAIMED** — cited from a source, or measured by the source, but not
  reproduced in this repo's automated harness.
- **DATA-GATED / HARDWARE-GATED** — the *code path* is real and tested, but the
  *accuracy/throughput claim* needs data or hardware we don't ship. We never
  fabricate the number; the code carries a typed error or a `weights_trained`/
  provenance flag instead.

## The hard gate (run on any machine with Rust + Python)

| Claim | Grade | Reproduce |
|---|---|---|
| Rust workspace: 3,128 tests, 0 failed | **MEASURED** | `cd v2 && cargo test --workspace --no-default-features` |
| Deterministic CSI pipeline proof (bit-exact SHA-256) | **MEASURED** | `python archive/v1/data/proof/verify.py` → `VERDICT: PASS` |

## Anti-slop assertion tests (each fails on the pre-fix code)

| Claim | Grade | Test (run via `cargo test -p <crate> <name>`) |
|---|---|---|
| Fusion crafted-input DoS panics are closed (ADR-156 §2.2) | **MEASURED** | `wifi-densepose-ruvector :: triangulation_out_of_range_index_returns_none_no_panic` |
| **The "Soul Signature" identity claim, honestly bounded:** on WiFi-only cardiac+respiratory channels two people are **not separable** (gap ≈ 0.0005) | **MEASURED** | `wifi-densepose-bfld :: cardiac_alone_cannot_separate_identity_matches_audit` |
| OccWorld `predict()` is real (input-dependent), not random noise | **MEASURED** | `wifi-densepose-occworld-candle :: predict_is_deterministic_for_same_input` |
| Pose runtime emits frames under its own default config (ADR-159 A1) | **MEASURED** | `cog-pose-estimation :: default_config_emits_frames_with_real_model` |
| Person-count flags untrained classes — no count inflation (ADR-159 A2) | **MEASURED** | `cog-person-count :: untrained_class_argmax_is_flagged_low_confidence` |
| Medical edge skills carry a "not a medical device" disclaimer (ADR-160 A1) | **MEASURED** | `wifi-densepose-wasm-edge :: a1_med_modules_have_clinical_disclaimer` (`--features std`) |
| Survivor dedup 3→1, count-inflation killed (ADR-158 §2) | **MEASURED** | `wifi-densepose-mat :: test_identical_vitals_no_location_dedup_to_one` (`--features mat`) |

## Measured performance (criterion; reproduce on your machine)

| Claim | Grade | Reproduce |
|---|---|---|
| PSD FFT-planner cache 2.0–3.1×, DTW band 2.4–4.1× (ADR-154) | **MEASURED** | `cd v2 && cargo bench -p wifi-densepose-signal` |
| fuse() double-clone removed ~2.17× marshalling (ADR-156) | **MEASURED** | `cd v2 && cargo bench -p wifi-densepose-ruvector --bench fusion_bench` |
| zero-copy ORT input ~1.48× (ADR-155) | **MEASURED** | `cd v2 && cargo bench -p wifi-densepose-nn --features onnx --bench onnx_bench` |
| pointcloud splats 9→2 passes ~1.24× (ADR-160 research) | **MEASURED** | `cd v2 && cargo bench -p wifi-densepose-pointcloud --bench splats_bench` |
| native wlanapi multi-BSSID scan 9.74 Hz (vs netsh ~2 Hz) | **MEASURED (Windows)** | `cd v2 && cargo test -p wifi-densepose-wifiscan -- --ignored measure_native_scan_rate` |
| wasm-edge `process_frame` hot-path latency (host proxy, ADR-163) | **MEASURED-on-host** (NOT the ESP32/WASM3 budget — needs hardware) | `cd v2/crates/wifi-densepose-wasm-edge && cargo bench --features std` |
| cog steady-state CPU infer latency ~305 µs (ADR-163; NOT the manifest cold-start) | **MEASURED-on-host** | `cd v2 && cargo bench -p cog-person-count -p cog-pose-estimation --no-default-features --bench infer_bench` |

## What we do NOT claim (the honest negatives — the strongest anti-slop signal)

| Capability | Status |
|---|---|
| **Named person-identity from WiFi** | **NOT achieved, and measured why.** The §3.6 matcher is real, but identity does not lock on WiFi-only channels (gap 0.0005). DATA-GATED on a real enrollment feeding the AETHER/body-resonance channel — never done. No named-identity claim is made. |
| WiFlow-STD ~96% PCK@20 | **CLAIMED-reproduced** on our RTX 5080 (`benchmarks/wiflow-std/RESULTS.md`); HARDWARE-GATED for you (needs an NVIDIA GPU + the MM-Fi dataset). The upstream *shipped checkpoint* was **REFUTED** (0.08% PCK) — we publish that. |
| OccWorld trajectory accuracy | DATA-GATED on a trained checkpoint; `predict()` carries `weights_trained=false` until one is loaded — never silently faked. |
| Edge-skill detection accuracy (seizure, weapon, affect, …) | UNVALIDATED — every such module is now disclaimer-gated as experimental/research; the DSP is real, the accuracy is not claimed. |
| 802.11bf-2025 OTA conformance | No commodity silicon ships a conformant interface as of 2026; ours is a simulation-tested forward-compat protocol model, not a certified implementation. |

## Provenance

Every claim above traces to a committed ADR (`docs/adr/ADR-154`…`ADR-163`), a
test, a criterion bench, `benchmarks/wiflow-std/RESULTS.md`, or
`benchmarks/edge-latency/RESULTS.md`. The history
includes published **retractions** (the 92.9% PCK retraction; the WiFlow-STD
shipped-checkpoint refutation; the NV-diamond BOM reality check) — a faker hides
failures; we commit them.
