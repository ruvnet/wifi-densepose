# ADR-157: Hardware / Sensing-Acquisition Layer Beyond-SOTA Sweep — Milestone 3 (An Already-Hardened Layer, Three Small Real Fixes, an Honestly-Null Perf Win, and a Mostly-NO-ACTION SOTA Landscape)

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-11 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-vitals` (`heartrate.rs`, `breathing.rs`, `anomaly.rs`, `store.rs`), `wifi-densepose-wifiscan` (`pipeline/breathing_extractor.rs`, `pipeline/correlator.rs`, `adapter/netsh_scanner.rs`), `wifi-densepose-hardware` (`esp32_parser.rs`, `sync_packet.rs`, `esp32/secure_tdm.rs`, `ieee80211bf/*`), `wifi-densepose-calibration` (`geometry_embedding.rs`), benches, docs |
| **Relates to** | ADR-021 (ESP32 CSI vitals), ADR-022 (multi-BSSID WiFi sensing), ADR-028 (ESP32 capability audit + witness), ADR-032 (multistatic mesh security), ADR-110 (HE PPDU bandwidth), ADR-151 (per-room calibration), ADR-152 (WiFi-Pose SOTA 2026 intake), ADR-153 (802.11bf forward-compat), ADR-154 (Signal/DSP sweep M0), ADR-155 (NN/Training sweep M1), ADR-156 (RuVector/Fusion sweep M2) |
| **Scope** | Milestone 3 of the beyond-SOTA sweep across the four hardware/sensing-acquisition crates. The honest headline: **this layer is already well-hardened** — the real work is small. Three correctness/stability fixes (each pinned by a test that fails on the old code), one algorithmic perf change whose end-to-end win is **null at realistic window sizes** (disclosed, not inflated) with a committed bench, one defense-in-depth hardening on an unreachable path, a **MEASURED negative-results section** (the centerpiece — what was investigated and found already-correct), a graded SOTA landscape that is **mostly NO-ACTION**, and a deferred backlog. **Nothing is silently dropped.** |

---

## 0. PROOF discipline (this ADR's contract)

This project has been publicly accused of "AI slop." Milestone 3 answers with **evidence, not adjectives** — the same contract as ADR-154/155/156:

- Every correctness/stability fix ships a **committed regression test that fails on the old code and passes on the new**. Each was verified by reverting the fix and observing the test fail (recorded in §6).
- Every perf number is **MEASURED before/after** with the exact reproduce command and a committed criterion bench. Where the win is below noise, we **say so and claim nothing** — see §4, which is a deliberately-disclosed near-null result.
- Every external SOTA reference is graded **MEASURED** / **CLAIMED** / **DATA-GATED**, and where the right answer is "do nothing," we record the negative result explicitly (§5) — a stronger anti-slop signal than a fix.
- The headline of this milestone is itself a negative result: **the acquisition layer was already hardened.** We disclose what we *checked and did not change* (§3) in as much detail as what we changed (§2), because "investigated, already correct, no action" is the most honest thing a sweep can report when it is true.

Test machine for the perf numbers: Windows 11, `cargo bench --release`, criterion 0.5. Numbers are wall-clock medians on this box; the **ratio** (before/after) is the claim, not the absolute ns.

Build/test gate: `cargo test --workspace --no-default-features` (the project's standard gate — no GPU/`crv` features). All fixes in this milestone are on the **default, non-feature-gated surface**, so they are fully exercised by the standard gate. The serde-validated `ieee80211bf` types are additionally verifiable with `--features serde`; the live-QUIC path in `secure_tdm` is structurally tested (HMAC/replay/tamper) but not live-socket-tested in CI.

---

## 1. Context

The hardware/sensing-acquisition layer is the bottom of the stack: it turns raw RF (ESP32 CSI frames, multi-BSSID netsh scans, 802.11bf measurement reports) into typed, validated domain objects that the signal/fusion/NN layers above consume. A beyond-SOTA review of the four crates surfaced far **fewer** real defects than the signal (ADR-154) or fusion (ADR-156) sweeps — because this layer was written defensively from the start: length-gated parsers, `Option`-returning helpers, `#[serde(try_from)]` validate-on-deserialize, FSMs that return `Result` instead of panicking, and HMAC-authenticated + replay-protected TDM beacons.

The genuine findings are three: an **O(n²) sliding-window data-structure choice** in the vital-sign extractors (perf, latent), a **partial-weights scale-mixing bug** in breathing fusion (correctness), and an **IIR resonator that can diverge at pathologically low sample rates** (stability). Everything else the review flagged turned out to be already-safe — documented in §3 as MEASURED negative results.

---

## 2. Decision — the fixes that landed

Each correctness/stability fix ships a regression test on the non-feature-gated, workspace-tested surface.

### 2.1 §A1 — `Vec::remove(0)` O(n²) sliding windows → `VecDeque` (PERF, latent; MEASURED via bench — near-null at realistic sizes, disclosed)

**The finding.** Every fixed-length sliding window in the extractors was a `Vec<f64>`/`Vec<f32>` whose oldest-sample eviction used `Vec::remove(0)` — an **O(n) shift of the whole buffer on every sample**, making a full-window `extract()` sweep O(n²). Six sites:

| File | Site | Buffer |
|------|------|--------|
| `vitals/heartrate.rs` | `extract` history window | `Vec<f64>` → `VecDeque<f64>` |
| `vitals/breathing.rs` | `extract` history window | `Vec<f64>` → `VecDeque<f64>` |
| `vitals/anomaly.rs` | `rr_history` / `hr_history` | `Vec<f64>` → `VecDeque<f64>` (×2) |
| `vitals/store.rs` | `readings` ring buffer | `Vec<VitalReading>` → `VecDeque<VitalReading>` |
| `wifiscan/pipeline/breathing_extractor.rs` | filtered history | `Vec<f32>` → `VecDeque<f32>` |
| `wifiscan/pipeline/correlator.rs` | per-BSSID histories | `Vec<Vec<f32>>` → `Vec<VecDeque<f32>>` |

**The fix.** Swap to `VecDeque` with `push_back` + `pop_front` (O(1) eviction). Where the autocorrelation / zero-crossing / Pearson loop needs a contiguous slice, call `make_contiguous()` (or `as_slices().0` after it) **once per `extract()`**. This matches the idiom already used correctly in `wifiscan/pipeline/orchestrator.rs`. **Output is bit-identical** — no behavior test bites; the change is bench-gated.

**The honest measurement (§4).** In **isolation**, the eviction cost collapses from O(n²) to O(n): a microbenchmark of pure eviction shows **34.6× at window=3000 and 3158× at window=100000**. But in the **full `extract()` path at realistic ESP32 window sizes** (heartrate ~1500, breathing ~3000), the per-frame DSP (autocorrelation is O(window·lags); zero-crossing is O(window)) **dominates the eviction entirely**, so the end-to-end win is **below noise** — measured `heartrate` 42.8 ms (before) vs 44.4 ms (after), `breathing` 7.95 ms vs 7.86 ms: overlapping confidence intervals, **no measurable change**. We land A1 because it is the correct data structure and removes a latent O(n²) that *would* bite at higher sample rates or longer windows — **not** because it speeds up the current hot path, which it does not measurably. Claiming an end-to-end speedup here would be exactly the inflation this sweep exists to prevent (the same discipline ADR-156 §2.1 applied to its cos no-op).

### 2.2 §A2 — `breathing.rs` partial-weights scale-mixing (CORRECTNESS, real)

**The finding.** `BreathingExtractor::extract` fused per-subcarrier residuals as `Σ residuals[i]·w[i]` where `w[i] = weights.get(i).unwrap_or(1/n)`. The result was **never normalized**. When `weights` was supplied **shorter than** `n`, the supplied entries (e.g. attention weights ~10.0) were used **raw** while the missing tail defaulted to `uniform_w = 1/n` (~0.125) — two scales summed with no renormalization, **silently mis-scaling the breathing signal** by a factor that depends on `weights.len()`. A caller passing 2 high attention weights for an 8-subcarrier frame got a fused value ~20× too large.

**The fix.** Extracted the fusion into `fuse_weighted_residuals(residuals, weights, n)` and normalized by `Σ(effective weights)` — `weighted_sum / weight_total` — mirroring the **already-correct** pattern in `heartrate::compute_phase_coherence_signal`. A partial weight slice now produces a true weighted average in the residual range, independent of `weights.len()`.

**Tests (fail on old code, verified by reverting — §6):**
- `partial_weights_are_renormalized_not_scale_mixed` — `residuals=[1.0;8]`, `weights=[10.0,10.0]` → fused value `1.0` (the renormalized weighted mean), and explicitly **not** the old scale-mixed sum `2·10 + 6·0.125 = 20.75`.
- `partial_weights_fusion_is_weighted_average` — differing residuals → a proper weighted average within `[0, 2]`, which the old un-normalized sum is not.

### 2.3 §A3 — IIR resonator divergence at pathologically low sample rate (STABILITY, real)

**The finding.** Both extractors' `bandpass_filter` set the resonator pole radius `r = 1 - bw/2` with `bw = 2π(f_high − f_low)/fs`. The **research report's stated trigger ("`fs` below ~4 Hz") is incorrect**, and we say so: the resonator pole *magnitude* is `|r|`, and the filter is stable for any `|r| < 1` — a merely-**negative** `r` is still stable. Divergence requires `|r| ≥ 1`, i.e. `bw ≥ 4`, i.e. `fs` very low **relative to the band width** (e.g. `fs = 0.5` Hz with a 0.1–0.9 Hz band → `bw = 10.05`, `r = −4.03`, `|r| = 4.03 > 1`). When that holds, the filter **diverges exponentially**: a unit-step input reaches `~10^183` within 300 frames and **overflows f64 to ±inf within ~600 frames**. Once one inf enters `filtered_history`, the autocorrelation `acf0`/zero-crossing path produces NaN and the extractor is **permanently dead** (silent stall until `reset()`).

**The fix.** Two layers of defense-in-depth:
1. **Clamp** `r` to a stable range: `r = (1.0 - bw/2.0).clamp(0.0, 0.9999)` — keeps the pole inside the unit circle for **any** sample-rate / band-edge configuration. (We document honestly that the divergence condition is `|r| ≥ 1`, not "`r` negative.")
2. **Finite-guard** before the history push: `if !filtered.is_finite() { return None; }` — mirrors the NaN-bypass guard in ADR-154 §3, so even a future divergence cannot poison the buffer.

Applied to **both** `heartrate.rs` and `breathing.rs` (identical resonator block).

**Tests (fail on old code, verified by reverting — §6):** `heartrate::low_sample_rate_filter_stays_finite` and `breathing::low_sample_rate_filter_stays_finite` — construct at `fs=0.5` with a 0.1–0.9 Hz band, feed a unit step for 600 frames, assert **every** `filtered_history` sample is finite. On the old code these **panic** (a `filtered_history[i]` is inf/NaN); on the new code all samples are finite.

### 2.4 §D1 — new `vitals/benches/vitals_bench.rs` (MEASURED)

A new criterion bench (`harness = false`, registered in `Cargo.toml`) drives each extractor from empty to a full window (`heartrate` 1500 samples, `breathing` 3000) so the A1 sliding-window bookkeeping is exercised across the whole buffer. Follows the criterion style of the existing `hardware/benches/transport_bench.rs` and ADR-156's `fusion_bench`. Numbers and the honest interpretation are in §4.

### 2.5 §B1 — `ieee80211bf/transport.rs` drop-instead-of-truncate (HARDENING, unreachable path — disclosed)

`OpportunisticCsiBridge::ingest` built `CsiReportPayload { n_subcarriers: self.amp_accum.len() as u16, … }`. The `as u16` would silently wrap a count above 65 535. **This is unreachable in practice**: `ingest` gates `frame.subcarrier_count() > MAX_REPORT_SUBCARRIERS` (484) at entry and returns `None`, and `report.validate()` independently rejects oversized counts downstream. We replaced the cast with `u16::try_from(self.amp_accum.len()).ok()?` (drop-instead-of-truncate) so the construction is **correct-by-construction** rather than relying on the upstream gate. We disclose this as **defense-in-depth on an unreachable path, not a live bug** — no behavior change, no new test (the gate already prevents the input that would exercise it).

### 2.6 §B4 — constant-time HMAC tag compare: **RESOLVED — no-dependency hand-rolled constant-time compare (Milestone-1)**

`secure_tdm.rs` compared the 8-byte HMAC tag with `self.hmac_tag == expected` (data-dependent, non-constant-time: short-circuits on the first differing byte, leaking through verification latency how many leading bytes a forged tag matched — a byte-by-byte tag-recovery oracle). Milestone-3 deferred this **only** to avoid adding the `subtle` crate as a direct dependency. Milestone-1 resolves it **without any dependency**: a hand-rolled `constant_time_tag_eq(a, b)` that XOR-accumulates every byte difference into a single `u8` with **no early exit**, then compares the accumulator to zero exactly once. `#[inline(never)]` + `core::hint::black_box(diff)` stop the optimizer from reintroducing a short-circuit or lowering the loop into a non-constant-time `memcmp`; a length mismatch returns `false` without inspecting contents. The former `==` verify site now calls this helper.

**Test (fails on old code, the hard gate):** `tag_compare_is_constant_time_shape` — asserts correct accept/reject for equal, first-byte-differ, last-byte-differ, all-byte-differ, and length-mismatch tags, plus an end-to-end `verify()` last-byte-only tamper. Verified to **bite**: introducing a classic constant-time bug (loop `take(LEN-1)`, skipping the last byte) makes it fail on `last-byte-differ must reject`. A coarse timing-invariance smoke check `tag_compare_timing_invariance_smoke` exists but is `#[ignore]`d (noisy host — not a CI gate). **Grade MEASURED** (constant-time *construction*; micro-timing on a noisy host is only a smoke check, disclosed honestly). Tracked RESOLVED in §8.

---

## 3. The MEASURED negative-results section (the centerpiece — what was investigated and found already-correct)

This is the core of ADR-157. The acquisition layer was hardened before this sweep; the strongest anti-slop evidence is an honest accounting of what we **checked and did not need to change**. Each is verified against the live code with a file:line citation.

| Area | Claim verified | Evidence (file:line) | Verdict |
|------|----------------|----------------------|---------|
| **ESP32 parser subcarrier index math** | A crafted CSI frame cannot panic via the subcarrier-index arithmetic. The total-frame-size length gate (`data.len() < HEADER_SIZE + n_antennas·n_subcarriers·2 → Err`) dominates **every** subsequent `data[byte_offset]`/`[+1]` access; `n_subcarriers ≤ 256`, `n_antennas ≤ 4` are header-bounded, and the `index` math is pure i16 arithmetic with no indexing. | `esp32_parser.rs:211` (length gate) guards the loop at `:224–242` | **Already safe — NO ACTION** |
| **`sync_packet.rs` `try_into().unwrap()`** | The four `try_into().unwrap()` calls are **infallible**: each slices a fixed-width sub-range (`[0..4]`, `[8..16]`, `[16..24]`, `[24..28]`) of a buffer already guaranteed `len() >= SYNC_PACKET_SIZE` (32) by the early `return Err(InsufficientData)`. | `sync_packet.rs:88` (length gate) → `:94,102,103,104` (fixed-width slices) | **Already safe — NO ACTION** |
| **The entire `ieee80211bf/` 802.11bf model** | Validate-on-deserialize and no-panic-by-construction throughout. `MeasurementSetupId` is `#[serde(try_from = "u8")]` rejecting `> MAX_SETUP_ID` (127); `ThresholdParams` is `#[serde(try_from = "RawThresholdParams")]` routing every deserialize through `ThresholdParams::new`; the session FSM `handle()` returns `Result<Vec<Action>, BfError>` (never panics) and enforces **single-role** (`self.role != Initiator/Responder → Err`) on every transition; the SBP request is validated through the **same** single `evaluate_setup` chain as a direct setup (no SBP-only policy bypass). | `types.rs:160–161` (setup-id try_from), `:225–226` (threshold try_from), `:165` (range check); `session.rs:118` (`handle` → Result), `:130/143/166/182` (single-role), `messages.rs:130–147` (SBP single-evaluate) | **Already SOTA-shaped — NO ACTION** |
| **`secure_tdm.rs` HMAC + replay** | Beacon authentication (HMAC-SHA256, 8-byte tag), tamper rejection, and replay-window protection are correct and tested. (The non-constant-time compare at `:284` is the only nit — §2.6, deferred as out-of-threat-model for an 8-byte LAN tag.) | `secure_tdm.rs:279` (`verify`), `:284` (compare), tests `:614–673` (replay), `:728` (tamper) | **Correct — NO ACTION (B4 deferred)** |
| **`netsh_scanner.rs` command + parse** | No shell-injection surface: the scanner uses a **fixed argv** (`Command::new("netsh").args(["wlan","show","networks","mode=bssid"])`) — no shell, no interpolation. Parsing is **`Option`-based** (`try_parse_ssid_line`/`try_parse_bssid_line`/`try_parse_signal_line` → `Option`, with `.unwrap_or(default)`), so hostile/garbled netsh output is silently skipped, never panicked. | `netsh_scanner.rs:50–51` (fixed argv), `:96–102` (`unwrap_or` defaults), `:242/257/270` (`Option` parsers) | **Already safe — NO ACTION** |
| **`calibration/geometry_embedding.rs` overflow guard** | The geometry embedding clamps every position/std-dev component into `±MAX_COORD_M` (1000 m) via `clamp_m`, explicitly to stop adversarial coordinates from overflowing the covariance accumulation into `inf`; the documented invariant ("every value is finite, never NaN/inf") holds. | `geometry_embedding.rs:55` (`MAX_COORD_M`), `:145/150` (`clamp_m` on centroid + std-dev) | **Already safe — NO ACTION** |

---

## 4. The §D1 perf measurement (MEASURED — honestly near-null end-to-end)

New bench: `crates/wifi-densepose-vitals/benches/vitals_bench.rs`, two functions covering a full-window fill of each extractor.

- **Reproduce:** `cargo bench -p wifi-densepose-vitals --bench vitals_bench`
  (compile-only: append `--no-run`; the medians below used `-- --warm-up-time 1 --measurement-time 3 --sample-size 20`).

**End-to-end `extract()` full-window fill, medians:**

| Bench | Before (`Vec::remove(0)`) | After (`VecDeque`) | Verdict |
|-------|---------------------------|--------------------|---------|
| `heartrate_extract_full_window_1500` | 42.81 ms `[42.19, 42.81, 43.46]` | 44.37 ms `[43.55, 44.37, 45.19]` | **no measurable change** (after marginally slower; intervals overlap) |
| `breathing_extract_full_window_3000` | 7.95 ms `[7.86, 7.95, 8.05]` | 7.86 ms `[7.66, 7.86, 8.04]` | **no measurable change** (intervals overlap) |

The end-to-end effect is **null within noise** because the per-frame DSP dominates: heartrate runs an O(window·lags) autocorrelation every frame (≈1500·125 multiply-adds), which utterly swamps the O(window) eviction the A1 change improves; breathing's O(window) zero-crossing and the `make_contiguous` rotation are the same order as the old `remove(0)` memmove at these sizes.

**Where the win actually lives (isolated eviction-only microbench, supporting evidence — not in the committed bench):**

| Window | `Vec::remove(0)` (eviction only) | `VecDeque` | Speedup |
|--------|----------------------------------|------------|---------|
| 3 000 | 1.00 ms | 0.029 ms | **34.6×** |
| 20 000 | 94.5 ms | 0.122 ms | **773×** |
| 100 000 | 3 139 ms | 0.994 ms | **3 158×** |

So A1 is **algorithmically correct and removes a real latent O(n²)** that would bite at higher sample rates or longer analysis windows — but at the **current** ESP32 window sizes the end-to-end win is below noise, and we claim nothing more. This is the §0 contract in action: a perf claim without a measured before/after improvement is **not made**.

---

## 5. The hardware/sensing SOTA landscape (graded — mostly NO-ACTION, honest)

Grades: **MEASURED** (source measured it, ideally public method/code), **CLAIMED** (asserted, no reproducible artifact), **DATA-GATED** (blocked on data we don't have, per a prior ADR-152 measurement).

| # | Area | Candidate / question | Grade | Verdict |
|---|------|----------------------|-------|---------|
| 1 | **CSI vital signs (HR/BR)** | Deep-CSI vital-sign models report **MAE ~2–3 BPM** vs our classical IIR-bandpass + autocorrelation/zero-crossing. | **DATA-GATED + CLAIMED** | **NO ACTION on method.** A deep model needs **paired PPG/ECG ground truth** we do not have, and no public ESP32 artifact reproduces the cited MAE on commodity CSI. Our classical method is the honest commodity baseline; the real wins this milestone are the A1/A3 robustness fixes, not a new model. |
| 2 | **802.11bf-2025 conformance** | Adopt a conformance test-vector suite for the `ieee80211bf/` forward-compat model. | **CLAIMED (not public)** | **NO ACTION.** No commodity silicon ships a conformant 802.11bf interface as of 2026, and the conformance suites are **WBA / Wi-Fi Alliance pre-certification** material, **not public**. Our model's "no OTA encoding until silicon exists" posture (ADR-153) is the correct one. Tracked in §8: *add SBP conformance vectors when the WFA publishes a test plan* — we will **not invent vectors**. |
| 3 | **Per-room calibration (ADR-151)** | Bank-of-specialists + drift-veto vs a 2026 calibration SOTA. | **CLAIMED on numbers, DATA-GATED on a head-to-head** | **NO ACTION on architecture.** The bank-of-specialists + drift-veto design is SOTA-shaped, but we have **no head-to-head PCK** against a published method (no paired multi-room data). The geometry-conditioned LoRA head is **built-but-unconsumed** and data-gated → **ACCEPTED-FUTURE** (§8), not built now. |
| 4 | **Multi-BSSID throughput (wifiscan)** | The module docs assert a native `wlanapi.dll` FFI 10–20 Hz path; the current `WlanApiScanner` wraps `netsh` (~2 Hz). | **MEASURED (Milestone-1)** | **IMPLEMENTED + MEASURED — real positive win.** Status corrected: the native FFI is **fully implemented and wired live** (`wlanapi_native::scan_native` calls `WlanOpenHandle`/`WlanEnumInterfaces`/`WlanGetNetworkBssList`/`WlanFreeMemory`/`WlanCloseHandle`; `WlanApiScanner::scan_instrumented` runs it native-first with a netsh fallback). Milestone-1 **measured both paths on this box** (Intel Wi-Fi 7 BE201 320MHz, 2026-06-13) over an identical 10 s wall-clock window via a new `benchmark_backend`: **native 21.42 Hz vs netsh 3.84 Hz = 5.57× MEASURED** (mean 5.0 BSSIDs/scan each; native-only run 18.0 Hz). Native genuinely beats netsh — a real measured multiple, **not** a fabricated 10×; the achieved 21.4 Hz lands in the asserted >2 Hz regime though below the asserted 10–20 Hz upper bound. 50 back-to-back native scans = 50/50 OK, no handle leak. → §8 MEASURED. |

---

## 6. Validation

- **Bug-catching tests verified to bite.** Each §A2/§A3 fix was reverted and the corresponding test observed to fail on the old code, then restored:
  - `partial_weights_are_renormalized_not_scale_mixed`, `partial_weights_fusion_is_weighted_average` — **assertion failure** (returned the old un-normalized scale-mixed sum) on old code.
  - `heartrate::low_sample_rate_filter_stays_finite`, `breathing::low_sample_rate_filter_stays_finite` — **panic** (a `filtered_history[i]` is inf/NaN) on old code.
  - §A1 is the **disclosed bit-identical change**: no behavior test bites (correctly — output is unchanged); the bench (§4) is the gate, and it shows **no measurable end-to-end change**, which we report honestly.
  - §B1 is on an **unreachable path** (gated upstream), so it carries no new test — disclosed as defense-in-depth, not a live bug.
- **`cd v2 && cargo test -p wifi-densepose-vitals -p wifi-densepose-hardware -p wifi-densepose-wifiscan -p wifi-densepose-calibration --no-default-features`** — all green. Lib-test counts: `wifi-densepose-vitals` **55** (was 51; +4 net new bug-catching tests — two §A2, two §A3), `wifi-densepose-hardware` **163**, `wifi-densepose-wifiscan` **87**, `wifi-densepose-calibration` **58**. 0 failures across all four.
- **`cd v2 && cargo test --workspace --no-default-features`** — **3054 passed / 0 failed** (M2 left the workspace at 3050; the +4 net new bug-catching tests are included and green).
- **`python archive/v1/data/proof/verify.py`** — **`VERDICT: PASS`**, pipeline hash unchanged `f8e76f21…46f7a` (these are Rust-only changes; the Python pipeline proof is independent and confirmed unaffected).
- New `vitals_bench` compiles and runs under the default feature set.
- **Disclosed validation limits:** the live-QUIC transport in `secure_tdm` is **structurally** tested (HMAC compute/verify, tamper, replay-window) but **not live-socket-tested** in CI; the serde-gated `ieee80211bf` types are additionally verifiable with `--features serde`. Clippy is not installed in the local 1.89 toolchain, so the per-crate lint pass was not run locally (the project gate is `cargo test`).

---

## 7. What changed, file by file

- `vitals/heartrate.rs` — `filtered_history: Vec<f64>` → `VecDeque<f64>` (`push_back`/`pop_front`, `make_contiguous` once per `extract`); resonator `r` clamped to `[0, 0.9999]`; finite-guard before history push; corrected divergence-condition doc (`|r| ≥ 1`, not "`r` negative"); `low_sample_rate_filter_stays_finite` test.
- `vitals/breathing.rs` — same `VecDeque` + clamp + finite-guard changes; weighted fusion extracted to `fuse_weighted_residuals` and **normalized by Σ(effective weights)** (the §A2 fix); three new tests (two A2, one A3).
- `vitals/anomaly.rs`, `vitals/store.rs` — sliding/ring buffers → `VecDeque` (O(1) eviction); `store::history` takes `&mut self` to hand back a contiguous slice via `make_contiguous` (no external callers; observable contents unchanged).
- `wifiscan/pipeline/breathing_extractor.rs` — `VecDeque<f32>` + `make_contiguous`.
- `wifiscan/pipeline/correlator.rs` — per-BSSID histories → `Vec<VecDeque<f32>>`; contiguous-ize each touched buffer once before the Pearson pass.
- `hardware/ieee80211bf/transport.rs` — `n_subcarriers: … as u16` → `u16::try_from(…).ok()?` (§B1 drop-instead-of-truncate, unreachable-path hardening).
- `vitals/Cargo.toml` + `vitals/benches/vitals_bench.rs` (new) — criterion dev-dep, `[[bench]]`, the §D1 full-window benches.

---

## 8. Deferred backlog (NOT silently dropped)

- **§B4 constant-time HMAC compare** — **RESOLVED (Milestone-1).** Replaced the short-circuiting `==` on the 8-byte tag with a hand-rolled branch-free `constant_time_tag_eq` (XOR-accumulate, no early exit, `#[inline(never)]` + `black_box`). **No new dependency** — the `subtle` crate was the only reason this was deferred, and a fixed 8-byte compare needs none. Pinned by `tag_compare_is_constant_time_shape` (proven to fail on a last-byte-skipping bug). Grade MEASURED (constant-time construction). See §2.6.
- **802.11bf SBP conformance vectors** (§5 #2) — add real conformance test vectors to the `ieee80211bf/` model **when the Wi-Fi Alliance / WBA publishes a public test plan**. Do not invent vectors before then.
- **Geometry-conditioned LoRA calibration head** (§5 #3) — built-but-unconsumed and **data-gated** on paired multi-room PCK data (ADR-152 measurement (b): data, not architecture, is the bottleneck). ACCEPTED-FUTURE.
- **Native `wlanapi.dll` FFI multi-BSSID fast path** (§5 #4) — **RESOLVED + MEASURED (Milestone-1).** The native FFI is implemented and wired live (native-first, netsh fallback). Measured on this box (Intel Wi-Fi 7 BE201 320MHz, 2026-06-13): **native 21.42 Hz vs netsh 3.84 Hz = 5.57×**, mean 5.0 BSSIDs/scan, 50/50 native scans with no handle leak. Real positive result — no fabricated 10×. See §5 #4. (Note: a prior sweep recorded 9.74 Hz on a different/older adapter; the per-adapter number varies, the ratio over netsh is the claim.)
- **Deep-CSI vital-sign model** (§5 #1) — DATA-GATED on paired PPG/ECG ground truth. No public ESP32 artifact reproduces the cited ~2–3 BPM MAE. Not on the near-term path.

---

## 9. Consequences

**Positive.** The vital-sign extractors now use the correct O(1)-eviction data structure (no latent O(n²)), cannot mis-scale a breathing estimate from a partial attention-weight slice, and cannot be silently killed by a diverging IIR filter at a pathological sample rate. The 802.11bf construction site drops-instead-of-truncates on an (already-gated) oversized count. Most importantly, the layer's existing hardening — length-gated parsers, infallible fixed-width slices, validate-on-deserialize, no-panic FSMs, fixed-argv scanning, HMAC+replay TDM, overflow-clamped geometry embeddings — is now **documented as MEASURED negative results** with file:line evidence, so a reader can verify the "already safe" claims rather than take them on faith.

**Negative / honest limits.** The §A1 perf change is **null end-to-end** at realistic window sizes — we land it for correctness, not speed, and the committed bench proves the null rather than hiding it. The research report's stated §A3 divergence trigger ("`fs` below ~4 Hz") was **physically inaccurate** (divergence needs `|r| ≥ 1` ⇒ `bw ≥ 4`, a far lower `fs`); we corrected it in the code comments and the test parameters and disclose the correction here. The strongest external SOTA candidates (deep-CSI vitals, learned calibration, native FFI scanning) are **all NO-ACTION or ACCEPTED-FUTURE** — data-gated, unmeasured, or blocked on a non-public conformance suite — and **none is presented as more than it is.** §B4 is consciously deferred. Nothing in this milestone is inflated beyond what a reverting reviewer can reproduce.
