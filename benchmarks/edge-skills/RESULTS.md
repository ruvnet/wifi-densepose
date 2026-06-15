# Edge-Skill Synthetic-Ground-Truth Validation — RESULTS

**Crate:** `v2/crates/wifi-densepose-wasm-edge` (workspace-EXCLUDED — build from its own dir)
**Branch:** `feat/edge-skills-synthetic-validation`
**ADR:** [ADR-160](../../docs/adr/ADR-160-edge-skill-library-honest-labeling.md)
**Date:** 2026-06-13
**Harness:** `tests/synthetic_validation.rs`

> **HONESTY BOUNDARY — read first.** Everything below is **synthetic-ground-truth
> validation**: a signal is *planted* with a known answer, the **real** detector
> is run, and detection accuracy / precision / recall / rate-error is **measured**.
> This is **NOT field accuracy.** A skill that recovers a planted sinusoid here is
> proven to do the math it claims on a *constructed* signal; it is **NOT** proven
> to work on real CSI in a real room. Skills whose detection target cannot be
> honestly planted (clinical, weapon, affect, sleep-stage, sign-language) are
> **NOT** given a number — they are listed under **DATA-GATED** with the real
> data each would require.

## Reproduce

```bash
cd v2/crates/wifi-densepose-wasm-edge   # workspace-excluded; build here
cargo test --features std --test synthetic_validation -- --nocapture
# also runs under the medical tier (med_* skills stay DATA-GATED, not validated):
cargo test --features std,medical-experimental --test synthetic_validation -- --nocapture
```

Each `MEASURED-on-synthetic | …` line printed by the harness is the source of the
table below. Numbers are deterministic (no RNG; pseudo-noise uses a fixed LCG seed).

---

## MEASURED-on-synthetic (constructible skills)

| Skill | What was planted (ground truth) | Result | Grade |
|-------|----------------------------------|--------|-------|
| **vital_trend** | BPM held N≥6 calls at each threshold band (brady/tachy-pnea <12 / >25, brady/tachy-cardia <50 / >120, apnea breathing<1.0 for ≥20) vs normal | **acc 1.000, prec 1.000, recall 1.000** (TP5 FP0 TN5 FN0) | MEASURED |
| **exo_time_crystal** | period-2 coordinated motion vs pseudo-noise + flat | **acc 1.000** (TP1 FP0 TN2 FN0) | MEASURED † |
| **exo_ghost_hunter** (hidden breathing) | phase sinusoid at lag-8 (breathing band 5–15) in an empty room vs flat phase | **acc 1.000**; planted score **1.000**, flat **0.000** | MEASURED |
| **occupancy** | 220-frame flat-amplitude calibration, then strong per-zone amplitude variance vs flat | **acc 1.000** (TP1 FP0 TN1 FN0) | MEASURED |
| **intrusion** | calibrate→arm (330 quiet frames), then per-subcarrier Δphase>1.5 + Δamp≫3σ vs quiet | **acc 1.000** (TP1 FP0 TN1 FN0) | MEASURED |
| **exo_rain_detect** | empty room, 60-frame baseline, then broadband variance (8/8 groups, ratio≫2.5) for ≥10 frames vs stable-low | **acc 1.000** (TP1 FP0 TN1 FN0) | MEASURED |
| **sig_flash_attention** | sustained high phase+amplitude in each of the 8 subcarrier groups; assert reported attention peak == planted group | **peak-localization 8/8 = 1.000** | MEASURED |
| **spt_spiking_tracker** | sparse (2-subcarrier) large phase-delta in each of the 4 zones; assert tracked zone == planted zone | **zone-localization 4/4 = 1.000** | MEASURED ‡ |
| **sig_optimal_transport** | sustained large frame-to-frame amplitude-distribution change vs stationary | **acc 1.000** (TP1 FP0 TN1 FN0) | MEASURED |
| **sig_mincut_person_match** | 2 persons with distinct stable per-region variance signatures over 40 frames | **person ids assigned, 0 id-swaps / 40 frames** | MEASURED |
| **lrn_dtw_gesture_learn** | stillness → 3 identical gesture rehearsals → enrollment | **template enrolled (templates=1)** | MEASURED (enroll) §|
| **sig_sparse_recovery** | 30 clean frames to init, then 8/32 (25%) nulled subcarriers | **dropout-detect + recovery-trigger = PASS** | MEASURED (trigger) ¶|

### Caveats on individual results

† **exo_time_crystal — honest discriminative limit.** A *pure* periodic signal
already has autocorrelation peaks at lag L **and** 2L (natural harmonics), so this
"period-doubling" detector cannot separate a true period-2 sub-harmonic from a
plain periodic signal — an earlier plant using a clean sine produced a *false
positive* (recorded during development). The construct it **can** discriminate
with known ground truth is **periodic-coordination vs aperiodic** (noise/flat),
which is what is measured (1.000). The original "sub-harmonic vs clean period"
claim is **NOT** validatable with this algorithm.

‡ **spt_spiking_tracker — plant must be sparse.** With weights init'd home=1.0 /
cross=0.25, firing all 8 inputs in a zone (8×0.25=2.0 > threshold 1.0) overdrives
*every* output neuron and the tracker collapses to zone 0 (measured 1/4 during
development). Firing only 2 inputs (home 2.0 fires, cross 0.5 silent) yields clean
4/4 zone localization. The validatable claim is *single-zone* localization.

§ **lrn_dtw_gesture_learn — enrollment validated; replay-match NOT.** The
deterministic, constructible part (stillness → 3 identical rehearsals → a template
is enrolled) is MEASURED. The DTW *replay match* (731) did **not** fire on the
identical replay in this run (`match_same=false`) — replay-recognition accuracy is
**reported, not asserted**, and is not claimed as validated.

¶ **sig_sparse_recovery — trigger validated; recovery accuracy is NEGATIVE.**
The dropout-detection + ISTA-recovery *trigger* pipeline fires correctly on >10%
planted nulls (asserted). But the **measured recovery accuracy is NOT a win**:
recovered RMSE **1.0045** vs unrecovered-null RMSE **0.9830** (**−2.2%**, i.e.
slightly *worse* than leaving the nulls at zero) on a neighbor-correlated signal.
The tridiagonal correlation model's fixed point does not equal the planted truth.
**The recovery's reconstruction quality is therefore NOT validated as effective on
synthetic data** — only its detection/trigger path is. Reported honestly; no
positive number claimed.

---

## DATA-GATED — NOT validatable on synthetic data

Planting a "seizure-like" / "weapon-like" / "happy-like" synthetic signal and
claiming the detector "works" validates **nothing real** and is exactly the
AI-slop this project fights. These skills run real DSP (per ADR-160, 0 stubs) and
keep their ADR-160 disclaimers, but get **no accuracy number** here. Each needs
the specific real, labelled data listed:

| Skill | Why not constructible on synthetic | Real data required |
|-------|------------------------------------|--------------------|
| `med_seizure_detect` | "seizure-like" motion is not a seizure; no ground-truth signature exists synthetically | Clinical EEG-/video-labelled tonic-clonic seizure CSI from instrumented patients |
| `med_sleep_apnea` | a planted breathing-pause is not clinical apnea (AHI scoring, hypopnea, desaturation) | Polysomnography-labelled (PSG) overnight CSI with scored apnea/hypopnea events |
| `med_cardiac_arrhythmia` | a synthetic HR sequence cannot encode true arrhythmia morphology | ECG-labelled CSI (AFib/PVC/etc.) from clinical monitoring |
| `med_respiratory_distress` | distress is a clinical gestalt, not a plantable rate | Clinician-labelled respiratory-distress CSI episodes |
| `med_gait_analysis` | clinical gait metrics need a reference motion-capture standard | Mocap-/force-plate-labelled gait CSI |
| `sec_weapon_detect` | a high variance ratio is RF reflectivity, **not** weapon discrimination (ADR-160 §A3 already renamed the event to `HIGH_METAL_REFLECTIVITY`) | Labelled metal-object-vs-no-object CSI with controlled object classes |
| `exo_emotion_detect` | affect is not recoverable from a planted heuristic; outputs are proxies (ADR-160 §A2) | Validated affect-labelled CSI (self-report / physiological ground truth) |
| `exo_happiness_score` | "happiness" is a gait-energy proxy, not a measured affect (ADR-160 §A2) | Validated affect/valence-labelled CSI |
| `exo_dream_stage` | sleep staging needs PSG reference (EEG/EOG/EMG) | PSG-staged overnight CSI |
| `exo_gesture_language` | coarse gesture clusters ≠ true sign language (ADR-160 §A4) | Labelled ASL letter/word CSI dataset |

> The above are **not failures** — they are the honest boundary. A smaller set of
> genuinely-measured skills plus this explicit gated list is the deliverable, per
> the prove-everything directive.

---

## Skills not in either list

The remaining edge skills (smart-building / retail / industrial occupancy-style,
the other `sig_*`/`lrn_*`/`spt_*`/`tmp_*`/`qnt_*`/`aut_*`/`ais_*` algorithm-named
modules) are **wired and exercised live** in the unified pipeline integration test
(`tests/pipeline_all.rs`, all 59 default / 64 medical skills run without panic over
300 synthetic frames) but were **not** given an individual planted-ground-truth
accuracy number here. They are honest REAL-DSP modules (ADR-160) whose physical
observable could be planted with more harness work; that is deferred, not claimed.

## Test counts (full crate suite)

```
DEFAULT  (--features std):                     631 passed, 0 failed
  (lib 504; budget 25; honest_labeling 10; pipeline_all 4; synthetic_validation 12; bench 1; vendor 75)
MEDICAL  (--features std,medical-experimental): 669 passed, 0 failed
  (lib 542; +16 same new tests; med_* stay DATA-GATED, not validated)
```

(M6 baseline was 615 / 653; the new pipeline_all (4) + synthetic_validation (12)
tests add 16 to each tier.)
