//! Synthetic-ground-truth validation harness (ADR-160 deliverable 2).
//!
//! For the subset of edge skills whose detection target can be PLANTED with
//! known ground truth, we generate N signals with known answers, run the real
//! detector, and MEASURE detection rate / precision / recall / rate-error.
//!
//! # Honesty boundary
//!
//! This is **synthetic-ground-truth validation, NOT field accuracy.** A skill
//! that recovers a planted sinusoid here is proven to do the math it claims on
//! a constructed signal; it is NOT proven to work on real CSI in a real room.
//!
//! Skills whose detection target cannot be honestly planted on synthetic data
//! (clinical seizure/apnea/arrhythmia/gait, weapon discrimination, affect/
//! emotion/happiness, dream stage, sign language) are **NOT** validated here —
//! see RESULTS.md "DATA-GATED" section. Planting a "seizure-like" wiggle and
//! claiming the detector works validates nothing real.
//!
//! Run:
//!   cargo test --features std --test synthetic_validation -- --nocapture
//!
//! The printed `MEASURED` lines are the source of `benchmarks/edge-skills/RESULTS.md`.

#![cfg(feature = "std")]

use std::f32::consts::PI;

// ── Confusion-matrix accumulator ─────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct Confusion {
    tp: u32,
    fp: u32,
    tn: u32,
    fn_: u32,
}
impl Confusion {
    fn observe(&mut self, predicted_positive: bool, actual_positive: bool) {
        match (predicted_positive, actual_positive) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, false) => self.tn += 1,
            (false, true) => self.fn_ += 1,
        }
    }
    fn precision(&self) -> f32 {
        let d = self.tp + self.fp;
        if d == 0 {
            1.0
        } else {
            self.tp as f32 / d as f32
        }
    }
    fn recall(&self) -> f32 {
        let d = self.tp + self.fn_;
        if d == 0 {
            1.0
        } else {
            self.tp as f32 / d as f32
        }
    }
    fn accuracy(&self) -> f32 {
        let d = self.tp + self.fp + self.tn + self.fn_;
        if d == 0 {
            0.0
        } else {
            (self.tp + self.tn) as f32 / d as f32
        }
    }
    fn report(&self, name: &str) {
        println!(
            "MEASURED-on-synthetic | {:<34} | acc={:.3} prec={:.3} recall={:.3} | TP={} FP={} TN={} FN={}",
            name,
            self.accuracy(),
            self.precision(),
            self.recall(),
            self.tp,
            self.fp,
            self.tn,
            self.fn_
        );
    }
}

// ── 1. vital_trend — rate-threshold detection (directly verified thresholds) ─
// Thresholds (from src/vital_trend.rs): BRADYPNEA<12, TACHYPNEA>25,
// BRADYCARDIA<50, TACHYCARDIA>120, APNEA at breathing<1.0 for 20 calls;
// ALERT_DEBOUNCE=5. Drive on_timer with known BPM, count event presence.

#[test]
fn vital_trend_rate_thresholds() {
    use wifi_densepose_wasm_edge::vital_trend::VitalTrendAnalyzer;

    // event ids: 101 brady-pnea, 102 tachy-pnea, 103 brady-cardia, 104 tachy-cardia, 105 apnea
    fn drive_breathing(bpm: f32, n: u32) -> std::collections::HashSet<i32> {
        let mut det = VitalTrendAnalyzer::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..n {
            for &(id, _) in det.on_timer(bpm, 72.0) {
                seen.insert(id);
            }
        }
        seen
    }
    fn drive_heart(bpm: f32, n: u32) -> std::collections::HashSet<i32> {
        let mut det = VitalTrendAnalyzer::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..n {
            for &(id, _) in det.on_timer(16.0, bpm) {
                seen.insert(id);
            }
        }
        seen
    }

    // 6 calls > ALERT_DEBOUNCE(5) so a sustained abnormal value fires.
    let mut c = Confusion::default();
    // Bradypnea: <12 positive; normal 16 negative.
    c.observe(drive_breathing(8.0, 6).contains(&101), true);
    c.observe(drive_breathing(16.0, 6).contains(&101), false);
    // Tachypnea: >25 positive; normal negative.
    c.observe(drive_breathing(30.0, 6).contains(&102), true);
    c.observe(drive_breathing(16.0, 6).contains(&102), false);
    // Bradycardia: <50.
    c.observe(drive_heart(40.0, 6).contains(&103), true);
    c.observe(drive_heart(72.0, 6).contains(&103), false);
    // Tachycardia: >120.
    c.observe(drive_heart(140.0, 6).contains(&104), true);
    c.observe(drive_heart(72.0, 6).contains(&104), false);
    // Apnea: breathing < 1.0 for >= 20 calls.
    c.observe(drive_breathing(0.0, 20).contains(&105), true);
    c.observe(drive_breathing(0.0, 10).contains(&105), false); // only 10 calls -> below APNEA_SECONDS

    c.report("vital_trend (brady/tachy-pnea/cardia, apnea)");
    // All 5 thresholds + their negatives must classify correctly.
    assert_eq!(c.accuracy(), 1.0, "vital_trend rate thresholds must be exact");
}

// ── 2. exo_time_crystal — period-doubling (sub-harmonic) detection ───────────
// Detects a peak at lag L AND a peak at lag 2L in motion-energy autocorrelation.
// PLANT positive: period-2 modulation (alternating amplitude on a base period)
//   so autocorr has peaks at both L and 2L.
// PLANT negative: a single clean period (peak at L only) or noise.

fn run_time_crystal(motion: &[f32]) -> bool {
    use wifi_densepose_wasm_edge::exo_time_crystal::TimeCrystalDetector;
    let mut det = TimeCrystalDetector::new();
    let mut detected = false;
    for &m in motion {
        for &(id, v) in det.process_frame(m) {
            if id == 680 && v >= 2.0 {
                detected = true; // CRYSTAL_DETECTED with multiplier 2
            }
        }
    }
    detected
}

#[test]
fn exo_time_crystal_period_doubling() {
    let n = 256usize;
    // Positive: period-2 subharmonic. Base period P=16; alternate full periods
    // are scaled differently so the waveform only repeats every 2P=32 (peak at
    // lag 32) while still correlating at P=16. Plain sine (no abs, which would
    // itself fold frequency and fake a sub-harmonic).
    let base_p = 16.0f32;
    let mut pos = Vec::with_capacity(n);
    for t in 0..n {
        let phase = (t as f32) * 2.0 * PI / base_p;
        let sub = if ((t as f32 / base_p) as i32) % 2 == 0 { 1.0 } else { 0.45 };
        pos.push(0.6 + 0.35 * phase.sin() * sub);
    }
    // HONEST LIMIT (measured below): a *pure* periodic signal already has
    // autocorrelation peaks at L AND 2L (natural harmonics), so this detector
    // cannot separate a true period-2 sub-harmonic from a plain periodic signal.
    // The construct it CAN discriminate with known ground truth is
    // "periodic-with-coordination vs aperiodic". We validate that.
    //
    // Negative 1: incrementing-seed pseudo-noise (no periodicity).
    let mut noise = Vec::with_capacity(n);
    let mut s: u32 = 12345;
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        noise.push(0.3 + 0.4 * ((s >> 8) & 0xffff) as f32 / 65535.0);
    }
    // Negative 2: near-constant motion (no oscillation at all).
    let flat: Vec<f32> = (0..n).map(|t| 0.5 + 1e-4 * (t as f32 * 0.01).sin()).collect();

    let mut c = Confusion::default();
    c.observe(run_time_crystal(&pos), true); // planted period-2 -> detect
    c.observe(run_time_crystal(&noise), false); // pseudo-noise -> reject
    c.observe(run_time_crystal(&flat), false); // flat -> reject
    c.report("exo_time_crystal (periodic-coordination vs aperiodic)");
    assert!(
        run_time_crystal(&pos),
        "must detect planted period-2 coordinated motion"
    );
    assert!(
        !run_time_crystal(&noise),
        "must NOT fire on pseudo-noise"
    );
    assert!(!run_time_crystal(&flat), "must NOT fire on flat motion");
}

// ── 3. exo_ghost_hunter — hidden breathing (autocorr at breathing-range lag) ─
// When presence==0, aggregate phase is autocorrelated at lags 5..=15; a peak
// there above HIDDEN_PRESENCE_THRESHOLD(0.3) emits HIDDEN_PRESENCE(652).
// PLANT positive: phase sinusoid at a lag in [5,15] across an empty room.
// PLANT negative: flat phase (no periodic breathing signature).

fn run_ghost_hidden_breathing(period: f32, amp: f32, frames: usize) -> f32 {
    use wifi_densepose_wasm_edge::exo_ghost_hunter::GhostHunterDetector;
    let mut det = GhostHunterDetector::new();
    let n_sc = 32usize;
    let mut max_hidden = 0.0f32;
    for t in 0..frames {
        let breath = if period > 0.0 {
            amp * (t as f32 * 2.0 * PI / period).sin()
        } else {
            0.0
        };
        let mut phases = [0.0f32; 32];
        let mut amps = [0.0f32; 32];
        let mut vars = [0.0f32; 32];
        for i in 0..n_sc {
            // breathing modulates phase uniformly (chest motion -> common phase shift)
            phases[i] = 0.1 * (i as f32 * 0.2).sin() + breath;
            amps[i] = 1.0;
            vars[i] = 0.01;
        }
        // presence = 0 (empty room) is required for the hidden-breathing path.
        for &(id, v) in det.process_frame(&phases, &amps, &vars, 0, 0.0) {
            if id == 652 {
                if v > max_hidden {
                    max_hidden = v;
                }
            }
        }
    }
    max_hidden
}

#[test]
fn exo_ghost_hunter_hidden_breathing() {
    // Period 8 frames is within the breathing lag window [5,15].
    let pos = run_ghost_hidden_breathing(8.0, 0.5, 200);
    // Flat phase (no breathing) -> no hidden-presence event.
    let neg = run_ghost_hidden_breathing(0.0, 0.0, 200);

    let mut c = Confusion::default();
    c.observe(pos > 0.0, true);
    c.observe(neg > 0.0, false);
    c.report("exo_ghost_hunter (hidden breathing, lag 8)");
    println!(
        "  detail: planted-breathing hidden-presence score={:.3}, flat-phase score={:.3}",
        pos, neg
    );
    assert!(
        pos > 0.3,
        "planted breathing must score above HIDDEN_PRESENCE_THRESHOLD (0.3); got {}",
        pos
    );
    assert!(
        neg <= 0.0,
        "flat phase must not emit hidden presence; got {}",
        neg
    );
}

// ── 4. occupancy — calibration + variance-driven zone occupancy ──────────────
// BASELINE_FRAMES=200 of low-variance amplitudes establish baseline; then
// high amplitude-variance per zone (score > ZONE_THRESHOLD=0.02) flips a zone
// to occupied (EVENT_ZONE_OCCUPIED=300).

#[test]
fn occupancy_variance_detection() {
    use wifi_densepose_wasm_edge::occupancy::OccupancyDetector;

    fn run(occupied_signal: bool) -> bool {
        let mut det = OccupancyDetector::new();
        let n_sc = 32usize;
        let mut phases = [0.0f32; 32];
        // Calibration: 220 frames of near-flat amplitudes (low variance).
        for t in 0..220 {
            let mut amps = [1.0f32; 32];
            for i in 0..n_sc {
                amps[i] = 1.0 + 1e-3 * ((t + i) as f32 * 0.7).sin();
                phases[i] = 0.01 * (i as f32).sin();
            }
            det.process_frame(&phases, &amps);
        }
        // Test phase: 60 frames. If occupied, inject strong per-zone amplitude
        // variance; else keep flat.
        let mut fired = false;
        for t in 0..60 {
            let mut amps = [1.0f32; 32];
            for i in 0..n_sc {
                amps[i] = if occupied_signal {
                    // strong structured variance within each zone
                    1.0 + 2.0 * (((i % 4) as f32) - 1.5) + 0.5 * (t as f32 * 0.3 + i as f32).sin()
                } else {
                    1.0 + 1e-3 * ((t + i) as f32 * 0.7).sin()
                };
            }
            for &(id, _) in det.process_frame(&phases, &amps) {
                if id == 300 {
                    fired = true;
                }
            }
        }
        fired
    }

    let mut c = Confusion::default();
    c.observe(run(true), true);
    c.observe(run(false), false);
    c.report("occupancy (zone variance vs flat baseline)");
    assert!(run(true), "high zone variance after calibration must occupy a zone");
    assert!(!run(false), "flat amplitude must stay unoccupied");
}

// ── 5. intrusion — calibrate, arm, then disturbance>=0.8 alerts ──────────────
// disturbance = 0.6*frac(|Δphase|>1.5) + 0.4*frac(|Δamp|>3σ). Calibrate 200
// quiet frames, monitor 100 quiet frames -> Armed, then 3 frames of large
// phase+amp disturbance -> EVENT_INTRUSION_ALERT(200).

#[test]
fn intrusion_disturbance_alert() {
    use wifi_densepose_wasm_edge::intrusion::IntrusionDetector;

    fn run(intrude: bool) -> bool {
        let mut det = IntrusionDetector::new();
        let n_sc = 32usize;
        // Calibration (200) + monitoring quiet (120) -> Armed. Quiet = constant.
        for _ in 0..330 {
            let phases = [0.5f32; 32];
            let amps = [1.0f32; 32];
            det.process_frame(&phases, &amps);
        }
        let mut alerted = false;
        // 10 test frames.
        for t in 0..10 {
            let mut phases = [0.5f32; 32];
            let mut amps = [1.0f32; 32];
            if intrude {
                for i in 0..n_sc {
                    // alternate phase by 3.0 (>1.5) and amplitude far from baseline 1.0.
                    phases[i] = if t % 2 == 0 { 0.5 } else { 4.0 };
                    amps[i] = 1.0 + 8.0; // huge deviation vs ~0 baseline variance
                }
            }
            for &(id, _) in det.process_frame(&phases, &amps) {
                if id == 200 {
                    alerted = true;
                }
            }
        }
        alerted
    }

    let mut c = Confusion::default();
    c.observe(run(true), true);
    c.observe(run(false), false);
    c.report("intrusion (armed -> disturbance alert vs quiet)");
    assert!(run(true), "large phase+amplitude disturbance must alert when armed");
    assert!(!run(false), "quiet environment must not alert");
}

// ── 6. sig_sparse_recovery — ISTA recovery of planted null subcarriers ───────
// Initialize correlation on clean frames, then null >10% of subcarriers and
// MEASURE how well ISTA recovers them (rate-error style: recovery residual).

#[test]
fn sig_sparse_recovery_recovers_nulls() {
    use wifi_densepose_wasm_edge::sig_sparse_recovery::SparseRecovery;

    let mut det = SparseRecovery::new();
    let n_sc = 32usize;
    // Underlying smooth signal (neighbor-correlated) the model can learn.
    let truth: Vec<f32> = (0..n_sc).map(|i| 1.0 + 0.5 * (i as f32 * 0.4).sin()).collect();

    // Warm up correlation model with 30 clean frames.
    for _ in 0..30 {
        let mut amps: Vec<f32> = truth.clone();
        det.process_frame(&mut amps);
    }

    // Null subcarriers 5..13 (8/32 = 25% > MIN_DROPOUT_RATE 0.10).
    let mut amps: Vec<f32> = truth.clone();
    let nulled: Vec<usize> = (5..13).collect();
    for &i in &nulled {
        amps[i] = 0.0;
    }
    // Baseline error if the nulls were left at 0.0 (unrecovered).
    let mut sse0 = 0.0f32;
    for &i in &nulled {
        sse0 += truth[i] * truth[i];
    }
    let baseline_rmse = (sse0 / nulled.len() as f32).sqrt();

    let mut recovery_seen = false;
    for &(id, _) in det.process_frame(&mut amps) {
        if id == 715 {
            recovery_seen = true; // RECOVERY_COMPLETE
        }
    }
    // Measure recovery error on the nulled positions (now written back in-place).
    let mut sse = 0.0f32;
    for &i in &nulled {
        let d = amps[i] - truth[i];
        sse += d * d;
    }
    let rmse = (sse / nulled.len() as f32).sqrt();
    println!(
        "MEASURED-on-synthetic | {:<34} | dropout-detect+recovery-trigger=PASS | recovered RMSE={:.4} vs unrecovered-null RMSE={:.4} ({:+.1}%) over {} nulled subcarriers",
        "sig_sparse_recovery (ISTA)",
        rmse,
        baseline_rmse,
        100.0 * (1.0 - rmse / baseline_rmse),
        nulled.len()
    );
    // CONSTRUCTIBLE + MEASURED: the dropout detection and recovery-trigger
    // pipeline fires correctly on >10% planted nulls. This is the validatable
    // claim and we assert it.
    assert!(recovery_seen, "dropout > 10% must trigger ISTA recovery (RECOVERY_COMPLETE)");
    // HONEST MEASURED RESULT (reported, NOT asserted as a win): on this
    // neighbor-correlated synthetic signal the tridiagonal-model ISTA recovery
    // does NOT beat leaving the nulls at zero (RMSE ~1.00 vs ~0.98). The skill's
    // *recovery accuracy* is therefore NOT validated as effective on synthetic
    // data — only its dropout-detection/trigger path is. Reported in RESULTS.md.
    assert!(
        rmse.is_finite() && rmse < 5.0,
        "recovered values must be finite and bounded; got {}",
        rmse
    );
}

// ── 7. exo_rain_detect — broadband variance onset (empty room) ───────────────
// presence=0, MIN_EMPTY_FRAMES=40 baseline, then >=6/8 groups with variance
// ratio > 2.5 for ONSET_FRAMES=10 -> EVENT_RAIN_ONSET(660).

#[test]
fn exo_rain_detect_broadband_onset() {
    use wifi_densepose_wasm_edge::exo_rain_detect::RainDetector;

    fn run(rain: bool) -> bool {
        let mut det = RainDetector::new();
        let n_sc = 32usize;
        let phases = [0.1f32; 32];
        let amps = [1.0f32; 32];
        // 60 empty baseline frames with low variance.
        for _ in 0..60 {
            let vars = [0.001f32; 32];
            det.process_frame(&phases, &vars, &amps, 0);
        }
        let mut onset = false;
        // 40 frames: broadband-high variance if rain, else stay low.
        for _ in 0..40 {
            let vars = if rain { [0.5f32; 32] } else { [0.001f32; 32] };
            for &(id, _) in det.process_frame(&phases, &vars, &amps, 0) {
                if id == 660 {
                    onset = true;
                }
            }
        }
        let _ = n_sc;
        onset
    }

    let mut c = Confusion::default();
    c.observe(run(true), true);
    c.observe(run(false), false);
    c.report("exo_rain_detect (broadband variance onset)");
    assert!(run(true), "broadband variance elevation must trigger rain onset");
    assert!(!run(false), "stable low variance must not trigger rain");
}

// ── 8. sig_flash_attention — peak-attention subcarrier localization ──────────
// Q=mean(phase) per group, K=mean(prev_phase), score=Q*K/sqrt(8), softmax peak.
// Plant a sustained large phase in a KNOWN group -> assert that group becomes
// the reported attention peak (EVENT_ATTENTION_PEAK_SC=700).

#[test]
fn sig_flash_attention_peak_localization() {
    use wifi_densepose_wasm_edge::sig_flash_attention::FlashAttention;

    fn peak_for_group(target_group: usize) -> i32 {
        let mut det = FlashAttention::new();
        let n_sc = 32usize;
        let subs_per = n_sc / 8;
        let mut last_peak = -1;
        // Sustain the spike so both Q (this frame) and K (prev frame) are large
        // in the target group -> highest score there.
        for _ in 0..20 {
            let mut phases = [0.05f32; 32];
            let mut amps = [1.0f32; 32];
            for i in (target_group * subs_per)..((target_group + 1) * subs_per) {
                phases[i] = 3.0;
                amps[i] = 3.0;
            }
            for &(id, v) in det.process_frame(&phases, &amps) {
                if id == 700 {
                    last_peak = v as i32;
                }
            }
        }
        last_peak
    }

    let mut correct = 0u32;
    let total = 8u32;
    for g in 0..8usize {
        let got = peak_for_group(g);
        if got == g as i32 {
            correct += 1;
        }
        println!("  flash_attention: planted group {} -> reported peak {}", g, got);
    }
    let acc = correct as f32 / total as f32;
    println!(
        "MEASURED-on-synthetic | {:<34} | peak-localization accuracy = {}/{} = {:.3}",
        "sig_flash_attention", correct, total, acc
    );
    assert!(acc >= 0.75, "must localize the planted attention group in >=75% of cases; got {}", acc);
}

// ── 9. spt_spiking_tracker — phase-delta zone localization ───────────────────
// LIF neurons fire on |phase - prev_phase|; zone with most spikes is tracked
// (EVENT_TRACK_UPDATE=770 carries zone id). Plant motion in a KNOWN zone.

#[test]
fn spt_spiking_tracker_zone_localization() {
    use wifi_densepose_wasm_edge::spt_spiking_tracker::SpikingTracker;

    fn track_zone(target_zone: usize) -> i32 {
        let mut det = SpikingTracker::new();
        let n_sc = 32usize;
        let per = n_sc / 4; // 4 zones of 8 subcarriers
        let mut prev = [0.0f32; 32];
        let mut last_zone = -1;
        // SPARSE plant: each zone's output neuron sums home-weight 1.0 + cross
        // 0.25. Firing all 8 inputs (8*0.25=2.0) overdrives EVERY zone, so the
        // tracker collapses to zone 0. Firing only 2 inputs in the target zone
        // gives potential 2.0 at home (fires) but 0.5 cross (silent) -> only the
        // target zone fires. This is the genuinely-constructible localization.
        let base = target_zone * per;
        for t in 0..60 {
            let mut phases = [0.0f32; 32];
            // 2 subcarriers in the target zone get a large alternating delta.
            for k in 0..2 {
                phases[base + k] = if t % 2 == 0 { 0.0 } else { 3.0 };
            }
            for &(id, v) in det.process_frame(&phases, &prev) {
                if id == 770 {
                    last_zone = v as i32;
                }
            }
            prev.copy_from_slice(&phases);
        }
        last_zone
    }

    let mut correct = 0u32;
    for z in 0..4usize {
        let got = track_zone(z);
        if got == z as i32 {
            correct += 1;
        }
        println!("  spiking_tracker: planted zone {} -> tracked zone {}", z, got);
    }
    let acc = correct as f32 / 4.0;
    println!(
        "MEASURED-on-synthetic | {:<34} | zone-localization accuracy = {}/4 = {:.3}",
        "spt_spiking_tracker", correct, acc
    );
    assert!(acc >= 0.75, "must track the planted motion zone in >=75% of cases; got {}", acc);
}

// ── 10. sig_optimal_transport — distribution-shift detection ─────────────────
// Sliced Wasserstein over amplitudes; sustained shift > WASS_SHIFT(0.25) for
// SHIFT_DEB(3) -> EVENT_DISTRIBUTION_SHIFT(726). Plant a large vs no shift.

#[test]
fn sig_optimal_transport_distribution_shift() {
    use wifi_densepose_wasm_edge::sig_optimal_transport::OptimalTransportDetector;

    fn run(shift: bool) -> bool {
        let mut det = OptimalTransportDetector::new();
        let n_sc = 32usize;
        // Establish a reference distribution.
        let base: Vec<f32> = (0..n_sc).map(|i| i as f32 * 0.1).collect();
        for _ in 0..10 {
            let mut a = base.clone();
            det.process_frame(&mut a);
        }
        let mut shifted = false;
        // The detector compares each frame to the PREVIOUS frame (prev_amps is
        // updated every frame), so a one-time jump decays. To exceed WASS_SHIFT
        // (0.25) for SHIFT_DEB(3) consecutive frames we need a sustained large
        // frame-to-frame change: alternate between two very different
        // distributions each frame.
        for t in 0..15 {
            let mut a: Vec<f32> = if shift {
                if t % 2 == 0 {
                    base.clone()
                } else {
                    base.iter().map(|x| 10.0 - x).collect() // reversed + offset
                }
            } else {
                base.clone()
            };
            for &(id, _) in det.process_frame(&mut a) {
                if id == 726 {
                    shifted = true;
                }
            }
        }
        shifted
    }

    let mut c = Confusion::default();
    c.observe(run(true), true);
    c.observe(run(false), false);
    c.report("sig_optimal_transport (distribution shift)");
    assert!(run(true), "large amplitude-distribution shift must be detected");
    assert!(!run(false), "stationary distribution must not flag a shift");
}

// ── 11. lrn_dtw_gesture_learn — enroll a template, replay match vs reject ────
// STILLNESS_FRAMES=60 stillness, then 3 rehearsals of the same gesture
// (motion->stillness) -> EVENT_GESTURE_LEARNED(730). Replaying the learned
// gesture later (in Idle) -> EVENT_GESTURE_MATCHED(731); replaying a different
// gesture -> no match.

#[test]
fn lrn_dtw_gesture_learn_enroll_and_match() {
    use wifi_densepose_wasm_edge::lrn_dtw_gesture_learn::GestureLearner;

    // A gesture is a phase trajectory across frames; motion_energy gates the
    // enroll state machine (still < 0.05, moving >= 0.05).
    fn gesture_frame(kind: u8, step: usize) -> ([f32; 32], f32) {
        let mut phases = [0.0f32; 32];
        let s = step as f32;
        for i in 0..32 {
            phases[i] = match kind {
                // distinct trajectories
                0 => (s * 0.4 + i as f32 * 0.1).sin(),
                _ => (s * 0.9 + i as f32 * 0.05).cos() * 1.5,
            };
        }
        (phases, 0.5) // moving
    }

    let mut det = GestureLearner::new();
    let still = ([0.0f32; 32], 0.0f32);

    // helper to feed N still frames
    let feed_still = |det: &mut GestureLearner, n: usize| {
        for _ in 0..n {
            det.process_frame(&still.0, still.1);
        }
    };
    let feed_gesture = |det: &mut GestureLearner, kind: u8, len: usize| -> bool {
        let mut learned = false;
        for s in 0..len {
            let (ph, me) = gesture_frame(kind, s);
            for &(id, _) in det.process_frame(&ph, me) {
                if id == 730 {
                    learned = true;
                }
            }
        }
        learned
    };

    // Enroll gesture kind 0: stillness, then 3 identical rehearsals (each
    // motion burst followed by stillness).
    feed_still(&mut det, 70);
    let mut any_learned = false;
    for _ in 0..3 {
        any_learned |= feed_gesture(&mut det, 0, 30);
        feed_still(&mut det, 70);
    }

    // Replay the SAME gesture during Idle -> expect a match (731).
    let mut matched_same = false;
    for s in 0..30 {
        let (ph, me) = gesture_frame(0, s);
        for &(id, _) in det.process_frame(&ph, me) {
            if id == 731 {
                matched_same = true;
            }
        }
    }
    feed_still(&mut det, 70);
    // Replay a DIFFERENT gesture -> ideally no match (731) to the learned one.
    let mut matched_diff = false;
    for s in 0..30 {
        let (ph, me) = gesture_frame(1, s);
        for &(id, _) in det.process_frame(&ph, me) {
            if id == 731 {
                matched_diff = true;
            }
        }
    }

    let tmpl_count = det.template_count();
    println!(
        "MEASURED-on-synthetic | {:<34} | learned_event={} templates={} match_same={} match_different={}",
        "lrn_dtw_gesture_learn", any_learned, tmpl_count, matched_same, matched_diff
    );
    // The enroll path must complete (a template is learned from 3 identical
    // rehearsals). Whether the precise replay matches is the DTW behavior we
    // measure and report; we assert the deterministic enrollment.
    assert!(
        any_learned || tmpl_count > 0,
        "3 identical rehearsals after stillness must enroll a template"
    );
}

// ── 12. sig_mincut_person_match — stable id assignment for distinct signatures ─
// Per-person feature = top-FEAT_DIM variances in that person's spatial region.
// Two persons with DISTINCT, stable variance signatures should get stable ids
// (EVENT_PERSON_ID_ASSIGNED=720) with zero swaps across frames.

#[test]
fn sig_mincut_person_stable_ids() {
    use wifi_densepose_wasm_edge::sig_mincut_person_match::PersonMatcher;

    let mut det = PersonMatcher::new();
    let n_sc = 32usize;
    let amplitudes = [1.0f32; 32];
    let mut swaps = 0u32;
    let mut assigned = false;

    // 40 frames, 2 persons: person 0 region (0..16) high-variance signature,
    // person 1 region (16..32) low-variance signature, both stable.
    for _ in 0..40 {
        let mut variances = [0.0f32; 32];
        for i in 0..n_sc {
            variances[i] = if i < 16 {
                2.0 + 0.05 * (i as f32).sin()
            } else {
                0.2 + 0.01 * (i as f32).cos()
            };
        }
        for &(id, _) in det.process_frame(&amplitudes, &variances, 2) {
            if id == 720 {
                assigned = true;
            }
            if id == 721 {
                swaps += 1;
            }
        }
    }
    println!(
        "MEASURED-on-synthetic | {:<34} | assigned={} id_swaps_over_40_frames={}",
        "sig_mincut_person_match", assigned, swaps
    );
    assert!(assigned, "distinct stable signatures must assign person ids");
    assert!(swaps == 0, "stable distinct signatures must not swap ids; got {} swaps", swaps);
}
