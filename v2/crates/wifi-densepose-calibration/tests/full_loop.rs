//! Full-loop integration test for the ADR-151 calibration pipeline (software half
//! of the §7 validation gap): a clean empty-room **baseline → enroll → extract →
//! train → infer** loop, driven end-to-end through the crates' public API in the
//! exact order the CLI (`calibrate` → `enroll` → `train-room` → `room-watch`)
//! wires the stages.
//!
//! CSI is synthetic but physically plausible:
//! - **empty room**: stable per-subcarrier amplitudes + small complex Gaussian
//!   noise (the ADR-135 roundtrip-test fingerprint) — never motion-flagged;
//! - **person present**: a common amplitude offset (extra multipath energy),
//!   small body sway, and a constant phase shift. Presence strength is free to
//!   exceed z = 2.0 — since the ADR-152 z-band-squeeze fix, anchor motion is
//!   measured from frame-to-frame deltas, not from the absolute deviation, so
//!   a strongly-reflecting *still* person is no longer misread as "moving";
//! - **breathing**: a few-percent periodic amplitude modulation (0.125–0.3 Hz)
//!   on a subset of subcarriers — visible in the mean-amplitude scalar the CLI
//!   uses, invisible to the per-frame *median* z (so still anchors stay still);
//! - **small movement**: per-frame amplitude jitter + a phase wobble that swings
//!   past the π/6 drift threshold.
//!
//! Deterministic (xorshift32, fixed seeds), no I/O, no hardware. What remains
//! hardware-only is the on-target run with real ESP32 CSI and a live operator.

use std::f32::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_calibration::extract::Features;
use wifi_densepose_calibration::{
    AnchorFeature, AnchorLabel, AnchorQualityGate, AnchorRecorder, EnrollmentEvent,
    EnrollmentSession, MixtureOfSpecialists, NodeGeometry, SpecialistBank, SpecialistKind,
};
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::{BaselineCalibration, CalibrationConfig, CalibrationRecorder};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32 + Box-Muller) — same pattern as
// wifi-densepose-signal/tests/calibration_roundtrip.rs.
// ---------------------------------------------------------------------------

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        assert_ne!(seed, 0, "xorshift seed must be non-zero");
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Synthetic room (HT20: 52 active subcarriers @ 20 Hz)
// ---------------------------------------------------------------------------

const N_SC: usize = 52;
const FS_HZ: f32 = 20.0;
/// Complex-noise std per quadrature ⇒ amplitude noise std ≈ NOISE_STD.
const NOISE_STD: f32 = 0.01;
/// Capture length per enrollment anchor (20 s @ 20 Hz; gate needs ≥ 60).
const ANCHOR_FRAMES: usize = 400;
/// Baseline / runtime window length (30 s @ 20 Hz; recorder needs ≥ 600).
const WINDOW_FRAMES: usize = 600;

/// What the person in the room is doing (None ⇒ empty room).
#[derive(Clone, Copy, Default)]
struct Person {
    /// Common amplitude offset in units of NOISE_STD (presence strength).
    /// Anything ≥ 1.5 reads as present; values above 2.0 are explicitly
    /// exercised to guard the ADR-152 z-band-squeeze fix (presence strength
    /// must not read as motion).
    presence_z: f32,
    /// Per-frame common amplitude jitter (body sway / fidgeting), in NOISE_STD.
    sway_z: f32,
    /// Respiration rate (Hz); 0 = no modulation.
    breathing_hz: f32,
    /// Relative amplitude-modulation depth on every 4th subcarrier.
    breathing_depth: f32,
    /// Constant phase shift from the body's multipath (radians).
    phase_shift: f32,
    /// Phase-wobble amplitude (radians) at 1.5 Hz — drives the motion flag.
    phase_wobble: f32,
}

/// Deterministic CSI source for one room. Time advances one frame per call.
struct RoomSim {
    rng: Rng,
    /// Static per-subcarrier amplitude fingerprint.
    amp: Vec<f32>,
    /// Static per-subcarrier phase fingerprint.
    phase: Vec<f32>,
    /// Frame counter (continuous room clock).
    t: u64,
}

impl RoomSim {
    fn new(seed: u32) -> Self {
        // Same HT20 fingerprint as the ADR-135 roundtrip test.
        let amp = (0..N_SC)
            .map(|k| 0.3 + 0.7 * (k as f32 * PI / N_SC as f32).sin().abs())
            .collect();
        let phase = (0..N_SC)
            .map(|k| (k as f32 * 0.1).rem_euclid(2.0 * PI) - PI)
            .collect();
        Self { rng: Rng::new(seed), amp, phase, t: 0 }
    }

    /// Generate the next CSI frame for the given occupancy.
    fn frame(&mut self, person: Option<&Person>) -> CsiFrame {
        let secs = self.t as f32 / FS_HZ;
        let (offset, wobble) = match person {
            Some(p) => {
                let sway = p.sway_z * NOISE_STD * self.rng.next_normal();
                (
                    p.presence_z * NOISE_STD + sway,
                    p.phase_shift + p.phase_wobble * (2.0 * PI * 1.5 * secs).sin(),
                )
            }
            None => (0.0, 0.0),
        };

        let mut data = Array2::<Complex64>::zeros((1, N_SC));
        for k in 0..N_SC {
            let mut a = self.amp[k] + offset;
            if let Some(p) = person {
                if p.breathing_hz > 0.0 && k % 4 == 0 {
                    a *= 1.0 + p.breathing_depth * (2.0 * PI * p.breathing_hz * secs).sin();
                }
            }
            let th = self.phase[k] + wobble;
            let re = a * th.cos() + NOISE_STD * self.rng.next_normal();
            let im = a * th.sin() + NOISE_STD * self.rng.next_normal();
            data[(0, k)] = Complex64::new(re as f64, im as f64);
        }

        let mut meta =
            CsiMetadata::new(DeviceId::new("full-loop-test"), FrequencyBand::Band2_4GHz, 6);
        meta.bandwidth_mhz = 20;
        meta.antenna_config = AntennaConfig::new(1, 1);
        self.t += 1;
        CsiFrame::new(meta, data)
    }
}

/// Per-frame scalar — mean amplitude across subcarriers/streams, the same
/// carrier the CLI's `frame_scalar` feeds into `Features::from_series`.
fn frame_scalar(frame: &CsiFrame) -> f32 {
    frame.mean_amplitude() as f32
}

/// Synthetic occupancy for each guided anchor in the canonical sequence.
fn anchor_person(label: AnchorLabel) -> Option<Person> {
    let p = match label {
        AnchorLabel::Empty => return None,
        // Strong reflector at z = 3.0 — every frame exceeds the baseline's
        // absolute motion threshold (z > 2.0). Pre-ADR-152 this anchor was
        // unenrollable ("too much motion"); the delta-based gate must accept it.
        AnchorLabel::StandStill => Person {
            presence_z: 3.0, sway_z: 0.25, phase_shift: 0.10, ..Default::default()
        },
        AnchorLabel::Sit => Person {
            presence_z: 1.65, sway_z: 0.25, phase_shift: 0.08, ..Default::default()
        },
        AnchorLabel::LieDown => Person {
            presence_z: 1.6, sway_z: 0.25, phase_shift: 0.06, ..Default::default()
        },
        AnchorLabel::BreatheSlow => Person {
            presence_z: 1.7, sway_z: 0.2, breathing_hz: 0.125, breathing_depth: 0.03,
            phase_shift: 0.08, ..Default::default()
        },
        AnchorLabel::BreatheNormal => Person {
            presence_z: 1.7, sway_z: 0.2, breathing_hz: 0.25, breathing_depth: 0.03,
            phase_shift: 0.08, ..Default::default()
        },
        AnchorLabel::SmallMove => Person {
            presence_z: 1.7, sway_z: 1.0, phase_shift: 0.10, phase_wobble: 1.0,
            ..Default::default()
        },
        AnchorLabel::SleepPosture => Person {
            presence_z: 1.6, sway_z: 0.2, breathing_hz: 0.2, breathing_depth: 0.03,
            phase_shift: 0.06, ..Default::default()
        },
    };
    Some(p)
}

/// Capture one anchor exactly as the CLI's `enroll` does: per-frame deviation
/// into the `AnchorRecorder`, scalar series for feature extraction, then the
/// quality-gate verdict.
fn capture_anchor(
    sim: &mut RoomSim,
    baseline: &BaselineCalibration,
    gate: &AnchorQualityGate,
    label: AnchorLabel,
    room_id: &str,
    at_unix_s: i64,
) -> (Option<AnchorFeature>, wifi_densepose_calibration::Anchor, Option<String>) {
    let person = anchor_person(label);
    let mut recorder = AnchorRecorder::new(label);
    let mut series = Vec::with_capacity(ANCHOR_FRAMES);
    for _ in 0..ANCHOR_FRAMES {
        let frame = sim.frame(person.as_ref());
        recorder.record_frame(baseline, &frame);
        series.push(frame_scalar(&frame));
    }
    let (anchor, reason) = recorder.finalize(gate, at_unix_s);
    let feature = anchor
        .quality
        .accepted
        .then(|| AnchorFeature::from_series(room_id, label, &series, FS_HZ));
    (feature, anchor, reason)
}

/// Generate a live feature window (Stage-5 runtime input).
fn live_window(sim: &mut RoomSim, person: Option<&Person>) -> Features {
    let series: Vec<f32> = (0..WINDOW_FRAMES)
        .map(|_| frame_scalar(&sim.frame(person)))
        .collect();
    Features::from_series(&series, FS_HZ)
}

// ---------------------------------------------------------------------------
// The full loop
// ---------------------------------------------------------------------------

#[test]
fn full_loop_baseline_enroll_extract_train_infer() {
    let room_id = "living-room";
    let mut sim = RoomSim::new(42);

    // -- Stage 1: clean empty-room baseline capture (ADR-135) ----------------
    let mut recorder = CalibrationRecorder::new(CalibrationConfig::ht20());
    let mut flagged_after_warmup = 0u32;
    for i in 0..WINDOW_FRAMES {
        let frame = sim.frame(None);
        let score = recorder.record(&frame).expect("baseline record");
        // Welford stats need a short warmup before the partial z is meaningful.
        if i >= 100 && score.motion_flagged {
            flagged_after_warmup += 1;
        }
    }
    assert_eq!(recorder.frames_recorded(), WINDOW_FRAMES as u32);
    assert_eq!(
        flagged_after_warmup, 0,
        "a static empty room must never be motion-flagged after warmup"
    );
    let baseline = recorder.finalize().expect("baseline finalize");
    assert_eq!(baseline.subcarriers.len(), N_SC);
    let baseline_id = baseline.calibration_uuid().to_string();

    // A fresh empty frame deviates negligibly from its own baseline.
    let check = baseline.deviation(&sim.frame(None)).expect("deviation");
    assert!(!check.motion_flagged, "empty frame flagged: {check:?}");
    assert!(
        check.amplitude_z_median < 1.0,
        "empty frame z {} should be < 1.0",
        check.amplitude_z_median
    );

    // -- Stage 2: guided-anchor enrollment with the quality gate -------------
    let gate = AnchorQualityGate::default();
    let mut session = EnrollmentSession::new(room_id, &baseline_id, 1_700_000_000);

    // Transceiver geometry recorded at session start (ADR-152 §2.1.1): a
    // two-node layout, one tape-measured, one unknown — all fields optional.
    let geometry = vec![
        NodeGeometry::new(1, "tape-measure")
            .with_position(0.0, 0.0, 1.2)
            .with_orientation(0.0, 0.0)
            .with_distance(2, 3.5),
        NodeGeometry::unknown(2),
    ];
    session.record_geometry(geometry.clone(), 1_700_000_000);
    assert_eq!(session.geometry(), Some(geometry.as_slice()));

    let mut features: Vec<AnchorFeature> = Vec::new();

    for (i, label) in AnchorLabel::SEQUENCE.into_iter().enumerate() {
        let at = 1_700_000_000 + (i as i64 + 1) * 30;
        let (feat, anchor, reason) =
            capture_anchor(&mut sim, &baseline, &gate, label, room_id, at);
        assert!(
            anchor.quality.accepted,
            "anchor {} rejected: {} (presence_z={:.2} motion={:.0}% frames={})",
            label.as_str(),
            reason.unwrap_or_default(),
            anchor.quality.presence_z,
            anchor.quality.motion_rate * 100.0,
            anchor.quality.frames,
        );
        match label {
            AnchorLabel::Empty => assert!(
                anchor.quality.presence_z < 1.0,
                "empty room must read empty, got z {}",
                anchor.quality.presence_z
            ),
            AnchorLabel::SmallMove => assert!(
                anchor.quality.motion_rate >= 0.3,
                "small-move motion {} too low",
                anchor.quality.motion_rate
            ),
            _ => assert!(
                anchor.quality.presence_z >= 1.5,
                "{} presence_z {} below gate",
                label.as_str(),
                anchor.quality.presence_z
            ),
        }
        features.push(feat.expect("accepted anchor yields a feature"));
        session.apply(EnrollmentEvent::AnchorAccepted { anchor });
    }
    assert!(session.is_complete(), "missing anchors: {:?}", session.missing());
    assert_eq!(session.progress(), (8, 8));
    session.apply(EnrollmentEvent::Completed { at: 1_700_000_300 });

    // -- Stage 3: feature extraction sanity ----------------------------------
    assert_eq!(features.len(), 8);
    let by_label = |l: AnchorLabel| {
        features
            .iter()
            .find(|f| f.label == l)
            .unwrap_or_else(|| panic!("no feature for {}", l.as_str()))
    };
    let breathe = by_label(AnchorLabel::BreatheNormal);
    assert!(
        (breathe.features.breathing_hz - 0.25).abs() < 0.04,
        "normal breathing extracted at {} Hz, injected 0.25 Hz",
        breathe.features.breathing_hz
    );
    assert!(
        breathe.features.breathing_score > 0.25,
        "breathing score {} too weak",
        breathe.features.breathing_score
    );
    let slow = by_label(AnchorLabel::BreatheSlow);
    assert!(
        (slow.features.breathing_hz - 0.125).abs() < 0.04,
        "slow breathing extracted at {} Hz, injected 0.125 Hz",
        slow.features.breathing_hz
    );
    let empty = by_label(AnchorLabel::Empty);
    assert!(
        empty.features.variance < breathe.features.variance,
        "empty variance {} should be below occupied {}",
        empty.features.variance,
        breathe.features.variance
    );

    // -- Stage 4: train the specialist bank + JSON persistence round-trip ----
    // The bank snapshots the geometry the enrollment recorded (ADR-152 §2.1.1).
    let bank = SpecialistBank::train(room_id, &baseline_id, &features, 1_700_000_400)
        .expect("bank training")
        .with_geometry(session.geometry().map(<[_]>::to_vec).unwrap_or_default());
    assert_eq!(bank.room_id, room_id);
    assert_eq!(bank.anchor_count, 8);
    let kinds = bank.trained_kinds();
    for kind in [
        SpecialistKind::Presence,
        SpecialistKind::Posture,
        SpecialistKind::Breathing,
        SpecialistKind::Heartbeat,
        SpecialistKind::Restlessness,
        SpecialistKind::Anomaly,
    ] {
        assert!(kinds.contains(&kind), "bank missing {kind:?} (got {kinds:?})");
    }

    // Persist and reload (JSON today) — the runtime below uses the *reloaded*
    // bank, so the round-trip is proven inside the loop, not as a side check.
    let json = bank.to_json().expect("bank to_json");
    let reloaded = SpecialistBank::from_json(&json).expect("bank from_json");
    assert_eq!(reloaded.room_id, bank.room_id);
    assert_eq!(reloaded.baseline_id, bank.baseline_id);
    assert_eq!(reloaded.anchor_count, bank.anchor_count);
    assert_eq!(
        reloaded.presence.as_ref().map(|p| p.threshold),
        bank.presence.as_ref().map(|p| p.threshold),
        "presence threshold must survive persistence"
    );
    assert_eq!(
        reloaded.geometry, geometry,
        "the enrollment geometry snapshot must survive bank persistence"
    );

    // -- Stage 5: runtime inference through the mixture ----------------------
    let mix = MixtureOfSpecialists::new(reloaded);

    // Positive case: a person breathing at a KNOWN 0.30 Hz (18 BPM) — a rate
    // never used during enrollment.
    let occupied = Person {
        presence_z: 1.7,
        sway_z: 0.25,
        breathing_hz: 0.30,
        breathing_depth: 0.04,
        phase_shift: 0.08,
        ..Default::default()
    };
    let f = live_window(&mut sim, Some(&occupied));
    let state = mix.infer(&f, &baseline_id);
    assert!(!state.stale, "bank trained against this baseline must be fresh");
    assert!(!state.vetoed, "plausible occupied window must not be vetoed");
    let presence = state.presence.expect("presence specialist trained");
    assert_eq!(presence.value, 1.0, "person in the room must be detected");
    let breathing = state.breathing.expect("breathing must be reported when present");
    assert!(
        (breathing.value - 18.0).abs() <= 2.0,
        "breathing {} BPM, injected 18 BPM",
        breathing.value
    );
    assert!(state.restlessness.is_some(), "restlessness specialist trained");

    // Motionless-person case (ADR-152 "variance-only presence" regression):
    // a strong reflector standing perfectly still — variance stays at the
    // empty-room level, only the scalar MEAN shifts. The mean channel of the
    // presence specialist must still detect them.
    let motionless = Person {
        presence_z: 3.0,
        sway_z: 0.05,
        phase_shift: 0.10,
        ..Default::default()
    };
    let f_still = live_window(&mut sim, Some(&motionless));
    let state = mix.infer(&f_still, &baseline_id);
    let presence = state.presence.expect("presence specialist trained");
    assert_eq!(
        presence.value, 1.0,
        "motionless person must be detected via the mean-shift channel \
         (variance {:.2e} vs empty-level)",
        f_still.variance
    );

    // Negative case: a fresh empty-room window must NOT report presence,
    // breathing, heartbeat, or posture.
    let f_empty = live_window(&mut sim, None);
    let state = mix.infer(&f_empty, &baseline_id);
    let presence = state.presence.expect("presence specialist trained");
    assert_eq!(presence.value, 0.0, "empty room must read absent");
    assert!(state.breathing.is_none(), "no breathing in an empty room");
    assert!(state.heartbeat.is_none(), "no heartbeat in an empty room");
    assert!(state.posture.is_none(), "no posture in an empty room");

    // Honest degradation: a drifted baseline flags the bank STALE.
    let state = mix.infer(&f, "some-other-baseline");
    assert!(state.stale, "baseline drift must mark readings STALE");
}
