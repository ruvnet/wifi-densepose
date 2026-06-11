//! Enrollment protocol — per-anchor capture with an adaptive quality gate
//! (ADR-151 Stage 2).
//!
//! Bad anchors poison small calibrated models far more than large ones, so an
//! anchor is only *accepted* when its captured statistics match what the anchor
//! is supposed to teach: a person present (or absent for `empty`), and the
//! expected stillness/motion. Failed anchors are re-prompted, not silently kept.
//!
//! Quality is measured against the ADR-135 empty-room baseline via
//! [`wifi_densepose_signal::BaselineCalibration::deviation`], whose
//! `CalibrationDeviationScore` gives per-frame amplitude z statistics
//! (presence strength).
//!
//! **Presence is the per-frame p90 of |z|, NOT the median** (ADR-151
//! presence-flatline bench, 2026-06-11): a person perturbs a *minority* of
//! subcarriers strongly, so the median of |z| over all bins floors at
//! `median(|N(0,1)|) ≈ 0.674` and never reaches presence gates — on real
//! ESP32-C6 HE20 captures the median read empty ≈ 0.40–0.42 vs person-moving
//! ≈ 0.74–0.75 (both below every gate), while the p90 separated cleanly:
//! empty ≈ 1.27–1.33 vs person-moving ≈ 2.26–2.29.
//!
//! **Motion is NOT taken from the score's `motion_flagged`** (ADR-152 finding,
//! "z-band squeeze"): that flag fires on `amplitude_z_median > 2.0` — deviation
//! from the *empty* baseline — which conflates presence strength with motion. A
//! strongly-reflecting person standing perfectly still (z > 2 on every frame)
//! would be rejected as "too much motion". Instead the recorder derives motion
//! from the frame-to-frame *change* in the deviation series (|Δz| and |Δφ|),
//! which is presence-independent: a still strong reflector has high z but a
//! flat z-series; a moving person has a jittery one. The |Δφ| term only
//! counts when the score reports `phase_usable` — an unsanitised phase
//! channel (no `PhaseSanitizer` in the path) makes |Δφ| pure noise that
//! inflates the motion rate to ~60–70 % on a still room (same bench).

use wifi_densepose_core::types::CsiFrame;
use wifi_densepose_signal::{BaselineCalibration, CalibrationDeviationScore};

use crate::anchor::{Anchor, AnchorLabel, AnchorQuality};

/// Default `empty` gate: maximum mean per-frame p90|z| for an empty room.
///
/// Re-derived from the ADR-151 presence-flatline bench (2026-06-11, real
/// ESP32-C6 HE20 node-8 captures replayed against a same-epoch baseline):
/// empty segments measured p90|z| ≈ 1.27–1.33; a person moving on the plate
/// measured ≈ 2.26–2.29. 1.6 sits above the empty band with margin while
/// rejecting the occupied band decisively. (The old gate of 1.0 was tuned to
/// the median statistic, which floored at ≈ 0.674 and could never separate.)
pub const EMPTY_MAX_Z: f32 = 1.6;

/// Default presence gate: minimum mean per-frame p90|z| to call a person
/// present. Same bench as [`EMPTY_MAX_Z`]: occupied ≈ 2.26–2.29, so 1.9
/// detects with margin while staying above the empty band (≤ 1.33). (The old
/// gate of 1.5 was unreachable by the median statistic — observed occupied
/// values were ≈ 0.74–0.75.)
pub const MIN_PRESENCE_Z: f32 = 1.9;

/// Thresholds for accepting an anchor.
#[derive(Debug, Clone, Copy)]
pub struct AnchorQualityGate {
    /// Minimum mean per-frame p90 amplitude z-score to consider a person
    /// present.
    pub min_presence_z: f32,
    /// For `empty`: maximum mean per-frame p90 z-score to consider the room
    /// truly empty.
    pub empty_max_z: f32,
    /// For "still" anchors: maximum motion-flag rate tolerated.
    pub max_still_motion: f32,
    /// For the "move" anchor: minimum motion-flag rate required.
    pub min_move_motion: f32,
    /// Minimum frames required to evaluate an anchor.
    pub min_frames: u32,
}

impl Default for AnchorQualityGate {
    fn default() -> Self {
        Self {
            min_presence_z: MIN_PRESENCE_Z,
            empty_max_z: EMPTY_MAX_Z,
            max_still_motion: 0.6,
            min_move_motion: 0.3,
            min_frames: 60,
        }
    }
}

impl AnchorQualityGate {
    /// Evaluate accumulated stats for `label`, returning the quality verdict
    /// and (on rejection) a human-readable reason.
    pub fn evaluate(
        &self,
        label: AnchorLabel,
        presence_z: f32,
        motion_rate: f32,
        frames: u32,
    ) -> (AnchorQuality, Option<String>) {
        let mut reason: Option<String> = None;

        if frames < self.min_frames {
            reason = Some(format!(
                "only {frames} frames (need ≥{}); is the ESP32 streaming?",
                self.min_frames
            ));
        } else if label.expects_presence() {
            if presence_z < self.min_presence_z {
                reason = Some(format!(
                    "no person detected (presence_z {presence_z:.2} < {:.2}) — move closer / face the sensor",
                    self.min_presence_z
                ));
            } else if label.expects_still() && motion_rate > self.max_still_motion {
                reason = Some(format!(
                    "too much motion ({:.0}% > {:.0}%) for a still anchor — hold still",
                    motion_rate * 100.0,
                    self.max_still_motion * 100.0
                ));
            } else if !label.expects_still() && motion_rate < self.min_move_motion {
                reason = Some(format!(
                    "not enough motion ({:.0}% < {:.0}%) — move a bit more",
                    motion_rate * 100.0,
                    self.min_move_motion * 100.0
                ));
            }
        } else {
            // `empty` anchor: the room must actually be empty.
            if presence_z > self.empty_max_z {
                reason = Some(format!(
                    "room not empty (presence_z {presence_z:.2} > {:.2}) — clear the room",
                    self.empty_max_z
                ));
            }
        }

        let quality = AnchorQuality {
            presence_z,
            motion_rate,
            frames,
            accepted: reason.is_none(),
        };
        (quality, reason)
    }
}

/// Frame-to-frame amplitude-z change above which a frame counts as motion.
///
/// Presence-independent by construction: a still person shifts the z *level*
/// but not its frame-to-frame delta (only noise-scale jitter survives), while
/// body movement modulates the reflected paths every frame. Sized well above
/// the delta the baseline's own noise floor produces (≲0.3σ) and well below
/// the delta even small limb movements produce (≳1σ). See ADR-152.
pub const Z_DELTA_MOTION: f32 = 0.5;

/// Frame-to-frame phase-drift change above which a frame counts as motion.
/// Same constant family as the absolute π/6 drift bound in
/// `CalibrationDeviationScore`, applied to the delta (static body phase shift
/// cancels out).
pub const PHASE_DELTA_MOTION: f32 = std::f32::consts::PI / 6.0;

/// Accumulates per-frame deviation statistics for a single anchor capture.
pub struct AnchorRecorder {
    label: AnchorLabel,
    /// Sum of per-frame p90|z| (the ADR-151 presence statistic).
    z_sum: f64,
    /// Sum of per-frame median|z| (legacy statistic, kept as a secondary
    /// diagnostic — see [`Self::presence_z_median`]).
    z_median_sum: f64,
    motion_count: u32,
    frames: u32,
    /// Previous frame's (amplitude_z_p90, phase_drift_median) for the
    /// delta-based motion measure (ADR-152 z-band-squeeze fix).
    prev: Option<(f32, f32)>,
}

impl AnchorRecorder {
    /// Start recording the given anchor.
    pub fn new(label: AnchorLabel) -> Self {
        Self {
            label,
            z_sum: 0.0,
            z_median_sum: 0.0,
            motion_count: 0,
            frames: 0,
            prev: None,
        }
    }

    /// The anchor being recorded.
    pub fn label(&self) -> AnchorLabel {
        self.label
    }

    /// Frames recorded so far.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Record a pre-computed deviation score (caller runs `baseline.deviation`).
    ///
    /// Presence accumulates the per-frame **p90** of |z| (ADR-151 — the
    /// median floors at ≈ 0.674 and cannot separate; see module docs).
    ///
    /// Motion is derived from the frame-to-frame change of the deviation
    /// series, NOT from `score.motion_flagged` — the flag conflates presence
    /// strength with motion (z-band squeeze, see module docs / ADR-152). The
    /// |Δφ| term only counts when `score.phase_usable` (unsanitised phase is
    /// noise, ADR-151 bench). The first frame of a capture is never motion
    /// (no predecessor).
    pub fn record_score(&mut self, score: &CalibrationDeviationScore) {
        let z = score.amplitude_z_p90;
        let phase = score.phase_drift_median;
        if let Some((pz, pp)) = self.prev {
            let phase_motion = score.phase_usable && (phase - pp).abs() > PHASE_DELTA_MOTION;
            if (z - pz).abs() > Z_DELTA_MOTION || phase_motion {
                self.motion_count += 1;
            }
        }
        self.prev = Some((z, phase));
        self.z_sum += z as f64;
        self.z_median_sum += score.amplitude_z_median as f64;
        self.frames += 1;
    }

    /// Convenience: record a CSI frame directly against a baseline.
    /// Frames that fail baseline geometry checks are skipped (not counted).
    pub fn record_frame(&mut self, baseline: &BaselineCalibration, frame: &CsiFrame) {
        if let Ok(score) = baseline.deviation(frame) {
            self.record_score(&score);
        }
    }

    /// Mean presence z-score over the capture (per-frame p90|z|, ADR-151).
    pub fn presence_z(&self) -> f32 {
        if self.frames == 0 {
            0.0
        } else {
            (self.z_sum / self.frames as f64) as f32
        }
    }

    /// Mean of the legacy per-frame median|z| over the capture. Diagnostic
    /// only — floors at ≈ 0.674 with a person present (ADR-151); never gate
    /// on it.
    pub fn presence_z_median(&self) -> f32 {
        if self.frames == 0 {
            0.0
        } else {
            (self.z_median_sum / self.frames as f64) as f32
        }
    }

    /// Fraction of frames flagged as motion.
    pub fn motion_rate(&self) -> f32 {
        if self.frames == 0 {
            0.0
        } else {
            self.motion_count as f32 / self.frames as f32
        }
    }

    /// Evaluate the capture against the gate and produce an `Anchor` (accepted
    /// or not) plus a rejection reason.
    pub fn finalize(
        &self,
        gate: &AnchorQualityGate,
        at_unix_s: i64,
    ) -> (Anchor, Option<String>) {
        let (quality, reason) =
            gate.evaluate(self.label, self.presence_z(), self.motion_rate(), self.frames);
        (
            Anchor {
                label: self.label,
                captured_at_unix_s: at_unix_s,
                quality,
            },
            reason,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a score the way `BaselineCalibration::deviation` actually would:
    /// `motion_flagged` is DERIVED from the median (z_med > 2.0 ⇒ flagged),
    /// never free. The old tests mocked `(z=3.0, motion=false)` — a
    /// combination the real producer can never emit, which is exactly how the
    /// z-band squeeze hid. `z` parameterises the p90 (the presence statistic);
    /// the median rides along at the real-data ratio (~0.33×, ADR-151 bench:
    /// person p90 ≈ 2.26 with median ≈ 0.75) so any code that accidentally
    /// gates on the median fails loudly.
    fn score(z: f32) -> CalibrationDeviationScore {
        let z_med = z / 3.0;
        CalibrationDeviationScore {
            amplitude_z_median: z_med,
            amplitude_z_p90: z,
            amplitude_z_max: z + 1.0,
            phase_drift_median: 0.05,
            phase_usable: true,
            motion_flagged: z_med > 2.0,
        }
    }

    /// Record a z-series and finalize against the default gate.
    fn run_series(label: AnchorLabel, zs: &[f32]) -> (Anchor, Option<String>) {
        let mut r = AnchorRecorder::new(label);
        for &z in zs {
            r.record_score(&score(z));
        }
        r.finalize(&AnchorQualityGate::default(), 100)
    }

    /// Constant z (a perfectly still capture at the given presence strength).
    fn run_still(label: AnchorLabel, z: f32, n: usize) -> (Anchor, Option<String>) {
        run_series(label, &vec![z; n])
    }

    /// Alternating z (every frame's |Δz| exceeds Z_DELTA_MOTION ⇒ all motion).
    fn run_jittery(label: AnchorLabel, z: f32, n: usize) -> (Anchor, Option<String>) {
        let zs: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { z } else { z + 2.0 * Z_DELTA_MOTION })
            .collect();
        run_series(label, &zs)
    }

    /// ADR-152 z-band-squeeze regression: a STRONGLY-reflecting still person
    /// (z = 3.0, so every frame is motion_flagged by the baseline heuristic)
    /// must still pass a still anchor — presence strength is not motion.
    #[test]
    fn still_anchor_with_strong_still_person_accepts() {
        let (a, reason) = run_still(AnchorLabel::StandStill, 3.0, 400);
        assert!(a.quality.accepted, "z-band squeeze is back: {reason:?}");
        assert!(reason.is_none());
        assert!(a.quality.motion_rate < 0.05, "flat z-series must read still");
    }

    #[test]
    fn still_anchor_rejects_when_no_presence() {
        let (a, reason) = run_still(AnchorLabel::Sit, 0.4, 400);
        assert!(!a.quality.accepted);
        assert!(reason.unwrap().contains("no person"));
    }

    #[test]
    fn still_anchor_rejects_on_motion() {
        let (a, reason) = run_jittery(AnchorLabel::LieDown, 3.0, 400);
        assert!(!a.quality.accepted);
        assert!(reason.unwrap().contains("motion"));
    }

    #[test]
    fn move_anchor_requires_motion() {
        let (still, r1) = run_still(AnchorLabel::SmallMove, 3.0, 400);
        assert!(!still.quality.accepted);
        assert!(r1.unwrap().contains("not enough motion"));
        let (moving, r2) = run_jittery(AnchorLabel::SmallMove, 3.0, 400);
        assert!(moving.quality.accepted, "reason: {r2:?}");
    }

    #[test]
    fn phase_delta_also_counts_as_motion() {
        // Constant z but a phase-drift series that swings past PHASE_DELTA_MOTION
        // every frame — motion must be detected from the phase channel alone.
        let mut r = AnchorRecorder::new(AnchorLabel::LieDown);
        for i in 0..400 {
            let mut s = score(2.5);
            s.phase_drift_median = if i % 2 == 0 { 0.0 } else { PHASE_DELTA_MOTION * 1.5 };
            r.record_score(&s);
        }
        let (a, reason) = r.finalize(&AnchorQualityGate::default(), 100);
        assert!(!a.quality.accepted);
        assert!(reason.unwrap().contains("motion"));
    }

    /// ADR-151: an UNSANITISED phase channel (`phase_usable = false`) must be
    /// excluded from motion — the same swinging phase series as above, but
    /// flagged unusable, must read still. (Bench: noise phase inflated the
    /// motion rate to ~60–70 % on a still room.)
    #[test]
    fn unusable_phase_does_not_create_motion() {
        let mut r = AnchorRecorder::new(AnchorLabel::LieDown);
        for i in 0..400 {
            let mut s = score(2.5);
            s.phase_usable = false;
            s.phase_drift_median = if i % 2 == 0 { 0.0 } else { PHASE_DELTA_MOTION * 1.5 };
            r.record_score(&s);
        }
        let (a, reason) = r.finalize(&AnchorQualityGate::default(), 100);
        assert!(a.quality.accepted, "noise phase counted as motion: {reason:?}");
        assert!(a.quality.motion_rate < 0.05);
    }

    /// ADR-151 median-floor regression: presence must gate on the per-frame
    /// p90, not the median. A real person reads p90 ≈ 2.26 with median ≈ 0.75
    /// (bench 2026-06-11); the median is below EVERY gate, so gating on it
    /// flatlines presence detection.
    #[test]
    fn presence_uses_p90_not_median_floor() {
        // score(2.26) ⇒ p90 = 2.26, median = 0.753 — the bench's occupied frame.
        let mut r = AnchorRecorder::new(AnchorLabel::StandStill);
        for _ in 0..400 {
            r.record_score(&score(2.26));
        }
        assert!((r.presence_z() - 2.26).abs() < 1e-3, "presence_z must be the p90 mean");
        assert!(
            (r.presence_z_median() - 2.26 / 3.0).abs() < 1e-3,
            "median kept as a secondary diagnostic"
        );
        let (a, reason) = r.finalize(&AnchorQualityGate::default(), 100);
        assert!(a.quality.accepted, "bench occupied level must pass presence: {reason:?}");

        // The same person fed to the EMPTY anchor must be rejected…
        let (occupied, reason) = run_still(AnchorLabel::Empty, 2.26, 400);
        assert!(!occupied.quality.accepted, "occupied room accepted as empty");
        assert!(reason.unwrap().contains("not empty"));
        // …while the bench's empty band still passes.
        let (empty, reason) = run_still(AnchorLabel::Empty, 1.33, 400);
        assert!(empty.quality.accepted, "bench empty level rejected: {reason:?}");
    }

    #[test]
    fn empty_anchor_rejects_when_occupied() {
        let (occupied, reason) = run_still(AnchorLabel::Empty, 3.0, 400);
        assert!(!occupied.quality.accepted);
        assert!(reason.unwrap().contains("not empty"));
        let (empty, _) = run_still(AnchorLabel::Empty, 0.3, 400);
        assert!(empty.quality.accepted);
    }

    #[test]
    fn too_few_frames_rejected() {
        let (a, reason) = run_still(AnchorLabel::Sit, 3.0, 10);
        assert!(!a.quality.accepted);
        assert!(reason.unwrap().contains("frames"));
    }
}
