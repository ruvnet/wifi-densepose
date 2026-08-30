//! Possible-distress primitive (§3.12.1 row 2).
//!
//! Enter `possible_distress = ON` when ALL of the following hold for
//! `distress_dwell` (default 60 s):
//! - sustained HR > `distress_hr_multiple` × rolling baseline (default 1.5×)
//! - motion is agitated (motion > 0.20)
//! - no fall recently
//!
//! Exit when HR returns to baseline OR motion calms below 0.10 for 30 s.
//! After exit there's a 5-min latch suppressing re-fire (refractory).
//!
//! Baseline is an exponential moving average over a long window so a
//! single high-HR sample doesn't shift the reference fast. Window is
//! parametric so deployments can tune for resident demographics.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

const REFRACTORY: Duration = Duration::from_secs(300);

/// Exponential moving average over heart-rate samples.
#[derive(Debug, Default, Clone)]
struct Ewma {
    value: Option<f64>,
    alpha: f64, // 0..1, smaller = longer memory
}

impl Ewma {
    fn new(alpha: f64) -> Self {
        Self { value: None, alpha }
    }
    fn update(&mut self, x: f64) {
        self.value = Some(match self.value {
            Some(v) => self.alpha * x + (1.0 - self.alpha) * v,
            None => x,
        });
    }
}

#[derive(Debug, Clone)]
pub struct PossibleDistress {
    pub active: bool,
    baseline: Ewma,
    enter_since: Option<Duration>,
    last_exit: Option<Duration>,
}

impl Default for PossibleDistress {
    fn default() -> Self {
        Self {
            active: false,
            baseline: Ewma::new(0.01), // ~100-sample memory at 1 Hz
            enter_since: None,
            last_exit: None,
        }
    }
}

impl PossibleDistress {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            // Still seed the baseline even in warmup so we don't fire
            // immediately after the warmup ends with a cold baseline.
            if let Some(hr) = snap.heart_rate_bpm {
                if snap.vital_confidence >= 0.5 {
                    self.baseline.update(hr);
                }
            }
            return PrimitiveState::Idle;
        }

        let hr = match snap.heart_rate_bpm {
            Some(v) if snap.vital_confidence >= 0.5 => v,
            _ => return PrimitiveState::Idle,
        };
        let baseline = match self.baseline.value {
            Some(b) if b > 0.0 => b,
            _ => {
                self.baseline.update(hr);
                return PrimitiveState::Idle;
            }
        };

        let hr_high = hr / baseline >= cfg.distress_hr_multiple;
        let agitated = snap.motion > 0.20;
        let no_fall = !snap.fall_detected;

        // Only update baseline when NOT active AND NOT in a candidate
        // distress event (low motion, HR near baseline). This keeps the
        // baseline anchored to resting HR rather than chasing elevated
        // samples — without this guard a sustained elevated HR drifts
        // the baseline up before the dwell completes.
        if !self.active && !agitated && !hr_high {
            self.baseline.update(hr);
        }

        if !self.active {
            // Refractory period after recent exit.
            if let Some(t) = self.last_exit {
                if snap.since_start.saturating_sub(t) < REFRACTORY {
                    return PrimitiveState::Idle;
                }
            }
            if hr_high && agitated && no_fall {
                let start = *self.enter_since.get_or_insert(snap.since_start);
                if snap.since_start.saturating_sub(start) >= cfg.distress_dwell {
                    self.active = true;
                    return PrimitiveState::Boolean {
                        active: true,
                        changed: true,
                        reason: Reason::new(&[
                            "hr_high>=1.5x",
                            "motion>20%",
                            "no_fall",
                            "dwell>=60s",
                        ]),
                    };
                }
            } else {
                self.enter_since = None;
            }
            PrimitiveState::Idle
        } else {
            // Active — check exit.
            let calm = snap.motion < 0.10 && hr / baseline < 1.2;
            if calm {
                self.active = false;
                self.enter_since = None;
                self.last_exit = Some(snap.since_start);
                return PrimitiveState::Boolean {
                    active: false,
                    changed: true,
                    reason: Reason::new(&["motion<10%", "hr_back_to_baseline"]),
                };
            }
            PrimitiveState::Idle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PrimitiveConfig {
        PrimitiveConfig::default()
    }

    fn snap(t_secs: u64, hr: Option<f64>, motion: f64) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence: true,
            motion,
            heart_rate_bpm: hr,
            vital_confidence: 0.8,
            ..Default::default()
        }
    }

    fn seed_baseline(p: &mut PossibleDistress, hr: f64) {
        // Warmup samples seed the EWMA baseline.
        for t in 0..60 {
            let _ = p.tick(&snap(t, Some(hr), 0.0), &cfg());
        }
    }

    #[test]
    fn does_not_fire_with_normal_hr() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        // Normal HR + low motion → no fire.
        for t in 60..200 {
            let s = snap(t, Some(72.0), 0.05);
            assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn fires_on_sustained_elevated_hr_with_motion() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        // Elevated HR (>1.5×70=105) + agitated motion, sustained 60s.
        let mut fired = false;
        for t in 60..200 {
            let s = snap(t, Some(120.0), 0.35);
            if matches!(
                p.tick(&s, &cfg()),
                PrimitiveState::Boolean { active: true, .. }
            ) {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "primitive must fire on sustained elevated HR + motion"
        );
        assert!(p.active);
    }

    #[test]
    fn does_not_fire_during_fall() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        for t in 60..200 {
            let mut s = snap(t, Some(120.0), 0.35);
            s.fall_detected = true;
            assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn exits_when_motion_calms_and_hr_normalises() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        // Trigger.
        for t in 60..200 {
            let s = snap(t, Some(120.0), 0.35);
            let _ = p.tick(&s, &cfg());
        }
        assert!(p.active);
        // Calm sample.
        let s_calm = snap(220, Some(75.0), 0.05);
        let state = p.tick(&s_calm, &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(!active && changed);
            }
            other => panic!("expected off/change, got {:?}", other),
        }
        assert!(!p.active);
    }

    #[test]
    fn refractory_blocks_immediate_refire() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        for t in 60..200 {
            let _ = p.tick(&snap(t, Some(120.0), 0.35), &cfg());
        }
        // Calm to exit.
        let _ = p.tick(&snap(220, Some(75.0), 0.05), &cfg());
        assert!(!p.active);
        // Try to re-fire 1 min after exit (refractory is 5 min).
        for t in 280..400 {
            let s = snap(t, Some(120.0), 0.35);
            assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn refire_allowed_after_refractory() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        for t in 60..200 {
            let _ = p.tick(&snap(t, Some(120.0), 0.35), &cfg());
        }
        let _ = p.tick(&snap(220, Some(75.0), 0.05), &cfg());
        // 6 min later — past refractory.
        let mut fired = false;
        for t in 600..800 {
            let s = snap(t, Some(120.0), 0.35);
            if matches!(
                p.tick(&s, &cfg()),
                PrimitiveState::Boolean { active: true, .. }
            ) {
                fired = true;
                break;
            }
        }
        assert!(fired);
    }

    #[test]
    fn baseline_does_not_track_during_active() {
        let mut p = PossibleDistress::new();
        seed_baseline(&mut p, 70.0);
        let initial = p.baseline.value.unwrap();
        for t in 60..200 {
            let _ = p.tick(&snap(t, Some(120.0), 0.35), &cfg());
        }
        assert!(p.active);
        // Many more elevated samples — baseline must not climb.
        for t in 200..400 {
            let _ = p.tick(&snap(t, Some(130.0), 0.35), &cfg());
        }
        let after = p.baseline.value.unwrap();
        // Baseline may move a little during pre-trigger window, but it
        // must not chase the 130-bpm samples during the active state.
        assert!(
            after < 100.0,
            "baseline {} drifted toward distress HR",
            after
        );
        assert!(initial < 100.0);
    }
}
