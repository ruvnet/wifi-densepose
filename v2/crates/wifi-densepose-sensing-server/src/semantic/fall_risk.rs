//! Fall-risk-elevated primitive (§3.12.1 row 7).
//!
//! Continuous 0..100 score derived from gait instability + near-fall
//! frequency over a rolling 24 h window. Emits a Scalar state every
//! tick when active; emits a one-shot event when the score crosses
//! `fall_risk_event_threshold` (default 70).
//!
//! v1 simplification: score = clamp(100, 10 * near_falls_24h +
//! 50 * recent_motion_variance), where:
//!   - near_falls_24h: count of `fall_detected` events in the trailing
//!     24 h window (we don't expose near-falls separately in the
//!     broadcast yet, so we approximate with confirmed falls)
//!   - recent_motion_variance: variance of motion over the trailing
//!     60 s.
//!
//! v2 will use the gait-instability score directly once it lands in
//! the pose tracker (see ADR-027 §A4).

use std::collections::VecDeque;
use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

const RECENT_MOTION_WINDOW: Duration = Duration::from_secs(60);
const FALL_HISTORY_WINDOW: Duration = Duration::from_secs(24 * 3600);

#[derive(Debug, Default, Clone)]
pub struct FallRiskElevated {
    pub last_score: f64,
    /// (timestamp, motion).
    motion_history: VecDeque<(Duration, f64)>,
    /// Timestamps of fall_detected=true events.
    fall_history: VecDeque<Duration>,
    /// True iff last emit was above the configured event threshold.
    above_threshold: bool,
}

impl FallRiskElevated {
    pub fn new() -> Self {
        Self::default()
    }

    fn variance(samples: &VecDeque<(Duration, f64)>) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mean = samples.iter().map(|(_, m)| m).sum::<f64>() / samples.len() as f64;
        let v = samples.iter().map(|(_, m)| (m - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        v
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }

        // Maintain rolling motion history.
        self.motion_history
            .push_back((snap.since_start, snap.motion));
        while let Some(&(t, _)) = self.motion_history.front() {
            if snap.since_start.saturating_sub(t) > RECENT_MOTION_WINDOW {
                self.motion_history.pop_front();
            } else {
                break;
            }
        }

        // Maintain rolling fall history.
        if snap.fall_detected {
            self.fall_history.push_back(snap.since_start);
        }
        while let Some(&t) = self.fall_history.front() {
            if snap.since_start.saturating_sub(t) > FALL_HISTORY_WINDOW {
                self.fall_history.pop_front();
            } else {
                break;
            }
        }

        let near_falls = self.fall_history.len() as f64;
        let var = Self::variance(&self.motion_history);
        let score = (10.0 * near_falls + 50.0 * var).clamp(0.0, 100.0);
        self.last_score = score;

        // Event on crossing threshold upward.
        let was_above = self.above_threshold;
        self.above_threshold = score >= cfg.fall_risk_event_threshold;
        if !was_above && self.above_threshold {
            return PrimitiveState::Event {
                event_type: "fall_risk_elevated",
                reason: Reason::new(&["score>=70", "crossed_threshold"]),
            };
        }
        PrimitiveState::Scalar {
            value: score,
            reason: Reason::new(&["score_published"]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PrimitiveConfig {
        PrimitiveConfig::default()
    }

    #[test]
    fn warmup_blocks_score() {
        let mut p = FallRiskElevated::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(30),
            motion: 0.5,
            ..Default::default()
        };
        assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
    }

    #[test]
    fn emits_scalar_when_active() {
        let mut p = FallRiskElevated::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(120),
            motion: 0.10,
            ..Default::default()
        };
        let state = p.tick(&s, &cfg());
        assert!(matches!(state, PrimitiveState::Scalar { .. }));
    }

    #[test]
    fn score_grows_with_falls() {
        let mut p = FallRiskElevated::new();
        // Establish baseline with no falls.
        let _ = p.tick(
            &RawSnapshot {
                since_start: Duration::from_secs(120),
                motion: 0.05,
                ..Default::default()
            },
            &cfg(),
        );
        let base_score = p.last_score;
        // Add some falls.
        for t in 121..125 {
            let s = RawSnapshot {
                since_start: Duration::from_secs(t),
                motion: 0.05,
                fall_detected: true,
                ..Default::default()
            };
            let _ = p.tick(&s, &cfg());
        }
        // Score should be higher than baseline.
        assert!(p.last_score > base_score);
    }

    #[test]
    fn emits_event_when_crossing_threshold() {
        let mut p = FallRiskElevated::new();
        // Inject 7 falls → score ≥ 70.
        let mut last_state = PrimitiveState::Idle;
        for t in 120..127 {
            let s = RawSnapshot {
                since_start: Duration::from_secs(t),
                motion: 0.05,
                fall_detected: true,
                ..Default::default()
            };
            last_state = p.tick(&s, &cfg());
        }
        // One of those ticks must have emitted the crossing event.
        // Since we only catch the last call's return, check the score.
        assert!(p.above_threshold, "should be above threshold");
        // The crossing-event return is on the first tick that crosses.
        // Verify the type via a fresh sequence.
        let mut p2 = FallRiskElevated::new();
        let _ = p2.tick(
            &RawSnapshot {
                since_start: Duration::from_secs(120),
                motion: 0.05,
                ..Default::default()
            },
            &cfg(),
        );
        let mut saw_event = false;
        for t in 121..130 {
            let s = RawSnapshot {
                since_start: Duration::from_secs(t),
                motion: 0.05,
                fall_detected: true,
                ..Default::default()
            };
            if matches!(p2.tick(&s, &cfg()), PrimitiveState::Event { .. }) {
                saw_event = true;
                break;
            }
        }
        assert!(saw_event, "should have emitted crossing event");
        // Suppress unused warning.
        let _ = last_state;
    }

    #[test]
    fn fall_history_evicts_after_24h() {
        let mut p = FallRiskElevated::new();
        // Inject fall.
        let _ = p.tick(
            &RawSnapshot {
                since_start: Duration::from_secs(120),
                motion: 0.05,
                fall_detected: true,
                ..Default::default()
            },
            &cfg(),
        );
        // 25 hours later — the fall should evict from the window.
        let _ = p.tick(
            &RawSnapshot {
                since_start: Duration::from_secs(120 + 25 * 3600),
                motion: 0.05,
                ..Default::default()
            },
            &cfg(),
        );
        assert!(p.fall_history.is_empty(), "fall must evict after 24h");
    }
}
