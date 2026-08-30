//! Elderly inactivity anomaly primitive (§3.12.1 row 4).
//!
//! Enter `elderly_inactivity_anomaly = ON` when current inactivity
//! duration exceeds `elderly_anomaly_multiple` × rolling median of
//! daily idle durations (default 2×).
//!
//! v1 implements this with a simplified rolling-quantile: the longest
//! idle stretch ever seen since process start, capped by the
//! `--semantic-baseline-window-days` flag (default 14 — but we don't
//! persist across restarts in v1, so the window is effectively
//! "uptime"). Per-resident persistent baselines arrive in v2 with the
//! `SemanticState` log-replay path.
//!
//! Refractory: max 1 firing per 24 h to prevent alert spam.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

const REFRACTORY: Duration = Duration::from_secs(24 * 3600);

#[derive(Debug, Default, Clone)]
pub struct ElderlyInactivityAnomaly {
    pub active: bool,
    idle_since: Option<Duration>,
    /// Longest idle stretch observed so far. The "baseline" the multiplier
    /// is applied against. Seeded to a sensible floor so the first day
    /// doesn't fire spuriously.
    longest_idle: Duration,
    last_fire: Option<Duration>,
}

const BASELINE_FLOOR: Duration = Duration::from_secs(30 * 60); // 30 min

impl ElderlyInactivityAnomaly {
    pub fn new() -> Self {
        Self {
            longest_idle: BASELINE_FLOOR,
            ..Default::default()
        }
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let still = snap.presence && snap.motion < 0.02;
        if !still {
            // Update baseline if we just emerged from a long stretch.
            if let Some(start) = self.idle_since {
                let dur = snap.since_start.saturating_sub(start);
                if dur > self.longest_idle {
                    self.longest_idle = dur;
                }
            }
            self.idle_since = None;
            if self.active {
                self.active = false;
                return PrimitiveState::Boolean {
                    active: false,
                    changed: true,
                    reason: Reason::new(&["motion_resumed"]),
                };
            }
            return PrimitiveState::Idle;
        }

        let start = *self.idle_since.get_or_insert(snap.since_start);
        let dur = snap.since_start.saturating_sub(start);
        let threshold_secs = (self.longest_idle.as_secs_f64()) * cfg.elderly_anomaly_multiple;
        let threshold = Duration::from_secs_f64(threshold_secs);

        if !self.active && dur >= threshold {
            // Refractory.
            if let Some(t) = self.last_fire {
                if snap.since_start.saturating_sub(t) < REFRACTORY {
                    return PrimitiveState::Idle;
                }
            }
            self.active = true;
            self.last_fire = Some(snap.since_start);
            return PrimitiveState::Boolean {
                active: true,
                changed: true,
                reason: Reason::new(&["presence=true", "motion<2%", "idle>2x_baseline"]),
            };
        }
        PrimitiveState::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PrimitiveConfig {
        PrimitiveConfig::default()
    }

    fn still_snap(t_secs: u64) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence: true,
            motion: 0.01,
            ..Default::default()
        }
    }

    #[test]
    fn fires_when_idle_exceeds_2x_baseline() {
        let mut p = ElderlyInactivityAnomaly::new();
        // baseline floor is 30 min → threshold = 60 min idle.
        let _ = p.tick(&still_snap(100), &cfg());
        let state = p.tick(&still_snap(100 + 61 * 60), &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(active && changed);
            }
            other => panic!("expected on, got {:?}", other),
        }
    }

    #[test]
    fn does_not_fire_before_threshold() {
        let mut p = ElderlyInactivityAnomaly::new();
        let _ = p.tick(&still_snap(100), &cfg());
        // 50 min idle, threshold is 60.
        let state = p.tick(&still_snap(100 + 50 * 60), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn motion_clears_active_state() {
        let mut p = ElderlyInactivityAnomaly::new();
        let _ = p.tick(&still_snap(100), &cfg());
        let _ = p.tick(&still_snap(100 + 61 * 60), &cfg());
        assert!(p.active);
        // Motion.
        let mut s = still_snap(100 + 61 * 60 + 1);
        s.motion = 0.10;
        let state = p.tick(&s, &cfg());
        match state {
            PrimitiveState::Boolean { active, .. } => assert!(!active),
            other => panic!("expected off, got {:?}", other),
        }
    }

    #[test]
    fn baseline_grows_to_observed_max() {
        let mut p = ElderlyInactivityAnomaly::new();
        // Establish a 90-min idle stretch — baseline should grow.
        let _ = p.tick(&still_snap(100), &cfg());
        let _ = p.tick(&still_snap(100 + 90 * 60), &cfg());
        // p is now active. Force exit.
        let mut s = still_snap(100 + 90 * 60 + 1);
        s.motion = 0.20;
        let _ = p.tick(&s, &cfg());
        // Baseline updated.
        assert!(p.longest_idle >= Duration::from_secs(89 * 60));
    }

    #[test]
    fn refractory_prevents_repeat_alerts() {
        let mut p = ElderlyInactivityAnomaly::new();
        let _ = p.tick(&still_snap(100), &cfg());
        let _ = p.tick(&still_snap(100 + 61 * 60), &cfg());
        // Motion clears.
        let mut s = still_snap(100 + 61 * 60 + 1);
        s.motion = 0.20;
        let _ = p.tick(&s, &cfg());
        // 5 hours later, another 1h+ idle — should NOT fire (still <24h).
        let _ = p.tick(&still_snap(100 + 5 * 3600), &cfg());
        let state = p.tick(&still_snap(100 + 5 * 3600 + 70 * 60), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }
}
