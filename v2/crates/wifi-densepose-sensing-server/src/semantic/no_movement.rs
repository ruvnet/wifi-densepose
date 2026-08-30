//! No-movement (safety check) primitive (§3.12.1 row 9).
//!
//! Enter `no_movement = ON` when `presence == true` AND motion < 0.01
//! for ≥`no_movement_dwell` (default 30 min).
//!
//! Exit on first frame with motion ≥ 0.01.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct NoMovement {
    pub active: bool,
    still_since: Option<Duration>,
}

impl NoMovement {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let still = snap.presence && snap.motion < 0.01;
        if !still {
            self.still_since = None;
            if self.active {
                self.active = false;
                return PrimitiveState::Boolean {
                    active: false,
                    changed: true,
                    reason: Reason::new(&["motion>=1%"]),
                };
            }
            return PrimitiveState::Idle;
        }
        let start = *self.still_since.get_or_insert(snap.since_start);
        let dwell = snap.since_start.saturating_sub(start);
        if !self.active && dwell >= cfg.no_movement_dwell {
            self.active = true;
            return PrimitiveState::Boolean {
                active: true,
                changed: true,
                reason: Reason::new(&["presence=true", "motion<1%", "dwell>=30min"]),
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
            motion: 0.005,
            ..Default::default()
        }
    }

    #[test]
    fn fires_after_full_dwell() {
        let mut p = NoMovement::new();
        // Establish start.
        let _ = p.tick(&still_snap(60 + 10), &cfg());
        // 30 min later — fire.
        let state = p.tick(&still_snap(60 + 10 + 30 * 60), &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(active && changed);
            }
            other => panic!("expected on/change, got {:?}", other),
        }
    }

    #[test]
    fn does_not_fire_with_motion() {
        let mut p = NoMovement::new();
        let mut s = still_snap(60 + 10);
        s.motion = 0.02;
        for t in 0..(30 * 60 + 5) {
            let mut s2 = s.clone();
            s2.since_start = Duration::from_secs(60 + 10 + t as u64);
            assert!(matches!(p.tick(&s2, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn brief_motion_resets_timer() {
        let mut p = NoMovement::new();
        let _ = p.tick(&still_snap(60 + 10), &cfg());
        // 25 min in — almost there.
        let _ = p.tick(&still_snap(60 + 10 + 25 * 60), &cfg());
        // Motion blip resets.
        let mut blip = still_snap(60 + 10 + 25 * 60 + 1);
        blip.motion = 0.05;
        let _ = p.tick(&blip, &cfg());
        // 5 min more — should NOT fire because timer reset.
        let state = p.tick(&still_snap(60 + 10 + 30 * 60 + 2), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
        assert!(!p.active);
    }

    #[test]
    fn exits_on_motion_after_active() {
        let mut p = NoMovement::new();
        let _ = p.tick(&still_snap(60 + 10), &cfg());
        let _ = p.tick(&still_snap(60 + 10 + 30 * 60), &cfg());
        assert!(p.active);
        let mut s = still_snap(60 + 10 + 30 * 60 + 1);
        s.motion = 0.10;
        let state = p.tick(&s, &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(!active && changed);
            }
            other => panic!("expected off/change, got {:?}", other),
        }
    }
}
