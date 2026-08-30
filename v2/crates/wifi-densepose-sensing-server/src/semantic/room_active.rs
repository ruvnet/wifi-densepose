//! Room-active primitive (§3.12.1 row 3).
//!
//! Enter `room_active = ON` when presence is true and motion has been
//! above `room_active_motion_threshold` (default 10 %) at any point in
//! a rolling `room_active_window` (default 30 s).
//!
//! Exit when no motion above threshold for `room_active_exit_idle`
//! (default 10 min) OR presence drops false.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct RoomActive {
    pub active: bool,
    last_motion: Option<Duration>,
}

impl RoomActive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let above_thresh = snap.motion >= cfg.room_active_motion_threshold;
        if above_thresh && snap.presence {
            self.last_motion = Some(snap.since_start);
        }

        let recent_motion = matches!(
            self.last_motion,
            Some(t) if snap.since_start.saturating_sub(t) < cfg.room_active_window
        );

        if !self.active && recent_motion && snap.presence {
            self.active = true;
            return PrimitiveState::Boolean {
                active: true,
                changed: true,
                reason: Reason::new(&["motion>10%", "presence=true", "window<30s"]),
            };
        }
        if self.active {
            let idle_long = matches!(
                self.last_motion,
                Some(t) if snap.since_start.saturating_sub(t) >= cfg.room_active_exit_idle
            ) || self.last_motion.is_none();
            if !snap.presence || idle_long {
                self.active = false;
                let mut tags = Vec::new();
                if !snap.presence {
                    tags.push("presence=false");
                }
                if idle_long {
                    tags.push("idle>=10min");
                }
                return PrimitiveState::Boolean {
                    active: false,
                    changed: true,
                    reason: Reason::new(&tags),
                };
            }
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

    fn snap(t_secs: u64, motion: f64, presence: bool) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence,
            motion,
            ..Default::default()
        }
    }

    #[test]
    fn does_not_fire_during_warmup() {
        let mut p = RoomActive::new();
        let s = snap(30, 0.5, true);
        assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
    }

    #[test]
    fn fires_on_high_motion_with_presence() {
        let mut p = RoomActive::new();
        let s = snap(120, 0.4, true);
        let state = p.tick(&s, &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(active);
                assert!(changed);
            }
            other => panic!("expected on/change, got {:?}", other),
        }
    }

    #[test]
    fn does_not_fire_without_presence() {
        let mut p = RoomActive::new();
        let state = p.tick(&snap(120, 0.4, false), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn does_not_fire_below_threshold() {
        let mut p = RoomActive::new();
        let state = p.tick(&snap(120, 0.05, true), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn exits_on_presence_drop() {
        let mut p = RoomActive::new();
        let _ = p.tick(&snap(120, 0.4, true), &cfg());
        let state = p.tick(&snap(125, 0.4, false), &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(!active);
                assert!(changed);
            }
            other => panic!("expected off/change, got {:?}", other),
        }
    }

    #[test]
    fn exits_on_extended_idle() {
        let mut p = RoomActive::new();
        let _ = p.tick(&snap(120, 0.4, true), &cfg());
        // Idle below threshold for >10 min.
        let state = p.tick(&snap(120 + 600, 0.02, true), &cfg());
        match state {
            PrimitiveState::Boolean { active, .. } => assert!(!active),
            other => panic!("expected off, got {:?}", other),
        }
    }
}
