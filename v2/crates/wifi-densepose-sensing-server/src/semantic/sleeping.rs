//! Someone-sleeping primitive (§3.12.1 row 1).
//!
//! **Definition (v1):**
//!
//! Enter `someone_sleeping = ON` when ALL of the following hold for
//! `sleep_dwell` (default 300 s):
//! - `presence == true`
//! - `motion < 0.05` (rolling)
//! - `breathing_rate_bpm ∈ [8.0, 20.0]` (rolling, conf ≥ 0.5)
//!
//! Exit when `motion > 0.15` for ≥30 s OR presence drops false.
//!
//! Heart-rate variability check is deferred to v2 because the broadcast
//! channel doesn't yet emit HRV; v1 fires on motion + BR + presence
//! which is the minimum that detects sleep cleanly in the ADR-079
//! paired-capture validation set.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct SomeoneSleeping {
    pub active: bool,
    enter_since: Option<Duration>,
    exit_since: Option<Duration>,
}

impl SomeoneSleeping {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process one snapshot, return state change (if any).
    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let br_ok = matches!(snap.breathing_rate_bpm, Some(bpm) if (8.0..=20.0).contains(&bpm))
            && snap.vital_confidence >= 0.5;
        let motion_low = snap.motion < 0.05;
        let presence_ok = snap.presence;

        if !self.active {
            if presence_ok && motion_low && br_ok {
                let start = *self.enter_since.get_or_insert(snap.since_start);
                if snap.since_start.saturating_sub(start) >= cfg.sleep_dwell {
                    self.active = true;
                    self.exit_since = None;
                    return PrimitiveState::Boolean {
                        active: true,
                        changed: true,
                        reason: Reason::new(&[
                            "presence=true",
                            "motion<5%",
                            "br=8-20bpm",
                            "dwell>=5min",
                        ]),
                    };
                }
            } else {
                self.enter_since = None;
            }
            PrimitiveState::Idle
        } else {
            // Active — check exit conditions.
            let exiting = !presence_ok || snap.motion > 0.15;
            if exiting {
                let start = *self.exit_since.get_or_insert(snap.since_start);
                // Presence-drop is immediate; motion-spike requires 30s dwell.
                if !presence_ok || snap.since_start.saturating_sub(start) >= Duration::from_secs(30)
                {
                    self.active = false;
                    self.enter_since = None;
                    self.exit_since = None;
                    let mut tags = Vec::new();
                    if !presence_ok {
                        tags.push("presence=false");
                    }
                    if snap.motion > 0.15 {
                        tags.push("motion>15%");
                    }
                    return PrimitiveState::Boolean {
                        active: false,
                        changed: true,
                        reason: Reason::new(&tags),
                    };
                }
            } else {
                self.exit_since = None;
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

    fn sleeping_snap(t_secs: u64) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence: true,
            motion: 0.02,
            breathing_rate_bpm: Some(13.0),
            vital_confidence: 0.85,
            ..Default::default()
        }
    }

    #[test]
    fn does_not_fire_during_warmup() {
        let mut p = SomeoneSleeping::new();
        let s = sleeping_snap(30);
        assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
        assert!(!p.active);
    }

    #[test]
    fn fires_after_dwell_post_warmup() {
        let mut p = SomeoneSleeping::new();
        // Tick after warmup but before dwell — idle.
        assert!(matches!(
            p.tick(&sleeping_snap(60 + 100), &cfg()),
            PrimitiveState::Idle
        ));
        // Tick after warmup + dwell — should activate (start was at t=160).
        let state = p.tick(&sleeping_snap(60 + 100 + 300), &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(active);
                assert!(changed);
            }
            other => panic!("expected boolean on/change, got {:?}", other),
        }
        assert!(p.active);
    }

    #[test]
    fn does_not_fire_when_motion_high() {
        let mut p = SomeoneSleeping::new();
        let mut s = sleeping_snap(60 + 100);
        s.motion = 0.30;
        for t in 0..600u64 {
            let mut s2 = s.clone();
            s2.since_start = Duration::from_secs(60 + 100 + t);
            assert!(matches!(p.tick(&s2, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn does_not_fire_when_br_out_of_range() {
        let mut p = SomeoneSleeping::new();
        let mut s = sleeping_snap(60 + 100);
        s.breathing_rate_bpm = Some(30.0); // too fast
        let s2 = {
            let mut x = s.clone();
            x.since_start = Duration::from_secs(60 + 100 + 600);
            x
        };
        let _ = p.tick(&s, &cfg());
        assert!(matches!(p.tick(&s2, &cfg()), PrimitiveState::Idle));
        assert!(!p.active);
    }

    #[test]
    fn exits_on_presence_false_immediately() {
        let mut p = SomeoneSleeping::new();
        let _ = p.tick(&sleeping_snap(60 + 100), &cfg());
        let _ = p.tick(&sleeping_snap(60 + 100 + 300), &cfg());
        assert!(p.active);
        // Presence drops.
        let mut s = sleeping_snap(60 + 100 + 301);
        s.presence = false;
        let state = p.tick(&s, &cfg());
        match state {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(!active);
                assert!(changed);
            }
            other => panic!("expected boolean off/change, got {:?}", other),
        }
        assert!(!p.active);
    }

    #[test]
    fn exits_on_sustained_motion_only_after_30s() {
        let mut p = SomeoneSleeping::new();
        let _ = p.tick(&sleeping_snap(60 + 100), &cfg());
        let _ = p.tick(&sleeping_snap(60 + 100 + 300), &cfg());
        assert!(p.active);
        // Motion spikes for 10 s — too short to exit.
        let mut s = sleeping_snap(60 + 100 + 310);
        s.motion = 0.20;
        let state = p.tick(&s, &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
        assert!(p.active);
        // Motion sustained 30 s → exit.
        let mut s2 = sleeping_snap(60 + 100 + 340);
        s2.motion = 0.20;
        let state2 = p.tick(&s2, &cfg());
        match state2 {
            PrimitiveState::Boolean {
                active, changed, ..
            } => {
                assert!(!active);
                assert!(changed);
            }
            other => panic!("expected boolean off/change, got {:?}", other),
        }
        assert!(!p.active);
    }

    #[test]
    fn brief_motion_blip_does_not_exit() {
        let mut p = SomeoneSleeping::new();
        let _ = p.tick(&sleeping_snap(60 + 100), &cfg());
        let _ = p.tick(&sleeping_snap(60 + 100 + 300), &cfg());
        assert!(p.active);
        // Motion spikes briefly then returns to low.
        let mut s_spike = sleeping_snap(60 + 100 + 305);
        s_spike.motion = 0.20;
        let _ = p.tick(&s_spike, &cfg());
        // Back to low motion within 30s.
        let s_calm = sleeping_snap(60 + 100 + 315);
        let state = p.tick(&s_calm, &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
        // Still active because exit dwell was reset by calm sample.
        assert!(p.active);
    }
}
