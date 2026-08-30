//! Meeting-in-progress primitive (§3.12.1 row 5).
//!
//! Enter `meeting_in_progress = ON` when person_count ≥ 2 AND motion
//! is sustained low-amplitude (people sitting still while talking) for
//! ≥`meeting_dwell` (default 10 min).
//!
//! Exit when person_count < 2 for ≥2 min.

use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

const EXIT_DWELL: Duration = Duration::from_secs(120);

#[derive(Debug, Default, Clone)]
pub struct MeetingInProgress {
    pub active: bool,
    enter_since: Option<Duration>,
    exit_since: Option<Duration>,
}

impl MeetingInProgress {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        // Low-amplitude motion: people seated/quiet but present.
        let suitable_motion = (0.01..0.20).contains(&snap.motion);
        let enough_persons = snap.n_persons >= cfg.meeting_min_persons;

        if !self.active {
            if enough_persons && suitable_motion {
                let start = *self.enter_since.get_or_insert(snap.since_start);
                if snap.since_start.saturating_sub(start) >= cfg.meeting_dwell {
                    self.active = true;
                    self.exit_since = None;
                    return PrimitiveState::Boolean {
                        active: true,
                        changed: true,
                        reason: Reason::new(&["n_persons>=2", "motion=1-20%", "dwell>=10min"]),
                    };
                }
            } else {
                self.enter_since = None;
            }
            PrimitiveState::Idle
        } else {
            let too_few = snap.n_persons < cfg.meeting_min_persons;
            if too_few {
                let start = *self.exit_since.get_or_insert(snap.since_start);
                if snap.since_start.saturating_sub(start) >= EXIT_DWELL {
                    self.active = false;
                    self.enter_since = None;
                    self.exit_since = None;
                    return PrimitiveState::Boolean {
                        active: false,
                        changed: true,
                        reason: Reason::new(&["n_persons<2", "dwell>=2min"]),
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

    fn meeting_snap(t_secs: u64, n: u32) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence: true,
            motion: 0.05,
            n_persons: n,
            ..Default::default()
        }
    }

    #[test]
    fn fires_after_dwell_with_2_plus_people() {
        let mut p = MeetingInProgress::new();
        let _ = p.tick(&meeting_snap(100, 3), &cfg());
        let state = p.tick(&meeting_snap(100 + 600, 3), &cfg());
        match state {
            PrimitiveState::Boolean { active, .. } => assert!(active),
            other => panic!("expected on, got {:?}", other),
        }
    }

    #[test]
    fn does_not_fire_with_1_person() {
        let mut p = MeetingInProgress::new();
        for t in 100..(100 + 1200) {
            assert!(matches!(
                p.tick(&meeting_snap(t, 1), &cfg()),
                PrimitiveState::Idle
            ));
        }
        assert!(!p.active);
    }

    #[test]
    fn does_not_fire_with_high_motion() {
        let mut p = MeetingInProgress::new();
        for t in 100..(100 + 1200) {
            let mut s = meeting_snap(t, 3);
            s.motion = 0.5;
            assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
        }
        assert!(!p.active);
    }

    #[test]
    fn exits_after_2_min_of_low_count() {
        let mut p = MeetingInProgress::new();
        let _ = p.tick(&meeting_snap(100, 3), &cfg());
        let _ = p.tick(&meeting_snap(100 + 600, 3), &cfg());
        assert!(p.active);
        // Drop to 1 person.
        let _ = p.tick(&meeting_snap(100 + 600 + 1, 1), &cfg());
        // <2 min: still active.
        let state = p.tick(&meeting_snap(100 + 600 + 60, 1), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
        assert!(p.active);
        // Past 2 min: exit.
        let state2 = p.tick(&meeting_snap(100 + 600 + 130, 1), &cfg());
        match state2 {
            PrimitiveState::Boolean { active, .. } => assert!(!active),
            other => panic!("expected off, got {:?}", other),
        }
    }
}
