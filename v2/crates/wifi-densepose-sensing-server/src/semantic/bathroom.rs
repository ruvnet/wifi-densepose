//! Bathroom-occupied primitive (§3.12.1 row 6).
//!
//! `bathroom_occupied = ON` iff `presence == true` AND any zone in
//! `active_zones` is configured as a bathroom (`cfg.bathroom_zone_tag`,
//! cross-referenced against `bed_zones`/`active_zones` via the
//! `--semantic-zones-file` config).
//!
//! Per §3.12.3 — explicitly safe in privacy mode because the entity is
//! a zone-derived boolean, not biometric.

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct BathroomOccupied {
    pub active: bool,
}

impl BathroomOccupied {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let occupied = snap.presence
            && snap
                .active_zones
                .iter()
                .any(|z| z == &cfg.bathroom_zone_tag);
        if occupied != self.active {
            self.active = occupied;
            let tag = if occupied {
                "presence=true,zone=bathroom"
            } else {
                "exit-bathroom"
            };
            return PrimitiveState::Boolean {
                active: occupied,
                changed: true,
                reason: Reason::new(&[tag]),
            };
        }
        PrimitiveState::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cfg() -> PrimitiveConfig {
        PrimitiveConfig::default()
    }

    #[test]
    fn fires_when_presence_in_bathroom_zone() {
        let mut p = BathroomOccupied::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(120),
            presence: true,
            active_zones: vec!["bathroom".into()],
            ..Default::default()
        };
        let state = p.tick(&s, &cfg());
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
    fn does_not_fire_for_other_zone() {
        let mut p = BathroomOccupied::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(120),
            presence: true,
            active_zones: vec!["kitchen".into()],
            ..Default::default()
        };
        let state = p.tick(&s, &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn requires_presence_true() {
        let mut p = BathroomOccupied::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(120),
            presence: false,
            active_zones: vec!["bathroom".into()],
            ..Default::default()
        };
        assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
    }

    #[test]
    fn warmup_blocks_initial_fire() {
        let mut p = BathroomOccupied::new();
        let s = RawSnapshot {
            since_start: Duration::from_secs(30),
            presence: true,
            active_zones: vec!["bathroom".into()],
            ..Default::default()
        };
        assert!(matches!(p.tick(&s, &cfg()), PrimitiveState::Idle));
    }

    #[test]
    fn emits_off_on_zone_exit() {
        let mut p = BathroomOccupied::new();
        let s_in = RawSnapshot {
            since_start: Duration::from_secs(120),
            presence: true,
            active_zones: vec!["bathroom".into()],
            ..Default::default()
        };
        let _ = p.tick(&s_in, &cfg());
        let s_out = RawSnapshot {
            since_start: Duration::from_secs(180),
            presence: true,
            active_zones: vec!["kitchen".into()],
            ..Default::default()
        };
        let state = p.tick(&s_out, &cfg());
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
