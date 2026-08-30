//! Multi-room transition primitive (§3.12.1 row 10).
//!
//! Edge-triggered event: when an `active_zones` set changes such that
//! one zone exited AND a different zone entered within
//! `multi_room_gap` (default 10 s), fire `multi_room_transition` with
//! the `from_zone` and `to_zone` baked into the reason tags.
//!
//! Useful for "who went from X to Y" automations (e.g. light the path,
//! announce arrival in next room).

use std::collections::HashSet;
use std::time::Duration;

use super::common::{PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct MultiRoomTransition {
    last_zones: HashSet<String>,
    last_exit: Option<(String, Duration)>,
}

impl MultiRoomTransition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            self.last_zones = snap.active_zones.iter().cloned().collect();
            return PrimitiveState::Idle;
        }
        let now: HashSet<String> = snap.active_zones.iter().cloned().collect();
        let added: Vec<&String> = now.difference(&self.last_zones).collect();
        let removed: Vec<&String> = self.last_zones.difference(&now).collect();

        let mut result = PrimitiveState::Idle;

        // Record the most recent exit.
        if let Some(exited) = removed.first() {
            self.last_exit = Some(((*exited).clone(), snap.since_start));
        }

        // Match exit with subsequent entry.
        if let (Some(entered), Some((from_zone, exit_t))) = (added.first(), self.last_exit.as_ref())
        {
            let gap = snap.since_start.saturating_sub(*exit_t);
            if gap <= cfg.multi_room_gap && from_zone.as_str() != entered.as_str() {
                let reason = Reason::new(&[
                    "zone_exit_to_entry",
                    Box::leak(format!("from={}", from_zone).into_boxed_str()),
                    Box::leak(format!("to={}", entered).into_boxed_str()),
                ]);
                result = PrimitiveState::Event {
                    event_type: "multi_room_transition",
                    reason,
                };
                // Consume the exit so we don't double-fire.
                self.last_exit = None;
            }
        }

        self.last_zones = now;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PrimitiveConfig {
        PrimitiveConfig::default()
    }

    fn zones_snap(t_secs: u64, zones: &[&str]) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(t_secs),
            presence: !zones.is_empty(),
            active_zones: zones.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn fires_when_zone_changes_quickly() {
        let mut p = MultiRoomTransition::new();
        let _ = p.tick(&zones_snap(120, &["kitchen"]), &cfg());
        // Exit kitchen.
        let _ = p.tick(&zones_snap(125, &[]), &cfg());
        // Enter living room within gap.
        let state = p.tick(&zones_snap(128, &["living"]), &cfg());
        match state {
            PrimitiveState::Event { event_type, reason } => {
                assert_eq!(event_type, "multi_room_transition");
                assert!(reason.tags.iter().any(|t| t.contains("from=kitchen")));
                assert!(reason.tags.iter().any(|t| t.contains("to=living")));
            }
            other => panic!("expected event, got {:?}", other),
        }
    }

    #[test]
    fn does_not_fire_after_long_gap() {
        let mut p = MultiRoomTransition::new();
        let _ = p.tick(&zones_snap(120, &["kitchen"]), &cfg());
        let _ = p.tick(&zones_snap(125, &[]), &cfg());
        // 15 s later — outside default 10 s gap.
        let state = p.tick(&zones_snap(140, &["living"]), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn does_not_fire_on_same_zone_re_entry() {
        let mut p = MultiRoomTransition::new();
        let _ = p.tick(&zones_snap(120, &["kitchen"]), &cfg());
        let _ = p.tick(&zones_snap(125, &[]), &cfg());
        let state = p.tick(&zones_snap(128, &["kitchen"]), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn warmup_blocks_event() {
        let mut p = MultiRoomTransition::new();
        let _ = p.tick(&zones_snap(30, &["kitchen"]), &cfg());
        let state = p.tick(&zones_snap(40, &["living"]), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn handles_simultaneous_zone_swap() {
        // Some sensing scenarios emit exit + enter in the same tick.
        let mut p = MultiRoomTransition::new();
        let _ = p.tick(&zones_snap(120, &["kitchen"]), &cfg());
        // Tick where kitchen left AND living entered simultaneously.
        let state = p.tick(&zones_snap(123, &["living"]), &cfg());
        match state {
            PrimitiveState::Event { event_type, .. } => {
                assert_eq!(event_type, "multi_room_transition");
            }
            other => panic!("expected event, got {:?}", other),
        }
    }
}
