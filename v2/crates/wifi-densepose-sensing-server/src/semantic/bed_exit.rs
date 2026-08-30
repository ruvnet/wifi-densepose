//! Bed-exit (overnight) primitive (§3.12.1 row 8).
//!
//! Edge-triggered event: fires once when "someone sleeping" transitions
//! to "no presence in any bed-tagged zone" between 22:00 and 06:00
//! local time.
//!
//! Inputs:
//! - `sleeping` from upstream (the someone_sleeping primitive — wired
//!   into the bus output so we don't re-derive it here)
//! - `active_zones` — list of zones currently reporting presence
//! - `bed_zones` — config list of zones tagged as bed-areas
//! - `local_seconds_since_midnight` — local-time of day
//!
//! For v1 we don't have direct cross-primitive wiring, so we
//! approximate "sleeping" with: was-presence-in-bed-zone, then
//! exited-bed-zone. Refine in v2 when the bus exposes `sleeping`
//! state to other primitives.

use super::common::{in_window, PrimitiveConfig, PrimitiveState, RawSnapshot, Reason};

#[derive(Debug, Default, Clone)]
pub struct BedExit {
    in_bed: bool,
}

impl BedExit {
    pub fn new() -> Self {
        Self::default()
    }

    fn in_bed_zone(snap: &RawSnapshot) -> bool {
        !snap.bed_zones.is_empty() && snap.active_zones.iter().any(|z| snap.bed_zones.contains(z))
    }

    pub fn tick(&mut self, snap: &RawSnapshot, cfg: &PrimitiveConfig) -> PrimitiveState {
        if snap.since_start < cfg.warmup {
            return PrimitiveState::Idle;
        }
        let now_in_bed = snap.presence && Self::in_bed_zone(snap);
        let was_in_bed = self.in_bed;
        self.in_bed = now_in_bed;

        if was_in_bed && !now_in_bed {
            // Only fire during overnight window.
            let (start, end) = cfg.bed_exit_window;
            if in_window(snap.local_seconds_since_midnight, start, end) {
                return PrimitiveState::Event {
                    event_type: "bed_exit",
                    reason: Reason::new(&["left_bed_zone", "overnight_window"]),
                };
            }
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

    fn in_bed_overnight(t: u64) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(120 + t),
            presence: true,
            active_zones: vec!["bedroom".into()],
            bed_zones: vec!["bedroom".into()],
            local_seconds_since_midnight: 2 * 3600, // 02:00
            ..Default::default()
        }
    }

    fn out_of_bed_overnight(t: u64) -> RawSnapshot {
        RawSnapshot {
            since_start: Duration::from_secs(120 + t),
            presence: true,
            active_zones: vec!["hall".into()],
            bed_zones: vec!["bedroom".into()],
            local_seconds_since_midnight: 2 * 3600,
            ..Default::default()
        }
    }

    #[test]
    fn fires_on_bed_to_non_bed_overnight() {
        let mut p = BedExit::new();
        let _ = p.tick(&in_bed_overnight(10), &cfg());
        let state = p.tick(&out_of_bed_overnight(20), &cfg());
        assert!(matches!(
            state,
            PrimitiveState::Event {
                event_type: "bed_exit",
                ..
            }
        ));
    }

    #[test]
    fn does_not_fire_during_day() {
        let mut p = BedExit::new();
        let mut s_in = in_bed_overnight(10);
        s_in.local_seconds_since_midnight = 14 * 3600; // 14:00
        let _ = p.tick(&s_in, &cfg());
        let mut s_out = out_of_bed_overnight(20);
        s_out.local_seconds_since_midnight = 14 * 3600;
        let state = p.tick(&s_out, &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn does_not_fire_without_prior_in_bed() {
        let mut p = BedExit::new();
        // Person never was in bed.
        let state = p.tick(&out_of_bed_overnight(20), &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn warmup_blocks_initial_transitions() {
        let mut p = BedExit::new();
        let mut s_in = in_bed_overnight(0);
        s_in.since_start = Duration::from_secs(30);
        assert!(matches!(p.tick(&s_in, &cfg()), PrimitiveState::Idle));
    }

    #[test]
    fn does_not_fire_when_bed_zones_unconfigured() {
        let mut p = BedExit::new();
        let mut s_in = in_bed_overnight(10);
        s_in.bed_zones.clear();
        let _ = p.tick(&s_in, &cfg());
        let mut s_out = out_of_bed_overnight(20);
        s_out.bed_zones.clear();
        let state = p.tick(&s_out, &cfg());
        assert!(matches!(state, PrimitiveState::Idle));
    }

    #[test]
    fn fires_just_after_midnight_window_start() {
        let mut p = BedExit::new();
        let mut s_in = in_bed_overnight(10);
        s_in.local_seconds_since_midnight = 22 * 3600 + 5; // 22:00:05
        let _ = p.tick(&s_in, &cfg());
        let mut s_out = out_of_bed_overnight(20);
        s_out.local_seconds_since_midnight = 22 * 3600 + 10;
        let state = p.tick(&s_out, &cfg());
        assert!(matches!(state, PrimitiveState::Event { .. }));
    }
}
