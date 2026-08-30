//! Shared types used by every semantic primitive's FSM.

use std::time::Duration;

/// Single observation snapshot the bus dispatches to every primitive.
///
/// All fields are derived from the existing broadcast channel —
/// primitives never touch raw CSI. This struct is a *projection* of
/// `VitalsSnapshot` + `sensing_update` (zones) so primitives are
/// schema-stable against future changes to the wire format.
#[derive(Debug, Clone, Default)]
pub struct RawSnapshot {
    pub node_id: String,
    pub since_start: Duration,
    pub timestamp_ms: i64,
    pub presence: bool,
    pub fall_detected: bool,
    pub motion: f64, // 0.0..=1.0
    pub motion_energy: f64,
    pub breathing_rate_bpm: Option<f64>,
    pub heart_rate_bpm: Option<f64>,
    pub n_persons: u32,
    pub rssi_dbm: Option<f64>,
    pub vital_confidence: f64,
    /// Zones currently reporting presence (e.g. `["bathroom", "kitchen"]`).
    pub active_zones: Vec<String>,
    /// Bed-tagged zones derived from `--semantic-zones-file`. Optional
    /// per-deployment.
    pub bed_zones: Vec<String>,
    /// Local time-of-day in seconds since midnight (0..86400). Used by
    /// time-gated primitives (bed_exit between 22:00 and 06:00).
    pub local_seconds_since_midnight: u32,
}

/// Output of one primitive on one snapshot.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveState {
    /// Boolean state with hysteresis. Includes change flag so the bus
    /// can decide whether to publish.
    Boolean {
        active: bool,
        changed: bool,
        reason: Reason,
    },
    /// Continuous score (e.g. fall risk 0..100). Always publish.
    Scalar { value: f64, reason: Reason },
    /// One-shot event (fall, bed exit, multi-room transition).
    Event {
        event_type: &'static str,
        reason: Reason,
    },
    /// No output this tick.
    Idle,
}

/// Human-readable explanation for HA users debugging an automation.
#[derive(Debug, Clone, PartialEq)]
pub struct Reason {
    /// Short tags suitable for `json_attributes` (e.g.
    /// `["motion<5%", "br=12bpm", "presence=true"]`).
    pub tags: Vec<String>,
}

impl Reason {
    pub fn new(tags: &[&str]) -> Self {
        Self {
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn empty() -> Self {
        Self { tags: Vec::new() }
    }
}

/// Per-deployment knobs. Loaded once at startup from
/// `--semantic-thresholds-file` if supplied, otherwise from defaults
/// committed to `docs/integrations/semantic-primitives-metrics.md`.
#[derive(Debug, Clone)]
pub struct PrimitiveConfig {
    /// First N seconds after process start during which no primitive
    /// fires (sensors settling, per §3.12.4).
    pub warmup: Duration,
    /// "Someone sleeping": min uninterrupted low-motion dwell.
    pub sleep_dwell: Duration,
    /// "Possible distress": HR multiple over rolling baseline.
    pub distress_hr_multiple: f64,
    /// "Possible distress": dwell at elevated HR before firing.
    pub distress_dwell: Duration,
    /// "Room active": motion threshold (0..1) sustained for the window.
    pub room_active_motion_threshold: f64,
    pub room_active_window: Duration,
    pub room_active_exit_idle: Duration,
    /// "Elderly inactivity anomaly": multiple over rolling baseline.
    pub elderly_anomaly_multiple: f64,
    /// "Meeting in progress": min persons + min dwell.
    pub meeting_min_persons: u32,
    pub meeting_dwell: Duration,
    /// "Bathroom occupied": zone tag to match.
    pub bathroom_zone_tag: String,
    /// "Fall risk": threshold for cross event firing.
    pub fall_risk_event_threshold: f64,
    /// "Bed exit": time window during which bed exits trigger (start, end).
    pub bed_exit_window: (u32, u32), // seconds-of-day; wraps midnight
    /// "No movement (safety)": dwell.
    pub no_movement_dwell: Duration,
    /// "Multi-room transition": max gap between zone exit + new zone enter.
    pub multi_room_gap: Duration,
}

impl Default for PrimitiveConfig {
    fn default() -> Self {
        Self {
            warmup: Duration::from_secs(60),
            sleep_dwell: Duration::from_secs(300),
            distress_hr_multiple: 1.5,
            distress_dwell: Duration::from_secs(60),
            room_active_motion_threshold: 0.10,
            room_active_window: Duration::from_secs(30),
            room_active_exit_idle: Duration::from_secs(600),
            elderly_anomaly_multiple: 2.0,
            meeting_min_persons: 2,
            meeting_dwell: Duration::from_secs(600),
            bathroom_zone_tag: "bathroom".into(),
            fall_risk_event_threshold: 70.0,
            bed_exit_window: (22 * 3600, 6 * 3600), // 22:00–06:00 local
            no_movement_dwell: Duration::from_secs(30 * 60),
            multi_room_gap: Duration::from_secs(10),
        }
    }
}

/// True iff `(start, end)` describes a wrap-around window (start > end,
/// e.g. 22:00–06:00). Used to test bed-exit time gating.
pub fn in_window(now: u32, start: u32, end: u32) -> bool {
    if start <= end {
        now >= start && now < end
    } else {
        // Wraps midnight.
        now >= start || now < end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_window_simple_range() {
        assert!(in_window(3 * 3600, 1 * 3600, 5 * 3600));
        assert!(!in_window(10 * 3600, 1 * 3600, 5 * 3600));
    }

    #[test]
    fn in_window_wrap_around_midnight() {
        // 22:00–06:00.
        assert!(in_window(23 * 3600, 22 * 3600, 6 * 3600)); // late evening
        assert!(in_window(2 * 3600, 22 * 3600, 6 * 3600)); // early morning
        assert!(!in_window(12 * 3600, 22 * 3600, 6 * 3600)); // noon — outside
        assert!(in_window(0, 22 * 3600, 6 * 3600)); // midnight tick
    }

    #[test]
    fn primitive_config_defaults_match_adr() {
        let c = PrimitiveConfig::default();
        // Spot-check key thresholds match §3.12 catalog.
        assert_eq!(c.warmup, Duration::from_secs(60));
        assert_eq!(c.sleep_dwell, Duration::from_secs(300));
        assert!((c.distress_hr_multiple - 1.5).abs() < 1e-9);
        assert_eq!(c.meeting_min_persons, 2);
        assert_eq!(c.bed_exit_window, (22 * 3600, 6 * 3600));
    }

    #[test]
    fn reason_empty_has_no_tags() {
        let r = Reason::empty();
        assert!(r.tags.is_empty());
    }

    #[test]
    fn reason_new_collects_string_owned() {
        let r = Reason::new(&["motion<5%", "br=12bpm"]);
        assert_eq!(
            r.tags,
            vec!["motion<5%".to_string(), "br=12bpm".to_string()]
        );
    }
}
