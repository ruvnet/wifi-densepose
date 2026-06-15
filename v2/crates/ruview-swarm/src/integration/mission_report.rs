//! Mission outcome report with victim confirmation details.
use serde::{Deserialize, Serialize};

/// A single confirmed victim with localization metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VictimReport {
    pub victim_id: u32,
    pub position: [f64; 3],         // [north, east, down] NED metres
    pub localization_error_m: f64,  // distance from ground-truth (sim only)
    pub uncertainty_m: f64,         // fusion uncertainty ellipse
    pub contributing_drones: Vec<u32>,
    pub fused_confidence: f32,
    pub detection_time_secs: f64,   // mission-elapsed time at confirmation
}

/// Complete mission outcome report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReport {
    pub profile: String,
    pub num_drones: usize,
    pub area_m2: f64,
    pub mission_duration_secs: f64,
    pub coverage_pct: f64,
    pub victims_total: usize,
    pub victims_confirmed: usize,
    pub detection_rate: f64,        // confirmed / total
    pub mean_localization_error_m: f64,
    pub collision_events: u32,
    pub victims: Vec<VictimReport>,
    pub sota_comparison: SotaComparison,
}

/// Comparison against the Wi2SAR published baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SotaComparison {
    pub wi2sar_localization_m: f64,    // 5.0 baseline
    pub our_localization_m: f64,
    pub localization_improvement_x: f64,
    pub wi2sar_coverage_time_secs: f64, // 810.0 for single drone over 160k m²
    pub our_coverage_time_secs: f64,
    pub beats_sota: bool,
}

impl MissionReport {
    pub fn detection_rate(&self) -> f64 {
        if self.victims_total == 0 {
            1.0
        } else {
            self.victims_confirmed as f64 / self.victims_total as f64
        }
    }

    /// Produce a human-readable summary line.
    pub fn summary(&self) -> String {
        format!(
            "{} mission: {}/{} victims confirmed ({:.0}%), mean error {:.2}m, {:.0}% coverage in {:.1}s, {} collisions — SOTA: {}",
            self.profile,
            self.victims_confirmed,
            self.victims_total,
            self.detection_rate() * 100.0,
            self.mean_localization_error_m,
            self.coverage_pct * 100.0,
            self.mission_duration_secs,
            self.collision_events,
            if self.sota_comparison.beats_sota { "BEATEN" } else { "not beaten" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sota() -> SotaComparison {
        SotaComparison {
            wi2sar_localization_m: 5.0,
            our_localization_m: 1.5,
            localization_improvement_x: 3.33,
            wi2sar_coverage_time_secs: 810.0,
            our_coverage_time_secs: 120.0,
            beats_sota: true,
        }
    }

    #[test]
    fn test_detection_rate_no_victims() {
        let report = MissionReport {
            profile: "sar".to_string(),
            num_drones: 2,
            area_m2: 160_000.0,
            mission_duration_secs: 100.0,
            coverage_pct: 0.5,
            victims_total: 0,
            victims_confirmed: 0,
            detection_rate: 1.0,
            mean_localization_error_m: 0.0,
            collision_events: 0,
            victims: vec![],
            sota_comparison: sample_sota(),
        };
        assert_eq!(report.detection_rate(), 1.0);
    }

    #[test]
    fn test_detection_rate_partial() {
        let report = MissionReport {
            profile: "sar".to_string(),
            num_drones: 4,
            area_m2: 160_000.0,
            mission_duration_secs: 100.0,
            coverage_pct: 0.8,
            victims_total: 4,
            victims_confirmed: 2,
            detection_rate: 0.5,
            mean_localization_error_m: 1.5,
            collision_events: 0,
            victims: vec![],
            sota_comparison: sample_sota(),
        };
        assert_eq!(report.detection_rate(), 0.5);
        assert!(report.summary().contains("sar mission"));
    }
}
