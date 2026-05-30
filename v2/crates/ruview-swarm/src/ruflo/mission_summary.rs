//! Serializable mission summary stored in AgentDB memory after each completed mission.
use serde::{Deserialize, Serialize};
use crate::orchestrator::MissionStats;

/// Serializable summary of a completed mission stored in AgentDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionSummary {
    pub mission_profile:      String,
    pub num_drones:           usize,
    pub area_width_m:         f64,
    pub area_height_m:        f64,
    pub victims_total:        usize,
    pub victims_confirmed:    u32,
    pub cells_covered:        u32,
    pub coverage_pct:         f64,
    pub elapsed_secs:         f64,
    pub collision_events:     u32,
    pub localization_error_m: Option<f64>,
}

impl MissionSummary {
    pub fn from_stats(
        stats: &MissionStats,
        profile: &str,
        num_drones: usize,
        area_width: f64,
        area_height: f64,
        victims_total: usize,
        coverage_pct: f64,
    ) -> Self {
        Self {
            mission_profile:      profile.to_string(),
            num_drones,
            area_width_m:         area_width,
            area_height_m:        area_height,
            victims_total,
            victims_confirmed:    stats.victims_confirmed,
            cells_covered:        stats.cells_covered,
            coverage_pct,
            elapsed_secs:         stats.elapsed_secs,
            collision_events:     stats.collision_events,
            localization_error_m: None,
        }
    }

    /// Pattern description for AgentDB pattern-store — human-readable.
    pub fn to_pattern_description(&self) -> String {
        format!(
            "{} mission: {} drones over {}x{}m, {} victims confirmed in {:.1}s, {:.0}% coverage, {} collisions",
            self.mission_profile,
            self.num_drones,
            self.area_width_m as u32,
            self.area_height_m as u32,
            self.victims_confirmed,
            self.elapsed_secs,
            self.coverage_pct * 100.0,
            self.collision_events,
        )
    }

    /// Pattern type tag for AgentDB.
    pub fn pattern_type(&self) -> &str {
        match self.mission_profile.as_str() {
            "sar"        => "sar-mission",
            "inspection" => "inspection-mission",
            "mine"       => "mine-mission",
            _            => "swarm-mission",
        }
    }

    /// Confidence score (0-1) for AgentDB based on mission outcomes.
    pub fn pattern_confidence(&self) -> f32 {
        let victim_score = if self.victims_total > 0 {
            self.victims_confirmed as f32 / self.victims_total as f32
        } else {
            0.5
        };
        let coverage_score = self.coverage_pct as f32;
        let collision_penalty = (self.collision_events as f32 * 0.1).min(0.5);
        ((victim_score * 0.5 + coverage_score * 0.5) - collision_penalty).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(victims_confirmed: u32, cells_covered: u32, collision_events: u32) -> MissionStats {
        MissionStats {
            cells_covered,
            victims_confirmed,
            collision_events,
            steps: 100,
            elapsed_secs: 30.0,
        }
    }

    #[test]
    fn test_pattern_type_tags() {
        let stats = make_stats(2, 80, 0);
        let s = MissionSummary::from_stats(&stats, "sar", 4, 400.0, 400.0, 3, 0.85);
        assert_eq!(s.pattern_type(), "sar-mission");

        let s2 = MissionSummary::from_stats(&stats, "custom", 2, 200.0, 200.0, 0, 0.5);
        assert_eq!(s2.pattern_type(), "swarm-mission");
    }

    #[test]
    fn test_pattern_confidence_penalises_collisions() {
        let no_collisions = make_stats(3, 80, 0);
        let with_collisions = make_stats(3, 80, 4);
        let s_good = MissionSummary::from_stats(&no_collisions, "sar", 4, 400.0, 400.0, 3, 0.9);
        let s_bad  = MissionSummary::from_stats(&with_collisions, "sar", 4, 400.0, 400.0, 3, 0.9);
        assert!(s_good.pattern_confidence() > s_bad.pattern_confidence());
    }

    #[test]
    fn test_to_pattern_description_contains_profile() {
        let stats = make_stats(1, 50, 0);
        let s = MissionSummary::from_stats(&stats, "inspection", 2, 100.0, 100.0, 1, 0.75);
        let desc = s.to_pattern_description();
        assert!(desc.contains("inspection"), "description should include profile: {desc}");
        assert!(desc.contains("2 drones"), "description should include drone count: {desc}");
    }
}
