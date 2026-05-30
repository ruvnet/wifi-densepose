//! Pre-built demo scenarios for rapid validation without hardware.
//!
//! Each scenario bundles a [`SwarmConfig`], victim positions, and a
//! [`SyntheticCsiGenerator`] so integration tests can drive a complete
//! swarm sim-loop with one call.

use crate::{
    config::SwarmConfig,
    types::Position3D,
};
use super::synthetic_csi::SyntheticCsiGenerator;

/// A self-contained demo scenario.
pub struct DemoScenario {
    pub name: String,
    pub config: SwarmConfig,
    pub num_drones: usize,
    pub victims: Vec<Position3D>,
}

/// Aggregate results produced after running a scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub victims_found: usize,
    pub victims_total: usize,
    pub coverage_time_secs: f64,
    pub localization_error_m: f64,
    pub collision_count: u32,
}

impl DemoScenario {
    /// Standard SAR rubble-field: 3 victims in a 400 × 400 m area.
    pub fn sar_rubble_field(num_drones: usize) -> Self {
        Self {
            name: "SAR Rubble Field".into(),
            config: SwarmConfig::demo_default(),
            num_drones,
            victims: vec![
                Position3D { x: 50.0,  y: 80.0,  z: 0.0 },
                Position3D { x: 150.0, y: 200.0, z: 0.0 },
                Position3D { x: 300.0, y: 100.0, z: 0.0 },
            ],
        }
    }

    /// Open-field search: single victim, easy detection conditions.
    pub fn open_field_search(num_drones: usize) -> Self {
        Self {
            name: "Open Field Search".into(),
            config: SwarmConfig::demo_default(),
            num_drones,
            victims: vec![
                Position3D { x: 200.0, y: 150.0, z: 0.0 },
            ],
        }
    }

    /// Mine/GPS-denied: victims in a narrow corridor, low speed.
    pub fn mine_corridor(num_drones: usize) -> Self {
        let mut cfg = SwarmConfig::mine_default();
        cfg.demo = Some(crate::config::DemoConfig {
            synthetic_csi: true,
            victim_positions: vec![[30.0, 10.0, -2.0], [80.0, 15.0, -2.0]],
            wind_noise_ms: 0.1,
            csi_noise_std: 0.08,
            packet_loss_pct: 10.0,
            replay_speed: 0.5,
        });
        Self {
            name: "Mine Corridor GPS-Denied".into(),
            config: cfg,
            num_drones,
            victims: vec![
                Position3D { x: 30.0, y: 10.0, z: -2.0 },
                Position3D { x: 80.0, y: 15.0, z: -2.0 },
            ],
        }
    }

    /// Build a [`SyntheticCsiGenerator`] from this scenario's config and victims.
    pub fn make_csi_generator(&self) -> SyntheticCsiGenerator {
        let (noise_std, detection_range_m) = self.config.demo.as_ref().map(|d| {
            (d.csi_noise_std, self.config.planning.csi_scan_width_m / 2.0)
        }).unwrap_or((0.05, 14.0));

        SyntheticCsiGenerator::new(self.victims.clone(), noise_std, detection_range_m)
    }

    /// Analytic estimate of coverage time (seconds) for this scenario.
    ///
    /// Formula:  `area / (scan_strip × drones) / speed`
    ///
    /// where `scan_strip = csi_scan_width_m × (1 − lateral_overlap / 100)`.
    pub fn estimate_coverage_time_secs(&self) -> f64 {
        let p = &self.config.planning;
        let m = &self.config.mission;
        let area = m.area_width_m * m.area_height_m;
        let scan_strip = p.csi_scan_width_m * (1.0 - p.lateral_overlap_pct / 100.0);
        if scan_strip <= 0.0 || p.max_speed_ms <= 0.0 || self.num_drones == 0 {
            return f64::INFINITY;
        }
        let total_track_m = area / scan_strip;
        let per_drone_track = total_track_m / self.num_drones as f64;
        per_drone_track / p.max_speed_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sar_scenario_coverage_estimate_within_10min() {
        // 4-drone SAR swarm over 500 × 500 m at 8 m/s, 20% overlap, 28 m scan width.
        // Analytic upper bound: area / (scan_strip × drones × speed)
        // = 250_000 / (22.4 × 4 × 8) ≈ 349 s (< 600 s = 10 min battery limit).
        let scenario = DemoScenario::sar_rubble_field(4);
        let t = scenario.estimate_coverage_time_secs();
        assert!(
            t < 600.0,
            "4-drone SAR coverage estimate {t:.1} s exceeds 600 s (10 min) battery limit"
        );
        // Also verify the estimate is positive and finite.
        assert!(t > 0.0 && t.is_finite(), "coverage estimate {t} must be positive and finite");
    }

    #[test]
    fn test_open_field_single_victim() {
        let scenario = DemoScenario::open_field_search(2);
        assert_eq!(scenario.victims.len(), 1);
        assert_eq!(scenario.num_drones, 2);
    }

    #[test]
    fn test_mine_scenario_low_speed() {
        let scenario = DemoScenario::mine_corridor(2);
        assert!(
            scenario.config.planning.max_speed_ms <= 3.0,
            "mine scenario max speed should be ≤ 3 m/s, got {}",
            scenario.config.planning.max_speed_ms
        );
    }

    #[test]
    fn test_make_csi_generator_victims_match() {
        let scenario = DemoScenario::sar_rubble_field(4);
        let gen = scenario.make_csi_generator();
        assert_eq!(gen.victims.len(), scenario.victims.len());
    }
}
