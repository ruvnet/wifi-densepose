//! TOML-based swarm configuration with mission profiles.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub swarm: SwarmParams,
    pub formation: FormationConfig,
    pub planning: PlanningConfig,
    pub security: SecurityConfig,
    pub mission: MissionConfig,
    pub demo: Option<DemoConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmParams {
    pub max_agents: usize,
    pub cluster_size: usize,
    pub raft_election_timeout_ms: u64,
    pub raft_heartbeat_ms: u64,
    pub gossip_fanout: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationConfig {
    /// "virtual_structure" | "leader_follower" | "reynolds"
    pub mode: String,
    pub min_separation_m: f64,
    pub grid_spacing_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    pub flight_altitude_m: f64,
    pub max_speed_ms: f64,
    /// Wi2SAR validated scan footprint width.
    pub csi_scan_width_m: f64,
    pub lateral_overlap_pct: f64,
    /// P(victim) threshold to trigger Phase 3 convergence.
    pub convergence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub mavlink_signing: bool,
    pub uwb_antispoofing: bool,
    pub uwb_tolerance_m: f64,
    pub geofence_hard_margin_m: f64,
    pub geofence_soft_margin_m: f64,
    /// Remote ID broadcast rate in Hz (FAA/EU requirement: ≥ 1 Hz).
    pub remote_id_broadcast_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionConfig {
    /// "sar" | "inspection" | "agriculture" | "mine" | "relay"
    pub profile: String,
    pub area_width_m: f64,
    pub area_height_m: f64,
    pub grid_resolution_m: f64,
    pub max_flight_time_mins: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    pub synthetic_csi: bool,
    /// Victim positions in NED [x, y, z].
    pub victim_positions: Vec<[f64; 3]>,
    pub wind_noise_ms: f64,
    pub csi_noise_std: f64,
    pub packet_loss_pct: f64,
    pub replay_speed: f64,
}

impl SwarmConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn sar_default() -> Self {
        Self {
            swarm: SwarmParams {
                max_agents: 12,
                cluster_size: 4,
                raft_election_timeout_ms: 300,
                raft_heartbeat_ms: 100,
                gossip_fanout: 3,
            },
            formation: FormationConfig {
                mode: "virtual_structure".into(),
                min_separation_m: 5.0,
                grid_spacing_m: 20.0,
            },
            planning: PlanningConfig {
                flight_altitude_m: 30.0,
                max_speed_ms: 8.0,
                csi_scan_width_m: 28.0,
                lateral_overlap_pct: 20.0,
                convergence_threshold: 0.75,
            },
            security: SecurityConfig {
                mavlink_signing: true,
                uwb_antispoofing: true,
                uwb_tolerance_m: 2.0,
                geofence_hard_margin_m: 20.0,
                geofence_soft_margin_m: 50.0,
                remote_id_broadcast_hz: 1.0,
            },
            mission: MissionConfig {
                profile: "sar".into(),
                area_width_m: 500.0,
                area_height_m: 500.0,
                grid_resolution_m: 5.0,
                max_flight_time_mins: 25.0,
            },
            demo: None,
        }
    }

    pub fn inspection_default() -> Self {
        let mut cfg = Self::sar_default();
        cfg.mission.profile = "inspection".into();
        cfg.planning.flight_altitude_m = 15.0;
        cfg.planning.max_speed_ms = 4.0;
        cfg.formation.mode = "leader_follower".into();
        cfg
    }

    pub fn agriculture_default() -> Self {
        let mut cfg = Self::sar_default();
        cfg.mission.profile = "agriculture".into();
        cfg.planning.flight_altitude_m = 10.0;
        cfg.planning.max_speed_ms = 6.0;
        cfg.planning.csi_scan_width_m = 15.0;
        cfg.formation.mode = "virtual_structure".into();
        cfg.formation.grid_spacing_m = 12.0;
        cfg
    }

    pub fn mine_default() -> Self {
        let mut cfg = Self::sar_default();
        cfg.mission.profile = "mine".into();
        cfg.planning.flight_altitude_m = 5.0;
        cfg.planning.max_speed_ms = 2.0;
        cfg.security.uwb_antispoofing = true; // GPS-denied: UWB only
        cfg
    }

    /// Wi2SAR reference configuration (400×400 m, 8 m/s, 4 drones) for ADR-148 SOTA benchmark.
    /// Produces 223 s coverage estimate — below the 240 s (4-min) SOTA target.
    /// Source: Wi2SAR (arxiv 2604.09115): single drone, 160,000 m², 13.5 min.
    pub fn wi2sar_reference() -> Self {
        let mut cfg = Self::sar_default();
        cfg.mission.area_width_m = 400.0;
        cfg.mission.area_height_m = 400.0;
        cfg.planning.max_speed_ms = 8.0;
        cfg.planning.csi_scan_width_m = 28.0;
        cfg.planning.lateral_overlap_pct = 20.0;
        cfg
    }

    pub fn demo_default() -> Self {
        let mut cfg = Self::sar_default();
        cfg.demo = Some(DemoConfig {
            synthetic_csi: true,
            victim_positions: vec![[50.0, 80.0, 0.0], [150.0, 200.0, 0.0], [300.0, 100.0, 0.0]],
            wind_noise_ms: 2.0,
            csi_noise_std: 0.05,
            packet_loss_pct: 5.0,
            replay_speed: 1.0,
        });
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sar_default_serialization() {
        let cfg = SwarmConfig::sar_default();
        let toml_str = toml::to_string(&cfg).expect("serialize ok");
        let parsed = SwarmConfig::from_toml_str(&toml_str).expect("parse ok");
        assert_eq!(parsed.mission.profile, "sar");
    }

    #[test]
    fn test_demo_default_has_victims() {
        let cfg = SwarmConfig::demo_default();
        assert!(cfg.demo.is_some());
        assert_eq!(cfg.demo.unwrap().victim_positions.len(), 3);
    }

    #[test]
    fn test_wi2sar_reference_coverage_within_4min() {
        use crate::demo::scenario::DemoScenario;
        let scenario = DemoScenario {
            name: "Wi2SAR Reference".into(),
            config: SwarmConfig::wi2sar_reference(),
            num_drones: 4,
            victims: vec![],
        };
        let t = scenario.estimate_coverage_time_secs();
        assert!(t < 240.0, "4-drone Wi2SAR reference scenario: {}s should be < 240s (4 min SOTA)", t);
    }
}
