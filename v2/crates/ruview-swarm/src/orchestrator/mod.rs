//! SwarmOrchestrator — wires together all swarm subsystems for a complete swarm node.
//!
//! Each physical drone runs one SwarmOrchestrator instance. In demo/sim mode it
//! runs N orchestrators in one process to simulate a full swarm.

use crate::{
    config::SwarmConfig,
    failsafe::{FailSafeMachine, FailSafeState},
    sensing::{
        multiview::MultiViewFusion,
        payload::{CsiPayloadPipeline, PayloadConfig},
    },
    planning::{
        coverage::CoverageStrategy,
        probability_grid::ProbabilityGrid,
    },
    types::{CsiDetection, DroneState, NodeId, Position3D, Velocity3D},
};
use std::collections::HashMap;

/// The complete per-drone swarm coordinator.
///
/// In production: backed by live CSI payload and PX4 flight controller.
/// In demo/sim: backed by synthetic CSI and simulated state.
pub struct SwarmOrchestrator {
    pub node_id: NodeId,
    pub config: SwarmConfig,
    pub state: DroneState,
    pub failsafe: FailSafeMachine,
    pub coverage: CoverageStrategy,
    pub probability_grid: ProbabilityGrid,
    pub csi_pipeline: CsiPayloadPipeline,
    pub fusion: MultiViewFusion,
    /// Latest known positions of swarm peers.
    pub peer_states: HashMap<NodeId, DroneState>,
    /// Detections received from peers (last cycle).
    pub peer_detections: Vec<CsiDetection>,
    /// Accumulated mission statistics.
    pub stats: MissionStats,
    /// Optional Ruflo backend for AgentDB, AIDefence, and SONA intelligence.
    /// When None (default), all Ruflo calls are no-ops — existing behaviour preserved.
    #[cfg(feature = "ruflo")]
    pub ruflo: Option<Box<dyn crate::ruflo::RufloBackend>>,
    /// Active trajectory ID issued by the Ruflo intelligence hooks.
    #[cfg(feature = "ruflo")]
    pub trajectory_id: Option<String>,
}

/// Accumulated metrics for one mission run.
#[derive(Debug, Clone, Default)]
pub struct MissionStats {
    pub cells_covered: u32,
    pub victims_confirmed: u32,
    pub collision_events: u32,
    pub steps: u64,
    pub elapsed_secs: f64,
}

impl SwarmOrchestrator {
    /// Create a new orchestrator in demo mode (synthetic CSI).
    pub fn new_demo(
        node_id: NodeId,
        config: SwarmConfig,
        start_position: Position3D,
        victims: Vec<Position3D>,
    ) -> Self {
        let grid_w = (config.mission.area_width_m / config.mission.grid_resolution_m).ceil() as u32;
        let grid_h = (config.mission.area_height_m / config.mission.grid_resolution_m).ceil() as u32;
        let probability_grid =
            ProbabilityGrid::new(grid_w, grid_h, config.mission.grid_resolution_m);

        let noise_std = config.demo.as_ref().map(|d| d.csi_noise_std).unwrap_or(0.05);
        let detection_range = config.planning.csi_scan_width_m;
        let convergence_threshold = config.planning.convergence_threshold;

        let csi_pipeline = CsiPayloadPipeline::new_synthetic(
            node_id,
            PayloadConfig {
                scan_freq_hz: 10.0,
                detection_range_m: detection_range,
                confidence_threshold: 0.5,
                esp32_baud_rate: 921_600,
            },
            victims,
            noise_std,
            node_id.0 as u64,
        );

        let state = DroneState {
            id: node_id,
            position: start_position,
            velocity: Velocity3D::default(),
            heading_rad: 0.0,
            altitude_agl_m: config.planning.flight_altitude_m,
            battery_pct: 100.0,
            link_quality: 1.0,
            timestamp_ms: 0,
        };

        Self {
            node_id,
            config: config.clone(),
            state,
            failsafe: FailSafeMachine::new(),
            coverage: CoverageStrategy::new(convergence_threshold),
            probability_grid,
            csi_pipeline,
            fusion: MultiViewFusion::default(),
            peer_states: HashMap::new(),
            peer_detections: Vec::new(),
            stats: MissionStats::default(),
            #[cfg(feature = "ruflo")]
            ruflo: None,
            #[cfg(feature = "ruflo")]
            trajectory_id: None,
        }
    }

    /// Process one simulation step (dt_secs: time elapsed since last step).
    /// Returns the current fail-safe state after evaluation.
    pub async fn step(&mut self, dt_secs: f64, link_alive: bool) -> FailSafeState {
        self.stats.steps += 1;
        self.stats.elapsed_secs += dt_secs;

        // 1. Drain stale peer detections from previous cycle.
        self.peer_detections.clear();

        // 2. Evaluate fail-safe state machine.
        let nearest_dist = self.nearest_peer_distance();
        let fs_state = self.failsafe.tick(&self.state, link_alive, nearest_dist);

        if fs_state != FailSafeState::Nominal && fs_state != FailSafeState::LowBatteryWarn {
            return fs_state; // safety takes over; skip mission logic
        }

        // 3. CSI scan at current position.
        let current_pos = self.state.position;
        if let Some(detection) = self.csi_pipeline.scan(&current_pos).await {
            if detection.confidence >= self.csi_pipeline.config.confidence_threshold {
                if let Some(victim_pos) = detection.victim_position {
                    let cell = self.pos_to_cell(&victim_pos);
                    self.probability_grid.update_bayesian(cell, detection.confidence, true);
                }
            }
        }

        // 4. Mark current cell as scanned.
        let cur_cell = self.pos_to_cell(&current_pos);
        let was_new = self.probability_grid.mark_scanned(cur_cell);
        if was_new {
            self.stats.cells_covered += 1;
        }

        // 5. Update coverage phase based on grid state.
        self.coverage.phase_transition(&self.probability_grid);

        // 6. Move toward next waypoint (proportional navigation for simulation).
        if let Some(target) = self.coverage.next_target(&self.state, &self.probability_grid) {
            self.move_toward(target, dt_secs);
        }

        // 7. Simple battery drain: 1% per 30 s at full speed.
        self.state.battery_pct -= (dt_secs / 30.0) as f32;
        self.state.battery_pct = self.state.battery_pct.max(0.0);
        self.state.timestamp_ms += (dt_secs * 1_000.0) as u64;

        fs_state
    }

    /// Multi-drone CSI fusion at the cluster-head level.
    /// Returns a fused detection if enough viewpoints agree.
    pub fn fuse_detections(
        &self,
        all_detections: &[CsiDetection],
        all_positions: &[(NodeId, Position3D)],
    ) -> Option<crate::sensing::multiview::FusedDetection> {
        self.fusion.fuse(all_detections, all_positions)
    }

    /// Accept an incoming peer state update (called by the swarm comm layer).
    pub fn receive_peer_state(&mut self, peer: DroneState) {
        self.peer_states.insert(peer.id, peer);
    }

    /// Accept an incoming CSI detection from a peer.
    pub fn receive_peer_detection(&mut self, det: CsiDetection) {
        self.peer_detections.push(det);
    }

    /// Attach a Ruflo backend for AgentDB pattern learning, AIDefence, and SONA.
    ///
    /// Call after `new_demo()`:
    /// ```ignore
    /// let orch = SwarmOrchestrator::new_demo(...)
    ///     .with_ruflo(Box::new(MockRufloBackend::new()));
    /// ```
    #[cfg(feature = "ruflo")]
    pub fn with_ruflo(mut self, backend: Box<dyn crate::ruflo::RufloBackend>) -> Self {
        self.ruflo = Some(backend);
        self
    }

    /// Start a Ruflo intelligence trajectory for this mission node.
    ///
    /// Call before the mission loop begins. If no backend is attached this is a no-op.
    #[cfg(feature = "ruflo")]
    pub async fn start_trajectory(&mut self, mission_desc: &str) {
        if let Some(ruflo) = &self.ruflo {
            match ruflo.trajectory_start(mission_desc, "swarm-specialist").await {
                Ok(tid) => self.trajectory_id = Some(tid),
                Err(e) => tracing::warn!("trajectory_start failed: {}", e),
            }
        }
    }

    /// End the Ruflo trajectory and persist the mission summary in AgentDB.
    ///
    /// Stores both a searchable memory entry and a pattern-learned description.
    /// If no backend is attached this is a no-op.
    #[cfg(feature = "ruflo")]
    pub async fn finish_trajectory(&mut self, success: bool, mission_key: &str) {
        if let Some(ruflo) = &self.ruflo {
            let tid = self.trajectory_id.take();
            if let Some(tid) = &tid {
                let _ = ruflo.trajectory_end(tid, success, None).await;
            }
            // Build and serialise mission summary.
            let summary = crate::ruflo::MissionSummary::from_stats(
                &self.stats,
                &self.config.mission.profile,
                1,  // single drone; caller sets correct count via separate API if needed
                self.config.mission.area_width_m,
                self.config.mission.area_height_m,
                0,  // caller sets victims_total; 0 = unknown
                self.probability_grid.coverage_pct(),
            );
            if let Ok(json) = serde_json::to_string(&summary) {
                let _ = ruflo.store_mission(mission_key, &json, "swarm-missions").await;
            }
            let _ = ruflo.store_pattern(
                &summary.to_pattern_description(),
                summary.pattern_type(),
                summary.pattern_confidence(),
            ).await;
        }
    }

    /// AIDefence-checked variant of `receive_peer_detection`.
    ///
    /// Returns `true` and enqueues the detection if it passes the safety check.
    /// Returns `false` (and drops the detection) if AIDefence flags it as unsafe.
    /// Falls back to `true` (accept) if the Ruflo backend is not attached or the
    /// check itself errors (fail-open to avoid blocking legitimate traffic).
    #[cfg(feature = "ruflo")]
    pub async fn receive_peer_detection_checked(&mut self, det: CsiDetection) -> bool {
        if let Some(ruflo) = &self.ruflo {
            // Serialise the detection to a string for AIDefence inspection.
            let repr = format!(
                "drone_id={:?} confidence={:.3} victim={:?}",
                det.drone_id, det.confidence, det.victim_position
            );
            match ruflo.mavlink_is_safe(&repr).await {
                Ok(false) => {
                    tracing::warn!(
                        "aidefence rejected peer detection from {:?}",
                        det.drone_id
                    );
                    return false;
                }
                Err(e) => tracing::debug!("aidefence check failed (proceeding): {}", e),
                _ => {}
            }
        }
        self.receive_peer_detection(det);
        true
    }

    /// Returns true when the mission is considered complete.
    pub fn is_mission_complete(&self) -> bool {
        self.probability_grid.coverage_pct() > 0.95
    }

    // ──────────────────────── private helpers ────────────────────────

    /// Distance to the nearest peer drone (f64::MAX if no peers).
    fn nearest_peer_distance(&self) -> f64 {
        self.peer_states
            .values()
            .map(|p| self.state.position.distance_to(&p.position))
            .fold(f64::MAX, f64::min)
    }

    /// Convert a world position to grid cell indices, clamped to grid bounds.
    fn pos_to_cell(&self, pos: &Position3D) -> (u32, u32) {
        let r = self.config.mission.grid_resolution_m;
        let w = (self.config.mission.area_width_m / r) as u32;
        let h = (self.config.mission.area_height_m / r) as u32;
        let xi = (pos.x / r).max(0.0) as u32;
        let yi = (pos.y / r).max(0.0) as u32;
        (xi.min(w.saturating_sub(1)), yi.min(h.saturating_sub(1)))
    }

    /// Simple proportional navigation: steer toward target at max planning speed.
    fn move_toward(&mut self, target: Position3D, dt_secs: f64) {
        let dx = target.x - self.state.position.x;
        let dy = target.y - self.state.position.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.5 {
            self.state.velocity = Velocity3D::default();
            return;
        }

        let speed = self.config.planning.max_speed_ms.min(dist / dt_secs);
        let vx = (dx / dist) * speed;
        let vy = (dy / dist) * speed;

        self.state.position.x += vx * dt_secs;
        self.state.position.y += vy * dt_secs;
        self.state.velocity = Velocity3D { vx, vy, vz: 0.0 };
        self.state.heading_rad = vy.atan2(vx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_orchestrator(node_id: u32, victims: Vec<Position3D>) -> SwarmOrchestrator {
        let cfg = SwarmConfig::demo_default();
        SwarmOrchestrator::new_demo(
            NodeId(node_id),
            cfg,
            Position3D { x: 10.0 * node_id as f64, y: 0.0, z: -30.0 },
            victims,
        )
    }

    #[tokio::test]
    async fn test_single_orchestrator_step() {
        let mut orch =
            demo_orchestrator(0, vec![Position3D { x: 50.0, y: 50.0, z: 0.0 }]);
        let state = orch.step(0.1, true).await;
        assert_eq!(state, FailSafeState::Nominal);
        assert_eq!(orch.stats.steps, 1);
    }

    #[tokio::test]
    async fn test_failsafe_triggers_on_link_loss() {
        let mut orch = demo_orchestrator(0, vec![]);
        // Lower the hold threshold so it trips well within a sub-second test run.
        orch.failsafe.link_loss_hold_secs = 0.001;
        orch.failsafe.link_loss_rth_secs = 0.1;

        // One tick to start the link-loss timer, then sleep briefly so the
        // real-time elapsed exceeds the tiny hold threshold.
        orch.step(0.1, false).await;
        std::thread::sleep(std::time::Duration::from_millis(5));

        let state = orch.step(0.1, false).await;
        assert_ne!(state, FailSafeState::Nominal, "link loss should trigger failsafe");
    }

    #[tokio::test]
    async fn test_multi_drone_coverage() {
        let victims = vec![Position3D { x: 50.0, y: 50.0, z: 0.0 }];
        let mut drones: Vec<SwarmOrchestrator> =
            (0..4).map(|i| demo_orchestrator(i, victims.clone())).collect();

        // 50 steps × 0.1 s dt = 5 simulated seconds
        for _ in 0..50 {
            for drone in &mut drones {
                drone.step(0.1, true).await;
            }
        }

        let total_cells: u32 = drones.iter().map(|d| d.stats.cells_covered).sum();
        assert!(total_cells > 0, "drones should have covered some cells");

        let elapsed = drones[0].stats.elapsed_secs;
        assert!((elapsed - 5.0).abs() < 0.01, "elapsed should be ~5 s, got {elapsed}");
    }

    #[tokio::test]
    async fn test_peer_state_exchange() {
        let mut orch0 = demo_orchestrator(0, vec![]);
        let mut orch1 = demo_orchestrator(1, vec![]);

        orch0.step(0.1, true).await;
        orch1.step(0.1, true).await;

        // Exchange states
        orch0.receive_peer_state(orch1.state.clone());
        orch1.receive_peer_state(orch0.state.clone());

        assert!(
            orch0.peer_states.contains_key(&NodeId(1)),
            "orch0 should know about orch1"
        );
    }

    #[tokio::test]
    async fn test_mission_complete_after_full_coverage() {
        let mut orch = demo_orchestrator(0, vec![]);
        // Manually mark every cell scanned.
        let w = orch.probability_grid.width;
        let h = orch.probability_grid.height;
        for y in 0..h {
            for x in 0..w {
                orch.probability_grid.mark_scanned((x, y));
            }
        }
        assert!(orch.is_mission_complete(), "should be complete at 100% coverage");
    }
}
