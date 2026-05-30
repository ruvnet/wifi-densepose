//! End-to-end 4-drone swarm simulation for integration testing.
//!
//! Simulates a complete SAR mission: systematic sweep → victim detection →
//! multi-drone convergence. Validates M3 (CSI integration) + M7 (mission profiles).

use crate::{
    config::SwarmConfig,
    orchestrator::SwarmOrchestrator,
    types::{NodeId, Position3D},
};

/// Result of an end-to-end simulated mission.
#[derive(Debug, Clone)]
pub struct SimMissionResult {
    pub total_cells_covered: u32,
    pub victims_detected: usize,
    pub elapsed_secs: f64,
    pub collision_events: u32,
    pub final_localization_error_m: Option<f64>,
    pub coverage_pct: f64,
}

/// Run an N-drone SAR swarm simulation using the Wi2SAR reference config.
///
/// Each step:
/// 1. Each drone calls `step()` advancing its state machine.
/// 2. All drone states are exchanged via simulated MAVLink broadcast.
/// 3. Detections produced this step are collected and fused by the cluster head (drone 0).
/// 4. Mission completes when coverage_pct > 90% or all steps are exhausted.
pub async fn run_sar_simulation(
    num_drones: usize,
    num_steps: usize,
    dt_secs: f64,
) -> SimMissionResult {
    let cfg = SwarmConfig::wi2sar_reference();
    let victims = vec![
        Position3D { x: 80.0,  y: 120.0, z: 0.0 },
        Position3D { x: 250.0, y: 180.0, z: 0.0 },
    ];

    // Stagger drone starting positions across the area so they cover different cells.
    let area_w = cfg.mission.area_width_m;
    let area_h = cfg.mission.area_height_m;
    let mut drones: Vec<SwarmOrchestrator> = (0..num_drones)
        .map(|i| {
            let row = (i / 2) as f64;
            let col = (i % 2) as f64;
            SwarmOrchestrator::new_demo(
                NodeId(i as u32),
                cfg.clone(),
                Position3D {
                    x: 10.0 + col * (area_w / 2.0),
                    y: 10.0 + row * (area_h / 2.0),
                    z: -cfg.planning.flight_altitude_m,
                },
                victims.clone(),
            )
        })
        .collect();

    let mut victims_detected = 0usize;
    let mut collision_events = 0u32;
    let mut final_localization_error: Option<f64> = None;

    for _step in 0..num_steps {
        // Step all drones (each step clears peer_detections internally).
        for drone in &mut drones {
            drone.step(dt_secs, true).await;
        }

        // Exchange simulated MAVLink state messages (full mesh broadcast).
        // Collect states first to avoid borrow conflicts.
        let states: Vec<_> = drones.iter().map(|d| d.state.clone()).collect();
        for drone in &mut drones {
            for state in &states {
                if state.id != drone.node_id {
                    drone.receive_peer_state(state.clone());
                }
            }
        }

        // Gather CSI detections injected by the payload pipelines this step.
        // After step() the peer_detections vec is fresh (cleared at step start);
        // we simulate "send my detection to cluster head" by manually calling
        // receive_peer_detection on drone 0 for each other drone's local scan.
        // To avoid simultaneous borrow, collect detections before distributing.
        let local_detections: Vec<_> = drones
            .iter()
            .filter_map(|d| d.peer_detections.first().cloned())
            .collect();

        if !local_detections.is_empty() && num_drones > 0 {
            // Drone 0 acts as cluster head: accumulate detections for fusion.
            for det in &local_detections {
                if det.drone_id != drones[0].node_id {
                    drones[0].receive_peer_detection(det.clone());
                }
            }

            // Attempt multi-drone fusion on cluster head.
            let all_dets: Vec<_> = drones[0].peer_detections.clone();
            if all_dets.len() >= 2 {
                let positions: Vec<(NodeId, Position3D)> = drones
                    .iter()
                    .map(|d| (d.node_id, d.state.position))
                    .collect();

                if let Some(fused) = drones[0].fuse_detections(&all_dets, &positions) {
                    if fused.confidence > 0.7 {
                        victims_detected += 1;

                        // Compute localization error vs nearest ground-truth victim.
                        let err = victims
                            .iter()
                            .map(|v| fused.estimated_position.distance_to(v))
                            .fold(f64::MAX, f64::min);
                        final_localization_error = Some(err);
                    }
                }
            }
        }

        // Check pairwise collision events (separation < 1.5 m).
        for i in 0..drones.len() {
            for j in (i + 1)..drones.len() {
                let dist = drones[i].state.position.distance_to(&drones[j].state.position);
                if dist < 1.5 {
                    collision_events += 1;
                }
            }
        }

        // Early exit when sufficient coverage achieved.
        let avg_coverage = drones
            .iter()
            .map(|d| d.probability_grid.coverage_pct())
            .sum::<f64>()
            / drones.len() as f64;
        if avg_coverage > 0.90 {
            break;
        }
    }

    let total_cells: u32 = drones.iter().map(|d| d.stats.cells_covered).sum();
    let elapsed = drones[0].stats.elapsed_secs;
    let avg_coverage = drones
        .iter()
        .map(|d| d.probability_grid.coverage_pct())
        .sum::<f64>()
        / drones.len() as f64;

    SimMissionResult {
        total_cells_covered: total_cells,
        victims_detected,
        elapsed_secs: elapsed,
        collision_events,
        final_localization_error_m: final_localization_error,
        coverage_pct: avg_coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_4drone_sar_simulation_runs_without_panic() {
        // Quick smoke test: 20 steps at 0.5 s each = 10 simulated seconds.
        let result = run_sar_simulation(4, 20, 0.5).await;
        assert!(result.elapsed_secs > 0.0, "simulation should advance time");
        assert_eq!(result.collision_events, 0, "no collisions with proper spacing");
    }

    #[tokio::test]
    async fn test_4drone_coverage_advances() {
        // 100 steps at 1 s each = 100 simulated seconds.
        let result = run_sar_simulation(4, 100, 1.0).await;
        assert!(result.total_cells_covered > 0, "drones should cover cells");
        assert!(result.coverage_pct > 0.0, "some coverage should occur");
    }

    #[tokio::test]
    async fn test_simulation_time_tracking() {
        let result = run_sar_simulation(2, 10, 0.1).await;
        // 10 steps × 0.1 s = 1.0 s elapsed.
        assert!(
            (result.elapsed_secs - 1.0).abs() < 0.05,
            "elapsed {}s should be ~1.0s",
            result.elapsed_secs
        );
    }
}
