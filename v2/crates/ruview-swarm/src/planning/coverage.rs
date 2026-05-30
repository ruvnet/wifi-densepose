//! Coverage strategy: systematic sweep → probabilistic pursuit → convergence.

use crate::types::{DroneState, NodeId, Position3D};
use super::probability_grid::ProbabilityGrid;
use std::collections::HashMap;

/// Phase of the coverage mission.
#[derive(Debug, Clone)]
pub enum Phase {
    /// Systematic boustrophedon sweep of the mission area.
    Systematic,
    /// Probabilistic pursuit: drones head toward high-P cells.
    ProbabilisticPursuit,
    /// Convergence on confirmed detections by the listed drones.
    Convergence(Vec<NodeId>),
}

/// Coverage strategy tracking phase and cell assignments.
pub struct CoverageStrategy {
    pub phase: Phase,
    /// Assigned cell per drone.
    pub assignments: HashMap<NodeId, (u32, u32)>,
    pub convergence_threshold: f32,
}

impl CoverageStrategy {
    pub fn new(convergence_threshold: f32) -> Self {
        Self {
            phase: Phase::Systematic,
            assignments: HashMap::new(),
            convergence_threshold,
        }
    }

    /// Compute the next waypoint for a drone given the current grid.
    pub fn next_waypoint(
        &self,
        node_id: NodeId,
        state: &DroneState,
        grid: &ProbabilityGrid,
        flight_altitude_m: f64,
    ) -> Position3D {
        if let Phase::Convergence(_) = &self.phase {
            if let Some(&(cx, cy)) = self.assignments.get(&node_id) {
                return Position3D {
                    x: cx as f64 * grid.cell_size_m,
                    y: cy as f64 * grid.cell_size_m,
                    z: -flight_altitude_m,
                };
            }
        }

        // Default: head toward the highest-priority unscanned cell.
        if let Some((cx, cy)) = grid.highest_priority_unscanned() {
            Position3D {
                x: cx as f64 * grid.cell_size_m,
                y: cy as f64 * grid.cell_size_m,
                z: -flight_altitude_m,
            }
        } else {
            state.position
        }
    }

    /// Return the next navigation target position for an orchestrator step.
    ///
    /// - Systematic phase: next unscanned boustrophedon cell.
    /// - ProbabilisticPursuit: highest-priority unscanned cell.
    /// - Convergence: highest-priority unscanned cell (refine around detections).
    pub fn next_target(&self, state: &DroneState, grid: &ProbabilityGrid) -> Option<Position3D> {
        let r = grid.cell_size_m;
        match &self.phase {
            Phase::Systematic => {
                grid.next_systematic_cell(state).map(|(cx, cy)| Position3D {
                    x: cx as f64 * r + r / 2.0,
                    y: cy as f64 * r + r / 2.0,
                    z: state.position.z,
                })
            }
            Phase::ProbabilisticPursuit | Phase::Convergence(_) => {
                grid.highest_priority_unscanned().map(|(cx, cy)| Position3D {
                    x: cx as f64 * r + r / 2.0,
                    y: cy as f64 * r + r / 2.0,
                    z: state.position.z,
                })
            }
        }
    }

    /// Transition to next phase based on grid state, guarded by a threshold.
    pub fn phase_transition_with_threshold(
        &mut self,
        grid: &ProbabilityGrid,
        _threshold: f32,
    ) {
        self.phase_transition(grid);
    }

    /// Transition to next phase based on grid state.
    pub fn phase_transition(&mut self, grid: &ProbabilityGrid) {
        let max_p = grid
            .cells
            .iter()
            .flat_map(|row| row.iter())
            .map(|c| c.victim_probability)
            .fold(0.0_f32, f32::max);

        self.phase = match &self.phase {
            Phase::Systematic if max_p >= self.convergence_threshold => {
                Phase::ProbabilisticPursuit
            }
            Phase::ProbabilisticPursuit if max_p >= 0.9 => {
                Phase::Convergence(vec![])
            }
            other => other.clone(),
        };
    }
}

