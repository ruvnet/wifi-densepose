use crate::types::DroneState;

/// Reward function for the MAPPO training loop.
///
/// Shaped reward components:
///   +coverage_reward         per new grid cell visited
///   +detection_reward        per confirmed victim detection
///   +triangulation_reward    per contribution to a triangulation event
///   idle_penalty             when no useful work done this step
///   collision_penalty        when nearest neighbour < min_separation_m
///   geofence_penalty         when drone breaches the mission boundary
///   battery_depletion_penalty when battery runs out outside RTH range
pub struct RewardCalculator {
    pub coverage_reward: f32,
    pub detection_reward: f32,
    pub triangulation_reward: f32,
    pub idle_penalty: f32,
    pub collision_penalty: f32,
    pub geofence_penalty: f32,
    pub battery_depletion_penalty: f32,
    pub min_separation_m: f64,
}

impl Default for RewardCalculator {
    fn default() -> Self {
        Self {
            coverage_reward: 10.0,
            detection_reward: 50.0,
            triangulation_reward: 5.0,
            idle_penalty: -2.0,
            collision_penalty: -100.0,
            geofence_penalty: -50.0,
            battery_depletion_penalty: -30.0,
            min_separation_m: 1.5,
        }
    }
}

/// Context needed to compute the reward for a single agent step.
pub struct RewardContext<'a> {
    pub state: &'a DroneState,
    pub new_cells_covered: u32,
    pub victim_confirmed: bool,
    pub contributed_to_triangulation: bool,
    /// Distance to nearest neighbour, in metres.
    pub nearest_neighbor_dist: f64,
    pub geofence_breached: bool,
    pub battery_depleted_without_rth: bool,
}

impl RewardCalculator {
    /// Compute the scalar reward for one agent at one timestep.
    pub fn compute(&self, ctx: &RewardContext) -> f32 {
        let mut reward = 0.0f32;

        reward += ctx.new_cells_covered as f32 * self.coverage_reward;

        if ctx.victim_confirmed {
            reward += self.detection_reward;
        }
        if ctx.contributed_to_triangulation {
            reward += self.triangulation_reward;
        }
        // Idle penalty only when no positive work was done.
        if ctx.new_cells_covered == 0 && !ctx.victim_confirmed {
            reward += self.idle_penalty;
        }
        if ctx.nearest_neighbor_dist < self.min_separation_m {
            reward += self.collision_penalty;
        }
        if ctx.geofence_breached {
            reward += self.geofence_penalty;
        }
        if ctx.battery_depleted_without_rth {
            reward += self.battery_depletion_penalty;
        }

        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DroneState, NodeId};

    fn mk_state() -> DroneState {
        DroneState::default_at_origin(NodeId(0))
    }

    #[test]
    fn detection_reward_dominates() {
        let calc = RewardCalculator::default();
        let state = mk_state();
        let ctx = RewardContext {
            state: &state,
            new_cells_covered: 1,
            victim_confirmed: true,
            contributed_to_triangulation: false,
            nearest_neighbor_dist: 10.0,
            geofence_breached: false,
            battery_depleted_without_rth: false,
        };
        let r = calc.compute(&ctx);
        // 10 (coverage) + 50 (detection) = 60
        assert!((r - 60.0).abs() < 1e-4, "reward={}", r);
    }

    #[test]
    fn collision_dominates_idle() {
        let calc = RewardCalculator::default();
        let state = mk_state();
        let ctx = RewardContext {
            state: &state,
            new_cells_covered: 0,
            victim_confirmed: false,
            contributed_to_triangulation: false,
            nearest_neighbor_dist: 0.5, // < 1.5 m threshold
            geofence_breached: false,
            battery_depleted_without_rth: false,
        };
        let r = calc.compute(&ctx);
        // -2 (idle) + -100 (collision) = -102
        assert!((r - (-102.0)).abs() < 1e-4, "reward={}", r);
    }

    #[test]
    fn test_collision_dominates() {
        let calc = RewardCalculator::default();
        let state = mk_state();
        // 3 covered cells = +30, victim = false, collision = -100 → net -70
        let ctx = RewardContext {
            state: &state,
            new_cells_covered: 3,
            victim_confirmed: false,
            contributed_to_triangulation: false,
            nearest_neighbor_dist: 1.0,  // collision (< 1.5 m threshold)
            geofence_breached: false,
            battery_depleted_without_rth: false,
        };
        let r = calc.compute(&ctx);
        assert!(r < 0.0, "collision (-100) should dominate coverage (+30), reward={}", r);
    }
}
