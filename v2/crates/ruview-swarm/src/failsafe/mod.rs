//! Fail-safe state machine: link loss, low battery, collision avoidance.

use crate::types::DroneState;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Fail-safe operating state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailSafeState {
    Nominal,
    AutonomousHold,
    LowBatteryWarn,
    ReturnToHome,
    EmergencyLand,
    EmergencyDiverge,
    ControlledDescent,
}

/// State machine driving fail-safe transitions.
pub struct FailSafeMachine {
    state: FailSafeState,
    link_loss_start: Option<Instant>,
    pub link_loss_hold_secs: f64,
    pub link_loss_rth_secs: f64,
    pub battery_warn_pct: f32,
    pub battery_rth_pct: f32,
    pub collision_dist_m: f64,
}

impl FailSafeMachine {
    pub fn new() -> Self {
        Self {
            state: FailSafeState::Nominal,
            link_loss_start: None,
            link_loss_hold_secs: 3.0,
            link_loss_rth_secs: 30.0,
            battery_warn_pct: 20.0,
            battery_rth_pct: 15.0,
            collision_dist_m: 1.5,
        }
    }

    /// Drive one tick. Returns the current state after evaluation.
    pub fn tick(
        &mut self,
        state: &DroneState,
        link_alive: bool,
        nearest_neighbor_dist: f64,
    ) -> FailSafeState {
        // Collision avoidance has highest priority
        if nearest_neighbor_dist < self.collision_dist_m {
            self.state = FailSafeState::EmergencyDiverge;
            return self.state.clone();
        }

        // Link loss handling
        if !link_alive {
            let start = self.link_loss_start.get_or_insert_with(Instant::now);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.link_loss_rth_secs {
                self.state = FailSafeState::ReturnToHome;
            } else if elapsed > self.link_loss_hold_secs {
                self.state = FailSafeState::AutonomousHold;
            }
            return self.state.clone();
        } else {
            // Link restored
            self.link_loss_start = None;
            if self.state == FailSafeState::AutonomousHold {
                self.state = FailSafeState::Nominal;
            }
        }

        // Battery checks
        if state.battery_pct <= self.battery_rth_pct {
            self.state = FailSafeState::ReturnToHome;
        } else if state.battery_pct <= self.battery_warn_pct {
            self.state = FailSafeState::LowBatteryWarn;
        } else if self.state == FailSafeState::LowBatteryWarn {
            // Recovered from low battery (charged on the fly / wrong reading)
            self.state = FailSafeState::Nominal;
        }

        self.state.clone()
    }

    pub fn current(&self) -> &FailSafeState {
        &self.state
    }

    pub fn force_land(&mut self) {
        self.state = FailSafeState::EmergencyLand;
    }
}

impl Default for FailSafeMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;

    fn good_state() -> DroneState {
        let mut s = DroneState::default_at_origin(NodeId(1));
        s.battery_pct = 80.0;
        s.link_quality = 1.0;
        s
    }

    #[test]
    fn test_nominal_when_healthy() {
        let mut fsm = FailSafeMachine::new();
        let s = good_state();
        let result = fsm.tick(&s, true, 10.0);
        assert_eq!(result, FailSafeState::Nominal);
    }

    #[test]
    fn test_low_battery_warn() {
        let mut fsm = FailSafeMachine::new();
        let mut s = good_state();
        s.battery_pct = 18.0;
        let result = fsm.tick(&s, true, 10.0);
        assert_eq!(result, FailSafeState::LowBatteryWarn);
    }

    #[test]
    fn test_battery_rth() {
        let mut fsm = FailSafeMachine::new();
        let mut s = good_state();
        s.battery_pct = 10.0;
        let result = fsm.tick(&s, true, 10.0);
        assert_eq!(result, FailSafeState::ReturnToHome);
    }

    #[test]
    fn test_collision_avoidance() {
        let mut fsm = FailSafeMachine::new();
        let s = good_state();
        let result = fsm.tick(&s, true, 0.5); // too close
        assert_eq!(result, FailSafeState::EmergencyDiverge);
    }
}
