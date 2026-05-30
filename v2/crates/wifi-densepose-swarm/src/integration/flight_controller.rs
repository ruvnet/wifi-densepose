//! Flight controller abstraction and simulated implementation.

use crate::types::{DroneState, NodeId, Position3D};
use async_trait::async_trait;
use tokio::sync::Mutex;

/// Flight controller operating mode.
#[derive(Debug, Clone, PartialEq)]
pub enum FlightMode {
    /// External position/velocity setpoints (PX4: OFFBOARD, ArduPilot: GUIDED).
    Offboard,
    Loiter,
    ReturnToLaunch,
    Land,
    Stabilize,
}

/// Abstraction over flight controller interfaces (PX4, ArduPilot, custom).
#[async_trait]
pub trait FlightController: Send + Sync {
    async fn set_target_position(
        &self,
        pos: &Position3D,
        speed_ms: f64,
    ) -> crate::SwarmResult<()>;

    async fn get_state(&self) -> crate::SwarmResult<DroneState>;

    async fn set_mode(&self, mode: FlightMode) -> crate::SwarmResult<()>;

    async fn arm(&self) -> crate::SwarmResult<()>;

    async fn disarm(&self) -> crate::SwarmResult<()>;

    async fn rtl(&self) -> crate::SwarmResult<()>;

    async fn emergency_land(&self) -> crate::SwarmResult<()>;
}

/// A simulated flight controller that immediately applies position commands.
/// Used in tests and demo mode.
pub struct SimulatedFlightController {
    pub state: Mutex<DroneState>,
}

impl SimulatedFlightController {
    pub fn new(id: NodeId) -> Self {
        Self {
            state: Mutex::new(DroneState::default_at_origin(id)),
        }
    }
}

#[async_trait]
impl FlightController for SimulatedFlightController {
    async fn set_target_position(
        &self,
        pos: &Position3D,
        _speed_ms: f64,
    ) -> crate::SwarmResult<()> {
        let mut state = self.state.lock().await;
        state.position = *pos;
        Ok(())
    }

    async fn get_state(&self) -> crate::SwarmResult<DroneState> {
        let state = self.state.lock().await;
        Ok(state.clone())
    }

    async fn set_mode(&self, _mode: FlightMode) -> crate::SwarmResult<()> {
        Ok(())
    }

    async fn arm(&self) -> crate::SwarmResult<()> {
        Ok(())
    }

    async fn disarm(&self) -> crate::SwarmResult<()> {
        Ok(())
    }

    async fn rtl(&self) -> crate::SwarmResult<()> {
        let mut state = self.state.lock().await;
        state.position = Position3D::zero();
        Ok(())
    }

    async fn emergency_land(&self) -> crate::SwarmResult<()> {
        let mut state = self.state.lock().await;
        state.altitude_agl_m = 0.0;
        state.position.z = 0.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_set_position_updates_state() {
        let fc = SimulatedFlightController::new(NodeId(0));
        let target = Position3D { x: 50.0, y: 30.0, z: -20.0 };
        fc.set_target_position(&target, 5.0).await.unwrap();
        let state = fc.get_state().await.unwrap();
        assert!((state.position.x - 50.0).abs() < 1e-6);
        assert!((state.position.y - 30.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_rtl_returns_to_origin() {
        let fc = SimulatedFlightController::new(NodeId(1));
        fc.set_target_position(
            &Position3D { x: 100.0, y: 100.0, z: -30.0 },
            5.0,
        )
        .await
        .unwrap();
        fc.rtl().await.unwrap();
        let state = fc.get_state().await.unwrap();
        assert!(state.position.x.abs() < 1e-6);
        assert!(state.position.y.abs() < 1e-6);
    }
}
