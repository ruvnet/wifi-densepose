//! Benchmark support utilities: scenario builders and timing helpers for criterion benchmarks.

use crate::types::{DroneState, NodeId, Position3D, Velocity3D};

/// Generate N drone states arranged in a grid.
pub fn grid_drone_states(n: usize, spacing_m: f64) -> Vec<DroneState> {
    let side = (n as f64).sqrt().ceil() as usize;
    (0..n)
        .map(|i| {
            let row = i / side;
            let col = i % side;
            DroneState {
                id: NodeId(i as u32),
                position: Position3D {
                    x: col as f64 * spacing_m,
                    y: row as f64 * spacing_m,
                    z: -30.0,
                },
                velocity: Velocity3D::default(),
                heading_rad: 0.0,
                altitude_agl_m: 30.0,
                battery_pct: 80.0,
                link_quality: 0.9,
                timestamp_ms: 0,
            }
        })
        .collect()
}

/// Generate N evenly-spaced positions in a circle.
pub fn circle_positions(n: usize, radius_m: f64) -> Vec<(NodeId, Position3D)> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (
                NodeId(i as u32),
                Position3D {
                    x: radius_m * angle.cos(),
                    y: radius_m * angle.sin(),
                    z: -30.0,
                },
            )
        })
        .collect()
}
