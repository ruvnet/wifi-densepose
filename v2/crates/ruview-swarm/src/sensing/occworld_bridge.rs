//! Bridge between OccWorld Python subprocess (ADR-147) and the Rust swarm planner.
use crate::types::Position3D;
use std::path::PathBuf;

/// A 3-D occupancy grid cell.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct VoxelCell {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,      // 0.0 = free, 1.0 = occupied
    pub semantic_class: u8,  // 0=free, 1=wall, 2=floor, 3=person, 4=furniture
}

/// Occupancy prior produced by OccWorld inference (ADR-147).
pub struct OccupancyPrior {
    pub voxels: Vec<VoxelCell>,
    pub resolution_m: f32,
    pub origin: (f32, f32, f32),
    pub timestamp_ms: u64,
}

impl OccupancyPrior {
    /// Extract free-space cells (occupancy < threshold) at a given altitude band.
    /// Used by RRT* as valid sampling space.
    pub fn free_cells_at_altitude(&self, target_z: f32, band_m: f32, threshold: f32) -> Vec<(f32, f32)> {
        self.voxels
            .iter()
            .filter(|v| v.occupancy < threshold && (v.z - target_z).abs() < band_m)
            .map(|v| (v.x, v.y))
            .collect()
    }

    /// Extract occupied cells (walls, debris). Used as obstacles for path planning.
    pub fn obstacle_cells(&self, threshold: f32) -> Vec<Position3D> {
        self.voxels
            .iter()
            .filter(|v| v.occupancy >= threshold)
            .map(|v| Position3D { x: v.x as f64, y: v.y as f64, z: v.z as f64 })
            .collect()
    }

    /// Cells where a person voxel is predicted (semantic_class == 3).
    /// Initializes the Bayesian probability grid with a prior.
    pub fn person_cells(&self) -> Vec<Position3D> {
        self.voxels
            .iter()
            .filter(|v| v.semantic_class == 3)
            .map(|v| Position3D { x: v.x as f64, y: v.y as f64, z: v.z as f64 })
            .collect()
    }

    /// Generate a synthetic 20 × 20 × 3 m room prior for demo mode.
    ///
    /// The room has wall voxels on the perimeter and free-space voxels in the
    /// interior, at the requested voxel resolution.
    pub fn synthetic_room(resolution_m: f32) -> Self {
        let mut voxels = Vec::new();
        let room = 20.0f32;
        let steps = (room / resolution_m) as i32;
        for xi in 0..steps {
            for yi in 0..steps {
                for zi in 0..15i32 { // 3 m height (15 × 0.2 m slices)
                    let x = xi as f32 * resolution_m - room / 2.0;
                    let y = yi as f32 * resolution_m - room / 2.0;
                    let z = zi as f32 * resolution_m;
                    let is_wall = xi == 0 || xi == steps - 1 || yi == 0 || yi == steps - 1;
                    voxels.push(VoxelCell {
                        x,
                        y,
                        z,
                        occupancy: if is_wall { 1.0 } else { 0.0 },
                        semantic_class: if is_wall { 1 } else if zi == 0 { 2 } else { 0 },
                    });
                }
            }
        }
        OccupancyPrior { voxels, resolution_m, origin: (0.0, 0.0, 0.0), timestamp_ms: 0 }
    }
}

/// Bridge to the OccWorld Python subprocess (ADR-147).
/// Provides 3-D occupancy priors for the RRT* path planner and the Bayesian
/// victim-probability grid. In demo mode, returns a synthetic room prior.
pub struct OccWorldBridge {
    /// Path to the OccWorld Python script.
    pub script_path: PathBuf,
    /// Cache of the last inference result.
    last_prior: Option<OccupancyPrior>,
}

impl Default for OccWorldBridge {
    fn default() -> Self {
        Self { script_path: PathBuf::from("occworld_infer.py"), last_prior: None }
    }
}

impl OccWorldBridge {
    pub fn new(script_path: PathBuf) -> Self {
        Self { script_path, last_prior: None }
    }

    /// Run a demo-mode inference using the synthetic room prior.
    /// No subprocess is spawned; the result is immediately available.
    pub async fn infer_demo(&mut self) -> &OccupancyPrior {
        self.last_prior = Some(OccupancyPrior::synthetic_room(0.2));
        self.last_prior.as_ref().unwrap()
    }

    /// Run OccWorld inference and return the occupancy prior.
    /// In demo mode: returns a synthetic prior with configurable obstacles.
    pub async fn infer(&mut self, demo_mode: bool) -> crate::SwarmResult<&OccupancyPrior> {
        if demo_mode {
            self.last_prior = Some(OccupancyPrior::synthetic_room(0.2));
        } else {
            // Production: spawn Python subprocess, read JSON output.
            // let output = tokio::process::Command::new("python3")
            //     .arg(&self.script_path)
            //     .arg("--mode=infer")
            //     .output().await?;
            // parse JSON output into OccupancyPrior.
            // Fallback to synthetic for now until subprocess integration is complete.
            self.last_prior = Some(OccupancyPrior::synthetic_room(0.2));
        }
        Ok(self.last_prior.as_ref().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_room_has_walls() {
        let prior = OccupancyPrior::synthetic_room(0.5);
        let obstacles = prior.obstacle_cells(0.5);
        assert!(!obstacles.is_empty(), "room should have wall voxels");
    }

    #[test]
    fn test_free_cells_at_altitude() {
        let prior = OccupancyPrior::synthetic_room(0.5);
        let free = prior.free_cells_at_altitude(1.5, 0.5, 0.5);
        assert!(!free.is_empty(), "room interior should have free cells");
    }
}
