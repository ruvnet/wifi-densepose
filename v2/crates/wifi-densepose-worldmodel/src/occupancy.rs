//! Converts WorldGraph PersonTrack ENU positions into an [`OccupancyGrid3D`]
//! tensor suitable for submission to the OccWorld inference server (ADR-147).
//!
//! ## Voxel encoding
//! | Class index | Meaning |
//! |-------------|---------|
//! | 17          | Free space (default) |
//! | 10          | Person occupancy |
//!
//! The grid footprint is defined by axis-aligned [`SceneBounds`] in the local
//! ENU coordinate frame.  The *z* / *up* dimension is always 16 voxels; the
//! floor voxel column for a given person is derived from their `up_m` value
//! clamped to `[0, depth-1]`.

use crate::OccupancyGrid3D;

/// Class index written into voxels that contain a detected person.
pub const CLASS_PERSON: u8 = 10;
/// Class index written into voxels that are free (unoccupied).
pub const CLASS_FREE: u8 = 17;

/// Number of voxels along the east/x axis (fixed at 200).
pub const GRID_WIDTH: usize = 200;
/// Number of voxels along the north/y axis (fixed at 200).
pub const GRID_HEIGHT: usize = 200;
/// Number of voxels along the up/z axis (fixed at 16).
pub const GRID_DEPTH: usize = 16;

/// Maximum height (metres) mapped onto the depth axis.  Points above this
/// value are clamped to the topmost voxel.
const MAX_HEIGHT_M: f32 = 3.2; // 3.2 m / 16 voxels = 0.2 m per z-voxel

/// A single person position expressed in local ENU metres.
#[derive(Debug, Clone)]
pub struct PersonPosition {
    /// Stable track identifier (mirrors `WorldNode::PersonTrack::track_id`).
    pub track_id: u64,
    /// East offset from installation origin, in metres.
    pub east_m: f64,
    /// North offset from installation origin, in metres.
    pub north_m: f64,
    /// Up offset (height above floor), in metres.
    pub up_m: f64,
}

/// Axis-aligned bounding box of the scene in the ENU plane.
///
/// Maps the *east* axis to the voxel *x* dimension and the *north* axis to
/// the voxel *y* dimension.
#[derive(Debug, Clone)]
pub struct SceneBounds {
    /// Western (minimum east) edge of the scene, in metres.
    pub min_e: f64,
    /// Southern (minimum north) edge of the scene, in metres.
    pub min_n: f64,
    /// Eastern (maximum east) edge of the scene, in metres.
    pub max_e: f64,
    /// Northern (maximum north) edge of the scene, in metres.
    pub max_n: f64,
}

impl SceneBounds {
    /// Returns `(east_extent_m, north_extent_m)`.  If either dimension
    /// is zero or negative a default of `1.0` is used to avoid division by
    /// zero.
    fn extents(&self) -> (f64, f64) {
        let e = (self.max_e - self.min_e).max(1.0);
        let n = (self.max_n - self.min_n).max(1.0);
        (e, n)
    }

    /// Maps a continuous ENU coordinate to `(vx, vy)` grid indices.
    /// Out-of-bounds positions are clamped to the grid extent.
    pub fn to_voxel_xy(&self, east_m: f64, north_m: f64) -> (usize, usize) {
        let (e_ext, n_ext) = self.extents();
        let fx = (east_m - self.min_e) / e_ext; // [0, 1]
        let fy = (north_m - self.min_n) / n_ext; // [0, 1]
        let vx = (fx * GRID_WIDTH as f64)
            .floor()
            .clamp(0.0, (GRID_WIDTH - 1) as f64) as usize;
        let vy = (fy * GRID_HEIGHT as f64)
            .floor()
            .clamp(0.0, (GRID_HEIGHT - 1) as f64) as usize;
        (vx, vy)
    }

    /// Maps a height value (metres) to a voxel *z* index in `[0, depth-1]`.
    pub fn to_voxel_z(up_m: f64) -> usize {
        let fz = (up_m as f32).clamp(0.0, MAX_HEIGHT_M) / MAX_HEIGHT_M;
        let vz = (fz * GRID_DEPTH as f32)
            .floor()
            .clamp(0.0, (GRID_DEPTH - 1) as f32) as usize;
        vz
    }
}

/// Converts a list of person positions from the WorldGraph into a flat
/// [`OccupancyGrid3D`] tensor.
///
/// The voxel buffer is laid out as `[x, y, z]` with stride order
/// `voxels[z * height * width + y * width + x]` (row-major, depth last).
///
/// # Arguments
/// * `persons`    – Slice of person ENU positions (may be empty).
/// * `bounds`     – Axis-aligned scene footprint used to define the grid.
/// * `resolution_m` – Informational only; the grid is always 200×200×16 —
///   this value is echoed back in the IPC request for the Python server.
///
/// # Returns
/// An [`OccupancyGrid3D`] with `width = 200`, `height = 200`, `depth = 16`.
pub fn worldgraph_to_occupancy(
    persons: &[PersonPosition],
    bounds: &SceneBounds,
    _resolution_m: f32,
) -> OccupancyGrid3D {
    let total = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let mut voxels = vec![CLASS_FREE; total];

    for p in persons {
        let (vx, vy) = bounds.to_voxel_xy(p.east_m, p.north_m);
        let vz = SceneBounds::to_voxel_z(p.up_m);

        let idx = vz * GRID_HEIGHT * GRID_WIDTH + vy * GRID_WIDTH + vx;
        // `idx` is always in-bounds given the clamping above.
        voxels[idx] = CLASS_PERSON;
    }

    OccupancyGrid3D {
        width: GRID_WIDTH as u32,
        height: GRID_HEIGHT as u32,
        depth: GRID_DEPTH as u32,
        voxels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> SceneBounds {
        SceneBounds {
            min_e: -10.0,
            min_n: -10.0,
            max_e: 10.0,
            max_n: 10.0,
        }
    }

    #[test]
    fn empty_persons_all_free() {
        let g = worldgraph_to_occupancy(&[], &default_bounds(), 0.1);
        assert!(g.voxels.iter().all(|&v| v == CLASS_FREE));
        assert_eq!(g.voxels.len(), GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH);
    }

    #[test]
    fn person_at_origin_maps_to_centre_voxel() {
        let bounds = default_bounds(); // ±10 m; centre = (100, 100) in 200×200
        let persons = vec![PersonPosition {
            track_id: 1,
            east_m: 0.0,
            north_m: 0.0,
            up_m: 0.0,
        }];
        let g = worldgraph_to_occupancy(&persons, &bounds, 0.1);

        // At ENU (0,0,0): vx=100, vy=100, vz=0
        let expected_idx = 0 * GRID_HEIGHT * GRID_WIDTH + 100 * GRID_WIDTH + 100;
        assert_eq!(g.voxels[expected_idx], CLASS_PERSON);
        // All other voxels must still be free
        let person_count = g.voxels.iter().filter(|&&v| v == CLASS_PERSON).count();
        assert_eq!(person_count, 1);
    }

    #[test]
    fn out_of_bounds_position_is_clamped() {
        let bounds = default_bounds();
        let persons = vec![PersonPosition {
            track_id: 2,
            east_m: 99.0,  // well outside max_e=10
            north_m: 99.0,
            up_m: 100.0,
        }];
        let g = worldgraph_to_occupancy(&persons, &bounds, 0.1);
        // Should not panic; exactly one person voxel set
        let person_count = g.voxels.iter().filter(|&&v| v == CLASS_PERSON).count();
        assert_eq!(person_count, 1);
    }

    #[test]
    fn multiple_persons_independent_voxels() {
        let bounds = default_bounds();
        let persons = vec![
            PersonPosition { track_id: 1, east_m: -5.0, north_m: -5.0, up_m: 0.5 },
            PersonPosition { track_id: 2, east_m: 5.0,  north_m: 5.0,  up_m: 1.5 },
        ];
        let g = worldgraph_to_occupancy(&persons, &bounds, 0.1);
        let person_count = g.voxels.iter().filter(|&&v| v == CLASS_PERSON).count();
        assert_eq!(person_count, 2);
    }

    #[test]
    fn grid_dimensions_correct() {
        let g = worldgraph_to_occupancy(&[], &default_bounds(), 0.4);
        assert_eq!(g.width, 200);
        assert_eq!(g.height, 200);
        assert_eq!(g.depth, 16);
        assert_eq!(g.voxels.len(), 200 * 200 * 16);
    }
}
