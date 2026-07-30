//! Extract a sparse point cloud from a dense [`ReflectivityImage`].
//!
//! A `nx * ny * nz` voxel grid is not a useful end product on its own --
//! real point-cloud consumers (visualization, `ruview-unified`'s
//! `GaussianMap`, downstream fusion) want a short list of "here is
//! something" points, not every voxel. This module does simple
//! threshold + local-maximum extraction: no clustering, no material
//! classification, no confidence calibration against real data (ADR-283
//! §5 -- explicitly out of scope for this crate).

use crate::geometry::Point3;
use crate::reconstruct::ReflectivityImage;
use serde::{Deserialize, Serialize};

/// A single detected point: a location and its reconstructed reflectivity
/// magnitude (relative, not calibrated to any physical unit).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointCloudPoint {
    /// World-space location, meters.
    pub position: Point3,
    /// Reconstructed reflectivity magnitude at this voxel.
    pub magnitude: f64,
}

/// Extract detected points from `image`: every voxel whose magnitude is
/// (a) at least `threshold_fraction` of the image's peak magnitude, and
/// (b) a local maximum among its 6-connected neighbors (so a single broad
/// blob yields one point, not every voxel inside it).
///
/// `threshold_fraction` must be in `(0.0, 1.0]`. A typical value is
/// `0.5` (a classic radar/SAR "half-power point" style threshold).
pub fn extract_point_cloud(image: &ReflectivityImage, threshold_fraction: f64) -> Vec<PointCloudPoint> {
    assert!(
        threshold_fraction > 0.0 && threshold_fraction <= 1.0,
        "threshold_fraction must be in (0, 1]"
    );
    let grid = &image.grid;
    let peak = image.magnitude.iter().cloned().fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let threshold = peak * threshold_fraction;

    let mag_at = |i: i64, j: i64, k: i64| -> f64 {
        if i < 0 || j < 0 || k < 0 || i as usize >= grid.nx || j as usize >= grid.ny || k as usize >= grid.nz {
            return f64::NEG_INFINITY;
        }
        let linear = grid.linear_index(i as usize, j as usize, k as usize);
        image.magnitude[linear]
    };

    let mut points = Vec::new();
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let here = mag_at(i as i64, j as i64, k as i64);
                if here < threshold {
                    continue;
                }
                let neighbors = [
                    mag_at(i as i64 - 1, j as i64, k as i64),
                    mag_at(i as i64 + 1, j as i64, k as i64),
                    mag_at(i as i64, j as i64 - 1, k as i64),
                    mag_at(i as i64, j as i64 + 1, k as i64),
                    mag_at(i as i64, j as i64, k as i64 - 1),
                    mag_at(i as i64, j as i64, k as i64 + 1),
                ];
                if neighbors.iter().all(|&n| here >= n) {
                    points.push(PointCloudPoint {
                        position: grid.voxel_center(i, j, k),
                        magnitude: here,
                    });
                }
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::linear_aperture;
    use crate::measurement::{simulate_measurement, FrequencySweep, ScatteringTarget};
    use crate::reconstruct::{backproject, VoxelGrid};

    #[test]
    fn single_target_yields_a_single_detected_point() {
        let poses = linear_aperture(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0), 21);
        let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);
        let target = ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0);
        let measurement = simulate_measurement(&poses, &sweep, &[target], 0.0, 3);
        let grid = VoxelGrid::new(Point3::new(-0.5, 1.6, -0.5), 0.05, 21, 17, 21);
        let image = backproject(&measurement, &poses, &sweep, &grid);

        let points = extract_point_cloud(&image, 0.5);
        assert!(!points.is_empty(), "must detect at least the true target");
        let best = points.iter().max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap()).unwrap();
        assert!(best.position.distance(&target.position) < 0.1);
    }

    #[test]
    fn empty_image_yields_no_points() {
        let grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 0.1, 3, 3, 3);
        let image = ReflectivityImage { grid, magnitude: vec![0.0; grid.len()] };
        assert!(extract_point_cloud(&image, 0.5).is_empty());
    }

    #[test]
    #[should_panic(expected = "threshold_fraction")]
    fn rejects_out_of_range_threshold() {
        let grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 0.1, 2, 2, 2);
        let image = ReflectivityImage { grid, magnitude: vec![1.0; grid.len()] };
        let _ = extract_point_cloud(&image, 1.5);
    }
}
