//! Delay-and-sum backprojection reconstruction (ADR-287 §2).
//!
//! Given a [`crate::measurement::Measurement`] recorded from known antenna
//! [`AntennaPose`]s across a known [`FrequencySweep`], reconstruct a 3D
//! reflectivity image on a regular voxel grid:
//!
//! ```text
//! I(x) = | (1 / (M*K)) * sum_m sum_k  y_{m,k} * R_{m,x}^2 * exp(+i * 4*pi * f_k * R_{m,x} / c) |
//! ```
//!
//! This is the matched-filter / frequency-domain backprojection kernel:
//! for a *correct* hypothesis voxel `x` coinciding with a real scatterer,
//! every (pose, frequency) term's phase-correction exactly cancels the
//! phase the forward model applied in [`crate::measurement`], so the sum
//! coheres constructively. For any other voxel the per-term phases are
//! effectively uncorrelated across the (pose, frequency) grid and the sum
//! averages toward zero. The `R_{m,x}^2` factor undoes the forward
//! model's `1/R^2` spreading-loss term (matched-filter gain
//! compensation), so voxel brightness reflects relative reflectivity
//! rather than falling off with range.

use crate::geometry::{AntennaPose, Point3};
use crate::measurement::{FrequencySweep, Measurement};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::resolution::SPEED_OF_LIGHT_M_PER_S;

/// A regular 3D grid of voxel centers over an axis-aligned box.
#[derive(Debug, Clone, Copy)]
pub struct VoxelGrid {
    /// Grid origin (the center of voxel `(0,0,0)`), meters.
    pub origin: Point3,
    /// Voxel edge length along each axis, meters.
    pub spacing: f64,
    /// Number of voxels along x.
    pub nx: usize,
    /// Number of voxels along y.
    pub ny: usize,
    /// Number of voxels along z.
    pub nz: usize,
}

impl VoxelGrid {
    /// Construct a grid.
    pub fn new(origin: Point3, spacing: f64, nx: usize, ny: usize, nz: usize) -> Self {
        assert!(spacing > 0.0, "voxel spacing must be positive");
        Self { origin, spacing, nx, ny, nz }
    }

    /// Total voxel count.
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// True if the grid has zero voxels along any axis.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// World-space center of voxel `(i, j, k)`.
    pub fn voxel_center(&self, i: usize, j: usize, k: usize) -> Point3 {
        Point3::new(
            self.origin.x + (i as f64) * self.spacing,
            self.origin.y + (j as f64) * self.spacing,
            self.origin.z + (k as f64) * self.spacing,
        )
    }

    /// Flatten a 3D voxel index into a linear index (row-major, x fastest).
    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    /// Recover the 3D voxel index `(i, j, k)` from a linear index.
    pub fn unflatten(&self, linear: usize) -> (usize, usize, usize) {
        let i = linear % self.nx;
        let j = (linear / self.nx) % self.ny;
        let k = linear / (self.nx * self.ny);
        (i, j, k)
    }
}

/// The reconstructed reflectivity image: one magnitude value per voxel,
/// row-major (`grid.linear_index`/`unflatten` order).
#[derive(Debug, Clone)]
pub struct ReflectivityImage {
    /// The grid this image was reconstructed on.
    pub grid: VoxelGrid,
    /// Per-voxel reflectivity magnitude, same length as `grid.len()`.
    pub magnitude: Vec<f64>,
}

impl ReflectivityImage {
    /// The voxel with the largest magnitude, and its world-space center.
    pub fn peak(&self) -> (Point3, f64) {
        let (idx, &mag) = self
            .magnitude
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("grid must have at least one voxel");
        let (i, j, k) = self.grid.unflatten(idx);
        (self.grid.voxel_center(i, j, k), mag)
    }
}

/// Reconstruct a reflectivity image from `measurement`, recorded at
/// `poses` across `sweep`, onto `grid`. Parallelized over voxels (rayon).
///
/// `poses` and `sweep` must describe the *same* geometry the measurement
/// was recorded with (or, when studying pose-error sensitivity, a
/// deliberately perturbed version of it -- see
/// `tests/physics_validation.rs`).
pub fn backproject(
    measurement: &Measurement,
    poses: &[AntennaPose],
    sweep: &FrequencySweep,
    grid: &VoxelGrid,
) -> ReflectivityImage {
    assert_eq!(poses.len(), measurement.n_poses, "pose count must match measurement");
    assert_eq!(sweep.n_steps, measurement.n_freqs, "frequency count must match measurement");

    let n = grid.len();

    let magnitude: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|linear| {
            let (i, j, k) = grid.unflatten(linear);
            let voxel = grid.voxel_center(i, j, k);
            focus_at_point(measurement, poses, sweep, &voxel)
        })
        .collect();

    ReflectivityImage { grid: *grid, magnitude }
}

/// Evaluate the coherent backprojection sum at a single world-space
/// `point`, without building a grid. This is the same matched-filter
/// kernel [`backproject`] evaluates per voxel; exposed directly so callers
/// (and tests) can measure focus quality exactly at a location of
/// interest -- e.g. a known target position -- rather than only at
/// whatever grid points happen to be sampled.
///
/// Takes `sweep` rather than a raw frequency slice specifically so the
/// evenly-spaced-frequencies guarantee ([`FrequencySweep::frequencies`])
/// is a type-level invariant, not a caller-observed precondition: the
/// implementation below relies on it (see the comment inside the pose
/// loop). Passing an arbitrary non-uniform frequency list is not possible
/// through this signature.
pub fn focus_at_point(measurement: &Measurement, poses: &[AntennaPose], sweep: &FrequencySweep, point: &Point3) -> f64 {
    assert_eq!(poses.len(), measurement.n_poses, "pose count must match measurement");
    assert_eq!(sweep.n_steps, measurement.n_freqs, "frequency count must match measurement");

    let n_terms = (measurement.n_poses * measurement.n_freqs) as f64;
    let k = sweep.n_steps;
    // Frequencies are evenly spaced by construction: f_kf = start_hz + kf *
    // delta_f. That makes the per-term phase phase_kf = 4*pi*f_kf*r/c an
    // arithmetic progression in kf, so instead of K trig evaluations
    // (Complex64::from_polar per frequency step) the phasor is evaluated
    // once and advanced by a fixed per-step rotation -- one complex
    // multiply per step instead of a sin/cos pair. Proven equivalent to
    // the direct per-frequency computation (independently reimplemented,
    // not reusing this code) in
    // `backprojection_incremental_rotation_matches_direct_per_frequency_computation`.
    let delta_f = if k > 1 { (sweep.stop_hz - sweep.start_hz) / (k - 1) as f64 } else { 0.0 };

    let mut acc = Complex64::new(0.0, 0.0);
    for (m, pose) in poses.iter().enumerate() {
        let r = pose.position.distance(point);
        if r < 1e-6 {
            continue;
        }
        let gain_compensation = r * r;
        let base_phase = 4.0 * PI * sweep.start_hz * r / SPEED_OF_LIGHT_M_PER_S;
        let step_phase = 4.0 * PI * delta_f * r / SPEED_OF_LIGHT_M_PER_S;
        let step = Complex64::from_polar(1.0, step_phase);
        let mut rot = Complex64::from_polar(1.0, base_phase);
        for kf in 0..k {
            acc += measurement.get(m, kf) * gain_compensation * rot;
            if kf + 1 < k {
                rot *= step;
            }
        }
    }
    acc.norm() / n_terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::linear_aperture;
    use crate::measurement::{simulate_measurement, ScatteringTarget};

    #[test]
    fn voxel_grid_flatten_unflatten_roundtrip() {
        let grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 0.1, 4, 5, 3);
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let lin = grid.linear_index(i, j, k);
                    assert_eq!(grid.unflatten(lin), (i, j, k));
                }
            }
        }
    }

    #[test]
    fn single_point_target_reconstructs_at_its_true_location() {
        let poses = linear_aperture(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0), 21);
        let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);
        let target = ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0);
        let measurement = simulate_measurement(&poses, &sweep, &[target], 0.0, 1);

        let grid = VoxelGrid::new(Point3::new(-0.5, 1.6, -0.5), 0.05, 21, 17, 21);
        let image = backproject(&measurement, &poses, &sweep, &grid);
        let (peak_loc, _peak_mag) = image.peak();

        let err = peak_loc.distance(&target.position);
        assert!(err < 0.1, "reconstructed peak {:?} should be within one voxel-ish of the true target {:?}, err={err}", peak_loc, target.position);
    }

    #[test]
    fn peak_at_target_is_far_above_background() {
        let poses = linear_aperture(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0), 21);
        let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);
        let target = ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0);
        let measurement = simulate_measurement(&poses, &sweep, &[target], 0.0, 2);

        let grid = VoxelGrid::new(Point3::new(-0.5, 1.6, -0.5), 0.05, 21, 17, 21);
        let image = backproject(&measurement, &poses, &sweep, &grid);
        let (_peak_loc, peak_mag) = image.peak();
        let mean_mag: f64 = image.magnitude.iter().sum::<f64>() / image.magnitude.len() as f64;

        assert!(peak_mag > mean_mag * 5.0, "coherent focus at the target must dominate the incoherent background: peak={peak_mag}, mean={mean_mag}");
    }

    /// Independent reference: the direct per-frequency computation
    /// `focus_at_point` used before the incremental-phasor-rotation
    /// optimization (one `Complex64::from_polar` per (pose, frequency)
    /// term, no recurrence). Deliberately reimplemented here rather than
    /// calling any shared helper, so this test cannot pass by construction.
    fn focus_at_point_direct_reference(
        measurement: &Measurement,
        poses: &[AntennaPose],
        sweep: &FrequencySweep,
        point: &Point3,
    ) -> f64 {
        let freqs = sweep.frequencies();
        let n_terms = (measurement.n_poses * measurement.n_freqs) as f64;
        let mut acc = Complex64::new(0.0, 0.0);
        for (m, pose) in poses.iter().enumerate() {
            let r = pose.position.distance(point);
            if r < 1e-6 {
                continue;
            }
            let gain_compensation = r * r;
            for (kf, &f) in freqs.iter().enumerate() {
                let phase = 4.0 * PI * f * r / SPEED_OF_LIGHT_M_PER_S;
                acc += measurement.get(m, kf) * gain_compensation * Complex64::from_polar(1.0, phase);
            }
        }
        acc.norm() / n_terms
    }

    /// PERF PROOF: the incremental-phasor-rotation `focus_at_point` (2
    /// trig evaluations/pose instead of K) matches the direct
    /// per-frequency reference to within f64 rounding, across several
    /// sweep sizes, ranges, and off-axis points (not just the on-target
    /// case, where errors could cancel).
    #[test]
    fn backprojection_incremental_rotation_matches_direct_per_frequency_computation() {
        let poses = linear_aperture(Point3::new(-0.7, 0.0, 0.0), Point3::new(0.6, 0.1, 0.0), 17);
        let targets = vec![
            ScatteringTarget::new(Point3::new(0.1, 2.3, -0.2), 1.0),
            ScatteringTarget::new(Point3::new(-0.4, 1.9, 0.3), 0.6),
        ];
        let test_points = [
            Point3::new(0.1, 2.3, -0.2),  // on a target
            Point3::new(-0.4, 1.9, 0.3),  // on the other target
            Point3::new(0.0, 2.0, 0.0),   // off-target
            Point3::new(-0.55, 2.6, 0.4), // off-target, far corner
        ];
        for &(n_steps, start_hz, stop_hz) in &[(1usize, 3.0e9, 3.0e9), (2, 2.0e9, 6.0e9), (8, 1.0e9, 9.0e9), (64, 2.4e9, 2.5e9)] {
            let sweep = FrequencySweep::new(start_hz, stop_hz, n_steps);
            let measurement = simulate_measurement(&poses, &sweep, &targets, 0.0, 42);
            for point in test_points {
                let fast = focus_at_point(&measurement, &poses, &sweep, &point);
                let reference = focus_at_point_direct_reference(&measurement, &poses, &sweep, &point);
                let scale = reference.max(1e-12);
                assert!(
                    (fast - reference).abs() / scale < 1e-9,
                    "incremental rotation diverged from the direct reference at n_steps={n_steps}, point={point:?}: fast={fast}, reference={reference}"
                );
            }
        }
    }
}
