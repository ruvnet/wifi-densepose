//! Forward measurement model: simulate the complex, stepped-frequency
//! returns a monostatic synthetic-aperture radar would record from a set
//! of point scatterers (ADR-287 §2).
//!
//! ```text
//! y_{m,k} = sum_j  sigma_j / R_{m,j}^2 * exp(-i * 4*pi * f_k * R_{m,j} / c)  +  noise
//! ```
//!
//! where `m` indexes antenna position, `k` indexes swept frequency, `j`
//! indexes point scatterer, `R_{m,j}` is the range from antenna position
//! `m` to scatterer `j`, and the `4*pi*f/c` phase term is the two-way
//! (round-trip) propagation phase for a monostatic (single antenna acting
//! as both transmitter and receiver) system. The `1/R^2` term is the
//! two-way free-space spreading loss (amplitude, not power).
//!
//! This is a deliberately simplified physical model: free-space
//! propagation only (no multipath, no per-material attenuation, no
//! antenna gain pattern). It exists to give the reconstruction algorithm
//! in [`crate::reconstruct`] a known ground truth to be checked against,
//! not to predict real hardware performance.

use crate::geometry::{AntennaPose, Point3};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::resolution::SPEED_OF_LIGHT_M_PER_S;

/// A stepped-frequency sweep: `n_steps` evenly spaced frequencies from
/// `start_hz` to `stop_hz` inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrequencySweep {
    /// Lowest swept frequency, Hz.
    pub start_hz: f64,
    /// Highest swept frequency, Hz.
    pub stop_hz: f64,
    /// Number of evenly spaced frequency steps (>= 1).
    pub n_steps: usize,
}

impl FrequencySweep {
    /// Construct a sweep. Panics if `n_steps == 0` or `stop_hz < start_hz`.
    pub fn new(start_hz: f64, stop_hz: f64, n_steps: usize) -> Self {
        assert!(n_steps > 0, "a frequency sweep needs at least one step");
        assert!(stop_hz >= start_hz, "stop_hz must be >= start_hz");
        Self { start_hz, stop_hz, n_steps }
    }

    /// The `n_steps` evenly spaced frequencies, Hz, ascending.
    pub fn frequencies(&self) -> Vec<f64> {
        if self.n_steps == 1 {
            return vec![self.start_hz];
        }
        (0..self.n_steps)
            .map(|i| {
                let t = i as f64 / (self.n_steps - 1) as f64;
                self.start_hz + (self.stop_hz - self.start_hz) * t
            })
            .collect()
    }

    /// Total swept bandwidth, Hz.
    pub fn bandwidth_hz(&self) -> f64 {
        self.stop_hz - self.start_hz
    }

    /// Center frequency, Hz.
    pub fn center_freq_hz(&self) -> f64 {
        (self.start_hz + self.stop_hz) / 2.0
    }
}

/// A single point scatterer: a location and a scalar reflectivity
/// (dimensionless; only relative magnitudes across scatterers matter).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScatteringTarget {
    /// Scatterer location, meters.
    pub position: Point3,
    /// Scalar reflectivity (>= 0 for a physical scatterer; the forward
    /// model does not enforce this so synthetic "negative" scatterers can
    /// be used to probe reconstruction linearity in tests).
    pub reflectivity: f64,
}

impl ScatteringTarget {
    /// Construct a target.
    pub fn new(position: Point3, reflectivity: f64) -> Self {
        Self { position, reflectivity }
    }
}

/// The complex, stepped-frequency measurement recorded at every
/// (antenna position, frequency) pair. Row-major: index `[m][k]` is at
/// `samples[m * n_freqs + k]`.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Number of antenna positions.
    pub n_poses: usize,
    /// Number of frequency steps.
    pub n_freqs: usize,
    /// Complex samples, row-major over (pose, frequency).
    pub samples: Vec<Complex64>,
}

impl Measurement {
    /// The complex sample at antenna position `m`, frequency index `k`.
    pub fn get(&self, m: usize, k: usize) -> Complex64 {
        self.samples[m * self.n_freqs + k]
    }
}

/// Simulate the forward measurement model for `targets` observed from
/// `poses` across the frequencies in `sweep`, with i.i.d. complex Gaussian
/// noise of standard deviation `noise_std` (per real/imaginary component)
/// added to every sample. Deterministic given `seed` (ChaCha20, the
/// nvsim/ruview-unified reproducibility commitment: same seed -> identical
/// output on every machine).
pub fn simulate_measurement(
    poses: &[AntennaPose],
    sweep: &FrequencySweep,
    targets: &[ScatteringTarget],
    noise_std: f64,
    seed: u64,
) -> Measurement {
    let freqs = sweep.frequencies();
    let n_poses = poses.len();
    let n_freqs = freqs.len();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_poses * n_freqs);

    for pose in poses {
        for &f in &freqs {
            let mut y = Complex64::new(0.0, 0.0);
            for t in targets {
                let r = pose.position.distance(&t.position);
                if r < 1e-6 {
                    // Degenerate: antenna co-located with target. Skip to
                    // avoid a divide-by-zero singularity in 1/R^2.
                    continue;
                }
                let amplitude = t.reflectivity / (r * r);
                let phase = -4.0 * PI * f * r / SPEED_OF_LIGHT_M_PER_S;
                y += Complex64::from_polar(amplitude, phase);
            }
            if noise_std > 0.0 {
                y += Complex64::new(gaussian(&mut rng, noise_std), gaussian(&mut rng, noise_std));
            }
            samples.push(y);
        }
    }

    Measurement { n_poses, n_freqs, samples }
}

/// Box-Muller transform: one N(0, std^2) sample from two uniform draws.
fn gaussian(rng: &mut ChaCha20Rng, std: f64) -> f64 {
    let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::linear_aperture;

    #[test]
    fn frequency_sweep_endpoints_and_count() {
        let sweep = FrequencySweep::new(1.0e9, 4.0e9, 5);
        let freqs = sweep.frequencies();
        assert_eq!(freqs.len(), 5);
        assert!((freqs[0] - 1.0e9).abs() < 1e-6);
        assert!((freqs[4] - 4.0e9).abs() < 1e-6);
        assert!((sweep.bandwidth_hz() - 3.0e9).abs() < 1e-6);
        assert!((sweep.center_freq_hz() - 2.5e9).abs() < 1e-6);
    }

    #[test]
    fn single_step_sweep_is_just_start_hz() {
        let sweep = FrequencySweep::new(2.0e9, 2.0e9, 1);
        assert_eq!(sweep.frequencies(), vec![2.0e9]);
    }

    #[test]
    fn simulate_measurement_is_deterministic_given_seed() {
        let poses = linear_aperture(Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0), 4);
        let sweep = FrequencySweep::new(2.0e9, 3.0e9, 8);
        let targets = vec![ScatteringTarget::new(Point3::new(0.15, 1.0, 0.0), 1.0)];
        let a = simulate_measurement(&poses, &sweep, &targets, 0.01, 42);
        let b = simulate_measurement(&poses, &sweep, &targets, 0.01, 42);
        for (sa, sb) in a.samples.iter().zip(b.samples.iter()) {
            assert_eq!(sa, sb, "same seed must give byte-identical measurements");
        }
    }

    #[test]
    fn different_seeds_give_different_noise() {
        let poses = linear_aperture(Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0), 4);
        let sweep = FrequencySweep::new(2.0e9, 3.0e9, 8);
        let targets = vec![ScatteringTarget::new(Point3::new(0.15, 1.0, 0.0), 1.0)];
        let a = simulate_measurement(&poses, &sweep, &targets, 0.05, 1);
        let b = simulate_measurement(&poses, &sweep, &targets, 0.05, 2);
        assert_ne!(a.samples, b.samples);
    }

    #[test]
    fn zero_noise_is_purely_deterministic_physics() {
        let poses = linear_aperture(Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0), 2);
        let sweep = FrequencySweep::new(2.0e9, 2.0e9, 1);
        let targets = vec![ScatteringTarget::new(Point3::new(0.0, 1.0, 0.0), 2.0)];
        let m = simulate_measurement(&poses, &sweep, &targets, 0.0, 7);
        // Antenna 0 is directly below the target at range 1.0 m.
        let r = 1.0_f64;
        let expected_amp = 2.0 / (r * r);
        assert!((m.get(0, 0).norm() - expected_amp).abs() < 1e-9);
    }

    #[test]
    fn no_noise_std_gt_zero_produces_nonzero_noise() {
        let poses = linear_aperture(Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0), 1);
        let sweep = FrequencySweep::new(2.0e9, 2.0e9, 1);
        let m = simulate_measurement(&poses, &sweep, &[], 1.0, 5);
        assert!(m.get(0, 0).norm() > 0.0, "no targets but noise_std>0 must still yield noise");
    }
}
