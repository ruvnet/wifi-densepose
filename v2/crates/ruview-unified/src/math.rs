//! Small, dependency-free numeric kernels shared across the crate.
//!
//! Everything here is deterministic and exact enough to be tested against
//! closed forms: `erf` is Abramowitz–Stegun 7.1.26 (|ε| ≤ 1.5e-7), the DFT is
//! the O(n²) definition (n ≤ 64 throughout this crate, so an FFT dependency
//! would buy nothing), and the resampler is linear interpolation on the
//! complex plane (amplitude/phase-continuous for the small bin ratios the
//! adapters use).

use num_complex::Complex64;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Error function, Abramowitz & Stegun 7.1.26 rational approximation.
///
/// Maximum absolute error 1.5e-7 — far below the opacity resolution the
/// Gaussian gain model needs.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Numerically stable logistic sigmoid.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// In-place stable softmax.
pub fn softmax(v: &mut [f64]) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// Magnitudes of the first `k` DFT coefficients of `x` (definition-form DFT).
///
/// Used for delay-domain (across subcarriers) and Doppler-domain (across
/// snapshots) token features. `k ≤ x.len()` is enforced by the callers.
#[must_use]
pub fn dft_magnitudes(x: &[Complex64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(k);
    for bin in 0..k {
        let mut acc = Complex64::new(0.0, 0.0);
        for (t, v) in x.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (bin as f64) * (t as f64) / (n as f64);
            acc += v * Complex64::new(ang.cos(), ang.sin());
        }
        out.push(acc.norm() / n as f64);
    }
    out
}

/// Precomputed twiddle table for repeated fixed-size DFTs.
///
/// The naive [`dft_magnitudes`] recomputes `cos`/`sin` per sample; the
/// tokenizer calls the transform once per token, so the table amortizes the
/// trig. The optimization is *proven equivalent* in `tokenizer::tests` and
/// its speedup is measured in `benches/unified_bench.rs`.
pub struct DftPlan {
    n: usize,
    k: usize,
    /// Row-major `k × n` twiddles: `exp(-2πi·bin·t/n)`.
    twiddles: Vec<Complex64>,
}

impl DftPlan {
    /// Builds a plan for length-`n` inputs and `k` output bins.
    #[must_use]
    pub fn new(n: usize, k: usize) -> Self {
        let mut twiddles = Vec::with_capacity(k * n);
        for bin in 0..k {
            for t in 0..n {
                let ang = -2.0 * std::f64::consts::PI * (bin as f64) * (t as f64) / (n as f64);
                twiddles.push(Complex64::new(ang.cos(), ang.sin()));
            }
        }
        Self { n, k, twiddles }
    }

    /// DFT magnitudes via the precomputed table; identical (to f64 rounding)
    /// to [`dft_magnitudes`] on the same input.
    ///
    /// # Panics
    /// If `x.len()` differs from the planned length.
    #[must_use]
    pub fn magnitudes(&self, x: &[Complex64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n, "DftPlan length mismatch");
        let mut out = Vec::with_capacity(self.k);
        for bin in 0..self.k {
            let row = &self.twiddles[bin * self.n..(bin + 1) * self.n];
            let mut acc = Complex64::new(0.0, 0.0);
            for (v, w) in x.iter().zip(row) {
                acc += v * w;
            }
            out.push(acc.norm() / self.n as f64);
        }
        out
    }
}

/// Linear interpolation of a complex series onto `m` uniformly spaced points.
///
/// Interpolates real and imaginary parts independently — adequate for the
/// small resampling ratios (≤ 2×) the adapters perform, and exactly identity
/// when `m == x.len()`.
#[must_use]
pub fn resample_complex(x: &[Complex64], m: usize) -> Vec<Complex64> {
    let n = x.len();
    if n == m {
        return x.to_vec();
    }
    if n == 1 {
        return vec![x[0]; m];
    }
    if m == 1 {
        // `(m - 1)` would divide by zero below; a single output point is the
        // mean of the series rather than an arbitrary NaN-poisoned sample.
        let sum: Complex64 = x.iter().copied().sum();
        return vec![sum / n as f64];
    }
    let mut out = Vec::with_capacity(m);
    for j in 0..m {
        let pos = (j as f64) * ((n - 1) as f64) / ((m - 1) as f64);
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(n - 1);
        let frac = pos - i0 as f64;
        out.push(x[i0] * (1.0 - frac) + x[i1] * frac);
    }
    out
}

/// Median of a slice (copies; slices here are ≤ a few hundred elements).
#[must_use]
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Least-squares slope of `y` against index `0..n` (used to detrend the
/// linear phase ramp that sampling-time offset imprints across subcarriers).
#[must_use]
pub fn linear_slope(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, v) in y.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (v - mean_y);
        den += dx * dx;
    }
    num / den
}

/// Deterministic RNG from a u64 seed (ChaCha20, the nvsim convention).
#[must_use]
pub fn seeded_rng(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
}

/// Xavier/Glorot-uniform init for a `rows × cols` weight matrix, flattened
/// row-major. Deterministic given the RNG state.
pub fn xavier_init(rng: &mut ChaCha20Rng, rows: usize, cols: usize) -> Vec<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    (0..rows * cols).map(|_| rng.gen_range(-limit..limit)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        // erf(0)=0, erf(∞)→1, erf(1)=0.8427007929 (tabulated).
        // Tolerances are the A&S 7.1.26 approximation bound (1.5e-7), not
        // machine epsilon — at x=0 the rational polynomial leaves ~1e-9.
        assert!(erf(0.0).abs() < 2e-7);
        assert!((erf(1.0) - 0.842_700_792_9).abs() < 2e-7);
        assert!((erf(-1.0) + 0.842_700_792_9).abs() < 2e-7);
        assert!((erf(3.0) - 0.999_977_909_5).abs() < 2e-7);
    }

    #[test]
    fn sigmoid_is_stable_and_symmetric() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!((sigmoid(500.0) - 1.0).abs() < 1e-12);
        assert!(sigmoid(-500.0) >= 0.0);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dft_finds_pure_tone() {
        // x[t] = exp(2πi·3t/16) has all its energy in bin 3.
        let n = 16;
        let x: Vec<Complex64> = (0..n)
            .map(|t| {
                let ang = 2.0 * std::f64::consts::PI * 3.0 * t as f64 / n as f64;
                Complex64::new(ang.cos(), ang.sin())
            })
            .collect();
        let mags = dft_magnitudes(&x, 8);
        assert!((mags[3] - 1.0).abs() < 1e-9);
        for (i, m) in mags.iter().enumerate() {
            if i != 3 {
                assert!(*m < 1e-9, "leakage at bin {i}: {m}");
            }
        }
    }

    #[test]
    fn dft_plan_matches_naive() {
        let mut rng = seeded_rng(7);
        let x: Vec<Complex64> = (0..24)
            .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
            .collect();
        let plan = DftPlan::new(24, 10);
        let a = dft_magnitudes(&x, 10);
        let b = plan.magnitudes(&x);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-12);
        }
    }

    #[test]
    fn resample_identity_and_endpoints() {
        let x: Vec<Complex64> = (0..10).map(|i| Complex64::new(i as f64, -(i as f64))).collect();
        assert_eq!(resample_complex(&x, 10), x);
        let y = resample_complex(&x, 25);
        assert_eq!(y.len(), 25);
        assert!((y[0] - x[0]).norm() < 1e-12);
        assert!((y[24] - x[9]).norm() < 1e-12);
    }

    #[test]
    fn slope_recovers_linear_ramp() {
        let y: Vec<f64> = (0..50).map(|i| 0.37 * i as f64 + 2.0).collect();
        assert!((linear_slope(&y) - 0.37).abs() < 1e-12);
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut a = seeded_rng(42);
        let mut b = seeded_rng(42);
        let va: Vec<f64> = (0..8).map(|_| a.gen_range(-1.0..1.0)).collect();
        let vb: Vec<f64> = (0..8).map(|_| b.gen_range(-1.0..1.0)).collect();
        assert_eq!(va, vb);
    }
}
