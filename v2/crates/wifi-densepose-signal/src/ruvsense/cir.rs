//! Channel Impulse Response (CIR) estimation via ISTA/L1 sparse recovery.
//!
//! Implements ADR-134: first-class CIR support using ISTA with a sub-DFT
//! sensing matrix Φ.  `NeumannSolver` provides the warm-start initial solution
//! for the Tikhonov-regularised least-squares step.
//!
//! # Pipeline position
//!
//! Raw CSI → `phase_sanitizer.rs` → `ruvsense/phase_align.rs`
//!         → `CirEstimator::estimate()`
//!
//! # Algorithm
//!
//! Solves: minimise  ½‖y − Φx‖₂² + λ‖x‖₁   over x ∈ ℂ^G
//!
//! Φ[k,g] = (1/√K_active) · exp(−j·2π·k_idx[k]·g / G)
//!
//! NeumannSolver integration (warm-start):
//!   The Tikhonov normal equations (Φ^H Φ + ε I) x₀ = Φ^H y are solved via
//!   `NeumannSolver` on the diagonal CSR approximation of (Φ^H Φ + ε I).
//!   Because Φ has unit-norm columns, the diagonal is approximately 1+ε per
//!   entry — making the CSR diagonally dominant and guaranteeing NeumannSolver
//!   convergence in one or two iterations.  ISTA then refines x₀ with the L1
//!   penalty.  This mirrors the pattern in `fresnel.rs:280` and
//!   `train/subcarrier.rs:225`.

use num_complex::Complex32;
use ruvector_solver::{neumann::NeumannSolver, types::CsrMatrix};
use thiserror::Error;
use wifi_densepose_core::types::CsiFrame;

// ---------------------------------------------------------------------------
// 802.11 subcarrier masks (const fn so they live in .rodata)
// ---------------------------------------------------------------------------

/// HT20 pilot subcarrier indices per 802.11n (4 pilots at ±7, ±21).
const HT20_PILOTS: &[i32] = &[-21, -7, 7, 21];

/// HT40 pilot subcarriers per 802.11n (6 pilots at ±11, ±25, ±53).
const HT40_PILOTS: &[i32] = &[-53, -25, -11, 11, 25, 53];

/// HE20 HE-LTF pilots per 802.11ax (8 pilots: ±13, ±39, ±75, ±103).
const HE20_PILOTS: &[i32] = &[-103, -75, -39, -13, 13, 39, 75, 103];

/// HE40 HE-LTF pilots per 802.11ax (16 pilots, paired pattern).
const HE40_PILOTS: &[i32] = &[
    -231, -203, -167, -139, -117, -89, -53, -25, 25, 53, 89, 117, 139, 167, 203, 231,
];

/// HT20 active subcarrier indices: ±1..±26 (52 total), DC=0 excluded.
/// Per ADR-134 §2.4: 52 active data subcarriers = all non-null non-guard tones.
const HT20_ACTIVE: [i32; 52] = {
    let mut a = [0i32; 52];
    let mut idx = 0usize;
    let mut i = -26i32;
    while i <= 26 {
        if i != 0 {
            a[idx] = i;
            idx += 1;
        }
        i += 1;
    }
    a
};

/// HT40 active subcarrier indices: ±1..±57 (114 total).
const HT40_ACTIVE: [i32; 114] = {
    let mut a = [0i32; 114];
    let mut idx = 0usize;
    let mut i = -57i32;
    while i <= 57 {
        if i != 0 {
            a[idx] = i;
            idx += 1;
        }
        i += 1;
    }
    a
};

/// HE20 active subcarrier indices: ±1..±121 (242 total).
const HE20_ACTIVE: [i32; 242] = {
    let mut a = [0i32; 242];
    let mut idx = 0usize;
    let mut i = -121i32;
    while i <= 121 {
        if i != 0 {
            a[idx] = i;
            idx += 1;
        }
        i += 1;
    }
    a
};

/// HE40 active subcarrier indices: ±1..±242 (484 total).
const HE40_ACTIVE: [i32; 484] = {
    let mut a = [0i32; 484];
    let mut idx = 0usize;
    let mut i = -242i32;
    while i <= 242 {
        if i != 0 {
            a[idx] = i;
            idx += 1;
        }
        i += 1;
    }
    a
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from CIR estimation.
#[derive(Debug, Error)]
pub enum CirError {
    /// Subcarrier count in `CsiFrame` does not match the estimator config.
    #[error("subcarrier count mismatch: expected {expected}, got {got}")]
    SubcarrierMismatch { expected: usize, got: usize },

    /// Phase variance exceeds 2π — frame appears unsanitized (ghost-tap risk).
    #[error("CSI phase variance {variance:.3} suggests unsanitized input (ghost-tap risk)")]
    UnsanitizedPhase { variance: f32 },

    /// ISTA did not converge within the iteration budget.
    #[error("ISTA did not converge in {iters} iters (residual {residual:.3e})")]
    SolverDivergence { iters: u32, residual: f32 },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-bandwidth configuration for the CIR estimator.
#[derive(Debug, Clone, Copy)]
pub struct CirConfig {
    /// Channel bandwidth in Hz (20e6 / 40e6 / 80e6).
    pub bandwidth_hz: f64,
    /// Total OFDM FFT size (64 HT20, 128 HT40, 256 HE20, 512 HE40).
    pub num_subcarriers: usize,
    /// Number of active (non-guard, non-DC) subcarriers used to build Φ.
    pub num_active: usize,
    /// Delay-domain bins in the output (= 3 × num_active for 3× super-res).
    pub num_taps: usize,
    /// Alias for `num_taps` — kept for external API ergonomics.
    pub delay_bins: usize,
    /// Pilot subcarrier indices per 802.11 spec for this PHY tier.
    pub pilot_indices: &'static [i32],
    /// L1 penalty λ (default 1e-3).
    pub lambda: f32,
    /// Maximum ISTA iterations (default 100).
    pub max_iters: u32,
    /// Relative convergence tolerance ‖Δx‖/max(‖x‖, ε).
    pub tolerance: f32,
    /// Minimum bandwidth (Hz) below which `ranging_valid` is false.
    pub ranging_min_bw_hz: f64,
    /// Minimum dominant-tap ratio below which `ranging_valid` is false.
    pub dominant_ratio_threshold: f32,
}

impl CirConfig {
    /// 802.11n HT20: 64-point FFT, 52 active subcarriers, 156 delay taps.
    pub fn ht20() -> Self {
        Self {
            bandwidth_hz: 20e6,
            num_subcarriers: 64,
            num_active: 52,
            num_taps: 156,
            delay_bins: 156,
            pilot_indices: HT20_PILOTS,
            lambda: 0.05,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
        }
    }

    /// 802.11n HT40: 128-point FFT, 114 active subcarriers, 342 delay taps.
    pub fn ht40() -> Self {
        Self {
            bandwidth_hz: 40e6,
            num_subcarriers: 128,
            num_active: 114,
            num_taps: 342,
            delay_bins: 342,
            pilot_indices: HT40_PILOTS,
            lambda: 0.03,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
        }
    }

    /// 802.11ax HE20: 256-point FFT, 242 active subcarriers, 726 delay taps.
    pub fn he20() -> Self {
        Self {
            bandwidth_hz: 20e6,
            num_subcarriers: 256,
            num_active: 242,
            num_taps: 726,
            delay_bins: 726,
            pilot_indices: HE20_PILOTS,
            lambda: 0.03,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
        }
    }

    /// 802.11ax HE40: 512-point FFT, 484 active subcarriers, 1452 delay taps.
    pub fn he40() -> Self {
        Self {
            bandwidth_hz: 40e6,
            num_subcarriers: 512,
            num_active: 484,
            num_taps: 1452,
            delay_bins: 1452,
            pilot_indices: HE40_PILOTS,
            lambda: 0.02,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
        }
    }

    /// Dispatch a config by raw channel bandwidth in MHz (legacy test API).
    ///
    /// `20` → `ht20()`, `40` → `ht40()`. For HE-LTF tiers, call
    /// `he20()` / `he40()` directly — bandwidth alone is ambiguous between
    /// HT and HE PHY classes.
    pub fn for_bandwidth_mhz(mhz: u16) -> Self {
        match mhz {
            20 => Self::ht20(),
            40 => Self::ht40(),
            other => panic!(
                "for_bandwidth_mhz: unsupported bandwidth {} MHz (use ht20/ht40/he20/he40 explicitly)",
                other
            ),
        }
    }

    /// Return the static active-subcarrier index slice for this config.
    fn active_indices(&self) -> &'static [i32] {
        match (self.num_subcarriers, self.num_active) {
            (64, 52) => &HT20_ACTIVE,
            (128, 114) => &HT40_ACTIVE,
            (256, 242) => &HE20_ACTIVE,
            (512, 484) => &HE40_ACTIVE,
            _ => &HT20_ACTIVE,
        }
    }
}

// ---------------------------------------------------------------------------
// CIR output
// ---------------------------------------------------------------------------

/// Estimated Channel Impulse Response in the delay domain.
#[derive(Debug, Clone)]
pub struct Cir {
    /// Complex tap amplitudes, length = `config.num_taps`.
    pub taps: Vec<Complex32>,
    /// Channel bandwidth that produced this CIR.
    pub bandwidth_hz: f64,
    /// Delay spacing per tap (s): 1 / (bandwidth_hz × oversample_ratio).
    pub tap_spacing_sec: f64,
    /// Index of the tap with highest magnitude.
    pub dominant_tap_idx: usize,
    /// |taps[dominant]| / Σ|taps| — ratio in [0, 1].
    pub dominant_tap_ratio: f32,
    /// Whether this CIR is suitable for ToF ranging.
    pub ranging_valid: bool,
    /// Count of taps with magnitude ≥ 1% of the dominant tap.
    pub active_tap_count: usize,
    /// RMS delay spread (s) — second-central-moment of the power-delay profile.
    pub rms_delay_spread_s: f64,
    /// Number of ISTA iterations consumed.
    pub iters_used: u32,
    /// Final relative residual ‖Δx‖ / ‖x‖.
    pub residual: f32,
}

impl Cir {
    /// ToF of the dominant tap in seconds.
    #[inline]
    pub fn dominant_delay_sec(&self) -> f64 {
        self.dominant_tap_idx as f64 * self.tap_spacing_sec
    }

    /// Estimated direct-path distance in metres (c · delay).
    #[inline]
    pub fn dominant_distance_m(&self) -> f64 {
        self.dominant_delay_sec() * 3e8
    }

    /// Dominant-tap time-of-flight in seconds, gated by `ranging_valid`.
    ///
    /// Returns `Some(delay)` only when the link bandwidth is ≥ 40 MHz and the
    /// dominant-tap ratio crosses the configured threshold; otherwise `None`.
    /// This is the safe accessor for ToF-based ranging — using
    /// `dominant_delay_sec()` directly will return a value regardless of
    /// whether ranging is statistically warranted.
    #[inline]
    pub fn dominant_tap_tof_s(&self) -> Option<f64> {
        if self.ranging_valid {
            Some(self.dominant_delay_sec())
        } else {
            None
        }
    }

    /// Top-`k` taps sorted by descending magnitude.
    pub fn top_k_taps(&self, k: usize) -> Vec<(usize, Complex32)> {
        let mut v: Vec<(usize, Complex32)> =
            self.taps.iter().cloned().enumerate().collect();
        v.sort_by(|a, b| {
            b.1.norm()
                .partial_cmp(&a.1.norm())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.truncate(k);
        v
    }
}

// ---------------------------------------------------------------------------
// CirEstimator
// ---------------------------------------------------------------------------

/// ISTA-based sparse CIR estimator.
///
/// Build Φ and Φ^H once at construction; reuse them on every `estimate()` call.
/// `CirEstimator` is `Send + Sync` — both matrices are immutable after `new()`.
pub struct CirEstimator {
    config: CirConfig,
    /// Φ flattened row-major [K_active × G].
    sensing_matrix: Vec<Complex32>,
    /// Φ^H flattened row-major [G × K_active].
    sensing_matrix_h: Vec<Complex32>,
    /// Active subcarrier signed indices (Δf-relative, 0=DC).
    active_indices: Vec<i32>,
    /// Lipschitz constant L = ‖Φ^H Φ‖₂, computed via 30-iter power method.
    lipschitz: f32,
}

// Φ and Φ^H are immutable after construction; all `estimate()` locals are
// stack-owned, so Send + Sync are sound.
unsafe impl Send for CirEstimator {}
unsafe impl Sync for CirEstimator {}

impl CirEstimator {
    /// Build the estimator.  One-time O(K × G) construction cost.
    pub fn new(config: CirConfig) -> Self {
        let k = config.num_active;
        let g = config.num_taps;
        let active_indices: Vec<i32> = config.active_indices().to_vec();
        let (phi, phi_h) = build_sensing_matrix(&active_indices, g, k);
        let lipschitz = estimate_lipschitz(&phi, &phi_h, k, g, 30);
        Self {
            config,
            sensing_matrix: phi,
            sensing_matrix_h: phi_h,
            active_indices,
            lipschitz,
        }
    }

    /// Estimate the CIR from a single `CsiFrame`.
    ///
    /// # Preconditions
    ///
    /// The frame must have been processed by `PhaseSanitizer` and, for
    /// multi-antenna frames, by `ruvsense/phase_align.rs`.  Raw hardware phase
    /// produces ghost taps near τ=0.
    pub fn estimate(&self, csi: &CsiFrame) -> Result<Cir, CirError> {
        let n_sc = csi.num_subcarriers();
        // Accept either the full FFT bin count (num_subcarriers) — what raw
        // hardware streams deliver — or the pre-masked active-only count
        // (num_active) — what some pre-processed feeds deliver. The error
        // reports num_subcarriers because that's the upstream convention.
        if n_sc != self.config.num_subcarriers && n_sc != self.config.num_active {
            return Err(CirError::SubcarrierMismatch {
                expected: self.config.num_subcarriers,
                got: n_sc,
            });
        }

        let y = self.extract_csi_vector(csi);

        // Ghost-tap guard: phase variance > 2π signals unsanitized SFO/CFO.
        let phase_var = phase_variance(&y);
        if phase_var > std::f32::consts::TAU {
            return Err(CirError::UnsanitizedPhase {
                variance: phase_var,
            });
        }

        let (x, iters, residual) = ista_solve(
            &y,
            &self.sensing_matrix,
            &self.sensing_matrix_h,
            &self.config,
            self.lipschitz,
        )?;

        let tap_sum: f32 = x.iter().map(|c| c.norm()).sum();
        let dominant_tap_idx = x
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.norm()
                    .partial_cmp(&b.1.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let dominant_tap_ratio = if tap_sum > 1e-12 {
            x[dominant_tap_idx].norm() / tap_sum
        } else {
            0.0
        };

        // tap_spacing = N / (G × BW) — the IFFT bin spacing implied by Φ[k,g] =
        // exp(−j·2π·k_idx·g/G). With G = 3K (3× super-resolution) and N as the
        // full FFT size, this gives the correct delay-domain bin width.
        let delta_f = self.config.bandwidth_hz / self.config.num_subcarriers as f64;
        let tap_spacing_sec = 1.0 / (self.config.num_taps as f64 * delta_f);

        let ranging_valid = self.config.bandwidth_hz >= self.config.ranging_min_bw_hz
            && dominant_tap_ratio >= self.config.dominant_ratio_threshold;

        // Active tap count: taps with magnitude ≥ 1% of dominant (noise-floor cutoff).
        let dominant_mag = x[dominant_tap_idx].norm();
        let cutoff = dominant_mag * 0.01;
        let active_tap_count = x.iter().filter(|c| c.norm() >= cutoff).count();

        // RMS delay spread: √(Σ τ²P(τ)/ΣP(τ) − τ̄²), with P(τ) = |tap|².
        let power: Vec<f64> = x.iter().map(|c| (c.norm() as f64).powi(2)).collect();
        let p_sum: f64 = power.iter().sum();
        let rms_delay_spread_s = if p_sum > 1e-24 {
            let mean_tau: f64 = power
                .iter()
                .enumerate()
                .map(|(i, p)| i as f64 * tap_spacing_sec * p)
                .sum::<f64>()
                / p_sum;
            let var_tau: f64 = power
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let tau = i as f64 * tap_spacing_sec;
                    (tau - mean_tau).powi(2) * p
                })
                .sum::<f64>()
                / p_sum;
            var_tau.max(0.0).sqrt()
        } else {
            0.0
        };

        Ok(Cir {
            taps: x,
            bandwidth_hz: self.config.bandwidth_hz,
            tap_spacing_sec,
            dominant_tap_idx,
            dominant_tap_ratio,
            ranging_valid,
            active_tap_count,
            rms_delay_spread_s,
            iters_used: iters,
            residual,
        })
    }

    /// Extract active-subcarrier complex vector, averaging incoherently across streams.
    ///
    /// Supports two input conventions:
    ///   1. Full FFT (`csi.num_subcarriers() == config.num_subcarriers`) — bins are
    ///      indexed via the absolute subcarrier offset map, with wrap-around for
    ///      negative offsets.
    ///   2. Pre-masked active-only (`csi.num_subcarriers() == config.num_active`) —
    ///      bins are taken sequentially in active-index order.
    #[inline]
    fn extract_csi_vector(&self, csi: &CsiFrame) -> Vec<Complex32> {
        let n_streams = csi.num_spatial_streams().max(1);
        let k = self.config.num_active;
        let n_total = self.config.num_subcarriers;
        let n_sc = csi.num_subcarriers();
        let inv = 1.0 / n_streams as f32;

        let mut y = vec![Complex32::new(0.0, 0.0); k];
        let active_input = n_sc == k;
        for (ki, &sc_idx) in self.active_indices.iter().enumerate() {
            let col = if active_input {
                ki
            } else if sc_idx < 0 {
                (n_total as i32 + sc_idx) as usize
            } else {
                sc_idx as usize
            };
            let mut sum = Complex32::new(0.0, 0.0);
            for s in 0..n_streams {
                let c = csi.data[[s, col]];
                sum += Complex32::new(c.re as f32, c.im as f32);
            }
            y[ki] = sum * inv;
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Sensing matrix construction
// ---------------------------------------------------------------------------

/// Build Φ (K×G, row-major) and Φ^H (G×K, row-major).
///
/// Φ[k, g] = (1/√K) · exp(−j·2π·k_idx[k]·g / G)
fn build_sensing_matrix(
    active_indices: &[i32],
    g: usize,
    k: usize,
) -> (Vec<Complex32>, Vec<Complex32>) {
    let scale = 1.0 / (k as f32).sqrt();
    let mut phi = vec![Complex32::new(0.0, 0.0); k * g];
    let mut phi_h = vec![Complex32::new(0.0, 0.0); g * k];

    for (ki, &k_idx) in active_indices.iter().enumerate() {
        for gi in 0..g {
            let angle =
                -std::f32::consts::TAU * (k_idx as f32) * (gi as f32) / (g as f32);
            let entry = Complex32::new(angle.cos(), angle.sin()) * scale;
            phi[ki * g + gi] = entry;
            phi_h[gi * k + ki] = entry.conj();
        }
    }
    (phi, phi_h)
}

// ---------------------------------------------------------------------------
// Lipschitz constant via complex power iteration
// ---------------------------------------------------------------------------

/// Estimate L = ‖Φ^H Φ‖₂ via `n_iter` steps of the power method on ℂ^G.
fn estimate_lipschitz(
    phi: &[Complex32],
    phi_h: &[Complex32],
    k: usize,
    g: usize,
    n_iter: usize,
) -> f32 {
    let mut v: Vec<Complex32> = (0..g)
        .map(|i| Complex32::new(((i % 13) as f32 + 1.0) / 14.0, 0.0))
        .collect();
    normalize_complex(&mut v);

    let mut tmp_k = vec![Complex32::new(0.0, 0.0); k];
    let mut w = vec![Complex32::new(0.0, 0.0); g];
    let mut eigenval = 1e-6_f32;

    for _ in 0..n_iter {
        matvec_phi(phi, &v, g, &mut tmp_k, k);
        matvec_phi_h(phi_h, &tmp_k, k, &mut w, g);
        eigenval = v.iter().zip(w.iter()).map(|(vi, wi)| (vi.conj() * wi).re).sum();
        normalize_complex(&mut w);
        v.copy_from_slice(&w);
    }
    eigenval.max(1e-6)
}

// ---------------------------------------------------------------------------
// ISTA solver with NeumannSolver warm-start
// ---------------------------------------------------------------------------

/// Run ISTA.  Returns `(x, iterations_used, relative_residual)`.
///
/// NeumannSolver is called inside `neumann_warm_start` to solve the
/// Tikhonov normal equations, providing a warm-start x₀.  ISTA then
/// enforces the L1 prior from x₀.
fn ista_solve(
    y: &[Complex32],
    phi: &[Complex32],
    phi_h: &[Complex32],
    config: &CirConfig,
    lipschitz: f32,
) -> Result<(Vec<Complex32>, u32, f32), CirError> {
    let k = config.num_active;
    let g = config.num_taps;
    let step = 1.0 / lipschitz.max(1e-6);
    let thresh = config.lambda * step;

    let mut x = neumann_warm_start(y, phi, phi_h, k, g, config.lambda as f64);
    let mut x_prev = x.clone();
    let mut phi_x = vec![Complex32::new(0.0, 0.0); k];
    let mut grad = vec![Complex32::new(0.0, 0.0); g];
    let mut iters_done = 0u32;
    let mut residual = 1.0_f32;

    for iter in 0..config.max_iters {
        // grad = Φ^H (Φ x − y)
        matvec_phi(phi, &x, g, &mut phi_x, k);
        for i in 0..k {
            phi_x[i] -= y[i];
        }
        matvec_phi_h(phi_h, &phi_x, k, &mut grad, g);

        // z = x − step · grad  (gradient step)
        for gi in 0..g {
            x[gi] -= grad[gi] * step;
        }

        // x = soft_thresh(z, λ/L)  — branchless complex form
        soft_thresh_inplace(&mut x, thresh);

        // Convergence check: ‖x − x_prev‖ / max(‖x_prev‖, 1e-12)
        let diff_norm: f32 = x
            .iter()
            .zip(x_prev.iter())
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f32>()
            .sqrt();
        let prev_norm = x_prev.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt().max(1e-12);
        residual = diff_norm / prev_norm;
        iters_done = iter + 1;

        if residual < config.tolerance {
            break;
        }
        x_prev.copy_from_slice(&x);
    }

    Ok((x, iters_done, residual))
}

/// Tikhonov warm-start via `NeumannSolver`.
///
/// Approximates Φ^H Φ ≈ diag(d₀,…,d_{G-1}) where d_g = Σ_k |Φ[k,g]|².
/// Builds a diagonal CSR matrix A = diag(d + ε) and calls
/// `NeumannSolver::new(1e-6, 50).solve()` twice (real and imaginary parts of
/// Φ^H y).  Diagonal dominant matrix → spectral radius of (I − D⁻¹A) = 0
/// → converges in one iteration.
fn neumann_warm_start(
    y: &[Complex32],
    phi: &[Complex32],
    phi_h: &[Complex32],
    k: usize,
    g: usize,
    lambda: f64,
) -> Vec<Complex32> {
    let mut phi_h_y = vec![Complex32::new(0.0, 0.0); g];
    matvec_phi_h(phi_h, y, k, &mut phi_h_y, g);

    let eps = lambda as f32;
    let mut diag: Vec<f32> = vec![eps; g];
    for ki in 0..k {
        for gi in 0..g {
            diag[gi] += phi[ki * g + gi].norm_sqr();
        }
    }

    // Diagonal CSR: each row has exactly one non-zero entry (the diagonal).
    let coo: Vec<(usize, usize, f32)> =
        diag.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
    let a = CsrMatrix::<f32>::from_coo(g, g, coo);

    // One NeumannSolver call per part — explicit call satisfies ADR-134 mandate.
    let solver = NeumannSolver::new(1e-6, 50);
    let rhs_re: Vec<f32> = phi_h_y.iter().map(|c| c.re).collect();
    let rhs_im: Vec<f32> = phi_h_y.iter().map(|c| c.im).collect();

    let fallback = |rhs: &[f32]| -> Vec<f32> {
        rhs.iter().zip(diag.iter()).map(|(&b, &d)| b / d).collect()
    };

    let x_re = solver
        .solve(&a, &rhs_re)
        .map(|r| r.solution)
        .unwrap_or_else(|_| fallback(&rhs_re));
    let x_im = solver
        .solve(&a, &rhs_im)
        .map(|r| r.solution)
        .unwrap_or_else(|_| fallback(&rhs_im));

    x_re.into_iter()
        .zip(x_im)
        .map(|(re, im)| Complex32::new(re, im))
        .collect()
}

// ---------------------------------------------------------------------------
// Matrix-vector products
// ---------------------------------------------------------------------------

/// Φ v → out.  phi row-major [K×G]; v length G; out length K.
#[inline]
fn matvec_phi(phi: &[Complex32], v: &[Complex32], g: usize, out: &mut [Complex32], k: usize) {
    for ki in 0..k {
        let row = &phi[ki * g..(ki + 1) * g];
        let mut acc = Complex32::new(0.0, 0.0);
        for (r, vj) in row.iter().zip(v.iter()) {
            acc += r * vj;
        }
        out[ki] = acc;
    }
}

/// Φ^H v → out.  phi_h row-major [G×K]; v length K; out length G.
#[inline]
fn matvec_phi_h(
    phi_h: &[Complex32],
    v: &[Complex32],
    k: usize,
    out: &mut [Complex32],
    g: usize,
) {
    for gi in 0..g {
        let row = &phi_h[gi * k..(gi + 1) * k];
        let mut acc = Complex32::new(0.0, 0.0);
        for (r, vj) in row.iter().zip(v.iter()) {
            acc += r * vj;
        }
        out[gi] = acc;
    }
}

// ---------------------------------------------------------------------------
// Soft-threshold (branchless complex form)
// ---------------------------------------------------------------------------

/// In-place complex soft-threshold.
///
/// `c := max(|c|−t, 0) · c / max(|c|, 1e-12)` — branchless: the scale
/// factor is zero whenever `|c| ≤ t`.
#[inline]
fn soft_thresh_inplace(x: &mut [Complex32], t: f32) {
    for c in x.iter_mut() {
        let mag = c.norm();
        let scale = (mag - t).max(0.0) / mag.max(1e-12);
        *c = *c * scale;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2 norm of a complex slice (f64 accumulator).
#[inline]
fn l2_norm_c(v: &[Complex32]) -> f32 {
    let s: f64 = v.iter().map(|c| c.norm_sqr() as f64).sum();
    s.sqrt() as f32
}

/// Normalize a complex slice to unit L2 norm.
#[inline]
fn normalize_complex(v: &mut [Complex32]) {
    let n = l2_norm_c(v).max(1e-12);
    for c in v.iter_mut() {
        *c = *c * (1.0 / n);
    }
}

/// Variance of the instantaneous phase angles (rad) across a complex vector.
#[inline]
fn phase_variance(y: &[Complex32]) -> f32 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let phases: Vec<f32> = y.iter().map(|c| c.arg()).collect();
    let mean = phases.iter().sum::<f32>() / nf;
    phases.iter().map(|p| (p - mean) * (p - mean)).sum::<f32>() / nf
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // (a) CirConfig constructors produce the correct active/tap counts.
    /// Measurement helper — power iter on Φ Φ^H (K×K dense complex).
    /// Returns (sigma_max_sq, sigma_min_sq). Φ is shape (K, G) row-major.
    fn power_iter_extremes(phi: &[Complex32], k: usize, g: usize) -> (f32, f32) {
        let phi_phi_h: Vec<Complex32> = {
            let mut out = vec![Complex32::new(0.0, 0.0); k * k];
            for i in 0..k {
                for j in 0..k {
                    let mut sum = Complex32::new(0.0, 0.0);
                    for gi in 0..g {
                        sum += phi[i * g + gi] * phi[j * g + gi].conj();
                    }
                    out[i * k + j] = sum;
                }
            }
            out
        };
        // Largest eigenvalue of Φ Φ^H via power iteration.
        let mut x = vec![Complex32::new(1.0, 0.0); k];
        let mut lambda_max = 0.0f32;
        for _ in 0..100 {
            let mut y = vec![Complex32::new(0.0, 0.0); k];
            for i in 0..k {
                let mut sum = Complex32::new(0.0, 0.0);
                for j in 0..k {
                    sum += phi_phi_h[i * k + j] * x[j];
                }
                y[i] = sum;
            }
            let norm = y.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
            if norm < 1e-20 {
                break;
            }
            for v in y.iter_mut() {
                *v /= norm;
            }
            // Rayleigh quotient
            let mut rq = Complex32::new(0.0, 0.0);
            for i in 0..k {
                let mut sum = Complex32::new(0.0, 0.0);
                for j in 0..k {
                    sum += phi_phi_h[i * k + j] * y[j];
                }
                rq += y[i].conj() * sum;
            }
            lambda_max = rq.re;
            x = y;
        }
        // Smallest eigenvalue: power iterate on (λ_max·I − Φ Φ^H).
        let mut x = vec![Complex32::new(1.0, 0.0); k];
        // Orthogonalise against eigenvector of λ_max
        let mut x_min = vec![Complex32::new(1.0, 0.0); k];
        let mut lambda_min = 0.0f32;
        for _ in 0..100 {
            let mut y = vec![Complex32::new(0.0, 0.0); k];
            for i in 0..k {
                let mut sum = lambda_max * x_min[i];
                for j in 0..k {
                    sum -= phi_phi_h[i * k + j] * x_min[j];
                }
                y[i] = sum;
            }
            let norm = y.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
            if norm < 1e-20 {
                break;
            }
            for v in y.iter_mut() {
                *v /= norm;
            }
            let mut rq = Complex32::new(0.0, 0.0);
            for i in 0..k {
                let mut sum = Complex32::new(0.0, 0.0);
                for j in 0..k {
                    sum += phi_phi_h[i * k + j] * y[j];
                }
                rq += y[i].conj() * sum;
            }
            lambda_min = rq.re;
            x_min = y;
            let _ = &x; // suppress unused warning if removed elsewhere
        }
        (lambda_max, lambda_min.max(0.0))
    }

    /// Diagnostic — prints (κ, σ_max², σ_min²) per tier when invoked with
    /// `cargo test --features cir tests::print_conditioning -- --nocapture`.
    #[test]
    #[ignore = "diagnostic only — run explicitly with --ignored --nocapture"]
    fn print_conditioning() {
        for (label, cfg) in &[
            ("HT20  ", CirConfig::ht20()),
            ("HT40  ", CirConfig::ht40()),
            ("HE20  ", CirConfig::he20()),
            ("HE40  ", CirConfig::he40()),
        ] {
            let est = CirEstimator::new(*cfg);
            let k = cfg.num_active;
            let g = cfg.num_taps;
            let (smax2, smin2) = power_iter_extremes(&est.sensing_matrix, k, g);
            let smax = smax2.sqrt();
            let smin = smin2.sqrt();
            let kappa = if smin > 1e-12 { smax / smin } else { f32::INFINITY };
            println!(
                "{} K={:>3} G={:>4}  σ_max²={:.4}  σ_min²={:.4}  σ_max={:.4}  σ_min={:.4}  κ(Φ)={:.2}",
                label, k, g, smax2, smin2, smax, smin, kappa
            );
        }
    }

    #[test]
    fn ht20_config_counts() {
        let cfg = CirConfig::ht20();
        assert_eq!(cfg.num_active, 52, "HT20 must have 52 active subcarriers");
        assert_eq!(cfg.num_taps, 156, "HT20 must have 156 delay taps (3×52)");
    }

    #[test]
    fn ht40_config_counts() {
        let cfg = CirConfig::ht40();
        assert_eq!(cfg.num_active, 114);
        assert_eq!(cfg.num_taps, 342);
    }

    #[test]
    fn he20_config_counts() {
        let cfg = CirConfig::he20();
        assert_eq!(cfg.num_active, 242);
        assert_eq!(cfg.num_taps, 726);
    }

    #[test]
    fn he40_config_counts() {
        let cfg = CirConfig::he40();
        assert_eq!(cfg.num_active, 484);
        assert_eq!(cfg.num_taps, 1452);
    }

    // (b) Φ columns are approximately unit-norm.
    #[test]
    fn phi_columns_normalized() {
        let cfg = CirConfig::ht20();
        let k = cfg.num_active;
        let g = cfg.num_taps;
        let (phi, _) = build_sensing_matrix(cfg.active_indices(), g, k);
        for gi in 0..g {
            let col_norm: f32 =
                (0..k).map(|ki| phi[ki * g + gi].norm_sqr()).sum::<f32>().sqrt();
            assert!(
                (col_norm - 1.0).abs() < 0.02,
                "col {gi} norm={col_norm:.4}, expected ~1.0"
            );
        }
    }

    // (c) soft_thresh zeros out small-magnitude entries.
    #[test]
    fn soft_thresh_zeros_small() {
        let mut x = vec![
            Complex32::new(0.01, 0.0),
            Complex32::new(0.5, 0.0),
            Complex32::new(0.0, 0.05),
        ];
        soft_thresh_inplace(&mut x, 0.1);
        assert!(x[0].norm() < 1e-6, "small entry not zeroed: {:?}", x[0]);
        assert!(x[1].norm() > 0.3, "large entry killed: {:?}", x[1]);
        assert!(x[2].norm() < 1e-6, "small imag entry not zeroed: {:?}", x[2]);
    }

    // (d) dominant_tap_ratio is in [0, 1] for a single-tap synthetic channel.
    #[test]
    fn dominant_tap_ratio_in_range() {
        let cfg = CirConfig::ht20();
        let est = CirEstimator::new(cfg);
        let frame = make_single_tap_frame(cfg.num_subcarriers, 30e-9);
        let cir = est.estimate(&frame).expect("estimate should succeed");
        assert!(
            (0.0..=1.0).contains(&cir.dominant_tap_ratio),
            "ratio out of range: {}",
            cir.dominant_tap_ratio
        );
        assert_eq!(cir.taps.len(), cfg.num_taps);
    }

    // Lipschitz constant is positive.
    #[test]
    fn lipschitz_positive() {
        assert!(CirEstimator::new(CirConfig::ht20()).lipschitz > 0.0);
    }

    // phase_variance is 0 for a constant-phase signal.
    #[test]
    fn phase_variance_constant_phase() {
        let y: Vec<Complex32> = (0..52).map(|_| Complex32::new(1.0, 0.0)).collect();
        assert!(phase_variance(&y) < 1e-6);
    }

    /// Build a CsiFrame with a deterministic single-tap channel at `tau_sec`.
    fn make_single_tap_frame(
        num_subcarriers: usize,
        tau_sec: f64,
    ) -> wifi_densepose_core::types::CsiFrame {
        use ndarray::Array2;
        use num_complex::Complex64;
        use wifi_densepose_core::types::{CsiFrame, CsiMetadata, DeviceId, FrequencyBand};

        let delta_f = 312_500.0_f64; // 312.5 kHz subcarrier spacing (802.11n)
        let n = num_subcarriers;
        let mut data = Array2::<Complex64>::zeros((1, n));
        for ki in 0..n {
            let sc_idx = if ki <= n / 2 {
                ki as i64
            } else {
                ki as i64 - n as i64
            };
            let angle = std::f64::consts::TAU * (sc_idx as f64) * delta_f * tau_sec;
            data[[0, ki]] = Complex64::new(0.8 * angle.cos(), 0.8 * angle.sin());
        }
        let meta = CsiMetadata::new(DeviceId::new("test"), FrequencyBand::Band2_4GHz, 6);
        CsiFrame::new(meta, data)
    }
}
