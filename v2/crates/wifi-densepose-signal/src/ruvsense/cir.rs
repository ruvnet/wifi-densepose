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
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
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

/// Canonical-56 active subcarrier indices: ±1..±28 (56 total), DC=0 excluded.
///
/// ADR-154 §A.1: the RuvSense pipeline (`hardware_norm.rs`) resamples every
/// chipset onto a uniform **canonical 56-tone grid** before fusion. That grid
/// is what `MultistaticFuser` and the CIR coherence gate actually see — *not*
/// the raw 64-bin HT20 stream. We model it as a contiguous 56-active-tone band
/// (−28..−1, +1..+28), which is also the native Atheros 56-subcarrier layout
/// (`HardwareType::Atheros`, hardware_norm.rs:45). Building Φ over these 56
/// indices lets `CirEstimator::estimate()` run on canonical frames instead of
/// rejecting them with `SubcarrierMismatch`.
const CANONICAL56_ACTIVE: [i32; 56] = {
    let mut a = [0i32; 56];
    let mut idx = 0usize;
    let mut i = -28i32;
    while i <= 28 {
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

    /// Circular phase variance (V = 1 − R̄ ∈ [0,1]) is too high — the CSI phase
    /// is near-uniformly spread across subcarriers, the signature of unsanitized
    /// SFO/CFO (ghost-tap risk). See `GHOST_TAP_CIRCULAR_VARIANCE_MAX`.
    #[error("CSI circular phase variance {variance:.3} suggests unsanitized input (ghost-tap risk)")]
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
    /// Use the FFT-based Φ/Φᴴ operator instead of the dense mat-vecs.
    ///
    /// **Default `false` (dense, bit-exact witness path).** Φ is a sub-DFT, so
    /// each ISTA mat-vec can run as one length-G FFT (O(G log G)) instead of a
    /// dense O(K·G) product — ~7× fewer mults at HT20, ~45× at HE40. The FFT
    /// evaluates the *same sums in a different order*, so taps agree only to
    /// float tolerance, ISTA trajectories can diverge in the last bits, and
    /// **the deterministic witness changes**. Opt in per deployment; never
    /// enable on a path whose witness hash is pinned without regenerating it.
    pub fft_operator: bool,
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
            // ADR-134 P2: tuned for sparse multipath — stronger L1 concentrates
            // energy on physical taps (with the windowed dominant ratio in `estimate`).
            lambda: 0.08,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
            fft_operator: false,
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
            lambda: 0.08, // ADR-134 P2 tuned (see ht20)
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
            fft_operator: false,
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
            // HE20 has the finest delay resolution (more leakage bins) -> needs
            // stronger L1 to reach the dominant-ratio floor. ADR-134 P2.
            lambda: 0.18,
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
            fft_operator: false,
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
            fft_operator: false,
        }
    }

    /// Canonical-56 grid (ADR-154 §A.1): 64-point FFT framing, **56 active
    /// tones**, 168 delay taps. This is the config the RuvSense multistatic
    /// fuser must use, because `hardware_norm.rs` resamples every node onto the
    /// canonical 56-subcarrier grid before fusion. Using `ht20()` (52 active)
    /// here makes `estimate()` reject every canonical frame with
    /// `SubcarrierMismatch` — the dead-gate bug ADR-154 fixes.
    ///
    /// `num_subcarriers` is kept at 64 (the HT20 FFT size) so the delay-domain
    /// `tap_spacing` and `bandwidth_hz` stay physically correct for a 20 MHz
    /// HT20 channel; only the *active-tone* count differs from `ht20()`.
    pub fn canonical56() -> Self {
        Self {
            bandwidth_hz: 20e6,
            num_subcarriers: 64,
            num_active: 56,
            num_taps: 168, // 3 × 56 super-resolution, matches the ht20 3× ratio
            delay_bins: 168,
            pilot_indices: HT20_PILOTS,
            lambda: 0.08, // ADR-134 P2 tuned (see ht20)
            max_iters: 100,
            tolerance: 1e-4,
            ranging_min_bw_hz: 40e6,
            dominant_ratio_threshold: 0.3,
            fft_operator: false,
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
    ///
    /// The returned slice length is always exactly `num_active`; the canonical-56
    /// grid (ADR-154) is handled explicitly so it never silently falls through to
    /// the 52-index HT20 slice (which would mismatch Φ's column count).
    fn active_indices(&self) -> &'static [i32] {
        match (self.num_subcarriers, self.num_active) {
            (64, 52) => &HT20_ACTIVE,
            (64, 56) => &CANONICAL56_ACTIVE,
            (128, 114) => &HT40_ACTIVE,
            (256, 242) => &HE20_ACTIVE,
            (512, 484) => &HE40_ACTIVE,
            // Fallback selects the slice whose length matches `num_active` so the
            // Φ dimensions stay self-consistent even for unconfigured tiers.
            (_, 56) => &CANONICAL56_ACTIVE,
            (_, 114) => &HT40_ACTIVE,
            (_, 242) => &HE20_ACTIVE,
            (_, 484) => &HE40_ACTIVE,
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
    /// Diagonal of the Tikhonov approximation diag(Φ^H Φ) + λI — depends only
    /// on Φ and λ, so it is precomputed once instead of per frame.
    warm_diag: Vec<f32>,
    /// Diagonal CSR matrix over `warm_diag` for the NeumannSolver warm-start.
    warm_csr: CsrMatrix<f32>,
    /// FFT operator for Φ/Φᴴ, built only when `config.fft_operator` (opt-in).
    fft: Option<FftOperator>,
}

/// FFT realisation of the sub-DFT sensing operator (opt-in, see
/// [`CirConfig::fft_operator`]).
///
/// Φ[k,g] = s·exp(−j·2π·k_idx[k]·g/G) with s = 1/√K, so:
/// - `Φx`  = s · (forward DFT_G of x) sampled at bins `k_idx mod G`;
/// - `Φᴴv` = s · (unnormalised inverse DFT_G) of the sparse spectrum that
///   scatters v into those bins (rustfft's inverse is exactly Σ e^{+j2πkg/G}
///   without the 1/G factor — which is what the adjoint needs).
///
/// Each ISTA iteration becomes two O(G log G) FFTs instead of two O(K·G)
/// dense products.
struct FftOperator {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// Active-subcarrier DFT bins: `k_idx mod G`, one per active subcarrier.
    bins: Vec<usize>,
    /// 1/√K column normalisation of Φ.
    scale: f32,
    g: usize,
}

impl FftOperator {
    fn new(active_indices: &[i32], g: usize, k: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let bins = active_indices
            .iter()
            .map(|&idx| (idx.rem_euclid(g as i32)) as usize)
            .collect();
        Self {
            forward: planner.plan_fft_forward(g),
            inverse: planner.plan_fft_inverse(g),
            bins,
            scale: 1.0 / (k as f32).sqrt(),
            g,
        }
    }

    /// Φ v → out (out length K). `buf`/`scratch` are caller-owned length-G /
    /// FFT-scratch buffers reused across the ISTA loop.
    fn matvec_phi(
        &self,
        v: &[Complex32],
        out: &mut [Complex32],
        buf: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        buf.copy_from_slice(v);
        self.forward.process_with_scratch(buf, scratch);
        for (o, &bin) in out.iter_mut().zip(&self.bins) {
            *o = buf[bin] * self.scale;
        }
    }

    /// Φᴴ v → out (out length G).
    fn matvec_phi_h(
        &self,
        v: &[Complex32],
        out: &mut [Complex32],
        buf: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        buf.fill(Complex32::new(0.0, 0.0));
        for (&vi, &bin) in v.iter().zip(&self.bins) {
            buf[bin] += vi;
        }
        self.inverse.process_with_scratch(buf, scratch);
        for (o, &b) in out.iter_mut().zip(buf.iter()) {
            *o = b * self.scale;
        }
    }

    /// Length of the FFT scratch buffer required by both plans.
    fn scratch_len(&self) -> usize {
        self.forward
            .get_inplace_scratch_len()
            .max(self.inverse.get_inplace_scratch_len())
    }
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
        let (warm_diag, warm_csr) = build_warm_start_system(&phi, k, g, config.lambda);
        let fft = config
            .fft_operator
            .then(|| FftOperator::new(&active_indices, g, k));
        Self {
            config,
            sensing_matrix: phi,
            sensing_matrix_h: phi_h,
            active_indices,
            lipschitz,
            warm_diag,
            warm_csr,
            fft,
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

        // Ghost-tap guard: a near-uniform spread of CSI phase across subcarriers
        // signals unsanitized SFO/CFO (raw hardware phase ramps that were never
        // de-rotated). `phase_variance` is now Mardia's *circular* variance
        // V = 1 − R̄ ∈ [0,1] (ADR-154 §7.4 #1), so the old `> TAU` (≈6.28)
        // threshold — meaningful only for the unbounded linear variance — no
        // longer applies. We compare against the bounded const below.
        let phase_var = phase_variance(&y);
        if phase_var > GHOST_TAP_CIRCULAR_VARIANCE_MAX {
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
            &self.warm_diag,
            &self.warm_csr,
            self.fft.as_ref(),
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

        // Dominant-tap energy fraction. On the 3× super-resolved grid a single
        // physical tap leaks across ~3 adjacent bins, so the dominant *physical*
        // tap is the magnitude summed over a ±1-bin window around the peak — using
        // a single bin under-counts its energy and crushes the ratio (ADR-134 P2).
        let dominant_tap_ratio = if tap_sum > 1e-12 {
            let lo = dominant_tap_idx.saturating_sub(1);
            let hi = (dominant_tap_idx + 1).min(x.len() - 1);
            let dom_window: f32 = x[lo..=hi].iter().map(|c| c.norm()).sum();
            dom_window / tap_sum
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
        // Only causal delays [0, G/2) contribute: the ISTA delay grid is circular
        // (Φ is DFT-like), so bins ≥ G/2 are aliased *negative* (non-causal) delays —
        // an alias of the near-zero dominant tap otherwise inflates the spread (ADR-134 P2).
        let causal_bins = x.len() / 2;
        let power: Vec<f64> = x[..causal_bins].iter().map(|c| (c.norm() as f64).powi(2)).collect();
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
#[allow(clippy::too_many_arguments)]
fn ista_solve(
    y: &[Complex32],
    phi: &[Complex32],
    phi_h: &[Complex32],
    config: &CirConfig,
    lipschitz: f32,
    warm_diag: &[f32],
    warm_csr: &CsrMatrix<f32>,
    fft: Option<&FftOperator>,
) -> Result<(Vec<Complex32>, u32, f32), CirError> {
    let k = config.num_active;
    let g = config.num_taps;
    let step = 1.0 / lipschitz.max(1e-6);
    let thresh = config.lambda * step;

    let mut x = neumann_warm_start(y, phi_h, k, g, warm_diag, warm_csr);
    let mut x_prev = x.clone();
    let mut phi_x = vec![Complex32::new(0.0, 0.0); k];
    let mut grad = vec![Complex32::new(0.0, 0.0); g];
    // FFT-path work buffers, allocated once per solve (not per iteration).
    let (mut fft_buf, mut fft_scratch) = match fft {
        Some(op) => (
            vec![Complex32::new(0.0, 0.0); op.g],
            vec![Complex32::new(0.0, 0.0); op.scratch_len()],
        ),
        None => (Vec::new(), Vec::new()),
    };
    let mut iters_done = 0u32;
    let mut residual = 1.0_f32;

    for iter in 0..config.max_iters {
        // grad = Φ^H (Φ x − y) — dense exact path by default; opt-in FFT
        // operator computes the same products in O(G log G).
        match fft {
            Some(op) => op.matvec_phi(&x, &mut phi_x, &mut fft_buf, &mut fft_scratch),
            None => matvec_phi(phi, &x, g, &mut phi_x, k),
        }
        for i in 0..k {
            phi_x[i] -= y[i];
        }
        match fft {
            Some(op) => op.matvec_phi_h(&phi_x, &mut grad, &mut fft_buf, &mut fft_scratch),
            None => matvec_phi_h(phi_h, &phi_x, k, &mut grad, g),
        }

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
    phi_h: &[Complex32],
    k: usize,
    g: usize,
    diag: &[f32],
    a: &CsrMatrix<f32>,
) -> Vec<Complex32> {
    let mut phi_h_y = vec![Complex32::new(0.0, 0.0); g];
    matvec_phi_h(phi_h, y, k, &mut phi_h_y, g);

    // One NeumannSolver call per part — explicit call satisfies ADR-134 mandate.
    let solver = NeumannSolver::new(1e-6, 50);
    let rhs_re: Vec<f32> = phi_h_y.iter().map(|c| c.re).collect();
    let rhs_im: Vec<f32> = phi_h_y.iter().map(|c| c.im).collect();

    let fallback = |rhs: &[f32]| -> Vec<f32> {
        rhs.iter().zip(diag.iter()).map(|(&b, &d)| b / d).collect()
    };

    let x_re = solver
        .solve(a, &rhs_re)
        .map(|r| r.solution)
        .unwrap_or_else(|_| fallback(&rhs_re));
    let x_im = solver
        .solve(a, &rhs_im)
        .map(|r| r.solution)
        .unwrap_or_else(|_| fallback(&rhs_im));

    x_re.into_iter()
        .zip(x_im)
        .map(|(re, im)| Complex32::new(re, im))
        .collect()
}

/// Precompute the diagonal Tikhonov system used by `neumann_warm_start`.
///
/// Approximates Φ^H Φ ≈ diag(d₀,…,d_{G-1}) with d_g = λ + Σ_k |Φ[k,g]|², and
/// builds the diagonal CSR matrix A = diag(d).  Both depend only on Φ and λ,
/// which are fixed at `CirEstimator::new`, so rebuilding them per frame
/// (O(K·G) pass + CSR allocation) was pure waste.  Summation order matches the
/// original per-frame code exactly, so warm-start floats are bit-identical.
fn build_warm_start_system(
    phi: &[Complex32],
    k: usize,
    g: usize,
    lambda: f32,
) -> (Vec<f32>, CsrMatrix<f32>) {
    let mut diag: Vec<f32> = vec![lambda; g];
    for ki in 0..k {
        for gi in 0..g {
            diag[gi] += phi[ki * g + gi].norm_sqr();
        }
    }

    // Diagonal CSR: each row has exactly one non-zero entry (the diagonal).
    let coo: Vec<(usize, usize, f32)> =
        diag.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
    let a = CsrMatrix::<f32>::from_coo(g, g, coo);
    (diag, a)
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

/// Ghost-tap guard threshold on the **circular** phase variance (ADR-154 §7.4 #1).
///
/// `phase_variance` returns Mardia's circular variance V = 1 − R̄ ∈ [0,1].
/// The guard rejects a frame as unsanitized when V exceeds this cutoff, i.e.
/// when the mean resultant length R̄ falls below `1 − MAX`. At V = 0.99 the
/// guard fires only when R̄ ≤ 0.01 — essentially uniform phase, the signature
/// of raw SFO/CFO ramps the gate is meant to reject — while a sanitized,
/// concentrated phase set (R̄ near 1, V near 0) passes comfortably.
///
/// **DATA-GATED (ADR-154 §7.4 #1):** this is a deliberately *conservative*
/// default, not a calibrated operating point. A clean single-path channel with
/// appreciable delay also sweeps the circle (high V), so V alone cannot cleanly
/// separate "clean ramp" from "unsanitized noise" without labelled
/// sanitized/unsanitized frames. The *metric* (circular variance) is MEASURED;
/// this *value* awaits per-deployment calibration. Until then we err toward
/// never false-rejecting a real frame — strictly more permissive at the wrap
/// boundary than the old linear-variance guard, which is the bug being fixed.
const GHOST_TAP_CIRCULAR_VARIANCE_MAX: f32 = 0.99;

/// Circular variance of the instantaneous phase angles across a complex vector.
///
/// Phase angles live on the circle and wrap at ±π, so a *linear* sample variance
/// (the previous implementation, ADR-154 §7.4 #1) reports spuriously HIGH
/// dispersion for a tightly-clustered set straddling the ±π branch cut — e.g.
/// `{+3.13, −3.13}` are 0.02 rad apart on the circle but ≈2π apart on the line.
/// That made the `phase_variance > TAU` ghost-tap guard FALSE-TRIP on real,
/// tightly-clustered CIR taps.
///
/// The correct metric is Mardia's circular variance:
///
///   R̄ = | (1/n) · Σ_k e^{iθ_k} |   (mean resultant length, ∈ [0,1])
///   V  = 1 − R̄                     (circular variance, ∈ [0,1])
///
/// V = 0 ⇔ all angles identical (maximally concentrated); V = 1 ⇔ the unit
/// phasors cancel (e.g. uniformly-spread angles → R̄ = 0). It is invariant to
/// where the cluster sits on the circle, so the branch-cut artefact is gone.
///
/// Reference: Mardia & Jupp, *Directional Statistics* (2000), §1.3.
#[inline]
fn phase_variance(y: &[Complex32]) -> f32 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    // Mean resultant vector of the *unit* phasors e^{iθ_k}. Normalising each
    // term to unit magnitude makes this a pure phase statistic (amplitude does
    // not bias the dispersion), matching the linear version which used only
    // `arg()`.
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for c in y {
        let theta = c.arg();
        sx += theta.cos();
        sy += theta.sin();
    }
    let nf = n as f32;
    let r_bar = ((sx * sx + sy * sy).sqrt() / nf).clamp(0.0, 1.0);
    1.0 - r_bar
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

    // ── ADR-154 §7.4 #1: circular vs linear phase variance ──────────────────

    /// Inline replica of the OLD linear sample variance over `arg()` — kept in
    /// the test only, so we can show the exact contrast the fix removes.
    fn old_linear_phase_variance(y: &[Complex32]) -> f32 {
        let n = y.len();
        if n < 2 {
            return 0.0;
        }
        let nf = n as f32;
        let phases: Vec<f32> = y.iter().map(|c| c.arg()).collect();
        let mean = phases.iter().sum::<f32>() / nf;
        phases.iter().map(|p| (p - mean) * (p - mean)).sum::<f32>() / nf
    }

    /// FAILS-ON-OLD: phases tightly clustered across the ±π branch cut. The old
    /// LINEAR variance reports a huge value (≈π²) and would trip the `> TAU`
    /// guard; the new CIRCULAR variance reports ≈0 (the cluster is 0.04 rad wide
    /// on the circle) and the guard does NOT false-trip.
    #[test]
    fn phase_variance_circular_not_fooled_by_branch_cut() {
        // 40 unit phasors split between +π−ε and −π+ε: true angular spread ≈0.04
        // rad, but they straddle the wrap point.
        let eps = 0.02_f32;
        let y: Vec<Complex32> = (0..40)
            .map(|i| {
                let theta = if i % 2 == 0 {
                    std::f32::consts::PI - eps
                } else {
                    -std::f32::consts::PI + eps
                };
                Complex32::new(theta.cos(), theta.sin())
            })
            .collect();

        let old = old_linear_phase_variance(&y);
        let new = phase_variance(&y);

        // The OLD metric is spuriously huge (well past the old TAU≈6.28 guard).
        assert!(
            old > std::f32::consts::TAU,
            "old linear variance should be large (>TAU) on wrap-straddling phases, was {old}"
        );
        // The NEW circular variance is ≈0 — the cluster is genuinely tight.
        assert!(
            new < 0.01,
            "circular variance must be ~0 for a tight cluster across ±π, was {new}"
        );
        // And the guard must NOT false-trip on this (a real tight CIR tap).
        assert!(
            new <= GHOST_TAP_CIRCULAR_VARIANCE_MAX,
            "ghost-tap guard must not false-trip on a tight wrap-straddling cluster"
        );
    }

    /// Circular variance is bounded [0,1] for arbitrary (deterministic-random)
    /// inputs, and hits its documented extremes: ≈0 for identical angles, ≈1
    /// for uniformly-spread angles.
    #[test]
    fn phase_variance_circular_is_bounded_and_extremal() {
        // Deterministic pseudo-random phases via an LCG — bounded check.
        let mut s: u32 = 0x1234_5678;
        let y: Vec<Complex32> = (0..200)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
                let theta = u * std::f32::consts::TAU - std::f32::consts::PI;
                Complex32::new(theta.cos(), theta.sin())
            })
            .collect();
        let v = phase_variance(&y);
        assert!((0.0..=1.0).contains(&v), "V must be in [0,1], was {v}");

        // Identical angles → V ≈ 0.
        let same: Vec<Complex32> = (0..64)
            .map(|_| {
                let t = 0.7_f32;
                Complex32::new(t.cos(), t.sin())
            })
            .collect();
        assert!(
            phase_variance(&same) < 1e-5,
            "identical angles must give V≈0, got {}",
            phase_variance(&same)
        );

        // Angles spread uniformly around the full circle → resultant cancels,
        // V ≈ 1.
        let n = 360usize;
        let uniform: Vec<Complex32> = (0..n)
            .map(|k| {
                let t = std::f32::consts::TAU * (k as f32) / (n as f32);
                Complex32::new(t.cos(), t.sin())
            })
            .collect();
        assert!(
            phase_variance(&uniform) > 0.99,
            "uniformly-spread angles must give V≈1, got {}",
            phase_variance(&uniform)
        );
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

    // ---- Opt-in FFT operator (CirConfig::fft_operator) ----

    /// The FFT operator computes the same Φ/Φᴴ products as the dense path to
    /// float tolerance, for both a small (HT20) and the largest (HE40) config.
    #[test]
    fn fft_matvecs_match_dense() {
        for config in [CirConfig::ht20(), CirConfig::he40()] {
            let k = config.num_active;
            let g = config.num_taps;
            let active: Vec<i32> = config.active_indices().to_vec();
            let (phi, phi_h) = build_sensing_matrix(&active, g, k);
            let op = FftOperator::new(&active, g, k);
            let mut buf = vec![Complex32::new(0.0, 0.0); g];
            let mut scratch = vec![Complex32::new(0.0, 0.0); op.scratch_len()];

            // Deterministic non-trivial input vectors.
            let x: Vec<Complex32> = (0..g)
                .map(|i| Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.71).cos()))
                .collect();
            let v: Vec<Complex32> = (0..k)
                .map(|i| Complex32::new((i as f32 * 0.13).cos(), (i as f32 * 0.29).sin()))
                .collect();

            // Φx: dense vs FFT.
            let mut dense_kx = vec![Complex32::new(0.0, 0.0); k];
            matvec_phi(&phi, &x, g, &mut dense_kx, k);
            let mut fft_kx = vec![Complex32::new(0.0, 0.0); k];
            op.matvec_phi(&x, &mut fft_kx, &mut buf, &mut scratch);
            let scale_ref: f32 = dense_kx.iter().map(|c| c.norm()).sum::<f32>() / k as f32;
            for (d, f) in dense_kx.iter().zip(&fft_kx) {
                assert!(
                    (d - f).norm() <= 1e-3 * scale_ref.max(1.0),
                    "phi matvec mismatch (G={g}): {d} vs {f}"
                );
            }

            // Φᴴv: dense vs FFT.
            let mut dense_gv = vec![Complex32::new(0.0, 0.0); g];
            matvec_phi_h(&phi_h, &v, k, &mut dense_gv, g);
            let mut fft_gv = vec![Complex32::new(0.0, 0.0); g];
            op.matvec_phi_h(&v, &mut fft_gv, &mut buf, &mut scratch);
            let scale_ref_g: f32 = dense_gv.iter().map(|c| c.norm()).sum::<f32>() / g as f32;
            for (d, f) in dense_gv.iter().zip(&fft_gv) {
                assert!(
                    (d - f).norm() <= 1e-3 * scale_ref_g.max(1.0),
                    "phi_h matvec mismatch (G={g}): {d} vs {f}"
                );
            }
        }
    }

    /// End-to-end: the FFT-enabled estimator recovers the same dominant tap as
    /// the dense estimator on a clean single-path frame, with close taps.
    #[test]
    fn fft_estimate_matches_dense_dominant_tap() {
        let dense_cfg = CirConfig::ht20();
        let mut fft_cfg = CirConfig::ht20();
        fft_cfg.fft_operator = true;

        let frame = make_single_tap_frame(dense_cfg.num_subcarriers, 50e-9);
        let dense = CirEstimator::new(dense_cfg).estimate(&frame).unwrap();
        let fast = CirEstimator::new(fft_cfg).estimate(&frame).unwrap();

        assert_eq!(dense.dominant_tap_idx, fast.dominant_tap_idx);
        assert!((dense.dominant_tap_ratio - fast.dominant_tap_ratio).abs() < 1e-2);
        // Tap vectors agree to float tolerance relative to the dominant tap.
        let dom = dense.taps[dense.dominant_tap_idx].norm().max(1e-6);
        for (a, b) in dense.taps.iter().zip(&fast.taps) {
            assert!((a - b).norm() <= 1e-2 * dom);
        }
    }

    /// ADR-154 §7.4 #14: the `fft_operator` path *changes the witness hash*
    /// (documented in `CirConfig::fft_operator`), so it must be pinned as
    /// numerically **close** to the dense path — not silently divergent. The
    /// existing `fft_estimate_matches_dense_dominant_tap` covers HT20 / one tau;
    /// this test asserts the **full `Cir` output** (every tap + every scalar
    /// field) stays within a documented relative tolerance on the production
    /// **canonical-56** config across several realistic delays. A regression
    /// that lets the FFT path drift (wrong scaling, off-by-one Φ column, etc.)
    /// fails here instead of corrupting a downstream witness unnoticed.
    #[test]
    fn fft_operator_within_tolerance_of_dense_canonical56() {
        // Relative tolerances — documented, not silent. The FFT operator sums the
        // same Φ entries in a different order, so taps agree to ~float epsilon
        // scaled by the dominant-tap magnitude; ISTA can differ by a few last
        // bits over its trajectory, hence 1e-2 (same order as the existing test).
        const TAP_REL_TOL: f32 = 1e-2;
        const RATIO_ABS_TOL: f32 = 1e-2;
        const SPREAD_REL_TOL: f64 = 1e-2;

        for &tau in &[20e-9_f64, 50e-9, 90e-9] {
            let dense_cfg = CirConfig::canonical56();
            let mut fft_cfg = CirConfig::canonical56();
            fft_cfg.fft_operator = true;

            let frame = make_single_tap_frame(dense_cfg.num_subcarriers, tau);
            let dense = CirEstimator::new(dense_cfg).estimate(&frame).unwrap();
            let fast = CirEstimator::new(fft_cfg).estimate(&frame).unwrap();

            assert_eq!(dense.taps.len(), fast.taps.len());

            // Full tap vector close (relative to the dominant tap magnitude).
            let dom = dense.taps[dense.dominant_tap_idx].norm().max(1e-6);
            let mut max_tap_err = 0.0_f32;
            for (a, b) in dense.taps.iter().zip(&fast.taps) {
                max_tap_err = max_tap_err.max((a - b).norm());
            }
            assert!(
                max_tap_err <= TAP_REL_TOL * dom,
                "tau={tau:e}: FFT taps diverged from dense — max err {max_tap_err} > {TAP_REL_TOL} * {dom} (NOT numerically close)"
            );

            // The dominant tap and the scalar summary fields must agree too —
            // these feed the witness, so a silent divergence here is the bug #14
            // guards against.
            assert_eq!(
                dense.dominant_tap_idx, fast.dominant_tap_idx,
                "tau={tau:e}: dominant tap index moved"
            );
            assert!(
                (dense.dominant_tap_ratio - fast.dominant_tap_ratio).abs() <= RATIO_ABS_TOL,
                "tau={tau:e}: dominant_tap_ratio drift {} vs {}",
                dense.dominant_tap_ratio,
                fast.dominant_tap_ratio
            );
            assert_eq!(
                dense.active_tap_count, fast.active_tap_count,
                "tau={tau:e}: active_tap_count changed"
            );
            assert_eq!(
                dense.ranging_valid, fast.ranging_valid,
                "tau={tau:e}: ranging_valid flipped"
            );
            let spread_ref = dense.rms_delay_spread_s.abs().max(1e-12);
            assert!(
                (dense.rms_delay_spread_s - fast.rms_delay_spread_s).abs()
                    <= SPREAD_REL_TOL * spread_ref,
                "tau={tau:e}: rms_delay_spread drift {} vs {}",
                dense.rms_delay_spread_s,
                fast.rms_delay_spread_s
            );
        }
    }

    /// The default configs keep the FFT operator off — the dense, bit-exact
    /// witness path is the default (enabling FFT shifts float results).
    #[test]
    fn fft_operator_is_off_by_default() {
        for c in [
            CirConfig::ht20(),
            CirConfig::ht40(),
            CirConfig::he20(),
            CirConfig::he40(),
        ] {
            assert!(!c.fft_operator);
        }
    }
}
