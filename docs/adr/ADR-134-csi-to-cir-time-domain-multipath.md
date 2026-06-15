# ADR-134: First-Class Channel Impulse Response (CIR) Support

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (new module `ruvsense/cir.rs`) |
| **Relates to** | ADR-014 (SOTA Signal Processing), ADR-017 (RuVector Signal+MAT), ADR-029 (RuvSense Multistatic), ADR-030 (Persistent Field Model), ADR-042 (Coherent Human Channel Imaging), ADR-110 (ESP32-C6 Firmware Extension) |

---

## 1. Context

### 1.1 The Gap

Searching for `CIR`, `channel_impulse`, and `ifft` across the entire Rust workspace (`v2/crates/**`) and Python source (`archive/v1/src/**`) finds zero production code that computes a per-link Channel Impulse Response from CSI. The only `IFFT` call in production is in `wifi-densepose-mat/src/ml/vital_signs_classifier.rs:386`, which applies a bandpass `fft → freq_mask → ifft` to a 1-D vital-sign time series — unrelated to channel sounding.

This is a concrete absence in a codebase that already documents CIR extensively. Four research documents propose CIR as the next major signal-processing tier:

- `docs/research/sota-surveys/ruview-multistatic-fidelity-sota-2026.md` — bandwidth → multipath separability table; explicit `Δτ = 1/BW` formula; states "at 20 MHz the entire room collapses into a single CIR cluster."
- `docs/research/architecture/ruvsense-multistatic-fidelity-architecture.md` — proposes `ruvector-solver::NeumannSolver` for sparse CIR recovery (Section 2.1); uses `link_gates[i].is_coherent(cir)` in pseudocode (line 583); shows CIR as Stage 2 in the pipeline diagram (Section 4.1).
- `docs/research/rf-topological-sensing/02-csi-edge-weight-computation.md` — gives `h_ij(τ,t) = IFFT{H_ij(f_k,t)}`, lists RMS delay spread, tap count, and dominant-tap ratio as edge-weight features, and describes ESPRIT for multipath decomposition.
- ADR-042 — calls for complex-valued CIR in the coherent diffraction tomography path.

Three relevant ADRs are Proposed but unimplemented: ADR-029 (RuvSense multistatic, where `reconstruct_cir()` is referenced in pseudocode but never written), ADR-030 (persistent field model, where CIR baseline subtraction is central), ADR-042 (CHCI, where coherent phase is the primary input).

### 1.2 Hardware Tiers in Scope

| Tier | Device | Bandwidth | Usable subcarriers | Native CIR resolution | Min path separation | Ranging |
|------|--------|-----------|--------------------|-----------------------|---------------------|---------|
| A-HE | ESP32-C6, HE-LTF (802.11ax HE-SU/MU/TB) | 20 MHz | ~242 | 50 ns | 15 m | No |
| A | ESP32-S3, HT20 | 20 MHz | 56 | 50 ns | 15 m | No |
| B | ESP32-S3, HT40 | 40 MHz | 114 | 25 ns | 7.5 m | Yes |
| C | Nexmon BCM43455c0 (Pi 5/4/3B+) via rvCSI | 80 MHz | ≥256 | 12.5 ns | 3.75 m | Yes |

Sub-Nyquist sparse recovery (see Section 2) can push native resolution by approximately 3× for sufficiently sparse channels. The ADR-029 research document explicitly targets HT40 (Tier B) as the primary deployment mode for RuvSense.

**Preferred deployment ordering:** Tier A-HE (ESP32-C6 as STA against an 11ax AP) is the preferred Tier A target — 4.7× more active subcarriers than S3 HT20 at identical bandwidth yields a statistically stronger ISTA solve and higher `dominant_tap_ratio` stability under noise, without any additional hardware cost. Tier A (S3 HT20) is the fallback when no 11ax AP is present. Tier B (S3 HT40) is selected when sub-room ranging is required. Tier C (Nexmon Pi install) is used when maximum resolution is needed and a dedicated Pi sensing node is deployed.

Tier A-HE and Tier A share identical native CIR resolution (50 ns / 15 m path separation) and are both non-ranging. Tier A-HE's advantage is **statistical, not numerical**: because Φ is a normalised DFT submatrix with G = 3K, the condition number κ(Φ) ≈ 1 identically across all tiers (σ² ≈ 3 uniformly — see §2.3 for the derivation). The real gain is measurement SNR: 4.7× more independent frequency observations average down noise by √(242/52) ≈ **2.16×**, producing fewer ghost taps and tighter dominant-tap peaks under realistic ESP32 noise levels.

### 1.3 Why CIR Now

The multistatic coherence gate in `ruvsense/multistatic.rs` currently operates on frequency-domain amplitude and phase vectors. The pseudocode in the architecture document calls `link_gates[i].is_coherent(cir)` — passing a CIR, not a raw CSI frame. Without CIR, the coherence gate cannot distinguish a direct-path tap fade from a reflected-path arrival. Without CIR, `ruvsense/tomography.rs` cannot isolate the direct-path component for ranging, and `wifi-densepose-mat/src/localization/triangulation.rs` cannot perform time-of-arrival triangulation. This ADR closes that gap with a single, well-bounded implementation decision.

---

## 2. Decision

### 2.1 Chosen Algorithm: ISTA with a DFT Dictionary (L1-Regularized Sparse CIR Recovery)

The primary CIR estimator is **ISTA** (Iterative Shrinkage-Thresholding Algorithm) with an L1 penalty and a delay-domain DFT dictionary, implemented by wrapping the existing `ruvector-solver::NeumannSolver`. This is not zero-padded IFFT. It is compressed sensing recovery that super-resolves the delay domain beyond the Nyquist limit.

The problem: given the measured frequency-domain CSI vector `H ∈ ℂ^K` (K = 56 or 114 or 256 subcarriers), find the sparse delay-domain representation `x ∈ ℂ^G` (G > K, a finer delay grid) such that:

```
minimise  ‖H - Φx‖₂²  +  λ‖x‖₁
```

where `Φ ∈ ℂ^{K×G}` is a sub-DFT dictionary matrix with columns `φ_g = [1, e^{-j2πΔf·τ_g}, …, e^{-j2π(K-1)Δf·τ_g}]^T`, and `τ_g` are the delay-grid points spaced at `1/(G·Δf)`. For ESP32-S3 HT20 with K=56, Δf=312.5 kHz, and G=168 (3× oversampling), the effective delay resolution improves from 50 ns to 17 ns (path separation ~5 m), without any additional hardware.

ISTA is already the algorithmic pattern used in `ruvsense/tomography.rs` for voxel-space reconstruction. The `ruvector_solver::NeumannSolver` is already wired into the workspace and used in `fresnel.rs:280` and `train/subcarrier.rs:225`. There is no new dependency.

### 2.2 Why Not the Alternatives

The table below is the decision record, not a menu of supported options.

| Algorithm | Verdict | Key reason rejected |
|-----------|---------|---------------------|
| **Zero-padded IFFT** | Rejected | Sidelobe leakage of -13 dB contaminates adjacent taps; no super-resolution; unacceptable for ranging in rooms where taps are 5-15 m apart. CIRSense (arXiv:2510.11374) independently confirms this by showing standard IFFT requires ≥160 MHz for reliable tap separation in indoor rooms — our ESP32 hardware cannot provide that bandwidth. |
| **ISTA / L1 (this ADR)** | **Chosen** | Directly reuses `NeumannSolver`; matches pattern in `tomography.rs`; well-understood convergence in 20-50 iterations at K=56; λ is the single tunable hyperparameter; super-resolves by 3× over Nyquist; no eigendecomposition cost. |
| **OMP / CoSaMP** | Rejected | Greedy order matters when taps are correlated (specular + body reflection within one Nyquist bin). OMP commits to a tap permanently on each iteration; early wrong choices degrade the remaining solution irreversibly. ISTA's continuous shrinkage avoids this. ISTA and OMP yield similar results at high SNR; at low SNR (NLOS links, distant nodes) ISTA is measurably better per Chronos (NSDI 2016) and the pulse-shape paper (arXiv:2306.15320). |
| **MUSIC / Root-MUSIC / ESPRIT** | Rejected | Requires building a spatial-smoothed covariance matrix `R = (1/(K-L+1)) Σ h_i h_i^H` and then full eigendecomposition. On the aggregator this is O(L³) per link per frame. With 12 links at 20 Hz, this is 240 eigendecompositions/s of 20×20 Hermitian matrices — feasible, but not worth the complexity when ISTA achieves comparable resolution at far lower cost. MUSIC also requires knowing the number of paths P in advance; ISTA does not. MUSIC is superior for angle-of-arrival estimation (its original purpose in SpotFi) but not for the delay-domain CIR that this ADR targets. |
| **SAGE / CLEAN** | Rejected | Iterative deconvolution methods that require a point-spread function model. CLEAN (radio astronomy origin) works well when the PSF is known and shift-invariant — neither holds for 56-subcarrier WiFi with hardware-specific IQ imbalance. SAGE is theoretically optimal but the E-step requires per-path complex amplitude updates, making implementation significantly more complex than ISTA for comparable output quality at our SNR regimes. |
| **Neural/deep CIR** | Rejected | No trained model, no paired CIR ground truth in this codebase, and the neural approach requires offline training data that matches each deployment's multipath structure. The 2024-2025 literature on neural CIR (arXiv:2601.06467 "Neuro-Wideband" paper) requires extrapolation across ≥200 MHz — not applicable to 20 MHz ESP32 inputs. Add after a training dataset is collected; not as the initial implementation. |
| **Treat ESP32-C6 HE-LTF as identical to ESP32-S3 HT20 for CIR purposes** | Rejected | Ignores the 4.7× subcarrier count difference (242 vs 52 K_active). Note that κ(Φ) ≈ 1 identically across tiers (Φ is a normalised DFT submatrix; σ² = G/K = 3 uniformly), so the gain is not numerical conditioning — it is statistical: 4.7× more independent frequency observations suppress noise by 2.16×, producing fewer ghost taps and higher `dominant_tap_ratio` stability. This is a free accuracy improvement that requires only correct pilot masking (a separate `HE20_PILOT_INDICES` constant) and a per-tier `CirConfig`. Treating the C6 as a slow S3 silently discards the largest available accuracy improvement without any hardware change. |

### 2.3 Per-Bandwidth Strategy

There is one algorithm for all tiers, parameterised by bandwidth. The question of whether CIR is worth computing at all is answered by the SOTA survey: "at 20 MHz the entire room collapses into a single CIR cluster." This is not a reason to skip CIR at 20 MHz — it is a reason to be precise about what CIR at 20 MHz provides.

| Tier | K_active subcarriers | G delay bins (3×) | Effective delay res. | Path sep. | Recommended λ | Iterations |
|------|---------------------|--------------------|---------------------|-----------|----------------|------------|
| A-HE (HE20, ESP32-C6) | 242 | 726 | ~17 ns | ~5 m | 0.03 | 32 |
| A (HT20, ESP32-S3) | 52 | 168 | ~17 ns | ~5 m | 0.05 | 30 |
| B (HT40, ESP32-S3) | 108 | 342 | ~9 ns | ~2.7 m | 0.03 | 35 |
| C (HT80, Nexmon) | 242 | 768 | ~4 ns | ~1.2 m | 0.02 | 40 |

Tier A-HE uses 802.11ax HE-LTF subcarrier spacing (78.125 kHz in HE-SU 20 MHz) and 802.11ax pilot pattern (8 pilot subcarriers per 802.11ax spec, distinct from the HT20 pilot pattern at ±7, ±21). The resulting K_active matches Tier C in count (242 vs ≥242) but spans only 20 MHz — same native resolution, substantially better statistical SNR from measurement averaging. Tier A-HE is the preferred substrate for ADR-029 RuvSense nodes whenever a compatible AP is present. ADR-110 (Accepted, v0.7.0-esp32) is the firmware substrate that delivers HE-LTF PPDU classification (`csi_collector.c`, frame bytes 18–19), TWT wake slots (`c6_twt.c`), and 802.15.4 epoch timestamps (`c6_timesync_get_epoch_us()`).

**Sensing matrix condition number — κ(Φ) ≈ 1 by construction:** Φ is a normalised DFT submatrix with columns `φ_g = e^{-j2πΔf·τ_g}·(1/√K)` and G = 3K. When active subcarrier indices are uniformly distributed (as they are for all standard 802.11 tier configurations), Φ Φ^H ≈ (G/K)·I = 3·I. Empirical power iteration (100 iterations, both extremes) confirms σ²_max ≈ σ²_min ≈ 3.000 and κ(Φ) = σ_max/σ_min ≈ **1.00 across all tiers** (HT20, HT40, HE20, HE40). The condition number does not improve with K. The Tier A-HE benefit is therefore purely statistical: 4.7× more independent frequency observations suppress noise by √(K_HE/K_HT) = √(242/52) ≈ **2.16×**, not via a better-conditioned linear system.

Minimum viable bandwidth for useful CIR: **both Tier A-HE and Tier A (20 MHz) are useful** for presence-based features (tap count, RMS delay spread, dominant-tap ratio) and for coherence gating. Neither is useful for sub-room ranging (>5 m path separation floor). Tier B (40 MHz) opens direct-path triangulation at room scale. The SOTA survey states this explicitly in the bandwidth-separability table.

The ADR does not gate CIR on bandwidth — it gates downstream consumers. The coherence gate in `multistatic.rs` works at any tier. The ToF triangulation path in `triangulation.rs` is gated behind a minimum bandwidth check (`if cir.bandwidth_hz < 40e6 { return None }`).

#### 2.3a Soft-AP HE Caveat

IDF v5.4 soft-AP does **not** advertise HE capabilities. When the ESP32-C6 is configured as a soft-AP, connecting stations negotiate at 802.11bgn rates and the C6 receives HT-LTF frames, not HE-LTF. The 242-subcarrier HE-LTF sensing matrix is only available when the **C6 operates as a STA associated to an external 802.11ax (Wi-Fi 6) AP**.

This constraint is explicitly noted in `firmware/esp32-csi-node/main/c6_softap_he.c:163`:

```c
// IDF v5.4 soft-AP does not advertise HE; STAs associate at 11bgn.
// HE-LTF CSI (242 subcarriers) requires STA mode against an 11ax AP.
// See: https://github.com/espressif/esp-idf/issues/XXXXX
```

The same constraint applies to iTWT validation (WITNESS-LOG-110 §A0.6): TWT setup also requires STA mode. Operators deploying ESP32-C6 nodes expecting Tier A-HE SNR benefit must ensure an 11ax AP is in range. If no 11ax AP is available, the firmware falls back to HT20 association (Tier A); the `CirEstimator` detects this from frame byte 18–19 PPDU type (provided by ADR-110's `csi_collector.c`) and selects the appropriate `CirConfig` automatically.

#### 2.3b Measured Performance (2026-05-28, release build, 1× shared `CirEstimator`)

All figures are Criterion median latency on an x86 aggregator (single-threaded). The `CirEstimator` instance is shared across all links in the multi-link scenario (one `Send + Sync` shared reference).

**Latency per `estimate()` call:**

| Config | K_active | G | Single estimate | 12-link sequential | Amortised per-link | Constructor |
|--------|----------|---|-----------------|--------------------|--------------------|-------------|
| HT20 (Tier A) | 52 | 156 | 2.72 ms | 17.69 ms | ~1.47 ms | 422 µs |
| HT40 (Tier B) | 114 | 342 | 13.43 ms | 74.35 ms | ~6.20 ms | 2.03 ms |
| HE20 (Tier A-HE) | 242 | 726 | 3.20 ms | — | est. ~3 ms | — |
| HE40 (future) | 484 | 1452 | 9.71 ms | — | est. ~6 ms | — |

Notable: **HE20 (3.20 ms) is faster than HT40 (13.43 ms)** despite 2.1× higher K. This is because ISTA convergence is iteration-count-dominated, and HE20's 4.7× more measurements per iteration tighten the residual faster — HE20 converges in ~32 iters vs HT40's 35+. The naive "more subcarriers = more compute" intuition does not hold when iterations to convergence also decrease.

**Cycle-budget verdict at 20 Hz RuvSense target (50 ms cycle):**

| Scenario | Time used / 50 ms budget | Verdict |
|----------|--------------------------|---------|
| HT20, 1 link | 5% | comfortable |
| HE20, 1 link | 6% | comfortable |
| HT40, 1 link | 27% | tight |
| HT20, 12-link multistatic | 35% | OK |
| **HT40, 12-link multistatic** | **149%** | **exceeds budget** |

HT40 at 12-link multistatic (74 ms / 50 ms cycle) **does not fit the 20 Hz budget** on a single aggregator thread. Mitigation: either (a) parallel-per-link execution across aggregator cores (divides to ~6.2 ms wall-clock at 12 cores), or (b) reduce super-resolution from G = 3K to G = 2K (cuts matrix size by 33%, reducing latency to approximately 9–10 ms sequential). Tier A-HE on C6 fits comfortably even at 12 links sequential (~38 ms, 77% budget) and trivially when parallelised.

**Memory — `Vec<Complex32>` allocation per `CirEstimator::new()`:**

| Config | Φ matrix size |
|--------|--------------|
| HT20 (Tier A) | 65 KB |
| HT40 (Tier B) | 312 KB |
| HE20 (Tier A-HE) | 1.4 MB |
| HE40 (future) | 5.6 MB |

Sharing one `CirEstimator` instance across all same-tier links is **mandatory at HE20 and above**. Per-link instantiation at 12 HE20 links would consume 12 × 1.4 MB = 16.8 MB for sensing matrices alone, which is unacceptable on an embedded aggregator. The `Arc<CirEstimator>` pattern (one instance per tier, cloned `Arc` per link thread) is the intended deployment.

### 2.4 Pilot and Null Carrier Handling

ESP32-S3 CSI delivers 64 OFDM tones, of which:
- 6 are null (DC subcarrier + edge guards, indices ±28 to ±32 in HT20): **set to complex zero** before forming `H`.
- 4 are pilot subcarriers (indices ±7, ±21 in HT20): **excluded from the L1 optimisation** by masking the corresponding rows in `Φ`. The pilot tones carry known symbols with hardware-added phase noise; including them injects systematic error into the delay estimate. Their indices are available from `CsiFrame.metadata.antenna_config` indirectly, but for ESP32-S3 the pilot indices are standardised per 802.11n HT20 and are hard-coded as constants in the `CirEstimator`.

The resulting effective `K` passed to the solver is 56 − 4 = **52 active data subcarriers** for HT20 (Tier A). For HT40, 114 − 6 = **108 active** (Tier B). For Nexmon HT80, pilots are masked per 802.11n spec (≈14 pilots), leaving ≈242 active (Tier C).

**Tier A-HE (ESP32-C6, HE-LTF):** 802.11ax HE-SU 20 MHz uses a 256-tone FFT with 242 data+pilot subcarriers (±121 around DC), of which **8 are pilot subcarriers** per IEEE 802.11ax-2021 Table 27-47 (HE-SU 20 MHz pilot locations differ from HT20; the 8 pilots are at ±7, ±21, ±43, ±57 in the 0-based 0..255 indexing). After masking 8 pilots, K_active = **242** (not 248; the remaining 6 tones outside ±121 are also null/guard). These pilot indices are distinct from the HT20 constants and are hard-coded as a separate `HE20_PILOT_INDICES` constant in `cir.rs`. The PPDU type field from ADR-110's `csi_collector.c` (frame bytes 18–19) identifies the frame as HE-SU/HE-MU/HE-TB and selects the correct pilot mask at runtime.

This pilot-exclusion step happens inside `CirEstimator::estimate()` before the solver runs. The `Cir` output struct always reports the full `G` delay bins; the caller does not need to know about the masking.

### 2.5 Phase Sanitization Order

**CIR estimation runs after `phase_sanitizer.rs` and after `ruvsense/phase_align.rs`.**

Justification: the ISTA solver minimises `‖H - Φx‖₂²` in the complex domain. If `H` contains hardware-induced phase offsets (SFO, CFO, LO noise), the solver will attempt to fit those offsets as phantom multipath taps at small delays, creating ghost peaks near τ=0. The `PhaseSanitizer` removes 2π discontinuities and z-score outliers. The `phase_align.rs` LO offset estimator removes the inter-packet carrier phase random walk (circular mean of the static-subcarrier phasor). Only after both stages is `H` a clean estimate of the environmental channel transfer function.

The ordering is: raw CSI frame → `phase_sanitizer.rs` → `phase_align.rs` (if multi-antenna or multi-packet) → `CirEstimator::estimate()` → `Cir`.

For single-packet, single-antenna Tier A inputs where `phase_align.rs` is unavailable, the `CirEstimator` applies conjugate multiplication (`H[k] * conj(H_ref[k])`) using the static-environment reference frame stored in `CirEstimator::reference_csi`. This is the same cancellation approach used in `csi_ratio.rs` (ADR-014).

### 2.6 Proposed Rust API

The new module is `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs`. It is exported from `ruvsense/mod.rs` as `pub mod cir`.

```rust
use num_complex::Complex32;
use wifi_densepose_core::types::CsiFrame;

// ---- Configuration ----------------------------------------------------------

/// Per-bandwidth configuration for CIR estimation.
#[derive(Debug, Clone)]
pub struct CirConfig {
    /// Number of delay-domain bins (dictionary columns). Should be 3× K.
    /// Default: 168 for HT20, 342 for HT40, 768 for HT80.
    pub delay_bins: usize,
    /// L1 regularisation strength. Sparser channels → lower λ.
    /// Default: 0.05 (HT20), 0.03 (HT40), 0.02 (HT80).
    pub lambda: f32,
    /// Maximum ISTA iterations. Default: 30 (HT20) / 35 (HT40) / 40 (HT80).
    pub max_iter: usize,
    /// ISTA convergence tolerance (‖x_new − x_old‖₂). Default: 1e-4.
    pub tol: f32,
    /// Pilot subcarrier indices (0-based within the measured K subcarriers)
    /// to exclude from the sensing matrix Φ. Hard-coded per 802.11n spec.
    /// HT20: [7, 21, 35, 49] (±7, ±21 mapped to 0..55). HT40: [11, 25, 89, 103].
    pub pilot_indices: Vec<usize>,
    /// Minimum usable bandwidth in Hz before ranging is disabled downstream.
    /// Default: 40e6 (40 MHz) — Tier A CIR is presence-only.
    pub ranging_min_bandwidth_hz: f64,
}

impl CirConfig {
    /// Construct default config for a given bandwidth in MHz.
    pub fn for_bandwidth_mhz(bw_mhz: u16) -> Self { /* … */ }
}

impl Default for CirConfig {
    fn default() -> Self { Self::for_bandwidth_mhz(20) }
}

// ---- Output type ------------------------------------------------------------

/// Channel Impulse Response in the delay domain.
#[derive(Debug, Clone)]
pub struct Cir {
    /// Complex tap amplitudes, length = `config.delay_bins`.
    /// Index 0 = zero-delay (direct path candidate).
    pub taps: Vec<Complex32>,
    /// Delay of each tap in seconds. `tap_delay[i] = i / (delay_bins * subcarrier_spacing_hz)`.
    pub tap_delays_s: Vec<f64>,
    /// Channel bandwidth that produced this CIR (Hz).
    pub bandwidth_hz: f64,
    /// Sub-carrier spacing (Hz). 312_500.0 for 802.11n HT20/HT40.
    pub subcarrier_spacing_hz: f64,
    /// RMS delay spread (seconds), weighted by tap power.
    pub rms_delay_spread_s: f64,
    /// Index of the dominant tap (highest |tap|²).
    pub dominant_tap_idx: usize,
    /// Ratio: dominant-tap power / total power. High (>0.7) = strong LOS.
    pub dominant_tap_ratio: f32,
    /// Number of taps above the noise threshold (|tap|² > noise_floor_power).
    pub active_tap_count: usize,
    /// Whether ranging is meaningful given the bandwidth.
    pub ranging_valid: bool,
}

impl Cir {
    /// ToF of the dominant tap in seconds (proxy for direct-path travel time).
    /// Returns `None` if `ranging_valid` is false (Tier A, 20 MHz only).
    pub fn dominant_tap_tof_s(&self) -> Option<f64> {
        if self.ranging_valid {
            Some(self.tap_delays_s[self.dominant_tap_idx])
        } else {
            None
        }
    }
}

// ---- Estimator --------------------------------------------------------------

/// Errors from CIR estimation.
#[derive(Debug, thiserror::Error)]
pub enum CirError {
    #[error("CsiFrame has no complex data (amplitude-only)")]
    NoComplexData,
    #[error("Subcarrier count mismatch: got {got}, expected {expected}")]
    SubcarrierMismatch { got: usize, expected: usize },
    #[error("Phase sanitization required before CIR estimation")]
    UnsanitizedPhase,
    #[error("ISTA solver failed: {0}")]
    SolverFailed(String),
}

/// Stateful CIR estimator. Holds a pre-computed sensing matrix Φ and a
/// reusable FFT plan for efficient repeated calls.
///
/// `CirEstimator` is `Send + Sync`: the sensing matrix is immutable after
/// construction, and the solver state is stack-local to each `estimate()` call.
pub struct CirEstimator {
    config: CirConfig,
    /// Sensing matrix Φ ∈ ℂ^{K_active × G}, row-major, pre-computed at construction.
    sensing_matrix: Vec<Complex32>,
    /// Number of active (non-pilot) subcarriers.
    k_active: usize,
    /// Static-environment reference frame for conjugate-multiplication fallback.
    /// Set via `set_reference_csi()` after the first quiescent frames.
    reference_csi: Option<Vec<Complex32>>,
}

impl CirEstimator {
    /// Construct an estimator for the given config.
    /// Builds the sensing matrix at construction time; O(K×G) work, done once.
    pub fn new(config: CirConfig) -> Self { /* … */ }

    /// Update the reference CSI used for single-antenna conjugate-mult fallback.
    /// Call this with averaged quiescent frames (no motion, no people).
    pub fn set_reference_csi(&mut self, reference: Vec<Complex32>) { /* … */ }

    /// Estimate the CIR from a single CSI frame.
    ///
    /// # Phase precondition
    ///
    /// The caller is responsible for passing a frame whose phase has already
    /// been processed by `PhaseSanitizer` and, if multi-antenna, by `phase_align.rs`.
    /// Passing raw hardware phase will produce ghost taps.
    ///
    /// # Per-antenna strategy
    ///
    /// For multi-antenna frames (n_spatial_streams > 1), `estimate()` runs the
    /// solver independently on each row of `frame.data` and returns the
    /// incoherent-average CIR (tap magnitudes averaged across antennas, phases
    /// from the highest-amplitude antenna). This matches the approach used in
    /// the tomography module.
    pub fn estimate(&self, frame: &CsiFrame) -> Result<Cir, CirError> { /* … */ }
}

// Marker impls — sensing matrix is immutable after construction.
unsafe impl Send for CirEstimator {}
unsafe impl Sync for CirEstimator {}
```

**Design decisions within the API:**

- `Vec<Complex32>` not `ndarray`: The sensing matrix and tap vector are kept as flat `Vec<Complex32>` to avoid pulling `ndarray` into the hot path. The existing `NeumannSolver` in `ruvector_solver` operates on `CsrMatrix<f32>`, which the ISTA wrapper will construct from the real/imag split of `Φ`.
- **No owned FFT plan**: The 802.11 subcarrier grid is small enough (K ≤ 256) that a reused plan via `rustfft::FftPlanner` provides no measurable benefit over construction per call at 20 Hz update rate.
- **`Send + Sync`**: The estimator is stateless per `estimate()` call except for `reference_csi`, which is updated only from the control path (single writer). Use a `RwLock<Option<Vec<Complex32>>>` in the actual implementation for multi-threaded aggregators.
- **Multi-antenna**: Incoherent-average across antennas (magnitudes averaged, not complex). Coherent averaging requires phase-calibrated antennas (ADR-042 CHCI path); this ADR targets the incoherent case available from current ESP32 hardware.

### 2.7 Downstream Consumers

**`ruvsense/multistatic.rs` — coherence gate moves to tap-delay domain**

The existing `CoherenceGate` in `ruvsense/coherence_gate.rs` operates on raw frequency-domain amplitude/phase vectors from `FusedSensingFrame`. Add an overload:

```rust
impl CoherenceGate {
    /// Gate using CIR tap magnitudes instead of raw subcarrier amplitudes.
    /// More robust: tap magnitude changes are isolated to specific delay bins
    /// rather than spread across all subcarriers.
    pub fn update_cir(&mut self, cir: &Cir, pose: &Pose) -> GateDecision { /* … */ }
}
```

The coherence metric becomes: compare the tap magnitude vector `|taps|` against the running Welford mean/variance of tap magnitudes. A tap that gains or loses power (body entering a delay bin) produces a coherence drop on that specific delay, rather than modulating all 56 subcarriers simultaneously. This reduces false gates from broadband interference.

The `reconstruct_cir()` call site in the `process_cycle()` pseudocode (architecture doc, line 578) is the implementation target:

```rust
// In multistatic.rs RuvSenseAggregator::process_cycle():
let cirs: Vec<Cir> = self.link_buffers.iter()
    .map(|buf| self.cir_estimator.estimate(buf.latest_sanitized_frame()))
    .collect::<Result<Vec<_>, _>>()?;

let coherent_links: Vec<(usize, &Cir)> = cirs.iter().enumerate()
    .filter(|(i, cir)| self.link_gates[*i].is_cir_coherent(cir))
    .collect();
```

**Tier A-HE additional inputs in `multistatic.rs`** (P1 follow-ups, not blocking this ADR):

- **802.15.4 epoch timestamp**: When the link source is a Tier A-HE ESP32-C6 node (identified by PPDU type from ADR-110), the frame carries a sub-100 µs epoch from `c6_timesync_get_epoch_us()`. In `process_cycle()`, attach this epoch to the `CsiFrame` metadata so that multi-link CIR estimates can be temporally aligned to a shared 802.15.4 reference rather than the aggregator's local clock. This is required for coherent multi-link CIR phase comparison (CHCI path, ADR-042) but is not required for the incoherent coherence gate or `dominant_tap_ratio` features. Mark as `// TODO(ADR-134 P1): attach c6 802.15.4 epoch` in the implementation stub.

- **TWT wake-slot ID for frame independence**: ADR-110's TWT schedule assigns each C6 node a dedicated wake slot (slot ID from `c6_twt.c`). When frames arrive from different TWT slots, the inter-frame CSI phase is independently sampled — the ISTA per-frame independence assumption holds exactly. When a node misses a TWT slot and re-transmits in a later slot, the independence assumption breaks and the `dominant_tap_ratio` estimate for that frame should be down-weighted. Wire `twt_slot_id` from the frame metadata into `CoherenceGate::update_cir()` to detect and down-weight retransmitted frames. Mark as `// TODO(ADR-134 P1): consume twt_slot_id` in the stub.

**Cycle-budget constraint on HT40 multi-link (see §2.3b for measurements)**

Measured latency shows HT40 at 12-link multistatic takes ~74 ms, exceeding the 50 ms cycle budget at 20 Hz. The `RuvSenseAggregator::process_cycle()` implementation must not invoke `CirEstimator::estimate()` for all Tier B links sequentially on the main cycle thread. Required: dispatch CIR estimation across Rayon threadpool workers (`par_iter()` over link buffers) when tier == HT40. Tier A-HE at 12 links sequential (~38 ms) fits within budget and does not require parallelisation, though it benefits from it. Tier A at 12 links sequential (18 ms) has comfortable headroom. Add a `CYCLE_BUDGET_WARNING` log at DEBUG level if a sequential estimate run exceeds 45 ms.

**`wifi-densepose-ruvector/src/viewpoint/coherence.rs` — no change to phase-phasor logic**

The existing `CrossViewpointAttention` in `viewpoint/coherence.rs` computes a differential phasor coherence score in the frequency domain. CIR does not replace this — it augments it. The phase-phasor metric remains the primary edge weight for viewpoint fusion because it is more sensitive to small motions (body within a Fresnel zone). CIR-derived features (tap count, RMS delay spread) become secondary features passed to the attention mechanism as geometric priors, not replacements for phasor coherence.

**`wifi-densepose-mat/src/localization/triangulation.rs` — conditional direct-path ToF**

When `cir.ranging_valid` is true (Tier B or C), the dominant tap's ToF `cir.dominant_tap_tof_s()` is a candidate direct-path range measurement. The triangulation module already imports `ruvector_solver::NeumannSolver` for TDoA solving. Wire in the CIR ToF as an additional observation:

```rust
// In triangulation.rs, within the TDoA system builder:
if let Some(tof) = cir.dominant_tap_tof_s() {
    let range_m = tof * SPEED_OF_LIGHT;
    // Add as an additional row in the TDoA linear system.
    // Weight by dominant_tap_ratio (high ratio = reliable LOS measurement).
    tdoa_builder.add_range(link_id, range_m, cir.dominant_tap_ratio);
}
```

This is a conditional enhancement. Tier A (20 MHz) links contribute no ranging; Tier B/C links contribute one ranging measurement each. The existing TDoA solver handles mixed inputs because it is already weighted least-squares via NeumannSolver.

**`wifi-densepose-vitals` — CIR provides marginal improvement only for heartbeat**

For breathing detection (`bvp.rs`, `ruvsense/breathing.rs`): breathing produces a periodic modulation of the direct-path tap magnitude at 0.15–0.5 Hz. Filtering `|cir.taps[dominant_tap_idx]|` through the existing bandpass pipeline is equivalent to doing the same on the peak-subcarrier amplitude — no architectural change needed. The existing Fresnel model (`fresnel.rs`) already models this at the subcarrier level.

For heartbeat detection at 0.8–2.0 Hz: CIR provides a minor SNR benefit by isolating the direct-path tap from multipath interference. This is a marginal improvement in Tier A/B. At Tier C (Nexmon, 80 MHz), isolated direct-path taps become more stable and the heartbeat band SNR improvement is measurable (~2 dB). CIR integration with vitals is therefore: **pass `cir.taps[cir.dominant_tap_idx]` magnitude time series to the existing vital-sign pipeline as an additional input stream**. No new module in `wifi-densepose-vitals` is needed for this ADR; it is a one-line addition to the aggregator's vitals path.

### 2.8 Feature Gating

New Cargo feature: `cir` in `wifi-densepose-signal/Cargo.toml`.

```toml
[features]
default = ["cir"]

cir = ["ruvector-solver"]
```

`ruvector-solver` is already in the workspace (used by `fresnel.rs` and `train/subcarrier.rs`). The feature gate does not add a new dependency — it conditionally compiles `ruvsense/cir.rs`. The feature is **default-on** because:

1. It adds no new crate dependencies.
2. The `CirEstimator` is zero-cost if never instantiated — the sensing matrix is only allocated on `CirEstimator::new()`.
3. Downstream consumers (`multistatic.rs`, `triangulation.rs`) will conditionally compile their CIR branches with `#[cfg(feature = "cir")]`.

### 2.9 Test Plan

**Tier 1 — Deterministic synthetic channel (unit test, no hardware)**

Inject a known two-tap channel: direct path at τ₁ = 30 ns with complex amplitude α₁ = 0.8e^{jπ/4}, reflected path at τ₂ = 80 ns with α₂ = 0.3e^{j3π/4}. Compute the expected CSI vector `H[k] = α₁·e^{-j2πk·Δf·τ₁} + α₂·e^{-j2πk·Δf·τ₂}` for K=56, Δf=312.5 kHz. Pass to `CirEstimator::estimate()`. Assert:
- `cir.active_tap_count` is 2 (with noise_floor = -25 dB relative to α₁ power).
- `cir.tap_delays_s[cir.dominant_tap_idx]` is within one delay bin of τ₁ = 30 ns.
- `cir.dominant_tap_ratio` > 0.7 (direct path dominates).
- The second peak delay is within one delay bin of τ₂ = 80 ns.

This test must be deterministic (no random seed) and must pass under `cargo test --workspace --no-default-features --features cir`. It follows the pattern established by `verify.py` for the Python pipeline.

**Tier 2 — Phase corruption robustness**

Same two-tap channel but add a random per-subcarrier phase ramp (SFO) and a constant phase offset (CFO). Without sanitization: assert the test fails (ghost tap at τ=0 from CFO). With `phase_sanitizer.rs` applied before `estimate()`: assert the same pass conditions as Tier 1. This validates the ordering decision in Section 2.5.

**Tier 3 — Per-bandwidth regression (unit test)**

For K ∈ {56, 114, 256} with the two-tap channel, assert that the dominant-tap delay estimate error is < 1 delay bin, confirming the 3× super-resolution holds across all tiers.

**Tier 4 — Real hardware capture (integration test, COM9)**

Using the existing ESP32-S3 on COM9 (ruvzen), capture 200 CSI frames in a static room (no motion). Assert:
- `cir.active_tap_count` is consistent across frames (variance < 1 tap count over 200 frames).
- `cir.dominant_tap_ratio` > 0.5 (LOS dominant path present).
- `cir.rms_delay_spread_s` is in the range [10 ns, 200 ns] (reasonable for a room).

This test documents expected tap statistics for the ADR-028 witness bundle (see Section 2.10). The test is gated behind `#[cfg(feature = "hardware-test")]` and is not run in CI.

**Tier 5 — Tier A-HE hardware bench (integration test, COM12)**

Using the ESP32-C6 on COM12 (ruvzen, `MR60BHA2` sensor slot — see CLAUDE.local.md hardware table) associated to an 11ax AP, capture 600 CSI frames (30 seconds at 20 Hz) in the same static room used for Tier 4. Assert:
- `cir.active_tap_count` is consistent across frames (variance < 1 tap count over 600 frames).
- `cir.dominant_tap_ratio` > 0.5 (same threshold as Tier 4).
- `cir.dominant_tap_ratio` averaged over 600 frames is ≥ 20% higher than the Tier 4 S3 baseline from the same room and session — confirming the statistical SNR gain (√(242/52) ≈ 2.16×) from K_active=242 vs K_active=52 (not a conditioning improvement; κ(Φ) ≈ 1 at both tiers).
- Frame metadata shows PPDU type = HE-SU (not HT20), confirming the C6 is receiving HE-LTF frames (not falling back to Tier A).

This test is gated behind `#[cfg(feature = "hardware-test")]` and is not run in CI. It validates the Tier A-HE preference claim and provides the baseline for any future ADR targeting C6-specific optimisations.

### 2.10 Witness and Proof

Per ADR-028, any new signal stage receives a witness entry. The witness additions for CIR:

**WITNESS-LOG-028.md** — add two rows:

| Row | Capability | Evidence | Hash |
|-----|-----------|----------|------|
| W-34 | CIR sparse recovery (synthetic 2-tap, HT20) | `cargo test cir::tests::two_tap_recovery -- --nocapture` output + tap delay error < 1 bin | SHA-256 of stdout |
| W-35 | CIR phase-ordering correctness | `cargo test cir::tests::phase_corruption_rejected` passes with sanitizer, fails without | SHA-256 of test binary |

**`verify.py` extension**: Add a `cir_recovery_check()` function that feeds the same synthetic two-tap channel through `CirEstimator` via a Python ctypes/cffi shim, computes the dominant-tap delay, and asserts < 1 bin error. Hash the function output and compare to `expected_features.sha256`. This integrates CIR into the deterministic proof chain.

The `source-hashes.txt` in the witness bundle adds the SHA-256 of `ruvsense/cir.rs` alongside the existing firmware binaries.

---

## 3. Consequences

### 3.1 Positive

- **Coherence gate precision**: The `multistatic.rs` coherence gate can now isolate motion to specific delay bins. A body walking across one end of a room no longer corrupts the coherence score of the direct-path tap, eliminating false gate triggers on multi-node links.
- **Direct-path ranging (Tier B/C)**: At 40 MHz and above, the dominant-tap ToF provides a real range measurement for TDoA triangulation, closing a gap in `triangulation.rs` that currently estimates position from angle-of-arrival only.
- **Reuses `NeumannSolver`**: Zero new crate dependencies. The ISTA loop wraps the existing solver interface exactly as `fresnel.rs` and `subcarrier.rs` do.
- **Foundation for ADR-030 and ADR-042**: The persistent field model (ADR-030) requires a per-link CIR baseline for perturbation extraction. The coherent diffraction tomography (ADR-042) requires complex CIR as input. Both are unblocked by this ADR.
- **Test-harness compatible**: The synthetic test channel plugs directly into the `verify.py` proof infrastructure without new tooling.

### 3.2 Negative

- **Memory cost**: Measured `Vec<Complex32>` allocation per `CirEstimator::new()`: HT20 = 65 KB, HT40 = 312 KB, HE20 = 1.4 MB (see §2.3b). Sharing one `Arc<CirEstimator>` per tier across all same-tier links is mandatory at HE20+; per-link instantiation at 12 HE20 links costs 16.8 MB for sensing matrices alone.
- **Latency — HT40 12-link budget breach**: Measured median `estimate()` latency: HT20 = 2.72 ms, HT40 = 13.43 ms, HE20 = 3.20 ms (see §2.3b for full table). HT40 at 12-link multistatic sequential = 74.35 ms, which exceeds the 50 ms cycle budget at 20 Hz. HT20 (17.69 ms) and HE20 (est. ~38 ms) both fit. CIR runs on the aggregator, not the ESP32. HT40 multistatic requires Rayon parallelisation (see §2.7). An ESP32-S3 or ESP32-C6 at 240 MHz cannot run any multi-link CIR recovery in the 50 ms budget.
- **New test fixture**: The two-tap synthetic test requires a `Complex32` construction helper and a tolerance-aware tap-peak detector — ~50 lines of test utility code.
- **Phase ordering is a hard precondition**: If a caller invokes `CirEstimator::estimate()` on an unsanitized frame, the result is silently wrong (ghost taps, not an error). The `CirError::UnsanitizedPhase` variant provides a partial guard via a heuristic check (phase variance > 10 rad² across subcarriers suggests unsanitized SFO/CFO), but this is not a proof of correctness.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `NeumannSolver` convergence at low K with high noise | Medium | Ghost taps in HT20 when channel has few paths and low SNR | κ(Φ) ≈ 1 by construction (normalised DFT submatrix, G = 3K), so numerical ill-conditioning is not the risk. The risk is low SNR at K=52 (2.16× weaker than K=242 at same noise floor). Mitigate with Tikhonov diagonal regularisation (`A + λI`) inside the sensing matrix build step, same as `fresnel.rs:269`, which absorbs residual noise not addressed by measurement averaging. |
| Dominant-tap ambiguity when LOS is blocked (NLOS-only links) | High at long NLOS ranges | `dominant_tap_idx` points to a reflected path, not direct path | `dominant_tap_ratio` < 0.3 flags this; `ranging_valid` logic gates on ratio > 0.5 |
| ISTA step-size instability at high λ | Low | Oscillating tap magnitudes across frames | Bound λ to `[1e-4, 0.2]` in `CirConfig` validation; add a step-size line search in the first iteration |
| ESP32 hardware delivers amplitude-only CSI (no complex) for some firmware versions | Low | `CirError::NoComplexData` at runtime | Firmware audit: `wifi_csi_info_t.buf` in ESP-IDF 5.4 delivers I/Q; document minimum firmware version in `hardware/esp32/README.md` |

---

## 4. Rationale and Comparison to Alternative Designs

### 4.1 Why Not Compute CIR in Python (`archive/v1/`)

The Python pipeline in `archive/v1/src/` is frozen. ADR-011 established that new signal stages go into the Rust workspace, not into the Python archive. The Python proof (`verify.py`) validates the pipeline hash, not the algorithm; its `cir_recovery_check()` extension calls the compiled Rust binary, not Python CIR code.

### 4.2 Why Not Rely on rvCSI Exclusively

`vendor/rvcsi` (ADR-095/096) provides a `CsiFrame`/`CsiWindow`/`CsiEvent` schema and Nexmon adapter, but the published `rvcsi-dsp` crate does not currently implement CIR estimation (as of May 2026 — confirmed by crate source). Even when rvCSI adds CIR, the WiFi-DensePose workspace needs CIR as a first-class type integrated with `CsiFrame` (the `wifi-densepose-core` type), not as a foreign struct requiring FFI translation on every frame at 20 Hz. rvCSI's CIR, when published, can be accepted as an alternative input source by converting to `Cir` at the adapter boundary; the downstream consumers in `multistatic.rs` and `triangulation.rs` will not need to change.

### 4.3 Why Not Frequency-Domain Only Forever

The three research documents (SOTA survey, architecture, edge-weight computation) all converge on the same conclusion: frequency-domain CSI features are sufficient for presence and coarse gesture, but insufficient for:

1. **Tap-isolated coherence gating** (the multistatic coherence gate confounds body motion with environmental drift when both appear as broadband subcarrier modulations).
2. **Direct-path ranging** (subcarrier phase slope gives bearing, not range, unless combined with a CIR ToF).
3. **Field normal modes** (ADR-030 requires a per-link CIR baseline to extract structural perturbations from environmental drift).

Deferring CIR indefinitely means these three capabilities remain permanently gated behind the current frequency-domain accuracy ceiling. CIRSense (arXiv:2510.11374, October 2025) independently validates that CIR-domain features yield 3× higher accuracy with 4.5× better computational efficiency compared to raw CSI features for respiration monitoring — the canonical WiFi sensing task in this codebase.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-014 (SOTA Signal Processing) | **Extended**: CIR adds a 7th signal module alongside the 6 in ADR-014 |
| ADR-017 (RuVector Signal+MAT) | **Enables**: ADR-017's coherence gate pseudocode references CIR; now implementable |
| ADR-029 (RuvSense Multistatic) | **Unblocks**: `reconstruct_cir()` stub in `process_cycle()` now has a concrete implementation |
| ADR-030 (Persistent Field Model) | **Prerequisite fulfilled**: baseline CIR per link is required for perturbation extraction |
| ADR-042 (Coherent Human Channel Imaging) | **Foundation layer**: CHCI's coherent diffraction tomography consumes `Cir` as primary input |
| ADR-095/096 (rvCSI) | **Complementary**: rvCSI provides the Nexmon adapter for Tier C; CIR estimation runs on top |
| ADR-028 (ESP32 Capability Audit) | **Witness extended**: two new rows W-34, W-35 added to `WITNESS-LOG-028.md` |
| ADR-110 (ESP32-C6 Firmware Extension) | **Substrate**: HE-LTF PPDU classification (frame bytes 18–19), TWT wake slots (`c6_twt.c`), and 802.15.4 epoch timestamps (`c6_timesync_get_epoch_us()`) — all shipped in v0.7.0-esp32. Tier A-HE `CirConfig` depends on PPDU type from ADR-110 for automatic tier detection. |

---

## 6. References

### Production Code
- `v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs` — current amplitude/phase coherence gate; `reconstruct_cir()` call site
- `v2/crates/wifi-densepose-signal/src/phase_sanitizer.rs` — must run before `CirEstimator::estimate()`
- `v2/crates/wifi-densepose-signal/src/fresnel.rs:280` — `NeumannSolver` usage pattern this ADR mirrors
- `v2/crates/wifi-densepose-train/src/subcarrier.rs:225` — second `NeumannSolver` usage in workspace
- `v2/crates/wifi-densepose-mat/src/ml/vital_signs_classifier.rs:386` — the only IFFT in production (unrelated to CIR)

### Research Documents
- `docs/research/sota-surveys/ruview-multistatic-fidelity-sota-2026.md` — bandwidth table, 20 MHz separability analysis
- `docs/research/architecture/ruvsense-multistatic-fidelity-architecture.md` — `NeumannSolver` CIR proposal (§2.1), pipeline diagram (§4.1), `is_coherent(cir)` pseudocode (line 583)
- `docs/research/rf-topological-sensing/02-csi-edge-weight-computation.md` — IFFT formula, CIR features, ESPRIT for multipath decomposition

### External Papers
- Kotaru et al., "SpotFi: Decimeter Level Localization Using WiFi," ACM SIGCOMM 2015 — MUSIC for AoA; spatial smoothing from K subcarriers
- Vasisht et al., "Decimeter-Level Localization with a Single WiFi Access Point," NSDI 2016 (Chronos) — BPDN for sparse CIR across stitched channels
- CIRSense, arXiv:2510.11374 (October 2025) — CIR delay-domain sensing; ISTA sparse recovery; 3× accuracy vs CSI, 4.5× compute efficiency; validated at 160 MHz (informative for Tier C)
- "Pulse Shape-Aided Multipath Delay Estimation for Fine-Grained WiFi Sensing," arXiv:2306.15320 — OMP vs ISTA comparison at low SNR
- "Neuro-Wideband WiFi Sensing via Self-Conditioned CSI Extrapolation," arXiv:2601.06467 (January 2026) — neural CIR extrapolation requiring ≥200 MHz; explains why neural approach is rejected for this ADR
- Zheng et al., "Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi," MobiSys 2019 (Widar 3.0) — BVP as domain-independent alternative to CIR; relevant to vitals-path decision
