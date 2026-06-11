//! Empty-room baseline calibration (ADR-135).
//!
//! Captures per-subcarrier amplitude and circular-phase statistics from a
//! quiescent (empty) room using Welford's online algorithm, then provides
//! real-time deviation scoring and in-place baseline subtraction.
//!
//! # Pipeline position
//!
//! Raw CSI → `phase_sanitizer.rs` → `phase_align.rs`
//!         → `CalibrationRecorder::record()`   (calibration mode)
//!         → `BaselineCalibration::subtract_in_place()`  (runtime mode)
//!         → `CirEstimator::estimate()`
//!
//! # Binary format (to_bytes / from_bytes)
//!
//! 16-byte header (all little-endian):
//!   magic:             u32 = 0xCA1B_0001
//!   version:           u8  = 1
//!   tier:              u8  (0=Ht20, 1=Ht40, 2=He20, 3=He40)
//!   reserved:          u16 = 0
//!   captured_at_unix_s: i64
//! Body:
//!   frame_count:       u64
//!   num_subcarriers:   u32
//!   for each subcarrier: amp_mean f32 LE, amp_variance f32 LE,
//!                         phase_mean f32 LE, phase_dispersion f32 LE
//!
//! SHA-256-stable: all writes are LE, no float branching.

use num_complex::Complex32;
use thiserror::Error;
use wifi_densepose_core::types::CsiFrame;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: u32 = 0xCA1B_0001;
const VERSION: u8 = 1;
const HEADER_LEN: usize = 16; // magic(4) + version(1) + tier(1) + reserved(2) + unix_s(8)
const SUBCARRIER_RECORD_LEN: usize = 16; // 4 × f32

/// Baseline phase channel is considered usable only when the median Von Mises
/// dispersion across subcarriers is at or below this value.
///
/// A sanitised pipeline (`PhaseSanitizer` + `phase_align.rs`) lands well under
/// the `CalibrationConfig::max_phase_variance` default of 0.3. The CLI UDP
/// path feeds *unsanitised* phase: the ADR-151 presence-flatline bench
/// (2026-06-11, real ESP32-C6 HE20 captures) measured baseline
/// `phase_dispersion` ≈ 0.965 — i.e. uniformly random phase — which makes
/// `phase_drift_median` ≈ π/2 pure noise on every frame. Gating on this
/// constant keeps the noise out of `motion_flagged`.
pub const PHASE_DISPERSION_USABLE_MAX: f32 = 0.5;

// ---------------------------------------------------------------------------
// PHY tier
// ---------------------------------------------------------------------------

/// 802.11 PHY tier identifies the subcarrier layout.
/// A mismatch between a stored baseline and a live frame triggers
/// `CalibrationError::TierMismatch` (ADR-135 §risk 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhyTier {
    /// 802.11n HT20: 64-FFT, 52 active subcarriers.
    Ht20,
    /// 802.11n HT40: 128-FFT, 114 active subcarriers.
    Ht40,
    /// 802.11ax HE20: 256-FFT, 242 active subcarriers.
    He20,
    /// 802.11ax HE40: 512-FFT, 484 active subcarriers.
    He40,
}

impl PhyTier {
    fn to_u8(self) -> u8 {
        match self {
            PhyTier::Ht20 => 0,
            PhyTier::Ht40 => 1,
            PhyTier::He20 => 2,
            PhyTier::He40 => 3,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(PhyTier::Ht20),
            1 => Some(PhyTier::Ht40),
            2 => Some(PhyTier::He20),
            3 => Some(PhyTier::He40),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Calibration capture configuration.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationConfig {
    /// PHY tier determines expected subcarrier count.
    pub tier: PhyTier,
    /// Total OFDM FFT bins (e.g. 64 HT20, 128 HT40, 256 HE20, 512 HE40).
    pub num_subcarriers: usize,
    /// Active (non-guard, non-DC) tones (52, 114, 242, 484).
    pub num_active: usize,
    /// Minimum frames before `finalize()` succeeds (default 600).
    pub min_frames: u32,
    /// Von Mises dispersion warn threshold — warn if any subcarrier exceeds this
    /// during recording (ADR-135 §risk 1). Default 0.3.
    pub max_phase_variance: f32,
}

impl CalibrationConfig {
    /// HT20 defaults: 64 FFT, 52 active, 600 frame minimum (30 s @ 20 Hz).
    pub fn ht20() -> Self {
        Self { tier: PhyTier::Ht20, num_subcarriers: 64, num_active: 52, min_frames: 600, max_phase_variance: 0.3 }
    }
    /// HT40 defaults: 128 FFT, 114 active.
    pub fn ht40() -> Self {
        Self { tier: PhyTier::Ht40, num_subcarriers: 128, num_active: 114, min_frames: 600, max_phase_variance: 0.3 }
    }
    /// HE20 defaults: 256 FFT, 242 active.
    pub fn he20() -> Self {
        Self { tier: PhyTier::He20, num_subcarriers: 256, num_active: 242, min_frames: 600, max_phase_variance: 0.3 }
    }
    /// HE40 defaults: 512 FFT, 484 active.
    pub fn he40() -> Self {
        Self { tier: PhyTier::He40, num_subcarriers: 512, num_active: 484, min_frames: 600, max_phase_variance: 0.3 }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from calibration operations.
#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("subcarrier count mismatch: expected {expected}, got {got}")]
    SubcarrierMismatch { expected: usize, got: usize },

    #[error("tier mismatch: baseline tier {baseline:?}, frame tier {frame:?}")]
    TierMismatch { baseline: PhyTier, frame: PhyTier },

    #[error("insufficient frames: have {got}, need {need}")]
    InsufficientFrames { got: u32, need: u32 },

    #[error("baseline serialization version mismatch: have v{got}, expected v{want}")]
    VersionMismatch { got: u8, want: u8 },

    #[error("buffer too short to deserialize baseline (have {got} bytes, need at least {need})")]
    TruncatedBuffer { got: usize, need: usize },

    #[error("invalid magic word: expected 0xCA1B0001, got 0x{got:08X}")]
    InvalidMagic { got: u32 },

    #[error("unknown tier byte: {0}")]
    UnknownTier(u8),
}

// ---------------------------------------------------------------------------
// Per-subcarrier running statistics
// ---------------------------------------------------------------------------

/// Per-subcarrier Welford amplitude + circular-phase accumulators.
///
/// Amplitude uses the standard Welford recurrence (as in `field_model::WelfordStats`
/// but inlined here into a struct-of-arrays to avoid pub-API churn on that type).
/// Phase uses sin/cos running sums — the standard technique for circular statistics.
#[derive(Debug, Clone)]
struct SubcarrierStats {
    amp_count: u64,
    amp_mean: f64,
    amp_m2: f64,
    phase_sin_sum: f64,
    phase_cos_sum: f64,
}

impl SubcarrierStats {
    fn new() -> Self {
        Self { amp_count: 0, amp_mean: 0.0, amp_m2: 0.0, phase_sin_sum: 0.0, phase_cos_sum: 0.0 }
    }

    /// Welford update for amplitude; circular update for phase.
    fn update(&mut self, c: Complex32) {
        let amp = c.norm() as f64;
        self.amp_count += 1;
        let delta = amp - self.amp_mean;
        self.amp_mean += delta / self.amp_count as f64;
        let delta2 = amp - self.amp_mean;
        self.amp_m2 += delta * delta2;

        let theta = c.arg() as f64;
        self.phase_sin_sum += theta.sin();
        self.phase_cos_sum += theta.cos();
    }

    /// Bessel-corrected sample variance (matches Welford convention).
    fn amp_variance(&self) -> f64 {
        if self.amp_count < 2 { 0.0 } else { self.amp_m2 / (self.amp_count - 1) as f64 }
    }

    /// Circular mean phase in `[-π, π]`.
    fn phase_mean(&self) -> f64 {
        self.phase_sin_sum.atan2(self.phase_cos_sum)
    }

    /// Von Mises dispersion `1 − R̄` in `[0, 1]`.
    fn phase_dispersion(&self) -> f64 {
        if self.amp_count == 0 { return 1.0; }
        let n = self.amp_count as f64;
        let r = (self.phase_sin_sum * self.phase_sin_sum + self.phase_cos_sum * self.phase_cos_sum).sqrt() / n;
        1.0 - r.min(1.0)
    }
}

// ---------------------------------------------------------------------------
// SubcarrierBaseline (public per-subcarrier summary)
// ---------------------------------------------------------------------------

/// Finalised per-subcarrier statistics from a baseline capture.
#[derive(Debug, Clone, Copy)]
pub struct SubcarrierBaseline {
    pub amp_mean: f32,
    pub amp_variance: f32,
    /// Circular mean phase in `[-π, π]` (radians).
    pub phase_mean: f32,
    /// Von Mises dispersion `1 − R̄` in `[0, 1]`; 0 = perfectly stationary.
    pub phase_dispersion: f32,
}

// ---------------------------------------------------------------------------
// BaselineCalibration
// ---------------------------------------------------------------------------

/// A fully finalised empty-room baseline (immutable after construction).
#[derive(Debug, Clone)]
pub struct BaselineCalibration {
    pub tier: PhyTier,
    pub captured_at_unix_s: i64,
    pub frame_count: u64,
    /// Per-subcarrier statistics, ordered by active-subcarrier index.
    pub subcarriers: Vec<SubcarrierBaseline>,
}

impl BaselineCalibration {
    /// Compute a per-frame deviation score against this baseline.
    pub fn deviation(&self, frame: &CsiFrame) -> Result<CalibrationDeviationScore, CalibrationError> {
        let n_sc = frame.num_subcarriers();
        let expected = self.subcarriers.len();
        if n_sc != expected && n_sc != self.tier_num_subcarriers() {
            return Err(CalibrationError::SubcarrierMismatch { expected, got: n_sc });
        }
        let y = extract_first_stream(frame, expected, self.tier_num_subcarriers());
        let mut z_amp = Vec::with_capacity(expected);
        let mut phase_drift = Vec::with_capacity(expected);
        for (ki, (c, baseline)) in y.iter().zip(self.subcarriers.iter()).enumerate() {
            let _ = ki;
            let amp = c.norm();
            let std = baseline.amp_variance.sqrt().max(1e-12_f32);
            z_amp.push((amp - baseline.amp_mean) / std);
            let theta = c.arg();
            let drift = circular_distance(theta, baseline.phase_mean);
            phase_drift.push(drift);
        }
        let amplitude_z_median = median_abs(&z_amp);
        let amplitude_z_p90 = percentile_abs(&z_amp, 0.90);
        let amplitude_z_max = z_amp.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let phase_drift_median = median_slice(&phase_drift);
        let phase_usable = self.phase_usable();
        let motion_flagged = amplitude_z_median > 2.0
            || (phase_usable && phase_drift_median > std::f32::consts::PI / 6.0);
        Ok(CalibrationDeviationScore {
            amplitude_z_median,
            amplitude_z_p90,
            amplitude_z_max,
            phase_drift_median,
            phase_usable,
            motion_flagged,
        })
    }

    /// Whether this baseline's phase channel carries signal (median Von Mises
    /// dispersion ≤ [`PHASE_DISPERSION_USABLE_MAX`]). False when the capture
    /// pipeline skipped `PhaseSanitizer` / `phase_align.rs` — see the constant's
    /// doc for the bench evidence.
    #[must_use]
    pub fn phase_usable(&self) -> bool {
        let disp: Vec<f32> = self.subcarriers.iter().map(|b| b.phase_dispersion).collect();
        median_slice(&disp) <= PHASE_DISPERSION_USABLE_MAX
    }

    /// Deterministic calibration epoch id (ADR-137 `CalibrationId`), derived
    /// from the immutable baseline fields — stable across reboots, changes only
    /// on recalibration. Deterministic (no RNG) so the ADR-136 witness replay
    /// stays reproducible.
    #[must_use]
    pub fn calibration_id(&self) -> super::fusion_quality::CalibrationId {
        // splitmix64 over (captured_at, frame_count, subcarrier_count, tier).
        let mut h = (self.captured_at_unix_s as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.frame_count.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add((self.subcarriers.len() as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
            .wrapping_add(self.tier as u64);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        super::fusion_quality::CalibrationId(h)
    }

    /// The ADR-136 `FrameMeta.calibration_id` value (a UUID derived
    /// deterministically from [`Self::calibration_id`]).
    #[must_use]
    pub fn calibration_uuid(&self) -> uuid::Uuid {
        uuid::Uuid::from_u128(self.calibration_id().0 as u128)
    }

    /// ADR-136 §2.4 calibration **Stage**: subtract the baseline AND stamp the
    /// frame's `calibration_id` provenance field. This is the only place that
    /// sets `calibration_id` (the append-only boundary rule).
    ///
    /// # Errors
    /// [`CalibrationError::SubcarrierMismatch`] if the frame's subcarrier count
    /// does not match this baseline.
    pub fn apply(&self, frame: &mut CsiFrame) -> Result<(), CalibrationError> {
        self.subtract_in_place(frame)?;
        frame.metadata.set_calibration(self.calibration_uuid());
        Ok(())
    }

    /// Subtract the amplitude baseline from `frame.data` in-place.
    /// Only amplitude mean is subtracted; phase is left untouched.
    pub fn subtract_in_place(&self, frame: &mut CsiFrame) -> Result<(), CalibrationError> {
        let n_sc = frame.num_subcarriers();
        let expected = self.subcarriers.len();
        if n_sc != expected && n_sc != self.tier_num_subcarriers() {
            return Err(CalibrationError::SubcarrierMismatch { expected, got: n_sc });
        }
        let n_streams = frame.num_spatial_streams();
        let n_total = self.tier_num_subcarriers();
        let active_input = n_sc == expected;
        for ki in 0..expected {
            let col = if active_input { ki } else { ki }; // sequential when active-only
            let baseline_amp = self.subcarriers[ki].amp_mean as f64;
            for s in 0..n_streams {
                let c = frame.data[[s, col]];
                let norm = c.norm();
                if norm > 1e-30 {
                    let scale = ((norm - baseline_amp).max(0.0)) / norm;
                    frame.data[[s, col]] = num_complex::Complex64::new(c.re * scale, c.im * scale);
                }
            }
            let _ = n_total;
        }
        Ok(())
    }

    /// Reference complex CSI vector: `amp_mean × exp(j × phase_mean)` per subcarrier.
    /// Pass to `CirEstimator::set_reference_csi()`.
    pub fn reference_csi_vector(&self) -> Vec<Complex32> {
        self.subcarriers.iter().map(|b| {
            let (sin, cos) = b.phase_mean.sin_cos();
            Complex32::new(b.amp_mean * cos, b.amp_mean * sin)
        }).collect()
    }

    /// Serialise to little-endian binary (see module-level format doc).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.subcarriers.len();
        let mut buf = Vec::with_capacity(HEADER_LEN + 8 + 4 + n * SUBCARRIER_RECORD_LEN);
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.push(VERSION);
        buf.push(self.tier.to_u8());
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        buf.extend_from_slice(&self.captured_at_unix_s.to_le_bytes());
        buf.extend_from_slice(&self.frame_count.to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        for sc in &self.subcarriers {
            buf.extend_from_slice(&sc.amp_mean.to_le_bytes());
            buf.extend_from_slice(&sc.amp_variance.to_le_bytes());
            buf.extend_from_slice(&sc.phase_mean.to_le_bytes());
            buf.extend_from_slice(&sc.phase_dispersion.to_le_bytes());
        }
        buf
    }

    /// Deserialise from little-endian binary produced by `to_bytes`.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, CalibrationError> {
        const MIN_LEN: usize = HEADER_LEN + 8 + 4; // header + frame_count + num_subcarriers
        if buf.len() < MIN_LEN {
            return Err(CalibrationError::TruncatedBuffer { got: buf.len(), need: MIN_LEN });
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(CalibrationError::InvalidMagic { got: magic });
        }
        let version = buf[4];
        if version != VERSION {
            return Err(CalibrationError::VersionMismatch { got: version, want: VERSION });
        }
        let tier_byte = buf[5];
        let tier = PhyTier::from_u8(tier_byte).ok_or(CalibrationError::UnknownTier(tier_byte))?;
        // reserved: buf[6..8] — ignored
        let captured_at_unix_s = i64::from_le_bytes(buf[8..16].try_into().unwrap());
        let frame_count = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let n = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;
        let needed = MIN_LEN + n * SUBCARRIER_RECORD_LEN;
        if buf.len() < needed {
            return Err(CalibrationError::TruncatedBuffer { got: buf.len(), need: needed });
        }
        let mut subcarriers = Vec::with_capacity(n);
        let mut off = 28usize;
        for _ in 0..n {
            let amp_mean = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()); off += 4;
            let amp_variance = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()); off += 4;
            let phase_mean = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()); off += 4;
            let phase_dispersion = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()); off += 4;
            subcarriers.push(SubcarrierBaseline { amp_mean, amp_variance, phase_mean, phase_dispersion });
        }
        Ok(Self { tier, captured_at_unix_s, frame_count, subcarriers })
    }

    /// Total FFT bins for this tier (used for dual-convention column selection).
    fn tier_num_subcarriers(&self) -> usize {
        match self.tier {
            PhyTier::Ht20 => 64,
            PhyTier::Ht40 => 128,
            PhyTier::He20 => 256,
            PhyTier::He40 => 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Deviation score
// ---------------------------------------------------------------------------

/// Per-frame deviation metrics against the static baseline.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationDeviationScore {
    /// Median of `|z_amp[k]|` across active subcarriers.
    ///
    /// **Not a presence statistic.** A person perturbs a *minority* of
    /// subcarriers strongly; the median of |z| over all bins floors at
    /// `median(|N(0,1)|) ≈ 0.674` and barely moves when someone enters the
    /// room (ADR-151 presence-flatline bench, 2026-06-11: empty ≈ 0.40–0.42
    /// vs person-moving ≈ 0.74–0.75 on real ESP32-C6 HE20 captures). Kept for
    /// drift scoring and backwards compatibility; presence gating uses
    /// [`Self::amplitude_z_p90`].
    pub amplitude_z_median: f32,
    /// 90th percentile of `|z_amp[k]|` across active subcarriers — the
    /// ADR-151 presence statistic. The upper tail is where a body's multipath
    /// perturbation lives: the same bench measured empty ≈ 1.27–1.33 vs
    /// person-moving ≈ 2.26–2.29 (same-epoch baseline), a clean separation
    /// the median never shows.
    pub amplitude_z_p90: f32,
    /// Max single-subcarrier `|z_amp[k]|`.
    pub amplitude_z_max: f32,
    /// Median circular distance (radians) between live and baseline phase.
    pub phase_drift_median: f32,
    /// Whether the baseline's phase channel carries signal (see
    /// [`PHASE_DISPERSION_USABLE_MAX`]). When false, `phase_drift_median` is
    /// noise and is excluded from `motion_flagged`.
    pub phase_usable: bool,
    /// Heuristic: `amplitude_z_median > 2.0 || (phase_usable &&
    /// phase_drift_median > π/6)`.
    pub motion_flagged: bool,
}

// ---------------------------------------------------------------------------
// CalibrationRecorder
// ---------------------------------------------------------------------------

/// Accumulates CSI frames from an empty room using Welford online statistics.
///
/// Phase precondition: the caller must pass frames processed by
/// `PhaseSanitizer` and `phase_align.rs`. Unsanitised phase produces
/// inflated `phase_dispersion` values.
pub struct CalibrationRecorder {
    config: CalibrationConfig,
    started_at_unix_s: i64,
    stats: Vec<SubcarrierStats>,
    frame_count: u32,
}

impl CalibrationRecorder {
    /// Create a new recorder for the given configuration.
    pub fn new(config: CalibrationConfig) -> Self {
        let stats = vec![SubcarrierStats::new(); config.num_active];
        Self { config, started_at_unix_s: unix_now_s(), stats, frame_count: 0 }
    }

    /// Ingest one sanitised CSI frame. Returns a deviation score from the
    /// current partial baseline so the operator can monitor room occupancy
    /// in real time.
    pub fn record(&mut self, frame: &CsiFrame) -> Result<CalibrationDeviationScore, CalibrationError> {
        let n_sc = frame.num_subcarriers();
        let expected_active = self.config.num_active;
        let expected_total = self.config.num_subcarriers;
        if n_sc != expected_active && n_sc != expected_total {
            return Err(CalibrationError::SubcarrierMismatch { expected: expected_active, got: n_sc });
        }
        let y = extract_first_stream(frame, expected_active, expected_total);
        for (ki, c) in y.iter().enumerate() {
            self.stats[ki].update(*c);
        }
        self.frame_count += 1;

        // Build deviation from partial baseline (after first frame).
        let mut z_amp_abs = Vec::with_capacity(expected_active);
        let mut phase_drift = Vec::with_capacity(expected_active);
        for (c, st) in y.iter().zip(self.stats.iter()) {
            let amp = c.norm();
            let std = (st.amp_variance() as f32).sqrt().max(1e-12_f32);
            z_amp_abs.push((amp - st.amp_mean as f32).abs() / std);
            phase_drift.push(circular_distance(c.arg(), st.phase_mean() as f32));
        }
        let amplitude_z_median = median_slice(&z_amp_abs);
        let amplitude_z_p90 = percentile_abs(&z_amp_abs, 0.90);
        let amplitude_z_max = z_amp_abs.iter().copied().fold(0.0_f32, f32::max);
        let phase_drift_median = median_slice(&phase_drift);
        let disp: Vec<f32> = self.stats.iter().map(|st| st.phase_dispersion() as f32).collect();
        let phase_usable = median_slice(&disp) <= PHASE_DISPERSION_USABLE_MAX;
        let motion_flagged = amplitude_z_median > 2.0
            || (phase_usable && phase_drift_median > std::f32::consts::PI / 6.0);
        Ok(CalibrationDeviationScore {
            amplitude_z_median,
            amplitude_z_p90,
            amplitude_z_max,
            phase_drift_median,
            phase_usable,
            motion_flagged,
        })
    }

    /// Number of frames recorded so far.
    pub fn frames_recorded(&self) -> u32 {
        self.frame_count
    }

    /// Consume the recorder and produce a finalised baseline.
    /// Returns `CalibrationError::InsufficientFrames` if fewer than
    /// `config.min_frames` frames were recorded.
    pub fn finalize(self) -> Result<BaselineCalibration, CalibrationError> {
        if self.frame_count < self.config.min_frames {
            return Err(CalibrationError::InsufficientFrames {
                got: self.frame_count,
                need: self.config.min_frames,
            });
        }
        let subcarriers = self.stats.iter().map(|st| SubcarrierBaseline {
            amp_mean: st.amp_mean as f32,
            amp_variance: st.amp_variance() as f32,
            phase_mean: st.phase_mean() as f32,
            phase_dispersion: st.phase_dispersion() as f32,
        }).collect();
        Ok(BaselineCalibration {
            tier: self.config.tier,
            captured_at_unix_s: self.started_at_unix_s,
            frame_count: self.frame_count as u64,
            subcarriers,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the first spatial stream as a `Vec<Complex32>`, honouring the
/// dual-convention used by `cir.rs::extract_csi_vector`: if the frame has
/// exactly `num_active` subcarriers they are taken sequentially; otherwise
/// the first `num_active` columns of the full FFT grid are used.
fn extract_first_stream(frame: &CsiFrame, num_active: usize, _num_total: usize) -> Vec<Complex32> {
    let n_sc = frame.num_subcarriers();
    let take = num_active.min(n_sc);
    (0..take).map(|ki| {
        let c = frame.data[[0, ki]];
        Complex32::new(c.re as f32, c.im as f32)
    }).collect()
}

/// Signed circular distance wrapped to `[0, π]`.
fn circular_distance(a: f32, b: f32) -> f32 {
    let mut d = (a - b).abs();
    if d > std::f32::consts::PI {
        d = 2.0 * std::f32::consts::PI - d;
    }
    d
}

/// Median of absolute values of a slice.
fn median_abs(v: &[f32]) -> f32 {
    let mut abs: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    median_in_place(&mut abs)
}

/// `p`-quantile (`0.0..=1.0`) of the absolute values of a slice, with linear
/// interpolation between order statistics (numpy's default convention — the
/// ADR-151 bench thresholds were derived with it).
fn percentile_abs(v: &[f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut abs: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = p.clamp(0.0, 1.0) * (abs.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    abs[lo] + (abs[hi] - abs[lo]) * frac
}

/// Median of a slice (non-destructive clone).
fn median_slice(v: &[f32]) -> f32 {
    let mut c = v.to_vec();
    median_in_place(&mut c)
}

fn median_in_place(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() { return 0.0; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 { (v[mid - 1] + v[mid]) / 2.0 } else { v[mid] }
}

/// Current Unix timestamp in seconds. Falls back to 0 if unavailable.
fn unix_now_s() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;
    use wifi_densepose_core::types::{CsiMetadata, CsiFrame};

    fn make_frame(data: Array2<Complex64>) -> CsiFrame {
        use wifi_densepose_core::types::{DeviceId, FrequencyBand};
        let meta = CsiMetadata::new(
            DeviceId::new("test-device"),
            FrequencyBand::Band2_4GHz,
            6,
        );
        CsiFrame::new(meta, data)
    }

    fn constant_frame(n_sc: usize, amp: f64, phase: f64) -> CsiFrame {
        let row = (0..n_sc).map(|_| Complex64::from_polar(amp, phase)).collect::<Vec<_>>();
        let arr = Array2::from_shape_vec((1, n_sc), row).unwrap();
        make_frame(arr)
    }

    // (a) Welford convergence: constant input → variance ≈ 0, mean = amp.
    #[test]
    fn welford_constant_input_converges() {
        let mut st = SubcarrierStats::new();
        let c = Complex32::new(1.0, 0.0);
        for _ in 0..600 {
            st.update(c);
        }
        assert!((st.amp_mean - 1.0).abs() < 1e-9);
        assert!(st.amp_variance() < 1e-20, "variance was {}", st.amp_variance());
    }

    // (b) Circular phase mean recovers known phase from N noisy samples.
    #[test]
    fn circular_phase_mean_recovery() {
        use std::f64::consts::PI;
        let mut st = SubcarrierStats::new();
        let target = PI / 4.0;
        // Feed 200 samples: 100 at target+0.05, 100 at target-0.05.
        for _ in 0..100 {
            st.update(Complex32::from_polar(1.0, (target + 0.05) as f32));
            st.update(Complex32::from_polar(1.0, (target - 0.05) as f32));
        }
        let recovered = st.phase_mean();
        assert!((recovered - target).abs() < 0.01, "phase error = {}", (recovered - target).abs());
        // Dispersion should be low (close to 0) for tight phase cluster.
        assert!(st.phase_dispersion() < 0.01, "dispersion = {}", st.phase_dispersion());
    }

    // (c) Round-trip: to_bytes → from_bytes preserves all baseline fields.
    #[test]
    fn round_trip_to_from_bytes() {
        let mut cfg = CalibrationConfig::ht20();
        cfg.min_frames = 2;
        let mut rec = CalibrationRecorder::new(cfg);
        let f1 = constant_frame(52, 0.8, 0.5);
        let f2 = constant_frame(52, 0.9, 0.6);
        rec.record(&f1).unwrap();
        rec.record(&f2).unwrap();
        let baseline = rec.finalize().unwrap();

        let bytes = baseline.to_bytes();
        let recovered = BaselineCalibration::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.frame_count, baseline.frame_count);
        assert_eq!(recovered.tier, baseline.tier);
        assert_eq!(recovered.subcarriers.len(), baseline.subcarriers.len());
        for (a, b) in recovered.subcarriers.iter().zip(baseline.subcarriers.iter()) {
            assert!((a.amp_mean - b.amp_mean).abs() < 1e-6, "amp_mean mismatch");
            assert!((a.phase_mean - b.phase_mean).abs() < 1e-6, "phase_mean mismatch");
            assert!((a.phase_dispersion - b.phase_dispersion).abs() < 1e-6, "dispersion mismatch");
        }
    }

    // ADR-136: calibration Stage stamps calibration_id deterministically.
    #[test]
    fn apply_stamps_calibration_id_deterministically() {
        let mut cfg = CalibrationConfig::ht20();
        cfg.min_frames = 2;
        let mut rec = CalibrationRecorder::new(cfg);
        rec.record(&constant_frame(52, 0.8, 0.5)).unwrap();
        rec.record(&constant_frame(52, 0.9, 0.6)).unwrap();
        let baseline = rec.finalize().unwrap();

        // id is stable across calls (no RNG).
        assert_eq!(baseline.calibration_id(), baseline.calibration_id());
        assert_eq!(baseline.calibration_uuid(), baseline.calibration_uuid());

        // apply() subtracts AND stamps the frame's provenance field.
        let mut frame = constant_frame(52, 1.0, 0.5);
        assert_eq!(frame.metadata.calibration_id, None);
        baseline.apply(&mut frame).unwrap();
        assert_eq!(frame.metadata.calibration_id, Some(baseline.calibration_uuid()));
    }

    // (d) Tier dispatch: each config constructor produces the correct counts.
    #[test]
    fn tier_dispatch_correct_counts() {
        let ht20 = CalibrationConfig::ht20();
        assert_eq!(ht20.num_subcarriers, 64);
        assert_eq!(ht20.num_active, 52);

        let ht40 = CalibrationConfig::ht40();
        assert_eq!(ht40.num_subcarriers, 128);
        assert_eq!(ht40.num_active, 114);

        let he20 = CalibrationConfig::he20();
        assert_eq!(he20.num_subcarriers, 256);
        assert_eq!(he20.num_active, 242);

        let he40 = CalibrationConfig::he40();
        assert_eq!(he40.num_subcarriers, 512);
        assert_eq!(he40.num_active, 484);
    }

    // Additional: insufficient frames → error.
    #[test]
    fn finalize_requires_min_frames() {
        let cfg = CalibrationConfig::ht20(); // min_frames = 600
        let mut rec = CalibrationRecorder::new(cfg);
        let f = constant_frame(52, 1.0, 0.0);
        rec.record(&f).unwrap();
        match rec.finalize() {
            Err(CalibrationError::InsufficientFrames { got: 1, need: 600 }) => {}
            other => panic!("expected InsufficientFrames, got {:?}", other),
        }
    }

    // Binary magic / version check.
    #[test]
    fn binary_magic_and_version() {
        let mut cfg = CalibrationConfig::ht20();
        cfg.min_frames = 1;
        let mut rec = CalibrationRecorder::new(cfg);
        rec.record(&constant_frame(52, 1.0, 0.0)).unwrap();
        let b = rec.finalize().unwrap().to_bytes();
        let magic = u32::from_le_bytes(b[0..4].try_into().unwrap());
        assert_eq!(magic, 0xCA1B_0001u32);
        assert_eq!(b[4], 1u8); // version = 1
    }

    // Subcarrier mismatch is rejected.
    #[test]
    fn subcarrier_mismatch_error() {
        let mut cfg = CalibrationConfig::ht20();
        cfg.min_frames = 1;
        let mut rec = CalibrationRecorder::new(cfg);
        let bad = constant_frame(50, 1.0, 0.0); // 50 ≠ 52, 50 ≠ 64
        assert!(matches!(
            rec.record(&bad),
            Err(CalibrationError::SubcarrierMismatch { expected: 52, got: 50 })
        ));
    }
}
