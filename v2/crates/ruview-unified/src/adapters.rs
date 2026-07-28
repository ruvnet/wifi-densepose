//! Hardware adapters — vendor formats in, canonical [`RfTensor`] out
//! (ADR-274 §2.3).
//!
//! Each adapter performs the same three-stage normalization so every
//! downstream consumer sees hardware-invariant data:
//!
//! 1. **Layout**: reshape the vendor capture to `(links, bins, snapshots)`
//!    (for FMCW radar this includes the fast-time DFT to range bins), then
//!    resample both the bin and snapshot axes to
//!    [`CANONICAL_BINS`] × [`CANONICAL_SNAPSHOTS`].
//! 2. **Amplitude**: divide each link by its median amplitude, removing
//!    front-end gain differences between chipsets (the offset is recorded in
//!    [`CalibrationMeta::gain_offset_db`] for provenance).
//! 3. **Phase**: per (link, snapshot), remove the constant phase offset (CFO
//!    residual) and the linear ramp across bins (sampling-time offset) via
//!    least squares — unless the capture declares itself phase-calibrated.

use std::collections::HashMap;

use ndarray::{Array3, Axis};
use num_complex::Complex64;
use wifi_densepose_core::types::{CsiFrame, FrequencyBand};

use crate::math::{linear_slope, median, resample_complex};
use crate::tensor::{
    CalibrationMeta, LinkGeometry, RfModality, RfTensor, CANONICAL_BINS, CANONICAL_SNAPSHOTS,
};
use crate::{Result, UnifiedError};

/// A raw capture from some hardware front-end, before normalization.
#[derive(Debug, Clone)]
pub enum RawCapture {
    /// 802.11 CSI: one [`CsiFrame`] per temporal snapshot (all frames must
    /// share the spatial-stream count) plus per-stream geometry.
    WifiCsi {
        /// Snapshots, oldest first.
        frames: Vec<CsiFrame>,
        /// One entry per spatial stream (link).
        links: Vec<LinkGeometry>,
        /// Age of the oldest snapshot at capture handoff, seconds.
        age_s: f64,
        /// Clock quality in `[0, 1]`.
        clock_quality: f64,
    },
    /// FMCW radar cube `(rx_channels, fast_time_samples, chirps)` before the
    /// range FFT, plus chirp parameters.
    FmcwRadarCube {
        /// Raw ADC cube.
        cube: Array3<Complex64>,
        /// Per-RX-channel geometry.
        links: Vec<LinkGeometry>,
        /// Carrier centre frequency, Hz.
        center_freq_hz: f64,
        /// Sweep bandwidth, Hz.
        bandwidth_hz: f64,
        /// Age in seconds.
        age_s: f64,
        /// Device identifier.
        device_id: String,
    },
    /// UWB channel impulse response `(links, taps, repeats)`.
    UwbCir {
        /// CIR taps.
        taps: Array3<Complex64>,
        /// Per-link geometry.
        links: Vec<LinkGeometry>,
        /// Carrier centre frequency, Hz.
        center_freq_hz: f64,
        /// Bandwidth, Hz.
        bandwidth_hz: f64,
        /// Age in seconds.
        age_s: f64,
        /// Device identifier.
        device_id: String,
    },
    /// Bluetooth 6.0 Channel Sounding capture: per-frequency-step phase
    /// samples plus optional round-trip timing (ADR-281 §2). Phase-based
    /// ranging and RTT are **separate evidence sources** — see
    /// [`ble_cs_range`] for the agreement/divergence contract.
    BleCs {
        /// The channel-sounding frame.
        frame: BleCsFrame,
        /// Initiator→reflector link geometry (single link).
        links: Vec<LinkGeometry>,
        /// Age in seconds.
        age_s: f64,
        /// Device identifier.
        device_id: String,
    },
    /// 5G NR uplink SRS frequency response `(links, comb_res, symbols)` with
    /// a comb factor (only every `comb`-th subcarrier is sounded).
    CellularSrs {
        /// Comb-sampled frequency response.
        comb_res: Array3<Complex64>,
        /// SRS comb factor (2 or 4).
        comb: usize,
        /// Per-link geometry.
        links: Vec<LinkGeometry>,
        /// Carrier centre frequency, Hz.
        center_freq_hz: f64,
        /// Sounded bandwidth, Hz.
        bandwidth_hz: f64,
        /// Age in seconds.
        age_s: f64,
        /// Device identifier.
        device_id: String,
    },
}

impl RawCapture {
    /// Modality this capture belongs to.
    #[must_use]
    pub fn modality(&self) -> RfModality {
        match self {
            Self::WifiCsi { .. } => RfModality::WifiCsi,
            Self::FmcwRadarCube { .. } => RfModality::FmcwRadar,
            Self::UwbCir { .. } => RfModality::UwbCir,
            Self::BleCs { .. } => RfModality::BleCs,
            Self::CellularSrs { .. } => RfModality::CellularSrs,
        }
    }
}

/// Bluetooth Channel Sounding frame (ADR-281 §2): tone phases across
/// frequency steps plus optional round-trip timing.
#[derive(Debug, Clone)]
pub struct BleCsFrame {
    /// Sounded frequencies, Hz (uniformly spaced, ascending).
    pub frequency_steps_hz: Vec<f64>,
    /// Measured round-trip tone phase at each step, radians (wrapped).
    pub phase_samples_rad: Vec<f64>,
    /// Round-trip time measurement, ns, when the mode includes RTT.
    pub round_trip_time_ns: Option<f64>,
}

/// Anomaly classes when the two CS evidence sources disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangingAnomaly {
    /// Phase-based and RTT distances diverge beyond tolerance — multipath
    /// bias, a relay attack, a timing fault, or a calibration problem.
    Divergent,
}

/// Distance evidence from one CS exchange. Phase-based ranging and RTT
/// are deliberately separate: agreement raises confidence, disagreement
/// is surfaced as an anomaly instead of silently averaged away.
#[derive(Debug, Clone)]
pub struct RangingEvidence {
    /// Distance from the unwrapped phase-vs-frequency slope, metres.
    pub phase_distance_m: f64,
    /// Distance from round-trip timing, metres (when measured).
    pub rtt_distance_m: Option<f64>,
    /// Whether the two sources agree within tolerance.
    pub agreement: bool,
    /// Confidence in `[0, 1]` (decays with divergence).
    pub confidence: f64,
    /// Set when the sources diverge.
    pub anomaly: Option<RangingAnomaly>,
}

/// Agreement tolerance between phase and RTT distances, metres.
const CS_AGREEMENT_TOLERANCE_M: f64 = 0.5;

/// Extracts ranging evidence from a CS frame.
///
/// Physics: the round-trip tone phase is `θ(f) = −4π·f·d/c (mod 2π)`, so
/// the unwrapped slope gives `d = |dθ/df|·c/(4π)`. RTT gives
/// `d = rtt·c/2` independently. Cross-validation is the security value:
/// a relay attack that defeats one mechanism generally cannot fake both
/// consistently.
pub fn ble_cs_range(frame: &BleCsFrame) -> Result<RangingEvidence> {
    let n = frame.frequency_steps_hz.len();
    if n < 2 || frame.phase_samples_rad.len() != n {
        return Err(UnifiedError::ShapeMismatch(format!(
            "CS frame needs >= 2 steps with matching phases, got {n} steps / {} phases",
            frame.phase_samples_rad.len()
        )));
    }
    // Boundary validation BEFORE unwrapping. Two DoS classes were found
    // here by `tests/security_boundaries.rs` property testing:
    // (a) a non-finite phase makes a loop-based unwrap spin forever
    //     (+inf minus anything stays +inf);
    // (b) a *finite but huge* phase (e.g. 1e300 rad) makes a loop-based
    //     unwrap take O(|Δ|/2π) ≈ 1e299 iterations.
    // Defense: reject implausible values (a tone phase is physically
    // meaningful mod 2π; |p| > 1e6 rad is garbage), and unwrap in O(1)
    // via modular arithmetic instead of a loop.
    const MAX_PLAUSIBLE_PHASE_RAD: f64 = 1e6;
    if frame
        .phase_samples_rad
        .iter()
        .any(|p| !p.is_finite() || p.abs() > MAX_PLAUSIBLE_PHASE_RAD)
        || frame.frequency_steps_hz.iter().any(|f| !f.is_finite())
    {
        return Err(UnifiedError::InvalidInput(
            "CS frame contains non-finite or implausible phase/frequency samples".into(),
        ));
    }
    if let Some(rtt) = frame.round_trip_time_ns {
        if !rtt.is_finite() {
            return Err(UnifiedError::InvalidInput("non-finite round-trip time".into()));
        }
    }
    let df = frame.frequency_steps_hz[1] - frame.frequency_steps_hz[0];
    if !(df.is_finite() && df > 0.0) {
        return Err(UnifiedError::InvalidInput("frequency steps must ascend uniformly".into()));
    }
    // O(1) unwrap per step: shift each raw phase by the multiple of 2π
    // that lands it within ±π of its predecessor.
    let tau = 2.0 * std::f64::consts::PI;
    let mut unwrapped = Vec::with_capacity(n);
    let mut prev = frame.phase_samples_rad[0];
    unwrapped.push(prev);
    for &p in &frame.phase_samples_rad[1..] {
        let delta = (p - prev + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
        let v = prev + delta;
        unwrapped.push(v);
        prev = v;
    }
    let slope_per_step = crate::math::linear_slope(&unwrapped);
    let c = 299_792_458.0;
    let phase_distance_m = (slope_per_step / df).abs() * c / (4.0 * std::f64::consts::PI);

    let rtt_distance_m = frame.round_trip_time_ns.map(|rtt| rtt * 1e-9 * c / 2.0);
    let (agreement, confidence, anomaly) = match rtt_distance_m {
        None => (true, 0.5, None), // single-source evidence: capped confidence
        Some(d_rtt) => {
            let divergence = (phase_distance_m - d_rtt).abs();
            if divergence <= CS_AGREEMENT_TOLERANCE_M {
                (true, (1.0 - divergence / CS_AGREEMENT_TOLERANCE_M).mul_add(0.5, 0.5), None)
            } else {
                (false, (-divergence).exp().min(0.2), Some(RangingAnomaly::Divergent))
            }
        }
    };
    Ok(RangingEvidence { phase_distance_m, rtt_distance_m, agreement, confidence, anomaly })
}

/// A hardware adapter: normalizes one family of raw captures into the
/// canonical tensor. Object-safe so the registry can hold heterogeneous
/// adapters behind one interface.
pub trait RfAdapter: Send + Sync {
    /// Modality this adapter accepts.
    fn modality(&self) -> RfModality;
    /// Stable hardware identifier, e.g. `"esp32s3-csi"`, `"iwl5300"`.
    fn hardware_id(&self) -> &str;
    /// Normalize a raw capture into the canonical tensor.
    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor>;
}

/// Shared stage 1–3 pipeline: resample to canonical dims, per-link median
/// amplitude normalization, per-(link, snapshot) phase detrend.
///
/// Crate-visible so [`crate::frame::RfFrameV2::to_canonical`] derives the
/// compatibility view through the exact same code path as every adapter
/// (ADR-279 §3 — one normalization, many entry points).
#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_grid(
    modality: RfModality,
    mut grid: Array3<Complex64>, // (links, bins, snapshots) at source resolution
    links: Vec<LinkGeometry>,
    center_freq_hz: f64,
    bandwidth_hz: f64,
    age_s: f64,
    timestamp_ns: u64,
    device_id: String,
    clock_quality: f64,
    uncertainty: f64,
    phase_calibrated: bool,
) -> Result<RfTensor> {
    let (n_links, n_bins, n_snaps) = grid.dim();
    if n_links == 0 || n_bins == 0 || n_snaps == 0 {
        return Err(UnifiedError::ShapeMismatch("empty raw capture".into()));
    }

    // Stage 1: resample bin axis then snapshot axis to canonical dims.
    if n_bins != CANONICAL_BINS {
        let mut resampled = Array3::zeros((n_links, CANONICAL_BINS, n_snaps));
        for l in 0..n_links {
            for s in 0..n_snaps {
                let col: Vec<Complex64> = (0..n_bins).map(|b| grid[[l, b, s]]).collect();
                for (b, v) in resample_complex(&col, CANONICAL_BINS).into_iter().enumerate() {
                    resampled[[l, b, s]] = v;
                }
            }
        }
        grid = resampled;
    }
    let n_snaps_now = grid.dim().2;
    if n_snaps_now != CANONICAL_SNAPSHOTS {
        let mut resampled = Array3::zeros((n_links, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        for l in 0..n_links {
            for b in 0..CANONICAL_BINS {
                let row: Vec<Complex64> = (0..n_snaps_now).map(|s| grid[[l, b, s]]).collect();
                for (s, v) in resample_complex(&row, CANONICAL_SNAPSHOTS).into_iter().enumerate() {
                    resampled[[l, b, s]] = v;
                }
            }
        }
        grid = resampled;
    }

    // Stage 2: per-link median amplitude normalization (gain invariance).
    let mut gain_offset_db = 0.0;
    for l in 0..n_links {
        let amps: Vec<f64> = grid.index_axis(Axis(0), l).iter().map(|z| z.norm()).collect();
        let med = median(&amps);
        if med > 0.0 {
            gain_offset_db += -20.0 * med.log10();
            grid.index_axis_mut(Axis(0), l).mapv_inplace(|z| z / med);
        }
    }
    gain_offset_db /= n_links as f64;

    // Stage 3: phase sanitization — remove per-(link, snapshot) constant
    // offset and linear ramp across bins (CFO residual + sampling-time
    // offset). Skipped when the front-end already phase-calibrates.
    if !phase_calibrated {
        for l in 0..n_links {
            for s in 0..CANONICAL_SNAPSHOTS {
                // Unwrap phase across bins before fitting the ramp.
                let mut phases = Vec::with_capacity(CANONICAL_BINS);
                let mut prev = grid[[l, 0, s]].arg();
                phases.push(prev);
                for b in 1..CANONICAL_BINS {
                    let mut p = grid[[l, b, s]].arg();
                    while p - prev > std::f64::consts::PI {
                        p -= 2.0 * std::f64::consts::PI;
                    }
                    while p - prev < -std::f64::consts::PI {
                        p += 2.0 * std::f64::consts::PI;
                    }
                    phases.push(p);
                    prev = p;
                }
                let slope = linear_slope(&phases);
                let mean = phases.iter().sum::<f64>() / CANONICAL_BINS as f64;
                let mid = (CANONICAL_BINS as f64 - 1.0) / 2.0;
                for b in 0..CANONICAL_BINS {
                    let correction = mean + slope * (b as f64 - mid);
                    let rot = Complex64::new(correction.cos(), -correction.sin());
                    grid[[l, b, s]] *= rot;
                }
            }
        }
    }

    RfTensor::new(
        modality,
        center_freq_hz,
        bandwidth_hz,
        grid,
        links,
        age_s,
        timestamp_ns,
        device_id,
        clock_quality,
        uncertainty,
        CalibrationMeta {
            clock_ppm: (1.0 - clock_quality) * 40.0,
            phase_calibrated: true, // after stage 3 the tensor is detrended
            gain_offset_db,
            baseline_id: None,
        },
    )
}

/// 802.11 CSI adapter (ESP32-S3 / Intel-style spatial-stream CSI).
pub struct WifiCsiAdapter {
    hardware_id: String,
}

impl WifiCsiAdapter {
    /// New adapter for the given hardware id (e.g. `"esp32s3-csi"`).
    #[must_use]
    pub fn new(hardware_id: impl Into<String>) -> Self {
        Self { hardware_id: hardware_id.into() }
    }
}

/// IEEE 802.11 channel-number → center-frequency, in MHz.
///
/// `CsiMetadata::frequency_band` alone only identifies which ~100 MHz-wide
/// band a capture came from (a fixed per-band constant), not the actual
/// channel — using the band constant directly misreports every channel
/// except the one it happens to match (2.4 GHz channel 6, 5 GHz channel 36),
/// by up to tens of MHz on 2.4 GHz and hundreds of MHz on 5/6 GHz. Falls back
/// to the band constant only when the channel number is out of the known
/// range (e.g. `0`, meaning "unknown").
fn channel_center_freq_mhz(band: FrequencyBand, channel: u8) -> f64 {
    match band {
        FrequencyBand::Band2_4GHz => match channel {
            1..=13 => 2407.0 + 5.0 * f64::from(channel),
            14 => 2484.0,
            _ => f64::from(band.center_frequency_mhz()),
        },
        FrequencyBand::Band5GHz if channel > 0 => 5000.0 + 5.0 * f64::from(channel),
        FrequencyBand::Band6GHz if channel > 0 => 5950.0 + 5.0 * f64::from(channel),
        FrequencyBand::Band5GHz | FrequencyBand::Band6GHz => f64::from(band.center_frequency_mhz()),
    }
}

impl RfAdapter for WifiCsiAdapter {
    fn modality(&self) -> RfModality {
        RfModality::WifiCsi
    }

    fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor> {
        let RawCapture::WifiCsi { frames, links, age_s, clock_quality } = raw else {
            return Err(UnifiedError::ModalityMismatch {
                adapter: self.hardware_id.clone(),
                got: raw.modality(),
            });
        };
        if frames.is_empty() {
            return Err(UnifiedError::ShapeMismatch("no CSI frames".into()));
        }
        let n_links = frames[0].num_spatial_streams();
        let n_bins = frames[0].num_subcarriers();
        for f in frames {
            if f.num_spatial_streams() != n_links || f.num_subcarriers() != n_bins {
                return Err(UnifiedError::ShapeMismatch(
                    "inconsistent CSI frame shapes within window".into(),
                ));
            }
        }
        let mut grid = Array3::zeros((n_links, n_bins, frames.len()));
        for (s, f) in frames.iter().enumerate() {
            for l in 0..n_links {
                for b in 0..n_bins {
                    grid[[l, b, s]] = f.data[[l, b]];
                }
            }
        }
        let meta = &frames[0].metadata;
        // SNR → uncertainty: 0 dB SNR ⇒ 1.0 (noise floor), ≥ 40 dB ⇒ 0.0.
        let snr_db =
            frames.iter().map(|f| f.metadata.snr_db()).sum::<f64>() / frames.len() as f64;
        let uncertainty = (1.0 - snr_db / 40.0).clamp(0.0, 1.0);
        let center_freq_hz = channel_center_freq_mhz(meta.frequency_band, meta.channel) * 1e6;
        let ts = &meta.timestamp;
        let timestamp_ns = u64::try_from(ts.seconds).unwrap_or(0) * 1_000_000_000 + u64::from(ts.nanos);
        normalize_grid(
            RfModality::WifiCsi,
            grid,
            links.clone(),
            center_freq_hz,
            f64::from(meta.bandwidth_mhz) * 1e6,
            *age_s,
            timestamp_ns,
            meta.device_id.as_str().to_string(),
            *clock_quality,
            uncertainty,
            false,
        )
    }
}

/// FMCW radar adapter: fast-time DFT to range bins, then shared pipeline.
pub struct FmcwRadarAdapter {
    hardware_id: String,
}

impl FmcwRadarAdapter {
    /// New adapter for the given radar front-end id (e.g. `"mr60bha2"`).
    #[must_use]
    pub fn new(hardware_id: impl Into<String>) -> Self {
        Self { hardware_id: hardware_id.into() }
    }
}

impl RfAdapter for FmcwRadarAdapter {
    fn modality(&self) -> RfModality {
        RfModality::FmcwRadar
    }

    fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor> {
        let RawCapture::FmcwRadarCube { cube, links, center_freq_hz, bandwidth_hz, age_s, device_id } =
            raw
        else {
            return Err(UnifiedError::ModalityMismatch {
                adapter: self.hardware_id.clone(),
                got: raw.modality(),
            });
        };
        let (n_rx, n_fast, n_chirps) = cube.dim();
        if n_rx == 0 || n_fast == 0 || n_chirps == 0 {
            return Err(UnifiedError::ShapeMismatch("empty radar cube".into()));
        }
        // Fast-time DFT: beat-frequency bin b ↔ range bin b. Positive-range
        // half only (b < n_fast/2) mirrors what commercial FMCW parts export.
        let n_range = (n_fast / 2).max(1);
        let mut grid = Array3::zeros((n_rx, n_range, n_chirps));
        for l in 0..n_rx {
            for c in 0..n_chirps {
                for b in 0..n_range {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for t in 0..n_fast {
                        let ang =
                            -2.0 * std::f64::consts::PI * (b as f64) * (t as f64) / (n_fast as f64);
                        acc += cube[[l, t, c]] * Complex64::new(ang.cos(), ang.sin());
                    }
                    grid[[l, b, c]] = acc / n_fast as f64;
                }
            }
        }
        // Range profiles are already phase-meaningful per bin; the linear
        // detrend would erase target range information, so radar declares
        // itself phase-calibrated and only amplitude-normalizes.
        normalize_grid(
            RfModality::FmcwRadar,
            grid,
            links.clone(),
            *center_freq_hz,
            *bandwidth_hz,
            *age_s,
            0,
            device_id.clone(),
            0.9,
            0.1,
            true,
        )
    }
}

/// UWB CIR adapter: taps are already a delay-domain profile.
pub struct UwbCirAdapter {
    hardware_id: String,
}

impl UwbCirAdapter {
    /// New adapter for the given UWB chip id (e.g. `"dw3000"`).
    #[must_use]
    pub fn new(hardware_id: impl Into<String>) -> Self {
        Self { hardware_id: hardware_id.into() }
    }
}

impl RfAdapter for UwbCirAdapter {
    fn modality(&self) -> RfModality {
        RfModality::UwbCir
    }

    fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor> {
        let RawCapture::UwbCir { taps, links, center_freq_hz, bandwidth_hz, age_s, device_id } = raw
        else {
            return Err(UnifiedError::ModalityMismatch {
                adapter: self.hardware_id.clone(),
                got: raw.modality(),
            });
        };
        normalize_grid(
            RfModality::UwbCir,
            taps.clone(),
            links.clone(),
            *center_freq_hz,
            *bandwidth_hz,
            *age_s,
            0,
            device_id.clone(),
            0.95, // UWB timestamps are hardware-disciplined
            0.05,
            true, // delay-domain taps: detrending would destroy ToF structure
        )
    }
}

/// 5G NR SRS adapter: de-comb by interpolating the unsounded subcarriers.
pub struct CellularSrsAdapter {
    hardware_id: String,
}

impl CellularSrsAdapter {
    /// New adapter for the given gNB/xApp source id (e.g. `"oai-srs-xapp"`).
    #[must_use]
    pub fn new(hardware_id: impl Into<String>) -> Self {
        Self { hardware_id: hardware_id.into() }
    }
}

impl RfAdapter for CellularSrsAdapter {
    fn modality(&self) -> RfModality {
        RfModality::CellularSrs
    }

    fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor> {
        let RawCapture::CellularSrs {
            comb_res,
            comb,
            links,
            center_freq_hz,
            bandwidth_hz,
            age_s,
            device_id,
        } = raw
        else {
            return Err(UnifiedError::ModalityMismatch {
                adapter: self.hardware_id.clone(),
                got: raw.modality(),
            });
        };
        if *comb == 0 {
            return Err(UnifiedError::InvalidInput("SRS comb factor must be >= 1".into()));
        }
        let (n_links, n_res, n_sym) = comb_res.dim();
        if n_links == 0 || n_res == 0 || n_sym == 0 {
            return Err(UnifiedError::ShapeMismatch("empty SRS capture".into()));
        }
        // De-comb: the comb-sampled response is a uniform subsampling of the
        // full band, so linear interpolation onto comb×n_res bins restores a
        // dense grid before canonical resampling.
        let dense_bins = n_res * comb;
        let mut grid = Array3::zeros((n_links, dense_bins, n_sym));
        for l in 0..n_links {
            for s in 0..n_sym {
                let col: Vec<Complex64> = (0..n_res).map(|b| comb_res[[l, b, s]]).collect();
                for (b, v) in resample_complex(&col, dense_bins).into_iter().enumerate() {
                    grid[[l, b, s]] = v;
                }
            }
        }
        normalize_grid(
            RfModality::CellularSrs,
            grid,
            links.clone(),
            *center_freq_hz,
            *bandwidth_hz,
            *age_s,
            0,
            device_id.clone(),
            0.99, // gNB reference clock
            0.1,
            false,
        )
    }
}

/// Bluetooth Channel Sounding adapter: tone phasors over frequency steps
/// become the canonical bin axis (delay structure preserved — the ranging
/// ramp *is* the signal, so phase detrending is skipped).
pub struct BleCsAdapter {
    hardware_id: String,
}

impl BleCsAdapter {
    /// New adapter for the given CS radio id (e.g. `"nrf54-cs"`).
    #[must_use]
    pub fn new(hardware_id: impl Into<String>) -> Self {
        Self { hardware_id: hardware_id.into() }
    }
}

impl RfAdapter for BleCsAdapter {
    fn modality(&self) -> RfModality {
        RfModality::BleCs
    }

    fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    fn normalize(&self, raw: &RawCapture) -> Result<RfTensor> {
        let RawCapture::BleCs { frame, links, age_s, device_id } = raw else {
            return Err(UnifiedError::ModalityMismatch {
                adapter: self.hardware_id.clone(),
                got: raw.modality(),
            });
        };
        let n = frame.frequency_steps_hz.len();
        if n < 2 || frame.phase_samples_rad.len() != n {
            return Err(UnifiedError::ShapeMismatch("malformed CS frame".into()));
        }
        let mut grid = Array3::zeros((1, n, 1));
        for (b, theta) in frame.phase_samples_rad.iter().enumerate() {
            grid[[0, b, 0]] = Complex64::from_polar(1.0, *theta);
        }
        let centre = (frame.frequency_steps_hz[0] + frame.frequency_steps_hz[n - 1]) / 2.0;
        let bandwidth = frame.frequency_steps_hz[n - 1] - frame.frequency_steps_hz[0];
        normalize_grid(
            RfModality::BleCs,
            grid,
            links.clone(),
            centre,
            bandwidth.max(1.0),
            *age_s,
            0,
            device_id.clone(),
            0.9,
            0.15,
            true, // the phase ramp is the measurement; never detrend it
        )
    }
}

/// Registry mapping hardware ids to adapters (ADR-274 §2.4). Lookup is
/// fail-closed: unknown hardware is an error, never a silent default.
#[derive(Default)]
pub struct AdapterRegistry {
    adapters: HashMap<String, Box<dyn RfAdapter>>,
}

impl AdapterRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry pre-populated with the four reference adapters.
    #[must_use]
    pub fn with_reference_adapters() -> Self {
        let mut r = Self::new();
        r.register(Box::new(WifiCsiAdapter::new("esp32s3-csi")));
        r.register(Box::new(FmcwRadarAdapter::new("mr60bha2")));
        r.register(Box::new(UwbCirAdapter::new("dw3000")));
        r.register(Box::new(CellularSrsAdapter::new("oai-srs-xapp")));
        r.register(Box::new(BleCsAdapter::new("nrf54-cs")));
        r
    }

    /// Registers an adapter under its hardware id (replaces any previous).
    pub fn register(&mut self, adapter: Box<dyn RfAdapter>) {
        self.adapters.insert(adapter.hardware_id().to_string(), adapter);
    }

    /// Normalizes a capture with the adapter registered for `hardware_id`.
    pub fn normalize(&self, hardware_id: &str, raw: &RawCapture) -> Result<RfTensor> {
        self.adapters
            .get(hardware_id)
            .ok_or_else(|| UnifiedError::UnknownHardware(hardware_id.to_string()))?
            .normalize(raw)
    }

    /// Registered hardware ids (sorted, for deterministic display).
    #[must_use]
    pub fn hardware_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.adapters.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use wifi_densepose_core::types::{CsiMetadata, DeviceId, FrequencyBand};

    fn test_links(n: usize) -> Vec<LinkGeometry> {
        (0..n)
            .map(|i| LinkGeometry {
                tx_pos: [0.0, 0.0, 2.0],
                rx_pos: [4.0, i as f64 * 0.05, 2.0],
            })
            .collect()
    }

    /// CSI frames with a known linear phase ramp + constant offset and a
    /// gain factor — exactly what stages 2–3 must remove.
    fn ramped_frames(n_frames: usize, n_streams: usize, n_sub: usize) -> Vec<CsiFrame> {
        (0..n_frames)
            .map(|s| {
                let meta = CsiMetadata::new(
                    DeviceId::new("esp32s3-a1"),
                    FrequencyBand::Band2_4GHz,
                    6,
                );
                let data = Array2::from_shape_fn((n_streams, n_sub), |(l, b)| {
                    let gain = 3.7 * (1.0 + l as f64);
                    let phase = 0.9 + 0.11 * b as f64 + 0.01 * s as f64;
                    Complex64::new(0.0, phase).exp() * gain
                });
                CsiFrame::new(meta, data)
            })
            .collect()
    }

    #[test]
    fn wifi_adapter_normalizes_shape_gain_and_phase() {
        let adapter = WifiCsiAdapter::new("esp32s3-csi");
        let raw = RawCapture::WifiCsi {
            frames: ramped_frames(12, 2, 114),
            links: test_links(2),
            age_s: 0.02,
            clock_quality: 0.7,
        };
        let t = adapter.normalize(&raw).expect("normalizes");
        assert_eq!(t.dims(), (2, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        assert_eq!(t.modality, RfModality::WifiCsi);

        // Gain invariance: median amplitude per link ≈ 1 after stage 2.
        for l in 0..2 {
            let amps: Vec<f64> =
                t.data.index_axis(Axis(0), l).iter().map(|z| z.norm()).collect();
            assert!((median(&amps) - 1.0).abs() < 1e-9, "link {l} median {:?}", median(&amps));
        }

        // Phase sanitization: the constant-plus-ramp phase must be gone.
        // Bound is 1e-4 rad: complex linear resampling (114→56 bins,
        // 12→8 snapshots) leaves second-order chord-vs-arc phase residue
        // of a few µrad on top of the exact detrend.
        for l in 0..2 {
            for s in 0..CANONICAL_SNAPSHOTS {
                for b in 0..CANONICAL_BINS {
                    assert!(
                        t.data[[l, b, s]].arg().abs() < 1e-4,
                        "residual phase at ({l},{b},{s}): {}",
                        t.data[[l, b, s]].arg()
                    );
                }
            }
        }
    }

    #[test]
    fn radar_adapter_localizes_beat_tone_to_range_bin() {
        // Beat tone at fast-time bin 9 of 64 ⇒ range profile peak at bin 9,
        // which the canonical resampler maps to 9 · (56−1)/(32−1) ≈ 16.
        let n_fast = 64;
        let cube = Array3::from_shape_fn((1, n_fast, 16), |(_, t, _)| {
            let ang = 2.0 * std::f64::consts::PI * 9.0 * t as f64 / n_fast as f64;
            Complex64::new(ang.cos(), ang.sin())
        });
        let adapter = FmcwRadarAdapter::new("mr60bha2");
        let raw = RawCapture::FmcwRadarCube {
            cube,
            links: test_links(1),
            center_freq_hz: 60e9,
            bandwidth_hz: 1e9,
            age_s: 0.0,
            device_id: "mr60".into(),
        };
        let t = adapter.normalize(&raw).expect("normalizes");
        assert_eq!(t.dims(), (1, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        let amps: Vec<f64> =
            (0..CANONICAL_BINS).map(|b| t.data[[0, b, 0]].norm()).collect();
        let peak = amps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected = (9.0 * (CANONICAL_BINS as f64 - 1.0) / 31.0).round() as usize;
        assert!(
            peak.abs_diff(expected) <= 1,
            "range peak at bin {peak}, expected ≈{expected}"
        );
    }

    #[test]
    fn srs_adapter_decombs_and_normalizes() {
        let adapter = CellularSrsAdapter::new("oai-srs-xapp");
        let comb_res = Array3::from_shape_fn((1, 24, 4), |(_, b, _)| {
            Complex64::new(1.0 + 0.01 * b as f64, 0.0)
        });
        let raw = RawCapture::CellularSrs {
            comb_res,
            comb: 2,
            links: test_links(1),
            center_freq_hz: 3.5e9,
            bandwidth_hz: 40e6,
            age_s: 0.001,
            device_id: "gnb-1".into(),
        };
        let t = adapter.normalize(&raw).expect("normalizes");
        assert_eq!(t.dims(), (1, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        assert_eq!(t.modality, RfModality::CellularSrs);
    }

    /// Synthesizes CS phases for a known distance: θ(f) = −4π·f·d/c.
    fn cs_frame(distance_m: f64, rtt_ns: Option<f64>) -> BleCsFrame {
        let c = 299_792_458.0;
        let steps: Vec<f64> = (0..40).map(|k| 2.402e9 + 1e6 * k as f64).collect();
        let phases: Vec<f64> = steps
            .iter()
            .map(|f| {
                let theta = -4.0 * std::f64::consts::PI * f * distance_m / c;
                theta.rem_euclid(2.0 * std::f64::consts::PI)
            })
            .collect();
        BleCsFrame { frequency_steps_hz: steps, phase_samples_rad: phases, round_trip_time_ns: rtt_ns }
    }

    #[test]
    fn ble_cs_phase_ranging_recovers_exact_distance() {
        let c = 299_792_458.0;
        for d in [1.5, 5.0, 12.0] {
            let rtt_ns = 2.0 * d / c * 1e9;
            let ev = ble_cs_range(&cs_frame(d, Some(rtt_ns))).expect("evidence");
            assert!(
                (ev.phase_distance_m - d).abs() < 1e-6,
                "phase ranging {} vs true {d}",
                ev.phase_distance_m
            );
            assert!((ev.rtt_distance_m.unwrap() - d).abs() < 1e-9);
            assert!(ev.agreement, "consistent sources must agree");
            assert!(ev.confidence > 0.9);
            assert!(ev.anomaly.is_none());
        }
    }

    #[test]
    fn ble_cs_flags_relay_style_divergence_instead_of_averaging() {
        // Phase says 5 m; a relay/timing fault inflates RTT to ~51 m.
        let ev = ble_cs_range(&cs_frame(5.0, Some(340.0))).expect("evidence");
        assert!((ev.phase_distance_m - 5.0).abs() < 1e-6);
        assert!(ev.rtt_distance_m.unwrap() > 50.0);
        assert!(!ev.agreement);
        assert_eq!(ev.anomaly, Some(RangingAnomaly::Divergent));
        assert!(ev.confidence < 0.25, "divergent evidence must not be trusted");
    }

    #[test]
    fn ble_cs_adapter_produces_canonical_tensor() {
        let adapter = BleCsAdapter::new("nrf54-cs");
        let raw = RawCapture::BleCs {
            frame: cs_frame(3.0, None),
            links: test_links(1),
            age_s: 0.01,
            device_id: "nrf54-a".into(),
        };
        let t = adapter.normalize(&raw).expect("normalizes");
        assert_eq!(t.dims(), (1, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        assert_eq!(t.modality, RfModality::BleCs);
        // The ranging ramp must survive (no detrend): phase varies across bins.
        let p0 = t.data[[0, 0, 0]].arg();
        let p_mid = t.data[[0, CANONICAL_BINS / 2, 0]].arg();
        assert!((p0 - p_mid).abs() > 1e-3, "phase ramp must be preserved");
    }

    #[test]
    fn registry_is_fail_closed_and_type_safe() {
        let registry = AdapterRegistry::with_reference_adapters();
        assert_eq!(
            registry.hardware_ids(),
            vec!["dw3000", "esp32s3-csi", "mr60bha2", "nrf54-cs", "oai-srs-xapp"]
        );

        // Unknown hardware ⇒ error, never a default adapter.
        let raw = RawCapture::UwbCir {
            taps: Array3::from_elem((1, 32, 4), Complex64::new(1.0, 0.0)),
            links: test_links(1),
            center_freq_hz: 6.5e9,
            bandwidth_hz: 500e6,
            age_s: 0.0,
            device_id: "dw".into(),
        };
        assert!(matches!(
            registry.normalize("unknown-chip", &raw),
            Err(UnifiedError::UnknownHardware(_))
        ));

        // Wrong modality for the adapter ⇒ typed mismatch error.
        assert!(matches!(
            registry.normalize("esp32s3-csi", &raw),
            Err(UnifiedError::ModalityMismatch { .. })
        ));

        // Right adapter succeeds.
        assert!(registry.normalize("dw3000", &raw).is_ok());
    }
}
