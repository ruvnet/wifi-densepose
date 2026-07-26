//! Canonical RF tensor — the single normalization target of every hardware
//! adapter (ADR-274 §2).
//!
//! Every modality (WiFi CSI, cellular SRS, FMCW radar range profiles, UWB
//! CIR) is normalized into the same `(links × bins × snapshots)` complex
//! tensor plus calibration metadata. Downstream code (tokenizer, encoder,
//! Gaussian memory) never sees vendor formats.

use ndarray::Array3;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{Result, UnifiedError};

/// Canonical number of frequency/delay bins after adapter resampling.
///
/// 56 matches the usable-subcarrier count of 20 MHz 802.11n CSI after guard
/// removal (and the 114→56 interpolation already used by
/// `wifi-densepose-train::subcarrier`), so the most common source needs no
/// resampling at all.
pub const CANONICAL_BINS: usize = 56;

/// Canonical number of temporal snapshots per tensor window.
pub const CANONICAL_SNAPSHOTS: usize = 8;

/// RF sensing modality of a capture or tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RfModality {
    /// 802.11 Channel State Information (per-subcarrier frequency response).
    WifiCsi,
    /// 802.11 channel impulse response (delay-domain taps).
    WifiCir,
    /// 802.11 beamforming feedback report (BFI/BFLD path).
    WifiBfReport,
    /// 5G NR uplink Sounding Reference Signal frequency response (O-RAN ISAC path).
    CellularSrs,
    /// FMCW radar range profile (post range-FFT complex bins).
    FmcwRadar,
    /// FMCW radar range–azimuth map.
    FmcwRangeAzimuth,
    /// FMCW radar Doppler–azimuth map.
    FmcwDopplerAzimuth,
    /// Ultra-wideband channel impulse response taps.
    UwbCir,
    /// Bluetooth Channel Sounding tones (phase-based ranging + RTT).
    BleCs,
    /// Output of the ADR-276 synthetic world generator (honest labeling:
    /// tensors of this modality must never be reported as measured).
    Synthetic,
}

/// Transmitter/receiver placement for one link, metres, room frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinkGeometry {
    /// Transmit antenna position `[x, y, z]` in metres.
    pub tx_pos: [f64; 3],
    /// Receive antenna position `[x, y, z]` in metres.
    pub rx_pos: [f64; 3],
}

impl LinkGeometry {
    /// Euclidean TX→RX distance in metres.
    #[must_use]
    pub fn distance_m(&self) -> f64 {
        let d: f64 = (0..3).map(|i| (self.tx_pos[i] - self.rx_pos[i]).powi(2)).sum();
        d.sqrt()
    }
}

/// Calibration contract carried alongside every canonical tensor (ADR-274 §2.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMeta {
    /// Oscillator quality in parts-per-million drift (lower is better).
    pub clock_ppm: f64,
    /// Whether per-link phase offsets have been calibrated out upstream.
    pub phase_calibrated: bool,
    /// Gain offset (dB) applied during normalization, for provenance.
    pub gain_offset_db: f64,
    /// Identifier of the empty-room baseline applied, if any (ADR-135).
    pub baseline_id: Option<String>,
}

impl Default for CalibrationMeta {
    fn default() -> Self {
        Self { clock_ppm: 20.0, phase_calibrated: false, gain_offset_db: 0.0, baseline_id: None }
    }
}

/// The canonical complex RF tensor: `(links, bins, snapshots)` plus the
/// metadata every downstream consumer needs (freshness, geometry, clock
/// quality, uncertainty, provenance).
#[derive(Debug, Clone)]
pub struct RfTensor {
    /// Source modality.
    pub modality: RfModality,
    /// Carrier centre frequency in Hz.
    pub center_freq_hz: f64,
    /// Occupied bandwidth in Hz (span of the bin axis).
    pub bandwidth_hz: f64,
    /// Complex samples, shape `(links, bins, snapshots)`.
    pub data: Array3<Complex64>,
    /// Per-link antenna geometry; `links.len() == data.dim().0`.
    pub links: Vec<LinkGeometry>,
    /// Age of the *oldest* snapshot in seconds at tensor construction time
    /// (the freshness signal the encoder fuses multiplicatively, ADR-274 §3).
    pub sample_age_s: f64,
    /// Capture timestamp, nanoseconds since epoch.
    pub timestamp_ns: u64,
    /// Source device identifier (chipset/firmware provenance key).
    pub device_id: String,
    /// Clock quality in `[0, 1]` (1 = disciplined reference, 0 = free-running).
    pub clock_quality: f64,
    /// Front-end uncertainty proxy in `[0, 1]` (0 = clean, 1 = at noise floor).
    pub uncertainty: f64,
    /// Calibration contract.
    pub calibration: CalibrationMeta,
}

impl RfTensor {
    /// Validated constructor — the only way to build an `RfTensor`.
    ///
    /// Boundary rules enforced here (so downstream modules may assume them):
    /// non-empty dims, geometry length matches the link axis, all samples
    /// finite, frequencies positive, `clock_quality`/`uncertainty` ∈ [0, 1],
    /// `sample_age_s` finite and non-negative.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modality: RfModality,
        center_freq_hz: f64,
        bandwidth_hz: f64,
        data: Array3<Complex64>,
        links: Vec<LinkGeometry>,
        sample_age_s: f64,
        timestamp_ns: u64,
        device_id: String,
        clock_quality: f64,
        uncertainty: f64,
        calibration: CalibrationMeta,
    ) -> Result<Self> {
        let (n_links, n_bins, n_snaps) = data.dim();
        if n_links == 0 || n_bins == 0 || n_snaps == 0 {
            return Err(UnifiedError::ShapeMismatch(format!(
                "empty tensor axis: ({n_links}, {n_bins}, {n_snaps})"
            )));
        }
        if links.len() != n_links {
            return Err(UnifiedError::ShapeMismatch(format!(
                "geometry describes {} links, data has {n_links}",
                links.len()
            )));
        }
        if !(center_freq_hz.is_finite() && center_freq_hz > 0.0) {
            return Err(UnifiedError::InvalidInput(format!(
                "center_freq_hz must be finite and positive, got {center_freq_hz}"
            )));
        }
        if !(bandwidth_hz.is_finite() && bandwidth_hz > 0.0) {
            return Err(UnifiedError::InvalidInput(format!(
                "bandwidth_hz must be finite and positive, got {bandwidth_hz}"
            )));
        }
        if !(0.0..=1.0).contains(&clock_quality) {
            return Err(UnifiedError::InvalidInput(format!(
                "clock_quality must be in [0,1], got {clock_quality}"
            )));
        }
        if !(0.0..=1.0).contains(&uncertainty) {
            return Err(UnifiedError::InvalidInput(format!(
                "uncertainty must be in [0,1], got {uncertainty}"
            )));
        }
        if !(sample_age_s.is_finite() && sample_age_s >= 0.0) {
            return Err(UnifiedError::InvalidInput(format!(
                "sample_age_s must be finite and >= 0, got {sample_age_s}"
            )));
        }
        for link in &links {
            for c in link.tx_pos.iter().chain(link.rx_pos.iter()) {
                if !c.is_finite() {
                    return Err(UnifiedError::InvalidInput(
                        "non-finite antenna coordinate".into(),
                    ));
                }
            }
        }
        if data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(UnifiedError::InvalidInput("non-finite complex sample".into()));
        }
        Ok(Self {
            modality,
            center_freq_hz,
            bandwidth_hz,
            data,
            links,
            sample_age_s,
            timestamp_ns,
            device_id,
            clock_quality,
            uncertainty,
            calibration,
        })
    }

    /// `(links, bins, snapshots)`.
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Carrier wavelength in metres.
    #[must_use]
    pub fn wavelength_m(&self) -> f64 {
        299_792_458.0 / self.center_freq_hz
    }

    /// Delay–Doppler magnitude map for one link (ADR-281 §3): IDFT over the
    /// frequency axis (→ delay bins) crossed with a DFT over the snapshot
    /// axis (→ Doppler bins). Shape `(bins, snapshots)`.
    ///
    /// Delay-Doppler-native modalities (OTFS ISAC, radar) must not be
    /// collapsed into scalar motion energy before storage — this transform
    /// keeps the native representation queryable locally.
    ///
    /// Implemented **separably** — delay IDFT per snapshot, then Doppler
    /// DFT per delay row: `O(B²S + S²B)` instead of the direct form's
    /// `O(B²S²)` (equivalence proven in tests, speedup measured in
    /// `benches/unified_bench.rs`).
    pub fn delay_doppler_map(&self, link: usize) -> crate::Result<ndarray::Array2<f64>> {
        let (n_links, n_bins, n_snaps) = self.dims();
        if link >= n_links {
            return Err(crate::UnifiedError::DimensionMismatch(format!(
                "link {link} out of range ({n_links} links)"
            )));
        }
        // Stage 1: per snapshot, IDFT over bins → delay columns.
        let mut delay = ndarray::Array2::from_elem((n_bins, n_snaps), Complex64::new(0.0, 0.0));
        for s in 0..n_snaps {
            for d in 0..n_bins {
                let mut acc = Complex64::new(0.0, 0.0);
                for b in 0..n_bins {
                    let ang = 2.0 * std::f64::consts::PI * (b * d) as f64 / n_bins as f64;
                    acc += self.data[[link, b, s]] * Complex64::new(ang.cos(), ang.sin());
                }
                delay[[d, s]] = acc / n_bins as f64;
            }
        }
        // Stage 2: per delay row, DFT over snapshots → Doppler.
        let mut out = ndarray::Array2::zeros((n_bins, n_snaps));
        for d in 0..n_bins {
            for v in 0..n_snaps {
                let mut acc = Complex64::new(0.0, 0.0);
                for s in 0..n_snaps {
                    let ang = -2.0 * std::f64::consts::PI * (s * v) as f64 / n_snaps as f64;
                    acc += delay[[d, s]] * Complex64::new(ang.cos(), ang.sin());
                }
                out[[d, v]] = acc.norm() / n_snaps as f64;
            }
        }
        Ok(out)
    }

    /// Direct (non-separable) reference implementation of
    /// [`Self::delay_doppler_map`] — kept for the equivalence test and the
    /// benchmark baseline.
    pub fn delay_doppler_map_direct(&self, link: usize) -> crate::Result<ndarray::Array2<f64>> {
        let (n_links, n_bins, n_snaps) = self.dims();
        if link >= n_links {
            return Err(crate::UnifiedError::DimensionMismatch(format!(
                "link {link} out of range ({n_links} links)"
            )));
        }
        let mut out = ndarray::Array2::zeros((n_bins, n_snaps));
        for d in 0..n_bins {
            for v in 0..n_snaps {
                let mut acc = Complex64::new(0.0, 0.0);
                for b in 0..n_bins {
                    for s in 0..n_snaps {
                        let ang = 2.0 * std::f64::consts::PI
                            * ((b * d) as f64 / n_bins as f64 - (s * v) as f64 / n_snaps as f64);
                        acc += self.data[[link, b, s]] * Complex64::new(ang.cos(), ang.sin());
                    }
                }
                out[[d, v]] = acc.norm() / (n_bins * n_snaps) as f64;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> LinkGeometry {
        LinkGeometry { tx_pos: [0.0, 0.0, 1.0], rx_pos: [3.0, 4.0, 1.0] }
    }

    fn valid_data(links: usize) -> Array3<Complex64> {
        Array3::from_elem((links, CANONICAL_BINS, CANONICAL_SNAPSHOTS), Complex64::new(1.0, 0.0))
    }

    fn build(data: Array3<Complex64>, links: Vec<LinkGeometry>) -> Result<RfTensor> {
        RfTensor::new(
            RfModality::WifiCsi,
            2.437e9,
            20e6,
            data,
            links,
            0.05,
            1_700_000_000_000_000_000,
            "test-device".into(),
            0.8,
            0.1,
            CalibrationMeta::default(),
        )
    }

    #[test]
    fn accepts_valid_tensor_and_reports_dims() {
        let t = build(valid_data(2), vec![link(), link()]).expect("valid tensor");
        assert_eq!(t.dims(), (2, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        // 2.437 GHz → λ ≈ 0.1230 m.
        assert!((t.wavelength_m() - 0.123_017).abs() < 1e-4);
        assert!((t.links[0].distance_m() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn delay_doppler_map_localizes_a_synthetic_target() {
        // A scatterer at delay bin 7 with Doppler bin 3:
        // H[b,s] = exp(-j2πb·7/B) · exp(+j2πs·3/S) ⇒ single peak at (7, 3).
        let (d0, v0) = (7usize, 3usize);
        let data = Array3::from_shape_fn((1, CANONICAL_BINS, CANONICAL_SNAPSHOTS), |(_, b, s)| {
            let ang = -2.0 * std::f64::consts::PI * (b * d0) as f64 / CANONICAL_BINS as f64
                + 2.0 * std::f64::consts::PI * (s * v0) as f64 / CANONICAL_SNAPSHOTS as f64;
            Complex64::new(ang.cos(), ang.sin())
        });
        let t = build(data, vec![link()]).expect("valid tensor");
        let map = t.delay_doppler_map(0).expect("map");
        assert!((map[[d0, v0]] - 1.0).abs() < 1e-9, "peak must be unit at ({d0},{v0})");
        for d in 0..CANONICAL_BINS {
            for v in 0..CANONICAL_SNAPSHOTS {
                if (d, v) != (d0, v0) {
                    assert!(map[[d, v]] < 1e-9, "leakage at ({d},{v}): {}", map[[d, v]]);
                }
            }
        }
        assert!(t.delay_doppler_map(5).is_err(), "out-of-range link is a typed error");
    }

    #[test]
    fn separable_delay_doppler_matches_direct_form() {
        // Random-ish structured tensor: several scatterers + noise floor.
        let data = Array3::from_shape_fn((1, CANONICAL_BINS, CANONICAL_SNAPSHOTS), |(_, b, s)| {
            let a1 = -2.0 * std::f64::consts::PI * (b * 3) as f64 / CANONICAL_BINS as f64
                + 2.0 * std::f64::consts::PI * (s * 2) as f64 / CANONICAL_SNAPSHOTS as f64;
            let a2 = -2.0 * std::f64::consts::PI * (b * 11) as f64 / CANONICAL_BINS as f64
                - 2.0 * std::f64::consts::PI * s as f64 / CANONICAL_SNAPSHOTS as f64;
            Complex64::new(a1.cos() + 0.4 * a2.cos() + 0.01 * ((b * 7 + s) % 5) as f64,
                           a1.sin() + 0.4 * a2.sin())
        });
        let t = build(data, vec![link()]).expect("valid tensor");
        let fast = t.delay_doppler_map(0).expect("separable");
        let direct = t.delay_doppler_map_direct(0).expect("direct");
        for d in 0..CANONICAL_BINS {
            for v in 0..CANONICAL_SNAPSHOTS {
                assert!(
                    (fast[[d, v]] - direct[[d, v]]).abs() < 1e-10,
                    "mismatch at ({d},{v}): {} vs {}",
                    fast[[d, v]],
                    direct[[d, v]]
                );
            }
        }
    }

    #[test]
    fn rejects_geometry_link_mismatch() {
        assert!(matches!(
            build(valid_data(2), vec![link()]),
            Err(UnifiedError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn rejects_non_finite_sample() {
        let mut data = valid_data(1);
        data[[0, 3, 2]] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(build(data, vec![link()]), Err(UnifiedError::InvalidInput(_))));
    }

    #[test]
    fn rejects_out_of_range_scalars() {
        let t = RfTensor::new(
            RfModality::WifiCsi,
            2.437e9,
            20e6,
            valid_data(1),
            vec![link()],
            -1.0, // negative age
            0,
            "d".into(),
            0.8,
            0.1,
            CalibrationMeta::default(),
        );
        assert!(matches!(t, Err(UnifiedError::InvalidInput(_))));

        let t = RfTensor::new(
            RfModality::WifiCsi,
            2.437e9,
            20e6,
            valid_data(1),
            vec![link()],
            0.0,
            0,
            "d".into(),
            1.5, // clock quality out of range
            0.1,
            CalibrationMeta::default(),
        );
        assert!(matches!(t, Err(UnifiedError::InvalidInput(_))));
    }
}
