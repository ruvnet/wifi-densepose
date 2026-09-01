//! Bounded decoder for the public VL53L8CH STM32 ASCII stream.
//!
//! The upstream firmware emits one row per zone:
//! `zone ambient distance_mm bin0 ... binN`. This decoder accumulates exactly
//! one unique row per zone and never mixes a malformed partial frame into the
//! next frame.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::protocol::{
    EvidenceLevel, FrameSource, Provenance, SensorPose, TransientFrame, TransientZone, Vec3,
};
use crate::{MAX_BINS, MAX_ZONES, TRANSIENT_SCHEMA_V1};

/// Static configuration sent to and expected from the ST firmware.
#[derive(Clone, Debug)]
pub struct StDecoderConfig {
    /// Capture session identifier.
    pub session_id: String,
    /// Grid height.
    pub height: u16,
    /// Grid width.
    pub width: u16,
    /// Histogram bins per zone.
    pub num_bins: u16,
    /// First firmware bin retained.
    pub start_bin: u16,
    /// Timing resolution in picoseconds.
    pub bin_width_ps: f32,
    /// Horizontal field of view in degrees.
    pub fov_x_degrees: f32,
    /// Vertical field of view in degrees.
    pub fov_y_degrees: f32,
    /// Add the reported ambient value back to every compact-normalized bin.
    pub add_back_ambient: bool,
    /// Require the upstream firmware's `D` frame-start marker.
    pub require_frame_marker: bool,
    /// Input source.
    pub source: FrameSource,
    /// Evidence level attached at ingest.
    pub evidence_level: EvidenceLevel,
    /// SHA-256 calibration digest or 64 zeroes while calibrating.
    pub calibration_hash: String,
    /// Sensor provenance.
    pub provenance: Provenance,
}

impl StDecoderConfig {
    fn validate(&self) -> Result<(), DecodeError> {
        let zones = usize::from(self.height) * usize::from(self.width);
        if self.height == 0
            || self.width == 0
            || zones > MAX_ZONES
            || !(8..=MAX_BINS).contains(&usize::from(self.num_bins))
        {
            return Err(DecodeError::InvalidConfiguration);
        }
        if !self.bin_width_ps.is_finite()
            || self.bin_width_ps <= 0.0
            || !self.fov_x_degrees.is_finite()
            || !self.fov_y_degrees.is_finite()
            || !(1.0..=120.0).contains(&self.fov_x_degrees)
            || !(1.0..=120.0).contains(&self.fov_y_degrees)
        {
            return Err(DecodeError::InvalidConfiguration);
        }
        Ok(())
    }
}

/// Incremental decoder for a trusted local serial device.
#[derive(Debug)]
pub struct StAsciiDecoder {
    config: StDecoderConfig,
    rows: BTreeMap<u16, ParsedRow>,
    next_sequence: u64,
    began_frame: bool,
}

#[derive(Debug)]
struct ParsedRow {
    ambient: u32,
    distance_m: f32,
    histogram: Vec<u16>,
}

impl StAsciiDecoder {
    /// Create a decoder after checking all configured dimensions.
    pub fn new(config: StDecoderConfig) -> Result<Self, DecodeError> {
        config.validate()?;
        Ok(Self {
            config,
            rows: BTreeMap::new(),
            next_sequence: 0,
            began_frame: false,
        })
    }

    /// Consume one firmware line and return a complete frame when all zones arrive.
    /// A malformed or duplicate row clears the partial frame before returning an
    /// error, preventing cross-frame row splicing.
    pub fn push_line(
        &mut self,
        line: &str,
        captured_at_unix_ms: u64,
        monotonic_ns: u64,
        sensor_pose: SensorPose,
    ) -> Result<Option<TransientFrame>, DecodeError> {
        if line.trim() == "D" {
            self.rows.clear();
            self.began_frame = true;
            return Ok(None);
        }
        if self.config.require_frame_marker && !self.began_frame {
            // Firmware boot messages and stale rows before the next marker are
            // deliberately ignored rather than incorporated into a frame.
            return Ok(None);
        }
        if line.len() > 4_096 || line.bytes().any(|b| b == 0) {
            self.rows.clear();
            self.began_frame = false;
            return Err(DecodeError::MalformedRow);
        }
        let fields: Vec<&str> = line.split_ascii_whitespace().collect();
        if fields.len() != usize::from(self.config.num_bins) + 3 {
            self.rows.clear();
            self.began_frame = false;
            return Err(DecodeError::MalformedRow);
        }
        macro_rules! parse_or_reset {
            ($raw:expr, $kind:ty) => {
                match parse::<$kind>($raw) {
                    Ok(value) => value,
                    Err(error) => {
                        self.rows.clear();
                        self.began_frame = false;
                        return Err(error);
                    }
                }
            };
        }
        let zone_id = parse_or_reset!(fields[0], u16);
        let zone_count = self.config.height * self.config.width;
        if zone_id >= zone_count || self.rows.contains_key(&zone_id) {
            self.rows.clear();
            self.began_frame = false;
            return Err(DecodeError::InvalidZone);
        }
        let ambient = parse_or_reset!(fields[1], u32);
        let distance_mm = parse_or_reset!(fields[2], f32);
        if !distance_mm.is_finite() || !(10.0..=10_000.0).contains(&distance_mm) {
            self.rows.clear();
            self.began_frame = false;
            return Err(DecodeError::InvalidDistance);
        }
        let mut histogram = Vec::with_capacity(usize::from(self.config.num_bins));
        for raw in &fields[3..] {
            let mut count = parse_or_reset!(raw, i64);
            if self.config.add_back_ambient {
                let Some(combined) = count.checked_add(i64::from(ambient)) else {
                    self.rows.clear();
                    self.began_frame = false;
                    return Err(DecodeError::HistogramOverflow);
                };
                count = combined;
            }
            if count > i64::from(u16::MAX) {
                self.rows.clear();
                self.began_frame = false;
                return Err(DecodeError::HistogramOverflow);
            }
            // The pinned public driver clips negative compact-normalized bins
            // to zero. Positive overflow is rejected rather than saturated.
            histogram.push(count.max(0) as u16);
        }
        self.rows.insert(
            zone_id,
            ParsedRow {
                ambient,
                distance_m: distance_mm / 1_000.0,
                histogram,
            },
        );
        if self.rows.len() != usize::from(zone_count) {
            return Ok(None);
        }

        let rows = std::mem::take(&mut self.rows);
        self.began_frame = false;
        let mut zones = Vec::with_capacity(rows.len());
        for (zone_id, row) in rows {
            zones.push(TransientZone {
                zone_id,
                wall_point_m: point_from_zone(
                    zone_id,
                    row.distance_m,
                    self.config.height,
                    self.config.width,
                    self.config.fov_x_degrees,
                    self.config.fov_y_degrees,
                ),
                distance_m: row.distance_m,
                ambient: row.ambient,
                histogram: row.histogram,
            });
        }
        let frame = TransientFrame {
            schema: TRANSIENT_SCHEMA_V1.into(),
            session_id: self.config.session_id.clone(),
            sequence: self.next_sequence,
            captured_at_unix_ms,
            monotonic_ns,
            source: self.config.source,
            evidence_level: self.config.evidence_level,
            bin_width_ps: self.config.bin_width_ps,
            start_bin: self.config.start_bin,
            sensor_pose,
            calibration_hash: self.config.calibration_hash.clone(),
            provenance: self.config.provenance.clone(),
            zones,
        };
        frame.validate().map_err(DecodeError::Contract)?;
        self.next_sequence += 1;
        Ok(Some(frame))
    }

    /// Encode the 13 little-endian `uint16` values expected by the upstream
    /// STM32 firmware. Network or serial I/O remains outside this pure function.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn firmware_config_bytes(
        &self,
        ranging_mode: u16,
        ranging_frequency_hz: u16,
        integration_time_ms: u16,
        subsample: u16,
    ) -> [u8; 26] {
        let values = [
            self.config.height * self.config.width,
            ranging_mode,
            ranging_frequency_hz,
            integration_time_ms,
            self.config.start_bin,
            self.config.num_bins,
            subsample,
            0,
            0,
            1,
            1,
            self.config.width,
            self.config.height,
        ];
        let mut bytes = [0_u8; 26];
        for (index, value) in values.iter().enumerate() {
            bytes[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

fn parse<T: std::str::FromStr>(raw: &str) -> Result<T, DecodeError> {
    raw.parse().map_err(|_| DecodeError::MalformedRow)
}

fn point_from_zone(
    zone_id: u16,
    distance_m: f32,
    height: u16,
    width: u16,
    fov_x_degrees: f32,
    fov_y_degrees: f32,
) -> Vec3 {
    let row = f32::from(zone_id / width) + 0.5;
    let col = f32::from(zone_id % width) + 0.5;
    let yaw = (col / f32::from(width) - 0.5) * fov_x_degrees.to_radians();
    let pitch = (row / f32::from(height) - 0.5) * fov_y_degrees.to_radians();
    let cp = pitch.cos();
    Vec3::new(
        distance_m * cp * yaw.sin(),
        distance_m * pitch.sin(),
        distance_m * cp * yaw.cos(),
    )
}

/// Serial framing failure.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Static configuration exceeds the supported sensor bounds.
    #[error("invalid ST decoder configuration")]
    InvalidConfiguration,
    /// A row had the wrong length or a non-numeric token.
    #[error("malformed ST histogram row")]
    MalformedRow,
    /// Zone id was out of range or duplicated.
    #[error("invalid or duplicate ST zone")]
    InvalidZone,
    /// Direct wall distance was outside 1 cm to 10 m.
    #[error("invalid ST wall distance")]
    InvalidDistance,
    /// A normalized histogram value could not be represented losslessly.
    #[error("ST histogram value exceeds the u16 contract")]
    HistogramOverflow,
    /// The assembled frame violated the shared contract.
    #[error(transparent)]
    Contract(#[from] crate::protocol::ContractError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Provenance, TransientKind};

    fn decoder() -> StAsciiDecoder {
        StAsciiDecoder::new(StDecoderConfig {
            session_id: "session-1".into(),
            height: 2,
            width: 2,
            num_bins: 8,
            start_bin: 30,
            bin_width_ps: 250.0,
            fov_x_degrees: 45.0,
            fov_y_degrees: 45.0,
            add_back_ambient: false,
            require_frame_marker: true,
            source: FrameSource::Live,
            evidence_level: EvidenceLevel::L1Measured,
            calibration_hash: "0".repeat(64),
            provenance: Provenance {
                sensor_id: "st-01".into(),
                sensor_model: "VL53L8CH".into(),
                firmware_version: "test".into(),
                transient_kind: TransientKind::CompactNormalizedHistogram,
                histogram_preserved: true,
                transport: "usb_serial".into(),
            },
        })
        .unwrap()
    }

    #[test]
    fn decodes_one_complete_frame_without_flattening_histograms() {
        let mut d = decoder();
        d.push_line("D", 100, 200, SensorPose::default()).unwrap();
        for zone in 0..4 {
            let result = d
                .push_line(
                    &format!("{zone} 3 800 0 1 2 3 4 5 6 7"),
                    100,
                    200,
                    SensorPose::default(),
                )
                .unwrap();
            if zone < 3 {
                assert!(result.is_none());
            } else {
                let frame = result.unwrap();
                assert_eq!(frame.zones.len(), 4);
                assert_eq!(frame.zones[0].histogram, vec![0, 1, 2, 3, 4, 5, 6, 7]);
                assert_eq!(frame.sequence, 0);
            }
        }
    }

    #[test]
    fn malformed_row_resets_partial_frame() {
        let mut d = decoder();
        d.push_line("D", 1, 1, SensorPose::default()).unwrap();
        d.push_line("0 3 800 0 1 2 3 4 5 6 7", 1, 1, SensorPose::default())
            .unwrap();
        assert!(d.push_line("bad", 1, 1, SensorPose::default()).is_err());
        d.push_line("D", 1, 1, SensorPose::default()).unwrap();
        assert!(d
            .push_line("1 3 800 0 1 2 3 4 5 6 7", 1, 1, SensorPose::default(),)
            .unwrap()
            .is_none());
    }

    #[test]
    fn positive_histogram_overflow_is_rejected_not_saturated() {
        let mut d = decoder();
        d.push_line("D", 1, 1, SensorPose::default()).unwrap();
        assert!(matches!(
            d.push_line("0 3 800 70000 1 2 3 4 5 6 7", 1, 1, SensorPose::default()),
            Err(DecodeError::HistogramOverflow)
        ));
    }

    #[test]
    fn firmware_packet_matches_upstream_little_endian_layout() {
        let bytes = decoder().firmware_config_bytes(1, 30, 10, 1);
        assert_eq!(&bytes[0..2], &4_u16.to_le_bytes());
        assert_eq!(&bytes[8..10], &30_u16.to_le_bytes());
        assert_eq!(&bytes[24..26], &2_u16.to_le_bytes());
    }
}
