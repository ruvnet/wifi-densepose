//! ADR-264 transport-neutral framing for Realtek RTL8720F radar reports.
//!
//! This is a RuView-owned wire contract around the public Ameba API boundary,
//! not a representation of Realtek's private structs. Upstream PR #1336 exposes
//! `wifi_radar_config(struct rtw_radar_action_parm *)`; the report callback ABI
//! remains vendor-gated. Keeping this codec byte-oriented lets host development,
//! replay, and fuzzing proceed without linking the Ameba SDK.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::radio_ops::crc32_ieee;

pub const RTL8720F_RADAR_MAGIC: u32 = 0x3152_5452; // "RTR1" in little endian
pub const RTL8720F_RADAR_VERSION: u8 = 1;
pub const RTL8720F_RADAR_HEADER_LEN: usize = 56;
pub const RTL8720F_RADAR_CRC_LEN: usize = 4;
/// Largest payload that can be carried in one IPv4 UDP datagram.
pub const RTL8720F_RADAR_MAX_FRAME_LEN: usize = 65_507;
pub const RTL8720F_RADAR_MAX_ELEMENTS: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ReportType {
    Cfr = 1,
    RangeNear = 2,
    RangeFar = 3,
    Interference = 4,
    Capabilities = 5,
}

impl TryFrom<u8> for ReportType {
    type Error = RadarParseError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Cfr),
            2 => Ok(Self::RangeNear),
            3 => Ok(Self::RangeFar),
            4 => Ok(Self::Interference),
            5 => Ok(Self::Capabilities),
            _ => Err(RadarParseError::UnknownReportType(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ElementFormat {
    /// TLV/opaque byte payload used by capabilities and interference reports.
    Bytes = 0,
    ComplexI16 = 1,
    ComplexF32 = 2,
    PowerU16 = 3,
    PowerF32 = 4,
}

impl ElementFormat {
    fn bytes_per_element(self) -> usize {
        match self {
            Self::Bytes => 1,
            Self::ComplexI16 | Self::PowerF32 => 4,
            Self::ComplexF32 => 8,
            Self::PowerU16 => 2,
        }
    }
}

impl TryFrom<u8> for ElementFormat {
    type Error = RadarParseError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Bytes),
            1 => Ok(Self::ComplexI16),
            2 => Ok(Self::ComplexF32),
            3 => Ok(Self::PowerU16),
            4 => Ok(Self::PowerF32),
            _ => Err(RadarParseError::UnknownElementFormat(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RadarFlags(pub u16);

impl RadarFlags {
    pub const CALIBRATED: u16 = 1 << 0;
    pub const INTERFERENCE_DETECTED: u16 = 1 << 1;
    pub const SATURATED: u16 = 1 << 2;
    pub const TIME_SYNCHRONIZED: u16 = 1 << 3;
    /// Frame was produced by a simulator/replay generator, never real hardware.
    pub const SYNTHETIC: u16 = 1 << 15;

    pub fn contains(self, flag: u16) -> bool {
        self.0 & flag != 0
    }
}

/// Deterministic, Rust-only RTL8720F source used until hardware is available.
/// It emits the same [`RadarFrame`] objects and wire bytes as the vendor adapter.
pub mod simulator {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SimulatorConfig {
        pub seed: u64,
        pub device_id: u64,
        pub center_freq_khz: u32,
        pub bandwidth_mhz: u16,
        pub frame_period_us: u64,
        pub cfr_bins: u16,
        pub range_bins: u16,
    }

    impl Default for SimulatorConfig {
        fn default() -> Self {
            Self {
                seed: 0x8720_F123_4567_89AB,
                device_id: 0x5254_4C38_3732_3046,
                center_freq_khz: 2_442_000,
                bandwidth_mhz: 40,
                frame_period_us: 15_000,
                cfr_bins: 128,
                range_bins: 32,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Rtl8720fSimulator {
        config: SimulatorConfig,
        rng: u64,
        sequence: u32,
        timestamp_us: u64,
        target_distance_m: f32,
        target_velocity_mps: f32,
    }

    impl Rtl8720fSimulator {
        pub fn new(config: SimulatorConfig) -> Result<Self, RadarParseError> {
            if !matches!(config.bandwidth_mhz, 20 | 40 | 70) {
                return Err(RadarParseError::InvalidBandwidth(config.bandwidth_mhz));
            }
            if config.cfr_bins == 0 || config.range_bins == 0 {
                return Err(RadarParseError::TooManyElements(0));
            }
            Ok(Self {
                rng: config.seed,
                config,
                sequence: 0,
                timestamp_us: 0,
                target_distance_m: 2.0,
                target_velocity_mps: 0.20,
            })
        }

        pub fn config(&self) -> &SimulatorConfig {
            &self.config
        }
        pub fn target_distance_m(&self) -> f32 {
            self.target_distance_m
        }
        pub fn target_velocity_mps(&self) -> f32 {
            self.target_velocity_mps
        }

        /// Emit the boot-time capabilities report as compact TLVs:
        /// type 1 = bandwidth bitset, 2 = CFR bins, 3 = range bins,
        /// 4 = minimum frame period in microseconds.
        pub fn capabilities_frame(&self) -> RadarFrame {
            let bandwidths = 0b0000_0111u8; // 20, 40, 70 MHz
            let mut bytes = vec![1, 1, bandwidths, 2, 2];
            bytes.extend_from_slice(&self.config.cfr_bins.to_le_bytes());
            bytes.extend_from_slice(&[3, 2]);
            bytes.extend_from_slice(&self.config.range_bins.to_le_bytes());
            bytes.extend_from_slice(&[4, 4]);
            bytes.extend_from_slice(&(self.config.frame_period_us as u32).to_le_bytes());
            self.frame(
                ReportType::Capabilities,
                0,
                0,
                RadarPayload::Bytes(bytes),
                0.0,
            )
        }

        pub fn next_frame(&mut self, report_type: ReportType) -> RadarFrame {
            let sequence = self.sequence;
            let timestamp_us = self.timestamp_us;
            self.sequence = self.sequence.wrapping_add(1);
            self.timestamp_us = self.timestamp_us.wrapping_add(self.config.frame_period_us);
            self.advance_target();

            match report_type {
                ReportType::Cfr => {
                    let values = (0..self.config.cfr_bins)
                        .map(|bin| {
                            let phase = bin as f32 * 0.17 + sequence as f32 * 0.05;
                            let noise_i = self.noise_i16(20);
                            let noise_q = self.noise_i16(20);
                            [
                                (phase.cos() * 1800.0) as i16 + noise_i,
                                (phase.sin() * 1800.0) as i16 + noise_q,
                            ]
                        })
                        .collect();
                    self.frame(
                        report_type,
                        sequence,
                        timestamp_us,
                        RadarPayload::ComplexI16(values),
                        self.config.bandwidth_mhz as f32 * 1_000_000.0
                            / self.config.cfr_bins as f32,
                    )
                }
                ReportType::RangeNear | ReportType::RangeFar => {
                    let bin_spacing = match self.config.bandwidth_mhz {
                        70 => 0.33,
                        40 => 0.59,
                        _ => 1.18,
                    };
                    let target_bin = (self.target_distance_m / bin_spacing).round() as usize;
                    let values = (0..self.config.range_bins as usize)
                        .map(|bin| {
                            let distance = bin.abs_diff(target_bin) as f32;
                            let peak = 1000.0 * (-0.5 * distance * distance).exp();
                            let leakage = if report_type == ReportType::RangeNear && bin < 2 {
                                250.0
                            } else {
                                0.0
                            };
                            (peak + leakage + self.noise_f32(12.0)).max(0.0)
                        })
                        .collect();
                    self.frame(
                        report_type,
                        sequence,
                        timestamp_us,
                        RadarPayload::PowerF32(values),
                        bin_spacing,
                    )
                }
                ReportType::Interference => {
                    // TLV: channel-busy %, detected-during-chirp, signed dBm.
                    let busy = (self.next_u32() % 35) as u8;
                    let detected = u8::from(busy > 25);
                    let dbm = (-90i8 + (self.next_u32() % 25) as i8) as u8;
                    let bytes = vec![1, 1, busy, 2, 1, detected, 3, 1, dbm];
                    let mut frame = self.frame(
                        report_type,
                        sequence,
                        timestamp_us,
                        RadarPayload::Bytes(bytes),
                        0.0,
                    );
                    if detected != 0 {
                        frame.flags.0 |= RadarFlags::INTERFERENCE_DETECTED;
                    }
                    frame
                }
                ReportType::Capabilities => self.capabilities_frame(),
            }
        }

        pub fn next_wire(&mut self, report_type: ReportType) -> Result<Vec<u8>, RadarParseError> {
            self.next_frame(report_type).to_bytes()
        }

        fn frame(
            &self,
            report_type: ReportType,
            sequence: u32,
            timestamp_us: u64,
            payload: RadarPayload,
            bin_spacing: f32,
        ) -> RadarFrame {
            RadarFrame {
                report_type,
                sequence,
                timestamp_us,
                device_id: self.config.device_id,
                center_freq_khz: self.config.center_freq_khz,
                bandwidth_mhz: self.config.bandwidth_mhz,
                flags: RadarFlags(RadarFlags::CALIBRATED | RadarFlags::SYNTHETIC),
                antenna_count: 1,
                scale: 1.0,
                bin_spacing,
                calibration_id: 0,
                payload,
            }
        }

        fn advance_target(&mut self) {
            let dt = self.config.frame_period_us as f32 / 1_000_000.0;
            self.target_distance_m += self.target_velocity_mps * dt;
            if self.target_distance_m >= 5.5 || self.target_distance_m <= 0.8 {
                self.target_velocity_mps = -self.target_velocity_mps;
                self.target_distance_m = self.target_distance_m.clamp(0.8, 5.5);
            }
        }

        fn next_u32(&mut self) -> u32 {
            // PCG-style state transition with xorshift output; deterministic and dependency-free.
            self.rng = self
                .rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let word = (((self.rng >> 18) ^ self.rng) >> 27) as u32;
            word.rotate_right((self.rng >> 59) as u32)
        }

        fn noise_i16(&mut self, amplitude: i16) -> i16 {
            (self.next_u32() % (amplitude as u32 * 2 + 1)) as i16 - amplitude
        }

        fn noise_f32(&mut self, amplitude: f32) -> f32 {
            let unit = self.next_u32() as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * amplitude
        }
    }

    impl Iterator for Rtl8720fSimulator {
        type Item = RadarFrame;

        fn next(&mut self) -> Option<Self::Item> {
            let report_type = match self.sequence % 4 {
                0 | 2 => ReportType::Cfr,
                1 => ReportType::RangeNear,
                _ => ReportType::RangeFar,
            };
            Some(self.next_frame(report_type))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RadarPayload {
    Bytes(Vec<u8>),
    ComplexI16(Vec<[i16; 2]>),
    ComplexF32(Vec<[f32; 2]>),
    PowerU16(Vec<u16>),
    PowerF32(Vec<f32>),
}

impl RadarPayload {
    pub fn format(&self) -> ElementFormat {
        match self {
            Self::Bytes(_) => ElementFormat::Bytes,
            Self::ComplexI16(_) => ElementFormat::ComplexI16,
            Self::ComplexF32(_) => ElementFormat::ComplexF32,
            Self::PowerU16(_) => ElementFormat::PowerU16,
            Self::PowerF32(_) => ElementFormat::PowerF32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(v) => v.len(),
            Self::ComplexI16(v) => v.len(),
            Self::ComplexF32(v) => v.len(),
            Self::PowerU16(v) => v.len(),
            Self::PowerF32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn encoded_len(&self) -> usize {
        self.len() * self.format().bytes_per_element()
    }

    fn validate_finite(&self) -> Result<(), RadarParseError> {
        let valid = match self {
            Self::ComplexF32(values) => values.iter().flatten().all(|v| v.is_finite()),
            Self::PowerF32(values) => values.iter().all(|v| v.is_finite()),
            _ => true,
        };
        if valid {
            Ok(())
        } else {
            Err(RadarParseError::NonFiniteValue)
        }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Bytes(values) => out.extend_from_slice(values),
            Self::ComplexI16(values) => values.iter().for_each(|value| {
                out.extend_from_slice(&value[0].to_le_bytes());
                out.extend_from_slice(&value[1].to_le_bytes());
            }),
            Self::ComplexF32(values) => values.iter().for_each(|value| {
                out.extend_from_slice(&value[0].to_le_bytes());
                out.extend_from_slice(&value[1].to_le_bytes());
            }),
            Self::PowerU16(values) => values
                .iter()
                .for_each(|value| out.extend_from_slice(&value.to_le_bytes())),
            Self::PowerF32(values) => values
                .iter()
                .for_each(|value| out.extend_from_slice(&value.to_le_bytes())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RadarFrame {
    pub report_type: ReportType,
    pub sequence: u32,
    pub timestamp_us: u64,
    pub device_id: u64,
    pub center_freq_khz: u32,
    pub bandwidth_mhz: u16,
    pub flags: RadarFlags,
    pub antenna_count: u8,
    pub scale: f32,
    pub bin_spacing: f32,
    pub calibration_id: u32,
    pub payload: RadarPayload,
}

impl RadarFrame {
    pub fn to_bytes(&self) -> Result<Vec<u8>, RadarParseError> {
        self.validate()?;
        let payload_len = self.payload.encoded_len();
        let frame_len = RTL8720F_RADAR_HEADER_LEN
            .checked_add(payload_len)
            .and_then(|value| value.checked_add(RTL8720F_RADAR_CRC_LEN))
            .ok_or(RadarParseError::LengthOverflow)?;
        if frame_len > RTL8720F_RADAR_MAX_FRAME_LEN {
            return Err(RadarParseError::FrameTooLarge(frame_len));
        }

        let mut out = Vec::with_capacity(frame_len);
        out.extend_from_slice(&RTL8720F_RADAR_MAGIC.to_le_bytes());
        out.push(RTL8720F_RADAR_VERSION);
        out.push(self.report_type as u8);
        out.extend_from_slice(&(RTL8720F_RADAR_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&(frame_len as u32).to_le_bytes());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.timestamp_us.to_le_bytes());
        out.extend_from_slice(&self.device_id.to_le_bytes());
        out.extend_from_slice(&self.center_freq_khz.to_le_bytes());
        out.extend_from_slice(&self.bandwidth_mhz.to_le_bytes());
        out.extend_from_slice(&self.flags.0.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u16).to_le_bytes());
        out.push(self.payload.format() as u8);
        out.push(self.antenna_count);
        out.extend_from_slice(&self.scale.to_le_bytes());
        out.extend_from_slice(&self.bin_spacing.to_le_bytes());
        out.extend_from_slice(&self.calibration_id.to_le_bytes());
        debug_assert_eq!(out.len(), RTL8720F_RADAR_HEADER_LEN);
        self.payload.encode_into(&mut out);
        let crc = crc32_ieee(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        Ok(out)
    }

    pub fn from_bytes(input: &[u8]) -> Result<(Self, usize), RadarParseError> {
        if input.len() < RTL8720F_RADAR_HEADER_LEN {
            return Err(RadarParseError::InsufficientData {
                needed: RTL8720F_RADAR_HEADER_LEN,
                got: input.len(),
            });
        }
        let magic = read_u32(input, 0);
        if magic != RTL8720F_RADAR_MAGIC {
            return Err(RadarParseError::InvalidMagic(magic));
        }
        if input[4] != RTL8720F_RADAR_VERSION {
            return Err(RadarParseError::UnsupportedVersion(input[4]));
        }
        let report_type = ReportType::try_from(input[5])?;
        let header_len = read_u16(input, 6) as usize;
        if header_len < RTL8720F_RADAR_HEADER_LEN {
            return Err(RadarParseError::InvalidHeaderLength(header_len));
        }
        let frame_len = read_u32(input, 8) as usize;
        if frame_len > RTL8720F_RADAR_MAX_FRAME_LEN {
            return Err(RadarParseError::FrameTooLarge(frame_len));
        }
        if frame_len < header_len + RTL8720F_RADAR_CRC_LEN {
            return Err(RadarParseError::InvalidFrameLength(frame_len));
        }
        if input.len() < frame_len {
            return Err(RadarParseError::InsufficientData {
                needed: frame_len,
                got: input.len(),
            });
        }

        let element_count = read_u16(input, 40) as usize;
        if element_count > RTL8720F_RADAR_MAX_ELEMENTS {
            return Err(RadarParseError::TooManyElements(element_count));
        }
        let format = ElementFormat::try_from(input[42])?;
        validate_type_format(report_type, format)?;
        let payload_len = element_count
            .checked_mul(format.bytes_per_element())
            .ok_or(RadarParseError::LengthOverflow)?;
        let expected_len = header_len
            .checked_add(payload_len)
            .and_then(|value| value.checked_add(RTL8720F_RADAR_CRC_LEN))
            .ok_or(RadarParseError::LengthOverflow)?;
        if expected_len != frame_len {
            return Err(RadarParseError::PayloadLengthMismatch {
                expected: expected_len,
                got: frame_len,
            });
        }

        let expected_crc = read_u32(input, frame_len - RTL8720F_RADAR_CRC_LEN);
        let actual_crc = crc32_ieee(&input[..frame_len - RTL8720F_RADAR_CRC_LEN]);
        if expected_crc != actual_crc {
            return Err(RadarParseError::CrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let scale = read_f32(input, 44);
        let bin_spacing = read_f32(input, 48);
        if !scale.is_finite() || !bin_spacing.is_finite() {
            return Err(RadarParseError::NonFiniteValue);
        }
        let payload = decode_payload(format, &input[header_len..header_len + payload_len])?;
        let frame = Self {
            report_type,
            sequence: read_u32(input, 12),
            timestamp_us: read_u64(input, 16),
            device_id: read_u64(input, 24),
            center_freq_khz: read_u32(input, 32),
            bandwidth_mhz: read_u16(input, 36),
            flags: RadarFlags(read_u16(input, 38)),
            antenna_count: input[43],
            scale,
            bin_spacing,
            calibration_id: read_u32(input, 52),
            payload,
        };
        frame.validate()?;
        Ok((frame, frame_len))
    }

    fn validate(&self) -> Result<(), RadarParseError> {
        if !matches!(self.bandwidth_mhz, 20 | 40 | 70) {
            return Err(RadarParseError::InvalidBandwidth(self.bandwidth_mhz));
        }
        if self.antenna_count == 0 || self.antenna_count > 8 {
            return Err(RadarParseError::InvalidAntennaCount(self.antenna_count));
        }
        if self.payload.len() > RTL8720F_RADAR_MAX_ELEMENTS
            || self.payload.len() > u16::MAX as usize
        {
            return Err(RadarParseError::TooManyElements(self.payload.len()));
        }
        if !self.scale.is_finite() || !self.bin_spacing.is_finite() {
            return Err(RadarParseError::NonFiniteValue);
        }
        validate_type_format(self.report_type, self.payload.format())?;
        self.payload.validate_finite()
    }
}

fn validate_type_format(
    report_type: ReportType,
    format: ElementFormat,
) -> Result<(), RadarParseError> {
    let valid = match report_type {
        ReportType::Cfr => matches!(
            format,
            ElementFormat::ComplexI16 | ElementFormat::ComplexF32
        ),
        ReportType::RangeNear | ReportType::RangeFar => {
            matches!(format, ElementFormat::PowerU16 | ElementFormat::PowerF32)
        }
        ReportType::Interference | ReportType::Capabilities => format == ElementFormat::Bytes,
    };
    if valid {
        Ok(())
    } else {
        Err(RadarParseError::InvalidTypeFormat {
            report_type,
            format,
        })
    }
}

fn decode_payload(format: ElementFormat, bytes: &[u8]) -> Result<RadarPayload, RadarParseError> {
    let payload = match format {
        ElementFormat::Bytes => RadarPayload::Bytes(bytes.to_vec()),
        ElementFormat::ComplexI16 => RadarPayload::ComplexI16(
            bytes
                .chunks_exact(4)
                .map(|c| [read_i16(c, 0), read_i16(c, 2)])
                .collect(),
        ),
        ElementFormat::ComplexF32 => RadarPayload::ComplexF32(
            bytes
                .chunks_exact(8)
                .map(|c| [read_f32(c, 0), read_f32(c, 4)])
                .collect(),
        ),
        ElementFormat::PowerU16 => {
            RadarPayload::PowerU16(bytes.chunks_exact(2).map(|c| read_u16(c, 0)).collect())
        }
        ElementFormat::PowerF32 => {
            RadarPayload::PowerF32(bytes.chunks_exact(4).map(|c| read_f32(c, 0)).collect())
        }
    };
    payload.validate_finite()?;
    Ok(payload)
}

fn read_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}
fn read_i16(buf: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([buf[offset], buf[offset + 1]])
}
fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}
fn read_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}
fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

#[derive(Debug, Error, PartialEq)]
pub enum RadarParseError {
    #[error("insufficient data: need {needed} bytes, got {got}")]
    InsufficientData { needed: usize, got: usize },
    #[error("invalid RTL8720F radar magic {0:#010x}")]
    InvalidMagic(u32),
    #[error("unsupported RTL8720F radar protocol version {0}")]
    UnsupportedVersion(u8),
    #[error("unknown radar report type {0}")]
    UnknownReportType(u8),
    #[error("unknown radar element format {0}")]
    UnknownElementFormat(u8),
    #[error("invalid header length {0}")]
    InvalidHeaderLength(usize),
    #[error("invalid frame length {0}")]
    InvalidFrameLength(usize),
    #[error("frame is too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("element count exceeds limit: {0}")]
    TooManyElements(usize),
    #[error("length arithmetic overflow")]
    LengthOverflow,
    #[error("payload/frame length mismatch: expected {expected}, got {got}")]
    PayloadLengthMismatch { expected: usize, got: usize },
    #[error("CRC mismatch: encoded {expected:#010x}, computed {actual:#010x}")]
    CrcMismatch { expected: u32, actual: u32 },
    #[error("non-finite floating-point value")]
    NonFiniteValue,
    #[error("invalid bandwidth {0} MHz")]
    InvalidBandwidth(u16),
    #[error("invalid antenna count {0}")]
    InvalidAntennaCount(u8),
    #[error("report {report_type:?} cannot use element format {format:?}")]
    InvalidTypeFormat {
        report_type: ReportType,
        format: ElementFormat,
    },
}

#[cfg(test)]
mod tests {
    use super::simulator::{Rtl8720fSimulator, SimulatorConfig};
    use super::*;

    fn cfr_frame() -> RadarFrame {
        RadarFrame {
            report_type: ReportType::Cfr,
            sequence: 42,
            timestamp_us: 123_456,
            device_id: 0x1122_3344_5566_7788,
            center_freq_khz: 2_442_000,
            bandwidth_mhz: 40,
            flags: RadarFlags(RadarFlags::CALIBRATED | RadarFlags::TIME_SYNCHRONIZED),
            antenna_count: 1,
            scale: 1.0 / 4096.0,
            bin_spacing: 312_500.0,
            calibration_id: 0xAABB_CCDD,
            payload: RadarPayload::ComplexI16(vec![[12, -7], [2048, -2048], [0, 1]]),
        }
    }

    #[test]
    fn cfr_round_trip_and_stream_consumption() {
        let frame = cfr_frame();
        let mut wire = frame.to_bytes().unwrap();
        let encoded_len = wire.len();
        wire.extend_from_slice(&[9, 8, 7]);
        let (decoded, consumed) = RadarFrame::from_bytes(&wire).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(consumed, encoded_len);
    }

    #[test]
    fn every_report_family_round_trips() {
        let payloads = [
            (
                ReportType::RangeNear,
                RadarPayload::PowerU16(vec![1, 2, u16::MAX]),
            ),
            (
                ReportType::RangeFar,
                RadarPayload::PowerF32(vec![0.0, 1.5, 9.25]),
            ),
            (
                ReportType::Interference,
                RadarPayload::Bytes(vec![1, 2, 0x34, 0x12]),
            ),
            (
                ReportType::Capabilities,
                RadarPayload::Bytes(vec![2, 1, 40]),
            ),
        ];
        for (report_type, payload) in payloads {
            let mut frame = cfr_frame();
            frame.report_type = report_type;
            frame.payload = payload;
            let (decoded, _) = RadarFrame::from_bytes(&frame.to_bytes().unwrap()).unwrap();
            assert_eq!(decoded, frame);
        }
    }

    #[test]
    fn single_bit_corruption_is_detected() {
        let mut wire = cfr_frame().to_bytes().unwrap();
        wire[RTL8720F_RADAR_HEADER_LEN + 1] ^= 0x01;
        assert!(matches!(
            RadarFrame::from_bytes(&wire),
            Err(RadarParseError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn truncation_is_reported_without_panicking() {
        let wire = cfr_frame().to_bytes().unwrap();
        for end in 0..wire.len() {
            assert!(RadarFrame::from_bytes(&wire[..end]).is_err());
        }
    }

    #[test]
    fn count_length_mismatch_fails_before_payload_decode() {
        let mut wire = cfr_frame().to_bytes().unwrap();
        wire[40..42].copy_from_slice(&100u16.to_le_bytes());
        let crc_offset = wire.len() - RTL8720F_RADAR_CRC_LEN;
        let crc = crc32_ieee(&wire[..crc_offset]);
        wire[crc_offset..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            RadarFrame::from_bytes(&wire),
            Err(RadarParseError::PayloadLengthMismatch { .. })
        ));
    }

    #[test]
    fn invalid_semantic_combinations_are_rejected() {
        let mut frame = cfr_frame();
        frame.payload = RadarPayload::PowerU16(vec![1]);
        assert!(matches!(
            frame.to_bytes(),
            Err(RadarParseError::InvalidTypeFormat { .. })
        ));
        frame.report_type = ReportType::RangeNear;
        frame.bandwidth_mhz = 80;
        assert_eq!(
            frame.to_bytes().unwrap_err(),
            RadarParseError::InvalidBandwidth(80)
        );
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut frame = cfr_frame();
        frame.scale = f32::NAN;
        assert_eq!(
            frame.to_bytes().unwrap_err(),
            RadarParseError::NonFiniteValue
        );
        frame.scale = 1.0;
        frame.payload = RadarPayload::ComplexF32(vec![[f32::INFINITY, 0.0]]);
        assert_eq!(
            frame.to_bytes().unwrap_err(),
            RadarParseError::NonFiniteValue
        );
    }

    #[test]
    fn arbitrary_short_inputs_never_panic() {
        let mut state = 0x1234_5678u32;
        for len in 0..256usize {
            let mut bytes = vec![0u8; len];
            for byte in &mut bytes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            let result = std::panic::catch_unwind(|| RadarFrame::from_bytes(&bytes));
            assert!(result.is_ok(), "parser panicked for {len} bytes");
        }
    }

    #[test]
    fn simulator_is_deterministic_and_uses_real_wire_boundary() {
        let mut a = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        let mut b = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        for _ in 0..12 {
            let a_wire = a.next_wire(ReportType::Cfr).unwrap();
            let b_wire = b.next_wire(ReportType::Cfr).unwrap();
            assert_eq!(a_wire, b_wire);
            let (decoded, consumed) = RadarFrame::from_bytes(&a_wire).unwrap();
            assert_eq!(consumed, a_wire.len());
            assert!(decoded.flags.contains(RadarFlags::SYNTHETIC));
        }
    }

    #[test]
    fn simulator_range_peak_tracks_ground_truth() {
        let mut sim = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        let frame = sim.next_frame(ReportType::RangeFar);
        let RadarPayload::PowerF32(power) = frame.payload else {
            panic!("expected power bins")
        };
        let peak = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let observed_m = peak as f32 * frame.bin_spacing;
        assert!((observed_m - sim.target_distance_m()).abs() <= frame.bin_spacing);
    }

    #[test]
    fn simulator_capabilities_are_explicitly_synthetic() {
        let sim = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        let frame = sim.capabilities_frame();
        assert_eq!(frame.report_type, ReportType::Capabilities);
        assert!(frame.flags.contains(RadarFlags::SYNTHETIC));
        let wire = frame.to_bytes().unwrap();
        assert_eq!(RadarFrame::from_bytes(&wire).unwrap().0, frame);
    }
}
