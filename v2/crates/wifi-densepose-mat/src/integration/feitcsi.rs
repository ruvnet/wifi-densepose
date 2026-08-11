//! Validated parser for FeitCSI binary CSI records (ADR-292).
//!
//! [FeitCSI](https://feitcsi.kuskosoft.com) is an open-source tool
//! (<https://github.com/KuskoSoft/FeitCSI>) that extracts 802.11ax channel
//! state information from Intel AX200/AX210 NICs at 20/40/80/160 MHz,
//! including the 6 GHz band. FeitCSI is GPL and is used strictly as an
//! *external* capture tool: RuView never links it, never configures the NIC,
//! and only parses the record files/streams its tooling produces.
//!
//! # Record layout (verified against FeitCSI source, `master` @ 2026-08-10)
//!
//! Each record is a packed 272-byte header followed by `csi_data_size` bytes
//! of raw CSI. Layout per `include/Csi.h` (`struct __attribute__((__packed__))
//! RawHeaderData`) and `Csi::save()` in `src/Csi.cpp`, which writes the raw
//! struct memory followed by the CSI buffer. FeitCSI runs on little-endian
//! x86 hosts and dumps native struct memory, so all fields are little-endian.
//!
//! | Offset | Size | Field            | Notes                                   |
//! |-------:|-----:|------------------|-----------------------------------------|
//! | 0      | 4    | `csiDataSize`    | u32, bytes of CSI payload after header  |
//! | 4      | 4    | reserved         | (`space4`)                              |
//! | 8      | 4    | `ftmClock`       | u32                                     |
//! | 12     | 8    | `timestamp`      | u64, device timestamp (microseconds)    |
//! | 20     | 26   | reserved         | (`space20`)                             |
//! | 46     | 1    | `numRx`          | u8, receive antennas                    |
//! | 47     | 1    | `numTx`          | u8, transmit streams                    |
//! | 48     | 4    | reserved         | (`space48`)                             |
//! | 52     | 4    | `numSubCarriers` | u32                                     |
//! | 56     | 4    | reserved         | (`space54`; upstream field name lags    |
//! |        |      |                  | the actual packed offset)               |
//! | 60     | 4    | `rssi1`          | u32, antenna A RSSI                     |
//! | 64     | 4    | `rssi2`          | u32, antenna B RSSI                     |
//! | 68     | 6    | `srcMac`         | source MAC address                      |
//! | 74     | 18   | reserved         | (`space75`)                             |
//! | 92     | 4    | `rateNflag`      | u32, iwlwifi rate flags (see below)     |
//! | 96     | 176  | reserved         | (`space96`, 44 × u32)                   |
//!
//! CSI payload: interleaved little-endian `i16` I/Q pairs, iterated
//! `for rx { for tx { for subcarrier { i16 real, i16 imag } } }` (per the
//! processing loops in `src/Csi.cpp`), i.e. 4 bytes per complex sample and
//! `csiDataSize == numRx * numTx * numSubCarriers * 4`.
//!
//! `rateNflag` uses the iwlwifi rate/flags encoding vendored by FeitCSI in
//! `lib/include/rs.h`: modulation type in bits 8..11 (`RATE_MCS_MOD_TYPE`,
//! 0=CCK, 1=legacy OFDM, 2=HT, 3=VHT, 4=HE, 5=EHT) and channel width in bits
//! 11..14 (`RATE_MCS_CHAN_WIDTH`, 0=20 MHz, 1=40, 2=80, 3=160, 4=320).
//!
//! # Failing loudly on format drift
//!
//! The on-disk format carries **no magic number or version field** (it is the
//! raw iwlwifi notification header), so the "version check" required by
//! ADR-292 is structural and strict:
//!
//! - the declared dimensions must agree exactly with the declared buffer
//!   length ([`FeitCsiError::DimensionMismatch`]);
//! - dimensions are hard-capped ([`MAX_SUBCARRIERS`], [`MAX_ANTENNAS`]) so a
//!   corrupt length can never cause unbounded allocation
//!   ([`FeitCsiError::CapExceeded`]);
//! - `rateNflag` values outside the vendored `rs.h` encoding (unknown
//!   modulation type, or a channel width this parser does not support, e.g.
//!   320 MHz EHT) are rejected as [`FeitCsiError::UnsupportedFormat`] instead
//!   of being misparsed.
//!
//! All input is untrusted: every read is length-checked, allocation is
//! bounded before it happens, and malformed input yields structured errors,
//! never a panic.

use super::hardware_adapter::{
    Bandwidth, CsiMetadata, CsiReadings, DeviceType, FrameControlType, SensorCsiReading,
    SubcarrierMapping, WidebandMeta, WifiBand,
};
use super::AdapterError;
use chrono::{DateTime, Utc};
use num_complex::Complex64;
use std::io::Read;

/// Size of the packed FeitCSI record header in bytes.
pub const HEADER_LEN: usize = 272;

/// Hard cap on the declared subcarrier count (802.11ax 160 MHz HE is 1992;
/// 4096 leaves headroom for future 802.11bf truncated-CIR shapes without
/// permitting unbounded allocation from a corrupt length field).
pub const MAX_SUBCARRIERS: u32 = 4096;

/// Hard cap on declared antenna/stream counts (AX210 is 2x2; 8 is generous).
pub const MAX_ANTENNAS: u8 = 8;

/// Bytes per complex CSI sample (i16 real + i16 imag).
const BYTES_PER_SAMPLE: usize = 4;

/// Upper bound on a single record's total size (header + max payload).
/// Used to bound stream-mode buffering.
pub const MAX_RECORD_BYTES: usize = HEADER_LEN
    + MAX_ANTENNAS as usize * MAX_ANTENNAS as usize * MAX_SUBCARRIERS as usize * BYTES_PER_SAMPLE;

// iwlwifi rate flag encoding, per FeitCSI `lib/include/rs.h`.
const RATE_MCS_MOD_TYPE_POS: u32 = 8;
const RATE_MCS_MOD_TYPE_MSK: u32 = 0x7 << RATE_MCS_MOD_TYPE_POS;
const RATE_MCS_CHAN_WIDTH_POS: u32 = 11;
const RATE_MCS_CHAN_WIDTH_MSK: u32 = 0x7 << RATE_MCS_CHAN_WIDTH_POS;

/// Structured errors for FeitCSI record parsing. Malformed input always
/// yields one of these — the parser never panics on untrusted bytes.
#[derive(Debug, thiserror::Error)]
pub enum FeitCsiError {
    /// The buffer ends before the declared record does.
    #[error("truncated FeitCSI record: need {needed} bytes, got {got}")]
    Truncated {
        /// Bytes required to complete the header or record.
        needed: usize,
        /// Bytes actually available.
        got: usize,
    },

    /// The declared CSI buffer length disagrees with the declared dimensions.
    #[error(
        "FeitCSI dimension mismatch: header declares csi_data_size={declared} \
         but num_rx={num_rx} * num_tx={num_tx} * num_subcarriers={num_subcarriers} \
         * 4 = {expected}"
    )]
    DimensionMismatch {
        /// `csiDataSize` from the header.
        declared: u32,
        /// Size implied by the dimension fields.
        expected: usize,
        /// Declared receive antenna count.
        num_rx: u8,
        /// Declared transmit stream count.
        num_tx: u8,
        /// Declared subcarrier count.
        num_subcarriers: u32,
    },

    /// A dimension field is zero — the record cannot contain CSI.
    #[error("FeitCSI record declares zero-sized dimension: {field}")]
    ZeroDimension {
        /// Which header field was zero.
        field: &'static str,
    },

    /// A dimension exceeds its hard cap; parsing stops before any
    /// allocation sized from the corrupt value.
    #[error("FeitCSI {field}={value} exceeds hard cap {cap} (corrupt or hostile length)")]
    CapExceeded {
        /// Which header field exceeded its cap.
        field: &'static str,
        /// The declared value.
        value: u64,
        /// The enforced cap.
        cap: u64,
    },

    /// `rateNflag` encodes a modulation type or channel width outside the
    /// layout this parser was written against — fail loudly instead of
    /// misparsing a newer/unknown format revision.
    #[error(
        "unsupported FeitCSI rate flags {rate_n_flags:#010x}: {reason} \
         (layout per FeitCSI master @ 2026-08-10; refusing to guess)"
    )]
    UnsupportedFormat {
        /// Raw `rateNflag` value.
        rate_n_flags: u32,
        /// Which sub-field was unrecognized.
        reason: &'static str,
    },

    /// I/O error while reading a record from a file or stream.
    #[error("FeitCSI I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<FeitCsiError> for AdapterError {
    fn from(e: FeitCsiError) -> Self {
        match e {
            FeitCsiError::UnsupportedFormat { .. } => AdapterError::UnsupportedAdapter(e.to_string()),
            FeitCsiError::Io(io) => AdapterError::Io(io),
            _ => AdapterError::DataFormat(e.to_string()),
        }
    }
}

/// Modulation type decoded from `rateNflag` (iwlwifi `RATE_MCS_MOD_TYPE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeitCsiModType {
    /// Legacy CCK (802.11b)
    Cck,
    /// Legacy OFDM (802.11a/g)
    LegacyOfdm,
    /// HT (802.11n)
    Ht,
    /// VHT (802.11ac)
    Vht,
    /// HE (802.11ax)
    He,
    /// EHT (802.11be)
    Eht,
}

/// Channel bandwidth decoded from `rateNflag` (iwlwifi `RATE_MCS_CHAN_WIDTH`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeitCsiBandwidth {
    /// 20 MHz
    Bw20,
    /// 40 MHz
    Bw40,
    /// 80 MHz
    Bw80,
    /// 160 MHz
    Bw160,
}

impl FeitCsiBandwidth {
    /// Bandwidth in MHz.
    pub fn mhz(&self) -> u16 {
        match self {
            Self::Bw20 => 20,
            Self::Bw40 => 40,
            Self::Bw80 => 80,
            Self::Bw160 => 160,
        }
    }

    /// Map to the adapter-level [`Bandwidth`] enum.
    pub fn to_bandwidth(&self) -> Bandwidth {
        match self {
            Self::Bw20 => Bandwidth::HT20,
            Self::Bw40 => Bandwidth::HT40,
            Self::Bw80 => Bandwidth::VHT80,
            Self::Bw160 => Bandwidth::VHT160,
        }
    }
}

/// Validated header fields of one FeitCSI record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeitCsiHeader {
    /// Declared CSI payload size in bytes (already validated against dims).
    pub csi_data_size: u32,
    /// FTM clock value.
    pub ftm_clock: u32,
    /// Device timestamp (microseconds, per upstream usage).
    pub timestamp_us: u64,
    /// Number of receive antennas.
    pub num_rx: u8,
    /// Number of transmit streams.
    pub num_tx: u8,
    /// Native subcarrier count of this frame.
    pub num_subcarriers: u32,
    /// Antenna A RSSI (raw u32 as stored on disk).
    pub rssi1: u32,
    /// Antenna B RSSI (raw u32 as stored on disk).
    pub rssi2: u32,
    /// Source MAC address.
    pub source_mac: [u8; 6],
    /// Raw iwlwifi rate flags.
    pub rate_n_flags: u32,
    /// Modulation type decoded from `rate_n_flags`.
    pub mod_type: FeitCsiModType,
    /// Channel bandwidth decoded from `rate_n_flags`.
    pub bandwidth: FeitCsiBandwidth,
}

/// One parsed FeitCSI record: validated header plus complex CSI, kept at
/// native dimensionality (`num_rx * num_tx * num_subcarriers` samples,
/// iterated rx-major, then tx, then subcarrier).
#[derive(Debug, Clone, PartialEq)]
pub struct FeitCsiRecord {
    /// Validated header.
    pub header: FeitCsiHeader,
    /// Complex CSI samples, flattened `[rx][tx][subcarrier]`.
    pub csi: Vec<Complex64>,
}

impl FeitCsiRecord {
    /// CSI for one (rx, tx) antenna pair as a subcarrier slice, or `None`
    /// when the indices are out of range.
    pub fn antenna_pair(&self, rx: u8, tx: u8) -> Option<&[Complex64]> {
        if rx >= self.header.num_rx || tx >= self.header.num_tx {
            return None;
        }
        let sc = self.header.num_subcarriers as usize;
        let start = (rx as usize * self.header.num_tx as usize + tx as usize) * sc;
        self.csi.get(start..start + sc)
    }

    /// Convert to adapter-level [`CsiReadings`], one [`SensorCsiReading`] per
    /// (rx, tx) antenna pair, carrying native subcarrier count, bandwidth and
    /// band as first-class frame metadata (ADR-292).
    ///
    /// The FeitCSI header does not record channel/band (the capture
    /// configuration owns that), so both are supplied by the caller. The
    /// timestamp is derived deterministically from the record's own device
    /// timestamp, never from wall-clock, so file replay is reproducible.
    pub fn to_readings(&self, band: WifiBand, channel: u8) -> CsiReadings {
        let sc = self.header.num_subcarriers as usize;
        let mut readings = Vec::with_capacity(self.header.num_rx as usize * self.header.num_tx as usize);

        // Interpret the on-disk u32 RSSI as two's-complement dBm (captures
        // store negative dBm values in the raw register field).
        let rssi_dbm = |raw: u32| raw as i32 as f64;
        let rssi = rssi_dbm(self.header.rssi1).max(rssi_dbm(self.header.rssi2));

        let mac = self.header.source_mac;
        let tx_mac = format!(
            "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
        );

        for rx in 0..self.header.num_rx {
            for tx in 0..self.header.num_tx {
                let pair = self
                    .antenna_pair(rx, tx)
                    .expect("indices bounded by validated header dims");
                let mut amplitudes = Vec::with_capacity(sc);
                let mut phases = Vec::with_capacity(sc);
                for c in pair {
                    amplitudes.push(c.norm());
                    phases.push(c.im.atan2(c.re));
                }
                readings.push(SensorCsiReading {
                    sensor_id: format!("feitcsi_rx{rx}_tx{tx}"),
                    amplitudes,
                    phases,
                    rssi,
                    noise_floor: -92.0,
                    tx_mac: Some(tx_mac.clone()),
                    rx_mac: None,
                    sequence_num: None,
                });
            }
        }

        // Deterministic timestamp from the device clock (microseconds since
        // capture epoch); replay of the same bytes yields the same output.
        let timestamp = DateTime::<Utc>::from_timestamp_micros(self.header.timestamp_us as i64)
            .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid"));

        CsiReadings {
            timestamp,
            readings,
            metadata: CsiMetadata {
                device_type: DeviceType::FeitCsi,
                channel,
                bandwidth: self.header.bandwidth.to_bandwidth(),
                num_subcarriers: sc,
                rssi: Some(rssi),
                noise_floor: None,
                fc_type: FrameControlType::Data,
                wideband: Some(WidebandMeta {
                    band,
                    bandwidth_mhz: self.header.bandwidth.mhz(),
                    native_subcarriers: sc,
                    mapping: None,
                }),
            },
        }
    }
}

/// Validate the fixed-size header. Enforces caps and dimension/length
/// consistency BEFORE any allocation is sized from untrusted fields.
fn validate_header(h: &[u8; HEADER_LEN]) -> Result<FeitCsiHeader, FeitCsiError> {
    let u32_at = |off: usize| u32::from_le_bytes([h[off], h[off + 1], h[off + 2], h[off + 3]]);

    let csi_data_size = u32_at(0);
    let ftm_clock = u32_at(8);
    let timestamp_us = u64::from_le_bytes([
        h[12], h[13], h[14], h[15], h[16], h[17], h[18], h[19],
    ]);
    let num_rx = h[46];
    let num_tx = h[47];
    let num_subcarriers = u32_at(52);
    let rssi1 = u32_at(60);
    let rssi2 = u32_at(64);
    let mut source_mac = [0u8; 6];
    source_mac.copy_from_slice(&h[68..74]);
    let rate_n_flags = u32_at(92);

    // Zero dimensions cannot carry CSI.
    if num_rx == 0 {
        return Err(FeitCsiError::ZeroDimension { field: "num_rx" });
    }
    if num_tx == 0 {
        return Err(FeitCsiError::ZeroDimension { field: "num_tx" });
    }
    if num_subcarriers == 0 {
        return Err(FeitCsiError::ZeroDimension {
            field: "num_subcarriers",
        });
    }

    // Hard caps: corrupt lengths must not size any allocation.
    if num_rx > MAX_ANTENNAS {
        return Err(FeitCsiError::CapExceeded {
            field: "num_rx",
            value: num_rx as u64,
            cap: MAX_ANTENNAS as u64,
        });
    }
    if num_tx > MAX_ANTENNAS {
        return Err(FeitCsiError::CapExceeded {
            field: "num_tx",
            value: num_tx as u64,
            cap: MAX_ANTENNAS as u64,
        });
    }
    if num_subcarriers > MAX_SUBCARRIERS {
        return Err(FeitCsiError::CapExceeded {
            field: "num_subcarriers",
            value: num_subcarriers as u64,
            cap: MAX_SUBCARRIERS as u64,
        });
    }

    // Dimensions vs declared buffer length. Capped dims bound this product
    // at 8 * 8 * 4096 * 4 = 1 MiB, so the arithmetic cannot overflow usize.
    let expected =
        num_rx as usize * num_tx as usize * num_subcarriers as usize * BYTES_PER_SAMPLE;
    if csi_data_size as usize != expected {
        return Err(FeitCsiError::DimensionMismatch {
            declared: csi_data_size,
            expected,
            num_rx,
            num_tx,
            num_subcarriers,
        });
    }

    // Format-revision check on the rate flags: reject encodings outside the
    // vendored rs.h layout this parser was written against.
    let mod_type = match (rate_n_flags & RATE_MCS_MOD_TYPE_MSK) >> RATE_MCS_MOD_TYPE_POS {
        0 => FeitCsiModType::Cck,
        1 => FeitCsiModType::LegacyOfdm,
        2 => FeitCsiModType::Ht,
        3 => FeitCsiModType::Vht,
        4 => FeitCsiModType::He,
        5 => FeitCsiModType::Eht,
        _ => {
            return Err(FeitCsiError::UnsupportedFormat {
                rate_n_flags,
                reason: "unknown modulation type (bits 8..11)",
            })
        }
    };
    let bandwidth = match (rate_n_flags & RATE_MCS_CHAN_WIDTH_MSK) >> RATE_MCS_CHAN_WIDTH_POS {
        0 => FeitCsiBandwidth::Bw20,
        1 => FeitCsiBandwidth::Bw40,
        2 => FeitCsiBandwidth::Bw80,
        3 => FeitCsiBandwidth::Bw160,
        // 4 = 320 MHz (EHT); not supported by this ingest revision.
        _ => {
            return Err(FeitCsiError::UnsupportedFormat {
                rate_n_flags,
                reason: "unsupported channel width (bits 11..14; 320 MHz+ not supported)",
            })
        }
    };

    Ok(FeitCsiHeader {
        csi_data_size,
        ftm_clock,
        timestamp_us,
        num_rx,
        num_tx,
        num_subcarriers,
        rssi1,
        rssi2,
        source_mac,
        rate_n_flags,
        mod_type,
        bandwidth,
    })
}

/// Decode a validated CSI payload (interleaved little-endian i16 I/Q pairs)
/// into complex samples. The caller has already validated `payload.len()`
/// against the header dimensions, so this performs exactly one bounded
/// allocation (`chunks_exact` is an exact-size iterator, so `collect`
/// reserves the final length up front) and the conversion loop itself is
/// allocation-free.
#[inline]
fn decode_csi(payload: &[u8]) -> Vec<Complex64> {
    payload
        .chunks_exact(BYTES_PER_SAMPLE)
        .map(|sample| {
            let re = i16::from_le_bytes([sample[0], sample[1]]) as f64;
            let im = i16::from_le_bytes([sample[2], sample[3]]) as f64;
            Complex64::new(re, im)
        })
        .collect()
}

/// Parse one record from the front of `buf`.
///
/// Returns the record and the number of bytes consumed, so callers can walk
/// a multi-record capture. All validation happens before any allocation is
/// sized from untrusted fields; malformed input yields a structured
/// [`FeitCsiError`], never a panic. The header is read in place (no copy)
/// and the payload is converted directly from the input slice, so the CSI
/// bytes are traversed exactly once.
pub fn parse_record(buf: &[u8]) -> Result<(FeitCsiRecord, usize), FeitCsiError> {
    if buf.len() < HEADER_LEN {
        return Err(FeitCsiError::Truncated {
            needed: HEADER_LEN,
            got: buf.len(),
        });
    }
    let header_bytes: &[u8; HEADER_LEN] = buf[..HEADER_LEN]
        .try_into()
        .expect("slice length checked above");
    let header = validate_header(header_bytes)?;

    let payload_len = header.csi_data_size as usize;
    let total = HEADER_LEN + payload_len;
    if buf.len() < total {
        return Err(FeitCsiError::Truncated {
            needed: total,
            got: buf.len(),
        });
    }

    // payload_len == validated expected size <= 1 MiB: bounded allocation.
    let csi = decode_csi(&buf[HEADER_LEN..total]);

    Ok((FeitCsiRecord { header, csi }, total))
}

/// Read one record from a byte stream (file, pipe, socket wrapper).
///
/// Returns `Ok(None)` on clean EOF (no bytes before end-of-stream); a
/// mid-record EOF is a [`FeitCsiError::Truncated`] error. `read_exact`
/// semantics mean a blocking pipe simply waits for the writer, so the same
/// code path serves file replay and streaming mode.
pub fn read_one_record<R: Read>(
    reader: &mut R,
) -> Result<Option<(FeitCsiRecord, usize)>, FeitCsiError> {
    let mut scratch = Vec::new();
    read_one_record_with_scratch(reader, &mut scratch)
}

/// [`read_one_record`] with a caller-owned scratch buffer for the raw
/// payload, so long-running replay/stream loops reuse one allocation across
/// records instead of allocating per record. The scratch is only ever
/// resized to the header-validated payload length (<= 1 MiB), never to an
/// untrusted value.
fn read_one_record_with_scratch<R: Read>(
    reader: &mut R,
    scratch: &mut Vec<u8>,
) -> Result<Option<(FeitCsiRecord, usize)>, FeitCsiError> {
    let mut header_bytes = [0u8; HEADER_LEN];
    let mut filled = 0usize;
    while filled < HEADER_LEN {
        let n = reader.read(&mut header_bytes[filled..])?;
        if n == 0 {
            if filled == 0 {
                return Ok(None); // clean EOF between records
            }
            return Err(FeitCsiError::Truncated {
                needed: HEADER_LEN,
                got: filled,
            });
        }
        filled += n;
    }

    let header = validate_header(&header_bytes)?;
    let payload_len = header.csi_data_size as usize; // validated, <= 1 MiB
    scratch.resize(payload_len, 0);
    reader.read_exact(scratch).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            FeitCsiError::Truncated {
                needed: HEADER_LEN + payload_len,
                got: HEADER_LEN, // header complete, payload short
            }
        } else {
            FeitCsiError::Io(e)
        }
    })?;

    let csi = decode_csi(scratch);

    Ok(Some((FeitCsiRecord { header, csi }, HEADER_LEN + payload_len)))
}

/// Streaming reader over any [`Read`] source (recorded capture file, or a
/// path/pipe an external FeitCSI process writes to). RuView never configures
/// the NIC — FeitCSI's own tooling owns capture, per least-authority.
///
/// Holds a reusable payload scratch buffer so a long-running stream performs
/// one bounded raw-byte allocation total (plus the per-record `Vec<Complex64>`
/// output), rather than one raw-byte allocation per record.
pub struct FeitCsiStreamReader<R: Read> {
    inner: R,
    scratch: Vec<u8>,
}

impl<R: Read> FeitCsiStreamReader<R> {
    /// Wrap a byte source.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            scratch: Vec::new(),
        }
    }

    /// Read the next record; `Ok(None)` on clean end-of-stream.
    pub fn read_next(&mut self) -> Result<Option<FeitCsiRecord>, FeitCsiError> {
        Ok(read_one_record_with_scratch(&mut self.inner, &mut self.scratch)?.map(|(rec, _)| rec))
    }
}

/// Deterministic file-replay reader for recorded FeitCSI captures.
pub struct FeitCsiFileReader {
    stream: FeitCsiStreamReader<std::io::BufReader<std::fs::File>>,
}

impl FeitCsiFileReader {
    /// Open a recorded capture for sequential replay.
    pub fn open(path: &str) -> Result<Self, FeitCsiError> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            stream: FeitCsiStreamReader::new(std::io::BufReader::new(file)),
        })
    }

    /// Read the next record; `Ok(None)` at end of capture.
    pub fn read_next(&mut self) -> Result<Option<FeitCsiRecord>, FeitCsiError> {
        self.stream.read_next()
    }
}

/// Convert wideband readings to the pipeline's subcarrier width via the
/// existing interpolation path (`wifi-densepose-signal`'s Catmull-Rom cubic
/// resampler from ADR-027), recording the native → pipeline mapping in frame
/// metadata so downstream consumers know the true spectral resolution
/// (ADR-292 §3).
///
/// This is the ONLY sanctioned native→pipeline conversion: it is explicit,
/// and the mapping is auditable in `metadata.wideband.mapping`.
pub fn resample_readings_to_pipeline(
    readings: &CsiReadings,
    pipeline_subcarriers: usize,
) -> Result<CsiReadings, AdapterError> {
    let normalizer =
        wifi_densepose_signal::HardwareNormalizer::with_canonical_subcarriers(pipeline_subcarriers)
            .map_err(|e| AdapterError::Config(format!("invalid pipeline width: {e}")))?;

    let native = readings.metadata.num_subcarriers;
    let mut out = readings.clone();
    for reading in &mut out.readings {
        reading.amplitudes = normalizer.resample_to_canonical(&reading.amplitudes);
        reading.phases = normalizer.resample_to_canonical(&reading.phases);
    }
    out.metadata.num_subcarriers = pipeline_subcarriers;

    let mapping = SubcarrierMapping {
        native,
        pipeline: pipeline_subcarriers,
        method: "catmull-rom-cubic",
    };
    match &mut out.metadata.wideband {
        Some(wb) => wb.mapping = Some(mapping),
        None => {
            // Preserve provenance even for frames that arrived without
            // wideband metadata: native resolution is still recorded.
            out.metadata.wideband = Some(WidebandMeta {
                band: WifiBand::Band5GHz,
                bandwidth_mhz: readings.metadata.bandwidth.mhz(),
                native_subcarriers: native,
                mapping: Some(mapping),
            });
        }
    }
    Ok(out)
}

/// Deterministic synthetic-fixture generation for tests and benchmarks.
///
/// FeitCSI capture fixtures are always generated in code (never checked in
/// as binary files, per repo policy). CSI samples are a pure function of the
/// sample index, so round-trips are checkable and replay is reproducible.
pub mod synth {
    use super::{BYTES_PER_SAMPLE, HEADER_LEN, RATE_MCS_CHAN_WIDTH_POS, RATE_MCS_MOD_TYPE_POS};

    /// Build the bytes of one synthetic FeitCSI record with the given
    /// dimensions, `rateNflag` channel-width value (0=20 MHz .. 3=160 MHz),
    /// modulation-type value (4=HE), and device timestamp.
    pub fn record_bytes(
        num_rx: u8,
        num_tx: u8,
        num_subcarriers: u32,
        chan_width_val: u32,
        mod_type_val: u32,
        timestamp_us: u64,
    ) -> Vec<u8> {
        let samples = num_rx as usize * num_tx as usize * num_subcarriers as usize;
        let csi_data_size = (samples * BYTES_PER_SAMPLE) as u32;

        let mut h = vec![0u8; HEADER_LEN];
        h[0..4].copy_from_slice(&csi_data_size.to_le_bytes());
        h[8..12].copy_from_slice(&0xAABBCCDDu32.to_le_bytes()); // ftm_clock
        h[12..20].copy_from_slice(&timestamp_us.to_le_bytes());
        h[46] = num_rx;
        h[47] = num_tx;
        h[52..56].copy_from_slice(&num_subcarriers.to_le_bytes());
        h[60..64].copy_from_slice(&(-42i32 as u32).to_le_bytes()); // rssi1
        h[64..68].copy_from_slice(&(-45i32 as u32).to_le_bytes()); // rssi2
        h[68..74].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        let rate_n_flags =
            (mod_type_val << RATE_MCS_MOD_TYPE_POS) | (chan_width_val << RATE_MCS_CHAN_WIDTH_POS);
        h[92..96].copy_from_slice(&rate_n_flags.to_le_bytes());

        for i in 0..samples {
            let re = (i as i64 % 200 - 100) as i16;
            let im = (i as i64 % 97 - 48) as i16;
            h.extend_from_slice(&re.to_le_bytes());
            h.extend_from_slice(&im.to_le_bytes());
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::synth::record_bytes as make_record_bytes;
    use super::*;

    /// HE (802.11ax) records at 20/80/160 MHz shapes parse with correct
    /// native dimensions, decoded bandwidth, and sample round-trip.
    #[test]
    fn test_parse_valid_he_shapes() {
        // (chan_width_val, expected bandwidth, HE tone count)
        let shapes = [
            (0u32, FeitCsiBandwidth::Bw20, 242u32),
            (2u32, FeitCsiBandwidth::Bw80, 996u32),
            (3u32, FeitCsiBandwidth::Bw160, 1992u32),
        ];
        for (cw, expected_bw, sc) in shapes {
            let bytes = make_record_bytes(2, 1, sc, cw, 4, 1_000_000);
            let (rec, consumed) = parse_record(&bytes).expect("valid record must parse");
            assert_eq!(consumed, bytes.len());
            assert_eq!(rec.header.num_subcarriers, sc);
            assert_eq!(rec.header.bandwidth, expected_bw);
            assert_eq!(rec.header.mod_type, FeitCsiModType::He);
            assert_eq!(rec.header.num_rx, 2);
            assert_eq!(rec.header.num_tx, 1);
            assert_eq!(rec.csi.len(), 2 * sc as usize);
            // Deterministic sample round-trip: index 5 → re = 5-100... check
            // the generator formula directly.
            let i = 5usize;
            assert_eq!(rec.csi[i].re, (i as i64 % 200 - 100) as f64);
            assert_eq!(rec.csi[i].im, (i as i64 % 97 - 48) as f64);
            // Antenna-pair accessor yields native-width slices.
            assert_eq!(rec.antenna_pair(0, 0).unwrap().len(), sc as usize);
            assert_eq!(rec.antenna_pair(1, 0).unwrap().len(), sc as usize);
            assert!(rec.antenna_pair(2, 0).is_none());
        }
    }

    /// A truncated buffer (header or payload cut short) is a structured
    /// error, not a panic.
    #[test]
    fn test_truncated_buffer() {
        let bytes = make_record_bytes(1, 1, 242, 0, 4, 0);

        // Header cut short.
        let r = parse_record(&bytes[..100]);
        assert!(matches!(
            r,
            Err(FeitCsiError::Truncated { needed, got: 100 }) if needed == HEADER_LEN
        ));

        // Payload cut short.
        let r = parse_record(&bytes[..bytes.len() - 1]);
        assert!(matches!(r, Err(FeitCsiError::Truncated { .. })));

        // Empty buffer.
        assert!(matches!(
            parse_record(&[]),
            Err(FeitCsiError::Truncated { .. })
        ));
    }

    /// csiDataSize that disagrees with the declared dimensions is rejected.
    #[test]
    fn test_dimension_mismatch() {
        let mut bytes = make_record_bytes(1, 1, 242, 0, 4, 0);
        // Corrupt the declared size (off by 4 bytes).
        let bad = (242 * BYTES_PER_SAMPLE as u32) + 4;
        bytes[0..4].copy_from_slice(&bad.to_le_bytes());
        let r = parse_record(&bytes);
        assert!(matches!(
            r,
            Err(FeitCsiError::DimensionMismatch {
                declared,
                expected,
                ..
            }) if declared == bad && expected == 242 * BYTES_PER_SAMPLE
        ));
    }

    /// Unknown rate-flag encodings fail loudly (the format has no magic, so
    /// this is the version/format check): 320 MHz width and out-of-range
    /// modulation types are refused rather than misparsed.
    #[test]
    fn test_unsupported_format_fails_loudly() {
        // Channel width value 4 = 320 MHz (EHT) — unsupported.
        let bytes = make_record_bytes(1, 1, 242, 4, 5, 0);
        assert!(matches!(
            parse_record(&bytes),
            Err(FeitCsiError::UnsupportedFormat { .. })
        ));

        // Modulation type 7 — outside the vendored rs.h encoding.
        let bytes = make_record_bytes(1, 1, 242, 0, 7, 0);
        assert!(matches!(
            parse_record(&bytes),
            Err(FeitCsiError::UnsupportedFormat { .. })
        ));
    }

    /// Corrupt dimension fields beyond the hard caps are rejected BEFORE any
    /// allocation is sized from them — a hostile length cannot cause
    /// unbounded allocation.
    #[test]
    fn test_allocation_cap_enforcement() {
        // Subcarrier count over the cap, with a consistent (huge) size field.
        let mut h = vec![0u8; HEADER_LEN];
        let huge_sc: u32 = 100_000;
        h[46] = 1;
        h[47] = 1;
        h[52..56].copy_from_slice(&huge_sc.to_le_bytes());
        h[0..4].copy_from_slice(&(huge_sc * 4).to_le_bytes());
        h[92..96].copy_from_slice(&(4u32 << RATE_MCS_MOD_TYPE_POS).to_le_bytes());
        let r = parse_record(&h);
        assert!(matches!(
            r,
            Err(FeitCsiError::CapExceeded {
                field: "num_subcarriers",
                value,
                cap,
            }) if value == huge_sc as u64 && cap == MAX_SUBCARRIERS as u64
        ));

        // Antenna count over the cap.
        let mut h = vec![0u8; HEADER_LEN];
        h[46] = 9; // num_rx > MAX_ANTENNAS
        h[47] = 1;
        h[52..56].copy_from_slice(&242u32.to_le_bytes());
        h[0..4].copy_from_slice(&(9 * 242 * 4u32).to_le_bytes());
        assert!(matches!(
            parse_record(&h),
            Err(FeitCsiError::CapExceeded { field: "num_rx", .. })
        ));

        // Zero dimension.
        let mut h = vec![0u8; HEADER_LEN];
        h[46] = 0;
        h[47] = 1;
        h[52..56].copy_from_slice(&242u32.to_le_bytes());
        assert!(matches!(
            parse_record(&h),
            Err(FeitCsiError::ZeroDimension { field: "num_rx" })
        ));
    }

    /// Streaming reader over an in-memory multi-record capture: reads all
    /// records in order, then clean EOF.
    #[test]
    fn test_stream_reader_multi_record() {
        let mut capture = Vec::new();
        for ts in [10u64, 20, 30] {
            capture.extend_from_slice(&make_record_bytes(1, 1, 242, 0, 4, ts));
        }
        let mut reader = FeitCsiStreamReader::new(std::io::Cursor::new(capture));
        let mut timestamps = Vec::new();
        while let Some(rec) = reader.read_next().expect("stream parse") {
            timestamps.push(rec.header.timestamp_us);
        }
        assert_eq!(timestamps, vec![10, 20, 30]);
    }

    /// A stream that ends mid-record reports Truncated, not clean EOF.
    #[test]
    fn test_stream_reader_mid_record_eof() {
        let bytes = make_record_bytes(1, 1, 242, 0, 4, 0);
        let cut = &bytes[..bytes.len() - 10];
        let mut reader = FeitCsiStreamReader::new(std::io::Cursor::new(cut.to_vec()));
        assert!(matches!(
            reader.read_next(),
            Err(FeitCsiError::Truncated { .. })
        ));
    }

    /// File replay is deterministic: two independent reads of the same
    /// synthetic capture yield byte-identical record sequences and identical
    /// converted readings (timestamps derive from the record, not wall-clock).
    #[test]
    fn test_replay_determinism() {
        let mut capture = Vec::new();
        for ts in [1_000u64, 2_000, 3_000] {
            capture.extend_from_slice(&make_record_bytes(2, 1, 996, 2, 4, ts));
        }
        let path = std::env::temp_dir().join(format!(
            "feitcsi_replay_test_{}.dat",
            std::process::id()
        ));
        std::fs::write(&path, &capture).unwrap();

        let read_all = || -> Vec<FeitCsiRecord> {
            let mut reader = FeitCsiFileReader::open(path.to_str().unwrap()).unwrap();
            let mut out = Vec::new();
            while let Some(rec) = reader.read_next().unwrap() {
                out.push(rec);
            }
            out
        };

        let first = read_all();
        let second = read_all();
        assert_eq!(first.len(), 3);
        assert_eq!(first, second, "replay must be deterministic");

        // Converted readings are also identical, including timestamps.
        let r1 = first[0].to_readings(WifiBand::Band6GHz, 37);
        let r2 = second[0].to_readings(WifiBand::Band6GHz, 37);
        assert_eq!(r1.timestamp, r2.timestamp);
        assert_eq!(r1.readings[0].amplitudes, r2.readings[0].amplitudes);

        let _ = std::fs::remove_file(&path);
    }

    /// Native → pipeline conversion goes through the explicit interpolation
    /// path and records the mapping in frame metadata.
    #[test]
    fn test_native_to_pipeline_mapping_recorded() {
        let bytes = make_record_bytes(1, 1, 1992, 3, 4, 500);
        let (rec, _) = parse_record(&bytes).unwrap();
        let native = rec.to_readings(WifiBand::Band6GHz, 37);

        // Native metadata is first-class.
        assert_eq!(native.metadata.num_subcarriers, 1992);
        let wb = native.metadata.wideband.as_ref().expect("wideband meta");
        assert_eq!(wb.band, WifiBand::Band6GHz);
        assert_eq!(wb.bandwidth_mhz, 160);
        assert_eq!(wb.native_subcarriers, 1992);
        assert!(wb.mapping.is_none(), "no mapping before conversion");

        // Explicit conversion to pipeline width.
        let converted = resample_readings_to_pipeline(&native, 56).unwrap();
        assert_eq!(converted.metadata.num_subcarriers, 56);
        assert_eq!(converted.readings[0].amplitudes.len(), 56);
        assert_eq!(converted.readings[0].phases.len(), 56);
        let wb = converted.metadata.wideband.as_ref().unwrap();
        assert_eq!(wb.native_subcarriers, 1992, "true resolution preserved");
        let mapping = wb.mapping.as_ref().expect("mapping recorded");
        assert_eq!(mapping.native, 1992);
        assert_eq!(mapping.pipeline, 56);
        assert_eq!(mapping.method, "catmull-rom-cubic");

        // Original frame is untouched.
        assert_eq!(native.metadata.num_subcarriers, 1992);
    }

    /// to_readings emits one reading per (rx, tx) pair at native width.
    #[test]
    fn test_to_readings_antenna_pairs() {
        let bytes = make_record_bytes(2, 2, 242, 0, 4, 0);
        let (rec, _) = parse_record(&bytes).unwrap();
        let readings = rec.to_readings(WifiBand::Band5GHz, 36);
        assert_eq!(readings.readings.len(), 4);
        for r in &readings.readings {
            assert_eq!(r.amplitudes.len(), 242);
            assert_eq!(r.phases.len(), 242);
        }
        assert_eq!(
            readings.readings[0].tx_mac.as_deref(),
            Some("AA:BB:CC:DD:EE:FF")
        );
        assert!(matches!(readings.metadata.device_type, DeviceType::FeitCsi));
        assert_eq!(readings.metadata.bandwidth, Bandwidth::HT20);
    }
}
