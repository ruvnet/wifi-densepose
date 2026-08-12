//! Widar3.0 ingest — Intel 5300 `.dat` "bfee" CSI log parser and dataset
//! adapter (ADR-291 §1).
//!
//! The Widar3.0 raw distribution ships CSI captured with the Intel 5300 NIC
//! and the Linux 802.11n CSI Tool, stored as framed binary `.dat` logs. This
//! module provides:
//!
//! - [`parse_bfee_bytes`] — a bounded, panic-free parser for the framed
//!   "bfee" record stream. Invalid records are **skipped with a warning**,
//!   never a panic: `.dat` files are untrusted input and are validated at the
//!   boundary (CLAUDE.md).
//! - [`WidarDataset`] — a [`CsiDataset`] implementation that maps each `.dat`
//!   recording into windowed [`CsiSample`]s (with subcarrier interpolation to
//!   the training pipeline's target count) and exposes per-window
//!   [`SampleMeta`] for the ADR-291 split protocols.
//! - [`encode_bfee_frame`] — a deterministic synthetic-fixture encoder used by
//!   unit tests and benches, so no binary dataset files are ever checked in.
//!
//! # Binary record layout (ADR-291)
//!
//! ```text
//! frame   : u16 LE field_len | u8 code            (code 0xBB = bfee record)
//!           field_len counts the code byte plus the payload, so the next
//!           frame starts field_len + 2 bytes later.
//! payload : 20-byte bfee header
//!             [0..4)   timestamp_low  u32 LE
//!             [4..6)   bfee_count     u16 LE
//!             [6..8)   reserved       (2 bytes, ignored)
//!             [8]      n_rx           u8   (1..=3)
//!             [9]      n_tx           u8   (1..=3)
//!             [10..13) rssi_a/b/c     u8 each
//!             [13]     noise          i8
//!             [14]     agc            u8
//!             [15]     antenna_sel    u8
//!             [16..18) len            u16 LE (packed CSI byte count)
//!             [18..20) rate           u16 LE
//!           then `len` bytes of packed CSI.
//! csi     : 10-bit two's-complement components, packed LSB-first with no
//!           inter-field padding, in order
//!             for sc in 0..30 { for rx in 0..n_rx { for tx in 0..n_tx {
//!                 real; imag; } } }
//!           `len` must equal ceil(30 * n_rx * n_tx * 2 * 10 / 8).
//! ```
//!
//! This is the layout specified by ADR-291. Note the original Linux CSI Tool
//! writes 8-bit components with per-group shift bits and a big-endian frame
//! length; if raw upstream logs are ingested unconverted, records fail the
//! `len` consistency check and are skipped with a warning rather than being
//! silently misdecoded.
//!
//! # Widar3.0 naming convention (assumed, tolerant)
//!
//! The Widar3.0 site was not reachable from this build environment, so the
//! convention below is **assumed** from the Widar3.0 paper/release notes and
//! the parser is deliberately tolerant (missing fields parse as `0`):
//!
//! ```text
//! <root>/[room1/]<date>/user1/user1-3-1-1-2-r5.dat
//!                              │     │ │ │ │  └ receiver id     (optional)
//!                              │     │ │ │ └ repetition number
//!                              │     │ │ └ face orientation (1..=5)
//!                              │     │ └ torso location    (1..=5)
//!                              │     └ gesture type
//!                              └ user id
//! ```
//!
//! The environment/room id is taken from the nearest ancestor directory named
//! `room<N>` (case-insensitive); when absent (the raw release groups by
//! capture date instead) it defaults to `0` and cross-environment splits over
//! such a tree will fail the leakage audit rather than silently pass.

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::dataset::{CsiDataset, CsiSample};
use crate::error::DatasetError;
use crate::protocols::SampleMeta;
use crate::subcarrier::interpolate_subcarriers;

/// Complex CSI component type used by the parser.
pub type Complex32 = Complex<f32>;

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Record code identifying a beamforming-feedback ("bfee") CSI record.
pub const BFEE_CODE: u8 = 0xBB;

/// Bytes in the per-frame header (`u16` length + `u8` code).
const FRAME_HEADER_LEN: usize = 3;

/// Bytes in the fixed bfee header that precedes the packed CSI payload.
const BFEE_HEADER_LEN: usize = 20;

/// Number of subcarrier groups reported by the Intel 5300 (30 groups over a
/// 20/40 MHz channel).
pub const WIDAR_SUBCARRIERS: usize = 30;

/// Bits per packed CSI component (10-bit two's complement).
const CSI_COMPONENT_BITS: usize = 10;

/// Maximum antenna count on either side (Intel 5300 has 3 antennas).
const MAX_ANTENNAS: usize = 3;

/// Upper bound on a single frame's `field_len`, derived from the largest
/// possible record (3×3 CSI ≈ 695 bytes) with generous slack. A larger value
/// means framing is lost; the parser stops instead of allocating unboundedly.
const MAX_FIELD_LEN: usize = 4096;

/// Upper bound on a `.dat` file accepted by [`WidarDataset::discover`].
/// Bounded allocation at the file boundary; larger files are skipped with a
/// warning.
const MAX_DAT_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Number of COCO keypoints emitted in [`CsiSample`]s. Widar is a gesture
/// dataset with no pose ground truth, so keypoints are zero with visibility
/// `0` (COCO "not labelled").
const NUM_KEYPOINTS: usize = 17;

/// Packed CSI byte length for a record with the given antenna counts:
/// `ceil(30 × n_rx × n_tx × 2 × 10 / 8)`.
#[must_use]
pub fn packed_csi_len(n_rx: usize, n_tx: usize) -> usize {
    (WIDAR_SUBCARRIERS * n_rx * n_tx * 2 * CSI_COMPONENT_BITS).div_ceil(8)
}

// ---------------------------------------------------------------------------
// BfeeRecord + parser
// ---------------------------------------------------------------------------

/// One decoded bfee CSI record.
#[derive(Debug, Clone)]
pub struct BfeeRecord {
    /// Low 32 bits of the NIC's 1 MHz clock at capture time.
    pub timestamp_low: u32,
    /// Running count of bfee measurements delivered by the NIC.
    pub bfee_count: u16,
    /// Number of receive antennas (1..=3).
    pub n_rx: u8,
    /// Number of transmit antennas (1..=3).
    pub n_tx: u8,
    /// RSSI at antenna A (dB above an internal reference).
    pub rssi_a: u8,
    /// RSSI at antenna B.
    pub rssi_b: u8,
    /// RSSI at antenna C.
    pub rssi_c: u8,
    /// Noise floor estimate in dBm.
    pub noise: i8,
    /// Automatic gain control setting.
    pub agc: u8,
    /// Antenna selection / permutation bits.
    pub antenna_sel: u8,
    /// Rate/flags field as logged by the driver.
    pub rate: u16,
    /// Complex CSI, shape `[n_tx, n_rx, 30]`.
    pub csi: Array3<Complex32>,
}

/// Outcome of parsing a byte buffer of framed bfee records.
#[derive(Debug, Clone)]
pub struct BfeeParse {
    /// Successfully decoded records, in file order.
    pub records: Vec<BfeeRecord>,
    /// Number of records skipped because they were truncated or corrupt.
    pub skipped: usize,
    /// Number of well-framed records with a non-bfee code (ignored, not an
    /// error — real logs interleave other record types).
    pub non_bfee: usize,
}

/// Parse a buffer of framed Intel 5300 bfee records (ADR-291 layout — see the
/// module docs for the exact binary format).
///
/// The parser never panics on malformed input: invalid or truncated records
/// are skipped with a `warn!` and counted in [`BfeeParse::skipped`]. When
/// framing is irrecoverably lost (a `field_len` beyond [`MAX_FIELD_LEN`] or a
/// record extending past the end of the buffer) parsing stops at that point.
#[must_use]
pub fn parse_bfee_bytes(bytes: &[u8]) -> BfeeParse {
    // Conservative lower-bound estimate (largest possible frame) so a clean
    // log skips the early Vec doublings without ever over-reserving.
    let max_frame = 2 + 1 + BFEE_HEADER_LEN + packed_csi_len(MAX_ANTENNAS, MAX_ANTENNAS);
    let mut records = Vec::with_capacity(bytes.len() / max_frame);
    let mut skipped = 0usize;
    let mut non_bfee = 0usize;
    let mut cursor = 0usize;

    while cursor + FRAME_HEADER_LEN <= bytes.len() {
        let field_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        if field_len == 0 {
            warn!("bfee frame at byte {cursor}: zero field_len, skipping frame header");
            skipped += 1;
            cursor += FRAME_HEADER_LEN;
            continue;
        }
        if field_len > MAX_FIELD_LEN {
            warn!(
                "bfee frame at byte {cursor}: field_len {field_len} exceeds bound \
                 {MAX_FIELD_LEN}; framing lost, abandoning remainder of buffer"
            );
            skipped += 1;
            break;
        }
        let frame_end = cursor + 2 + field_len;
        if frame_end > bytes.len() {
            warn!(
                "bfee frame at byte {cursor}: truncated (needs {} bytes, {} remain)",
                field_len + 2,
                bytes.len() - cursor
            );
            skipped += 1;
            break;
        }

        let code = bytes[cursor + 2];
        let payload = &bytes[cursor + FRAME_HEADER_LEN..frame_end];
        cursor = frame_end;

        if code != BFEE_CODE {
            debug!("skipping non-bfee record code {code:#04x}");
            non_bfee += 1;
            continue;
        }

        match parse_bfee_payload(payload) {
            Ok(record) => records.push(record),
            Err(reason) => {
                warn!("skipping corrupt bfee record: {reason}");
                skipped += 1;
            }
        }
    }

    let tail = bytes.len().saturating_sub(cursor);
    if tail > 0 && tail < FRAME_HEADER_LEN {
        // A dangling partial frame header at EOF is a truncation, not silence.
        warn!("bfee buffer ends with {tail} dangling byte(s) (truncated frame header)");
        skipped += 1;
    }

    BfeeParse {
        records,
        skipped,
        non_bfee,
    }
}

/// Decode the 20-byte bfee header + packed CSI payload of a single record.
fn parse_bfee_payload(payload: &[u8]) -> Result<BfeeRecord, String> {
    if payload.len() < BFEE_HEADER_LEN {
        return Err(format!(
            "payload too short: {} < {BFEE_HEADER_LEN} header bytes",
            payload.len()
        ));
    }

    // Header slices are in-bounds by the length check above.
    let timestamp_low = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let bfee_count = u16::from_le_bytes([payload[4], payload[5]]);
    // payload[6..8] reserved.
    let n_rx = payload[8];
    let n_tx = payload[9];
    let rssi_a = payload[10];
    let rssi_b = payload[11];
    let rssi_c = payload[12];
    let noise = payload[13] as i8;
    let agc = payload[14];
    let antenna_sel = payload[15];
    let csi_len = u16::from_le_bytes([payload[16], payload[17]]) as usize;
    let rate = u16::from_le_bytes([payload[18], payload[19]]);

    if !(1..=MAX_ANTENNAS).contains(&(n_rx as usize)) {
        return Err(format!("n_rx {n_rx} out of range 1..=3"));
    }
    if !(1..=MAX_ANTENNAS).contains(&(n_tx as usize)) {
        return Err(format!("n_tx {n_tx} out of range 1..=3"));
    }
    let expected = packed_csi_len(n_rx as usize, n_tx as usize);
    if csi_len != expected {
        return Err(format!(
            "csi len field {csi_len} does not match {expected} expected for \
             n_rx={n_rx}, n_tx={n_tx}"
        ));
    }
    let body = &payload[BFEE_HEADER_LEN..];
    if body.len() < csi_len {
        return Err(format!(
            "packed CSI truncated: {} bytes present, {csi_len} declared",
            body.len()
        ));
    }
    let body = &body[..csi_len];

    // Unpack: for sc { for rx { for tx { real; imag } } }, 10 bits each,
    // LSB-first. A streaming bit accumulator reads each payload byte exactly
    // once (instead of re-assembling a 3-byte window per component), and the
    // components are written through the contiguous backing slice — the
    // `[n_tx, n_rx, 30]` array is standard C order, so the destination index
    // is `(tx * n_rx + rx) * 30 + sc`.
    let (n_rx_u, n_tx_u) = (n_rx as usize, n_tx as usize);
    let mut csi = Array3::<Complex32>::zeros((n_tx_u, n_rx_u, WIDAR_SUBCARRIERS));
    let flat = csi
        .as_slice_mut()
        .expect("freshly allocated Array3 is contiguous");
    let mut bits = BitReader::new(body);
    for sc in 0..WIDAR_SUBCARRIERS {
        for rx in 0..n_rx_u {
            for tx in 0..n_tx_u {
                let re = bits.next_i10();
                let im = bits.next_i10();
                flat[(tx * n_rx_u + rx) * WIDAR_SUBCARRIERS + sc] =
                    Complex32::new(re as f32, im as f32);
            }
        }
    }

    Ok(BfeeRecord {
        timestamp_low,
        bfee_count,
        n_rx,
        n_tx,
        rssi_a,
        rssi_b,
        rssi_c,
        noise,
        agc,
        antenna_sel,
        rate,
        csi,
    })
}

/// Streaming LSB-first bit reader over a packed CSI payload.
///
/// Each payload byte is loaded into the accumulator exactly once; reads past
/// the slice end yield zero bits — callers bound the total bit count via the
/// `csi_len` consistency check, so that is belt-and-braces, not a format
/// feature. The accumulator never holds more than 17 bits, so `u32` cannot
/// overflow.
struct BitReader<'a> {
    body: &'a [u8],
    pos: usize,
    acc: u32,
    acc_bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(body: &'a [u8]) -> Self {
        BitReader {
            body,
            pos: 0,
            acc: 0,
            acc_bits: 0,
        }
    }

    /// Next 10-bit two's-complement integer (branchless sign extension).
    #[inline]
    fn next_i10(&mut self) -> i16 {
        while self.acc_bits < CSI_COMPONENT_BITS as u32 {
            let byte = self.body.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.acc |= (byte as u32) << self.acc_bits;
            self.acc_bits += 8;
        }
        let v = self.acc & 0x3FF;
        self.acc >>= CSI_COMPONENT_BITS;
        self.acc_bits -= CSI_COMPONENT_BITS as u32;
        // Shift the 10-bit value to the top of an i32 and arithmetic-shift
        // back down: sign extension without a branch.
        (((v << 22) as i32) >> 22) as i16
    }
}

/// Write a 10-bit two's-complement integer at `bit_off` into a zeroed buffer.
fn write_i10(buf: &mut [u8], bit_off: usize, value: i16) {
    let v = (value as i32 & 0x3FF) as u32;
    let byte = bit_off >> 3;
    let shift = bit_off & 7;
    let merged = v << shift;
    buf[byte] |= (merged & 0xFF) as u8;
    if byte + 1 < buf.len() {
        buf[byte + 1] |= ((merged >> 8) & 0xFF) as u8;
    }
    if byte + 2 < buf.len() {
        buf[byte + 2] |= ((merged >> 16) & 0xFF) as u8;
    }
}

/// Encode one framed bfee record from synthetic CSI values — the fixture
/// generator used by unit tests and benches (ADR-291: fixtures are generated
/// in code, never checked in as binary files).
///
/// `csi` is `(real, imag)` pairs in the packing order
/// `for sc { for rx { for tx { .. } } }` and must contain exactly
/// `30 × n_rx × n_tx` entries with each component in `-512..=511`.
///
/// # Panics
///
/// Panics on programmer error: antenna counts outside `1..=3`, a wrong `csi`
/// length, or out-of-range components. This is a fixture builder for trusted
/// test inputs, not a boundary parser.
#[must_use]
pub fn encode_bfee_frame(
    timestamp_low: u32,
    bfee_count: u16,
    n_rx: u8,
    n_tx: u8,
    csi: &[(i16, i16)],
) -> Vec<u8> {
    assert!(
        (1..=MAX_ANTENNAS).contains(&(n_rx as usize)),
        "n_rx must be 1..=3"
    );
    assert!(
        (1..=MAX_ANTENNAS).contains(&(n_tx as usize)),
        "n_tx must be 1..=3"
    );
    let expected_pairs = WIDAR_SUBCARRIERS * n_rx as usize * n_tx as usize;
    assert_eq!(
        csi.len(),
        expected_pairs,
        "csi must contain 30 × n_rx × n_tx complex pairs"
    );
    for &(re, im) in csi {
        assert!(
            (-512..=511).contains(&re) && (-512..=511).contains(&im),
            "10-bit components must be in -512..=511"
        );
    }

    let csi_len = packed_csi_len(n_rx as usize, n_tx as usize);
    let mut packed = vec![0u8; csi_len];
    let mut bit_off = 0usize;
    for &(re, im) in csi {
        write_i10(&mut packed, bit_off, re);
        bit_off += CSI_COMPONENT_BITS;
        write_i10(&mut packed, bit_off, im);
        bit_off += CSI_COMPONENT_BITS;
    }

    let field_len = 1 + BFEE_HEADER_LEN + csi_len; // code + header + payload
    let mut frame = Vec::with_capacity(2 + field_len);
    frame.extend_from_slice(&(field_len as u16).to_le_bytes());
    frame.push(BFEE_CODE);
    frame.extend_from_slice(&timestamp_low.to_le_bytes());
    frame.extend_from_slice(&bfee_count.to_le_bytes());
    frame.extend_from_slice(&[0, 0]); // reserved
    frame.push(n_rx);
    frame.push(n_tx);
    frame.extend_from_slice(&[33, 34, 35]); // rssi a/b/c
    frame.push((-92i8) as u8); // noise
    frame.push(30); // agc
    frame.push(0b0000_0110); // antenna_sel
    frame.extend_from_slice(&(csi_len as u16).to_le_bytes());
    frame.extend_from_slice(&0x4404u16.to_le_bytes()); // rate
    frame.extend_from_slice(&packed);
    frame
}

// ---------------------------------------------------------------------------
// Widar naming convention
// ---------------------------------------------------------------------------

/// Domain metadata parsed from a Widar3.0 `.dat` path (see the module docs
/// for the assumed naming convention). Fields the path does not encode are
/// `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WidarFileMeta {
    /// User (subject) id, e.g. `1` for `user1-…`.
    pub user: u32,
    /// Gesture type id (second dash field).
    pub gesture: u32,
    /// Torso location id (third dash field).
    pub location: u32,
    /// Face orientation id (fourth dash field).
    pub orientation: u32,
    /// Repetition number (fifth dash field).
    pub repetition: u32,
    /// Receiver id from a trailing `-r<N>` field; `0` when absent.
    pub receiver: u32,
    /// Room/environment id from a `room<N>` ancestor directory; `0` when the
    /// tree does not encode one.
    pub room: u32,
}

/// Parse Widar3.0 domain metadata from a `.dat` path. Tolerant: returns
/// `None` only when the file stem yields no user id at all; any other missing
/// field parses as `0`.
#[must_use]
pub fn parse_widar_path(path: &Path) -> Option<WidarFileMeta> {
    let stem = path.file_stem()?.to_str()?;
    let mut fields = stem.split('-');

    // First field: "user1" / "id1" / bare digits — take the numeric suffix.
    let user = trailing_number(fields.next()?)?;

    let mut meta = WidarFileMeta {
        user,
        ..WidarFileMeta::default()
    };

    let positional: [&mut u32; 4] = [
        &mut meta.gesture,
        &mut meta.location,
        &mut meta.orientation,
        &mut meta.repetition,
    ];
    let mut pos = 0usize;
    for field in fields {
        let lower_r = field.len() >= 2
            && (field.starts_with('r') || field.starts_with('R'))
            && field[1..].chars().all(|c| c.is_ascii_digit());
        if lower_r {
            meta.receiver = field[1..].parse().unwrap_or(0);
            continue;
        }
        if pos < positional.len() {
            *positional[pos] = field.parse().unwrap_or(0);
            pos += 1;
        }
    }

    // Room from the nearest `room<N>` ancestor directory (case-insensitive).
    for ancestor in path.ancestors().skip(1) {
        if let Some(name) = ancestor.file_name().and_then(|n| n.to_str()) {
            let lower = name.to_ascii_lowercase();
            if let Some(digits) = lower.strip_prefix("room") {
                if let Ok(room) = digits.parse::<u32>() {
                    meta.room = room;
                    break;
                }
            }
        }
    }

    Some(meta)
}

/// Numeric suffix of a token like `user1` → `1` (also accepts bare digits).
fn trailing_number(token: &str) -> Option<u32> {
    let digits: String = token.chars().skip_while(|c| !c.is_ascii_digit()).collect();
    digits.parse().ok()
}

// ---------------------------------------------------------------------------
// WidarDataset
// ---------------------------------------------------------------------------

/// An indexed `.dat` recording in the Widar scan.
#[derive(Debug, Clone)]
struct WidarEntry {
    path: PathBuf,
    meta: WidarFileMeta,
    /// Antenna dims established by the first valid record of the file.
    n_tx: usize,
    n_rx: usize,
    /// Number of valid records with matching antenna dims.
    num_frames: usize,
    window_frames: usize,
}

impl WidarEntry {
    /// Number of stride-1 windows this recording contributes.
    fn num_windows(&self) -> usize {
        if self.num_frames < self.window_frames {
            0
        } else {
            self.num_frames - self.window_frames + 1
        }
    }
}

/// Dataset adapter for Widar3.0 `.dat` recordings (ADR-291 §1).
///
/// Scanning parses every file once at construction to count valid records;
/// [`CsiDataset::get`] re-reads the file lazily and cuts the requested
/// stride-1 window. Each `.dat` file is treated as **one continuous
/// recording** for the leakage audit ([`crate::protocols::leakage`]): its
/// [`SampleMeta::recording_id`] is the file's index in the sorted scan.
///
/// Widar has no pose ground truth, so [`CsiSample::keypoints`] are zeros with
/// visibility `0` ("not labelled"); `subject_id` carries the user id and
/// `action_id` the gesture id.
pub struct WidarDataset {
    entries: Vec<WidarEntry>,
    /// Prefix-sum of window counts (length = entries.len() + 1).
    cumulative: Vec<usize>,
    window_frames: usize,
    target_subcarriers: usize,
    /// Root directory stored for display / debug purposes.
    #[allow(dead_code)]
    root: PathBuf,
}

impl WidarDataset {
    /// Scan `root` recursively for `.dat` recordings and build a window index.
    ///
    /// Unreadable, oversized, or record-free files are skipped with a
    /// warning; a root with no usable recordings is an error.
    ///
    /// # Errors
    ///
    /// [`DatasetError::DataNotFound`] when `root` does not exist or yields no
    /// usable recording; I/O errors for filesystem access failures.
    pub fn discover(
        root: &Path,
        window_frames: usize,
        target_subcarriers: usize,
    ) -> Result<Self, DatasetError> {
        if window_frames == 0 {
            return Err(DatasetError::invalid_format(
                root,
                "window_frames must be > 0",
            ));
        }
        if !root.exists() {
            return Err(DatasetError::not_found(
                root,
                "Widar root directory not found",
            ));
        }

        let mut dat_paths: Vec<PathBuf> = walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("dat"))
                    .unwrap_or(false)
            })
            .collect();
        dat_paths.sort();

        let mut entries = Vec::new();
        for path in dat_paths {
            match Self::scan_file(&path, window_frames) {
                Ok(Some(entry)) => entries.push(entry),
                Ok(None) => {}
                Err(e) => warn!("Skipping {}: {e}", path.display()),
            }
        }

        if entries.is_empty() {
            return Err(DatasetError::not_found(
                root,
                "no usable Widar .dat recordings found under root",
            ));
        }

        let mut cumulative = vec![0usize; entries.len() + 1];
        for (i, e) in entries.iter().enumerate() {
            cumulative[i + 1] = cumulative[i] + e.num_windows();
        }

        info!(
            "WidarDataset: scanned {} recordings, {} total windows (root={})",
            entries.len(),
            cumulative.last().copied().unwrap_or(0),
            root.display()
        );

        Ok(WidarDataset {
            entries,
            cumulative,
            window_frames,
            target_subcarriers,
            root: root.to_path_buf(),
        })
    }

    /// Scan one `.dat` file: size bound, record count, antenna dims,
    /// path metadata. `Ok(None)` means "valid scan, nothing usable".
    fn scan_file(path: &Path, window_frames: usize) -> Result<Option<WidarEntry>, DatasetError> {
        let file_len = std::fs::metadata(path)
            .map_err(|e| DatasetError::io_error(path, e))?
            .len();
        if file_len > MAX_DAT_FILE_BYTES {
            warn!(
                "Skipping {}: {file_len} bytes exceeds the {MAX_DAT_FILE_BYTES}-byte bound",
                path.display()
            );
            return Ok(None);
        }

        let meta = match parse_widar_path(path) {
            Some(m) => m,
            None => {
                warn!(
                    "{}: file name does not follow the Widar convention; using zeroed metadata",
                    path.display()
                );
                WidarFileMeta::default()
            }
        };

        let bytes = std::fs::read(path).map_err(|e| DatasetError::io_error(path, e))?;
        let parse = parse_bfee_bytes(&bytes);
        if parse.skipped > 0 {
            warn!(
                "{}: skipped {} invalid record(s) ({} valid)",
                path.display(),
                parse.skipped,
                parse.records.len()
            );
        }
        let Some(first) = parse.records.first() else {
            warn!("Skipping {}: no valid bfee records", path.display());
            return Ok(None);
        };
        let (n_tx, n_rx) = (first.n_tx as usize, first.n_rx as usize);
        let num_frames = parse
            .records
            .iter()
            .filter(|r| r.n_tx as usize == n_tx && r.n_rx as usize == n_rx)
            .count();
        if num_frames < parse.records.len() {
            warn!(
                "{}: dropped {} record(s) with antenna dims differing from the first \
                 ({n_tx}×{n_rx})",
                path.display(),
                parse.records.len() - num_frames
            );
        }
        if num_frames < window_frames {
            debug!(
                "{}: {} frame(s) < window {window_frames}; contributes no windows",
                path.display(),
                num_frames
            );
        }
        Ok(Some(WidarEntry {
            path: path.to_path_buf(),
            meta,
            n_tx,
            n_rx,
            num_frames,
            window_frames,
        }))
    }

    /// Resolve a global window index to `(entry_index, frame_offset)`.
    fn locate(&self, idx: usize) -> Option<(usize, usize)> {
        let total = self.cumulative.last().copied().unwrap_or(0);
        if idx >= total {
            return None;
        }
        let entry_idx = self
            .cumulative
            .partition_point(|&c| c <= idx)
            .saturating_sub(1);
        Some((entry_idx, idx - self.cumulative[entry_idx]))
    }

    /// Split-protocol metadata for the window at `idx` (ADR-291 §2): user →
    /// subject, room → environment, plus orientation/gesture, and the owning
    /// `.dat` file as the continuous `recording_id`.
    ///
    /// # Errors
    ///
    /// [`DatasetError::IndexOutOfBounds`] when `idx >= self.len()`.
    pub fn sample_meta(&self, idx: usize) -> Result<SampleMeta, DatasetError> {
        let (entry_idx, offset) = self.locate(idx).ok_or(DatasetError::IndexOutOfBounds {
            idx,
            len: self.cumulative.last().copied().unwrap_or(0),
        })?;
        let m = &self.entries[entry_idx].meta;
        Ok(SampleMeta {
            subject_id: m.user,
            environment_id: m.room,
            orientation_id: m.orientation,
            gesture_id: m.gesture,
            recording_id: entry_idx as u64,
            window_index: offset as u64,
        })
    }

    /// [`SampleMeta`] for every window, in index order — the input to
    /// [`crate::protocols::SplitPlan::partition`].
    pub fn sample_metas(&self) -> Vec<SampleMeta> {
        (0..self.len())
            .map(|i| {
                self.sample_meta(i)
                    .expect("index < len is always locatable")
            })
            .collect()
    }

    /// Number of `.dat` recordings behind this dataset.
    #[must_use]
    pub fn num_recordings(&self) -> usize {
        self.entries.len()
    }
}

impl CsiDataset for WidarDataset {
    fn len(&self) -> usize {
        self.cumulative.last().copied().unwrap_or(0)
    }

    fn get(&self, idx: usize) -> Result<CsiSample, DatasetError> {
        let total = self.len();
        let (entry_idx, offset) = self
            .locate(idx)
            .ok_or(DatasetError::IndexOutOfBounds { idx, len: total })?;
        let entry = &self.entries[entry_idx];

        let bytes =
            std::fs::read(&entry.path).map_err(|e| DatasetError::io_error(&entry.path, e))?;
        let parse = parse_bfee_bytes(&bytes);
        let records: Vec<&BfeeRecord> = parse
            .records
            .iter()
            .filter(|r| r.n_tx as usize == entry.n_tx && r.n_rx as usize == entry.n_rx)
            .collect();

        let t_end = offset + self.window_frames;
        if t_end > records.len() {
            // The file changed on disk since discovery.
            return Err(DatasetError::invalid_format(
                &entry.path,
                format!(
                    "window [{offset}, {t_end}) exceeds {} valid frame(s); \
                     file changed since scan?",
                    records.len()
                ),
            ));
        }

        let (n_tx, n_rx) = (entry.n_tx, entry.n_rx);
        let mut amplitude =
            Array4::<f32>::zeros((self.window_frames, n_tx, n_rx, WIDAR_SUBCARRIERS));
        let mut phase = Array4::<f32>::zeros((self.window_frames, n_tx, n_rx, WIDAR_SUBCARRIERS));
        for (t, record) in records[offset..t_end].iter().enumerate() {
            for tx in 0..n_tx {
                for rx in 0..n_rx {
                    for sc in 0..WIDAR_SUBCARRIERS {
                        let c = record.csi[[tx, rx, sc]];
                        amplitude[[t, tx, rx, sc]] = c.norm();
                        phase[[t, tx, rx, sc]] = c.arg();
                    }
                }
            }
        }

        let amplitude = if WIDAR_SUBCARRIERS != self.target_subcarriers {
            interpolate_subcarriers(&amplitude, self.target_subcarriers)
        } else {
            amplitude
        };
        let phase = if WIDAR_SUBCARRIERS != self.target_subcarriers {
            interpolate_subcarriers(&phase, self.target_subcarriers)
        } else {
            phase
        };

        Ok(CsiSample {
            amplitude,
            phase,
            keypoints: Array2::zeros((NUM_KEYPOINTS, 2)),
            keypoint_visibility: Array1::zeros(NUM_KEYPOINTS),
            subject_id: entry.meta.user,
            action_id: entry.meta.gesture,
            frame_id: offset as u64,
        })
    }

    fn name(&self) -> &str {
        "WidarDataset"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Deterministic synthetic CSI pattern for record `t`: values derived
    /// from the pair index, folded into the 10-bit range.
    fn synthetic_csi(t: usize, n_rx: usize, n_tx: usize) -> Vec<(i16, i16)> {
        (0..WIDAR_SUBCARRIERS * n_rx * n_tx)
            .map(|i| {
                let re = ((t * 37 + i * 13) % 1024) as i16 - 512;
                let im = ((t * 17 + i * 7) % 1024) as i16 - 512;
                (re, im)
            })
            .collect()
    }

    fn synthetic_file(num_records: usize, n_rx: u8, n_tx: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        for t in 0..num_records {
            let csi = synthetic_csi(t, n_rx as usize, n_tx as usize);
            bytes.extend_from_slice(&encode_bfee_frame(
                1000 + t as u32,
                t as u16,
                n_rx,
                n_tx,
                &csi,
            ));
        }
        bytes
    }

    // ----- parser: valid fixtures ------------------------------------------

    #[test]
    fn parse_roundtrips_valid_records() {
        let bytes = synthetic_file(5, 3, 2);
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 5);
        assert_eq!(parse.skipped, 0);
        assert_eq!(parse.non_bfee, 0);

        let r = &parse.records[2];
        assert_eq!(r.timestamp_low, 1002);
        assert_eq!(r.bfee_count, 2);
        assert_eq!(r.n_rx, 3);
        assert_eq!(r.n_tx, 2);
        assert_eq!(r.noise, -92);
        assert_eq!(r.rate, 0x4404);
        assert_eq!(r.csi.shape(), &[2, 3, WIDAR_SUBCARRIERS]);

        // Bit-exact roundtrip of every component, including negatives.
        let csi = synthetic_csi(2, 3, 2);
        let mut i = 0usize;
        for sc in 0..WIDAR_SUBCARRIERS {
            for rx in 0..3 {
                for tx in 0..2 {
                    let (re, im) = csi[i];
                    assert_abs_diff_eq!(r.csi[[tx, rx, sc]].re, re as f32, epsilon = 0.0);
                    assert_abs_diff_eq!(r.csi[[tx, rx, sc]].im, im as f32, epsilon = 0.0);
                    i += 1;
                }
            }
        }
    }

    #[test]
    fn parse_sign_extends_extremes() {
        let n = WIDAR_SUBCARRIERS;
        let mut csi = vec![(0i16, 0i16); n];
        csi[0] = (-512, 511);
        csi[n - 1] = (-1, 1);
        let bytes = encode_bfee_frame(7, 1, 1, 1, &csi);
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        let r = &parse.records[0];
        assert_eq!(r.csi[[0, 0, 0]], Complex32::new(-512.0, 511.0));
        assert_eq!(r.csi[[0, 0, n - 1]], Complex32::new(-1.0, 1.0));
    }

    #[test]
    fn parse_is_deterministic() {
        let bytes = synthetic_file(3, 2, 2);
        let a = parse_bfee_bytes(&bytes);
        let b = parse_bfee_bytes(&bytes);
        assert_eq!(a.records.len(), b.records.len());
        for (ra, rb) in a.records.iter().zip(&b.records) {
            assert_eq!(ra.csi, rb.csi);
        }
    }

    // ----- parser: truncated / corrupt fixtures ----------------------------

    #[test]
    fn parse_empty_buffer_is_empty() {
        let parse = parse_bfee_bytes(&[]);
        assert!(parse.records.is_empty());
        assert_eq!(parse.skipped, 0);
    }

    #[test]
    fn parse_truncated_record_is_skipped_not_panic() {
        let mut bytes = synthetic_file(2, 2, 1);
        // Chop the last record mid-payload.
        let cut = bytes.len() - 10;
        bytes.truncate(cut);
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn parse_dangling_header_bytes_counted() {
        let mut bytes = synthetic_file(1, 1, 1);
        bytes.extend_from_slice(&[0x07, 0x00]); // 2 dangling bytes < frame header
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn parse_zero_field_len_resyncs() {
        let mut bytes = vec![0u8, 0u8, 0xBB]; // zero-length frame
        bytes.extend_from_slice(&synthetic_file(1, 1, 1));
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn parse_oversized_field_len_stops_bounded() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u16::MAX.to_le_bytes());
        bytes.push(BFEE_CODE);
        bytes.extend_from_slice(&vec![0u8; 64]);
        let parse = parse_bfee_bytes(&bytes);
        assert!(parse.records.is_empty());
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn parse_non_bfee_code_is_ignored() {
        let mut bytes = Vec::new();
        // A well-framed record with a different code.
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.push(0xC1);
        bytes.extend_from_slice(&[1, 2, 3]);
        bytes.extend_from_slice(&synthetic_file(1, 1, 1));
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        assert_eq!(parse.non_bfee, 1);
        assert_eq!(parse.skipped, 0);
    }

    #[test]
    fn parse_corrupt_antenna_count_is_skipped() {
        let mut bytes = synthetic_file(2, 2, 2);
        // First frame: corrupt n_rx (payload byte 8 → frame offset 3 + 8).
        bytes[3 + 8] = 9;
        let parse = parse_bfee_bytes(&bytes);
        assert_eq!(parse.records.len(), 1);
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn parse_len_field_mismatch_is_skipped() {
        let mut bytes = synthetic_file(1, 1, 1);
        // Corrupt the csi len field (payload bytes 16..18 → frame offset 19).
        bytes[3 + 16] = 0xFF;
        let parse = parse_bfee_bytes(&bytes);
        assert!(parse.records.is_empty());
        assert_eq!(parse.skipped, 1);
    }

    #[test]
    fn packed_len_matches_formula() {
        // 30 × n_rx × n_tx × 2 comps × 10 bits, ceil to bytes.
        assert_eq!(packed_csi_len(1, 1), 75);
        assert_eq!(packed_csi_len(3, 1), 225);
        assert_eq!(packed_csi_len(3, 3), 675);
    }

    // ----- naming convention -----------------------------------------------

    #[test]
    fn parses_full_widar_name() {
        let m = parse_widar_path(Path::new("/data/room2/20181130/user1/user1-3-1-4-2-r5.dat"))
            .unwrap();
        assert_eq!(
            m,
            WidarFileMeta {
                user: 1,
                gesture: 3,
                location: 1,
                orientation: 4,
                repetition: 2,
                receiver: 5,
                room: 2,
            }
        );
    }

    #[test]
    fn parses_name_without_receiver_or_room() {
        let m = parse_widar_path(Path::new("user12/user12-6-2-3-1.dat")).unwrap();
        assert_eq!(m.user, 12);
        assert_eq!(m.gesture, 6);
        assert_eq!(m.orientation, 3);
        assert_eq!(m.receiver, 0);
        assert_eq!(m.room, 0);
    }

    #[test]
    fn tolerates_short_names() {
        let m = parse_widar_path(Path::new("user3-2.dat")).unwrap();
        assert_eq!(m.user, 3);
        assert_eq!(m.gesture, 2);
        assert_eq!(m.orientation, 0);
        assert!(parse_widar_path(Path::new("nodigits.dat")).is_none());
    }

    // ----- WidarDataset end-to-end on synthetic files ----------------------

    fn write_synthetic_tree(root: &Path) {
        // Two users, one recording each, in room1/room2.
        for (user, room) in [(1u32, 1u32), (2, 2)] {
            let dir = root.join(format!("room{room}")).join(format!("user{user}"));
            std::fs::create_dir_all(&dir).unwrap();
            let file = dir.join(format!("user{user}-1-1-{user}-1-r1.dat"));
            std::fs::write(&file, synthetic_file(6, 2, 1)).unwrap();
        }
    }

    #[test]
    fn widar_dataset_discovers_and_windows() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_tree(tmp.path());

        let ds = WidarDataset::discover(tmp.path(), 4, 56).unwrap();
        assert_eq!(ds.num_recordings(), 2);
        // 6 frames, window 4 ⇒ 3 windows per recording.
        assert_eq!(ds.len(), 6);

        let s = ds.get(0).unwrap();
        assert_eq!(s.amplitude.shape(), &[4, 1, 2, 56]);
        assert_eq!(s.phase.shape(), &[4, 1, 2, 56]);
        assert_eq!(s.keypoints.shape(), &[17, 2]);
        assert_eq!(s.subject_id, 1);
        assert_eq!(s.action_id, 1);

        // Second recording's windows carry the second user's metadata.
        let s2 = ds.get(3).unwrap();
        assert_eq!(s2.subject_id, 2);
        assert_eq!(s2.frame_id, 0);

        // Out of bounds is an error, not a panic.
        assert!(matches!(
            ds.get(6),
            Err(DatasetError::IndexOutOfBounds { idx: 6, len: 6 })
        ));
    }

    #[test]
    fn widar_dataset_native_subcarriers_skip_interpolation() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_tree(tmp.path());
        let ds = WidarDataset::discover(tmp.path(), 4, WIDAR_SUBCARRIERS).unwrap();
        let s = ds.get(0).unwrap();
        assert_eq!(s.amplitude.shape(), &[4, 1, 2, WIDAR_SUBCARRIERS]);
        // Amplitude of the first component must equal |re + j·im| of the fixture.
        let csi = synthetic_csi(0, 2, 1);
        let (re, im) = csi[0];
        let expected = ((re as f32).powi(2) + (im as f32).powi(2)).sqrt();
        assert_abs_diff_eq!(s.amplitude[[0, 0, 0, 0]], expected, epsilon = 1e-4);
    }

    #[test]
    fn widar_sample_meta_maps_domains() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_tree(tmp.path());
        let ds = WidarDataset::discover(tmp.path(), 4, 56).unwrap();

        let metas = ds.sample_metas();
        assert_eq!(metas.len(), ds.len());
        // Windows 0..3 belong to recording 0 (user1, room1, orientation 1).
        assert_eq!(metas[0].subject_id, 1);
        assert_eq!(metas[0].environment_id, 1);
        assert_eq!(metas[0].orientation_id, 1);
        assert_eq!(metas[0].recording_id, 0);
        assert_eq!(metas[2].window_index, 2);
        // Windows 3..6 belong to recording 1 (user2, room2, orientation 2).
        assert_eq!(metas[3].subject_id, 2);
        assert_eq!(metas[3].environment_id, 2);
        assert_eq!(metas[3].orientation_id, 2);
        assert_eq!(metas[3].recording_id, 1);

        assert!(ds.sample_meta(999).is_err());
    }

    #[test]
    fn widar_dataset_skips_corrupt_file_keeps_valid() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_tree(tmp.path());
        // A garbage .dat file must not abort discovery.
        std::fs::write(tmp.path().join("user9-1-1-1-1.dat"), [0xFFu8; 64]).unwrap();
        let ds = WidarDataset::discover(tmp.path(), 4, 56).unwrap();
        assert_eq!(ds.num_recordings(), 2);
    }

    #[test]
    fn widar_dataset_missing_root_errors() {
        assert!(matches!(
            WidarDataset::discover(Path::new("/nonexistent/widar"), 4, 56),
            Err(DatasetError::DataNotFound { .. })
        ));
    }
}
