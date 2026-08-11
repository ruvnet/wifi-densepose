//! Ground-truth reference ingest, time alignment, and agreement metrics
//! (ADR-290).
//!
//! Every credible vitals result ships with reference-sensor ground truth
//! (chest strap, pulse oximeter, ECG). This module makes a `MEASURED` vitals
//! claim reachable for RuView by providing:
//!
//! 1. **Reference ingest** ([`ReferenceSeries`]): timestamped reference
//!    samples for one measurand, parsed from an untrusted
//!    `timestamp_ms,value` CSV export with row-numbered structured errors.
//! 2. **Time alignment** ([`align`]): constant-offset estimation by
//!    maximizing normalized cross-correlation over a bounded lag window on a
//!    common nearest-sample grid, plus an optional linear clock-drift fit.
//!    Alignment parameters are returned in [`AlignmentResult`], never
//!    silently applied.
//! 3. **Agreement metrics** ([`AgreementReport`]): paired-sample count,
//!    coverage, MAE, RMSE, bias, Bland-Altman 95% limits of agreement, and
//!    percent-within-tolerance. A mandatory [`SessionScope`] states subject
//!    count, motion, propagation, and distance band — a report without scope
//!    cannot exist.
//! 4. **Evidence tagging** ([`GradedAgreementReport`]):
//!    [`EvidenceGrade::Measured`] is constructible only through
//!    [`GradedAgreementReport::measured`], which requires non-zero paired
//!    samples, minimum coverage, and a non-blank reproducer command —
//!    enforcement lives in the constructor, not in documentation.
//!
//! Agreement against consumer reference devices is engineering evidence,
//! not medical validation, and never a camera-grade or clinical claim.

use crate::store::VitalSignStore;
use crate::types::{VitalReading, VitalStatus};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Bounds for untrusted input
// ---------------------------------------------------------------------------

/// Maximum number of data rows accepted from a reference CSV.
pub const MAX_CSV_ROWS: usize = 1_000_000;

/// Maximum absolute timestamp in milliseconds (`2^52` ms, far beyond any
/// realistic unix-millis session). Keeps all i64 offset/span arithmetic in
/// this module overflow-free and every timestamp exactly representable as
/// `f64`.
pub const MAX_TIMESTAMP_ABS_MS: i64 = 1 << 52;

/// Maximum plausible physiological value in BPM/BrPM accepted at the input
/// boundary.
pub const MAX_VALUE_BPM: f64 = 300.0;

/// Maximum number of resampled grid points for alignment or agreement.
pub const MAX_GRID_POINTS: usize = 10_000_000;

/// Minimum coverage fraction required to grade a report `MEASURED`.
pub const MIN_MEASURED_COVERAGE: f64 = 0.5;

/// Expected CSV header line.
const CSV_HEADER: &str = "timestamp_ms,value";

/// Maximum length of untrusted text echoed back inside an error.
const MAX_ERROR_ECHO: usize = 64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Structured error for ground-truth ingest, alignment, and agreement.
///
/// For CSV input, `row` is the 1-based line number in the file (the header
/// is line 1). For in-memory constructors ([`ReferenceSeries::new`],
/// [`EstimateSeries::new`], [`EstimateSeries::from_readings`]), `row` is the
/// zero-based index of the offending sample/reading.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundTruthError {
    /// Input contained no header line.
    MissingHeader,
    /// Header line did not match `timestamp_ms,value`. Carries a bounded
    /// echo of what was found.
    BadHeader {
        /// The (truncated) header text encountered.
        found: String,
    },
    /// A data row did not have exactly two comma-separated fields.
    WrongFieldCount {
        /// Offending row.
        row: usize,
        /// Number of fields found.
        found: usize,
    },
    /// A timestamp field failed to parse as an integer.
    BadTimestamp {
        /// Offending row.
        row: usize,
    },
    /// A timestamp is outside `±`[`MAX_TIMESTAMP_ABS_MS`].
    TimestampOutOfRange {
        /// Offending row.
        row: usize,
    },
    /// A value field failed to parse as a finite number.
    BadValue {
        /// Offending row.
        row: usize,
    },
    /// A value is outside `[0, `[`MAX_VALUE_BPM`]`]`.
    ValueOutOfRange {
        /// Offending row.
        row: usize,
        /// The out-of-range value.
        value: f64,
    },
    /// Timestamps must be strictly increasing; sorting is never applied
    /// silently.
    NonMonotonicTimestamp {
        /// Offending row.
        row: usize,
    },
    /// No usable samples were present.
    NoSamples,
    /// More data rows than [`MAX_CSV_ROWS`] (bounded allocation).
    TooManyRows {
        /// Row limit that was exceeded.
        max: usize,
    },
    /// Estimate and reference series measure different quantities.
    MeasurandMismatch {
        /// Measurand of the estimate series.
        estimate: Measurand,
        /// Measurand of the reference series.
        reference: Measurand,
    },
    /// An alignment/agreement configuration parameter is invalid.
    InvalidConfig(&'static str),
    /// The resampled grid would exceed [`MAX_GRID_POINTS`].
    GridTooLarge {
        /// Grid points that would be required.
        points: u64,
    },
    /// Not enough overlapping valid samples for a statistic.
    InsufficientOverlap {
        /// Minimum overlapping pairs required.
        required: usize,
        /// Best overlap actually found.
        found: usize,
    },
    /// Overlapping samples exist but at least one side has zero variance,
    /// so normalized cross-correlation is undefined.
    ConstantSignal,
    /// A report failed the `MEASURED` evidence gate; the message states the
    /// failed requirement.
    NotMeasured(&'static str),
}

impl fmt::Display for GroundTruthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHeader => write!(f, "missing CSV header line '{CSV_HEADER}'"),
            Self::BadHeader { found } => {
                write!(f, "bad CSV header: expected '{CSV_HEADER}', found '{found}'")
            }
            Self::WrongFieldCount { row, found } => {
                write!(f, "row {row}: expected 2 comma-separated fields, found {found}")
            }
            Self::BadTimestamp { row } => {
                write!(f, "row {row}: timestamp is not a valid integer")
            }
            Self::TimestampOutOfRange { row } => {
                write!(f, "row {row}: timestamp outside ±{MAX_TIMESTAMP_ABS_MS} ms")
            }
            Self::BadValue { row } => write!(f, "row {row}: value is not a finite number"),
            Self::ValueOutOfRange { row, value } => {
                write!(f, "row {row}: value {value} outside [0, {MAX_VALUE_BPM}]")
            }
            Self::NonMonotonicTimestamp { row } => {
                write!(f, "row {row}: timestamps must be strictly increasing")
            }
            Self::NoSamples => write!(f, "no usable samples"),
            Self::TooManyRows { max } => write!(f, "more than {max} data rows"),
            Self::MeasurandMismatch { estimate, reference } => write!(
                f,
                "measurand mismatch: estimate is {estimate:?}, reference is {reference:?}"
            ),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::GridTooLarge { points } => {
                write!(f, "resampled grid of {points} points exceeds {MAX_GRID_POINTS}")
            }
            Self::InsufficientOverlap { required, found } => write!(
                f,
                "insufficient overlap: required {required} paired samples, found {found}"
            ),
            Self::ConstantSignal => {
                write!(f, "constant signal: normalized cross-correlation undefined")
            }
            Self::NotMeasured(msg) => write!(f, "MEASURED evidence gate failed: {msg}"),
        }
    }
}

impl std::error::Error for GroundTruthError {}

// ---------------------------------------------------------------------------
// Reference series
// ---------------------------------------------------------------------------

/// Quantity a reference or estimate series measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Measurand {
    /// Heart rate, beats per minute.
    HeartRateBpm,
    /// Breathing (respiratory) rate, breaths per minute.
    BreathingRateBrpm,
}

impl Measurand {
    /// Default agreement tolerance for this measurand (ADR-290: ±2 bpm for
    /// heart rate, ±1 brpm for breathing).
    #[must_use]
    pub fn default_tolerance_bpm(self) -> f64 {
        match self {
            Self::HeartRateBpm => 2.0,
            Self::BreathingRateBrpm => 1.0,
        }
    }
}

/// Measurement principle of a reference device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MeasurementPrinciple {
    /// Electrocardiography (e.g. chest strap ECG).
    Ecg,
    /// Photoplethysmography (e.g. pulse oximeter, optical wrist sensor).
    Ppg,
    /// Respiratory effort band / chest expansion.
    RespiratoryBand,
    /// Capnography.
    Capnography,
    /// Manually counted.
    Manual,
    /// Anything else; state it in the device model string.
    Other,
}

/// Metadata identifying the reference device a series came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReferenceDevice {
    /// Device make, e.g. `"Polar"`.
    pub make: String,
    /// Device model, e.g. `"H10"`.
    pub model: String,
    /// Measurement principle.
    pub principle: MeasurementPrinciple,
}

/// One timestamped sample.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReferenceSample {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Value in BPM (heart rate) or BrPM (breathing rate).
    pub value: f64,
}

/// Validate one sample at index/row `row` against the previous timestamp.
fn validate_sample(
    row: usize,
    timestamp_ms: i64,
    value: f64,
    prev_ts: Option<i64>,
) -> Result<(), GroundTruthError> {
    if timestamp_ms.abs() > MAX_TIMESTAMP_ABS_MS {
        return Err(GroundTruthError::TimestampOutOfRange { row });
    }
    if !value.is_finite() {
        return Err(GroundTruthError::BadValue { row });
    }
    if !(0.0..=MAX_VALUE_BPM).contains(&value) {
        return Err(GroundTruthError::ValueOutOfRange { row, value });
    }
    if let Some(prev) = prev_ts {
        if timestamp_ms <= prev {
            return Err(GroundTruthError::NonMonotonicTimestamp { row });
        }
    }
    Ok(())
}

/// Validate an in-memory sample slice (row = zero-based index).
fn validate_samples(samples: &[ReferenceSample]) -> Result<(), GroundTruthError> {
    if samples.is_empty() {
        return Err(GroundTruthError::NoSamples);
    }
    let mut prev: Option<i64> = None;
    for (i, s) in samples.iter().enumerate() {
        validate_sample(i, s.timestamp_ms, s.value, prev)?;
        prev = Some(s.timestamp_ms);
    }
    Ok(())
}

/// A reference-device time series for one measurand.
///
/// Samples are guaranteed non-empty, finite, in-range, and strictly
/// increasing in time — the invariant is enforced by every constructor, so
/// downstream alignment/agreement code never re-checks it.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct ReferenceSeries {
    measurand: Measurand,
    device: ReferenceDevice,
    samples: Vec<ReferenceSample>,
}

impl ReferenceSeries {
    /// Build a series from in-memory samples, validating the invariant.
    ///
    /// Errors use the zero-based sample index as `row`.
    pub fn new(
        measurand: Measurand,
        device: ReferenceDevice,
        samples: Vec<ReferenceSample>,
    ) -> Result<Self, GroundTruthError> {
        validate_samples(&samples)?;
        Ok(Self {
            measurand,
            device,
            samples,
        })
    }

    /// Parse an untrusted `timestamp_ms,value` CSV export.
    ///
    /// The first non-blank line must be the header `timestamp_ms,value`
    /// (a UTF-8 BOM is tolerated). Blank lines are skipped; every other
    /// line must be `<i64>,<finite f64 in [0, 300]>`. Malformed rows are
    /// rejected with 1-based line numbers; non-monotonic timestamps are an
    /// error, never silently sorted. At most [`MAX_CSV_ROWS`] data rows are
    /// accepted.
    pub fn parse_csv(
        measurand: Measurand,
        device: ReferenceDevice,
        text: &str,
    ) -> Result<Self, GroundTruthError> {
        Self::parse_csv_bounded(measurand, device, text, MAX_CSV_ROWS)
    }

    /// [`Self::parse_csv`] with an explicit row limit (tested directly).
    fn parse_csv_bounded(
        measurand: Measurand,
        device: ReferenceDevice,
        text: &str,
        max_rows: usize,
    ) -> Result<Self, GroundTruthError> {
        let mut saw_header = false;
        let mut samples: Vec<ReferenceSample> = Vec::new();
        let mut prev_ts: Option<i64> = None;

        for (idx, raw) in text.lines().enumerate() {
            let row = idx + 1;
            let line = raw.trim_start_matches('\u{feff}').trim();
            if line.is_empty() {
                continue;
            }
            if !saw_header {
                if line != CSV_HEADER {
                    let mut found: String = line.chars().take(MAX_ERROR_ECHO).collect();
                    if found.len() < line.len() {
                        found.push('…');
                    }
                    return Err(GroundTruthError::BadHeader { found });
                }
                saw_header = true;
                continue;
            }
            if samples.len() >= max_rows {
                return Err(GroundTruthError::TooManyRows { max: max_rows });
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() != 2 {
                return Err(GroundTruthError::WrongFieldCount {
                    row,
                    found: fields.len(),
                });
            }
            let timestamp_ms: i64 = fields[0]
                .trim()
                .parse()
                .map_err(|_| GroundTruthError::BadTimestamp { row })?;
            let value: f64 = fields[1]
                .trim()
                .parse()
                .map_err(|_| GroundTruthError::BadValue { row })?;
            validate_sample(row, timestamp_ms, value, prev_ts)?;
            prev_ts = Some(timestamp_ms);
            samples.push(ReferenceSample {
                timestamp_ms,
                value,
            });
        }

        if !saw_header {
            return Err(GroundTruthError::MissingHeader);
        }
        if samples.is_empty() {
            return Err(GroundTruthError::NoSamples);
        }
        Ok(Self {
            measurand,
            device,
            samples,
        })
    }

    /// The measurand this series records.
    #[must_use]
    pub fn measurand(&self) -> Measurand {
        self.measurand
    }

    /// The reference device metadata.
    #[must_use]
    pub fn device(&self) -> &ReferenceDevice {
        &self.device
    }

    /// The validated samples (strictly increasing timestamps).
    #[must_use]
    pub fn samples(&self) -> &[ReferenceSample] {
        &self.samples
    }
}

// ---------------------------------------------------------------------------
// Estimate series (CSI-derived)
// ---------------------------------------------------------------------------

/// A CSI-derived estimate series, extracted from [`VitalReading`]s, carrying
/// the same validated-invariant as [`ReferenceSeries`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct EstimateSeries {
    measurand: Measurand,
    samples: Vec<ReferenceSample>,
}

impl EstimateSeries {
    /// Build from in-memory samples, validating the invariant.
    ///
    /// Errors use the zero-based sample index as `row`.
    pub fn new(
        measurand: Measurand,
        samples: Vec<ReferenceSample>,
    ) -> Result<Self, GroundTruthError> {
        validate_samples(&samples)?;
        Ok(Self { measurand, samples })
    }

    /// Extract one measurand from a slice of pipeline readings.
    ///
    /// Readings whose selected estimate has [`VitalStatus::Unavailable`] are
    /// skipped (Degraded/Unreliable estimates are kept — honest agreement
    /// statistics must include them). `timestamp_secs` is converted to unix
    /// milliseconds; non-finite or out-of-range timestamps/values are
    /// structured errors carrying the zero-based reading index as `row`.
    pub fn from_readings(
        measurand: Measurand,
        readings: &[VitalReading],
    ) -> Result<Self, GroundTruthError> {
        let mut samples: Vec<ReferenceSample> = Vec::new();
        let mut prev_ts: Option<i64> = None;
        for (row, reading) in readings.iter().enumerate() {
            let est = match measurand {
                Measurand::HeartRateBpm => &reading.heart_rate,
                Measurand::BreathingRateBrpm => &reading.respiratory_rate,
            };
            if est.status == VitalStatus::Unavailable {
                continue;
            }
            let ts_ms_f = reading.timestamp_secs * 1000.0;
            if !ts_ms_f.is_finite() || ts_ms_f.abs() > MAX_TIMESTAMP_ABS_MS as f64 {
                return Err(GroundTruthError::TimestampOutOfRange { row });
            }
            #[allow(clippy::cast_possible_truncation)]
            let timestamp_ms = ts_ms_f.round() as i64;
            validate_sample(row, timestamp_ms, est.value_bpm, prev_ts)?;
            prev_ts = Some(timestamp_ms);
            samples.push(ReferenceSample {
                timestamp_ms,
                value: est.value_bpm,
            });
        }
        if samples.is_empty() {
            return Err(GroundTruthError::NoSamples);
        }
        Ok(Self { measurand, samples })
    }

    /// Extract one measurand from everything currently held in a
    /// [`VitalSignStore`] session.
    ///
    /// Takes `&mut` because [`VitalSignStore::history`] rotates its ring
    /// buffer in place; contents are unchanged.
    pub fn from_store(
        measurand: Measurand,
        store: &mut VitalSignStore,
    ) -> Result<Self, GroundTruthError> {
        let n = store.len();
        Self::from_readings(measurand, store.history(n))
    }

    /// The measurand this series records.
    #[must_use]
    pub fn measurand(&self) -> Measurand {
        self.measurand
    }

    /// The validated samples (strictly increasing timestamps).
    #[must_use]
    pub fn samples(&self) -> &[ReferenceSample] {
        &self.samples
    }
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Number of grid points spanning `[start, end]` at `step` ms, bounded by
/// [`MAX_GRID_POINTS`].
fn grid_len(start_ms: i64, end_ms: i64, step_ms: i64) -> Result<usize, GroundTruthError> {
    debug_assert!(end_ms >= start_ms && step_ms > 0);
    let points = (end_ms - start_ms) / step_ms + 1;
    let points_u = points as u64;
    if points_u > MAX_GRID_POINTS as u64 {
        return Err(GroundTruthError::GridTooLarge { points: points_u });
    }
    Ok(points as usize)
}

/// Nearest-sample resampling onto a uniform grid.
///
/// A grid point at time `t` takes the value of the nearest sample if that
/// sample is within `max_dist_ms`; otherwise the grid point is `None`. No
/// interpolation is performed, so physiological values are never bridged
/// across gaps: with `max_dist_ms = max_gap_ms / 2`, two samples further
/// apart than `max_gap_ms` leave uncovered grid points between them.
fn resample_nearest(
    samples: &[ReferenceSample],
    grid_start_ms: i64,
    step_ms: i64,
    n_points: usize,
    max_dist_ms: i64,
) -> Vec<Option<f64>> {
    debug_assert!(!samples.is_empty());
    let mut out = Vec::with_capacity(n_points);
    let mut j = 0usize;
    for i in 0..n_points {
        let t = grid_start_ms + (i as i64) * step_ms;
        while j + 1 < samples.len()
            && (samples[j + 1].timestamp_ms - t).abs() < (samples[j].timestamp_ms - t).abs()
        {
            j += 1;
        }
        let dist = (samples[j].timestamp_ms - t).abs();
        out.push(if dist <= max_dist_ms {
            Some(samples[j].value)
        } else {
            None
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Time alignment
// ---------------------------------------------------------------------------

/// Configuration for [`align`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlignmentConfig {
    /// Bounded lag search window in milliseconds (default ±30 s).
    pub max_lag_ms: i64,
    /// Common resampling grid step in milliseconds (also the offset
    /// resolution; default 1000).
    pub grid_step_ms: i64,
    /// Maximum gap in milliseconds across which values may be carried to a
    /// grid point (nearest-sample within `max_gap_ms / 2`; default 5000).
    pub max_gap_ms: i64,
    /// Minimum overlapping valid pairs required at a candidate lag
    /// (default 10).
    pub min_overlap: usize,
    /// Whether to additionally fit a linear clock drift (default false).
    pub fit_drift: bool,
    /// Number of windows for the drift fit (default 4, minimum 2).
    pub drift_windows: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            max_lag_ms: 30_000,
            grid_step_ms: 1000,
            max_gap_ms: 5000,
            min_overlap: 10,
            fit_drift: false,
            drift_windows: 4,
        }
    }
}

impl AlignmentConfig {
    fn validate(&self) -> Result<(), GroundTruthError> {
        if self.grid_step_ms <= 0 {
            return Err(GroundTruthError::InvalidConfig("grid_step_ms must be > 0"));
        }
        if self.max_gap_ms <= 0 {
            return Err(GroundTruthError::InvalidConfig("max_gap_ms must be > 0"));
        }
        if self.max_lag_ms < 0 || self.max_lag_ms > MAX_TIMESTAMP_ABS_MS {
            return Err(GroundTruthError::InvalidConfig(
                "max_lag_ms must be in [0, 2^52]",
            ));
        }
        if self.min_overlap < 2 {
            return Err(GroundTruthError::InvalidConfig("min_overlap must be >= 2"));
        }
        if self.fit_drift && self.drift_windows < 2 {
            return Err(GroundTruthError::InvalidConfig(
                "drift_windows must be >= 2 when fit_drift is set",
            ));
        }
        Ok(())
    }
}

/// Optional linear clock-drift fit: the estimated offset as a linear
/// function of time, `offset(t) ≈ offset_at_start_ms + rate_ppm * 1e-6 * t`,
/// with `t` measured from the start of the common grid.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DriftFit {
    /// Fitted offset at the start of the common grid, milliseconds.
    pub offset_at_start_ms: f64,
    /// Fitted clock rate difference, parts per million (positive: the
    /// estimate clock runs slow relative to the reference clock).
    pub rate_ppm: f64,
    /// Number of windows that produced a usable local offset.
    pub windows_used: usize,
}

/// Result of [`align`]. Parameters are reported here and must be passed
/// explicitly to [`AgreementReport::compute`] — they are never silently
/// applied to any series.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlignmentResult {
    /// Estimated constant clock offset in milliseconds: add this to
    /// estimate timestamps to map them onto the reference clock.
    pub offset_ms: i64,
    /// Peak normalized cross-correlation at the chosen offset, in `[-1, 1]`.
    pub peak_ncc: f64,
    /// Number of overlapping valid grid pairs at the chosen offset.
    pub n_overlap: usize,
    /// Grid step used, milliseconds (the offset resolution).
    pub grid_step_ms: i64,
    /// Optional linear clock-drift fit (requested via
    /// [`AlignmentConfig::fit_drift`]; `None` if too few windows aligned).
    pub drift: Option<DriftFit>,
}

/// Best lag found by a bounded normalized cross-correlation search.
struct LagSearch {
    lag_steps: i64,
    ncc: f64,
    n_overlap: usize,
}

/// Search lags `-max_lag_steps..=max_lag_steps` for the maximum normalized
/// cross-correlation between `est[i]` and `refg[i + lag]` over pairs where
/// both grids hold a value. Ties prefer the smaller `|lag|` (deterministic:
/// lags are scanned in increasing order).
fn best_lag(
    est: &[Option<f64>],
    refg: &[Option<f64>],
    max_lag_steps: i64,
    min_overlap: usize,
) -> Result<LagSearch, GroundTruthError> {
    let n = est.len() as i64;
    let mut best: Option<LagSearch> = None;
    let mut max_overlap_seen = 0usize;

    for lag in -max_lag_steps..=max_lag_steps {
        let i_lo = 0.max(-lag);
        let i_hi = n.min(n - lag);
        if i_hi <= i_lo {
            continue;
        }
        // Zip the two aligned windows once per lag: no per-lag allocation,
        // and no per-element bounds check inside the O(lags × n) hot loop.
        let est_win = &est[i_lo as usize..i_hi as usize];
        let ref_win = &refg[(i_lo + lag) as usize..(i_hi + lag) as usize];
        let mut count = 0usize;
        let (mut se, mut sr, mut see, mut srr, mut ser) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for (&ev, &rv) in est_win.iter().zip(ref_win) {
            let (Some(e), Some(r)) = (ev, rv) else {
                continue;
            };
            count += 1;
            se += e;
            sr += r;
            see += e * e;
            srr += r * r;
            ser += e * r;
        }
        max_overlap_seen = max_overlap_seen.max(count);
        if count < min_overlap {
            continue;
        }
        let nf = count as f64;
        let var_e = see - se * se / nf;
        let var_r = srr - sr * sr / nf;
        if var_e <= 0.0 || var_r <= 0.0 {
            continue;
        }
        let ncc = (ser - se * sr / nf) / (var_e * var_r).sqrt();
        let take = match &best {
            None => true,
            Some(b) => ncc > b.ncc || (ncc == b.ncc && lag.abs() < b.lag_steps.abs()),
        };
        if take {
            best = Some(LagSearch {
                lag_steps: lag,
                ncc,
                n_overlap: count,
            });
        }
    }

    best.ok_or({
        if max_overlap_seen < min_overlap {
            GroundTruthError::InsufficientOverlap {
                required: min_overlap,
                found: max_overlap_seen,
            }
        } else {
            GroundTruthError::ConstantSignal
        }
    })
}

/// Fit a linear clock drift from per-window constant offsets.
///
/// The common grid is split into `cfg.drift_windows` equal windows; each
/// window runs its own bounded lag search, and the resulting
/// (window-center-time, local-offset) points are fit by least squares.
/// Returns `None` when fewer than two windows align.
fn fit_drift(
    est: &[Option<f64>],
    refg: &[Option<f64>],
    max_lag_steps: i64,
    cfg: &AlignmentConfig,
) -> Option<DriftFit> {
    let n = est.len();
    let windows = cfg.drift_windows;
    let mut xs: Vec<f64> = Vec::with_capacity(windows);
    let mut ys: Vec<f64> = Vec::with_capacity(windows);
    for w in 0..windows {
        let lo = w * n / windows;
        let hi = ((w + 1) * n / windows).min(n);
        if hi <= lo {
            continue;
        }
        if let Ok(local) = best_lag(&est[lo..hi], &refg[lo..hi], max_lag_steps, cfg.min_overlap) {
            let center_ms = ((lo + hi) as f64 / 2.0) * cfg.grid_step_ms as f64;
            xs.push(center_ms);
            ys.push((local.lag_steps * cfg.grid_step_ms) as f64);
        }
    }
    if xs.len() < 2 {
        return None;
    }
    let nf = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / nf;
    let y_mean = ys.iter().sum::<f64>() / nf;
    let sxx: f64 = xs.iter().map(|x| (x - x_mean) * (x - x_mean)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;
    Some(DriftFit {
        offset_at_start_ms: intercept,
        rate_ppm: slope * 1.0e6,
        windows_used: xs.len(),
    })
}

/// Estimate the constant clock offset between a CSI-derived estimate series
/// and a reference series by maximizing normalized cross-correlation over a
/// bounded lag window on a common nearest-sample grid.
///
/// Grid points further than `max_gap_ms / 2` from any sample are treated as
/// gaps and never bridged. The returned offset has `grid_step_ms`
/// resolution and is **reported, not applied** — pass it explicitly to
/// [`AgreementReport::compute`].
pub fn align(
    estimate: &EstimateSeries,
    reference: &ReferenceSeries,
    cfg: &AlignmentConfig,
) -> Result<AlignmentResult, GroundTruthError> {
    cfg.validate()?;
    if estimate.measurand != reference.measurand {
        return Err(GroundTruthError::MeasurandMismatch {
            estimate: estimate.measurand,
            reference: reference.measurand,
        });
    }
    let e = estimate.samples();
    let r = reference.samples();
    let start = e[0].timestamp_ms.min(r[0].timestamp_ms);
    let end = e[e.len() - 1]
        .timestamp_ms
        .max(r[r.len() - 1].timestamp_ms);
    let n = grid_len(start, end, cfg.grid_step_ms)?;
    let max_dist = cfg.max_gap_ms / 2;
    let eg = resample_nearest(e, start, cfg.grid_step_ms, n, max_dist);
    let rg = resample_nearest(r, start, cfg.grid_step_ms, n, max_dist);
    let max_lag_steps = cfg.max_lag_ms / cfg.grid_step_ms;

    let global = best_lag(&eg, &rg, max_lag_steps, cfg.min_overlap)?;
    let drift = if cfg.fit_drift {
        fit_drift(&eg, &rg, max_lag_steps, cfg)
    } else {
        None
    };

    Ok(AlignmentResult {
        offset_ms: global.lag_steps * cfg.grid_step_ms,
        peak_ncc: global.ncc,
        n_overlap: global.n_overlap,
        grid_step_ms: cfg.grid_step_ms,
        drift,
    })
}

// ---------------------------------------------------------------------------
// Session scope
// ---------------------------------------------------------------------------

/// Subject motion state during a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MotionState {
    /// Subject seated/lying, minimal movement.
    Static,
    /// Subject moving during the session.
    Moving,
}

/// RF propagation condition between sensor and subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Propagation {
    /// Clear line of sight.
    LineOfSight,
    /// Obstructed within the same room (furniture, people).
    NonLineOfSight,
    /// Signal traverses at least one wall.
    ThroughWall,
}

/// Coarse sensor-to-subject distance band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistanceBand {
    /// Up to 2 m.
    Near,
    /// 2 m to 5 m.
    Mid,
    /// Beyond 5 m.
    Far,
}

/// Mandatory scope statement for an agreement report (ADR-290): a vitals
/// number without its scope is systematically misleading, so a report
/// cannot be constructed without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionScope {
    /// Number of people in the sensing area during the session.
    pub subject_count: u32,
    /// Subject motion state.
    pub motion: MotionState,
    /// RF propagation condition.
    pub propagation: Propagation,
    /// Sensor-to-subject distance band.
    pub distance_band: DistanceBand,
}

// ---------------------------------------------------------------------------
// Agreement metrics
// ---------------------------------------------------------------------------

/// Configuration for [`AgreementReport::compute`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgreementConfig {
    /// Pairing grid step in milliseconds (default 1000).
    pub grid_step_ms: i64,
    /// Maximum gap in milliseconds across which values may be carried to a
    /// grid point (nearest-sample within `max_gap_ms / 2`; default 5000).
    pub max_gap_ms: i64,
    /// Agreement tolerance in BPM; `None` uses
    /// [`Measurand::default_tolerance_bpm`] (±2 bpm HR, ±1 brpm breathing).
    pub tolerance_bpm: Option<f64>,
}

impl Default for AgreementConfig {
    fn default() -> Self {
        Self {
            grid_step_ms: 1000,
            max_gap_ms: 5000,
            tolerance_bpm: None,
        }
    }
}

impl AgreementConfig {
    fn validate(&self) -> Result<(), GroundTruthError> {
        if self.grid_step_ms <= 0 {
            return Err(GroundTruthError::InvalidConfig("grid_step_ms must be > 0"));
        }
        if self.max_gap_ms <= 0 {
            return Err(GroundTruthError::InvalidConfig("max_gap_ms must be > 0"));
        }
        if let Some(t) = self.tolerance_bpm {
            if !t.is_finite() || t <= 0.0 {
                return Err(GroundTruthError::InvalidConfig(
                    "tolerance_bpm must be finite and > 0",
                ));
            }
        }
        Ok(())
    }
}

/// Agreement statistics between an aligned estimate series and a reference
/// series. Differences are `estimate - reference` in BPM.
///
/// The [`SessionScope`] field is mandatory by type: no report exists
/// without its scope. `applied_offset_ms` records the alignment that was
/// explicitly applied for pairing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AgreementReport {
    /// Measurand compared.
    pub measurand: Measurand,
    /// Reference device the estimates were compared against.
    pub device: ReferenceDevice,
    /// Mandatory session scope.
    pub scope: SessionScope,
    /// Constant clock offset (ms) that was explicitly applied to estimate
    /// timestamps for pairing.
    pub applied_offset_ms: i64,
    /// Number of paired samples.
    pub n_pairs: usize,
    /// Fraction of the overlapping-span grid where both series had a valid
    /// sample, in `[0, 1]`.
    pub coverage: f64,
    /// Mean absolute error, BPM.
    pub mae_bpm: f64,
    /// Root-mean-square error, BPM.
    pub rmse_bpm: f64,
    /// Mean error (bias), BPM.
    pub bias_bpm: f64,
    /// Bland-Altman lower 95% limit of agreement (`bias - 1.96·SD`), BPM.
    pub loa_lower_bpm: f64,
    /// Bland-Altman upper 95% limit of agreement (`bias + 1.96·SD`), BPM.
    pub loa_upper_bpm: f64,
    /// Tolerance used for `within_tolerance_fraction`, BPM.
    pub tolerance_bpm: f64,
    /// Fraction of pairs with `|estimate - reference| <= tolerance_bpm`.
    pub within_tolerance_fraction: f64,
}

impl AgreementReport {
    /// Compute agreement statistics between an estimate and a reference
    /// series, applying the given constant clock offset (typically
    /// [`AlignmentResult::offset_ms`]) to the estimate timestamps.
    ///
    /// The offset is a required, explicit argument — alignment is never
    /// applied silently — and is echoed back in `applied_offset_ms`.
    /// Pairing uses nearest-sample resampling on a grid over the
    /// overlapping span; gaps wider than `max_gap_ms` are never bridged.
    /// At least two pairs are required (Bland-Altman limits need a sample
    /// standard deviation).
    pub fn compute(
        estimate: &EstimateSeries,
        reference: &ReferenceSeries,
        applied_offset_ms: i64,
        cfg: &AgreementConfig,
        scope: SessionScope,
    ) -> Result<Self, GroundTruthError> {
        cfg.validate()?;
        if applied_offset_ms.abs() > MAX_TIMESTAMP_ABS_MS {
            return Err(GroundTruthError::InvalidConfig(
                "applied_offset_ms out of range",
            ));
        }
        if estimate.measurand != reference.measurand {
            return Err(GroundTruthError::MeasurandMismatch {
                estimate: estimate.measurand,
                reference: reference.measurand,
            });
        }
        let e = estimate.samples();
        let r = reference.samples();
        // Overlapping span on the reference clock; |ts| <= 2^52 and
        // |offset| <= 2^52 keep the sums well inside i64.
        let e_start = e[0].timestamp_ms + applied_offset_ms;
        let e_end = e[e.len() - 1].timestamp_ms + applied_offset_ms;
        let start = e_start.max(r[0].timestamp_ms);
        let end = e_end.min(r[r.len() - 1].timestamp_ms);
        if end < start {
            return Err(GroundTruthError::InsufficientOverlap {
                required: 2,
                found: 0,
            });
        }
        let n_grid = grid_len(start, end, cfg.grid_step_ms)?;
        let max_dist = cfg.max_gap_ms / 2;
        let eg = resample_nearest(
            e,
            start - applied_offset_ms,
            cfg.grid_step_ms,
            n_grid,
            max_dist,
        );
        let rg = resample_nearest(r, start, cfg.grid_step_ms, n_grid, max_dist);

        let diffs: Vec<f64> = eg
            .iter()
            .zip(&rg)
            .filter_map(|(ev, rv)| match (ev, rv) {
                (Some(ev), Some(rv)) => Some(ev - rv),
                _ => None,
            })
            .collect();
        let n_pairs = diffs.len();
        if n_pairs < 2 {
            return Err(GroundTruthError::InsufficientOverlap {
                required: 2,
                found: n_pairs,
            });
        }

        let nf = n_pairs as f64;
        let bias = diffs.iter().sum::<f64>() / nf;
        let mae = diffs.iter().map(|d| d.abs()).sum::<f64>() / nf;
        let rmse = (diffs.iter().map(|d| d * d).sum::<f64>() / nf).sqrt();
        let var = diffs.iter().map(|d| (d - bias) * (d - bias)).sum::<f64>() / (nf - 1.0);
        let sd = var.sqrt();
        let tolerance = cfg
            .tolerance_bpm
            .unwrap_or_else(|| estimate.measurand.default_tolerance_bpm());
        let within = diffs.iter().filter(|d| d.abs() <= tolerance).count() as f64 / nf;

        Ok(Self {
            measurand: estimate.measurand,
            device: reference.device.clone(),
            scope,
            applied_offset_ms,
            n_pairs,
            coverage: nf / n_grid as f64,
            mae_bpm: mae,
            rmse_bpm: rmse,
            bias_bpm: bias,
            loa_lower_bpm: bias - 1.96 * sd,
            loa_upper_bpm: bias + 1.96 * sd,
            tolerance_bpm: tolerance,
            within_tolerance_fraction: within,
        })
    }
}

// ---------------------------------------------------------------------------
// Evidence grading
// ---------------------------------------------------------------------------

/// Proof-of-measurement payload for [`EvidenceGrade::Measured`].
///
/// Has no public constructor: the only way to obtain one is
/// [`GradedAgreementReport::measured`], which enforces the gate. This makes
/// `MEASURED` unconstructible without passing the gate (ADR-288-style
/// enforcement in types).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct MeasuredEvidence {
    reproducer: String,
}

impl MeasuredEvidence {
    /// The exact command line that reproduces the reported numbers.
    #[must_use]
    pub fn reproducer(&self) -> &str {
        &self.reproducer
    }
}

/// Evidence grade per CLAUDE.md tagging rules.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum EvidenceGrade {
    /// Measured against a real reference device with a reproducer command.
    /// Only constructible through [`GradedAgreementReport::measured`].
    Measured(MeasuredEvidence),
    /// Real data, but the comparison does not meet the `MEASURED` gate
    /// (or is quoted rather than reproduced here).
    Claimed,
    /// Computed on synthetic/generated input.
    Synthetic,
}

impl EvidenceGrade {
    /// Stable uppercase tag (`MEASURED` / `CLAIMED` / `SYNTHETIC`).
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Measured(_) => "MEASURED",
            Self::Claimed => "CLAIMED",
            Self::Synthetic => "SYNTHETIC",
        }
    }
}

/// An [`AgreementReport`] paired with its [`EvidenceGrade`].
///
/// Fields are private; the constructors are the policy:
///
/// - [`Self::measured`] requires a reference device (structurally present in
///   every computed report), `n_pairs > 0`, coverage of at least
///   [`MIN_MEASURED_COVERAGE`], and a non-blank reproducer command.
/// - [`Self::claimed`] and [`Self::synthetic`] are always available.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct GradedAgreementReport {
    report: AgreementReport,
    evidence: EvidenceGrade,
}

impl GradedAgreementReport {
    /// Grade a report `MEASURED`. The gate is enforced here, not in docs:
    /// zero pairs, coverage below [`MIN_MEASURED_COVERAGE`], or a blank
    /// reproducer are structured errors.
    pub fn measured(
        report: AgreementReport,
        reproducer: &str,
    ) -> Result<Self, GroundTruthError> {
        if report.n_pairs == 0 {
            return Err(GroundTruthError::NotMeasured(
                "zero paired samples against the reference device",
            ));
        }
        if !report.coverage.is_finite() || report.coverage < MIN_MEASURED_COVERAGE {
            return Err(GroundTruthError::NotMeasured(
                "coverage below the minimum for a measured claim",
            ));
        }
        if reproducer.trim().is_empty() {
            return Err(GroundTruthError::NotMeasured(
                "a measured number without a reproducer command is not measured",
            ));
        }
        Ok(Self {
            report,
            evidence: EvidenceGrade::Measured(MeasuredEvidence {
                reproducer: reproducer.trim().to_string(),
            }),
        })
    }

    /// Grade a report `CLAIMED` (real data, gate not met or not reproduced
    /// here).
    #[must_use]
    pub fn claimed(report: AgreementReport) -> Self {
        Self {
            report,
            evidence: EvidenceGrade::Claimed,
        }
    }

    /// Grade a report `SYNTHETIC` (generated input).
    #[must_use]
    pub fn synthetic(report: AgreementReport) -> Self {
        Self {
            report,
            evidence: EvidenceGrade::Synthetic,
        }
    }

    /// The underlying agreement report.
    #[must_use]
    pub fn report(&self) -> &AgreementReport {
        &self.report
    }

    /// The evidence grade.
    #[must_use]
    pub fn evidence(&self) -> &EvidenceGrade {
        &self.evidence
    }
}

// ---------------------------------------------------------------------------
// Session evaluation (VitalSignStore integration)
// ---------------------------------------------------------------------------

/// Alignment plus agreement for one store session, with the alignment
/// parameters visible in both places.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct SessionEvaluation {
    /// The estimated alignment (offset and optional drift fit).
    pub alignment: AlignmentResult,
    /// Agreement computed with `alignment.offset_ms` explicitly applied
    /// (echoed in `report.applied_offset_ms`). Drift is reported only,
    /// never applied.
    pub report: AgreementReport,
}

/// Evaluate everything currently held in a [`VitalSignStore`] session
/// against a reference series: extract the matching measurand, estimate the
/// constant clock offset, and compute agreement with that offset explicitly
/// applied (and reported in the result).
///
/// Takes `&mut` store because reading history rotates its ring buffer in
/// place; contents are unchanged.
pub fn evaluate_session(
    store: &mut VitalSignStore,
    reference: &ReferenceSeries,
    align_cfg: &AlignmentConfig,
    agree_cfg: &AgreementConfig,
    scope: SessionScope,
) -> Result<SessionEvaluation, GroundTruthError> {
    let estimate = EstimateSeries::from_store(reference.measurand(), store)?;
    let alignment = align(&estimate, reference, align_cfg)?;
    let report = AgreementReport::compute(
        &estimate,
        reference,
        alignment.offset_ms,
        agree_cfg,
        scope,
    )?;
    Ok(SessionEvaluation { alignment, report })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{VitalEstimate, VitalReading, VitalStatus};

    fn device() -> ReferenceDevice {
        ReferenceDevice {
            make: "Polar".to_string(),
            model: "H10".to_string(),
            principle: MeasurementPrinciple::Ecg,
        }
    }

    fn scope() -> SessionScope {
        SessionScope {
            subject_count: 1,
            motion: MotionState::Static,
            propagation: Propagation::LineOfSight,
            distance_band: DistanceBand::Near,
        }
    }

    /// Deterministic aperiodic test signal (BPM range), aperiodic within
    /// the ±30 s lag window thanks to incommensurate periods.
    fn synth(t_secs: f64) -> f64 {
        70.0 + 5.0 * (2.0 * std::f64::consts::PI * t_secs / 47.0).sin()
            + 3.0 * (2.0 * std::f64::consts::PI * t_secs / 113.0).sin()
    }

    fn series_1hz(start_ms: i64, n: usize, f: impl Fn(f64) -> f64) -> Vec<ReferenceSample> {
        (0..n)
            .map(|i| {
                let ts = start_ms + (i as i64) * 1000;
                ReferenceSample {
                    timestamp_ms: ts,
                    value: f(ts as f64 / 1000.0),
                }
            })
            .collect()
    }

    // -- CSV parsing --------------------------------------------------------

    #[test]
    fn csv_parses_valid_input() {
        let csv = "timestamp_ms,value\n1000,72.5\n2000,73.0\n3000,71.5\n";
        let s = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap();
        assert_eq!(s.samples().len(), 3);
        assert_eq!(s.samples()[0].timestamp_ms, 1000);
        assert!((s.samples()[2].value - 71.5).abs() < f64::EPSILON);
        assert_eq!(s.measurand(), Measurand::HeartRateBpm);
        assert_eq!(s.device().make, "Polar");
    }

    #[test]
    fn csv_tolerates_crlf_blank_lines_and_bom() {
        let csv = "\u{feff}timestamp_ms,value\r\n\r\n1000,72\r\n2000,73\r\n\r\n";
        let s = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap();
        assert_eq!(s.samples().len(), 2);
    }

    #[test]
    fn csv_rejects_empty_input() {
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), "").unwrap_err();
        assert_eq!(err, GroundTruthError::MissingHeader);
    }

    #[test]
    fn csv_rejects_bad_header() {
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), "time,bpm\n1,2\n")
            .unwrap_err();
        assert!(matches!(err, GroundTruthError::BadHeader { .. }));
    }

    #[test]
    fn csv_bad_header_echo_is_bounded() {
        let long = "x".repeat(10_000);
        let err =
            ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), &long).unwrap_err();
        let GroundTruthError::BadHeader { found } = err else {
            panic!("expected BadHeader");
        };
        assert!(found.chars().count() <= MAX_ERROR_ECHO + 1);
    }

    #[test]
    fn csv_rejects_wrong_field_count_with_row_number() {
        let csv = "timestamp_ms,value\n1000,72\n2000,73,extra\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::WrongFieldCount { row: 3, found: 3 });

        let csv = "timestamp_ms,value\njustonefield\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::WrongFieldCount { row: 2, found: 1 });
    }

    #[test]
    fn csv_rejects_bad_timestamp_with_row_number() {
        let csv = "timestamp_ms,value\n1000,72\nnot_a_ts,73\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::BadTimestamp { row: 3 });
    }

    #[test]
    fn csv_rejects_bad_and_nonfinite_values() {
        let csv = "timestamp_ms,value\n1000,abc\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::BadValue { row: 2 });

        let csv = "timestamp_ms,value\n1000,NaN\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::BadValue { row: 2 });

        let csv = "timestamp_ms,value\n1000,inf\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::BadValue { row: 2 });
    }

    #[test]
    fn csv_rejects_out_of_range_value() {
        let csv = "timestamp_ms,value\n1000,400\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert!(matches!(err, GroundTruthError::ValueOutOfRange { row: 2, .. }));

        let csv = "timestamp_ms,value\n1000,-1\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert!(matches!(err, GroundTruthError::ValueOutOfRange { row: 2, .. }));
    }

    #[test]
    fn csv_rejects_non_monotonic_timestamps() {
        // Decreasing.
        let csv = "timestamp_ms,value\n2000,72\n1000,73\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::NonMonotonicTimestamp { row: 3 });

        // Duplicate.
        let csv = "timestamp_ms,value\n2000,72\n2000,73\n";
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), csv).unwrap_err();
        assert_eq!(err, GroundTruthError::NonMonotonicTimestamp { row: 3 });
    }

    #[test]
    fn csv_rejects_timestamp_out_of_range() {
        let csv = format!("timestamp_ms,value\n{},72\n", i64::MAX);
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), &csv).unwrap_err();
        // i64::MAX parses fine but exceeds the 2^52 bound.
        assert_eq!(err, GroundTruthError::TimestampOutOfRange { row: 2 });
    }

    #[test]
    fn csv_rejects_header_only() {
        let err = ReferenceSeries::parse_csv(Measurand::HeartRateBpm, device(), "timestamp_ms,value\n")
            .unwrap_err();
        assert_eq!(err, GroundTruthError::NoSamples);
    }

    #[test]
    fn csv_row_limit_is_enforced() {
        let csv = "timestamp_ms,value\n1000,70\n2000,71\n3000,72\n";
        let err =
            ReferenceSeries::parse_csv_bounded(Measurand::HeartRateBpm, device(), csv, 2)
                .unwrap_err();
        assert_eq!(err, GroundTruthError::TooManyRows { max: 2 });
    }

    #[test]
    fn series_new_validates_and_reports_index() {
        let err = ReferenceSeries::new(Measurand::HeartRateBpm, device(), vec![]).unwrap_err();
        assert_eq!(err, GroundTruthError::NoSamples);

        let samples = vec![
            ReferenceSample {
                timestamp_ms: 2000,
                value: 70.0,
            },
            ReferenceSample {
                timestamp_ms: 1000,
                value: 71.0,
            },
        ];
        let err = ReferenceSeries::new(Measurand::HeartRateBpm, device(), samples).unwrap_err();
        assert_eq!(err, GroundTruthError::NonMonotonicTimestamp { row: 1 });
    }

    // -- Alignment ----------------------------------------------------------

    fn cfg_default() -> AlignmentConfig {
        AlignmentConfig::default()
    }

    #[test]
    fn alignment_recovers_zero_offset() {
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 300, synth),
        )
        .unwrap();
        let estimate = EstimateSeries::new(
            Measurand::HeartRateBpm,
            series_1hz(0, 300, synth),
        )
        .unwrap();
        let result = align(&estimate, &reference, &cfg_default()).unwrap();
        assert_eq!(result.offset_ms, 0);
        assert!(result.peak_ncc > 0.999);
        assert!(result.drift.is_none());
    }

    #[test]
    fn alignment_recovers_known_positive_offset() {
        // Estimate device stamps events 7 s early: an estimate sample at
        // its own clock time t carries the reference value at t + 7 s, so
        // reference_time = estimate_time + 7000.
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 600, synth),
        )
        .unwrap();
        let est_samples: Vec<ReferenceSample> = (0..600)
            .map(|i| ReferenceSample {
                timestamp_ms: (i as i64) * 1000 - 7000,
                value: synth(i as f64),
            })
            .collect();
        let estimate = EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap();
        let result = align(&estimate, &reference, &cfg_default()).unwrap();
        assert_eq!(result.offset_ms, 7000);
        assert!(result.peak_ncc > 0.999);
    }

    #[test]
    fn alignment_recovers_known_negative_offset() {
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 600, synth),
        )
        .unwrap();
        let est_samples: Vec<ReferenceSample> = (0..600)
            .map(|i| ReferenceSample {
                timestamp_ms: (i as i64) * 1000 + 11_000,
                value: synth(i as f64),
            })
            .collect();
        let estimate = EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap();
        let result = align(&estimate, &reference, &cfg_default()).unwrap();
        assert_eq!(result.offset_ms, -11_000);
        assert!(result.peak_ncc > 0.999);
    }

    #[test]
    fn alignment_rejects_measurand_mismatch() {
        let reference = ReferenceSeries::new(
            Measurand::BreathingRateBrpm,
            device(),
            series_1hz(0, 60, |t| {
                15.0 + (t / 20.0).sin()
            }),
        )
        .unwrap();
        let estimate = EstimateSeries::new(
            Measurand::HeartRateBpm,
            series_1hz(0, 60, synth),
        )
        .unwrap();
        let err = align(&estimate, &reference, &cfg_default()).unwrap_err();
        assert!(matches!(err, GroundTruthError::MeasurandMismatch { .. }));
    }

    #[test]
    fn alignment_rejects_insufficient_overlap() {
        // Series 10 minutes apart with a ±30 s window: no lag overlaps.
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 60, synth),
        )
        .unwrap();
        let estimate = EstimateSeries::new(
            Measurand::HeartRateBpm,
            series_1hz(600_000, 60, synth),
        )
        .unwrap();
        let err = align(&estimate, &reference, &cfg_default()).unwrap_err();
        assert!(matches!(err, GroundTruthError::InsufficientOverlap { .. }));
    }

    #[test]
    fn alignment_rejects_constant_signal() {
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 60, |_| 70.0),
        )
        .unwrap();
        let estimate = EstimateSeries::new(
            Measurand::HeartRateBpm,
            series_1hz(0, 60, |_| 70.0),
        )
        .unwrap();
        let err = align(&estimate, &reference, &cfg_default()).unwrap_err();
        assert_eq!(err, GroundTruthError::ConstantSignal);
    }

    #[test]
    fn alignment_rejects_invalid_config() {
        let reference = ReferenceSeries::new(
            Measurand::HeartRateBpm,
            device(),
            series_1hz(0, 60, synth),
        )
        .unwrap();
        let estimate = EstimateSeries::new(
            Measurand::HeartRateBpm,
            series_1hz(0, 60, synth),
        )
        .unwrap();
        let cfg = AlignmentConfig {
            grid_step_ms: 0,
            ..AlignmentConfig::default()
        };
        assert!(matches!(
            align(&estimate, &reference, &cfg).unwrap_err(),
            GroundTruthError::InvalidConfig(_)
        ));
        let cfg = AlignmentConfig {
            fit_drift: true,
            drift_windows: 1,
            ..AlignmentConfig::default()
        };
        assert!(matches!(
            align(&estimate, &reference, &cfg).unwrap_err(),
            GroundTruthError::InvalidConfig(_)
        ));
    }

    #[test]
    fn alignment_drift_fit_recovers_synthetic_drift() {
        // The mapping reference_time = estimate_time + offset(t) with
        // offset(t) = 2000 ms + 0.01 * t (1% clock-rate error).
        let a_ms = 2000.0;
        let b = 0.01;
        let n_est = 2000usize;
        let est_samples: Vec<ReferenceSample> = (0..n_est)
            .map(|i| {
                let est_ts = (i as i64) * 1000;
                let ref_time_s = (est_ts as f64 + a_ms + b * est_ts as f64) / 1000.0;
                ReferenceSample {
                    timestamp_ms: est_ts,
                    value: synth(ref_time_s),
                }
            })
            .collect();
        let ref_samples = series_1hz(0, 2101, synth);
        let reference =
            ReferenceSeries::new(Measurand::HeartRateBpm, device(), ref_samples).unwrap();
        let estimate = EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap();
        let cfg = AlignmentConfig {
            fit_drift: true,
            ..AlignmentConfig::default()
        };
        let result = align(&estimate, &reference, &cfg).unwrap();
        let drift = result.drift.expect("drift fit should succeed");
        assert_eq!(drift.windows_used, 4);
        // True rate is 10_000 ppm; local offsets quantize to the 1 s grid,
        // so allow a generous but decisive tolerance.
        assert!(
            (drift.rate_ppm - 10_000.0).abs() < 2000.0,
            "rate_ppm = {}",
            drift.rate_ppm
        );
        assert!(
            (drift.offset_at_start_ms - a_ms).abs() < 1500.0,
            "offset_at_start_ms = {}",
            drift.offset_at_start_ms
        );
    }

    #[test]
    fn resampling_does_not_bridge_wide_gaps() {
        let samples = vec![
            ReferenceSample {
                timestamp_ms: 0,
                value: 70.0,
            },
            ReferenceSample {
                timestamp_ms: 10_000,
                value: 71.0,
            },
        ];
        // max_dist 500 ms: grid points between the two samples stay None.
        let grid = resample_nearest(&samples, 0, 1000, 11, 500);
        assert_eq!(grid[0], Some(70.0));
        assert_eq!(grid[10], Some(71.0));
        for g in &grid[1..10] {
            assert_eq!(*g, None);
        }
    }

    // -- Agreement ----------------------------------------------------------

    fn agree_cfg(tolerance: Option<f64>) -> AgreementConfig {
        AgreementConfig {
            grid_step_ms: 1000,
            max_gap_ms: 2000,
            tolerance_bpm: tolerance,
        }
    }

    /// Fixture: diffs (est - ref) = [1, -1, 2, 0] over four 1 Hz pairs.
    ///
    /// Hand-computed: bias = 0.5, MAE = 1.0, RMSE = sqrt(1.5),
    /// SD (n-1) = sqrt(5/3), LoA = 0.5 ± 1.96*sqrt(5/3).
    fn fixture_pair() -> (EstimateSeries, ReferenceSeries) {
        let ref_samples: Vec<ReferenceSample> = (0..4)
            .map(|i| ReferenceSample {
                timestamp_ms: i * 1000,
                value: 70.0,
            })
            .collect();
        let est_values = [71.0, 69.0, 72.0, 70.0];
        let est_samples: Vec<ReferenceSample> = est_values
            .iter()
            .enumerate()
            .map(|(i, v)| ReferenceSample {
                timestamp_ms: (i as i64) * 1000,
                value: *v,
            })
            .collect();
        (
            EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap(),
            ReferenceSeries::new(Measurand::HeartRateBpm, device(), ref_samples).unwrap(),
        )
    }

    #[test]
    fn agreement_matches_hand_computed_fixture() {
        let (estimate, reference) = fixture_pair();
        let report =
            AgreementReport::compute(&estimate, &reference, 0, &agree_cfg(None), scope()).unwrap();

        assert_eq!(report.n_pairs, 4);
        assert!((report.coverage - 1.0).abs() < 1e-12);
        assert!((report.bias_bpm - 0.5).abs() < 1e-12);
        assert!((report.mae_bpm - 1.0).abs() < 1e-12);
        assert!((report.rmse_bpm - 1.5f64.sqrt()).abs() < 1e-12);
        let sd = (5.0f64 / 3.0).sqrt();
        assert!((report.loa_lower_bpm - (0.5 - 1.96 * sd)).abs() < 1e-12);
        assert!((report.loa_upper_bpm - (0.5 + 1.96 * sd)).abs() < 1e-12);
        // Default HR tolerance ±2 bpm: all four diffs are within.
        assert!((report.tolerance_bpm - 2.0).abs() < f64::EPSILON);
        assert!((report.within_tolerance_fraction - 1.0).abs() < 1e-12);
        assert_eq!(report.applied_offset_ms, 0);
        assert_eq!(report.scope, scope());
    }

    #[test]
    fn agreement_within_tolerance_with_explicit_tolerance() {
        let (estimate, reference) = fixture_pair();
        let report =
            AgreementReport::compute(&estimate, &reference, 0, &agree_cfg(Some(1.0)), scope())
                .unwrap();
        // Diffs [1, -1, 2, 0]: three of four within ±1.
        assert!((report.within_tolerance_fraction - 0.75).abs() < 1e-12);
        assert!((report.tolerance_bpm - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn agreement_breathing_default_tolerance_is_1_brpm() {
        let ref_samples = series_1hz(0, 10, |_| 15.0);
        let est_samples = series_1hz(0, 10, |_| 16.5);
        let reference =
            ReferenceSeries::new(Measurand::BreathingRateBrpm, device(), ref_samples).unwrap();
        let estimate = EstimateSeries::new(Measurand::BreathingRateBrpm, est_samples).unwrap();
        let report =
            AgreementReport::compute(&estimate, &reference, 0, &agree_cfg(None), scope()).unwrap();
        assert!((report.tolerance_bpm - 1.0).abs() < f64::EPSILON);
        // All diffs are +1.5 brpm: none within ±1.
        assert!((report.within_tolerance_fraction - 0.0).abs() < 1e-12);
        assert!((report.bias_bpm - 1.5).abs() < 1e-9);
    }

    #[test]
    fn agreement_coverage_reflects_unbridged_gaps() {
        // Reference covers 0..=10 s; estimate is missing 4..=7 s. With
        // max_gap 1000 (max_dist 500), the four gap grid points stay
        // unpaired: 7 pairs over an 11-point grid.
        let ref_samples = series_1hz(0, 11, synth);
        let est_samples: Vec<ReferenceSample> = (0..11)
            .filter(|i| !(4..=7).contains(i))
            .map(|i| ReferenceSample {
                timestamp_ms: (i as i64) * 1000,
                value: synth(i as f64),
            })
            .collect();
        let reference =
            ReferenceSeries::new(Measurand::HeartRateBpm, device(), ref_samples).unwrap();
        let estimate = EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap();
        let cfg = AgreementConfig {
            grid_step_ms: 1000,
            max_gap_ms: 1000,
            tolerance_bpm: None,
        };
        let report = AgreementReport::compute(&estimate, &reference, 0, &cfg, scope()).unwrap();
        assert_eq!(report.n_pairs, 7);
        assert!((report.coverage - 7.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn agreement_applies_offset_explicitly() {
        // Estimate timestamps 5 s behind the reference clock; passing the
        // alignment offset pairs them exactly.
        let ref_samples = series_1hz(0, 120, synth);
        let est_samples: Vec<ReferenceSample> = (0..120)
            .map(|i| ReferenceSample {
                timestamp_ms: (i as i64) * 1000 - 5000,
                value: synth(i as f64),
            })
            .collect();
        let reference =
            ReferenceSeries::new(Measurand::HeartRateBpm, device(), ref_samples).unwrap();
        let estimate = EstimateSeries::new(Measurand::HeartRateBpm, est_samples).unwrap();
        let report =
            AgreementReport::compute(&estimate, &reference, 5000, &agree_cfg(None), scope())
                .unwrap();
        assert_eq!(report.applied_offset_ms, 5000);
        assert!(report.mae_bpm < 1e-9);
        // Without the offset the same series disagree.
        let misaligned =
            AgreementReport::compute(&estimate, &reference, 0, &agree_cfg(None), scope()).unwrap();
        assert!(misaligned.mae_bpm > report.mae_bpm);
    }

    #[test]
    fn agreement_rejects_disjoint_and_mismatched_series() {
        let (estimate, reference) = fixture_pair();
        // No overlap after a huge offset.
        let err = AgreementReport::compute(
            &estimate,
            &reference,
            1_000_000,
            &agree_cfg(None),
            scope(),
        )
        .unwrap_err();
        assert!(matches!(err, GroundTruthError::InsufficientOverlap { .. }));

        // Measurand mismatch.
        let breathing = EstimateSeries::new(
            Measurand::BreathingRateBrpm,
            series_1hz(0, 4, |_| 15.0),
        )
        .unwrap();
        let err = AgreementReport::compute(&breathing, &reference, 0, &agree_cfg(None), scope())
            .unwrap_err();
        assert!(matches!(err, GroundTruthError::MeasurandMismatch { .. }));
    }

    // -- Evidence grading ---------------------------------------------------

    fn good_report() -> AgreementReport {
        let (estimate, reference) = fixture_pair();
        AgreementReport::compute(&estimate, &reference, 0, &agree_cfg(None), scope()).unwrap()
    }

    #[test]
    fn measured_grade_requires_reproducer() {
        let graded = GradedAgreementReport::measured(
            good_report(),
            "cargo test -p wifi-densepose-vitals groundtruth",
        )
        .unwrap();
        assert_eq!(graded.evidence().tag(), "MEASURED");
        let EvidenceGrade::Measured(evidence) = graded.evidence() else {
            panic!("expected Measured");
        };
        assert_eq!(
            evidence.reproducer(),
            "cargo test -p wifi-densepose-vitals groundtruth"
        );

        let err = GradedAgreementReport::measured(good_report(), "   ").unwrap_err();
        assert!(matches!(err, GroundTruthError::NotMeasured(_)));
    }

    #[test]
    fn measured_grade_rejects_zero_pairs() {
        let mut report = good_report();
        report.n_pairs = 0;
        let err = GradedAgreementReport::measured(report, "cargo test").unwrap_err();
        assert!(matches!(err, GroundTruthError::NotMeasured(_)));
    }

    #[test]
    fn measured_grade_rejects_low_coverage() {
        let mut report = good_report();
        report.coverage = MIN_MEASURED_COVERAGE - 0.01;
        let err = GradedAgreementReport::measured(report, "cargo test").unwrap_err();
        assert!(matches!(err, GroundTruthError::NotMeasured(_)));
    }

    #[test]
    fn claimed_and_synthetic_grades_are_always_constructible() {
        let claimed = GradedAgreementReport::claimed(good_report());
        assert_eq!(claimed.evidence().tag(), "CLAIMED");
        let synthetic = GradedAgreementReport::synthetic(good_report());
        assert_eq!(synthetic.evidence().tag(), "SYNTHETIC");
        assert_eq!(synthetic.report().n_pairs, 4);
    }

    // -- VitalSignStore integration -----------------------------------------

    fn reading(ts_secs: f64, hr: f64, rr: f64, hr_status: VitalStatus) -> VitalReading {
        VitalReading {
            respiratory_rate: VitalEstimate {
                value_bpm: rr,
                confidence: 0.9,
                status: VitalStatus::Valid,
            },
            heart_rate: VitalEstimate {
                value_bpm: hr,
                confidence: 0.85,
                status: hr_status,
            },
            subcarrier_count: 56,
            signal_quality: 0.9,
            timestamp_secs: ts_secs,
        }
    }

    #[test]
    fn estimate_series_from_readings_skips_unavailable() {
        let readings = vec![
            reading(0.0, 70.0, 15.0, VitalStatus::Valid),
            reading(1.0, 0.0, 15.0, VitalStatus::Unavailable),
            reading(2.0, 72.0, 15.0, VitalStatus::Degraded),
        ];
        let series = EstimateSeries::from_readings(Measurand::HeartRateBpm, &readings).unwrap();
        assert_eq!(series.samples().len(), 2);
        assert_eq!(series.samples()[0].timestamp_ms, 0);
        assert_eq!(series.samples()[1].timestamp_ms, 2000);
        assert!((series.samples()[1].value - 72.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_series_from_readings_rejects_bad_input() {
        let readings = vec![
            reading(1.0, 70.0, 15.0, VitalStatus::Valid),
            reading(1.0, 71.0, 15.0, VitalStatus::Valid),
        ];
        let err = EstimateSeries::from_readings(Measurand::HeartRateBpm, &readings).unwrap_err();
        assert_eq!(err, GroundTruthError::NonMonotonicTimestamp { row: 1 });

        let readings = vec![reading(f64::NAN, 70.0, 15.0, VitalStatus::Valid)];
        let err = EstimateSeries::from_readings(Measurand::HeartRateBpm, &readings).unwrap_err();
        assert_eq!(err, GroundTruthError::TimestampOutOfRange { row: 0 });

        let readings = vec![reading(0.0, 0.0, 15.0, VitalStatus::Unavailable)];
        let err = EstimateSeries::from_readings(Measurand::HeartRateBpm, &readings).unwrap_err();
        assert_eq!(err, GroundTruthError::NoSamples);
    }

    #[test]
    fn evaluate_session_end_to_end_recovers_offset_and_agrees() {
        // Store readings at 1 Hz on the estimate clock; the reference
        // device stamps the same physiological signal 5 s later
        // (reference_time = estimate_time + 5000).
        let mut store = VitalSignStore::new(1000);
        for i in 0..300 {
            store.push(reading(i as f64, synth(i as f64 + 5.0), 15.0, VitalStatus::Valid));
        }
        let ref_samples: Vec<ReferenceSample> = (0..300)
            .map(|i| ReferenceSample {
                timestamp_ms: (i as i64) * 1000 + 5000,
                value: synth(i as f64 + 5.0),
            })
            .collect();
        let reference =
            ReferenceSeries::new(Measurand::HeartRateBpm, device(), ref_samples).unwrap();

        let eval = evaluate_session(
            &mut store,
            &reference,
            &AlignmentConfig::default(),
            &AgreementConfig::default(),
            scope(),
        )
        .unwrap();

        assert_eq!(eval.alignment.offset_ms, 5000);
        assert_eq!(eval.report.applied_offset_ms, 5000);
        assert!(eval.report.mae_bpm < 1e-9);
        assert!(eval.report.coverage > 0.99);
        assert!(eval.report.n_pairs >= 290);

        // And the result meets the MEASURED gate.
        let graded = GradedAgreementReport::measured(
            eval.report,
            "cargo test -p wifi-densepose-vitals evaluate_session_end_to_end",
        )
        .unwrap();
        assert_eq!(graded.evidence().tag(), "MEASURED");
    }

    #[test]
    fn error_display_is_stable() {
        let err = GroundTruthError::NonMonotonicTimestamp { row: 7 };
        assert_eq!(
            err.to_string(),
            "row 7: timestamps must be strictly increasing"
        );
        assert_eq!(
            GroundTruthError::MissingHeader.to_string(),
            "missing CSV header line 'timestamp_ms,value'"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn graded_report_serializes() {
        let graded = GradedAgreementReport::measured(good_report(), "cargo test").unwrap();
        let json = serde_json::to_string(&graded).unwrap();
        assert!(json.contains("Measured"));
        assert!(json.contains("cargo test"));
    }
}
