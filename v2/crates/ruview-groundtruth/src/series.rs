//! Timestamped reference and estimate series with boundary validation
//! (ADR-300 §1, reusing ADR-290's ingest discipline).
//!
//! Both a reference (independent observer) and an RF estimate are sequences of
//! timestamped [`Reading`]s for one [`Measurand`]. Timestamps must be strictly
//! increasing (non-monotonic input is rejected, never silently sorted), scalar
//! values must be finite, and each reading's family must match the measurand.
//! Sample counts are bounded to cap allocation on untrusted input.

use serde::{Deserialize, Serialize};

use crate::error::{check_bound, check_nonempty, GroundTruthError};
use crate::model::{DataProvenance, Measurand, Reading};
use crate::source::ReferenceSource;

/// The largest series length accepted, bounding allocation on untrusted input.
pub const MAX_SAMPLES: usize = 1_000_000;

/// A single timestamped observation on the validation plane: a producer-stamped
/// Unix-millisecond time and a [`Reading`]. Time is always injected, never read
/// from a clock inside this crate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceObservation {
    /// Producer-supplied observation time, Unix milliseconds.
    pub at_unix_ms: i64,
    /// The observed value or label.
    pub reading: Reading,
}

impl ReferenceObservation {
    /// A scalar (continuous) observation at `at_unix_ms`.
    #[must_use]
    pub fn scalar(at_unix_ms: i64, value: f64) -> Self {
        Self {
            at_unix_ms,
            reading: Reading::Scalar(value),
        }
    }

    /// A label (categorical) observation at `at_unix_ms`.
    #[must_use]
    pub fn label(at_unix_ms: i64, label: impl Into<String>) -> Self {
        Self {
            at_unix_ms,
            reading: Reading::Label(label.into()),
        }
    }
}

/// Validate a sample vector: non-empty, bounded, strictly increasing
/// timestamps, finite scalars, and reading family matching `measurand`.
fn validate_samples(
    measurand: Measurand,
    samples: &[ReferenceObservation],
) -> Result<(), GroundTruthError> {
    if samples.is_empty() {
        return Err(GroundTruthError::EmptySeries);
    }
    if samples.len() > MAX_SAMPLES {
        return Err(GroundTruthError::TooManySamples {
            len: samples.len(),
            max: MAX_SAMPLES,
        });
    }
    let want = measurand.kind();
    let mut prev: Option<i64> = None;
    for (index, s) in samples.iter().enumerate() {
        if s.reading.kind() != want {
            return Err(GroundTruthError::ReadingKindMismatch {
                index,
                measurand: measurand.label(),
            });
        }
        if let Reading::Scalar(v) = &s.reading {
            if !v.is_finite() {
                return Err(GroundTruthError::NonFiniteValue { index });
            }
        }
        if let Reading::Label(l) = &s.reading {
            check_bound("label", l)?;
        }
        if let Some(p) = prev {
            if s.at_unix_ms <= p {
                return Err(GroundTruthError::NonMonotonic {
                    index,
                    prev_ms: p,
                    this_ms: s.at_unix_ms,
                });
            }
        }
        prev = Some(s.at_unix_ms);
    }
    Ok(())
}

/// A validated series of independent reference observations for one measurand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceSeries {
    /// The named reference source.
    pub source: ReferenceSource,
    /// The measurand observed.
    pub measurand: Measurand,
    samples: Vec<ReferenceObservation>,
}

impl ReferenceSeries {
    /// Ingest and validate a reference series at the boundary.
    ///
    /// # Errors
    /// [`GroundTruthError::EmptySeries`], [`GroundTruthError::TooManySamples`],
    /// [`GroundTruthError::NonMonotonic`], [`GroundTruthError::NonFiniteValue`],
    /// [`GroundTruthError::ReadingKindMismatch`], or
    /// [`GroundTruthError::TooLong`].
    pub fn new(
        source: ReferenceSource,
        measurand: Measurand,
        samples: Vec<ReferenceObservation>,
    ) -> Result<Self, GroundTruthError> {
        validate_samples(measurand, &samples)?;
        Ok(Self {
            source,
            measurand,
            samples,
        })
    }

    /// The validated samples, in time order.
    #[must_use]
    pub fn samples(&self) -> &[ReferenceObservation] {
        &self.samples
    }

    /// The number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the series is empty. Always `false` for a constructed series
    /// (empty input is rejected), provided so clippy's `len`-without-`is_empty`
    /// lint is satisfied.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// A validated series of RF-estimate observations for one measurand, carrying
/// the producing model version and whether the data is real or synthetic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimateSeries {
    /// The measurand estimated.
    pub measurand: Measurand,
    /// The model version that produced the estimates (ADR-136).
    pub model_version: String,
    /// Whether the estimates are real inference or synthetic input.
    pub provenance: DataProvenance,
    samples: Vec<ReferenceObservation>,
}

impl EstimateSeries {
    /// Ingest and validate an estimate series at the boundary. `model_version`
    /// must be non-empty and length-bounded.
    ///
    /// # Errors
    /// As [`ReferenceSeries::new`], plus [`GroundTruthError::EmptyField`] for a
    /// missing `model_version`.
    pub fn new(
        measurand: Measurand,
        model_version: impl Into<String>,
        provenance: DataProvenance,
        samples: Vec<ReferenceObservation>,
    ) -> Result<Self, GroundTruthError> {
        let model_version = model_version.into();
        check_bound("model_version", &model_version)?;
        check_nonempty("model_version", &model_version)?;
        validate_samples(measurand, &samples)?;
        Ok(Self {
            measurand,
            model_version,
            provenance,
            samples,
        })
    }

    /// The validated samples, in time order.
    #[must_use]
    pub fn samples(&self) -> &[ReferenceObservation] {
        &self.samples
    }

    /// The number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the series is empty (always `false` for a constructed series).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
