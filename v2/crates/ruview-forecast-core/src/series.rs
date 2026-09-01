//! Validated feature schemas, time series, and train-only scaler state.

use crate::digest::CanonicalWriter;
use crate::{CanonicalDigest, DataPolicy, ForecastError, SourceState};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

/// Maximum number of variates in one schema.
pub const MAX_FEATURES: usize = 256;
/// Maximum UTF-8 byte length of a feature name.
pub const MAX_FEATURE_NAME_LEN: usize = 128;
/// Maximum UTF-8 byte length of a unit string.
pub const MAX_UNIT_LEN: usize = 64;
/// Maximum UTF-8 byte length of a provenance reference.
pub const MAX_SOURCE_REFERENCE_LEN: usize = 1_024;
/// Maximum rows accepted in one in-memory series.
pub const MAX_SERIES_ROWS: usize = 16_384;
/// Maximum flattened values accepted in one in-memory series.
pub const MAX_SERIES_VALUES: usize = 4_194_304;

/// One named variate and its physical unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureSpec {
    name: String,
    unit: String,
}

impl FeatureSpec {
    /// Construct a bounded, canonical feature description.
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Result<Self, ForecastError> {
        let name = name.into();
        let unit = unit.into();
        validate_text("feature_name", &name, MAX_FEATURE_NAME_LEN, false)?;
        validate_text("feature_unit", &unit, MAX_UNIT_LEN, false)?;
        Ok(Self { name, unit })
    }

    /// Stable feature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Physical unit, such as `dBm` or `Hz`.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureSpecWire {
    name: String,
    unit: String,
}

impl<'de> Deserialize<'de> for FeatureSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FeatureSpecWire::deserialize(deserializer)?;
        Self::new(wire.name, wire.unit).map_err(serde::de::Error::custom)
    }
}

/// Ordered multivariate feature schema.
///
/// Order is load-bearing because series values are row-major by this schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureSchema {
    features: Vec<FeatureSpec>,
}

impl FeatureSchema {
    /// Validate a non-empty, bounded schema with unique feature names.
    pub fn new(features: Vec<FeatureSpec>) -> Result<Self, ForecastError> {
        if features.is_empty() {
            return Err(ForecastError::EmptyField { field: "features" });
        }
        if features.len() > MAX_FEATURES {
            return Err(ForecastError::LimitExceeded {
                field: "features",
                actual: features.len(),
                max: MAX_FEATURES,
            });
        }
        let mut names = BTreeSet::new();
        for feature in &features {
            if !names.insert(feature.name.as_str()) {
                return Err(ForecastError::DuplicateFeature {
                    name: feature.name.clone(),
                });
            }
        }
        Ok(Self { features })
    }

    /// Ordered features.
    #[must_use]
    pub fn features(&self) -> &[FeatureSpec] {
        &self.features
    }

    /// Number of variates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether the schema has no variates. A validated schema is never empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Deterministic digest of ordered names and units.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"feature-schema-v1");
        writer.usize(self.features.len());
        for feature in &self.features {
            writer.string(&feature.name);
            writer.string(&feature.unit);
        }
        writer.finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureSchemaWire {
    features: Vec<FeatureSpec>,
}

impl<'de> Deserialize<'de> for FeatureSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FeatureSchemaWire::deserialize(deserializer)?;
        Self::new(wire.features).map_err(serde::de::Error::custom)
    }
}

/// Bounded multivariate observations with explicit missingness.
///
/// `values` and `observed_mask` are row-major with flat index
/// `row * variates + variate`. Every supplied float must be finite. Missing
/// slots are canonicalized to `0.0`, but remain missing because their mask is
/// false; downstream code must never interpret the placeholder as observed.
/// The in-process cell ceiling is not a transport byte limit: adapters must
/// cap request or file bytes before Serde allocates these vectors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimeSeries {
    schema: FeatureSchema,
    timestamps_ms: Vec<u64>,
    values: Vec<f32>,
    observed_mask: Vec<bool>,
    source: SourceState,
    policy: DataPolicy,
}

impl TimeSeries {
    /// Construct and validate a bounded row-major series.
    pub fn new(
        schema: FeatureSchema,
        timestamps_ms: Vec<u64>,
        mut values: Vec<f32>,
        observed_mask: Vec<bool>,
        source: SourceState,
        policy: DataPolicy,
    ) -> Result<Self, ForecastError> {
        if timestamps_ms.is_empty() {
            return Err(ForecastError::EmptyField {
                field: "timestamps_ms",
            });
        }
        if timestamps_ms.len() > MAX_SERIES_ROWS {
            return Err(ForecastError::LimitExceeded {
                field: "series_rows",
                actual: timestamps_ms.len(),
                max: MAX_SERIES_ROWS,
            });
        }
        for (index, pair) in timestamps_ms.windows(2).enumerate() {
            if pair[0] >= pair[1] {
                return Err(ForecastError::NonMonotonicTimestamp { index: index + 1 });
            }
        }
        let expected = checked_product(
            "series_values",
            &[timestamps_ms.len(), schema.len()],
            MAX_SERIES_VALUES,
        )?;
        check_shape("values", expected, values.len())?;
        check_shape("observed_mask", expected, observed_mask.len())?;
        check_finite("values", &values)?;
        for (value, observed) in values.iter_mut().zip(&observed_mask) {
            if !observed || *value == 0.0 {
                *value = 0.0;
            }
        }
        source.validate()?;
        if policy.retention_until_ms() < timestamps_ms[timestamps_ms.len() - 1] {
            return Err(ForecastError::PrivacyDenied {
                operation: "series construction",
                reason: "retention expires before the final observation",
            });
        }
        Ok(Self {
            schema,
            timestamps_ms,
            values,
            observed_mask,
            source,
            policy,
        })
    }

    /// Feature schema.
    #[must_use]
    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    /// Strictly increasing timestamps in milliseconds.
    #[must_use]
    pub fn timestamps_ms(&self) -> &[u64] {
        &self.timestamps_ms
    }

    /// Canonical row-major values. Inspect [`Self::observed_mask`] before use.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Row-major observation mask.
    #[must_use]
    pub fn observed_mask(&self) -> &[bool] {
        &self.observed_mask
    }

    /// Provenance state fixed at construction.
    #[must_use]
    pub fn source(&self) -> &SourceState {
        &self.source
    }

    /// Privacy, tenant, purpose, consent, and retention binding.
    #[must_use]
    pub fn policy(&self) -> &DataPolicy {
        &self.policy
    }

    /// Number of timestamped rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.timestamps_ms.len()
    }

    /// Number of variates per row.
    #[must_use]
    pub fn variates(&self) -> usize {
        self.schema.len()
    }

    /// Last timestamp in this non-empty series.
    #[must_use]
    pub fn end_timestamp_ms(&self) -> u64 {
        self.timestamps_ms[self.timestamps_ms.len() - 1]
    }

    /// Return one observed value, or `None` for an out-of-range/missing slot.
    #[must_use]
    pub fn value(&self, row: usize, variate: usize) -> Option<f32> {
        let index = self.flat_index(row, variate)?;
        self.observed_mask[index].then_some(self.values[index])
    }

    /// Return whether one in-range slot is observed.
    #[must_use]
    pub fn is_observed(&self, row: usize, variate: usize) -> bool {
        self.flat_index(row, variate)
            .is_some_and(|index| self.observed_mask[index])
    }

    /// Find the most recent observed value for a variate.
    #[must_use]
    pub fn last_observed(&self, variate: usize) -> Option<f32> {
        if variate >= self.variates() {
            return None;
        }
        (0..self.rows())
            .rev()
            .find_map(|row| self.value(row, variate))
    }

    /// Deterministic digest of schema, timestamps, values, mask, and source.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"time-series-v1");
        writer.digest(self.schema.canonical_digest());
        writer.usize(self.rows());
        for timestamp in &self.timestamps_ms {
            writer.u64(*timestamp);
        }
        writer.usize(self.values.len());
        for (value, observed) in self.values.iter().zip(&self.observed_mask) {
            writer.bool(*observed);
            writer.f32(*value);
        }
        self.source.write_canonical(&mut writer);
        writer.digest(self.policy.canonical_digest());
        writer.finish()
    }

    fn flat_index(&self, row: usize, variate: usize) -> Option<usize> {
        if row >= self.rows() || variate >= self.variates() {
            return None;
        }
        row.checked_mul(self.variates())?.checked_add(variate)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeSeriesWire {
    schema: FeatureSchema,
    timestamps_ms: Vec<u64>,
    values: Vec<f32>,
    observed_mask: Vec<bool>,
    source: SourceState,
    policy: DataPolicy,
}

impl<'de> Deserialize<'de> for TimeSeries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TimeSeriesWire::deserialize(deserializer)?;
        Self::new(
            wire.schema,
            wire.timestamps_ms,
            wire.values,
            wire.observed_mask,
            wire.source,
            wire.policy,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Standardization parameters fitted only from an explicitly supplied
/// training series. Missing values do not contribute to moments.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StandardScaler {
    schema_digest: CanonicalDigest,
    means: Vec<f64>,
    scales: Vec<f64>,
    observed_counts: Vec<u64>,
    fit_series_digest: CanonicalDigest,
}

impl StandardScaler {
    /// Fit population mean and standard deviation over observed training data.
    /// Constant features use scale `1.0`; absent features are rejected.
    pub fn fit_training_partition(series: &TimeSeries) -> Result<Self, ForecastError> {
        let variates = series.variates();
        let mut counts = vec![0_u64; variates];
        let mut means = vec![0.0_f64; variates];
        let mut m2 = vec![0.0_f64; variates];
        for row in 0..series.rows() {
            for feature in 0..variates {
                if let Some(value) = series.value(row, feature) {
                    counts[feature] =
                        counts[feature]
                            .checked_add(1)
                            .ok_or(ForecastError::SizeOverflow {
                                field: "scaler_observed_count",
                            })?;
                    let count = counts[feature] as f64;
                    let value = f64::from(value);
                    let delta = value - means[feature];
                    means[feature] += delta / count;
                    let delta2 = value - means[feature];
                    m2[feature] += delta * delta2;
                }
            }
        }
        let mut scales = Vec::with_capacity(variates);
        for feature in 0..variates {
            if counts[feature] == 0 {
                return Err(ForecastError::NoObservedValue { feature });
            }
            let variance = m2[feature] / counts[feature] as f64;
            let scale = variance.sqrt();
            scales.push(if scale > 0.0 { scale } else { 1.0 });
        }
        Self::from_parts(
            series.schema.canonical_digest(),
            means,
            scales,
            counts,
            series.canonical_digest(),
        )
    }

    /// Ordered feature means.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Ordered positive feature scales.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Observed training values contributing to each feature.
    #[must_use]
    pub fn observed_counts(&self) -> &[u64] {
        &self.observed_counts
    }

    /// Digest of the schema this state applies to.
    #[must_use]
    pub const fn schema_digest(&self) -> CanonicalDigest {
        self.schema_digest
    }

    /// Digest of the exact training series used to fit this state.
    #[must_use]
    pub const fn fit_series_digest(&self) -> CanonicalDigest {
        self.fit_series_digest
    }

    /// Transform a compatible series while preserving its missing-value mask.
    pub fn transform(&self, series: &TimeSeries) -> Result<Vec<f32>, ForecastError> {
        if series.schema.canonical_digest() != self.schema_digest {
            return Err(ForecastError::DigestMismatch {
                field: "scaler_schema_digest",
            });
        }
        let mut transformed = Vec::with_capacity(series.values.len());
        for (index, value) in series.values.iter().enumerate() {
            if !series.observed_mask[index] {
                transformed.push(0.0);
                continue;
            }
            let feature = index % series.variates();
            let normalized =
                ((f64::from(*value) - self.means[feature]) / self.scales[feature]) as f32;
            if !normalized.is_finite() {
                return Err(ForecastError::NonFinite {
                    field: "normalized_values",
                    index,
                });
            }
            transformed.push(if normalized == 0.0 { 0.0 } else { normalized });
        }
        Ok(transformed)
    }

    /// Deterministic digest of the complete fitted state and fit receipt.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"standard-scaler-v1");
        writer.digest(self.schema_digest);
        writer.digest(self.fit_series_digest);
        writer.usize(self.means.len());
        for ((mean, scale), count) in self
            .means
            .iter()
            .zip(&self.scales)
            .zip(&self.observed_counts)
        {
            writer.f64(*mean);
            writer.f64(*scale);
            writer.u64(*count);
        }
        writer.finish()
    }

    fn from_parts(
        schema_digest: CanonicalDigest,
        means: Vec<f64>,
        scales: Vec<f64>,
        observed_counts: Vec<u64>,
        fit_series_digest: CanonicalDigest,
    ) -> Result<Self, ForecastError> {
        if schema_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "scaler_schema_digest",
            });
        }
        if fit_series_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "fit_series_digest",
            });
        }
        if means.is_empty() || means.len() > MAX_FEATURES {
            return Err(ForecastError::LimitExceeded {
                field: "scaler_features",
                actual: means.len(),
                max: MAX_FEATURES,
            });
        }
        check_shape("scaler_scales", means.len(), scales.len())?;
        check_shape("scaler_observed_counts", means.len(), observed_counts.len())?;
        for (index, mean) in means.iter().enumerate() {
            if !mean.is_finite() {
                return Err(ForecastError::NonFinite {
                    field: "scaler_means",
                    index,
                });
            }
        }
        for (index, scale) in scales.iter().enumerate() {
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(ForecastError::NonFinite {
                    field: "scaler_scales",
                    index,
                });
            }
        }
        if let Some(feature) = observed_counts.iter().position(|count| *count == 0) {
            return Err(ForecastError::NoObservedValue { feature });
        }
        Ok(Self {
            schema_digest,
            means,
            scales,
            observed_counts,
            fit_series_digest,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StandardScalerWire {
    schema_digest: CanonicalDigest,
    means: Vec<f64>,
    scales: Vec<f64>,
    observed_counts: Vec<u64>,
    fit_series_digest: CanonicalDigest,
}

impl<'de> Deserialize<'de> for StandardScaler {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StandardScalerWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.schema_digest,
            wire.means,
            wire.scales,
            wire.observed_counts,
            wire.fit_series_digest,
        )
        .map_err(serde::de::Error::custom)
    }
}

pub(crate) fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
    allow_empty: bool,
) -> Result<(), ForecastError> {
    if !allow_empty && value.is_empty() {
        return Err(ForecastError::EmptyField { field });
    }
    if value.len() > max {
        return Err(ForecastError::TextTooLong {
            field,
            actual: value.len(),
            max,
        });
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(ForecastError::InvalidText { field });
    }
    Ok(())
}

pub(crate) fn checked_product(
    field: &'static str,
    dimensions: &[usize],
    max: usize,
) -> Result<usize, ForecastError> {
    let size = dimensions.iter().try_fold(1_usize, |accumulator, value| {
        accumulator.checked_mul(*value)
    });
    let size = size.ok_or(ForecastError::SizeOverflow { field })?;
    if size > max {
        return Err(ForecastError::LimitExceeded {
            field,
            actual: size,
            max,
        });
    }
    Ok(size)
}

pub(crate) fn check_shape(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ForecastError> {
    if actual != expected {
        return Err(ForecastError::ShapeMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn check_finite(field: &'static str, values: &[f32]) -> Result<(), ForecastError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ForecastError::NonFinite { field, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataPolicy, PrivacyClass, SourceState};

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"series-test", value)
    }

    fn policy() -> DataPolicy {
        DataPolicy::new(
            PrivacyClass::P1,
            "tenant",
            "account",
            "workspace",
            "unit test",
            digest(b"policy"),
            None,
            None,
            None,
            10_000,
            true,
        )
        .unwrap()
    }

    fn schema() -> FeatureSchema {
        FeatureSchema::new(vec![
            FeatureSpec::new("rssi_mean", "dBm").unwrap(),
            FeatureSpec::new("motion_power", "ratio").unwrap(),
        ])
        .unwrap()
    }

    fn series() -> TimeSeries {
        TimeSeries::new(
            schema(),
            vec![1_000, 2_000, 3_000],
            vec![-50.0, 1.0, -49.0, 99.0, -48.0, 3.0],
            vec![true, true, true, false, true, true],
            SourceState::synthetic("unit-test-generator").unwrap(),
            policy(),
        )
        .unwrap()
    }

    #[test]
    fn rejects_duplicate_schema_and_bad_shapes() {
        let duplicate = FeatureSchema::new(vec![
            FeatureSpec::new("x", "u").unwrap(),
            FeatureSpec::new("x", "u").unwrap(),
        ]);
        assert!(matches!(
            duplicate,
            Err(ForecastError::DuplicateFeature { .. })
        ));

        let bad = TimeSeries::new(
            schema(),
            vec![1, 2],
            vec![1.0],
            vec![true],
            SourceState::claimed("test").unwrap(),
            policy(),
        );
        assert!(matches!(bad, Err(ForecastError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_nonfinite_and_nonmonotonic_inputs() {
        let nonfinite = TimeSeries::new(
            schema(),
            vec![1, 2],
            vec![1.0, 2.0, f32::NAN, 4.0],
            vec![true; 4],
            SourceState::claimed("test").unwrap(),
            policy(),
        );
        assert!(matches!(nonfinite, Err(ForecastError::NonFinite { .. })));
        let unordered = TimeSeries::new(
            schema(),
            vec![2, 2],
            vec![1.0; 4],
            vec![true; 4],
            SourceState::claimed("test").unwrap(),
            policy(),
        );
        assert!(matches!(
            unordered,
            Err(ForecastError::NonMonotonicTimestamp { .. })
        ));
    }

    #[test]
    fn missing_values_are_masked_and_canonicalized() {
        let value = series();
        assert_eq!(value.value(1, 1), None);
        assert_eq!(value.values()[3], 0.0);
        let mut same = value.clone();
        same.values[3] = -0.0;
        assert_eq!(value.canonical_digest(), same.canonical_digest());
    }

    #[test]
    fn scaler_records_fit_series_and_preserves_mask() {
        let input = series();
        let scaler = StandardScaler::fit_training_partition(&input).unwrap();
        assert_eq!(scaler.observed_counts(), &[3, 2]);
        assert_eq!(scaler.fit_series_digest(), input.canonical_digest());
        let transformed = scaler.transform(&input).unwrap();
        assert_eq!(transformed[3], 0.0);
        assert!(transformed.iter().all(|value| value.is_finite()));

        let encoded = serde_json::to_string(&scaler).unwrap();
        let decoded: StandardScaler = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, scaler);
    }

    #[test]
    fn validated_series_round_trips_through_serde() {
        let value = series();
        let json = serde_json::to_string(&value).unwrap();
        let decoded: TimeSeries = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }
}
