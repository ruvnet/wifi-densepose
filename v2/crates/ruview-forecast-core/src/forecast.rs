//! Forecast requests, outputs, quantiles, and backend-neutral inference trait.

use crate::digest::CanonicalWriter;
use crate::series::{
    check_finite, check_shape, checked_product, validate_text, MAX_SERIES_VALUES,
    MAX_SOURCE_REFERENCE_LEN,
};
use crate::{
    ArtifactReceipt, CanonicalDigest, ForecastError, ForecastReceipt, SourceState, TimeSeries,
    MAX_HORIZON,
};
use serde::{Deserialize, Deserializer, Serialize};

/// Maximum quantiles in one forecast distribution.
pub const MAX_QUANTILES: usize = 32;
/// Maximum supported forecast cadence, one day per step.
pub const MAX_STEP_MS: u64 = 86_400_000;
/// Maximum future span of one request, one leap-year.
pub const MAX_FORECAST_SPAN_MS: u64 = 31_622_400_000;

/// Strictly increasing probabilities in the open interval `(0, 1)`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuantileSet {
    values: Vec<f32>,
}

impl QuantileSet {
    /// Validate a non-empty bounded quantile set.
    pub fn new(values: Vec<f32>) -> Result<Self, ForecastError> {
        if values.is_empty() {
            return Err(ForecastError::EmptyField { field: "quantiles" });
        }
        if values.len() > MAX_QUANTILES {
            return Err(ForecastError::LimitExceeded {
                field: "quantiles",
                actual: values.len(),
                max: MAX_QUANTILES,
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 || *value >= 1.0 {
                return Err(ForecastError::InvalidQuantile {
                    index,
                    value: *value,
                });
            }
            if index > 0 && values[index - 1] >= *value {
                return Err(ForecastError::QuantilesNotIncreasing { index });
            }
        }
        Ok(Self { values })
    }

    /// Ordered probabilities.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of requested quantiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this set is empty. A validated set is never empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Exact index of a probability, if requested.
    #[must_use]
    pub fn index_of(&self, probability: f32) -> Option<usize> {
        self.values.iter().position(|value| *value == probability)
    }

    /// Deterministic digest of ordered IEEE-754 probabilities.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"quantile-set-v1");
        writer.usize(self.values.len());
        for value in &self.values {
            writer.f32(*value);
        }
        writer.finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantileSetWire {
    values: Vec<f32>,
}

impl<'de> Deserialize<'de> for QuantileSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = QuantileSetWire::deserialize(deserializer)?;
        Self::new(wire.values).map_err(serde::de::Error::custom)
    }
}

/// Borrowed, validated inference request.
pub struct ForecastRequest<'a> {
    series: &'a TimeSeries,
    horizon: usize,
    step_ms: u64,
    quantiles: &'a QuantileSet,
}

impl<'a> ForecastRequest<'a> {
    /// Construct a request without copying observations or quantiles.
    pub fn new(
        series: &'a TimeSeries,
        horizon: usize,
        step_ms: u64,
        quantiles: &'a QuantileSet,
    ) -> Result<Self, ForecastError> {
        if horizon == 0 {
            return Err(ForecastError::ZeroValue { field: "horizon" });
        }
        if horizon > MAX_HORIZON {
            return Err(ForecastError::LimitExceeded {
                field: "horizon",
                actual: horizon,
                max: MAX_HORIZON,
            });
        }
        if step_ms == 0 {
            return Err(ForecastError::ZeroValue { field: "step_ms" });
        }
        validate_future_span(series.end_timestamp_ms(), horizon, step_ms)?;
        checked_product(
            "forecast_values",
            &[horizon, series.variates(), quantiles.len()],
            MAX_SERIES_VALUES,
        )?;
        Ok(Self {
            series,
            horizon,
            step_ms,
            quantiles,
        })
    }

    /// Input observations.
    #[must_use]
    pub const fn series(&self) -> &'a TimeSeries {
        self.series
    }

    /// Number of future steps.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Forecast cadence in milliseconds.
    #[must_use]
    pub const fn step_ms(&self) -> u64 {
        self.step_ms
    }

    /// Requested distribution probabilities.
    #[must_use]
    pub const fn quantiles(&self) -> &'a QuantileSet {
        self.quantiles
    }

    /// Deterministic digest binding exact input, horizon, cadence, and quantiles.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"forecast-request-v1");
        writer.digest(self.series.canonical_digest());
        writer.usize(self.horizon);
        writer.u64(self.step_ms);
        writer.digest(self.quantiles.canonical_digest());
        writer.finish()
    }
}

/// Backend-neutral forecast implementation.
///
/// Implementations must be deterministic for fixed model state and request,
/// perform no evidence upgrade, and return only values accepted by
/// [`Forecast::issue`].
pub trait Forecaster: Send + Sync {
    /// Stable model family identifier.
    fn model_id(&self) -> &str;

    /// Produce a validated forecast distribution.
    fn forecast(&self, request: &ForecastRequest<'_>) -> Result<ForecastOutcome, ForecastError>;
}

/// Fail-closed reason for returning no numeric forecast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AbstentionReason {
    /// Context is shorter than the artifact contract.
    InsufficientHistory,
    /// The most recent observation is too old for the artifact contract.
    StaleInput,
    /// Too few context cells are observed.
    LowCoverage,
    /// Input is outside the calibrated distribution.
    OutOfDistribution,
    /// Required analogue retrieval is unavailable or isolated.
    RetrievalUnavailable,
    /// Model activation or runtime is unavailable.
    ModelUnavailable,
    /// Model output failed structural or finite-value validation.
    InvalidOutput,
}

/// Valid advisory abstention with the same request/artifact bindings as an
/// ordinary forecast, but no fabricated numeric values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Abstention {
    reason: AbstentionReason,
    detail: String,
    artifact: ArtifactReceipt,
    request_digest: CanonicalDigest,
    input_source: SourceState,
    input_policy_digest: CanonicalDigest,
}

impl Abstention {
    /// Construct an abstention bound to an exact request and artifact.
    pub fn new(
        request: &ForecastRequest<'_>,
        artifact: ArtifactReceipt,
        reason: AbstentionReason,
        detail: impl Into<String>,
    ) -> Result<Self, ForecastError> {
        let detail = detail.into();
        validate_text(
            "abstention_detail",
            &detail,
            MAX_SOURCE_REFERENCE_LEN,
            false,
        )?;
        Ok(Self {
            reason,
            detail,
            artifact,
            request_digest: request.canonical_digest(),
            input_source: request.series.source().clone(),
            input_policy_digest: request.series.policy().canonical_digest(),
        })
    }

    /// Fail-closed reason.
    #[must_use]
    pub const fn reason(&self) -> AbstentionReason {
        self.reason
    }

    /// Bounded operator-facing detail. It must not contain sensitive values.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }

    /// Artifact that abstained.
    #[must_use]
    pub fn artifact(&self) -> &ArtifactReceipt {
        &self.artifact
    }

    /// Verify bindings against a request and trusted artifact receipt.
    pub fn verify_against(
        &self,
        request: &ForecastRequest<'_>,
        verified_artifact: &ArtifactReceipt,
    ) -> Result<(), ForecastError> {
        if self.request_digest != request.canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "abstention_request_digest",
            });
        }
        if self.artifact.canonical_digest() != verified_artifact.canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "abstention_artifact_receipt",
            });
        }
        if self.input_source != *request.series.source() {
            return Err(ForecastError::DigestMismatch {
                field: "abstention_input_source",
            });
        }
        if self.input_policy_digest != request.series.policy().canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "abstention_input_policy",
            });
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AbstentionWire {
    reason: AbstentionReason,
    detail: String,
    artifact: ArtifactReceipt,
    request_digest: CanonicalDigest,
    input_source: SourceState,
    input_policy_digest: CanonicalDigest,
}

impl<'de> Deserialize<'de> for Abstention {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AbstentionWire::deserialize(deserializer)?;
        validate_text(
            "abstention_detail",
            &wire.detail,
            MAX_SOURCE_REFERENCE_LEN,
            false,
        )
        .map_err(serde::de::Error::custom)?;
        if wire.request_digest.is_zero() || wire.input_policy_digest.is_zero() {
            return Err(serde::de::Error::custom("abstention digest is zero"));
        }
        Ok(Self {
            reason: wire.reason,
            detail: wire.detail,
            artifact: wire.artifact,
            request_digest: wire.request_digest,
            input_source: wire.input_source,
            input_policy_digest: wire.input_policy_digest,
        })
    }
}

/// Valid inference result: numeric forecast or explicit abstention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ForecastOutcome {
    /// Structurally valid forecast distribution.
    Forecast(Forecast),
    /// Advisory fail-closed result with no fabricated values.
    Abstained(Abstention),
}

/// Validated multivariate quantile forecast.
///
/// Values use flat layout `step * variates * quantiles + variate * quantiles
/// + quantile`. Every value is finite, and probabilities for each
/// `(step, variate)` are non-crossing.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Forecast {
    origin_ms: u64,
    step_ms: u64,
    horizon: usize,
    variates: usize,
    quantiles: QuantileSet,
    values: Vec<f32>,
    receipt: ForecastReceipt,
}

impl Forecast {
    /// Validate output values and issue an immutable content receipt.
    pub fn issue(
        request: &ForecastRequest<'_>,
        mut values: Vec<f32>,
        artifact: ArtifactReceipt,
        source: SourceState,
    ) -> Result<Self, ForecastError> {
        validate_values(
            request.horizon,
            request.series.variates(),
            request.quantiles,
            &mut values,
        )?;
        let origin_ms = request.series.end_timestamp_ms();
        let output_digest = output_digest(
            origin_ms,
            request.step_ms,
            request.horizon,
            request.series.variates(),
            request.quantiles,
            &values,
            &source,
        );
        let receipt = ForecastReceipt::new(
            artifact,
            request.canonical_digest(),
            output_digest,
            request.series.source().clone(),
            request.series.policy().canonical_digest(),
            source,
        )?;
        Ok(Self {
            origin_ms,
            step_ms: request.step_ms,
            horizon: request.horizon,
            variates: request.series.variates(),
            quantiles: request.quantiles.clone(),
            values,
            receipt,
        })
    }

    /// Last input timestamp, immediately before the first forecast step.
    #[must_use]
    pub const fn origin_ms(&self) -> u64 {
        self.origin_ms
    }

    /// Forecast cadence in milliseconds.
    #[must_use]
    pub const fn step_ms(&self) -> u64 {
        self.step_ms
    }

    /// Number of future steps.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Number of forecast variates.
    #[must_use]
    pub const fn variates(&self) -> usize {
        self.variates
    }

    /// Forecast probabilities.
    #[must_use]
    pub fn quantiles(&self) -> &QuantileSet {
        &self.quantiles
    }

    /// Flat output values in step-major, variate-major, quantile-major order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Immutable artifact/request/output receipt.
    #[must_use]
    pub fn receipt(&self) -> &ForecastReceipt {
        &self.receipt
    }

    /// Read one forecast value, returning `None` for an out-of-range index.
    #[must_use]
    pub fn value(&self, step: usize, variate: usize, quantile: usize) -> Option<f32> {
        if step >= self.horizon || variate >= self.variates || quantile >= self.quantiles.len() {
            return None;
        }
        let row = step.checked_mul(self.variates)?;
        let slot = row.checked_add(variate)?;
        let base = slot.checked_mul(self.quantiles.len())?;
        self.values.get(base.checked_add(quantile)?).copied()
    }

    /// Recompute only the canonical payload-integrity digest.
    ///
    /// This does not authenticate the artifact or request. Use
    /// [`Self::verify_against`] with an artifact already verified by the
    /// activation boundary for a complete binding check.
    pub fn verify_payload_integrity(&self) -> Result<(), ForecastError> {
        let actual = output_digest(
            self.origin_ms,
            self.step_ms,
            self.horizon,
            self.variates,
            &self.quantiles,
            &self.values,
            self.receipt.source(),
        );
        if actual != self.receipt.output_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_output_digest",
            });
        }
        Ok(())
    }

    /// Verify content bindings against an exact request and trusted artifact.
    pub fn verify_against(
        &self,
        request: &ForecastRequest<'_>,
        verified_artifact: &ArtifactReceipt,
    ) -> Result<(), ForecastError> {
        self.verify_payload_integrity()?;
        if self.receipt.request_digest() != request.canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_request_digest",
            });
        }
        if self.receipt.artifact().canonical_digest() != verified_artifact.canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_artifact_receipt",
            });
        }
        if self.receipt.input_source() != request.series.source() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_input_source",
            });
        }
        if self.receipt.input_policy_digest() != request.series.policy().canonical_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_input_policy",
            });
        }
        Ok(())
    }

    /// Deterministic digest of output plus its complete receipt.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"forecast-v1");
        writer.digest(self.receipt.output_digest());
        writer.digest(self.receipt.canonical_digest());
        writer.finish()
    }

    fn from_parts(
        origin_ms: u64,
        step_ms: u64,
        horizon: usize,
        variates: usize,
        quantiles: QuantileSet,
        mut values: Vec<f32>,
        receipt: ForecastReceipt,
    ) -> Result<Self, ForecastError> {
        if step_ms == 0 {
            return Err(ForecastError::ZeroValue { field: "step_ms" });
        }
        if horizon == 0 || horizon > MAX_HORIZON {
            return Err(ForecastError::LimitExceeded {
                field: "horizon",
                actual: horizon,
                max: MAX_HORIZON,
            });
        }
        if variates == 0 {
            return Err(ForecastError::ZeroValue { field: "variates" });
        }
        validate_future_span(origin_ms, horizon, step_ms)?;
        validate_values(horizon, variates, &quantiles, &mut values)?;
        let actual = output_digest(
            origin_ms,
            step_ms,
            horizon,
            variates,
            &quantiles,
            &values,
            receipt.source(),
        );
        if actual != receipt.output_digest() {
            return Err(ForecastError::DigestMismatch {
                field: "forecast_output_digest",
            });
        }
        Ok(Self {
            origin_ms,
            step_ms,
            horizon,
            variates,
            quantiles,
            values,
            receipt,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastWire {
    origin_ms: u64,
    step_ms: u64,
    horizon: usize,
    variates: usize,
    quantiles: QuantileSet,
    values: Vec<f32>,
    receipt: ForecastReceipt,
}

impl<'de> Deserialize<'de> for Forecast {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ForecastWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.origin_ms,
            wire.step_ms,
            wire.horizon,
            wire.variates,
            wire.quantiles,
            wire.values,
            wire.receipt,
        )
        .map_err(serde::de::Error::custom)
    }
}

fn validate_values(
    horizon: usize,
    variates: usize,
    quantiles: &QuantileSet,
    values: &mut [f32],
) -> Result<(), ForecastError> {
    let expected = checked_product(
        "forecast_values",
        &[horizon, variates, quantiles.len()],
        MAX_SERIES_VALUES,
    )?;
    check_shape("forecast_values", expected, values.len())?;
    check_finite("forecast_values", values)?;
    for value in values.iter_mut() {
        if *value == 0.0 {
            *value = 0.0;
        }
    }
    for (chunk_index, chunk) in values.chunks_exact(quantiles.len()).enumerate() {
        for index in 1..chunk.len() {
            if chunk[index - 1] > chunk[index] {
                return Err(ForecastError::QuantileCrossing {
                    index: chunk_index * quantiles.len() + index,
                });
            }
        }
    }
    Ok(())
}

fn output_digest(
    origin_ms: u64,
    step_ms: u64,
    horizon: usize,
    variates: usize,
    quantiles: &QuantileSet,
    values: &[f32],
    source: &SourceState,
) -> CanonicalDigest {
    let mut writer = CanonicalWriter::new(b"forecast-output-v1");
    writer.u64(origin_ms);
    writer.u64(step_ms);
    writer.usize(horizon);
    writer.usize(variates);
    writer.digest(quantiles.canonical_digest());
    writer.usize(values.len());
    for value in values {
        writer.f32(*value);
    }
    source.write_canonical(&mut writer);
    writer.finish()
}

fn validate_future_span(origin_ms: u64, horizon: usize, step_ms: u64) -> Result<(), ForecastError> {
    if step_ms > MAX_STEP_MS {
        return Err(ForecastError::DurationLimitExceeded {
            field: "step_ms",
            actual_ms: step_ms,
            max_ms: MAX_STEP_MS,
        });
    }
    let horizon_u64 = u64::try_from(horizon).map_err(|_| ForecastError::SizeOverflow {
        field: "forecast_span_ms",
    })?;
    let span = step_ms
        .checked_mul(horizon_u64)
        .ok_or(ForecastError::SizeOverflow {
            field: "forecast_span_ms",
        })?;
    if span > MAX_FORECAST_SPAN_MS {
        return Err(ForecastError::DurationLimitExceeded {
            field: "forecast_span_ms",
            actual_ms: span,
            max_ms: MAX_FORECAST_SPAN_MS,
        });
    }
    origin_ms
        .checked_add(span)
        .ok_or(ForecastError::SizeOverflow {
            field: "forecast_end_timestamp_ms",
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataPolicy, FeatureSchema, FeatureSpec, PrivacyClass};

    fn digest(label: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"forecast-test", label)
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

    fn request_data() -> (TimeSeries, QuantileSet) {
        let schema = FeatureSchema::new(vec![FeatureSpec::new("x", "ratio").unwrap()]).unwrap();
        let series = TimeSeries::new(
            schema,
            vec![1, 2, 3],
            vec![1.0, 2.0, 3.0],
            vec![true; 3],
            SourceState::synthetic("fixture").unwrap(),
            policy(),
        )
        .unwrap();
        let quantiles = QuantileSet::new(vec![0.1, 0.5, 0.9]).unwrap();
        (series, quantiles)
    }

    fn artifact(source: SourceState) -> ArtifactReceipt {
        ArtifactReceipt::new(
            "test-model",
            "1",
            digest(b"artifact"),
            digest(b"config"),
            policy().canonical_digest(),
            source,
        )
        .unwrap()
    }

    #[test]
    fn quantiles_reject_invalid_and_unsorted_values() {
        assert!(matches!(
            QuantileSet::new(vec![0.0]),
            Err(ForecastError::InvalidQuantile { .. })
        ));
        assert!(matches!(
            QuantileSet::new(vec![0.5, 0.5]),
            Err(ForecastError::QuantilesNotIncreasing { .. })
        ));
    }

    #[test]
    fn request_rejects_timestamp_and_duration_overflow() {
        let schema = FeatureSchema::new(vec![FeatureSpec::new("x", "ratio").unwrap()]).unwrap();
        let series = TimeSeries::new(
            schema,
            vec![u64::MAX - 1],
            vec![1.0],
            vec![true],
            SourceState::synthetic("fixture").unwrap(),
            DataPolicy::new(
                PrivacyClass::P1,
                "tenant",
                "account",
                "workspace",
                "unit test",
                digest(b"overflow-policy"),
                None,
                None,
                None,
                u64::MAX,
                true,
            )
            .unwrap(),
        )
        .unwrap();
        let quantiles = QuantileSet::new(vec![0.5]).unwrap();
        assert!(matches!(
            ForecastRequest::new(&series, 2, 1, &quantiles),
            Err(ForecastError::SizeOverflow { .. })
        ));
    }

    #[test]
    fn issue_binds_request_and_rejects_crossing() {
        let (series, quantiles) = request_data();
        let request = ForecastRequest::new(&series, 1, 1_000, &quantiles).unwrap();
        let source = SourceState::synthetic("forecast").unwrap();
        let invalid = Forecast::issue(
            &request,
            vec![3.0, 2.0, 4.0],
            artifact(SourceState::claimed("artifact").unwrap()),
            source,
        );
        assert!(matches!(
            invalid,
            Err(ForecastError::QuantileCrossing { .. })
        ));
    }

    #[test]
    fn output_cannot_upgrade_synthetic_input() {
        let (series, quantiles) = request_data();
        let request = ForecastRequest::new(&series, 1, 1_000, &quantiles).unwrap();
        let invalid = Forecast::issue(
            &request,
            vec![2.0, 2.0, 2.0],
            artifact(SourceState::claimed("artifact").unwrap()),
            SourceState::claimed("improper upgrade").unwrap(),
        );
        assert!(matches!(
            invalid,
            Err(ForecastError::EvidenceEscalation { .. })
        ));
    }

    #[test]
    fn issued_forecast_round_trips_and_verifies() {
        let (series, quantiles) = request_data();
        let request = ForecastRequest::new(&series, 2, 1_000, &quantiles).unwrap();
        let forecast = Forecast::issue(
            &request,
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            artifact(SourceState::claimed("artifact").unwrap()),
            SourceState::synthetic("derived fixture").unwrap(),
        )
        .unwrap();
        forecast.verify_payload_integrity().unwrap();
        let json = serde_json::to_string(&forecast).unwrap();
        let decoded: Forecast = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, forecast);
        assert_eq!(decoded.value(1, 0, 2), Some(4.0));
    }
}
