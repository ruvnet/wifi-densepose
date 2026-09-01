//! Deterministic baseline forecasters required for honest evaluation.

use crate::{
    Abstention, AbstentionReason, ArtifactReceipt, CanonicalDigest, Forecast, ForecastError,
    ForecastOutcome, ForecastRequest, Forecaster, SourceState,
};

const LAST_VALUE_ID: &str = "ruview-last-value-baseline";
const SEASONAL_ID: &str = "ruview-seasonal-naive-baseline";

/// Last-observed-value baseline repeated across every future step/quantile.
#[derive(Clone, Copy, Debug, Default)]
pub struct LastValueForecaster;

impl LastValueForecaster {
    /// Construct the stateless baseline.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Forecaster for LastValueForecaster {
    fn model_id(&self) -> &str {
        LAST_VALUE_ID
    }

    fn forecast(&self, request: &ForecastRequest<'_>) -> Result<ForecastOutcome, ForecastError> {
        let artifact = baseline_artifact(LAST_VALUE_ID, b"last-value-v1", request)?;
        let mut last = Vec::with_capacity(request.series().variates());
        for feature in 0..request.series().variates() {
            let Some(value) = request.series().last_observed(feature) else {
                return Ok(ForecastOutcome::Abstained(Abstention::new(
                    request,
                    artifact,
                    AbstentionReason::LowCoverage,
                    "a variate has no observed context value",
                )?));
            };
            last.push(value);
        }
        let mut values = Vec::with_capacity(
            request.horizon() * request.series().variates() * request.quantiles().len(),
        );
        for _ in 0..request.horizon() {
            for value in &last {
                values.extend(std::iter::repeat_n(*value, request.quantiles().len()));
            }
        }
        issue_baseline(request, artifact, values, "last-value derived forecast")
    }
}

/// Seasonal-naive baseline using a fixed row period.
#[derive(Clone, Copy, Debug)]
pub struct SeasonalNaiveForecaster {
    period: usize,
}

impl SeasonalNaiveForecaster {
    /// Construct with a nonzero period no larger than the context ceiling.
    pub fn new(period: usize) -> Result<Self, ForecastError> {
        if period == 0 {
            return Err(ForecastError::ZeroValue {
                field: "seasonal_period",
            });
        }
        if period > crate::MAX_CONTEXT_LENGTH {
            return Err(ForecastError::LimitExceeded {
                field: "seasonal_period",
                actual: period,
                max: crate::MAX_CONTEXT_LENGTH,
            });
        }
        Ok(Self { period })
    }

    /// Seasonal row period.
    #[must_use]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl Forecaster for SeasonalNaiveForecaster {
    fn model_id(&self) -> &str {
        SEASONAL_ID
    }

    fn forecast(&self, request: &ForecastRequest<'_>) -> Result<ForecastOutcome, ForecastError> {
        let mut config = Vec::from(b"seasonal-naive-v1".as_slice());
        config.extend_from_slice(&(self.period as u64).to_be_bytes());
        let artifact = baseline_artifact(SEASONAL_ID, &config, request)?;
        if request.series().rows() < self.period {
            return Ok(ForecastOutcome::Abstained(Abstention::new(
                request,
                artifact,
                AbstentionReason::InsufficientHistory,
                "context is shorter than the seasonal period",
            )?));
        }
        let base = request.series().rows() - self.period;
        let mut values = Vec::with_capacity(
            request.horizon() * request.series().variates() * request.quantiles().len(),
        );
        for step in 0..request.horizon() {
            let row = base + (step % self.period);
            for feature in 0..request.series().variates() {
                let Some(value) = request.series().value(row, feature) else {
                    return Ok(ForecastOutcome::Abstained(Abstention::new(
                        request,
                        artifact,
                        AbstentionReason::LowCoverage,
                        "required seasonal context slot is missing",
                    )?));
                };
                values.extend(std::iter::repeat_n(value, request.quantiles().len()));
            }
        }
        issue_baseline(request, artifact, values, "seasonal-naive derived forecast")
    }
}

fn baseline_artifact(
    model_id: &str,
    config: &[u8],
    request: &ForecastRequest<'_>,
) -> Result<ArtifactReceipt, ForecastError> {
    ArtifactReceipt::new(
        model_id,
        "1",
        CanonicalDigest::of_bytes(b"built-in-baseline-artifact", model_id.as_bytes()),
        CanonicalDigest::of_bytes(b"built-in-baseline-config", config),
        request.series().policy().canonical_digest(),
        SourceState::claimed("built-in deterministic baseline")?,
    )
}

fn issue_baseline(
    request: &ForecastRequest<'_>,
    artifact: ArtifactReceipt,
    values: Vec<f32>,
    reference: &str,
) -> Result<ForecastOutcome, ForecastError> {
    let source =
        SourceState::derived_forecast(reference, request.series().source(), artifact.source())?;
    Ok(ForecastOutcome::Forecast(Forecast::issue(
        request, values, artifact, source,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataPolicy, FeatureSchema, FeatureSpec, PrivacyClass, QuantileSet, TimeSeries};

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"baseline-test", value)
    }

    fn series(mask: Vec<bool>) -> TimeSeries {
        let schema = FeatureSchema::new(vec![FeatureSpec::new("x", "ratio").unwrap()]).unwrap();
        let policy = DataPolicy::new(
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
        .unwrap();
        TimeSeries::new(
            schema,
            vec![1, 2, 3, 4],
            vec![1.0, 2.0, 3.0, 4.0],
            mask,
            SourceState::synthetic("fixture").unwrap(),
            policy,
        )
        .unwrap()
    }

    #[test]
    fn last_value_is_deterministic() {
        let series = series(vec![true; 4]);
        let quantiles = QuantileSet::new(vec![0.1, 0.5, 0.9]).unwrap();
        let request = ForecastRequest::new(&series, 2, 1, &quantiles).unwrap();
        let first = LastValueForecaster::new().forecast(&request).unwrap();
        let second = LastValueForecaster::new().forecast(&request).unwrap();
        assert_eq!(first, second);
        let ForecastOutcome::Forecast(value) = first else {
            panic!("expected forecast");
        };
        assert_eq!(value.values(), &[4.0; 6]);
    }

    #[test]
    fn seasonal_missing_slot_abstains() {
        let series = series(vec![true, true, false, true]);
        let quantiles = QuantileSet::new(vec![0.5]).unwrap();
        let request = ForecastRequest::new(&series, 2, 1, &quantiles).unwrap();
        let result = SeasonalNaiveForecaster::new(2)
            .unwrap()
            .forecast(&request)
            .unwrap();
        assert!(matches!(result, ForecastOutcome::Abstained(_)));
    }
}
