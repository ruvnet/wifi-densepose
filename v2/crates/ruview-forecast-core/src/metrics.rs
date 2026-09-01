//! Deterministic, allocation-free forecast metrics.

use crate::series::{check_finite, check_shape};
use crate::{Forecast, ForecastError};

/// Mean absolute error over observed targets.
pub fn mae(actual: &[f32], predicted: &[f32], observed: &[bool]) -> Result<f64, ForecastError> {
    validate_metric_arrays(actual, predicted, observed)?;
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for ((actual, predicted), observed) in actual.iter().zip(predicted).zip(observed) {
        if *observed {
            total += (f64::from(*actual) - f64::from(*predicted)).abs();
            count += 1;
        }
    }
    divide_observed(total, count)
}

/// Mean pinball loss for one quantile over observed targets.
pub fn pinball_loss(
    actual: &[f32],
    predicted: &[f32],
    observed: &[bool],
    quantile: f32,
) -> Result<f64, ForecastError> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
        return Err(ForecastError::InvalidQuantile {
            index: 0,
            value: quantile,
        });
    }
    validate_metric_arrays(actual, predicted, observed)?;
    let q = f64::from(quantile);
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for ((actual, predicted), observed) in actual.iter().zip(predicted).zip(observed) {
        if *observed {
            let residual = f64::from(*actual) - f64::from(*predicted);
            total += if residual >= 0.0 {
                q * residual
            } else {
                (q - 1.0) * residual
            };
            count += 1;
        }
    }
    divide_observed(total, count)
}

/// Weighted quantile loss over every distribution value in a forecast.
///
/// Uses `2 * sum(pinball) / (quantiles * sum(abs(actual)))`; all-zero observed
/// targets are rejected because that normalized score is undefined.
pub fn weighted_quantile_loss(
    actual: &[f32],
    observed: &[bool],
    forecast: &Forecast,
) -> Result<f64, ForecastError> {
    let expected =
        forecast
            .horizon()
            .checked_mul(forecast.variates())
            .ok_or(ForecastError::SizeOverflow {
                field: "metric_targets",
            })?;
    check_shape("metric_actual", expected, actual.len())?;
    check_shape("metric_observed", expected, observed.len())?;
    check_finite("metric_actual", actual)?;
    let mut loss = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut observed_count = 0_u64;
    for index in 0..expected {
        if !observed[index] {
            continue;
        }
        observed_count += 1;
        scale += f64::from(actual[index]).abs();
        let step = index / forecast.variates();
        let variate = index % forecast.variates();
        for (quantile_index, quantile) in forecast.quantiles().values().iter().enumerate() {
            let prediction = forecast
                .value(step, variate, quantile_index)
                .expect("validated forecast shape");
            let residual = f64::from(actual[index]) - f64::from(prediction);
            let q = f64::from(*quantile);
            loss += if residual >= 0.0 {
                q * residual
            } else {
                (q - 1.0) * residual
            };
        }
    }
    if observed_count == 0 {
        return Err(ForecastError::NoObservedTargets);
    }
    if scale == 0.0 {
        return Err(ForecastError::MetricUndefined {
            reason: "weighted quantile loss requires nonzero absolute targets",
        });
    }
    Ok(2.0 * loss / (forecast.quantiles().len() as f64 * scale))
}

/// [`weighted_quantile_loss`] broken out per horizon step (index `0..horizon`)
/// instead of collapsed into one aggregate number. Uses the identical
/// per-cell pinball formula and the same domain checks; only the reduction
/// changes, so summing this function's outputs' contributions reproduces
/// [`weighted_quantile_loss`]'s aggregate exactly. Useful for spotting
/// whether error grows with lead time, which a single aggregate number
/// hides.
pub fn weighted_quantile_loss_by_horizon(
    actual: &[f32],
    observed: &[bool],
    forecast: &Forecast,
) -> Result<Vec<f64>, ForecastError> {
    let horizon = forecast.horizon();
    let variates = forecast.variates();
    let expected = horizon
        .checked_mul(variates)
        .ok_or(ForecastError::SizeOverflow {
            field: "metric_targets",
        })?;
    check_shape("metric_actual", expected, actual.len())?;
    check_shape("metric_observed", expected, observed.len())?;
    check_finite("metric_actual", actual)?;
    let mut loss = vec![0.0_f64; horizon];
    let mut scale = vec![0.0_f64; horizon];
    let mut observed_count = vec![0_u64; horizon];
    for index in 0..expected {
        if !observed[index] {
            continue;
        }
        let step = index / variates;
        let variate = index % variates;
        observed_count[step] += 1;
        scale[step] += f64::from(actual[index]).abs();
        for (quantile_index, quantile) in forecast.quantiles().values().iter().enumerate() {
            let prediction = forecast
                .value(step, variate, quantile_index)
                .expect("validated forecast shape");
            let residual = f64::from(actual[index]) - f64::from(prediction);
            let q = f64::from(*quantile);
            loss[step] += if residual >= 0.0 {
                q * residual
            } else {
                (q - 1.0) * residual
            };
        }
    }
    let quantile_count = forecast.quantiles().len() as f64;
    let mut result = Vec::with_capacity(horizon);
    for step in 0..horizon {
        if observed_count[step] == 0 {
            return Err(ForecastError::NoObservedTargets);
        }
        if scale[step] == 0.0 {
            return Err(ForecastError::MetricUndefined {
                reason: "weighted quantile loss requires nonzero absolute targets",
            });
        }
        result.push(2.0 * loss[step] / (quantile_count * scale[step]));
    }
    Ok(result)
}

/// Fraction of observed targets inside inclusive lower/upper intervals.
pub fn interval_coverage(
    actual: &[f32],
    lower: &[f32],
    upper: &[f32],
    observed: &[bool],
) -> Result<f64, ForecastError> {
    check_shape("metric_lower", actual.len(), lower.len())?;
    check_shape("metric_upper", actual.len(), upper.len())?;
    check_shape("metric_observed", actual.len(), observed.len())?;
    check_finite("metric_actual", actual)?;
    check_finite("metric_lower", lower)?;
    check_finite("metric_upper", upper)?;
    let mut covered = 0_u64;
    let mut count = 0_u64;
    for index in 0..actual.len() {
        if lower[index] > upper[index] {
            return Err(ForecastError::QuantileCrossing { index });
        }
        if observed[index] {
            count += 1;
            covered += u64::from(actual[index] >= lower[index] && actual[index] <= upper[index]);
        }
    }
    divide_observed(covered as f64, count)
}

fn validate_metric_arrays(
    actual: &[f32],
    predicted: &[f32],
    observed: &[bool],
) -> Result<(), ForecastError> {
    check_shape("metric_predicted", actual.len(), predicted.len())?;
    check_shape("metric_observed", actual.len(), observed.len())?;
    check_finite("metric_actual", actual)?;
    check_finite("metric_predicted", predicted)?;
    Ok(())
}

fn divide_observed(total: f64, count: u64) -> Result<f64, ForecastError> {
    if count == 0 {
        return Err(ForecastError::NoObservedTargets);
    }
    Ok(total / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_mae_and_pinball_are_deterministic() {
        let actual = [1.0, 10.0, 3.0];
        let predicted = [2.0, -99.0, 1.0];
        let mask = [true, false, true];
        assert_eq!(mae(&actual, &predicted, &mask).unwrap(), 1.5);
        assert_eq!(pinball_loss(&actual, &predicted, &mask, 0.5).unwrap(), 0.75);
    }

    #[test]
    fn coverage_rejects_crossed_intervals_and_empty_masks() {
        assert!(matches!(
            interval_coverage(&[1.0], &[2.0], &[0.0], &[true]),
            Err(ForecastError::QuantileCrossing { .. })
        ));
        assert!(matches!(
            interval_coverage(&[1.0], &[0.0], &[2.0], &[false]),
            Err(ForecastError::NoObservedTargets)
        ));
    }

    #[test]
    fn per_horizon_wql_sums_to_the_aggregate() {
        use crate::{
            ArtifactReceipt, CanonicalDigest, DataPolicy, FeatureSchema, FeatureSpec,
            ForecastRequest, PrivacyClass, QuantileSet, SourceState, TimeSeries,
        };

        let schema = FeatureSchema::new(vec![
            FeatureSpec::new("a", "ratio").unwrap(),
            FeatureSpec::new("b", "ratio").unwrap(),
        ])
        .unwrap();
        let policy = DataPolicy::new(
            PrivacyClass::P1,
            "tenant",
            "account",
            "workspace",
            "unit test",
            CanonicalDigest::of_bytes(b"metrics-test", b"policy"),
            None,
            None,
            None,
            10_000,
            true,
        )
        .unwrap();
        let series = TimeSeries::new(
            schema.clone(),
            vec![1, 2, 3, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![true; 8],
            SourceState::synthetic("fixture").unwrap(),
            policy.clone(),
        )
        .unwrap();
        let quantiles = QuantileSet::new(vec![0.1, 0.5, 0.9]).unwrap();
        let horizon = 3;
        let variates = 2;
        let request = ForecastRequest::new(&series, horizon, 1, &quantiles).unwrap();

        // Deterministic, hand-built forecast: a constant prediction offset
        // from a synthetic actual series so every cell has a nonzero,
        // distinguishable pinball contribution.
        let artifact = ArtifactReceipt::new(
            "fixture-model",
            "1",
            CanonicalDigest::of_bytes(b"metrics-test", b"model"),
            CanonicalDigest::of_bytes(b"metrics-test", b"config"),
            policy.canonical_digest(),
            SourceState::claimed("fixture").unwrap(),
        )
        .unwrap();
        let source = SourceState::derived_forecast("fixture", series.source(), artifact.source())
            .unwrap();
        let mut values = Vec::with_capacity(horizon * variates * quantiles.len());
        for step in 0..horizon {
            for variate in 0..variates {
                let base = (step * variates + variate) as f32;
                for quantile_index in 0..quantiles.len() {
                    values.push(base + quantile_index as f32 * 0.1);
                }
            }
        }
        let forecast = Forecast::issue(&request, values, artifact, source).unwrap();

        let actual: Vec<f32> = (0..horizon * variates).map(|i| i as f32 + 0.5).collect();
        let observed = vec![true; horizon * variates];

        let aggregate = weighted_quantile_loss(&actual, &observed, &forecast).unwrap();
        let by_horizon = weighted_quantile_loss_by_horizon(&actual, &observed, &forecast).unwrap();
        assert_eq!(by_horizon.len(), horizon);

        // Reconstruct the aggregate from the per-step numerators/denominators
        // (not just averaging the per-step ratios, which would be a
        // different, incorrect reduction) to prove the two functions agree.
        let quantile_count = quantiles.len() as f64;
        let mut total_loss = 0.0_f64;
        let mut total_scale = 0.0_f64;
        for step in 0..horizon {
            let step_scale: f64 = (0..variates)
                .map(|variate| f64::from(actual[step * variates + variate]).abs())
                .sum();
            total_scale += step_scale;
            total_loss += by_horizon[step] * quantile_count * step_scale / 2.0;
        }
        let reconstructed = 2.0 * total_loss / (quantile_count * total_scale);
        assert!(
            (reconstructed - aggregate).abs() < 1e-9,
            "reconstructed {reconstructed} vs aggregate {aggregate}"
        );
    }
}
