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
}
