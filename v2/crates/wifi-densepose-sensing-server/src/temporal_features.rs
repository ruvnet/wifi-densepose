//! Causal, time-axis CSI features for the lightweight presence path.

use std::collections::VecDeque;

#[derive(Debug, Default, PartialEq)]
pub(crate) struct TemporalFrequencyFeatures {
    /// Frame-to-frame normalized RMS change, reported as a percentage.
    pub(crate) motion_band_power: f64,
    /// Fraction of temporal spectral power in the physiological 0.1–0.5 Hz band.
    pub(crate) breathing_band_power: f64,
    pub(crate) dominant_freq_hz: f64,
    pub(crate) breathing_rate_hz: f64,
    pub(crate) change_points: usize,
}

/// Extract features on the time axis. CSI subcarrier indices are spatial/frequency
/// bins, not elapsed time, so treating an index as Hz creates physically meaningless
/// signals. The bounded, decimated periodogram keeps this path cheap enough for ESP32
/// streaming while retaining the 0.1–5 Hz band used by CSI presence/vital pipelines.
pub(crate) fn extract_temporal_frequency_features(
    frame_history: &VecDeque<Vec<f64>>,
    current_amplitudes: &[f64],
    sample_rate_hz: f64,
) -> TemporalFrequencyFeatures {
    let sample_rate_hz = sample_rate_hz.clamp(1.0, 200.0);
    let previous = frame_history.back();
    let n_cmp = previous
        .map(|frame| frame.len().min(current_amplitudes.len()))
        .unwrap_or(0);
    let current_energy = if current_amplitudes.is_empty() {
        0.0
    } else {
        current_amplitudes
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / current_amplitudes.len() as f64
    };
    let (motion_ratio, change_points) = previous.map_or((0.0, 0), |previous| {
        if n_cmp == 0 {
            return (0.0, 0);
        }
        let diff_energy = (0..n_cmp)
            .map(|index| (current_amplitudes[index] - previous[index]).powi(2))
            .sum::<f64>()
            / n_cmp as f64;
        let changed = (0..n_cmp)
            .filter(|&index| {
                let reference = previous[index]
                    .abs()
                    .max(current_amplitudes[index].abs())
                    .max(1e-6);
                (current_amplitudes[index] - previous[index]).abs() / reference >= 0.03
            })
            .count();
        (
            (diff_energy / (current_energy + 1e-9))
                .sqrt()
                .clamp(0.0, 1.0),
            changed,
        )
    });

    let max_source_samples = (sample_rate_hz * 30.0).ceil() as usize;
    let start = frame_history.len().saturating_sub(max_source_samples);
    let mut source: Vec<f64> = frame_history
        .iter()
        .skip(start)
        .map(|amplitudes| mean_amplitude(amplitudes))
        .collect();
    source.push(mean_amplitude(current_amplitudes));

    // Frequencies above 5 Hz are irrelevant to this feature contract. Downsampling
    // makes periodogram cost nearly independent of high CSI packet rates.
    let decimation = (sample_rate_hz / 10.0).ceil().max(1.0) as usize;
    if decimation > 1 {
        source = source
            .chunks(decimation)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect();
    }
    let effective_rate_hz = sample_rate_hz / decimation as f64;
    let minimum_samples = (effective_rate_hz * 8.0).ceil() as usize;
    if source.len() < minimum_samples.max(16) {
        return motion_only(motion_ratio, change_points);
    }

    let sample_count = source.len();
    let mean = source.iter().sum::<f64>() / sample_count as f64;
    let detrended: Vec<f64> = source
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let hann = 0.5
                - 0.5
                    * (std::f64::consts::TAU * index as f64
                        / sample_count.saturating_sub(1) as f64)
                        .cos();
            (value - mean) * hann
        })
        .collect();
    if detrended.iter().map(|value| value * value).sum::<f64>() <= 1e-9 {
        return motion_only(motion_ratio, change_points);
    }

    let first_bin = ((0.1 * sample_count as f64 / effective_rate_hz).ceil() as usize).max(1);
    let max_frequency_hz = 5.0_f64.min(effective_rate_hz * 0.45);
    let last_bin = (max_frequency_hz * sample_count as f64 / effective_rate_hz).floor() as usize;
    let mut total_power = 0.0;
    let mut breathing_power = 0.0;
    let mut breathing_bins = 0usize;
    let mut best_power = 0.0;
    let mut best_frequency_hz = 0.0;
    let mut best_breathing_power = 0.0;
    let mut best_breathing_hz = 0.0;

    for bin in first_bin..=last_bin.max(first_bin) {
        let omega = std::f64::consts::TAU * bin as f64 / sample_count as f64;
        let (real, imaginary) =
            detrended
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(real, imaginary), (index, value)| {
                    let angle = omega * index as f64;
                    (real + value * angle.cos(), imaginary - value * angle.sin())
                });
        let power = real * real + imaginary * imaginary;
        let frequency_hz = bin as f64 * effective_rate_hz / sample_count as f64;
        total_power += power;
        if power > best_power {
            best_power = power;
            best_frequency_hz = frequency_hz;
        }
        if (0.1..=0.5).contains(&frequency_hz) {
            breathing_power += power;
            breathing_bins += 1;
            if power > best_breathing_power {
                best_breathing_power = power;
                best_breathing_hz = frequency_hz;
            }
        }
    }

    let breathing_band_power = if total_power > 0.0 {
        (breathing_power / total_power).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let average_breathing_power = if breathing_bins > 0 {
        breathing_power / breathing_bins as f64
    } else {
        0.0
    };
    let breathing_rate_hz =
        if breathing_band_power >= 0.25 && best_breathing_power >= average_breathing_power * 1.5 {
            best_breathing_hz
        } else {
            0.0
        };

    TemporalFrequencyFeatures {
        motion_band_power: motion_ratio * 100.0,
        breathing_band_power,
        dominant_freq_hz: best_frequency_hz,
        breathing_rate_hz,
        change_points,
    }
}

fn mean_amplitude(amplitudes: &[f64]) -> f64 {
    if amplitudes.is_empty() {
        0.0
    } else {
        amplitudes.iter().sum::<f64>() / amplitudes.len() as f64
    }
}

fn motion_only(motion_ratio: f64, change_points: usize) -> TemporalFrequencyFeatures {
    TemporalFrequencyFeatures {
        motion_band_power: motion_ratio * 100.0,
        change_points,
        ..TemporalFrequencyFeatures::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_current_frame_with_previous_frame() {
        let history = VecDeque::from([vec![10.0; 4]]);
        let temporal = extract_temporal_frequency_features(&history, &[20.0; 4], 20.0);

        assert!(temporal.motion_band_power > 20.0);
        assert_eq!(temporal.change_points, 4);
    }

    #[test]
    fn reports_temporal_hertz_instead_of_subcarrier_indices() {
        let sample_rate_hz = 20.0;
        let breathing_hz = 0.25;
        let history = (0..400)
            .map(|index| {
                let time = index as f64 / sample_rate_hz;
                let amplitude = 20.0 + (std::f64::consts::TAU * breathing_hz * time).sin();
                vec![amplitude; 8]
            })
            .collect();

        let temporal = extract_temporal_frequency_features(&history, &[20.0; 8], sample_rate_hz);

        assert!((temporal.dominant_freq_hz - breathing_hz).abs() <= 0.05);
        assert!(temporal.breathing_band_power > 0.50);
        assert!((temporal.breathing_rate_hz - breathing_hz).abs() <= 0.05);
    }

    #[test]
    fn rejects_frequency_claims_until_enough_time_has_elapsed() {
        let history = std::iter::repeat_with(|| vec![20.0; 8]).take(10).collect();
        let temporal = extract_temporal_frequency_features(&history, &[20.0; 8], 20.0);

        assert_eq!(temporal.dominant_freq_hz, 0.0);
        assert_eq!(temporal.breathing_rate_hz, 0.0);
    }
}
