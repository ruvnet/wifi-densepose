//! Benchmark for ground-truth time alignment (ADR-293).
//!
//! Aligns an hour-scale synthetic session (3600 s) against a reference
//! series with a known 12 s clock offset, over the default ±30 s lag
//! window. Variants: 1 Hz estimate vs 1 Hz reference (same-rate), 0.5 Hz
//! estimate vs 1 Hz reference (rate-mismatched, the realistic CSI case),
//! and same-rate with the windowed drift fit enabled. The lag search is
//! O(lags × grid points) with no per-lag allocation; this bench tracks that
//! cost at realistic session length. All input is generated in code and
//! fully deterministic; measurement time is kept short deliberately.
//!
//! Reproduce:
//!   cargo bench -p wifi-densepose-vitals --bench groundtruth_bench
//! Compile-only:
//!   cargo bench -p wifi-densepose-vitals --bench groundtruth_bench --no-run

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use wifi_densepose_vitals::groundtruth::{
    align, AlignmentConfig, EstimateSeries, Measurand, MeasurementPrinciple, ReferenceDevice,
    ReferenceSample, ReferenceSeries,
};

/// Session length in seconds (one hour).
const SESSION_SECS: i64 = 3600;

/// Known clock offset injected into the estimate series, milliseconds.
const OFFSET_MS: i64 = 12_000;

/// Deterministic aperiodic heart-rate-like signal (incommensurate periods).
fn synth(t_secs: f64) -> f64 {
    70.0 + 5.0 * (2.0 * std::f64::consts::PI * t_secs / 47.0).sin()
        + 3.0 * (2.0 * std::f64::consts::PI * t_secs / 113.0).sin()
}

/// Reference series: `SESSION_SECS` samples at 1 Hz on the reference clock.
fn reference_1hz() -> ReferenceSeries {
    ReferenceSeries::new(
        Measurand::HeartRateBpm,
        ReferenceDevice {
            make: "Synthetic".to_string(),
            model: "bench".to_string(),
            principle: MeasurementPrinciple::Other,
        },
        (0..SESSION_SECS)
            .map(|i| ReferenceSample {
                timestamp_ms: i * 1000,
                value: synth(i as f64),
            })
            .collect(),
    )
    .expect("valid reference")
}

/// Estimate series at `period_ms` sampling, shifted `OFFSET_MS` earlier.
fn estimate(period_ms: i64) -> EstimateSeries {
    let n = (SESSION_SECS * 1000) / period_ms;
    EstimateSeries::new(
        Measurand::HeartRateBpm,
        (0..n)
            .map(|i| {
                let t_ms = i * period_ms;
                ReferenceSample {
                    timestamp_ms: t_ms - OFFSET_MS,
                    value: synth(t_ms as f64 / 1000.0),
                }
            })
            .collect(),
    )
    .expect("valid estimate")
}

fn bench_align_hour_session(c: &mut Criterion) {
    let reference = reference_1hz();
    let est_1hz = estimate(1000);
    let est_half_hz = estimate(2000);
    let cfg = AlignmentConfig::default();

    // 1 Hz estimate vs 1 Hz reference: exact offset recovery expected.
    c.bench_function("groundtruth_align_1h_est1hz_ref1hz_pm30s", |b| {
        b.iter(|| {
            let result = align(black_box(&est_1hz), black_box(&reference), black_box(&cfg))
                .expect("alignment succeeds");
            assert_eq!(result.offset_ms, OFFSET_MS);
            black_box(result);
        });
    });

    // 0.5 Hz estimate vs 1 Hz reference: the realistic CSI-pipeline case.
    // Nearest-sample resampling quantizes, so allow one grid step of slack.
    c.bench_function("groundtruth_align_1h_est0p5hz_ref1hz_pm30s", |b| {
        b.iter(|| {
            let result = align(
                black_box(&est_half_hz),
                black_box(&reference),
                black_box(&cfg),
            )
            .expect("alignment succeeds");
            assert!((result.offset_ms - OFFSET_MS).abs() <= cfg.grid_step_ms);
            black_box(result);
        });
    });

    // Same-rate alignment with the windowed linear drift fit enabled.
    let cfg_drift = AlignmentConfig {
        fit_drift: true,
        ..AlignmentConfig::default()
    };
    c.bench_function("groundtruth_align_1h_est1hz_ref1hz_pm30s_drift", |b| {
        b.iter(|| {
            let result = align(
                black_box(&est_1hz),
                black_box(&reference),
                black_box(&cfg_drift),
            )
            .expect("alignment succeeds");
            black_box(result);
        });
    });
}

/// Short measurement window: each iteration is an hour-scale alignment, so
/// default criterion settings would make the suite needlessly slow.
fn short_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = short_config();
    targets = bench_align_hour_session
}
criterion_main!(benches);
