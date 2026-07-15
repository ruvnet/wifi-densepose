//! Benchmarks for the vital-sign extractor hot paths (ADR-157 §D1).
//!
//! The extractors maintain a fixed-length sliding window of filtered samples.
//! The window was previously a `Vec<f64>` whose oldest-sample eviction used
//! `Vec::remove(0)` — an O(n) shift of the whole buffer on every sample, making
//! a full-window `extract()` sweep O(n²). ADR-157 §A1 switched the window to a
//! `VecDeque<f64>` (O(1) `push_back` + `pop_front`, with one `make_contiguous`
//! per call for the autocorrelation / zero-crossing loop).
//!
//! These benches measure the payoff: a full-window fill of each extractor
//! (`HeartRateExtractor` ~1500 samples, `BreathingExtractor` ~3000 samples).
//! Each iteration drives the extractor from empty to a full window so the
//! per-sample eviction cost (the thing A1 changed) is exercised across the
//! entire buffer.
//!
//! Reproduce:
//!   cargo bench -p wifi-densepose-vitals --bench vitals_bench
//! Compile-only:
//!   cargo bench -p wifi-densepose-vitals --bench vitals_bench --no-run

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wifi_densepose_vitals::{BreathingExtractor, HeartRateExtractor};

/// Drive a heart-rate extractor from empty to a full window.
///
/// `fs = 100 Hz`, `window = 15 s` -> 1500 samples. A few coherent subcarriers
/// of a synthetic cardiac sinusoid are fed each frame; the point of the bench
/// is the sliding-window bookkeeping, not the signal content.
fn bench_heartrate_full_window(c: &mut Criterion) {
    let sample_rate = 100.0;
    let window_secs = 15.0;
    let n_frames = (sample_rate * window_secs) as usize; // 1500
    let heart_freq = 1.2; // 72 BPM

    c.bench_function("heartrate_extract_full_window_1500", |b| {
        b.iter(|| {
            let mut ext = HeartRateExtractor::new(4, sample_rate, window_secs);
            for i in 0..n_frames {
                let t = i as f64 / sample_rate;
                let base = (2.0 * std::f64::consts::PI * heart_freq * t).sin();
                let residuals = [base * 0.1, base * 0.08, base * 0.12, base * 0.09];
                let phases = [0.0, 0.01, 0.02, 0.03];
                black_box(ext.extract(black_box(&residuals), black_box(&phases)));
            }
            black_box(ext.history_len());
        });
    });
}

/// Drive a breathing extractor from empty to a full window.
///
/// `fs = 100 Hz`, `window = 30 s` -> 3000 samples.
fn bench_breathing_full_window(c: &mut Criterion) {
    let sample_rate = 100.0;
    let window_secs = 30.0;
    let n_frames = (sample_rate * window_secs) as usize; // 3000
    let breathing_freq = 0.25; // 15 BPM

    c.bench_function("breathing_extract_full_window_3000", |b| {
        b.iter(|| {
            let mut ext = BreathingExtractor::new(4, sample_rate, window_secs);
            for i in 0..n_frames {
                let t = i as f64 / sample_rate;
                let s = (2.0 * std::f64::consts::PI * breathing_freq * t).sin();
                let residuals = [s, s * 0.9, s * 1.1, s * 0.95];
                let weights = [0.25, 0.25, 0.25, 0.25];
                black_box(ext.extract(black_box(&residuals), black_box(&weights)));
            }
            black_box(ext.history_len());
        });
    });
}

criterion_group!(benches, bench_heartrate_full_window, bench_breathing_full_window);
criterion_main!(benches);
