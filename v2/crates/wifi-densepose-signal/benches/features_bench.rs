//! ADR-154 perf benchmarks: FFT-planner caching (PSD) and DTW Sakoe-Chiba band.
//!
//! These benches back the *measured* before/after claims in
//! `docs/adr/ADR-154-signal-dsp-beyond-sota.md`. Every claim in that ADR has a
//! reproduce command pointing here — no perf number ships without a bench.
//!
//! Reproduce (compile-only):
//!   cargo bench -p wifi-densepose-signal --no-default-features \
//!     --bench features_bench --no-run
//!
//! Reproduce (full run, writes target/criterion/ HTML):
//!   cargo bench -p wifi-densepose-signal --no-default-features --bench features_bench
//!
//! Two groups:
//!   * `psd_fft_planner`    — `from_csi_data` (re-plans every call) vs
//!                            `from_csi_data_with_fft` (cached plan). Same output
//!                            (proved bit-identical in features.rs tests).
//!   * `dtw_sakoe_chiba`    — full-row baseline (walks 1..=m, the pre-ADR-154
//!                            behaviour) vs the banded loop (walks the band only).
//!                            Both functions are inlined here because the crate's
//!                            `dtw_distance` is private; the banded copy is a
//!                            faithful transcription of the shipped fix.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use rustfft::FftPlanner;
use std::time::Duration;

use wifi_densepose_signal::{CsiData, PowerSpectralDensity};

// ---------------------------------------------------------------------------
// PSD: fresh-planner vs cached-planner
// ---------------------------------------------------------------------------

fn make_csi(subcarriers: usize) -> CsiData {
    use std::f64::consts::PI;
    let antennas = 4;
    let mut amplitude = Array2::zeros((antennas, subcarriers));
    let mut phase = Array2::zeros((antennas, subcarriers));
    for i in 0..antennas {
        for j in 0..subcarriers {
            amplitude[[i, j]] = 0.5 + 0.3 * ((j as f64 / subcarriers as f64) * PI).sin();
            phase[[i, j]] = (j as f64 / subcarriers as f64) * 2.0 * PI - PI;
        }
    }
    CsiData::builder()
        .amplitude(amplitude)
        .phase(phase)
        .bandwidth(20.0e6)
        .build()
        .unwrap()
}

fn bench_psd_fft_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("psd_fft_planner");
    group.measurement_time(Duration::from_secs(4));

    for &fft_size in &[64usize, 128, 256] {
        let csi = make_csi(fft_size);
        group.throughput(Throughput::Elements(1));

        // BEFORE: re-plans a FftPlanner on every frame.
        group.bench_with_input(
            BenchmarkId::new("fresh_planner", fft_size),
            &fft_size,
            |b, &n| {
                b.iter(|| {
                    let psd = PowerSpectralDensity::from_csi_data(black_box(&csi), black_box(n));
                    black_box(psd.total_power)
                });
            },
        );

        // AFTER: plan once, reuse across frames (the FeatureExtractor path).
        let mut planner = FftPlanner::<f64>::new();
        let plan = planner.plan_fft_forward(fft_size);
        group.bench_with_input(
            BenchmarkId::new("cached_planner", fft_size),
            &fft_size,
            |b, &n| {
                b.iter(|| {
                    let psd = PowerSpectralDensity::from_csi_data_with_fft(
                        black_box(&csi),
                        black_box(n),
                        black_box(&plan),
                    );
                    black_box(psd.total_power)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// DTW: full-row baseline vs Sakoe-Chiba band
// ---------------------------------------------------------------------------

#[inline]
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Pre-ADR-154 behaviour: iterate the FULL 1..=m row, `continue` on out-of-band.
fn dtw_fullrow(seq_a: &[Vec<f64>], seq_b: &[Vec<f64>], band_width: usize) -> f64 {
    let (n, m) = (seq_a.len(), seq_b.len());
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        curr[0] = f64::INFINITY;
        let j_start = if band_width >= i {
            1
        } else {
            i.saturating_sub(band_width).max(1)
        };
        let j_end = (i + band_width).min(m);
        for j in 1..=m {
            if j < j_start || j > j_end {
                curr[j] = f64::INFINITY;
                continue;
            }
            let cost = euclidean(&seq_a[i - 1], &seq_b[j - 1]);
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Post-ADR-154: iterate the band only (transcription of the shipped fix).
fn dtw_banded(seq_a: &[Vec<f64>], seq_b: &[Vec<f64>], band_width: usize) -> f64 {
    let (n, m) = (seq_a.len(), seq_b.len());
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        curr[0] = f64::INFINITY;
        let j_start = if band_width >= i {
            1
        } else {
            i.saturating_sub(band_width).max(1)
        };
        let j_end = (i + band_width).min(m);
        if j_start >= 1 && j_start - 1 <= m {
            curr[j_start - 1] = f64::INFINITY;
        }
        for j in j_start..=j_end {
            let cost = euclidean(&seq_a[i - 1], &seq_b[j - 1]);
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        if j_end + 1 <= m {
            curr[j_end + 1] = f64::INFINITY;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let lo = n.saturating_sub(band_width).max(1);
    let hi = (n + band_width).min(m);
    if m >= lo && m <= hi {
        prev[m]
    } else {
        f64::INFINITY
    }
}

fn make_seq(len: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((s >> 33) as f64) / (u32::MAX as f64);
            vec![x, 1.0 - x, x * 0.5]
        })
        .collect()
}

fn bench_dtw_band(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_sakoe_chiba");
    group.measurement_time(Duration::from_secs(4));

    // The ADR claim case: n = m = 200, band = 5.
    for &(n, band) in &[(100usize, 5usize), (200, 5), (200, 10)] {
        let a = make_seq(n, 0x1234);
        let b = make_seq(n, 0x9abc);
        // Cells touched ≈ full: n*n; banded: n*(2*band+1).
        group.throughput(Throughput::Elements((n * n) as u64));

        group.bench_with_input(
            BenchmarkId::new("full_row", format!("n{n}_band{band}")),
            &band,
            |bch, &bw| {
                bch.iter(|| black_box(dtw_fullrow(black_box(&a), black_box(&b), bw)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("banded", format!("n{n}_band{band}")),
            &band,
            |bch, &bw| {
                bch.iter(|| black_box(dtw_banded(black_box(&a), black_box(&b), bw)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_psd_fft_planner, bench_dtw_band);
criterion_main!(benches);
