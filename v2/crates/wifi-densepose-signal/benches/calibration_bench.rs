//! Criterion benchmarks for the empty-room baseline calibration module (ADR-135).
//!
//! Measures per-call throughput of CalibrationRecorder and BaselineCalibration
//! across HT20 (K=52), HT40 (K=114), HE20 (K=256, all bins; #1009), and HE40 (K=484).
//!
//! Run (compile-only — no execution):
//!   cargo bench -p wifi-densepose-signal --no-default-features --bench calibration_bench --no-run
//!
//! Run to completion (generates HTML in target/criterion/):
//!   cargo bench -p wifi-densepose-signal --no-default-features --bench calibration_bench

use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::calibration::{
    BaselineCalibration, CalibrationConfig, CalibrationRecorder,
};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32, seed=42) — duplicated locally.
// ---------------------------------------------------------------------------

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        assert_ne!(seed, 0);
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Tier specification table
// ---------------------------------------------------------------------------

struct TierSpec {
    label: &'static str,
    n_active: usize,
    bandwidth_mhz: u16,
    config: CalibrationConfig,
}

fn tiers() -> Vec<TierSpec> {
    vec![
        TierSpec { label: "ht20", n_active: 52,  bandwidth_mhz: 20, config: CalibrationConfig::ht20() },
        TierSpec { label: "ht40", n_active: 114, bandwidth_mhz: 40, config: CalibrationConfig::ht40() },
        // Issue #1009 §1b: HE20 records all 256 delivered bins (he20().num_active == 256).
        TierSpec { label: "he20", n_active: 256, bandwidth_mhz: 20, config: CalibrationConfig::he20() },
        TierSpec { label: "he40", n_active: 484, bandwidth_mhz: 40, config: CalibrationConfig::he40() },
    ]
}

// ---------------------------------------------------------------------------
// Synthetic CSI frame builder (stationary, seed=42)
// ---------------------------------------------------------------------------

fn make_frame(n_active: usize, bandwidth_mhz: u16, rng: &mut Rng) -> CsiFrame {
    let noise_std = 0.01_f64;
    let mut data = Array2::<Complex64>::zeros((1, n_active));
    for k in 0..n_active {
        let amp = 0.3 + 0.7 * (k as f64 * PI / n_active as f64).sin().abs();
        let phase = (k as f64 * 0.1).rem_euclid(2.0 * PI) - PI;
        let re = amp * phase.cos() + noise_std * rng.next_normal();
        let im = amp * phase.sin() + noise_std * rng.next_normal();
        data[(0, k)] = Complex64::new(re, im);
    }
    let mut meta = CsiMetadata::new(DeviceId::new("bench"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = bandwidth_mhz;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

/// Build a `CalibrationRecorder` that has already absorbed 600 frames.
fn pre_loaded_recorder(spec: &TierSpec) -> CalibrationRecorder {
    let mut rng = Rng::new(42);
    let mut recorder = CalibrationRecorder::new(spec.config.clone());
    for _ in 0..600 {
        let frame = make_frame(spec.n_active, spec.bandwidth_mhz, &mut rng);
        recorder.record(&frame).expect("record should succeed in bench setup");
    }
    recorder
}

/// Build a finalised `BaselineCalibration` for deviation and to_bytes benches.
fn finalised_baseline(spec: &TierSpec) -> BaselineCalibration {
    pre_loaded_recorder(spec)
        .finalize()
        .expect("finalize should succeed in bench setup")
}

// ---------------------------------------------------------------------------
// Bench 1: bench_recorder_record/<tier> — single record() call (hot path)
// ---------------------------------------------------------------------------

fn bench_recorder_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_recorder_record");
    for spec in tiers() {
        group.throughput(Throughput::Elements(spec.n_active as u64));
        let mut rng = Rng::new(42);
        let frame = make_frame(spec.n_active, spec.bandwidth_mhz, &mut rng);
        let mut recorder = CalibrationRecorder::new(spec.config.clone());

        group.bench_with_input(
            BenchmarkId::from_parameter(spec.label),
            &frame,
            |b, f| {
                b.iter(|| {
                    // Accumulate into a shared recorder — measures per-call cost of record().
                    black_box(recorder.record(black_box(f)).ok())
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 2: bench_recorder_finalize/<tier> — finalize() from 600 pre-loaded frames
// ---------------------------------------------------------------------------

fn bench_recorder_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_recorder_finalize");
    for spec in tiers() {
        group.throughput(Throughput::Elements(spec.n_active as u64));

        group.bench_function(BenchmarkId::from_parameter(spec.label), |b| {
            b.iter_with_setup(
                || pre_loaded_recorder(&spec),
                |recorder| {
                    black_box(recorder.finalize().ok())
                },
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 3: bench_deviation/<tier> — deviation() on a single frame
// ---------------------------------------------------------------------------

fn bench_deviation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_deviation");
    for spec in tiers() {
        group.throughput(Throughput::Elements(spec.n_active as u64));
        let baseline = finalised_baseline(&spec);
        let mut rng = Rng::new(42);
        let frame = make_frame(spec.n_active, spec.bandwidth_mhz, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(spec.label),
            &frame,
            |b, f| {
                b.iter(|| {
                    black_box(baseline.deviation(black_box(f)).ok())
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 4: bench_record_600/<tier> — full 600-frame record session
// ---------------------------------------------------------------------------

fn bench_record_600(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_record_600");
    for spec in tiers() {
        group.throughput(Throughput::Elements(600 * spec.n_active as u64));

        // Pre-build 600 frames to avoid contaminating bench with frame construction.
        let mut rng = Rng::new(42);
        let frames: Vec<CsiFrame> = (0..600)
            .map(|_| make_frame(spec.n_active, spec.bandwidth_mhz, &mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(spec.label),
            &frames,
            |b, fs| {
                b.iter_with_setup(
                    || CalibrationRecorder::new(spec.config.clone()),
                    |mut recorder| {
                        for f in fs {
                            black_box(recorder.record(black_box(f)).ok());
                        }
                        black_box(recorder)
                    },
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 5: bench_to_bytes/<tier> — serialisation cost (to_bytes)
// ---------------------------------------------------------------------------

fn bench_to_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_to_bytes");
    for spec in tiers() {
        group.throughput(Throughput::Elements(spec.n_active as u64));
        let baseline = finalised_baseline(&spec);

        group.bench_function(BenchmarkId::from_parameter(spec.label), |b| {
            b.iter(|| {
                black_box(baseline.to_bytes())
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_recorder_record,
    bench_recorder_finalize,
    bench_deviation,
    bench_record_600,
    bench_to_bytes,
);
criterion_main!(benches);
