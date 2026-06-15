//! Criterion benchmarks for the CIR estimator (ADR-134).
//!
//! Measures per-call throughput of `CirEstimator::estimate()` across all
//! four hardware tiers (HT20, HT40, HE20, HE40) and the 12-link amortization
//! pattern used by the RuvSense multistatic aggregator.
//!
//! Run (compile-only check):
//!   cargo bench -p wifi-densepose-signal --no-default-features --bench cir_bench --no-run
//!
//! Run to completion (slow — generates HTML reports in target/criterion/):
//!   cargo bench -p wifi-densepose-signal --no-default-features --bench cir_bench

#![cfg(feature = "cir")]

use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::cir::{CirConfig, CirEstimator};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32, seed=42)
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
// Synthetic CSI generator — 3-tap deterministic channel (seed=42)
// ---------------------------------------------------------------------------

/// Build a 3-tap deterministic CSI vector for the given config.
///
/// Tap parameters mirror `cir_synthetic.rs`:
///   direct path: τ=10 ns, amplitude 1.0
///   reflection 1: τ=80 ns, amplitude 0.6
///   reflection 2: τ=180 ns, amplitude 0.3
///
/// SNR = 20 dB, seed = 42.
fn synth_csi(cfg: &CirConfig) -> Vec<Complex64> {
    let k_active = cfg.delay_bins / 3;
    let delta_f = 312_500.0_f64; // Hz

    let taps: &[(f64, f64, f64)] = &[
        (10e-9, 1.0, PI / 4.0),
        (80e-9, 0.6, PI),
        (180e-9, 0.3, -PI / 3.0),
    ];

    // Forward projection
    let mut h: Vec<Complex64> = (0..k_active)
        .map(|k| {
            let val: Complex64 = taps
                .iter()
                .map(|(tau, amp, phase)| {
                    let angle = -2.0 * PI * k as f64 * delta_f * tau;
                    let re = amp * phase.cos() * angle.cos() - amp * phase.sin() * angle.sin();
                    let im = amp * phase.cos() * angle.sin() + amp * phase.sin() * angle.cos();
                    Complex64::new(re, im)
                })
                .sum();
            val
        })
        .collect();

    // Add AWGN at SNR=20 dB, seed=42
    let signal_power: f64 = h.iter().map(|c| c.norm_sqr()).sum::<f64>() / k_active as f64;
    let noise_power = signal_power / 10_f64.powf(20.0 / 10.0);
    let noise_std = (noise_power / 2.0).sqrt();

    let mut rng = Rng::new(42);
    for sample in h.iter_mut() {
        let n_i = noise_std * rng.next_normal();
        let n_q = noise_std * rng.next_normal();
        *sample += Complex64::new(n_i, n_q);
    }

    h
}

// ---------------------------------------------------------------------------
// CsiFrame construction
// ---------------------------------------------------------------------------

fn make_frame(bandwidth_mhz: u16, csi: Vec<Complex64>) -> CsiFrame {
    let k = csi.len();
    let mut data = Array2::zeros((1, k));
    for (i, &v) in csi.iter().enumerate() {
        data[(0, i)] = v;
    }
    let mut meta = CsiMetadata::new(DeviceId::new("bench"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = bandwidth_mhz;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

// ---------------------------------------------------------------------------
// Benchmark 1: single estimate() call per tier
// ---------------------------------------------------------------------------

fn bench_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("cir_estimate");

    let tiers: &[(&str, u16)] = &[
        ("ht20", 20),
        ("ht40", 40),
        ("he20", 20),   // HE20: same BW as HT20, different pilot mask — same for_bandwidth_mhz(20)
        ("he40", 40),   // HE40: same BW as HT40
    ];

    for &(label, bw_mhz) in tiers {
        let cfg = CirConfig::for_bandwidth_mhz(bw_mhz);
        let k_active = cfg.delay_bins / 3;

        group.throughput(Throughput::Elements(k_active as u64));

        let est = CirEstimator::new(cfg.clone());
        let csi = synth_csi(&cfg);
        let frame = make_frame(bw_mhz, csi);

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &frame,
            |b, f| {
                b.iter(|| {
                    black_box(est.estimate(black_box(f)).ok())
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 1b: opt-in FFT operator (CirConfig::fft_operator = true)
// ---------------------------------------------------------------------------

/// Same workload as `cir_estimate`, with the O(G log G) FFT Φ/Φᴴ operator
/// enabled. Compare against `cir_estimate/<tier>` for the dense baseline.
fn bench_estimate_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("cir_estimate_fft");

    let tiers: &[(&str, u16)] = &[("ht20", 20), ("ht40", 40), ("he40", 40)];

    for &(label, bw_mhz) in tiers {
        let mut cfg = CirConfig::for_bandwidth_mhz(bw_mhz);
        cfg.fft_operator = true;
        let k_active = cfg.delay_bins / 3;

        group.throughput(Throughput::Elements(k_active as u64));

        let est = CirEstimator::new(cfg.clone());
        let csi = synth_csi(&cfg);
        let frame = make_frame(bw_mhz, csi);

        group.bench_with_input(BenchmarkId::from_parameter(label), &frame, |b, f| {
            b.iter(|| black_box(est.estimate(black_box(f)).ok()));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: 12-link amortisation (shared estimator across links)
// ---------------------------------------------------------------------------

/// Simulates the RuvSense multistatic aggregator pattern: one shared
/// CirEstimator instance processes 12 sequential links per call.
/// This measures the per-cycle cost of a full mesh with 12 active links.
fn bench_estimate_12link(c: &mut Criterion) {
    let mut group = c.benchmark_group("cir_estimate_12link");

    for &(label, bw_mhz) in &[("ht20", 20u16), ("ht40", 40u16)] {
        let cfg = CirConfig::for_bandwidth_mhz(bw_mhz);
        let k_active = cfg.delay_bins / 3;

        // 12 distinct pre-built CSI frames (seeded differently to prevent
        // the compiler from deduplicating them).  Vary seed per link.
        let frames: Vec<CsiFrame> = (1u32..=12)
            .map(|seed| {
                let k = k_active;
                let delta_f = 312_500.0_f64;
                let mut rng = Rng::new(seed * 7 + 1); // deterministic per-link seed

                let signal_power = 1.0_f64;
                let noise_power = signal_power / 10_f64.powf(20.0 / 10.0);
                let noise_std = (noise_power / 2.0).sqrt();

                let csi: Vec<Complex64> = (0..k)
                    .map(|k_idx| {
                        let angle = -2.0 * PI * k_idx as f64 * delta_f * 30e-9;
                        let mut c = Complex64::new(angle.cos(), angle.sin());
                        c += Complex64::new(noise_std * rng.next_normal(), noise_std * rng.next_normal());
                        c
                    })
                    .collect();
                make_frame(bw_mhz, csi)
            })
            .collect();

        let est = CirEstimator::new(cfg.clone());

        group.throughput(Throughput::Elements(12 * k_active as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &frames,
            |b, fs| {
                b.iter(|| {
                    for f in fs {
                        black_box(est.estimate(black_box(f)).ok());
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: estimator construction cost (sensing matrix build)
// ---------------------------------------------------------------------------

/// Measures the one-time cost of CirEstimator::new() for each tier.
/// This is amortised over many frames but useful to understand cold-start cost.
fn bench_estimator_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cir_estimator_new");

    for &(label, bw_mhz) in &[("ht20", 20u16), ("ht40", 40u16)] {
        group.bench_function(label, |b| {
            b.iter(|| {
                let cfg = CirConfig::for_bandwidth_mhz(bw_mhz);
                black_box(CirEstimator::new(cfg))
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
    bench_estimate,
    bench_estimate_fft,
    bench_estimate_12link,
    bench_estimator_construction,
);
criterion_main!(benches);
