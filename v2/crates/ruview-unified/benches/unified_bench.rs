//! Criterion benchmarks for the unified RF spatial world model hot paths
//! (ADR-273 §7): encoder inference (the edge-latency budget), tokenization
//! (with vs without precomputed DFT twiddles), Gaussian map queries (spatial
//! hash vs linear-scan baseline), channel-gain evaluation, and synthetic
//! window generation.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use ruview_unified::encoder::{EncoderConfig, RfEncoder};
use ruview_unified::gaussian::map::GaussianMap;
use ruview_unified::gaussian::primitive::{Provenance, RfGaussian};
use ruview_unified::gaussian::{channel_gain, observe_link};
use ruview_unified::math::{dft_magnitudes, DftPlan};
use ruview_unified::synth::{SynthConfig, SynthGenerator};
use ruview_unified::tokenizer::RfTokenizer;

fn synth_corpus() -> Vec<ruview_unified::synth::LabeledWindow> {
    SynthGenerator::new(SynthConfig {
        seed: 11,
        n_rooms: 2,
        windows_per_room: 4,
        links: 3,
        snapshot_dt_s: 0.05,
    })
    .generate()
}

fn bench_encoder_forward(c: &mut Criterion) {
    let corpus = synth_corpus();
    let tokenizer = RfTokenizer::new();
    let window = tokenizer.tokenize(&corpus[0].tensor);
    let encoder = RfEncoder::new(EncoderConfig::default(), 42);
    c.bench_function("encoder_encode_window_21tok_d128", |b| {
        b.iter(|| std::hint::black_box(encoder.encode(&window)));
    });
}

fn bench_tokenizer(c: &mut Criterion) {
    let corpus = synth_corpus();
    let tokenizer = RfTokenizer::new();
    c.bench_function("tokenize_3link_56bin_8snap", |b| {
        b.iter(|| std::hint::black_box(tokenizer.tokenize(&corpus[0].tensor)));
    });
}

fn bench_dft_plan_vs_naive(c: &mut Criterion) {
    use num_complex::Complex64;
    let x: Vec<Complex64> =
        (0..8).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect();
    let plan = DftPlan::new(8, 4);
    c.bench_function("dft8x4_naive", |b| {
        b.iter(|| std::hint::black_box(dft_magnitudes(&x, 4)));
    });
    c.bench_function("dft8x4_planned", |b| {
        b.iter(|| std::hint::black_box(plan.magnitudes(&x)));
    });
}

fn populated_map(n_side: usize) -> GaussianMap {
    let mut map = GaussianMap::new(1.0);
    for x in 0..n_side {
        for y in 0..n_side {
            let g = RfGaussian::new(
                [x as f64 * 1.5, y as f64 * 1.5, 1.0],
                [0.3, 0.3, 0.3],
                [1.0, 0.0, 0.0, 0.0],
                0.4,
                0.9,
                0,
                600.0,
                Provenance { device_id: "bench".into(), model_version: 1, synthetic: true },
            )
            .expect("valid");
            map.insert(g);
        }
    }
    map
}

fn bench_gaussian_map(c: &mut Criterion) {
    // Two sizes to show the hash/linear crossover honestly: at 1,024
    // Gaussians a brute-force scan is competitive for small-radius queries;
    // at 16,384 the hash wins decisively.
    for (label, side) in [("1k", 32usize), ("16k", 128)] {
        let map = populated_map(side);
        let centre = [side as f64 * 0.75, side as f64 * 0.75, 1.0];
        c.bench_function(&format!("map{label}_query_radius_hash"), |b| {
            b.iter(|| std::hint::black_box(map.query_radius(centre, 3.0)));
        });
        c.bench_function(&format!("map{label}_query_radius_linear"), |b| {
            b.iter(|| std::hint::black_box(map.query_radius_linear(centre, 3.0)));
        });
        c.bench_function(&format!("map{label}_segment_corridor_hash"), |b| {
            b.iter(|| {
                std::hint::black_box(map.query_near_segment(
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    3.0,
                ))
            });
        });
        c.bench_function(&format!("map{label}_segment_corridor_linear"), |b| {
            b.iter(|| {
                std::hint::black_box(map.query_near_segment_linear(
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    3.0,
                ))
            });
        });
        c.bench_function(&format!("map{label}_channel_gain"), |b| {
            b.iter(|| {
                std::hint::black_box(channel_gain(
                    &map,
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    2.437e9,
                ))
            });
        });
    }
    c.bench_function("map_observe_link_inverse_update", |b| {
        b.iter_batched(
            || populated_map(8),
            |mut m| {
                std::hint::black_box(observe_link(
                    &mut m,
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    2.437e9,
                    1e-4,
                    0.7,
                    1,
                ))
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_synth_generation(c: &mut Criterion) {
    c.bench_function("synth_generate_1room_4windows_3links", |b| {
        b.iter(|| {
            std::hint::black_box(
                SynthGenerator::new(SynthConfig {
                    seed: 3,
                    n_rooms: 1,
                    windows_per_room: 4,
                    links: 3,
                    snapshot_dt_s: 0.05,
                })
                .generate(),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_encoder_forward,
    bench_tokenizer,
    bench_dft_plan_vs_naive,
    bench_gaussian_map,
    bench_synth_generation
);
criterion_main!(benches);
