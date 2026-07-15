//! ADR-155 M2 §4 — native (pure-Rust) DensePose conv benchmark.
//!
//! `DensePoseHead::apply_conv_layer` is a pure-Rust naive 6-nested-loop
//! convolution (the §8 "native-conv naive-loop" backlog item). This bench
//! measures `forward()` (which runs the shared-conv + segmentation + UV conv
//! stacks through that naive loop) on a representative single-layer config so a
//! perf claim can be made (or refused) with a MEASURED before/after — never a
//! fabricated number.
//!
//! Reproduce:
//!   cargo bench -p wifi-densepose-nn --no-default-features --bench native_conv_bench
//!
//! The bench is `--no-default-features` (no `onnx`/`ort` download needed): the
//! conv path is pure-Rust and benchable on any host.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array4};
use std::hint::black_box;
use wifi_densepose_nn::densepose::{ConvLayerWeights, DensePoseWeights};
use wifi_densepose_nn::{DensePoseConfig, DensePoseHead, Tensor};

/// Build a single same-padding conv layer `in_ch -> out_ch`, kernel `k`, with a
/// bias (no batch-norm) — deterministic, small, representative of one stage.
fn conv_layer(in_ch: usize, out_ch: usize, k: usize) -> ConvLayerWeights {
    let weight = Array4::from_shape_fn((out_ch, in_ch, k, k), |(o, i, kh, kw)| {
        // Deterministic, bounded weights.
        ((o + i + kh + kw) as f32 * 0.013).sin()
    });
    ConvLayerWeights {
        weight,
        bias: Some(Array1::from_shape_fn(out_ch, |o| o as f32 * 0.01)),
        bn_gamma: None,
        bn_beta: None,
        bn_mean: None,
        bn_var: None,
    }
}

/// A head whose shared-conv stack is one `ch->ch` conv, with empty seg/uv heads,
/// so the bench isolates a single conv-layer cost.
fn single_conv_head(ch: usize, k: usize) -> DensePoseHead {
    let mut config = DensePoseConfig::new(ch, 1, 2);
    config.kernel_size = k;
    config.padding = k / 2; // same padding
    config.hidden_channels = vec![ch];
    let weights = DensePoseWeights {
        shared_conv: vec![conv_layer(ch, ch, k)],
        segmentation_head: vec![],
        uv_head: vec![],
    };
    DensePoseHead::with_weights(config, weights).expect("valid head")
}

fn bench_native_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_conv");
    // (channels, spatial, kernel) — a modest map and a larger one.
    for &(ch, hw, k) in &[(16usize, 32usize, 3usize), (32, 32, 3)] {
        let head = single_conv_head(ch, k);
        let input = Tensor::Float4D(Array4::from_shape_fn((1, ch, hw, hw), |(_, c, y, x)| {
            ((c + y + x) as f32 * 0.001).cos()
        }));
        // Throughput in output elements processed.
        group.throughput(Throughput::Elements((ch * hw * hw) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("ch{ch}_hw{hw}_k{k}")),
            &input,
            |bencher, inp| {
                bencher.iter(|| {
                    let out = head.forward(black_box(inp)).expect("forward ok");
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_native_conv);
criterion_main!(benches);
