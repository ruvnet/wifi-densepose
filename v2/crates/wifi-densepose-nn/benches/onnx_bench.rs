//! ADR-155 ONNX backend micro-benchmarks.
//!
//! Two measured concerns:
//!
//! * **WIN 2 — input copy.** `OnnxSession::run` builds the ORT input from the
//!   ndarray. `input_copy_contiguous` measures the difference between the old
//!   element-wise `iter().cloned().collect()` and the new
//!   `as_slice().to_vec()` zero-copy-when-contiguous path. `input_copy_strided`
//!   confirms the fallback still works on a non-contiguous view.
//!
//! * **WIN 1 — concurrency.** `onnx_concurrency` runs real inference over a
//!   shared `Arc<OnnxBackend>` at 1/2/4/8 threads. It documents the current
//!   serialized behaviour (ort 2.0.0-rc.11 `Session::run` is `&mut self`, so the
//!   backend holds a write lock). It is the harness that would show the speedup
//!   if a `&self` run path becomes available.
//!
//! Requires the `onnx` feature and a real ORT runtime. The fixture model is
//! `tests/fixtures/tiny_conv.onnx` (input `[1,3,8,8]` -> Conv -> Relu).
//!
//! Reproduce:
//!   cargo bench -p wifi-densepose-nn --no-default-features --features onnx --bench onnx_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array4;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use wifi_densepose_nn::inference::Backend;
use wifi_densepose_nn::onnx::OnnxBackend;
use wifi_densepose_nn::tensor::Tensor;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_conv.onnx")
}

/// Representative input shape matching the fixture model.
const SHAPE: [usize; 4] = [1, 3, 8, 8];

/// Old path: full element-wise iterator copy.
#[inline]
fn copy_iter(arr: &Array4<f32>) -> Vec<f32> {
    arr.iter().cloned().collect()
}

/// New path: zero-copy `as_slice()` when contiguous, else iterator fallback.
#[inline]
fn copy_slice(arr: &Array4<f32>) -> Vec<f32> {
    match arr.as_slice() {
        Some(slice) => slice.to_vec(),
        None => arr.iter().cloned().collect(),
    }
}

/// WIN 2 — input copy, before vs after, on a standard-layout (contiguous) array.
fn bench_input_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_input_copy");

    // A larger, realistic CSI-like input to make the copy cost visible.
    let big_shape = [1usize, 256, 64, 64];
    let arr: Array4<f32> = Array4::from_shape_fn(big_shape, |(_, c, h, w)| (c + h + w) as f32);
    let n = big_shape.iter().product::<usize>() as u64;
    group.throughput(Throughput::Elements(n));

    group.bench_function("contiguous_iter_clone_before", |b| {
        b.iter(|| black_box(copy_iter(black_box(&arr))))
    });
    group.bench_function("contiguous_as_slice_after", |b| {
        b.iter(|| black_box(copy_slice(black_box(&arr))))
    });

    // Non-contiguous (transposed view) — confirms the fallback still works and
    // measures it. `permuted_axes` yields a non-standard layout, so `as_slice()`
    // returns None and we hit the iterator fallback.
    let strided = arr.view().permuted_axes([0, 2, 3, 1]).to_owned();
    group.bench_function("strided_iter_clone_before", |b| {
        b.iter(|| black_box(strided.iter().cloned().collect::<Vec<f32>>()))
    });
    group.bench_function("strided_as_slice_after", |b| {
        b.iter(|| {
            black_box(match strided.as_slice() {
                Some(s) => s.to_vec(),
                None => strided.iter().cloned().collect::<Vec<f32>>(),
            })
        })
    });

    group.finish();
}

/// WIN 2 — end-to-end single inference (input build + ORT run) with the real model.
fn bench_single_inference(c: &mut Criterion) {
    let path = fixture_path();
    if !path.exists() {
        eprintln!("skip onnx single inference: fixture missing at {path:?}");
        return;
    }
    let backend = match OnnxBackend::from_file(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip onnx single inference: failed to load model: {e}");
            return;
        }
    };
    let input_name = backend.input_names()[0].clone();
    let input = Tensor::from_array4(Array4::from_elem(SHAPE, 0.5f32));

    let mut group = c.benchmark_group("onnx_single_inference");
    group.bench_function("infer", |b| {
        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.clone(), input.clone());
            black_box(backend.run(inputs).unwrap())
        })
    });
    group.finish();
}

/// WIN 1 — concurrency harness: shared `Arc<OnnxBackend>` across N threads.
fn bench_concurrency(c: &mut Criterion) {
    let path = fixture_path();
    if !path.exists() {
        eprintln!("skip onnx concurrency: fixture missing at {path:?}");
        return;
    }
    let backend = match OnnxBackend::from_file(&path) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("skip onnx concurrency: failed to load model: {e}");
            return;
        }
    };
    let input_name = backend.input_names()[0].clone();

    let mut group = c.benchmark_group("onnx_concurrency");
    // Fixed total work (inferences) per iteration, split across threads. Lower
    // wall time at higher thread counts == real concurrency gain.
    const TOTAL: usize = 64;

    for threads in [1usize, 2, 4, 8] {
        group.throughput(Throughput::Elements(TOTAL as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let per = TOTAL / threads;
                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let backend = Arc::clone(&backend);
                            let name = input_name.clone();
                            thread::spawn(move || {
                                let input = Tensor::from_array4(Array4::from_elem(SHAPE, 0.5f32));
                                for _ in 0..per {
                                    let mut inputs = HashMap::new();
                                    inputs.insert(name.clone(), input.clone());
                                    black_box(backend.run(inputs).unwrap());
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_input_copy,
    bench_single_inference,
    bench_concurrency,
);
criterion_main!(benches);
