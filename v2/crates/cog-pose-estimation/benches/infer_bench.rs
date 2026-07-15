//! Criterion bench for `cog-pose-estimation` steady-state inference latency
//! (ADR-163, closing the ADR-159/160 deferred "cog inference latency bench" item).
//!
//! ## What this measures — and what the manifest's `cold_start_ms_avg` does NOT
//!
//! The pose cog's manifest (`cog/artifacts/manifests/x86_64/manifest.json`)
//! cites `build_metadata.cold_start_ms_avg: 5.4` (30 invocations, measured on
//! ruvultra / RTX 5080 host, candle 0.9 cpu). **That is a cold-start number** —
//! it folds in one-time weight load / mmap / first-forward allocation.
//!
//! This bench measures the **steady-state** per-frame cost instead:
//! `InferenceEngine::infer` over a FIXED CSI window on `Device::Cpu` with the
//! **real** shipped `pose_v1.safetensors`, after a warm-up forward. Steady-state
//! and cold-start are different measurements; we label both honestly and do not
//! claim this reproduces the 5.4 ms manifest figure (different machine, different
//! measurement). See `benchmarks/edge-latency/RESULTS.md`.
//!
//! Run (cog crates are normal workspace members):
//!   cd v2 && cargo bench -p cog-pose-estimation --no-default-features
//!   cd v2 && cargo bench -p cog-pose-estimation --no-default-features -- --warm-up-time 1 --measurement-time 2

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};

use cog_pose_estimation::inference::{
    CsiWindow, InferenceEngine, INPUT_SUBCARRIERS, INPUT_TIMESTEPS,
};

/// Deterministic fixed CSI window (seed-stable LCG).
fn fixed_window() -> CsiWindow {
    let mut s = 0x00C0_FFEEu32;
    let data: Vec<f32> = (0..INPUT_SUBCARRIERS * INPUT_TIMESTEPS)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 16) as f32 / 32768.0 // [0, 1)
        })
        .collect();
    CsiWindow { data }
}

fn real_weights() -> Option<std::path::PathBuf> {
    let candidates = [
        "cog/artifacts/pose_v1.safetensors",
        "v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors",
        "crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors",
    ];
    candidates
        .iter()
        .map(Path::new)
        .find(|p| p.exists())
        .map(|p| p.to_path_buf())
}

fn bench_infer(c: &mut Criterion) {
    let window = fixed_window();

    match real_weights() {
        Some(path) => {
            let engine =
                InferenceEngine::with_weights(Some(&path)).expect("load real pose_v1 weights");
            assert!(
                engine.backend().starts_with("candle-"),
                "expected real Candle backend, got {} — bench would measure the stub",
                engine.backend()
            );
            let _ = engine.infer(&window).expect("warmup infer");

            c.bench_function("cog_pose_estimation::infer[cpu_real_weights_steady_state]", |b| {
                b.iter(|| {
                    black_box(engine.infer(black_box(&window)).expect("infer"));
                });
            });
        }
        None => {
            eprintln!(
                "NOTE: pose_v1.safetensors not found — skipping the real-weights infer bench. \
                 (The committed RESULTS.md numbers require the in-repo weights.)"
            );
            c.bench_function("cog_pose_estimation::infer[SKIPPED_no_weights]", |b| {
                b.iter(|| black_box(1 + 1));
            });
        }
    }
}

criterion_group!(benches, bench_infer);
criterion_main!(benches);
