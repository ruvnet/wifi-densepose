//! Criterion bench for `cog-person-count` steady-state inference latency
//! (ADR-163, closing the ADR-159/160 deferred "cog inference latency bench" item).
//!
//! ## What this measures — and what the manifest's `cold_start_ms` does NOT
//!
//! This benches **steady-state** `InferenceEngine::infer` over a FIXED CSI
//! window on `Device::Cpu` with the **real** shipped `count_v1.safetensors`
//! weights — i.e. the per-frame cost once the model is loaded and warm.
//!
//! The cog manifest's `build_metadata.cold_start_ms_avg` (in the pose cog;
//! person-count's manifest carries comparable provenance) is a **DIFFERENT
//! measurement**: it includes one-time weight load / mmap / first-forward
//! allocation. Cold-start is a startup cost paid once; steady-state infer is the
//! recurring per-frame cost. They are not comparable and we do not conflate them.
//! `cold_start` was measured on ruvultra (RTX 5080 host, candle 0.9 cpu); this
//! bench runs on whatever machine you run it on — see `benchmarks/edge-latency/RESULTS.md`
//! for the host the committed numbers were taken on.
//!
//! If the weights file is absent the engine falls back to the zero-confidence
//! stub; we skip the bench in that case rather than benchmark the stub (which
//! would be a meaningless number) — the bench prints a notice and measures a
//! no-op so criterion still produces a (clearly-labelled) datapoint.
//!
//! Run (cog crates are normal workspace members):
//!   cd v2 && cargo bench -p cog-person-count --no-default-features
//!   cd v2 && cargo bench -p cog-person-count --no-default-features -- --warm-up-time 1 --measurement-time 2

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};

use cog_person_count::inference::{CsiWindow, InferenceEngine, INPUT_SUBCARRIERS, INPUT_TIMESTEPS};

/// Deterministic fixed CSI window (seed-stable LCG), normalised-ish amplitudes.
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

/// Locate the real weights from the crate dir or the repo root.
fn real_weights() -> Option<std::path::PathBuf> {
    let candidates = [
        "cog/artifacts/count_v1.safetensors",
        "v2/crates/cog-person-count/cog/artifacts/count_v1.safetensors",
        "crates/cog-person-count/cog/artifacts/count_v1.safetensors",
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
                InferenceEngine::with_weights(Some(&path)).expect("load real count_v1 weights");
            assert!(
                engine.backend().starts_with("candle-"),
                "expected real Candle backend, got {} — bench would measure the stub",
                engine.backend()
            );
            // Sanity: one real inference before timing.
            let _ = engine.infer(&window).expect("warmup infer");

            c.bench_function("cog_person_count::infer[cpu_real_weights_steady_state]", |b| {
                b.iter(|| {
                    black_box(engine.infer(black_box(&window)).expect("infer"));
                });
            });
        }
        None => {
            eprintln!(
                "NOTE: count_v1.safetensors not found — skipping the real-weights infer bench. \
                 (The committed RESULTS.md numbers require the in-repo weights.)"
            );
            c.bench_function("cog_person_count::infer[SKIPPED_no_weights]", |b| {
                b.iter(|| black_box(1 + 1));
            });
        }
    }
}

criterion_group!(benches, bench_infer);
criterion_main!(benches);
