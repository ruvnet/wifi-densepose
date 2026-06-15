//! ADR-156 §finding 4/5 — cross-viewpoint fusion hot-path benchmark.
//!
//! Two groups:
//!
//! 1. **`fusion_pipeline`** — end-to-end `MultistaticArray::fuse()` at realistic
//!    array sizes (2–8 viewpoints) and the AETHER embedding dimension (128).
//!    This is the production fusion path exercised once per TDM cycle.
//!
//! 2. **`embedding_extract`** — an isolated A/B of the embedding-marshalling step
//!    that finding 4 fixed: the OLD code cloned every viewpoint embedding
//!    *twice* (once into `extracted`, once into `embeddings`); the NEW code
//!    clones once (out of the borrowed `viewpoints`) and then *moves* into the
//!    attention input. The `before_double_clone` / `after_single_clone` benches
//!    measure exactly that difference so the perf claim is MEASURED, not asserted.
//!
//! Run with:
//! ```bash
//! cargo bench -p wifi-densepose-ruvector --bench fusion_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint;
use wifi_densepose_ruvector::viewpoint::attention::ViewpointGeometry;
use wifi_densepose_ruvector::viewpoint::{FusionConfig, MultistaticArray, ViewpointEmbedding};

/// Deterministic pseudo-random embedding (LCG — no `rand` dev-dep needed).
fn make_embedding(dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..dim)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// Build a coherent array of `n` viewpoints with `dim`-d embeddings, gate open.
fn make_array(n: usize, dim: usize) -> MultistaticArray {
    let config = FusionConfig {
        embed_dim: dim,
        coherence_threshold: 0.5,
        coherence_hysteresis: 0.0,
        min_snr_db: 0.0,
        ..FusionConfig::default()
    };
    let mut array = MultistaticArray::new(1, config);
    for _ in 0..60 {
        array.push_phase_diff(0.1); // coherent → gate opens
    }
    for i in 0..n {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
        let r = 3.0;
        array
            .submit_viewpoint(ViewpointEmbedding {
                node_id: i as u32,
                embedding: make_embedding(dim, i as u32 + 1),
                azimuth: angle,
                elevation: 0.0,
                baseline: r,
                position: (r * angle.cos(), r * angle.sin()),
                snr_db: 15.0,
            })
            .unwrap();
    }
    array
}

fn bench_fusion_pipeline(c: &mut Criterion) {
    let dim = 128; // AETHER embedding dimension (ADR-024)
    let mut group = c.benchmark_group("fusion_pipeline");
    for n in [2usize, 4, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut array = make_array(n, dim);
            b.iter(|| {
                let fused = array.fuse_ungated().unwrap();
                hint::black_box(&fused);
            });
        });
    }
    group.finish();
}

// --- Finding 4 A/B: double-clone vs single-move embedding marshalling ---------

/// OLD behaviour: clone every embedding into `extracted`, then clone AGAIN into
/// the attention input vector (two heap allocations + two memcpys per viewpoint).
fn extract_double_clone(viewpoints: &[ViewpointEmbedding]) -> Vec<Vec<f32>> {
    type Ext = (u32, Vec<f32>, f32, (f32, f32));
    let extracted: Vec<Ext> = viewpoints
        .iter()
        .map(|v| (v.node_id, v.embedding.clone(), v.azimuth, v.position))
        .collect();
    // Second clone (the bug).
    let embeddings: Vec<Vec<f32>> = extracted.iter().map(|(_, e, _, _)| e.clone()).collect();
    let _geom: Vec<ViewpointGeometry> = extracted
        .iter()
        .map(|(_, _, az, pos)| ViewpointGeometry {
            azimuth: *az,
            position: *pos,
        })
        .collect();
    embeddings
}

/// NEW behaviour: clone once into `extracted`, then MOVE into the attention
/// input (one heap allocation + one memcpy per viewpoint).
fn extract_single_clone(viewpoints: &[ViewpointEmbedding]) -> Vec<Vec<f32>> {
    type Ext = (u32, Vec<f32>, f32, (f32, f32));
    let extracted: Vec<Ext> = viewpoints
        .iter()
        .map(|v| (v.node_id, v.embedding.clone(), v.azimuth, v.position))
        .collect();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(extracted.len());
    let mut _geom: Vec<ViewpointGeometry> = Vec::with_capacity(extracted.len());
    for (_, emb, az, pos) in extracted {
        _geom.push(ViewpointGeometry { azimuth: az, position: pos });
        embeddings.push(emb); // move
    }
    embeddings
}

fn bench_embedding_extract(c: &mut Criterion) {
    let dim = 128;
    let n = 8; // max realistic multistatic array
    let viewpoints: Vec<ViewpointEmbedding> = (0..n)
        .map(|i| ViewpointEmbedding {
            node_id: i as u32,
            embedding: make_embedding(dim, i as u32 + 1),
            azimuth: 0.0,
            elevation: 0.0,
            baseline: 3.0,
            position: (0.0, 0.0),
            snr_db: 15.0,
        })
        .collect();

    let mut group = c.benchmark_group("embedding_extract");
    group.bench_function("before_double_clone", |b| {
        b.iter(|| black_box(extract_double_clone(black_box(&viewpoints))));
    });
    group.bench_function("after_single_clone", |b| {
        b.iter(|| black_box(extract_single_clone(black_box(&viewpoints))));
    });
    group.finish();
}

criterion_group!(benches, bench_fusion_pipeline, bench_embedding_extract);
criterion_main!(benches);
