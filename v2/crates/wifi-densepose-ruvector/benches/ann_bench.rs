//! Criterion bench for the ADR-261 graph-ANN index: linear scan vs float HNSW
//! vs quantized HNSW, on the shared `ann_measure` fixture.
//!
//! The authoritative recall/QPS numbers in ADR-261 come from the
//! `--no-default-features --release` test report
//! (`ann_bench_report` in `src/ann_measure.rs`), which is deterministic and
//! gate-runnable. This criterion bench times the same operations through the
//! criterion harness for stable per-op medians:
//!
//! ```text
//! cargo bench -p wifi-densepose-ruvector --bench ann_bench
//! ```
//!
//! Build is excluded from the timed region (done once in setup); only the query
//! path is measured. The fixture and both indices are identical to the report's,
//! so the bench and the report can never measure different graphs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wifi_densepose_ruvector::ann_measure::{build_indices, queries, AnnBenchParams};

fn bench_ann(c: &mut Criterion) {
    // Modest N so the bench builds quickly; the report covers the larger N.
    let p = AnnBenchParams::default_fixture(10_000);
    let (float_idx, quant_idx, _v) = build_indices(p);
    let qs = queries(p);
    let k = p.k;

    let mut group = c.benchmark_group("ann_query");
    group.sample_size(20);

    // Linear scan (brute force) — the no-index baseline.
    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            let mut sink = 0u64;
            for q in &qs {
                sink = sink.wrapping_add(float_idx.brute_force(black_box(q), k).len() as u64);
            }
            black_box(sink)
        })
    });

    // Float HNSW at a mid beam width.
    for &ef in &[64usize, 128] {
        group.bench_function(format!("float_hnsw_ef{ef}"), |b| {
            b.iter(|| {
                let mut sink = 0u64;
                for q in &qs {
                    sink = sink.wrapping_add(float_idx.search(black_box(q), k, ef).len() as u64);
                }
                black_box(sink)
            })
        });
    }

    // Quantized HNSW at matched beam widths + rerank.
    for &ef in &[64usize, 128] {
        let rr = k * 5;
        group.bench_function(format!("quant_hnsw_ef{ef}_rr{rr}"), |b| {
            b.iter(|| {
                let mut sink = 0u64;
                for q in &qs {
                    sink = sink
                        .wrapping_add(quant_idx.search_quantized(black_box(q), k, ef, rr).len() as u64);
                }
                black_box(sink)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ann);
criterion_main!(benches);
