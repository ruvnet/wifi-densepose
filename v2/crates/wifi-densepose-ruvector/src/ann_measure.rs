//! Deterministic, `--no-default-features`-runnable **ANN benchmark measurement**
//! for ADR-261 — the single source of truth for the QPS/recall numbers the ADR
//! quotes for **linear scan**, **float HNSW**, and **quantized HNSW**.
//!
//! Both the criterion bench (`benches/ann_bench.rs`) and the in-crate report test
//! ([`tests::ann_bench_report`]) call into here, so they can never silently
//! measure different things. The numbers in ADR-261 §6 come from running:
//!
//! ```text
//! cd v2 && cargo test -p wifi-densepose-ruvector --no-default-features --release \
//!   ann_bench_report -- --nocapture
//! ```
//!
//! # What is measured, and the honesty contract
//!
//! On one fixed planted-cluster fixture (documented dim/N/K/seed), for each
//! method we measure:
//! - **recall@10** vs the brute-force exact top-10 (the ground truth),
//! - **QPS** = queries / total wall-clock query time (warm; build excluded),
//! at matched recall operating points found by sweeping `ef` (HNSW) and
//! `(ef, rerank)` (quantized).
//!
//! The reported **ratio** is the claim, not the absolute QPS (which is
//! machine-specific). We do **not** tune the quantized path to manufacture a
//! win: if at our scale quantized does not beat float HNSW, the report says so
//! and the ADR records the honest negative + the expected larger-N crossover.

use std::collections::HashSet;
use std::time::Instant;

use crate::hnsw::{HnswIndex, HnswParams, Metric};
use crate::hnsw_quantized::QuantizedHnswIndex;

/// SplitMix64 — the crate-wide deterministic PRNG (mirrors `coverage.rs`).
#[inline]
fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
#[inline]
fn unif01(state: &mut u64) -> f32 {
    ((split_mix64(state) >> 40) as f32) / ((1u64 << 24) as f32)
}
#[inline]
fn gauss(state: &mut u64) -> f32 {
    let u1 = unif01(state).max(1e-7);
    let u2 = unif01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// ANN benchmark fixture parameters, documented in the ADR-261 report.
#[derive(Debug, Clone, Copy)]
pub struct AnnBenchParams {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of indexed vectors (N).
    pub n: usize,
    /// Number of planted clusters (near-neighbour structure).
    pub clusters: usize,
    /// Number of queries timed.
    pub n_queries: usize,
    /// Top-K.
    pub k: usize,
    /// Intra-cluster Gaussian jitter.
    pub noise: f32,
    /// Master fixture seed.
    pub seed: u64,
    /// Graph construction/level seed.
    pub graph_seed: u64,
    /// Rotation seed for the quantized 1-bit codes.
    pub rot_seed: u64,
}

impl AnnBenchParams {
    /// The default ADR-261 fixture: AETHER-shape 128-d, planted clusters.
    pub fn default_fixture(n: usize) -> Self {
        Self {
            dim: 128,
            n,
            clusters: 64,
            n_queries: 200,
            k: 10,
            noise: 0.35,
            seed: 0xADADADAD_0000_0261,
            graph_seed: 0x6261_5247_4148_4E53,
            rot_seed: 0x5EED_C0DE_1234_5678,
        }
    }
}

/// The fixture vectors for `p` (deterministic planted clusters).
pub fn fixture(p: AnnBenchParams) -> Vec<Vec<f32>> {
    let centres: Vec<Vec<f32>> = (0..p.clusters)
        .map(|c| {
            let mut s = p.seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
            (0..p.dim).map(|_| gauss(&mut s) * 3.0).collect()
        })
        .collect();
    (0..p.n)
        .map(|i| {
            let c = i % p.clusters;
            let mut s = p.seed ^ (i as u64).wrapping_mul(0x9E37);
            (0..p.dim)
                .map(|d| centres[c][d] + gauss(&mut s) * p.noise)
                .collect()
        })
        .collect()
}

/// The timed query set for `p` (drawn from the same clusters, disjoint seed).
pub fn queries(p: AnnBenchParams) -> Vec<Vec<f32>> {
    let centres: Vec<Vec<f32>> = (0..p.clusters)
        .map(|c| {
            let mut s = p.seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
            (0..p.dim).map(|_| gauss(&mut s) * 3.0).collect()
        })
        .collect();
    (0..p.n_queries)
        .map(|q| {
            let c = q % p.clusters;
            let mut s = p.seed ^ 0xDEAD_0000_0000 ^ (q as u64).wrapping_mul(0x2545_F491);
            (0..p.dim)
                .map(|d| centres[c][d] + gauss(&mut s) * p.noise)
                .collect()
        })
        .collect()
}

/// Per-method measurement: recall@K and QPS.
#[derive(Debug, Clone, Copy)]
pub struct MethodResult {
    /// Mean recall@K vs brute-force ground truth.
    pub recall: f64,
    /// Queries per second (warm wall-clock).
    pub qps: f64,
    /// Mean query latency in microseconds.
    pub latency_us: f64,
}

/// Ground-truth brute-force top-K id sets for every query (computed once).
/// Public so the criterion bench and the report test share one definition.
pub fn ground_truth(idx: &HnswIndex, queries: &[Vec<f32>], k: usize) -> Vec<HashSet<u32>> {
    queries
        .iter()
        .map(|q| idx.brute_force(q, k).into_iter().map(|(id, _)| id).collect())
        .collect()
}

/// Measure **linear scan** (brute force): recall is 1.0 by definition; QPS is the
/// timed exact scan. This is the no-index baseline.
pub fn measure_linear(
    idx: &HnswIndex,
    queries: &[Vec<f32>],
    truth: &[HashSet<u32>],
    k: usize,
) -> MethodResult {
    let mut recall_acc = 0.0f64;
    let start = Instant::now();
    let mut sink = 0u64;
    for (qi, q) in queries.iter().enumerate() {
        let got = idx.brute_force(q, k);
        let hit = got.iter().filter(|(id, _)| truth[qi].contains(id)).count();
        recall_acc += hit as f64 / k as f64;
        sink = sink.wrapping_add(got.len() as u64);
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    MethodResult {
        recall: recall_acc / queries.len() as f64,
        qps: queries.len() as f64 / elapsed,
        latency_us: elapsed / queries.len() as f64 * 1e6,
    }
}

/// Measure **float HNSW** at a given beam width `ef`.
pub fn measure_float_hnsw(
    idx: &HnswIndex,
    queries: &[Vec<f32>],
    truth: &[HashSet<u32>],
    k: usize,
    ef: usize,
) -> MethodResult {
    let mut recall_acc = 0.0f64;
    let start = Instant::now();
    let mut sink = 0u64;
    for (qi, q) in queries.iter().enumerate() {
        let got = idx.search(q, k, ef);
        let hit = got.iter().filter(|(id, _)| truth[qi].contains(id)).count();
        recall_acc += hit as f64 / k as f64;
        sink = sink.wrapping_add(got.len() as u64);
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    MethodResult {
        recall: recall_acc / queries.len() as f64,
        qps: queries.len() as f64 / elapsed,
        latency_us: elapsed / queries.len() as f64 * 1e6,
    }
}

/// Measure **quantized HNSW** at a given `(ef, rerank)`.
pub fn measure_quantized_hnsw(
    qidx: &QuantizedHnswIndex,
    queries: &[Vec<f32>],
    truth: &[HashSet<u32>],
    k: usize,
    ef: usize,
    rerank: usize,
) -> MethodResult {
    let mut recall_acc = 0.0f64;
    let start = Instant::now();
    let mut sink = 0u64;
    for (qi, q) in queries.iter().enumerate() {
        let got = qidx.search_quantized(q, k, ef, rerank);
        let hit = got.iter().filter(|(id, _)| truth[qi].contains(id)).count();
        recall_acc += hit as f64 / k as f64;
        sink = sink.wrapping_add(got.len() as u64);
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    MethodResult {
        recall: recall_acc / queries.len() as f64,
        qps: queries.len() as f64 / elapsed,
        latency_us: elapsed / queries.len() as f64 * 1e6,
    }
}

/// Build both indices for `p` (shared insertion order + graph seed so the float
/// and quantized graphs are identical — the only variable is scoring).
pub fn build_indices(p: AnnBenchParams) -> (HnswIndex, QuantizedHnswIndex, Vec<Vec<f32>>) {
    let vectors = fixture(p);
    let params = HnswParams {
        m: 16,
        ef_construction: 200,
        ef_search: 64,
        seed: p.graph_seed,
    };
    let mut float_idx = HnswIndex::new(p.dim, Metric::L2, params);
    for v in &vectors {
        float_idx.insert(v);
    }
    let quant_idx =
        QuantizedHnswIndex::build(&vectors, p.dim, Metric::L2, params, p.rot_seed, p.k * 4);
    (float_idx, quant_idx, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_and_queries_are_deterministic() {
        let p = AnnBenchParams::default_fixture(500);
        assert_eq!(fixture(p), fixture(p));
        assert_eq!(queries(p), queries(p));
        let p2 = AnnBenchParams {
            seed: p.seed ^ 1,
            ..p
        };
        assert_ne!(fixture(p)[0], fixture(p2)[0]);
    }

    #[test]
    fn linear_recall_is_one() {
        // Linear scan IS the ground truth, so recall must be exactly 1.0.
        let p = AnnBenchParams::default_fixture(800);
        let (float_idx, _q, _v) = build_indices(p);
        let qs = queries(p);
        let truth = ground_truth(&float_idx, &qs, p.k);
        let r = measure_linear(&float_idx, &qs, &truth, p.k);
        assert!((r.recall - 1.0).abs() < 1e-9, "linear recall {} != 1.0", r.recall);
        assert!(r.qps > 0.0);
    }

    /// The ADR-261 measurement report. Prints the linear / float-HNSW /
    /// quantized-HNSW recall@10 + QPS table and the QPS ratios at matched recall.
    /// Run with `--release --nocapture` for the numbers the ADR quotes.
    #[test]
    fn ann_bench_report() {
        // N here is the small/CI-friendly default so the standard (debug) test
        // gate stays fast; the ADR's headline numbers are taken at the larger N
        // under --release (documented in the ADR with the exact command). This
        // test asserts only structural invariants so it is gate-safe at any N.
        let n: usize = std::env::var("ANN_BENCH_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let p = AnnBenchParams::default_fixture(n);
        let (float_idx, quant_idx, _v) = build_indices(p);
        let qs = queries(p);
        let truth = ground_truth(&float_idx, &qs, p.k);

        println!("\n=== ADR-261 ANN benchmark (planted-cluster synthetic) ===");
        println!(
            "dim={} N={} clusters={} queries={} K={} noise={} graph_seed=0x{:X} rot_seed=0x{:X}",
            p.dim, p.n, p.clusters, p.n_queries, p.k, p.noise, p.graph_seed, p.rot_seed
        );
        println!("metric=L2  M=16 ef_construction=200  (debug build unless --release)");
        println!(
            "{:<28} {:>9} {:>12} {:>12}",
            "method", "recall@10", "QPS", "lat(us)"
        );

        let lin = measure_linear(&float_idx, &qs, &truth, p.k);
        println!(
            "{:<28} {:>8.4} {:>12.1} {:>12.1}",
            "linear scan (brute)", lin.recall, lin.qps, lin.latency_us
        );

        // Float HNSW across an ef sweep.
        let mut float_ops: Vec<(usize, MethodResult)> = Vec::new();
        for &ef in &[16usize, 32, 64, 128, 256] {
            let r = measure_float_hnsw(&float_idx, &qs, &truth, p.k, ef);
            println!(
                "{:<28} {:>8.4} {:>12.1} {:>12.1}",
                format!("float-HNSW ef={ef}"),
                r.recall,
                r.qps,
                r.latency_us
            );
            float_ops.push((ef, r));
        }

        // Quantized HNSW across (ef, rerank) sweep.
        let mut quant_ops: Vec<((usize, usize), MethodResult)> = Vec::new();
        for &ef in &[32usize, 64, 128, 256] {
            for &rr in &[p.k * 2, p.k * 5, p.k * 10] {
                let r = measure_quantized_hnsw(&quant_idx, &qs, &truth, p.k, ef, rr);
                println!(
                    "{:<28} {:>8.4} {:>12.1} {:>12.1}",
                    format!("quant-HNSW ef={ef} rr={rr}"),
                    r.recall,
                    r.qps,
                    r.latency_us
                );
                quant_ops.push(((ef, rr), r));
            }
        }

        // Equal-recall comparison: pick, for a target recall, the FASTEST op of
        // each method that meets it, then report the QPS ratios.
        println!("\n--- equal-recall QPS ratios ---");
        for &target in &[0.90f64, 0.95, 0.99] {
            let best_float = float_ops
                .iter()
                .filter(|(_, r)| r.recall >= target)
                .max_by(|a, b| a.1.qps.partial_cmp(&b.1.qps).unwrap());
            let best_quant = quant_ops
                .iter()
                .filter(|(_, r)| r.recall >= target)
                .max_by(|a, b| a.1.qps.partial_cmp(&b.1.qps).unwrap());
            match (best_float, best_quant) {
                (Some((fef, fr)), Some(((qef, qrr), qr))) => {
                    let ratio = qr.qps / fr.qps;
                    let hnsw_vs_lin = fr.qps / lin.qps;
                    println!(
                        "recall>={:.2}: float ef={} {:.0} QPS | quant ef={} rr={} {:.0} QPS | quant/float={:.2}x | float/linear={:.2}x",
                        target, fef, fr.qps, qef, qrr, qr.qps, ratio, hnsw_vs_lin
                    );
                }
                (Some((fef, fr)), None) => {
                    let hnsw_vs_lin = fr.qps / lin.qps;
                    println!(
                        "recall>={:.2}: float ef={} {:.0} QPS | quant: NO op met this recall | float/linear={:.2}x",
                        target, fef, fr.qps, hnsw_vs_lin
                    );
                }
                _ => {
                    println!("recall>={:.2}: neither method met this recall at the swept ops", target);
                }
            }
        }
        println!("=========================================================\n");

        // Structural assertions (gate-safe, any N):
        // - linear scan is exact,
        // - the best float-HNSW op clears the correctness gate,
        // - quantized's best op is at least useful (recall well above random).
        assert!((lin.recall - 1.0).abs() < 1e-9);
        let best_float_recall = float_ops.iter().map(|(_, r)| r.recall).fold(0.0, f64::max);
        assert!(
            best_float_recall >= 0.95,
            "best float-HNSW recall {best_float_recall:.4} below 0.95 gate"
        );
        let best_quant_recall = quant_ops.iter().map(|(_, r)| r.recall).fold(0.0, f64::max);
        // Honest floor: the 1-bit Hamming traversal is a COARSE angle proxy, so
        // at large N its best recall lands well below the float gate (MEASURED
        // ~0.74 at N=10k — see ADR-261 §6). We assert only that it is clearly
        // useful (>> random: random top-10 of N=10k is ~0.001), which catches a
        // fully-broken traversal/rerank without pretending the quantized variant
        // matches float HNSW. The honest negative IS the result.
        assert!(
            best_quant_recall >= 0.30,
            "best quant-HNSW recall {best_quant_recall:.4} below the 0.30 not-broken floor"
        );
    }
}
