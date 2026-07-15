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
/// and quantized graphs are identical — the only variable is scoring). The
/// quantized index uses the legacy **1-bit** code (ADR-261 §6); use
/// [`build_indices_bits`] for the multi-bit scaling study (§11).
pub fn build_indices(p: AnnBenchParams) -> (HnswIndex, QuantizedHnswIndex, Vec<Vec<f32>>) {
    build_indices_bits(p, 1)
}

/// Build the float HNSW + a `bits`-bit quantized HNSW over the same fixture,
/// sharing the graph seed and insertion order so the *only* variable between the
/// float and quantized search is the traversal score. `bits ∈ {1, 2, 4}` (clamped
/// in [`QuantizedHnswIndex::build_bits`]). The float index is **independent of
/// `bits`** — callers sweeping `bits` should build the float index once and reuse
/// it (the quantized graph is identical across `bits`; only the per-node code
/// changes).
pub fn build_indices_bits(
    p: AnnBenchParams,
    bits: u32,
) -> (HnswIndex, QuantizedHnswIndex, Vec<Vec<f32>>) {
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
    let quant_idx = QuantizedHnswIndex::build_bits(
        &vectors,
        p.dim,
        Metric::L2,
        params,
        p.rot_seed,
        bits,
        p.k * 4,
    );
    (float_idx, quant_idx, vectors)
}

/// Build only the `bits`-bit quantized index for `p`, reusing a fixture the
/// caller already has (avoids regenerating `N×dim` floats per bit-depth in the
/// scaling sweep). The graph seed/insertion order match [`build_indices_bits`],
/// so this quantized graph is identical to that one's at the same `p`.
pub fn build_quant_bits(p: AnnBenchParams, vectors: &[Vec<f32>], bits: u32) -> QuantizedHnswIndex {
    let params = HnswParams {
        m: 16,
        ef_construction: 200,
        ef_search: 64,
        seed: p.graph_seed,
    };
    QuantizedHnswIndex::build_bits(vectors, p.dim, Metric::L2, params, p.rot_seed, bits, p.k * 4)
}

/// The fastest operating point of a method that meets `target` recall, as
/// `(qps, recall, label)`; `None` if no swept op met it.
type BestOp = Option<(f64, f64, String)>;

/// Sweep float HNSW over a fixed `ef` ladder; return the fastest op meeting
/// `target` recall.
pub fn best_float_op(
    idx: &HnswIndex,
    qs: &[Vec<f32>],
    truth: &[HashSet<u32>],
    k: usize,
    target: f64,
) -> BestOp {
    let mut best: BestOp = None;
    for &ef in &[16usize, 32, 64, 128, 256] {
        let r = measure_float_hnsw(idx, qs, truth, k, ef);
        if r.recall >= target && best.as_ref().map(|b| r.qps > b.0).unwrap_or(true) {
            best = Some((r.qps, r.recall, format!("ef={ef}")));
        }
    }
    best
}

/// Sweep quant HNSW over a fixed `(ef, rerank)` ladder; return the fastest op
/// meeting `target` recall, plus the best recall reached anywhere on the ladder
/// (so a not-found verdict can report how close it got).
pub fn best_quant_op(
    qidx: &QuantizedHnswIndex,
    qs: &[Vec<f32>],
    truth: &[HashSet<u32>],
    k: usize,
    target: f64,
) -> (BestOp, f64) {
    let mut best: BestOp = None;
    let mut best_recall_seen = 0.0f64;
    for &ef in &[32usize, 64, 128, 256, 512] {
        for &rr in &[k * 2, k * 5, k * 10, k * 20] {
            let r = measure_quantized_hnsw(qidx, qs, truth, k, ef, rr);
            best_recall_seen = best_recall_seen.max(r.recall);
            if r.recall >= target && best.as_ref().map(|b| r.qps > b.0).unwrap_or(true) {
                best = Some((r.qps, r.recall, format!("ef={ef} rr={rr}")));
            }
        }
    }
    (best, best_recall_seen)
}

/// One row of the ADR-261 §11 scaling study: at a fixed `(N, b)`, the equal-recall
/// (≥ `target`) operating points for float vs quant HNSW and their QPS ratio.
#[derive(Debug, Clone)]
pub struct ScalingRow {
    /// Indexed vector count.
    pub n: usize,
    /// Traversal-code bit-depth (1, 2, or 4).
    pub bits: u32,
    /// Packed bytes per node of the quant code at this `b`.
    pub bytes_per_node: usize,
    /// Fastest float-HNSW op meeting `target` recall (qps, recall, label).
    pub float_op: BestOp,
    /// Fastest quant-HNSW op meeting `target` recall (qps, recall, label).
    pub quant_op: BestOp,
    /// Best recall the quant ladder reached at this `(N, b)` (≤ `target` ⇒ no op).
    pub quant_best_recall: f64,
    /// quant/float QPS ratio at equal recall, if both met `target`.
    pub ratio: Option<f64>,
}

/// Run the ADR-261 §11 multi-bit scaling study: for each `N ∈ ns` and each
/// `b ∈ bits_set`, measure the equal-recall (≥ `target`) QPS ratio of quant-HNSW
/// vs float-HNSW on the shared fixture. Deterministic and `--no-default-features`
/// runnable. Returns one [`ScalingRow`] per `(N, b)`; the caller prints the table
/// and decides the crossover verdict. The float index is built once per `N` and
/// reused across `b` (the quant graph is identical across `b`).
pub fn run_scaling_study(
    base: AnnBenchParams,
    ns: &[usize],
    bits_set: &[u32],
    target: f64,
) -> Vec<ScalingRow> {
    let mut rows = Vec::new();
    for &n in ns {
        let p = AnnBenchParams { n, ..base };
        let (float_idx, _q1, vectors) = build_indices_bits(p, 1);
        let qs = queries(p);
        let truth = ground_truth(&float_idx, &qs, p.k);
        let float_op = best_float_op(&float_idx, &qs, &truth, p.k, target);
        for &b in bits_set {
            let qidx = build_quant_bits(p, &vectors, b);
            let (quant_op, quant_best_recall) =
                best_quant_op(&qidx, &qs, &truth, p.k, target);
            let ratio = match (&float_op, &quant_op) {
                (Some((fqps, _, _)), Some((qqps, _, _))) => Some(qqps / fqps),
                _ => None,
            };
            rows.push(ScalingRow {
                n,
                bits: qidx.bits(),
                bytes_per_node: qidx.bytes_per_node(),
                float_op: float_op.clone(),
                quant_op,
                quant_best_recall,
                ratio,
            });
        }
    }
    rows
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

    /// The ADR-261 §11 **multi-bit scaling study**. Sweeps `N` and `b ∈ {1,2,4}`,
    /// printing the `(N, b) → recall / QPS / quant-vs-float ratio at equal recall`
    /// surface and the crossover verdict. This is the source of truth for the §11
    /// table. Run for the published numbers with:
    ///
    /// ```text
    /// cd v2 && ANN_SCALE_NS=10000,100000,250000 \
    ///   cargo test -p wifi-densepose-ruvector --no-default-features --release \
    ///   scaling_report -- --nocapture --ignored
    /// ```
    ///
    /// Marked `#[ignore]` so the default (debug) gate stays fast: it builds and
    /// queries several indices up to large `N`, which is minutes under `--release`
    /// and far too slow in debug. The CI-safe structural invariants are checked by
    /// `scaling_study_small_is_consistent` below at tiny `N`.
    #[test]
    #[ignore = "scaling study — run explicitly with --release --ignored; minutes at large N"]
    fn scaling_report() {
        // N ladder: default 10k→100k→250k (a clean 25× span that builds+queries in
        // a few minutes under --release on the test box). Override with
        // ANN_SCALE_NS=a,b,c. The largest feasible N is documented in the ADR with
        // the measured build/query time at the cap.
        let ns: Vec<usize> = std::env::var("ANN_SCALE_NS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![10_000, 100_000, 250_000]);
        let bits_set = [1u32, 2, 4];
        let target = 0.90f64;
        let base = AnnBenchParams::default_fixture(ns[0]);

        println!("\n=== ADR-261 §11 multi-bit scaling study (planted-cluster synthetic) ===");
        println!(
            "dim={} clusters={} queries={} K={} noise={} graph_seed=0x{:X} rot_seed=0x{:X}",
            base.dim, base.clusters, base.n_queries, base.k, base.noise, base.graph_seed, base.rot_seed
        );
        println!("metric=L2  M=16 ef_construction=200  target recall >= {target:.2}  (use --release for QPS)");
        println!(
            "{:<9} {:>4} {:>9} {:>10} {:>22} {:>22} {:>12}",
            "N", "bits", "B/node", "q_best_rec", "float@target", "quant@target", "quant/float"
        );

        let rows = run_scaling_study(base, &ns, &bits_set, target);
        for row in &rows {
            let float_s = row
                .float_op
                .as_ref()
                .map(|(q, r, l)| format!("{l} {q:.0}QPS r={r:.3}"))
                .unwrap_or_else(|| "none".to_string());
            let quant_s = row
                .quant_op
                .as_ref()
                .map(|(q, r, l)| format!("{l} {q:.0}QPS r={r:.3}"))
                .unwrap_or_else(|| "none".to_string());
            let ratio_s = row
                .ratio
                .map(|x| format!("{x:.2}x"))
                .unwrap_or_else(|| "—".to_string());
            println!(
                "{:<9} {:>4} {:>9} {:>10.3} {:>22} {:>22} {:>12}",
                row.n, row.bits, row.bytes_per_node, row.quant_best_recall, float_s, quant_s, ratio_s
            );
        }

        // Crossover verdict: report whether the quant/float ratio EVER exceeds 1.0
        // at equal recall, and the per-bit trend of the best-quant-recall as N grows
        // (is quant getting closer to the equal-recall regime, or not).
        println!("\n--- crossover verdict (quant-HNSW > float-HNSW at equal recall?) ---");
        let crossover: Vec<&ScalingRow> = rows
            .iter()
            .filter(|r| r.ratio.map(|x| x > 1.0).unwrap_or(false))
            .collect();
        if crossover.is_empty() {
            println!("NO crossover at any measured (N, b): quant never met target recall AND beat float QPS.");
        } else {
            for r in &crossover {
                println!(
                    "CROSSOVER at N={} b={}: quant/float = {:.2}x at recall >= {target:.2}",
                    r.n, r.bits, r.ratio.unwrap()
                );
            }
        }
        for &b in &bits_set {
            let trend: Vec<(usize, f64)> = rows
                .iter()
                .filter(|r| r.bits == b)
                .map(|r| (r.n, r.quant_best_recall))
                .collect();
            let trend_s: Vec<String> = trend
                .iter()
                .map(|(n, r)| format!("N={n}:{r:.3}"))
                .collect();
            println!("b={b} best-quant-recall trend: {}", trend_s.join("  "));
        }
        println!("======================================================================\n");

        // Structural invariants (gate-safe at any N): at least one float op met
        // target at every N (the baseline must work), and quant recall is in range.
        for &n in &ns {
            let any_float = rows.iter().any(|r| r.n == n && r.float_op.is_some());
            assert!(any_float, "no float-HNSW op met target recall at N={n} — baseline broken");
        }
        for r in &rows {
            assert!(
                (0.0..=1.0).contains(&r.quant_best_recall),
                "quant recall out of range at N={} b={}: {}",
                r.n,
                r.bits,
                r.quant_best_recall
            );
        }
    }

    /// CI-safe structural check for the scaling study at tiny `N` (debug-fast):
    /// the study runs end-to-end, bytes/node scales with `b`, and the float
    /// baseline meets target at the smallest N. Does **not** assert any crossover
    /// (that is the §11 measured question, answered by `scaling_report`).
    #[test]
    fn scaling_study_small_is_consistent() {
        let base = AnnBenchParams::default_fixture(1500);
        let ns = [1500usize, 3000];
        let bits_set = [1u32, 2, 4];
        let rows = run_scaling_study(base, &ns, &bits_set, 0.90);
        assert_eq!(rows.len(), ns.len() * bits_set.len());
        // Bytes/node scales with b at dim=128 (D=128): 16 / 32 / 64.
        for r in rows.iter().filter(|r| r.n == 1500) {
            let expect = match r.bits {
                1 => 16,
                2 => 32,
                _ => 64,
            };
            assert_eq!(r.bytes_per_node, expect, "B/node wrong for b={}", r.bits);
        }
        // Float baseline must meet target at the smallest N.
        assert!(
            rows.iter().any(|r| r.n == 1500 && r.float_op.is_some()),
            "float baseline failed target at small N"
        );
    }
}
