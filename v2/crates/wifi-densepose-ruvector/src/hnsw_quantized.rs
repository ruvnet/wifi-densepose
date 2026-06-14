//! A **SymphonyQG-style quantized-traversal HNSW** — ADR-261 (multi-bit, §11).
//!
//! # The SymphonyQG bet (what we are testing)
//!
//! [SymphonyQG (SIGMOD 2025)](../../../../../docs/adr/ADR-261-ruvector-graph-ann-index.md)
//! unifies **quantization with graph traversal**: instead of computing the full
//! float distance at every node the beam search visits (the cost that dominates
//! float HNSW — one `O(d)` float dot/diff per visited node), it scores traversal
//! candidates with a **cheap quantized distance** and only computes the exact
//! float distance for the *final* candidate set, which it **reranks**. The bet:
//! the quantized score is cheap enough — and accurate enough to keep the beam on
//! the right path — that you visit roughly as many nodes but pay far less per
//! node, and recover the small recall loss with a final exact rerank. Source
//! reports **3.5–17× QPS over HNSW at equal recall**.
//!
//! # Our implementation (honest scope)
//!
//! We are **not** reproducing SymphonyQG's exact system (their RaBitQ-fused codes,
//! their SIMD layout, their refined graph). We build the **direction** of the
//! claim from the pieces this crate already has, so the comparison is
//! apples-to-apples on *our* hardware:
//!
//! - **Same graph** as the float [`crate::HnswIndex`] — identical structure,
//!   identical seed, identical level assignment. The *only* variable between the
//!   float and quantized search is **how a candidate is scored during traversal**,
//!   so any QPS/recall difference is attributable to the quantization, not to a
//!   different graph.
//! - **Quantized score = `b`-bit code over the RaBitQ Pass-2 rotated coordinates**
//!   ([`crate::rotation`] + the multi-bit scalar quantizer mirrored from
//!   [ADR-156 §10](../../../../../docs/adr/ADR-156-ruvector-fusion-beyond-sota.md)'s
//!   `coverage::measure_multibit`). Each node stores a `b`-bit-per-dimension code
//!   over the padded rotation length `D = next_pow2(dim)`. During traversal we
//!   compare query-code vs node-code by the **L1 distance over the per-dim
//!   codes** — a few machine words of integer work, no per-dimension float work.
//!   For `b == 1` the codes are `{0, 1}` and the L1 distance is **exactly the
//!   1-bit Hamming distance** of the original ADR-261 construction, so `b == 1`
//!   is fully backward-compatible.
//! - **Exact float rerank** of the final beam: the top `rerank` candidates by
//!   code-L1 are re-scored with the true float metric and the best `k` returned.
//!
//! Higher `b` keeps the traversal beam on-path better than 1-bit (ADR-156 §10
//! measured 1/2/3/4-bit strict-K coverage at ~46/54/67/74%), at a memory cost
//! that scales linearly with `b` (bytes/node = `ceil(D·b/8)`). **Whether the
//! extra bits net a QPS win at equal recall — and at what N a crossover with
//! float HNSW appears, if any — is the measured question ADR-261 §11 answers.**
//! We report the real number, win or lose, and do not tune to manufacture a
//! speedup.
//!
//! # Determinism & robustness
//!
//! The graph seed drives everything (level assignment), so the quantized index
//! is as reproducible as the float one. Empty/degenerate inputs are guarded
//! exactly as in [`crate::hnsw`] — no panic on empty index, `k > n`, `k == 0`,
//! single node, ragged query, or zero dim.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::hnsw::{HnswIndex, HnswParams, Metric};
use crate::rotation::Rotation;

/// Symmetric clamp range for the uniform mid-rise scalar quantizer, in rotated-
/// coordinate units. The normalized FHT (`1/√D`) puts AETHER-shape rotated
/// coordinates roughly in `[-3, 3]`; out-of-range coords clamp to the end codes.
/// This is the **same `RANGE = 3.0`** as ADR-156 §10's `coverage::measure_multibit`,
/// so the multi-bit code here is the same scheme that module measured.
const RANGE: f32 = 3.0;

/// A `b`-bit-per-dimension scalar code of a rotated embedding over the padded
/// length `D`, compared by per-dim L1.
///
/// For `bits == 1` the per-dim code is `{0, 1}` (sign), and L1 over those codes
/// is exactly POPCNT Hamming — so the 1-bit case is bit-for-bit the original
/// ADR-261 construction. For `bits ∈ {2, 4}` the code is a uniform mid-rise
/// quantizer with `2^bits` levels over `[-RANGE, RANGE]`.
#[derive(Debug, Clone)]
struct Code {
    /// Per-dimension codes (`0..2^bits`), one entry per padded dimension `D`.
    /// Kept unpacked as `u8` for branch-free L1; the *reported* memory cost is
    /// the packed footprint (`ceil(D·bits/8)`), since a production node would
    /// store the packed form. (We measure the packed bytes/node explicitly in
    /// [`QuantizedHnswIndex::bytes_per_node`].)
    codes: Vec<u8>,
}

impl Code {
    /// L1 distance over the per-dimension codes — the multi-bit generalization
    /// of Hamming. At `bits == 1` (codes in `{0,1}`) this equals the popcount of
    /// the XOR, i.e. the 1-bit Hamming distance.
    #[inline]
    fn l1(&self, other: &Code) -> u32 {
        let n = self.codes.len().min(other.codes.len());
        let mut acc = 0u32;
        for i in 0..n {
            acc += (self.codes[i] as i32 - other.codes[i] as i32).unsigned_abs();
        }
        acc
    }
}

/// Quantize the rotated coordinates of `embedding` to a `bits`-bit-per-dimension
/// [`Code`] over the padded rotation length `D = rotation.padded_dim()`.
///
/// `bits == 1` reduces to sign-quantization (code `1` iff the rotated coord ≥ 0),
/// preserving the original 1-bit construction; `bits ∈ {2, 4}` uses a uniform
/// mid-rise quantizer with `2^bits` levels over `[-RANGE, RANGE]`, identical to
/// ADR-156 §10's `measure_multibit`.
fn encode(embedding: &[f32], rotation: &Rotation, bits: u32) -> Code {
    let rotated = rotation.apply_padded(embedding);
    let levels = 1u32 << bits; // 2^bits codes per dim
    let codes: Vec<u8> = rotated
        .iter()
        .map(|&x| {
            if bits == 1 {
                // Sign code: identical to the original 1-bit construction.
                u8::from(x >= 0.0)
            } else {
                let t = ((x + RANGE) / (2.0 * RANGE)).clamp(0.0, 1.0); // → [0,1]
                let code = (t * (levels - 1) as f32).round() as u32;
                code.min(levels - 1) as u8
            }
        })
        .collect();
    Code { codes }
}

/// Packed bytes a node's `bits`-bit code occupies over padded length `D`:
/// `ceil(D·bits/8)`. The memory cost reported by ADR-261 §11 (1-bit → `D/8`,
/// 2-bit → `D/4`, 4-bit → `D/2`).
#[inline]
fn packed_bytes(padded_dim: usize, bits: u32) -> usize {
    (padded_dim * bits as usize).div_ceil(8)
}

/// Min-heap node for the quantized beam (closest code-L1 at the top).
#[derive(Debug, Clone, Copy)]
struct HScored {
    /// Code-L1 distance (quantized score) — the traversal key.
    dist: u32,
    id: u32,
}
impl PartialEq for HScored {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for HScored {}
impl Ord for HScored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for HScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
/// Reversed wrapper for a min-heap (smallest code-L1 at the top).
#[derive(Debug, Clone, Copy)]
struct MinH(HScored);
impl PartialEq for MinH {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinH {}
impl Ord for MinH {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}
impl PartialOrd for MinH {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A SymphonyQG-style HNSW: the same graph as [`HnswIndex`], traversed by a
/// **cheap `b`-bit code-L1 score**, with a final **exact-float rerank**.
///
/// Built by inserting the same vectors in the same order with the same seed as
/// a float [`HnswIndex`], so the two indices share identical graph structure and
/// only differ in how the beam is scored. The shared [`Rotation`] (seed + dim)
/// is the index/query frame for the `b`-bit codes. `bits ∈ {1, 2, 4}` selects
/// the traversal-code resolution; `bits == 1` is the original 1-bit Hamming
/// construction.
#[derive(Debug, Clone)]
pub struct QuantizedHnswIndex {
    /// The underlying graph (built with the float metric for exact rerank).
    graph: HnswIndex,
    /// Per-node `b`-bit codes, indexed by id (parallel to graph vectors).
    codes: Vec<Code>,
    /// The rotation frame shared by index and query codes.
    rotation: Rotation,
    /// Bits per dimension of the traversal code (`1`, `2`, or `4`).
    bits: u32,
    /// Number of final candidates to exact-float rerank (≥ k at query time).
    default_rerank: usize,
}

impl QuantizedHnswIndex {
    /// Build a 1-bit quantized index (the original ADR-261 construction).
    ///
    /// Equivalent to [`QuantizedHnswIndex::build_bits`] with `bits = 1`; kept as
    /// the backward-compatible entry point so existing callers and tests are
    /// unchanged.
    pub fn build(
        vectors: &[Vec<f32>],
        dim: usize,
        metric: Metric,
        params: HnswParams,
        rotation_seed: u64,
        default_rerank: usize,
    ) -> Self {
        Self::build_bits(vectors, dim, metric, params, rotation_seed, 1, default_rerank)
    }

    /// Build a `bits`-bit quantized index over `vectors`, mirroring a float
    /// [`HnswIndex`] built with the same `(dim, metric, params)` and insertion
    /// order. The `rotation_seed` fixes the code frame (index and query share it).
    ///
    /// `bits` is clamped to `{1, 2, 4}` (the resolutions ADR-261 §11 sweeps): any
    /// other value is rounded up to the nearest of these so the constructor is
    /// total. `default_rerank` is how many top-code-L1 candidates get an exact
    /// float re-score before returning the best `k`; it is clamped to `≥ k` at
    /// query time. A larger rerank recovers more recall at more float cost — the
    /// knob that, alongside `ef`, sets the equal-recall operating point.
    pub fn build_bits(
        vectors: &[Vec<f32>],
        dim: usize,
        metric: Metric,
        params: HnswParams,
        rotation_seed: u64,
        bits: u32,
        default_rerank: usize,
    ) -> Self {
        let bits = clamp_bits(bits);
        let rotation = Rotation::new(rotation_seed, dim);
        let mut graph = HnswIndex::new(dim, metric, params);
        let mut codes = Vec::with_capacity(vectors.len());
        for v in vectors {
            graph.insert(v);
            codes.push(encode(v, &rotation, bits));
        }
        Self {
            graph,
            codes,
            rotation,
            bits,
            default_rerank: default_rerank.max(1),
        }
    }

    /// Number of indexed points.
    #[inline]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// True iff empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Borrow the underlying float graph (for shared-graph benchmark parity:
    /// the float-HNSW baseline runs on *this* graph so the only variable is
    /// scoring).
    #[inline]
    pub fn graph(&self) -> &HnswIndex {
        &self.graph
    }

    /// The rerank width this index defaults to.
    #[inline]
    pub fn default_rerank(&self) -> usize {
        self.default_rerank
    }

    /// Bits per dimension of the traversal code.
    #[inline]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Packed memory footprint of one node's traversal code, in bytes:
    /// `ceil(D·bits/8)` where `D = next_pow2(dim)` is the padded rotation length.
    /// This is the per-node cost ADR-261 §11 reports for each `b`.
    #[inline]
    pub fn bytes_per_node(&self) -> usize {
        packed_bytes(self.rotation.padded_dim(), self.bits)
    }

    /// SymphonyQG-style search: traverse the graph scoring candidates by the
    /// **`b`-bit code-L1**, collect a beam of `ef`, then **exact-float rerank**
    /// the top `rerank` (clamped ≥ k) and return the best `k` as `(id, float_dist)`.
    ///
    /// Degenerate cases mirror [`HnswIndex::search`]: empty ⇒ empty; `k == 0` ⇒
    /// empty; `k > n` ⇒ all; never panics.
    pub fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank: usize,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let ef = ef.max(k).max(1);
        let rerank = rerank.max(k);
        let q_code = encode(query, &self.rotation, self.bits);

        // Entry point: the graph's entry (highest-level node).
        let entry = match self.graph.entry_point() {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Greedy-descend upper layers by code-L1, then beam-search layer 0.
        let mut ep = entry;
        let mut layer = self.graph.top_level();
        while layer > 0 {
            ep = self.greedy_code(&q_code, ep, layer);
            layer -= 1;
        }
        let beam = self.beam_code(&q_code, ep, ef);

        // Exact-float rerank of the top `rerank` code-L1 candidates.
        let mut cand: Vec<HScored> = beam;
        cand.sort_by_key(|c| c.dist);
        cand.truncate(rerank);
        let mut reranked: Vec<(u32, f32)> = cand
            .iter()
            .filter_map(|c| {
                self.graph
                    .vector(c.id)
                    .map(|v| (c.id, self.graph.metric().distance(query, v)))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Search using the index's default `ef` (from graph params) and rerank.
    #[inline]
    pub fn search_default(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_quantized(query, k, self.graph.params_ef_search(), self.default_rerank)
    }

    /// Greedy single-best descent on a layer scored by code-L1.
    fn greedy_code(&self, q_code: &Code, start: u32, layer: usize) -> u32 {
        let mut best = start;
        let mut best_d = self.codes[best as usize].l1(q_code);
        loop {
            let mut improved = false;
            for &nbr in self.graph.neighbours(best, layer) {
                let d = self.codes[nbr as usize].l1(q_code);
                if d < best_d {
                    best_d = d;
                    best = nbr;
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Beam search on layer 0 scored by code-L1. Returns the `ef` best-code nodes
    /// (unsorted). Iterative — bounded by the visited set + the ef beam.
    fn beam_code(&self, q_code: &Code, ep: u32, ef: usize) -> Vec<HScored> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<MinH> = BinaryHeap::new();
        let mut results: BinaryHeap<HScored> = BinaryHeap::new(); // max-heap: worst at top

        let d0 = self.codes[ep as usize].l1(q_code);
        let s0 = HScored { dist: d0, id: ep };
        visited.insert(ep);
        candidates.push(MinH(s0));
        results.push(s0);

        while let Some(MinH(cur)) = candidates.pop() {
            let worst = results.peek().map(|s| s.dist).unwrap_or(u32::MAX);
            if cur.dist > worst && results.len() >= ef {
                break;
            }
            for &nbr in self.graph.neighbours(cur.id, 0) {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.codes[nbr as usize].l1(q_code);
                let worst = results.peek().map(|s| s.dist).unwrap_or(u32::MAX);
                if results.len() < ef || d < worst {
                    let s = HScored { dist: d, id: nbr };
                    candidates.push(MinH(s));
                    results.push(s);
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_vec()
    }
}

/// Clamp a requested bit-depth to the supported `{1, 2, 4}` set (round up to the
/// nearest supported value; `0` → `1`, `3` → `4`, `> 4` → `4`).
#[inline]
fn clamp_bits(bits: u32) -> u32 {
    match bits {
        0 | 1 => 1,
        2 => 2,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_mix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif01(state: &mut u64) -> f32 {
        ((split_mix64(state) >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn gauss(state: &mut u64) -> f32 {
        let u1 = unif01(state).max(1e-7);
        let u2 = unif01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    fn planted(dim: usize, n: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        let centres: Vec<Vec<f32>> = (0..clusters)
            .map(|c| {
                let mut s = seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
                (0..dim).map(|_| gauss(&mut s) * 3.0).collect()
            })
            .collect();
        (0..n)
            .map(|i| {
                let c = i % clusters;
                let mut s = seed ^ (i as u64).wrapping_mul(0x9E37);
                (0..dim).map(|d| centres[c][d] + gauss(&mut s) * 0.35).collect()
            })
            .collect()
    }
    fn params(seed: u64) -> HnswParams {
        HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed,
        }
    }

    #[test]
    fn empty_quantized_search_is_empty_no_panic() {
        let idx = QuantizedHnswIndex::build(&[], 8, Metric::Cosine, params(1), 0x42, 16);
        assert!(idx.is_empty());
        assert!(idx.search_quantized(&[0.0; 8], 5, 16, 16).is_empty());
    }

    #[test]
    fn single_node_quantized_returns_itself() {
        let v = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let idx = QuantizedHnswIndex::build(&v, 4, Metric::L2, params(2), 0x7, 8);
        let r = idx.search_quantized(&v[0], 3, 16, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn k_zero_and_k_gt_n_no_panic() {
        let vectors = planted(16, 40, 4, 0xABCD);
        let idx = QuantizedHnswIndex::build(&vectors, 16, Metric::L2, params(3), 0x9, 32);
        assert!(idx.search_quantized(&vectors[0], 0, 16, 16).is_empty());
        let r = idx.search_quantized(&vectors[0], 1000, 64, 64);
        assert_eq!(r.len(), 40);
    }

    #[test]
    fn ragged_query_no_panic() {
        let vectors = planted(16, 30, 3, 0x55);
        let idx = QuantizedHnswIndex::build(&vectors, 16, Metric::Cosine, params(4), 0xB, 16);
        assert!(!idx.search_quantized(&[1.0, 2.0, 3.0], 3, 16, 16).is_empty());
        let long: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert!(!idx.search_quantized(&long, 3, 16, 16).is_empty());
    }

    #[test]
    fn quantized_is_deterministic() {
        let vectors = planted(32, 300, 8, 0x2468);
        let a = QuantizedHnswIndex::build(&vectors, 32, Metric::Cosine, params(0xFEED), 0xC0DE, 32);
        let b = QuantizedHnswIndex::build(&vectors, 32, Metric::Cosine, params(0xFEED), 0xC0DE, 32);
        let q = &vectors[100];
        assert_eq!(
            a.search_quantized(q, 10, 64, 32),
            b.search_quantized(q, 10, 64, 32),
            "quantized search must be deterministic"
        );
    }

    /// Recall@10 of quantized-HNSW vs brute-force ground truth, averaged over
    /// queries. With an exact-float rerank, recall should be high (the rerank
    /// repairs most of the 1-bit traversal's coarseness). This is the quantized
    /// variant's correctness gate.
    #[test]
    fn quantized_recall_at_10_is_high_with_rerank() {
        let dim = 64;
        let n = 2000;
        let clusters = 32;
        let seed = 0x9999;
        let vectors = planted(dim, n, clusters, seed);
        // Generous rerank so the exact float repairs the coarse Hamming beam.
        let idx = QuantizedHnswIndex::build(&vectors, dim, Metric::L2, params(0xAAAA), 0x5EED, 64);

        let mut total = 0.0f64;
        let n_queries = 64;
        for q in 0..n_queries {
            let c = q % clusters;
            let mut cs = seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
            let centre: Vec<f32> = (0..dim).map(|_| gauss(&mut cs) * 3.0).collect();
            let mut s = seed ^ 0xDEAD_0000 ^ (q as u64).wrapping_mul(0x2545_F491);
            let qv: Vec<f32> = (0..dim).map(|d| centre[d] + gauss(&mut s) * 0.35).collect();
            let truth: HashSet<u32> = idx
                .graph()
                .brute_force(&qv, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got = idx.search_quantized(&qv, 10, 128, 64);
            let hit = got.iter().filter(|(id, _)| truth.contains(id)).count();
            total += hit as f64 / 10.0;
        }
        let recall = total / n_queries as f64;
        // The 1-bit code is coarse, so we do not demand the float 0.95 gate here;
        // but with a 64-wide rerank over an ef=128 beam it must be clearly useful
        // (well above random). ADR-261 reports the exact number; this gate just
        // catches a broken traversal/rerank.
        assert!(
            recall >= 0.80,
            "quantized recall@10 = {recall:.4} too low — traversal or rerank bug"
        );
    }

    #[test]
    fn zero_dim_no_panic() {
        let vectors = vec![vec![], vec![]];
        let idx = QuantizedHnswIndex::build(&vectors, 0, Metric::Cosine, params(5), 0x1, 4);
        let r = idx.search_quantized(&[], 2, 16, 4);
        assert_eq!(r.len(), 2);
    }

    // ----- multi-bit (ADR-261 §11) -----

    /// `bits == 1` via `build_bits` is byte-for-byte the legacy `build` 1-bit
    /// construction: same codes, same search output. Backward-compatibility pin.
    #[test]
    fn one_bit_build_bits_matches_legacy_build() {
        let vectors = planted(32, 400, 8, 0x1B17);
        let legacy = QuantizedHnswIndex::build(&vectors, 32, Metric::L2, params(0x5151), 0xC0DE, 40);
        let viabits =
            QuantizedHnswIndex::build_bits(&vectors, 32, Metric::L2, params(0x5151), 0xC0DE, 1, 40);
        assert_eq!(legacy.bits(), 1);
        assert_eq!(viabits.bits(), 1);
        let q = &vectors[123];
        assert_eq!(
            legacy.search_quantized(q, 10, 64, 40),
            viabits.search_quantized(q, 10, 64, 40),
            "build_bits(…,1,…) must equal legacy build(…)"
        );
    }

    /// Unsupported bit-depths round up to the supported `{1,2,4}` set so the
    /// constructor is total (no panic, predictable resolution).
    #[test]
    fn bits_are_clamped_to_supported_set() {
        let vectors = planted(16, 50, 4, 0xB175);
        for (req, exp) in [(0u32, 1u32), (1, 1), (2, 2), (3, 4), (4, 4), (7, 4)] {
            let idx = QuantizedHnswIndex::build_bits(
                &vectors,
                16,
                Metric::L2,
                params(0x9),
                0xB,
                req,
                16,
            );
            assert_eq!(idx.bits(), exp, "bits {req} should clamp to {exp}");
            // and it must still search without panic
            assert!(!idx.search_quantized(&vectors[0], 5, 32, 20).is_empty());
        }
    }

    /// Bytes/node scales linearly with `bits`: for a power-of-two dim `D`,
    /// 1-bit → D/8, 2-bit → D/4, 4-bit → D/2.
    #[test]
    fn bytes_per_node_scales_with_bits() {
        let vectors = planted(128, 20, 4, 0xBEEF);
        let b1 = QuantizedHnswIndex::build_bits(&vectors, 128, Metric::L2, params(1), 0x5, 1, 16);
        let b2 = QuantizedHnswIndex::build_bits(&vectors, 128, Metric::L2, params(1), 0x5, 2, 16);
        let b4 = QuantizedHnswIndex::build_bits(&vectors, 128, Metric::L2, params(1), 0x5, 4, 16);
        assert_eq!(b1.bytes_per_node(), 16, "128-d 1-bit = 16 B/node");
        assert_eq!(b2.bytes_per_node(), 32, "128-d 2-bit = 32 B/node");
        assert_eq!(b4.bytes_per_node(), 64, "128-d 4-bit = 64 B/node");
    }

    /// More bits must not *reduce* recall at a fixed (ef, rerank): the multi-bit
    /// code is a strictly finer angle proxy than 1-bit, so the traversal beam can
    /// only land on equal-or-better candidates for the rerank to repair. This is
    /// the core ADR-261 §11 hypothesis (multi-bit keeps the beam on-path better),
    /// pinned as a regression gate. We assert a small tolerance for ties.
    #[test]
    fn more_bits_does_not_reduce_recall() {
        let dim = 64;
        let n = 3000;
        let clusters = 32;
        let seed = 0x7A11;
        let vectors = planted(dim, n, clusters, seed);
        let recall_for = |bits: u32| -> f64 {
            let idx = QuantizedHnswIndex::build_bits(
                &vectors,
                dim,
                Metric::L2,
                params(0xA11A),
                0x5EED,
                bits,
                // Modest rerank so traversal quality — not a huge rerank pool —
                // is what drives the recall difference between bit depths.
                20,
            );
            let mut total = 0.0f64;
            let n_queries = 64;
            for q in 0..n_queries {
                let c = q % clusters;
                let mut cs = seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
                let centre: Vec<f32> = (0..dim).map(|_| gauss(&mut cs) * 3.0).collect();
                let mut s = seed ^ 0xDEAD_0000 ^ (q as u64).wrapping_mul(0x2545_F491);
                let qv: Vec<f32> = (0..dim).map(|d| centre[d] + gauss(&mut s) * 0.35).collect();
                let truth: HashSet<u32> = idx
                    .graph()
                    .brute_force(&qv, 10)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let got = idx.search_quantized(&qv, 10, 64, 20);
                let hit = got.iter().filter(|(id, _)| truth.contains(id)).count();
                total += hit as f64 / 10.0;
            }
            total / n_queries as f64
        };
        let r1 = recall_for(1);
        let r2 = recall_for(2);
        let r4 = recall_for(4);
        // 2-bit and 4-bit must be at least as good as 1-bit (small tie tolerance).
        assert!(
            r2 + 0.02 >= r1,
            "2-bit recall {r2:.4} regressed vs 1-bit {r1:.4}"
        );
        assert!(
            r4 + 0.02 >= r1,
            "4-bit recall {r4:.4} regressed vs 1-bit {r1:.4}"
        );
    }
}
