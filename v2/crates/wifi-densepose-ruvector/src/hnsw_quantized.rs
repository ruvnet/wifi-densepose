//! A **SymphonyQG-style quantized-traversal HNSW** — ADR-261.
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
//! - **Quantized score = 1-bit Hamming over the RaBitQ Pass-2 rotated sign code**
//!   ([`crate::rotation`] + the sign-quantization in [`crate::sketch`]). Each
//!   node stores its `ceil(D/8)`-byte sign code (`D = next_pow2(dim)`). During
//!   traversal we compare query-code vs node-code by **POPCNT Hamming** — a few
//!   machine words, no per-dimension float work.
//! - **Exact float rerank** of the final beam: the top `rerank` candidates by
//!   Hamming are re-scored with the true float metric and the best `k` returned.
//!
//! This trades a small recall hit (the 1-bit code is a coarse angle proxy — the
//! same ~46%-strict limitation ADR-156 §10 measured) for far cheaper per-node
//! scoring, recovered by the float rerank. **Whether that nets a QPS win at our
//! test scale is the measured question ADR-261 answers** — and at small N the
//! float distance is cheap enough that the Hamming saving may not pay off. We
//! report the real number, win or lose, and do not tune to manufacture a speedup.
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

/// A 1-bit Pass-2 sign code for one vector, over the padded rotation length `D`.
/// Stored as packed bytes; compared by POPCNT Hamming.
#[derive(Debug, Clone)]
struct Code {
    bits: Vec<u8>,
}

impl Code {
    /// Hamming distance to another code of the same length (popcount of XOR).
    #[inline]
    fn hamming(&self, other: &Code) -> u32 {
        let n = self.bits.len().min(other.bits.len());
        let mut acc = 0u32;
        for i in 0..n {
            acc += (self.bits[i] ^ other.bits[i]).count_ones();
        }
        acc
    }
}

/// Build the packed 1-bit sign code of a rotated embedding over the padded
/// length `D = rotation.padded_dim()`. Bit set ⇒ rotated coord ≥ 0.
fn encode(embedding: &[f32], rotation: &Rotation) -> Code {
    let rotated = rotation.apply_padded(embedding);
    let d = rotated.len();
    let mut bits = vec![0u8; d.div_ceil(8)];
    for (i, &c) in rotated.iter().enumerate() {
        if c >= 0.0 {
            bits[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    Code { bits }
}

/// Min-heap node for the quantized beam (closest Hamming at the top).
#[derive(Debug, Clone, Copy)]
struct HScored {
    /// Hamming distance (quantized score) — the traversal key.
    ham: u32,
    id: u32,
}
impl PartialEq for HScored {
    fn eq(&self, other: &Self) -> bool {
        self.ham == other.ham && self.id == other.id
    }
}
impl Eq for HScored {}
impl Ord for HScored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ham.cmp(&other.ham).then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for HScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
/// Reversed wrapper for a min-heap (smallest Hamming at the top).
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
/// **cheap 1-bit Hamming score**, with a final **exact-float rerank**.
///
/// Built by inserting the same vectors in the same order with the same seed as
/// a float [`HnswIndex`], so the two indices share identical graph structure and
/// only differ in how the beam is scored. The shared [`Rotation`] (seed + dim)
/// is the index/query frame for the 1-bit codes.
#[derive(Debug, Clone)]
pub struct QuantizedHnswIndex {
    /// The underlying graph (built with the float metric for exact rerank).
    graph: HnswIndex,
    /// Per-node 1-bit Pass-2 codes, indexed by id (parallel to graph vectors).
    codes: Vec<Code>,
    /// The rotation frame shared by index and query codes.
    rotation: Rotation,
    /// Number of final candidates to exact-float rerank (≥ k at query time).
    default_rerank: usize,
}

impl QuantizedHnswIndex {
    /// Build a quantized index over `vectors`, mirroring a float [`HnswIndex`]
    /// built with the same `(dim, metric, params)` and insertion order. The
    /// `rotation_seed` fixes the 1-bit code frame (index and query share it).
    ///
    /// `default_rerank` is how many top-Hamming candidates get an exact float
    /// re-score before returning the best `k`; it is clamped to `≥ k` at query
    /// time. A larger rerank recovers more recall at more float cost — the knob
    /// that, alongside `ef`, sets the equal-recall operating point.
    pub fn build(
        vectors: &[Vec<f32>],
        dim: usize,
        metric: Metric,
        params: HnswParams,
        rotation_seed: u64,
        default_rerank: usize,
    ) -> Self {
        let rotation = Rotation::new(rotation_seed, dim);
        let mut graph = HnswIndex::new(dim, metric, params);
        let mut codes = Vec::with_capacity(vectors.len());
        for v in vectors {
            graph.insert(v);
            codes.push(encode(v, &rotation));
        }
        Self {
            graph,
            codes,
            rotation,
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

    /// SymphonyQG-style search: traverse the graph scoring candidates by **1-bit
    /// Hamming**, collect a beam of `ef`, then **exact-float rerank** the top
    /// `rerank` (clamped ≥ k) and return the best `k` as `(id, float_dist)`.
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
        let q_code = encode(query, &self.rotation);

        // Entry point: the graph's entry (highest-level node).
        let entry = match self.graph.entry_point() {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Greedy-descend upper layers by Hamming, then beam-search layer 0.
        let mut ep = entry;
        let mut layer = self.graph.top_level();
        while layer > 0 {
            ep = self.greedy_hamming(&q_code, ep, layer);
            layer -= 1;
        }
        let beam = self.beam_hamming(&q_code, ep, ef);

        // Exact-float rerank of the top `rerank` Hamming candidates.
        let mut cand: Vec<HScored> = beam;
        cand.sort_by_key(|c| c.ham);
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

    /// Greedy single-best descent on a layer scored by Hamming.
    fn greedy_hamming(&self, q_code: &Code, start: u32, layer: usize) -> u32 {
        let mut best = start;
        let mut best_h = self.codes[best as usize].hamming(q_code);
        loop {
            let mut improved = false;
            for &nbr in self.graph.neighbours(best, layer) {
                let h = self.codes[nbr as usize].hamming(q_code);
                if h < best_h {
                    best_h = h;
                    best = nbr;
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Beam search on layer 0 scored by Hamming. Returns the `ef` best-Hamming
    /// nodes (unsorted). Iterative — bounded by the visited set + the ef beam.
    fn beam_hamming(&self, q_code: &Code, ep: u32, ef: usize) -> Vec<HScored> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<MinH> = BinaryHeap::new();
        let mut results: BinaryHeap<HScored> = BinaryHeap::new(); // max-heap: worst at top

        let h0 = self.codes[ep as usize].hamming(q_code);
        let s0 = HScored { ham: h0, id: ep };
        visited.insert(ep);
        candidates.push(MinH(s0));
        results.push(s0);

        while let Some(MinH(cur)) = candidates.pop() {
            let worst = results.peek().map(|s| s.ham).unwrap_or(u32::MAX);
            if cur.ham > worst && results.len() >= ef {
                break;
            }
            for &nbr in self.graph.neighbours(cur.id, 0) {
                if !visited.insert(nbr) {
                    continue;
                }
                let h = self.codes[nbr as usize].hamming(q_code);
                let worst = results.peek().map(|s| s.ham).unwrap_or(u32::MAX);
                if results.len() < ef || h < worst {
                    let s = HScored { ham: h, id: nbr };
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
}
