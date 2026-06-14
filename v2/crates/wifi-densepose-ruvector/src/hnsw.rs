//! A correct, dependency-free **float HNSW** graph-ANN index — ADR-261.
//!
//! # Why this exists
//!
//! The ruvector crate's retrieval path (AETHER re-ID hot-cache, the `sketch.rs`
//! 1-bit prefilter, room fingerprinting) is, at its core, an **approximate
//! nearest-neighbour** problem: dense float embedding in, top-K similar ids out.
//! Until now the crate had **no graph index** — every `topk` was a linear scan
//! (`O(N·d)` per query) or a 1-bit Hamming prefilter over a linear scan. That is
//! fine at the small N the unit fixtures use, but it is `O(N)` per query and does
//! not scale.
//!
//! [ADR-156 §5 #1](../../../../../docs/adr/ADR-156-ruvector-fusion-beyond-sota.md)
//! lists **SymphonyQG** (SIGMOD 2025) as the lead beyond-SOTA ANN candidate,
//! claiming **3.5–17× QPS over HNSW at equal recall** — but graded that claim
//! **CLAIMED**, *"not reproduced on our hardware (no HNSW baseline exists to
//! compare against)."* You cannot measure a ratio against a baseline you do not
//! have. This module **builds that missing HNSW baseline**; [`crate::hnsw_quantized`]
//! builds the quantized-rerank variant that tests the *direction* of the
//! SymphonyQG bet. ADR-261 reports the **measured** ratio.
//!
//! # The algorithm (Malkov & Yashunin, TPAMI 2018)
//!
//! HNSW = a multi-layer navigable small-world graph. Each inserted point gets a
//! random **level** `ℓ` (geometrically distributed, mean `1/ln(M)`); it appears
//! in all layers `0..=ℓ`. Layer 0 holds every point; higher layers are
//! exponentially sparser "express lanes". A search:
//!
//! 1. Enters at the top layer's single entry point.
//! 2. **Greedy-descends** each layer above 0: repeatedly hop to the neighbour
//!    closest to the query until no neighbour is closer, then drop a layer.
//! 3. At layer 0, runs a **best-first beam search** with beam width `ef`,
//!    keeping the `ef` closest candidates seen, and returns the closest `k`.
//!
//! Construction inserts each point by searching for its `ef_construction`
//! nearest existing neighbours at each of its layers, then connecting it to a
//! pruned subset chosen by the **neighbour-selection heuristic** (Algorithm 4 in
//! the paper): prefer neighbours that are closer to the new point than to any
//! already-selected neighbour, which keeps the graph navigable (diverse edges)
//! instead of clumping all edges toward one cluster.
//!
//! # Determinism (the proof contract)
//!
//! Level assignment is the only randomness, and it is driven by a **seeded
//! SplitMix64** PRNG (the exact pattern from [`crate::rotation`]) — never
//! `Date::now`, an OS RNG, or `rand` without a seed. Two indices built from the
//! same `(seed, params, insertion order)` are bit-identical, pinned by
//! [`tests::hnsw_is_deterministic_for_seed`]. This matters for reproducible
//! benchmarks: the recall/QPS numbers in ADR-261 must be regenerable.
//!
//! # Robustness (no panic on degenerate input)
//!
//! Empty index, `k > n`, `k == 0`, a single node, zero-dimension vectors,
//! ragged-length queries, and `ef < k` are all handled without panicking —
//! pinned by the `*_no_panic` / degenerate tests. Graph traversal is bounded by
//! the visited-set and the candidate beam, so there is no unbounded recursion
//! (the search is iterative, using explicit heaps).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Distance metric for the index. Both are computed over `Vec<f32>` with an
/// `f64` accumulator for numerical stability on long vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared euclidean distance `Σ (a_i − b_i)²`. Monotone in euclidean
    /// distance, so top-K ranking is identical; we skip the sqrt.
    L2,
    /// Cosine **distance** `1 − cos(a, b)`. Smaller = more similar. This is
    /// AETHER's actual angular metric and what the `sketch.rs` sign code
    /// approximates, so it is the default for ruvector re-ID.
    Cosine,
}

impl Metric {
    /// Distance between two equal-length slices under this metric.
    ///
    /// Ragged lengths are handled charitably (compared over the shorter prefix);
    /// a degenerate (zero-norm) cosine input yields the maximum cosine distance
    /// `1.0` rather than a NaN. Never panics.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        match self {
            Metric::L2 => {
                let mut acc = 0.0f64;
                for i in 0..n {
                    let d = a[i] as f64 - b[i] as f64;
                    acc += d * d;
                }
                acc as f32
            }
            Metric::Cosine => {
                let mut dot = 0.0f64;
                let mut na = 0.0f64;
                let mut nb = 0.0f64;
                for i in 0..n {
                    let (x, y) = (a[i] as f64, b[i] as f64);
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                }
                let denom = (na * nb).sqrt();
                if denom < 1e-12 {
                    1.0
                } else {
                    (1.0 - dot / denom) as f32
                }
            }
        }
    }
}

/// Construction / search hyper-parameters for an [`HnswIndex`].
///
/// Defaults follow the paper's recommended starting points (`M = 16`,
/// `ef_construction = 200`). `ef_search` is the query-time beam width; larger
/// `ef_search` trades QPS for recall — the knob the ADR-261 benchmark sweeps to
/// find the equal-recall operating point.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbours per node on layers ≥ 1. Layer 0 uses `2·M` (`m_max0`),
    /// the paper's standard asymmetry (the base layer needs higher degree).
    pub m: usize,
    /// Candidate list size during construction (`efConstruction`). Larger =
    /// better-connected graph, slower build.
    pub ef_construction: usize,
    /// Default beam width at query time (`ef`). Overridable per-query in
    /// [`HnswIndex::search`].
    pub ef_search: usize,
    /// Seed for the level-assignment PRNG. Fixed ⇒ reproducible graph.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x1157_0000_0000_0001u64,
        }
    }
}

/// A min-distance ordering wrapper: a `BinaryHeap<Candidate>` is a **max-heap**,
/// so we negate the comparison to make `peek()` the *closest* candidate when we
/// want a min-heap, or use it directly for a max-heap of the *farthest*. We keep
/// two explicit newtypes to make the intent unmistakable at each call site.
#[derive(Debug, Clone, Copy)]
struct Scored {
    dist: f32,
    id: u32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Scored {}

/// Max-heap ordering: larger `dist` is "greater" ⇒ at the top. Ties broken by
/// id so the order is total and deterministic.
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `Reverse`-equivalent for a min-heap (closest at top) without pulling in
/// `std::cmp::Reverse` boilerplate at every site.
#[derive(Debug, Clone, Copy)]
struct MinScored(Scored);
impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinScored {}
impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0) // reversed
    }
}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A multi-layer HNSW graph index over dense `Vec<f32>` embeddings.
///
/// IDs are the **insertion index** (`0..len`), returned by [`HnswIndex::search`]
/// alongside the distance. The original vectors are retained (the graph needs
/// them for distance computation at query time), so memory is
/// `O(N·d) + O(N·M)` — the float vectors plus the adjacency lists.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    metric: Metric,
    params: HnswParams,
    dim: usize,
    /// Stored vectors, indexed by id.
    vectors: Vec<Vec<f32>>,
    /// `links[id][layer]` = neighbour ids of `id` on `layer`. A node of level
    /// `ℓ` has `ℓ+1` layers (`0..=ℓ`).
    links: Vec<Vec<Vec<u32>>>,
    /// Per-node top level.
    levels: Vec<usize>,
    /// Current entry point id (the highest-level node), or `None` if empty.
    entry: Option<u32>,
    /// Highest level currently present in the graph.
    top_level: usize,
    /// PRNG state for level assignment (advances per insert).
    rng_state: u64,
}

impl HnswIndex {
    /// Create an empty index with the given metric and parameters.
    ///
    /// `dim` is the expected embedding dimension. Inserts of a different length
    /// are accepted charitably (the metric compares over the shorter prefix), so
    /// a wrong-length vector degrades recall rather than panicking — but callers
    /// should keep dimension uniform.
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self {
            metric,
            params,
            dim,
            vectors: Vec::new(),
            links: Vec::new(),
            levels: Vec::new(),
            entry: None,
            top_level: 0,
            rng_state: params.seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Number of indexed points.
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True iff the index holds no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The metric this index ranks by.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The expected embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The current entry-point id (highest-level node), or `None` if empty.
    /// Exposed so the quantized variant ([`crate::hnsw_quantized`]) can traverse
    /// the **same** graph with a different (quantized) score.
    #[inline]
    pub fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    /// The highest level currently present in the graph.
    #[inline]
    pub fn top_level(&self) -> usize {
        self.top_level
    }

    /// The default query-time beam width (`ef_search`) from this index's params.
    #[inline]
    pub fn params_ef_search(&self) -> usize {
        self.params.ef_search
    }

    /// Borrow the neighbour ids of `id` on `layer`. Returns an empty slice if the
    /// id is unknown or the node does not reach that layer — never panics. Used
    /// by the quantized variant to walk the shared graph.
    #[inline]
    pub fn neighbours(&self, id: u32, layer: usize) -> &[u32] {
        match self.links.get(id as usize).and_then(|l| l.get(layer)) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }

    /// `m_max` for a layer: `2·M` on layer 0, `M` above. The base layer carries
    /// every node and needs higher degree to stay connected (the paper's
    /// asymmetric degree cap).
    #[inline]
    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    /// Draw the next node's level from a geometric distribution with parameter
    /// `m_l = 1/ln(M)` — the paper's level generator — using the **seeded**
    /// SplitMix64 stream. `floor(−ln(U) · m_l)` with `U ∈ (0, 1]`.
    fn assign_level(&mut self) -> usize {
        let m = self.params.m.max(2) as f64;
        let m_l = 1.0 / m.ln();
        // Uniform in (0, 1] from the top 53 bits of a SplitMix64 word.
        let r = split_mix64(&mut self.rng_state);
        let u = (((r >> 11) as f64) + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let level = (-(u.ln()) * m_l).floor();
        if level.is_finite() && level >= 0.0 {
            level as usize
        } else {
            0
        }
    }

    /// Insert `embedding` with the next sequential id. Returns the assigned id.
    ///
    /// Builds the node's adjacency by searching the existing graph for its
    /// nearest neighbours at each of its layers and connecting via the
    /// neighbour-selection heuristic. The first insert becomes the entry point.
    pub fn insert(&mut self, embedding: &[f32]) -> u32 {
        let id = self.vectors.len() as u32;
        let vec = embedding.to_vec();
        let node_level = self.assign_level();

        // Push the node into the arrays UP FRONT with empty per-layer link lists.
        // This is load-bearing: the bidirectional wiring below does
        // `self.links[nbr][l].push(id)`, after which a neighbour points at `id`;
        // a subsequent traversal step in the SAME insert can hop to that
        // neighbour and read `self.links[id]`. If `id`'s links did not exist yet
        // that read panics (the bug the recall gate caught). The new node has no
        // *incoming* edges until we add them, and empty outgoing lists, so it is
        // unreachable by the searches that run before its edges are wired —
        // pushing it early is safe and keeps every `self.links[*]` index valid.
        self.vectors.push(vec.clone());
        self.links.push(vec![Vec::new(); node_level + 1]);
        self.levels.push(node_level);

        // First node: it is the entry point, no neighbours to connect.
        if self.entry.is_none() {
            self.entry = Some(id);
            self.top_level = node_level;
            return id;
        }

        let entry = self.entry.unwrap();
        let mut ep = entry;

        // Phase 1: greedy-descend from the top of the graph down to the layer
        // just above the node's own top level, refining the single entry point.
        let mut layer = self.top_level;
        while layer > node_level {
            ep = self.greedy_closest(&vec, ep, layer);
            if layer == 0 {
                break;
            }
            layer -= 1;
        }

        // Phase 2: from min(node_level, top_level) down to 0, search for
        // ef_construction candidates, select neighbours, and wire bidirectional
        // edges (pruning the neighbour's list if it overflows m_max).
        let start = node_level.min(self.top_level);
        let mut layer = start as isize;
        while layer >= 0 {
            let l = layer as usize;
            let candidates =
                self.search_layer(&vec, &[ep], self.params.ef_construction.max(1), l);
            let selected = self.select_neighbours(&vec, &candidates, self.m_max(l));

            // Connect node -> selected (write straight into the node's slot).
            self.links[id as usize][l] = selected.iter().map(|s| s.id).collect();

            // Connect selected -> node (bidirectional), pruning if needed.
            for s in &selected {
                let nbr = s.id as usize;
                self.links[nbr][l].push(id);
                if self.links[nbr][l].len() > self.m_max(l) {
                    self.prune_neighbours(nbr as u32, l);
                }
            }

            // Move the entry for the next-lower layer to the closest candidate.
            if let Some(best) = candidates
                .iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal))
            {
                ep = best.id;
            }
            layer -= 1;
        }

        if node_level > self.top_level {
            self.top_level = node_level;
            self.entry = Some(id);
        }
        id
    }

    /// Greedy single-best descent on one layer: hop to the neighbour closest to
    /// `query` until no neighbour improves. Iterative (bounded by the graph) —
    /// no recursion.
    fn greedy_closest(&self, query: &[f32], start: u32, layer: usize) -> u32 {
        let mut best = start;
        let mut best_d = self.metric.distance(query, &self.vectors[best as usize]);
        loop {
            let mut improved = false;
            for &nbr in &self.links[best as usize][layer] {
                let d = self.metric.distance(query, &self.vectors[nbr as usize]);
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

    /// Beam search on one layer (paper Algorithm 2): best-first expansion from
    /// `entry_points`, keeping the `ef` closest results. Returns the result set
    /// (unsorted; callers sort/truncate). Bounded by a visited set + the `ef`
    /// result heap — no recursion, no unbounded growth.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<Scored> {
        let mut visited: HashSet<u32> = HashSet::new();
        // `candidates`: min-heap (closest first) of nodes to expand.
        let mut candidates: BinaryHeap<MinScored> = BinaryHeap::new();
        // `results`: max-heap (farthest first) of the best-ef found so far, so
        // the top is the current worst and is cheap to evict.
        let mut results: BinaryHeap<Scored> = BinaryHeap::new();

        for &ep in entry_points {
            if ep as usize >= self.vectors.len() {
                continue;
            }
            let d = self.metric.distance(query, &self.vectors[ep as usize]);
            let s = Scored { dist: d, id: ep };
            visited.insert(ep);
            candidates.push(MinScored(s));
            results.push(s);
        }
        // Cap results at ef from the start.
        while results.len() > ef {
            results.pop();
        }

        while let Some(MinScored(cur)) = candidates.pop() {
            // Stop when the closest unexpanded candidate is farther than the
            // current worst result and the result set is already full.
            let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if cur.dist > worst && results.len() >= ef {
                break;
            }
            for &nbr in &self.links[cur.id as usize][layer] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.metric.distance(query, &self.vectors[nbr as usize]);
                let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    let s = Scored { dist: d, id: nbr };
                    candidates.push(MinScored(s));
                    results.push(s);
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_vec()
    }

    /// Neighbour-selection heuristic (paper Algorithm 4): from `candidates`,
    /// greedily pick up to `m` that are **closer to the new point than to any
    /// already-picked neighbour**, giving diverse, navigable edges instead of a
    /// clump. Candidates are considered nearest-first.
    fn select_neighbours(&self, _base: &[f32], candidates: &[Scored], m: usize) -> Vec<Scored> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        let mut selected: Vec<Scored> = Vec::with_capacity(m);
        for cand in sorted {
            if selected.len() >= m {
                break;
            }
            // Keep `cand` only if it is closer to `base` than to every already
            // selected neighbour — the diversity condition.
            let cand_vec = &self.vectors[cand.id as usize];
            let mut keep = true;
            for sel in &selected {
                let d_cand_sel = self.metric.distance(cand_vec, &self.vectors[sel.id as usize]);
                if d_cand_sel < cand.dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(cand);
            }
        }
        // If the diversity filter left us short (sparse graph), backfill with the
        // remaining nearest candidates so the node is not under-connected.
        if selected.len() < m {
            let chosen: HashSet<u32> = selected.iter().map(|s| s.id).collect();
            let mut rest: Vec<Scored> = candidates
                .iter()
                .filter(|c| !chosen.contains(&c.id))
                .copied()
                .collect();
            rest.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            for c in rest {
                if selected.len() >= m {
                    break;
                }
                selected.push(c);
            }
        }
        selected
    }

    /// Re-prune a node's neighbour list on `layer` back down to `m_max` using
    /// the selection heuristic, after a bidirectional edge pushed it over cap.
    fn prune_neighbours(&mut self, id: u32, layer: usize) {
        let base = self.vectors[id as usize].clone();
        let current: Vec<Scored> = self.links[id as usize][layer]
            .iter()
            .map(|&nbr| Scored {
                dist: self.metric.distance(&base, &self.vectors[nbr as usize]),
                id: nbr,
            })
            .collect();
        let kept = self.select_neighbours(&base, &current, self.m_max(layer));
        self.links[id as usize][layer] = kept.iter().map(|s| s.id).collect();
    }

    /// Search for the `k` nearest neighbours of `query`, using beam width `ef`
    /// (clamped to at least `k`). Returns up to `k` `(id, distance)` pairs sorted
    /// ascending by distance.
    ///
    /// Degenerate cases return cleanly: empty index ⇒ empty vec; `k == 0` ⇒ empty
    /// vec; `k > len` ⇒ all points; a single node ⇒ that node. Never panics.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let ef = ef.max(k).max(1);

        // Greedy-descend the upper layers to a good layer-0 entry point.
        let mut ep = entry;
        let mut layer = self.top_level;
        while layer > 0 {
            ep = self.greedy_closest(query, ep, layer);
            layer -= 1;
        }
        // Beam search on layer 0.
        let mut results = self.search_layer(query, &[ep], ef, 0);
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results.into_iter().map(|s| (s.id, s.dist)).collect()
    }

    /// Search using the index's configured default `ef_search`.
    #[inline]
    pub fn search_default(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search(query, k, self.params.ef_search)
    }

    /// Borrow a stored vector by id (for the quantized variant / reranking).
    #[inline]
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id as usize).map(|v| v.as_slice())
    }

    /// Brute-force exact top-K linear scan over the stored vectors — the ANN
    /// **ground truth** and the linear-scan baseline the benchmark measures
    /// against. `O(N·d)` per query. Returns up to `k` `(id, distance)` ascending.
    pub fn brute_force(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(u32, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, self.metric.distance(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// SplitMix64 step — the same deterministic PRNG used by [`crate::rotation`].
/// Public-domain (Sebastiano Vigna). Dependency-free and reproducible.
#[inline]
pub(crate) fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64-driven uniform in [0,1) for building fixtures (mirrors
    /// `coverage.rs`'s style so the planted-cluster geometry matches).
    fn unif01(state: &mut u64) -> f32 {
        let r = split_mix64(state);
        ((r >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn gauss(state: &mut u64) -> f32 {
        let u1 = unif01(state).max(1e-7);
        let u2 = unif01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Build a planted-cluster fixture: `n` vectors of `dim`, in `clusters`
    /// Gaussian clusters. Returns the vectors. Deterministic from `seed`.
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

    fn build(vectors: &[Vec<f32>], metric: Metric, seed: u64) -> HnswIndex {
        let params = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed,
        };
        let mut idx = HnswIndex::new(vectors[0].len(), metric, params);
        for v in vectors {
            idx.insert(v);
        }
        idx
    }

    /// Recall@k of HNSW search vs brute-force ground truth, averaged over queries
    /// drawn from the same planted clusters.
    fn recall_at_k(
        idx: &HnswIndex,
        vectors: &[Vec<f32>],
        dim: usize,
        clusters: usize,
        k: usize,
        ef: usize,
        n_queries: usize,
        seed: u64,
    ) -> f64 {
        let centres_seed = seed; // reuse fixture seed for matching cluster geometry
        let mut total = 0.0f64;
        for q in 0..n_queries {
            let c = q % clusters;
            let mut s = centres_seed ^ 0xDEAD_0000 ^ (q as u64).wrapping_mul(0x2545_F491);
            // A query near cluster centre c: regenerate the centre then jitter.
            let mut cs = centres_seed ^ (0xC0FFEE_u64.wrapping_mul(c as u64 + 1));
            let centre: Vec<f32> = (0..dim).map(|_| gauss(&mut cs) * 3.0).collect();
            let qv: Vec<f32> = (0..dim).map(|d| centre[d] + gauss(&mut s) * 0.35).collect();

            let truth: HashSet<u32> = idx.brute_force(&qv, k).into_iter().map(|(id, _)| id).collect();
            let got = idx.search(&qv, k, ef);
            let hit = got.iter().filter(|(id, _)| truth.contains(id)).count();
            total += hit as f64 / k as f64;
            let _ = vectors;
        }
        total / n_queries as f64
    }

    #[test]
    fn empty_index_search_is_empty_no_panic() {
        let idx = HnswIndex::new(8, Metric::L2, HnswParams::default());
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 8], 5, 16).is_empty());
        assert!(idx.brute_force(&[0.0; 8], 5).is_empty());
    }

    #[test]
    fn single_node_returns_itself() {
        let mut idx = HnswIndex::new(4, Metric::L2, HnswParams::default());
        let id = idx.insert(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(id, 0);
        let r = idx.search(&[1.0, 2.0, 3.0, 4.0], 5, 16);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
        assert!(r[0].1 < 1e-6);
    }

    #[test]
    fn k_zero_and_k_gt_n_no_panic() {
        let vectors = planted(16, 40, 4, 0xABCD);
        let idx = build(&vectors, Metric::L2, 0x1234);
        assert!(idx.search(&vectors[0], 0, 16).is_empty());
        // k > n returns all n.
        let r = idx.search(&vectors[0], 1000, 64);
        assert_eq!(r.len(), 40);
    }

    #[test]
    fn ragged_query_no_panic() {
        let vectors = planted(16, 30, 3, 0x55);
        let idx = build(&vectors, Metric::Cosine, 0x66);
        // Short and long queries must not panic.
        assert!(!idx.search(&[1.0, 2.0, 3.0], 3, 16).is_empty());
        let long: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert!(!idx.search(&long, 3, 16).is_empty());
    }

    #[test]
    fn self_query_ranks_self_first() {
        let vectors = planted(32, 200, 8, 0x77);
        let idx = build(&vectors, Metric::L2, 0x88);
        for &probe in &[0usize, 50, 137, 199] {
            let r = idx.search(&vectors[probe], 1, 64);
            assert_eq!(r.len(), 1);
            assert_eq!(r[0].0, probe as u32, "self-query should return the stored self");
        }
    }

    #[test]
    fn hnsw_is_deterministic_for_seed() {
        // Same (seed, params, insertion order) ⇒ identical level assignment and
        // identical search output.
        let vectors = planted(24, 150, 6, 0x2222);
        let a = build(&vectors, Metric::Cosine, 0xFEED);
        let b = build(&vectors, Metric::Cosine, 0xFEED);
        assert_eq!(a.levels, b.levels, "level assignment must be deterministic");
        let q = &vectors[42];
        assert_eq!(a.search(q, 10, 64), b.search(q, 10, 64));
        // A different seed (almost surely) changes the level structure.
        let c = build(&vectors, Metric::Cosine, 0x1357);
        assert_ne!(a.levels, c.levels, "different seed should change levels");
    }

    #[test]
    fn recall_at_10_meets_correctness_gate_l2() {
        // THE CORRECTNESS GATE (ADR-261): HNSW recall@10 vs brute-force must be
        // >= 0.95 at a reasonable ef. Low recall ⇒ a bug in the graph.
        let dim = 64;
        let n = 2000;
        let clusters = 32;
        let seed = 0x9999;
        let vectors = planted(dim, n, clusters, seed);
        let idx = build(&vectors, Metric::L2, 0xAAAA);
        let recall = recall_at_k(&idx, &vectors, dim, clusters, 10, 128, 64, seed);
        assert!(
            recall >= 0.95,
            "HNSW recall@10 (L2) = {recall:.4} below the 0.95 correctness gate — graph bug"
        );
    }

    #[test]
    fn recall_at_10_meets_correctness_gate_cosine() {
        let dim = 64;
        let n = 2000;
        let clusters = 32;
        let seed = 0xBBBB;
        let vectors = planted(dim, n, clusters, seed);
        let idx = build(&vectors, Metric::Cosine, 0xCCCC);
        let recall = recall_at_k(&idx, &vectors, dim, clusters, 10, 128, 64, seed);
        assert!(
            recall >= 0.95,
            "HNSW recall@10 (cosine) = {recall:.4} below the 0.95 correctness gate — graph bug"
        );
    }

    #[test]
    fn higher_ef_does_not_reduce_recall() {
        // Monotonicity sanity: more beam width should not hurt recall.
        let dim = 48;
        let vectors = planted(dim, 1000, 16, 0xD00D);
        let idx = build(&vectors, Metric::L2, 0xE00E);
        let lo = recall_at_k(&idx, &vectors, dim, 16, 10, 16, 48, 0xD00D);
        let hi = recall_at_k(&idx, &vectors, dim, 16, 10, 128, 48, 0xD00D);
        assert!(hi + 1e-9 >= lo, "recall dropped with larger ef: {lo:.3} -> {hi:.3}");
    }

    #[test]
    fn zero_dim_no_panic() {
        // Degenerate zero-dimension index: inserts and searches must not panic.
        let mut idx = HnswIndex::new(0, Metric::Cosine, HnswParams::default());
        idx.insert(&[]);
        idx.insert(&[]);
        let r = idx.search(&[], 2, 16);
        assert_eq!(r.len(), 2);
    }
}
