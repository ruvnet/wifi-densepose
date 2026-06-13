//! RaBitQ **unbiased distance estimator** — the real Gao & Long (SIGMOD 2024)
//! contribution, on top of the Pass-2 rotation ([`crate::rotation`]).
//!
//! ## Why this exists (ADR-156 Milestone-2)
//!
//! Pass-1 ([`crate::sketch`]) and Pass-2 ([`crate::rotation`]) use only the
//! **sign** of each rotated coordinate and rank candidates by **Hamming /
//! bit distance** — a coarse, monotone-but-lossy proxy for the true angle.
//! ADR-156 §10 measured that sign-only Pass-2 leaves strict-K
//! (`candidate_k == K`) top-K coverage at **~46%**, well below the ADR-084
//! **≥90%** bar, and only clears 90% with ~3× over-fetch.
//!
//! RaBitQ's *actual* algorithmic contribution is not the sign bits — it is an
//! **unbiased estimator of the inner product / squared distance** recovered
//! from the 1-bit code **plus a few bytes of per-vector side information**.
//! That estimate is far sharper than the raw Hamming proxy, so it can
//! **rerank** the candidate set and (the question this module measures) close
//! the strict-K coverage gap.
//!
//! ## The estimator (paper formula + our simplification, stated honestly)
//!
//! Notation follows the paper. Let `P` be the Pass-2 orthogonal rotation
//! ([`crate::Rotation`], `R = H·D`). For a data vector `o_raw` and a query
//! `q_raw`:
//!
//! 1. **Centroid.** The paper centres each vector on its (per-cluster)
//!    centroid `c`: residual `o_r = o_raw − c`. **We use a zero / global
//!    centroid `c = 0`** (`o_r = o_raw`). This is an explicit simplification
//!    (no IVF/k-means cluster structure in the current sketch path) — it costs
//!    accuracy when the data is far off-origin, and we document it rather than
//!    hide it. With `c = 0`, the residual *is* the raw vector.
//!
//! 2. **Unit residual + 1-bit code.** `o = o_r / ‖o_r‖`. Rotate:
//!    `o' = P·o`. The 1-bit code is `x̄_i = sign(o'_i) · (1/√D)`, so `x̄`
//!    is a **unit vector** in `{±1/√D}^D` (the corner of the hypercube nearest
//!    `o'`). `D` is the rotation's padded dimension (`next_pow2(dim)`), because
//!    the FHT operates on the padded length and `x̄` is unit over that length.
//!
//! 3. **Per-vector side information** (the "few bytes"): we store, per sketch,
//!    - `residual_norm = ‖o_r‖` (an `f32`), and
//!    - `x_dot_o = ⟨x̄, o'⟩` (an `f32`), the cosine between the code and the
//!      rotated unit residual. This is the quantity the paper calls `⟨x̄, o⟩`
//!      (after rotation); it lies in `(0, 1]` and is `1` only when `o'`
//!      already sits exactly on a hypercube corner.
//!
//!    That is **8 bytes/vector** of side info (2× `f32`).
//!
//! 4. **Query-time estimate.** Rotate the query residual: `q' = P·q_r`. The
//!    **unbiased estimator of `⟨o', q'⟩`** (equivalently `⟨o, q_r⟩`, since `P`
//!    is orthogonal) is
//!
//!    ```text
//!        ⟨o', q'⟩  ≈  ⟨x̄, q'⟩ / ⟨x̄, o'⟩  =  ⟨x̄, q'⟩ / x_dot_o
//!    ```
//!
//!    This is RaBitQ Eq. (in the paper, the estimator `<q, o> ≈ <q̄, ...>`):
//!    the random rotation makes the quantization error of `x̄` (relative to
//!    `o'`) orthogonal **in expectation** to `q'`, so dividing the measured
//!    `⟨x̄, q'⟩` by `x_dot_o` is **unbiased** for `⟨o', q'⟩`, with the paper's
//!    `O(1/√D)` error bound. The only per-candidate cost is one length-`D`
//!    dot product `⟨x̄, q'⟩` — which, because `x̄ ∈ {±1/√D}`, is just a signed
//!    sum of the query coordinates (`±` chosen by the stored sign bits),
//!    i.e. as cheap as the Hamming proxy plus one multiply.
//!
//! 5. **Inner product and squared distance.** Un-normalize:
//!    `⟨o_r, q_r⟩ = ‖o_r‖ · ⟨o, q_r⟩`. Then
//!
//!    ```text
//!        ‖q_r − o_r‖²  =  ‖q_r‖²  +  ‖o_r‖²  −  2·⟨o_r, q_r⟩
//!    ```
//!
//!    For **ranking** a candidate set against one fixed query, `‖q_r‖²` is a
//!    per-query constant and can be dropped; we keep it in
//!    [`DistanceEstimator::estimate_sq_distance`] so the value is a genuine
//!    distance estimate (used by the unbiasedness test), and expose the
//!    cheaper ranking key separately.
//!
//! ## What is unbiased, and what we measure
//!
//! The estimator of `⟨o', q'⟩` is unbiased over the random rotation. We pin
//! that on a small hand-checkable fixture (`estimator_unbiased_on_fixture`):
//! averaging the estimate over many random rotation seeds converges to the true
//! inner product within tolerance. We then measure whether **reranking the
//! candidate set by this estimate** closes the strict-K coverage gap that the
//! sign-only Pass-2 left at ~46% — reported honestly in ADR-156 §10 / §11
//! whether it clears 90% or not.
//!
//! ## Backward compatibility
//!
//! This module is **purely additive**. It introduces an *extended* sketch type
//! ([`EstimatorSketch`]) and bank ([`EstimatorBank`]) that carry the side info;
//! the Pass-1 [`crate::Sketch`] / Pass-2 [`crate::SketchBank`] paths and the
//! [`crate::WireSketch`] wire format are **untouched**. Nothing on the existing
//! surface changes.

use crate::rotation::{next_pow2, Rotation};

/// The per-vector side information RaBitQ needs to turn a 1-bit code into an
/// **unbiased** distance estimate (§ module docs step 3).
///
/// Two `f32`s = **8 bytes/vector** on top of the packed sign bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideInfo {
    /// `‖o_r‖` — L2 norm of the (zero-centroid) residual = the raw vector norm.
    pub residual_norm: f32,
    /// `⟨x̄, o'⟩` — dot product of the unit 1-bit code with the rotated unit
    /// residual. In `(0, 1]`; the paper's `⟨x̄, o⟩`. Drives the unbiased
    /// rescaling `⟨x̄, q'⟩ / x_dot_o`.
    pub x_dot_o: f32,
}

/// A Pass-2 sketch **plus** the RaBitQ side information, sufficient to compute
/// the unbiased distance estimate at query time.
///
/// Stores the packed sign bits over the **padded** rotation length `D`
/// (`next_pow2(dim)`) — the frame `x̄` actually lives in — together with the
/// [`SideInfo`]. Construct via [`EstimatorSketch::from_embedding`]; the index
/// and the query **must** use the same [`Rotation`] (same seed + dim), exactly
/// as for a Pass-2 sketch.
#[derive(Debug, Clone)]
pub struct EstimatorSketch {
    /// Sign bits of the rotated *padded* unit residual, MSB-first per byte.
    /// Length is `ceil(D / 8)` where `D = next_pow2(dim)`. Bit set ⇒ `o'_i ≥ 0`
    /// ⇒ code coordinate `+1/√D`; clear ⇒ `−1/√D`.
    bits: Vec<u8>,
    /// Padded rotation dimension `D = next_pow2(dim)`; the code is unit over `D`.
    padded_dim: usize,
    /// Source embedding dimension (for compatibility checks / reporting).
    embedding_dim: usize,
    /// The RaBitQ side info for the unbiased estimate.
    side: SideInfo,
}

impl EstimatorSketch {
    /// Build an estimator sketch from a dense embedding and a [`Rotation`].
    ///
    /// Zero-centroid (`c = 0`): the residual is the raw embedding. The vector is
    /// rotated through `rotation` over its padded length `D = next_pow2(dim)`,
    /// the sign of each rotated coordinate is packed, and the side info
    /// (`‖o_r‖`, `⟨x̄, o'⟩`) is computed in the same pass.
    ///
    /// A zero (or all-equal-to-its-own-mean) input yields `residual_norm = 0`;
    /// its estimate degenerates to `0` (handled in
    /// [`EstimatorBank`]) rather than dividing by zero.
    pub fn from_embedding(embedding: &[f32], rotation: &Rotation) -> Self {
        Self::from_embedding_centred(embedding, rotation, None)
    }

    /// Build an estimator sketch with an **explicit centroid** `c` subtracted
    /// before rotation (the paper's per-cluster centroid; `o_r = o_raw − c`).
    ///
    /// Pass `None` for the zero-centroid simplification (`c = 0`, identical to
    /// [`EstimatorSketch::from_embedding`]). Pass `Some(centroid)` (length `dim`)
    /// to centre on a shared global / cluster centroid — the index and the query
    /// **must** use the *same* centroid, exactly as they must share the rotation.
    /// This path exists so ADR-156 can **measure the cost of the zero-centroid
    /// simplification** honestly rather than assert it.
    pub fn from_embedding_centred(
        embedding: &[f32],
        rotation: &Rotation,
        centroid: Option<&[f32]>,
    ) -> Self {
        let dim = rotation.dim();
        let padded = next_pow2(dim);
        // Residual o_r = o_raw − c (c = 0 when centroid is None). Build it once.
        let residual: Vec<f32> = (0..dim)
            .map(|i| {
                let v = embedding.get(i).copied().unwrap_or(0.0);
                let c = centroid.and_then(|c| c.get(i)).copied().unwrap_or(0.0);
                v - c
            })
            .collect();
        let residual_norm = {
            let mut acc = 0.0f64;
            for &v in &residual {
                acc += (v as f64) * (v as f64);
            }
            acc.sqrt() as f32
        };

        // Rotate the RESIDUAL over the PADDED length so the code frame matches
        // what `x_dot_o` and the query dot product use.
        let rotated_padded = rotation.apply_padded(&residual);
        debug_assert_eq!(rotated_padded.len(), padded);

        // 1-bit code over the padded length: x̄_i = sign(o'_i)/√D on the *unit*
        // residual. Since o' = P·o = P·(o_r/‖o_r‖) = (P·o_r)/‖o_r‖, and sign is
        // scale-invariant, sign(o'_i) == sign((P·o_r)_i) == sign(rotated_padded_i).
        // ⟨x̄, o'⟩ = (1/√D)·Σ sign(o'_i)·o'_i = (1/√D)·Σ |o'_i|
        //         = (1/√D)·(Σ|(P·o_r)_i|) / ‖o_r‖.
        let inv_sqrt_d = 1.0f32 / (padded as f32).sqrt();
        let mut bits = vec![0u8; padded.div_ceil(8)];
        let mut sum_abs = 0.0f64; // Σ |(P·o_r)_i|
        for (i, &c) in rotated_padded.iter().enumerate() {
            if c >= 0.0 {
                bits[i / 8] |= 1 << (7 - (i % 8));
            }
            sum_abs += (c as f64).abs();
        }
        // ⟨x̄, o'⟩ with o' the rotated *unit* residual.
        let x_dot_o = if residual_norm > 0.0 {
            (inv_sqrt_d as f64 * sum_abs / residual_norm as f64) as f32
        } else {
            0.0
        };

        Self {
            bits,
            padded_dim: padded,
            embedding_dim: dim,
            side: SideInfo {
                residual_norm,
                x_dot_o,
            },
        }
    }

    /// The padded rotation dimension `D` the code lives in.
    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Source embedding dimension.
    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// The RaBitQ side information.
    #[inline]
    pub fn side_info(&self) -> SideInfo {
        self.side
    }

    /// `‖o_r‖` of the residual (zero-centroid ⇒ raw vector norm).
    #[inline]
    pub fn residual_norm(&self) -> f32 {
        self.side.residual_norm
    }

    /// Side-information byte cost (excluding the packed sign bits): 8 bytes.
    pub const SIDE_INFO_BYTES: usize = 2 * std::mem::size_of::<f32>();

    /// `⟨x̄, q'⟩` — the dot product of this sketch's unit 1-bit code with a
    /// rotated query `q'` (length `padded_dim`). Because `x̄_i = ±1/√D`, this is
    /// `(1/√D)·Σ ±q'_i` with the sign taken from the stored bit. The single
    /// per-candidate cost of the estimator.
    #[inline]
    fn code_dot(&self, q_rotated_padded: &[f32]) -> f32 {
        debug_assert_eq!(q_rotated_padded.len(), self.padded_dim);
        let inv_sqrt_d = 1.0f32 / (self.padded_dim as f32).sqrt();
        let mut acc = 0.0f32;
        for (i, &q) in q_rotated_padded.iter().enumerate() {
            let bit = (self.bits[i / 8] >> (7 - (i % 8))) & 1;
            if bit == 1 {
                acc += q;
            } else {
                acc -= q;
            }
        }
        acc * inv_sqrt_d
    }
}

/// A pre-rotated query, computed **once** per query and reused across all
/// candidates. Carries `q' = P·q_r` (over the padded length) and `‖q_r‖²`.
#[derive(Debug, Clone)]
pub struct EstimatorQuery {
    /// `q' = P·q_r` over the padded rotation length.
    q_rotated_padded: Vec<f32>,
    /// `‖q_r‖²` — per-query constant in the squared-distance expansion.
    q_norm_sq: f32,
}

impl EstimatorQuery {
    /// Pre-rotate a query embedding through `rotation` (zero-centroid).
    pub fn new(query: &[f32], rotation: &Rotation) -> Self {
        Self::new_centred(query, rotation, None)
    }

    /// Pre-rotate a query residual `q_r = q − c` through `rotation`. The
    /// centroid **must** match the one used to build the bank's sketches.
    pub fn new_centred(query: &[f32], rotation: &Rotation, centroid: Option<&[f32]>) -> Self {
        let dim = rotation.dim();
        let residual: Vec<f32> = (0..dim)
            .map(|i| {
                let v = query.get(i).copied().unwrap_or(0.0);
                let c = centroid.and_then(|c| c.get(i)).copied().unwrap_or(0.0);
                v - c
            })
            .collect();
        let mut q_norm_sq = 0.0f64;
        for &v in &residual {
            q_norm_sq += (v as f64) * (v as f64);
        }
        Self {
            q_rotated_padded: rotation.apply_padded(&residual),
            q_norm_sq: q_norm_sq as f32,
        }
    }
}

/// Computes RaBitQ unbiased estimates from an [`EstimatorSketch`] + a
/// pre-rotated [`EstimatorQuery`].
///
/// Stateless — the methods are associated functions. Kept as a type for
/// discoverability and to group the estimator formula in one place.
pub struct DistanceEstimator;

impl DistanceEstimator {
    /// Unbiased estimate of `⟨o_r, q_r⟩` (the inner product of the residuals).
    ///
    /// `⟨o_r, q_r⟩ = ‖o_r‖ · (⟨x̄, q'⟩ / ⟨x̄, o'⟩)`. Returns `0.0` when the
    /// stored `x_dot_o` is non-positive (degenerate / zero residual), which
    /// cannot happen for a non-zero input but keeps the call total.
    pub fn estimate_inner_product(sketch: &EstimatorSketch, query: &EstimatorQuery) -> f32 {
        let x_dot_o = sketch.side.x_dot_o;
        if x_dot_o <= 0.0 {
            return 0.0;
        }
        let code_dot_q = sketch.code_dot(&query.q_rotated_padded);
        // ⟨o, q_r⟩ ≈ ⟨x̄, q'⟩ / x_dot_o   (unit residual o)
        let inner_unit = code_dot_q / x_dot_o;
        sketch.side.residual_norm * inner_unit
    }

    /// Unbiased estimate of the **squared euclidean distance** `‖q_r − o_r‖²`.
    ///
    /// `= ‖q_r‖² + ‖o_r‖² − 2·⟨o_r, q_r⟩`, using the estimated inner product.
    /// This is the value the unbiasedness test checks.
    pub fn estimate_sq_distance(sketch: &EstimatorSketch, query: &EstimatorQuery) -> f32 {
        let ip = Self::estimate_inner_product(sketch, query);
        let o_norm = sketch.side.residual_norm;
        query.q_norm_sq + o_norm * o_norm - 2.0 * ip
    }

    /// The cheap **euclidean ranking key** for nearest-neighbour reranking:
    /// monotone in the estimated squared distance with the per-query constant
    /// `‖q_r‖²` dropped. Smaller = nearer. Equals `‖o_r‖² − 2·⟨o_r, q_r⟩`.
    ///
    /// Use this (not [`Self::estimate_sq_distance`]) for top-K reranking under a
    /// **euclidean** ground truth — it avoids adding the same `q_norm_sq` to
    /// every candidate. For a **cosine** ground truth (AETHER / the coverage
    /// harness), use [`Self::cosine_ranking_key`] instead.
    #[inline]
    pub fn ranking_key(sketch: &EstimatorSketch, query: &EstimatorQuery) -> f32 {
        let ip = Self::estimate_inner_product(sketch, query);
        let o_norm = sketch.side.residual_norm;
        o_norm * o_norm - 2.0 * ip
    }

    /// The cheap **cosine ranking key**: smaller = nearer in cosine distance.
    ///
    /// Cosine distance is `1 − ⟨o_r,q_r⟩ / (‖o_r‖·‖q_r‖)`. `‖q_r‖` is a
    /// per-query constant, so ranking by cosine distance ascending is ranking by
    /// `⟨o_r,q_r⟩ / ‖o_r‖` **descending**, i.e. by `−⟨o, q_r⟩` ascending. And
    /// `⟨o, q_r⟩ = ⟨x̄, q'⟩ / x_dot_o` — the unit-residual inner product, which
    /// needs **only the code and `x_dot_o`**, not even `residual_norm`. We
    /// return `−⟨o, q_r⟩` so "smaller = nearer" matches the euclidean key's
    /// convention.
    ///
    /// This is the correct key when the sketch is used (as in ADR-084) as an
    /// **angular** sensor graded against a cosine top-K: the 1-bit code is a
    /// rotated-angle estimator, and dividing by `x_dot_o` is the RaBitQ unbiased
    /// rescale of that angle's inner product.
    #[inline]
    pub fn cosine_ranking_key(sketch: &EstimatorSketch, query: &EstimatorQuery) -> f32 {
        let x_dot_o = sketch.side.x_dot_o;
        if x_dot_o <= 0.0 {
            return 0.0;
        }
        // ⟨o, q_r⟩ = ⟨x̄, q'⟩ / x_dot_o ; nearer in cosine ⇒ larger ⇒ negate.
        -(sketch.code_dot(&query.q_rotated_padded) / x_dot_o)
    }
}

/// A bank of [`EstimatorSketch`]es with stable IDs, reranked by the RaBitQ
/// **unbiased distance estimate** instead of raw Hamming.
///
/// All sketches share one [`Rotation`] (the index/query frame). The bank rotates
/// every inserted embedding and every query through it, so the estimator is
/// always computed in a consistent frame.
///
/// # Invariants
/// - All sketches share the bank's `embedding_dim` and `Rotation`.
/// - IDs are caller-assigned and stable.
#[derive(Debug, Clone)]
pub struct EstimatorBank {
    rotation: Rotation,
    entries: Vec<(u32, EstimatorSketch)>,
    embedding_dim: usize,
    /// Optional shared centroid subtracted from every embedding/query before
    /// rotation. `None` = zero-centroid (the default simplification).
    centroid: Option<Vec<f32>>,
}

impl EstimatorBank {
    /// Create an empty bank over `rotation`'s dimension and frame (zero-centroid).
    pub fn new(rotation: Rotation) -> Self {
        let embedding_dim = rotation.dim();
        Self {
            rotation,
            entries: Vec::new(),
            embedding_dim,
            centroid: None,
        }
    }

    /// Create an empty bank that subtracts `centroid` from every embedding and
    /// query before rotation (the paper's centroid path). Used by ADR-156 to
    /// measure the cost of the zero-centroid simplification.
    pub fn with_centroid(rotation: Rotation, centroid: Vec<f32>) -> Self {
        let embedding_dim = rotation.dim();
        Self {
            rotation,
            entries: Vec::new(),
            embedding_dim,
            centroid: Some(centroid),
        }
    }

    /// The rotation (index/query frame) this bank uses.
    #[inline]
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// Number of stored sketches.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Source embedding dimension.
    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Insert a raw embedding, sketching it (with side info) through the bank's
    /// rotation. The stored code and the queries share one rotated frame.
    pub fn insert_embedding(&mut self, id: u32, embedding: &[f32]) {
        let sketch = EstimatorSketch::from_embedding_centred(
            embedding,
            &self.rotation,
            self.centroid.as_deref(),
        );
        self.entries.push((id, sketch));
    }

    /// Insert a pre-built [`EstimatorSketch`] (must have been built with this
    /// bank's rotation; the caller is responsible for that).
    pub fn insert(&mut self, id: u32, sketch: EstimatorSketch) {
        self.entries.push((id, sketch));
    }

    /// Top-K nearest neighbours by the **RaBitQ unbiased estimate**, ascending
    /// by [`DistanceEstimator::ranking_key`]. Returns up to `k` `(id, key)`
    /// pairs. If `k == 0` or the bank is empty, returns empty. If the bank has
    /// fewer than `k`, returns all of them.
    ///
    /// The query is rotated **once**; every candidate then costs one
    /// length-`D` signed-sum dot product — the estimator is as cheap per
    /// candidate as Hamming plus a multiply.
    pub fn topk_estimated(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.topk_by(query, k, DistanceEstimator::ranking_key)
    }

    /// Top-K by the estimated **cosine** distance
    /// ([`DistanceEstimator::cosine_ranking_key`]) — the correct rerank when the
    /// sketch is graded against a cosine top-K (AETHER / the coverage harness).
    pub fn topk_estimated_cosine(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.topk_by(query, k, DistanceEstimator::cosine_ranking_key)
    }

    /// Shared top-K driver parameterised on the ranking-key function. Rotates
    /// the query once, scores every candidate with `key`, returns the `k`
    /// smallest keys ascending.
    fn topk_by(
        &self,
        query: &[f32],
        k: usize,
        key: fn(&EstimatorSketch, &EstimatorQuery) -> f32,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let q = EstimatorQuery::new_centred(query, &self.rotation, self.centroid.as_deref());
        let mut scored: Vec<(u32, f32)> = self
            .entries
            .iter()
            .map(|(id, sk)| (*id, key(sk, &q)))
            .collect();
        // Ascending by ranking key. Total ordering via partial_cmp with a
        // NaN-safe fallback (estimates are finite for finite input).
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Brute-force true inner product of two residuals (zero-centroid).
    fn true_inner(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn estimator_is_deterministic() {
        // Same (seed, dim) rotation + same vectors ⇒ identical estimate, twice.
        let dim = 64;
        let rot = Rotation::new(0xC0DE_1234_5678_9ABC, dim);
        let o: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.21).sin() + 0.3).collect();
        let qv: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).cos() - 0.2).collect();

        let s1 = EstimatorSketch::from_embedding(&o, &rot);
        let s2 = EstimatorSketch::from_embedding(&o, &rot);
        let q1 = EstimatorQuery::new(&qv, &rot);
        let q2 = EstimatorQuery::new(&qv, &Rotation::new(0xC0DE_1234_5678_9ABC, dim));

        let e1 = DistanceEstimator::estimate_inner_product(&s1, &q1);
        let e2 = DistanceEstimator::estimate_inner_product(&s2, &q2);
        assert_eq!(e1, e2, "estimator must be deterministic for a fixed seed");

        // Bank topk is deterministic too.
        let mut bank = EstimatorBank::new(Rotation::new(7, dim));
        for id in 0..16u32 {
            let v: Vec<f32> = (0..dim).map(|i| ((i + id as usize) as f32 * 0.07).sin()).collect();
            bank.insert_embedding(id, &v);
        }
        let a = bank.topk_estimated(&qv, 5);
        let b = bank.topk_estimated(&qv, 5);
        assert_eq!(a, b, "topk_estimated must be deterministic");
    }

    #[test]
    fn estimator_unbiased_on_fixture() {
        // The core unbiasedness claim: averaging the estimate of ⟨o_r, q_r⟩ over
        // MANY random rotation seeds converges to the true inner product.
        //
        // Hand-checkable small case: two fixed vectors, known true inner
        // product, average the estimator over many seeds and assert it lands
        // within a tolerance that a BIASED estimator would miss.
        let dim = 32;
        let o: Vec<f32> = (0..dim).map(|i| ((i % 7) as f32 - 3.0) * 0.4 + 0.5).collect();
        let qv: Vec<f32> = (0..dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3 - 0.1).collect();
        let truth = true_inner(&o, &qv);

        let n_seeds = 4000u64;
        let mut acc = 0.0f64;
        for seed in 0..n_seeds {
            let rot = Rotation::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xABCD, dim);
            let sk = EstimatorSketch::from_embedding(&o, &rot);
            let q = EstimatorQuery::new(&qv, &rot);
            acc += DistanceEstimator::estimate_inner_product(&sk, &q) as f64;
        }
        let mean = (acc / n_seeds as f64) as f32;

        // Tolerance scaled to the magnitudes involved. The estimator is
        // unbiased, so the Monte-Carlo mean must be CLOSE to truth; a sign-only
        // Hamming proxy (or a biased rescale) would be systematically off.
        let scale = l2(&o) * l2(&qv);
        let tol = 0.06 * scale; // ~6% of the ‖o‖‖q‖ envelope over 4000 seeds
        assert!(
            (mean - truth).abs() < tol,
            "estimator biased: mean={mean:.4} truth={truth:.4} tol={tol:.4} (scale={scale:.4})"
        );
    }

    #[test]
    fn estimator_self_distance_is_small() {
        // Estimating the distance of a vector to itself should be ~0 (the
        // estimate of ⟨o,o⟩ ≈ ‖o‖², so ‖q-o‖² ≈ 0). Not exactly 0 (1-bit code),
        // but small relative to ‖o‖².
        let dim = 128;
        let rot = Rotation::new(0xBEEF_CAFE, dim);
        let o: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).cos() + 0.2).collect();
        let sk = EstimatorSketch::from_embedding(&o, &rot);
        let q = EstimatorQuery::new(&o, &rot);
        let sq = DistanceEstimator::estimate_sq_distance(&sk, &q);
        let o_norm_sq = l2(&o) * l2(&o);
        assert!(
            sq.abs() < 0.25 * o_norm_sq,
            "self sq-distance estimate {sq:.3} too large vs ‖o‖²={o_norm_sq:.3}"
        );
    }

    #[test]
    fn side_info_is_eight_bytes() {
        assert_eq!(EstimatorSketch::SIDE_INFO_BYTES, 8);
    }

    #[test]
    fn x_dot_o_in_unit_range() {
        // ⟨x̄, o'⟩ ∈ (0, 1] for any non-zero input (it's the cosine between the
        // rotated residual and its nearest hypercube corner).
        let dim = 96;
        let rot = Rotation::new(0x1357_9BDF, dim);
        for s in 0..20u32 {
            let v: Vec<f32> = (0..dim).map(|i| (((i + s as usize) * 13 % 23) as f32 - 11.0) * 0.2).collect();
            let sk = EstimatorSketch::from_embedding(&v, &rot);
            let x = sk.side_info().x_dot_o;
            assert!(x > 0.0 && x <= 1.0 + 1e-5, "x_dot_o out of (0,1]: {x}");
        }
    }

    #[test]
    fn zero_input_does_not_panic() {
        let dim = 64;
        let rot = Rotation::new(1, dim);
        let sk = EstimatorSketch::from_embedding(&vec![0.0f32; dim], &rot);
        assert_eq!(sk.residual_norm(), 0.0);
        let q = EstimatorQuery::new(&vec![1.0f32; dim], &rot);
        // No divide-by-zero; degenerate estimate is 0 inner product.
        assert_eq!(DistanceEstimator::estimate_inner_product(&sk, &q), 0.0);
    }

    #[test]
    fn centroid_path_self_query_ranks_self_first() {
        // The paper-faithful centroid path (o_r = o − c) must still rank a
        // stored vector first when queried with itself, with a shared centroid.
        let dim = 64;
        let rot = Rotation::new(0x9999, dim);
        let centroid: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut bank = EstimatorBank::with_centroid(rot, centroid.clone());
        let target: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos() + 1.5).collect();
        bank.insert_embedding(7, &target);
        for id in 0..24u32 {
            let v: Vec<f32> = (0..dim)
                .map(|i| ((i as f32 + id as f32) * 0.09).sin() + 1.4)
                .collect();
            bank.insert_embedding(id, &v);
        }
        let top = bank.topk_estimated_cosine(&target, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 7, "centroid-path self-query should rank self first");
    }

    #[test]
    fn centroid_zero_matches_default() {
        // from_embedding_centred(None) must be byte-identical to from_embedding.
        let dim = 48;
        let rot = Rotation::new(0x4242, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin() - 0.1).collect();
        let a = EstimatorSketch::from_embedding(&v, &rot);
        let b = EstimatorSketch::from_embedding_centred(&v, &rot, None);
        assert_eq!(a.residual_norm(), b.residual_norm());
        assert_eq!(a.side_info(), b.side_info());
    }

    #[test]
    fn bank_self_query_ranks_self_first() {
        // A bank queried with one of its own stored vectors should rank that id
        // first under the estimator (its estimated distance to itself is the
        // smallest).
        let dim = 128;
        let rot = Rotation::new(0xABCD_1234, dim);
        let mut bank = EstimatorBank::new(rot);
        let target: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.19).sin() * 2.0).collect();
        bank.insert_embedding(99, &target);
        for id in 0..32u32 {
            let v: Vec<f32> = (0..dim)
                .map(|i| ((i as f32 + id as f32 * 3.0) * 0.05).cos())
                .collect();
            bank.insert_embedding(id, &v);
        }
        let top = bank.topk_estimated(&target, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 99, "self-query should rank the stored self first");
    }
}
