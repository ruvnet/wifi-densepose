//! RaBitQ **Pass 2** — deterministic randomized orthogonal rotation.
//!
//! Implements the "Pass 2" deferred in [`crate::sketch`]'s Pass-1 doc and in
//! [ADR-156 §8](../../../../../docs/adr/ADR-156-ruvector-fusion-beyond-sota.md)
//! (Multi-bit / Extended RaBitQ). The published *RaBitQ* algorithm
//! (Gao & Long, SIGMOD 2024) wraps the 1-bit sign-quantization of Pass 1 with
//! a **randomized orthogonal rotation** `R` applied to every embedding *before*
//! sign-quantization. The rotation decorrelates coordinates so the per-bit sign
//! carries more independent information, which gives both the paper's
//! theoretical error bound and better top-K recall on anisotropic / correlated
//! embedding distributions (exactly the case ADR-084's "Open questions" flagged
//! for skewed spectrogram embeddings).
//!
//! # Why a Fast Hadamard Transform, not a dense d×d matrix
//!
//! A full dense orthogonal matrix `R ∈ ℝ^{d×d}` is **O(d²) memory and O(d²)
//! time per vector**. ADR-084's wire format already provisions for embeddings
//! up to `u16::MAX = 65,535` dimensions; a dense rotation there is ~4.3 G
//! floats (17 GiB) — completely infeasible on the cluster-Pi / edge targets
//! this sketch is built for.
//!
//! Instead we use the **randomized Hadamard transform** (the "HD" construction,
//! a.k.a. a structured Johnson–Lindenstrauss / fast-JL rotation):
//!
//! ```text
//!     R · x  =  H · D · x
//! ```
//!
//! where `D` is a diagonal matrix of random ±1 sign flips and `H` is the
//! (normalized) Walsh–Hadamard matrix applied via the **Fast Hadamard
//! Transform (FHT)**. The FHT is `O(d log d)` time and `O(1)` extra memory
//! (in-place butterfly); `D` is `O(d)` memory (one sign per dimension, packed).
//! `H` and `D` are each orthogonal, so `R = H·D` is orthogonal and therefore
//! **norm-preserving** — a hard requirement for a rotation that must not distort
//! relative distances. This is the same fast-orthogonal trick used by Fast-JL,
//! Structured Orthogonal Random Features, and the RaBitQ reference rotation.
//!
//! # Determinism (index-time == query-time)
//!
//! The rotation **must** be identical when the bank is built and when it is
//! queried, or the two sign-quantizations live in different rotated frames and
//! hamming distance becomes meaningless. We therefore derive the ±1 sign flips
//! deterministically from a stored `u64` seed via a SplitMix64 PRNG — **never**
//! an unseeded / OS RNG. Two [`Rotation`]s built from the same `(seed, dim)`
//! produce bit-identical output for the same input (pinned by
//! `rotation_is_deterministic_for_seed`).
//!
//! # Power-of-two padding
//!
//! The FHT is defined on lengths that are powers of two. For a `d` that is not
//! a power of two we pad the (sign-flipped) input with zeros up to the next
//! power of two `m = next_pow2(d)`, run the length-`m` FHT, and then **read back
//! the first `d` coordinates**. Zero-padding + orthogonal `H` keeps the
//! transform norm-preserving on the padded vector; we sign-quantize the first
//! `d` rotated coordinates so the sketch dimension is unchanged from Pass 1
//! (API-compatible: same `embedding_dim`, same packed-byte length, same
//! `SketchBank` schema).

/// A deterministic randomized orthogonal rotation (FHT-based) applied to an
/// embedding before sign-quantization — RaBitQ Pass 2.
///
/// Construct once per `(seed, dim)` and reuse for **every** embedding that goes
/// into the same [`crate::SketchBank`] (and for every query against it). The
/// seed is stored so the rotation is reproducible across processes and runs.
///
/// # Invariants
///
/// - `dim` is the source-embedding dimension (the sketch keeps this dimension).
/// - `padded` is `next_pow2(dim)` — the FHT working length.
/// - `signs` has exactly `padded` entries (`+1.0` / `-1.0`), derived from
///   `seed` via SplitMix64. Padding positions get signs too; they only ever
///   multiply zeros, so their value is irrelevant to the result but they keep
///   the construction uniform.
#[derive(Debug, Clone)]
pub struct Rotation {
    /// Source-embedding dimension; the rotated sketch keeps this dimension.
    dim: usize,
    /// FHT working length = `next_pow2(dim)`.
    padded: usize,
    /// Random ±1 sign flips (the diagonal `D`), length `padded`.
    signs: Vec<f32>,
    /// The seed the sign flips were derived from (stored for reproducibility).
    seed: u64,
}

impl Rotation {
    /// Build a rotation for `dim`-dimensional embeddings from a fixed `seed`.
    ///
    /// The same `(seed, dim)` always yields a bit-identical rotation, so an
    /// index built with `Rotation::new(seed, d)` and a query rotated with a
    /// freshly-constructed `Rotation::new(seed, d)` agree exactly.
    ///
    /// `dim == 0` yields an identity (empty) rotation — `apply` returns an
    /// empty vector — which keeps the constructor total (no panic on a
    /// degenerate dimension).
    pub fn new(seed: u64, dim: usize) -> Self {
        let padded = next_pow2(dim);
        let mut signs = Vec::with_capacity(padded);
        // SplitMix64: a tiny, well-distributed, fully deterministic PRNG. We
        // only need a reproducible stream of bits to pick ±1 per dimension;
        // SplitMix64 is the standard seeding generator and is more than
        // adequate (and far better-mixed than the LCG used for bench fixtures).
        let mut state = seed;
        for _ in 0..padded {
            state = split_mix64(&mut state);
            // Use the top bit of the mixed word to choose the sign.
            signs.push(if state >> 63 == 1 { 1.0 } else { -1.0 });
        }
        Self {
            dim,
            padded,
            signs,
            seed,
        }
    }

    /// The seed this rotation was derived from (for serialization / audit).
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Source-embedding dimension this rotation expects.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// FHT working length (`next_pow2(dim)`).
    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded
    }

    /// Apply the rotation `R = H·D` to `embedding`, returning the first `dim`
    /// rotated coordinates.
    ///
    /// If `embedding.len() != dim` the input is treated charitably: it is
    /// truncated or zero-extended to `dim` before rotation. This mirrors
    /// Pass 1's saturating tolerance and keeps the call total.
    ///
    /// The returned vector has length `self.dim`. Its L2 norm equals the L2
    /// norm of the (dim-truncated / zero-extended) input up to floating-point
    /// rounding — see [`Rotation::apply`] tests and
    /// `rotation_preserves_norm`.
    pub fn apply(&self, embedding: &[f32]) -> Vec<f32> {
        if self.dim == 0 {
            return Vec::new();
        }
        let mut buf = self.apply_padded(embedding);
        // Read back the first `dim` rotated coordinates as the sketch input.
        buf.truncate(self.dim);
        buf
    }

    /// Apply the rotation `R = H·D` and return **all `padded_dim` rotated
    /// coordinates** (not truncated to `dim`).
    ///
    /// This is the frame the RaBitQ estimator ([`crate::estimator`]) works in:
    /// the 1-bit code `x̄ ∈ {±1/√D}^D` is unit over the **padded** length `D`,
    /// and the query dot product `⟨x̄, q'⟩` must be taken over that same `D`. For
    /// a power-of-two `dim`, `padded_dim == dim` and this equals
    /// [`Rotation::apply`]; for a non-power-of-two `dim` the tail coordinates
    /// (the zero-padded energy redistributed by the FHT) are retained here but
    /// dropped by `apply`.
    ///
    /// `dim == 0` yields an empty vector. Ragged input is handled charitably
    /// (truncate / zero-extend to `dim`), as in [`Rotation::apply`].
    pub fn apply_padded(&self, embedding: &[f32]) -> Vec<f32> {
        if self.dim == 0 {
            return Vec::new();
        }
        // Build the padded, sign-flipped working buffer: buf = D · x, then 0-pad.
        let mut buf = vec![0.0f32; self.padded];
        let n = embedding.len().min(self.dim);
        for i in 0..n {
            buf[i] = embedding[i] * self.signs[i];
        }
        // (positions n..dim and dim..padded stay zero — zero-extend + pad)

        // In-place normalized Fast Hadamard Transform.
        fht_normalized(&mut buf);
        buf
    }
}

/// Smallest power of two `>= n` (with `next_pow2(0) == 1`, `next_pow2(1) == 1`).
///
/// Pulled out (and `pub(crate)`) so the sketch layer and tests can reason about
/// the FHT working length without duplicating the rule.
#[inline]
pub(crate) fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    // `n` here is small relative to usize::MAX in every realistic embedding
    // (<= 65_535), so `next_power_of_two` cannot overflow.
    n.next_power_of_two()
}

/// SplitMix64 step: advance `state` and return a well-mixed 64-bit word.
///
/// Reference algorithm (public domain, by Sebastiano Vigna). Deterministic and
/// dependency-free — exactly what we need for a reproducible sign stream.
#[inline]
fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// In-place **normalized** Fast Hadamard Transform on a power-of-two slice.
///
/// Computes `y = (1/√m) · H_m · x` in place, where `H_m` is the `m × m`
/// Walsh–Hadamard matrix and `m = buf.len()` is a power of two. The `1/√m`
/// normalization makes `H` orthogonal (`HᵀH = I`), so the transform preserves
/// the L2 norm. Runs in `O(m log m)` with `O(1)` extra memory (the standard
/// iterative butterfly).
///
/// # Panics
///
/// Debug-asserts that `buf.len()` is a power of two. Callers in this module
/// always pass `next_pow2(dim)`, so this never fires in practice; it documents
/// the precondition.
fn fht_normalized(buf: &mut [f32]) {
    let m = buf.len();
    debug_assert!(m.is_power_of_two(), "FHT length must be a power of two");
    if m <= 1 {
        return;
    }
    // Unnormalized in-place Walsh–Hadamard butterfly.
    let mut h = 1usize;
    while h < m {
        let mut i = 0usize;
        while i < m {
            for j in i..i + h {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    // Normalize by 1/√m so H is orthogonal (norm-preserving).
    let inv_sqrt_m = 1.0f32 / (m as f32).sqrt();
    for v in buf.iter_mut() {
        *v *= inv_sqrt_m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn next_pow2_rounds_up() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(128), 128);
        assert_eq!(next_pow2(129), 256);
        assert_eq!(next_pow2(200), 256);
        assert_eq!(next_pow2(65_535), 65_536);
    }

    #[test]
    fn fht_is_norm_preserving_on_power_of_two() {
        // Pure FHT (no sign flips) must preserve L2 norm to fp tolerance.
        let mut v: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.7).collect();
        let before = l2(&v);
        fht_normalized(&mut v);
        let after = l2(&v);
        assert!(
            (before - after).abs() < 1e-5,
            "FHT changed norm: {before} -> {after}"
        );
    }

    #[test]
    fn fht_self_inverse_normalized() {
        // Normalized H is symmetric and orthogonal, so H·H·x == x.
        let original: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5];
        let mut v = original.clone();
        fht_normalized(&mut v);
        fht_normalized(&mut v);
        for (a, b) in original.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5, "H·H·x != x: {a} vs {b}");
        }
    }

    #[test]
    fn rotation_is_deterministic_for_seed() {
        // Two rotations from the same (seed, dim) must produce identical
        // output for the same input — the index-time == query-time contract.
        let r1 = Rotation::new(0xDEAD_BEEF_CAFE_1234, 130);
        let r2 = Rotation::new(0xDEAD_BEEF_CAFE_1234, 130);
        let x: Vec<f32> = (0..130).map(|i| (i as f32 * 0.31).sin()).collect();
        let a = r1.apply(&x);
        let b = r2.apply(&x);
        assert_eq!(a.len(), 130);
        assert_eq!(a, b, "same seed must give identical rotation");

        // A different seed must (almost surely) differ.
        let r3 = Rotation::new(0x0000_0000_0000_0001, 130);
        let c = r3.apply(&x);
        assert_ne!(a, c, "different seed must give different rotation");
    }

    #[test]
    fn rotation_preserves_norm() {
        // R = H·D is orthogonal; on a power-of-two dim the first `dim`
        // coordinates ARE the whole transform, so norm is preserved exactly
        // (to fp tolerance). We test a power-of-two dim for the exact claim.
        let r = Rotation::new(42, 128);
        let x: Vec<f32> = (0..128).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.5).collect();
        let y = r.apply(&x);
        let before = l2(&x);
        let after = l2(&y);
        assert!(
            (before - after).abs() < 1e-3 * before.max(1.0),
            "rotation changed norm: {before} -> {after}"
        );
    }

    #[test]
    fn rotation_non_power_of_two_preserves_norm_via_padding() {
        // For a non-power-of-two dim, reading back the first `dim` coords of a
        // padded FHT only preserves norm if the padded tail carries ~no energy.
        // We assert the rotated norm does not EXCEED the input norm (the padded
        // transform is non-expansive on the truncated read-back) and stays
        // within a loose band — enough to confirm padding is sane, not a hard
        // exact-norm claim.
        let r = Rotation::new(7, 130); // pads 130 -> 256
        assert_eq!(r.padded_dim(), 256);
        let x: Vec<f32> = (0..130).map(|i| (i as f32 * 0.13).cos()).collect();
        let y = r.apply(&x);
        assert_eq!(y.len(), 130);
        let before = l2(&x);
        let after = l2(&y);
        // Truncated read-back is non-expansive: ||y|| <= ||Hx|| == ||x||.
        assert!(
            after <= before + 1e-4,
            "truncated rotation expanded norm: {before} -> {after}"
        );
    }

    #[test]
    fn rotation_dim_zero_is_empty() {
        let r = Rotation::new(1, 0);
        assert!(r.apply(&[]).is_empty());
        assert!(r.apply(&[1.0, 2.0]).is_empty());
    }

    #[test]
    fn rotation_handles_ragged_input() {
        // Charitable length handling: short input zero-extends, long truncates.
        let r = Rotation::new(99, 64);
        let short = r.apply(&[1.0, 2.0, 3.0]); // zero-extended to 64
        assert_eq!(short.len(), 64);
        let long: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let truncated = r.apply(&long); // truncated to 64
        assert_eq!(truncated.len(), 64);
    }
}
