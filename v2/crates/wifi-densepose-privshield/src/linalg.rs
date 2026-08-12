//! Minimal, dependency-free vector algebra over `f32` slices.
//!
//! VEIL deliberately avoids `ndarray`/BLAS: the vectors are short (tens of
//! elements — a flattened compressed-beamforming angle report), the crate is
//! a WASM-ready leaf, and keeping the math inline makes the energy-conservation
//! proof in [`crate::compliance`] auditable line-by-line.

/// Euclidean inner product. Panics if lengths differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared L2 norm.
#[must_use]
pub fn norm_sq(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// L2 norm.
#[must_use]
pub fn norm(a: &[f32]) -> f32 {
    norm_sq(a).sqrt()
}

/// Squared Euclidean distance. Panics if lengths differ.
#[must_use]
pub fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dist_sq: length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Scale in place.
pub fn scale_inplace(a: &mut [f32], k: f32) {
    for x in a.iter_mut() {
        *x *= k;
    }
}

/// Normalize `a` to a target L2 norm in place. No-op if `a` is (near) zero.
pub fn set_norm_inplace(a: &mut [f32], target: f32) {
    let n = norm(a);
    if n > 1e-12 {
        scale_inplace(a, target / n);
    }
}

/// Apply a Givens rotation to coordinates `(i, j)` of `v` by angle `theta`.
///
/// A Givens rotation is the exact primitive 802.11 compressed beamforming
/// feedback is built from (the ψ/φ angles a beamformee reports). It is an
/// **orthogonal** operation: it preserves `‖v‖` to machine precision, which is
/// precisely why composing extra keyed Givens rotations adds *no transmit
/// energy* — the compliance argument in [`crate::compliance`].
pub fn apply_givens(v: &mut [f32], i: usize, j: usize, theta: f32) {
    debug_assert!(i < v.len() && j < v.len() && i != j);
    let (c, s) = (theta.cos(), theta.sin());
    let (vi, vj) = (v[i], v[j]);
    v[i] = c * vi - s * vj;
    v[j] = s * vi + c * vj;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn givens_preserves_norm() {
        let mut v = vec![0.3, -1.2, 0.7, 2.1, -0.5];
        let before = norm(&v);
        apply_givens(&mut v, 1, 3, 0.9);
        apply_givens(&mut v, 0, 4, -2.3);
        apply_givens(&mut v, 2, 3, 1.1);
        let after = norm(&v);
        assert!((before - after).abs() < 1e-5, "{before} vs {after}");
    }

    #[test]
    fn givens_is_invertible() {
        let orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut v = orig.clone();
        apply_givens(&mut v, 0, 2, 0.7);
        apply_givens(&mut v, 0, 2, -0.7);
        for (a, b) in orig.iter().zip(&v) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
