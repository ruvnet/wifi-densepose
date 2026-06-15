//! Geometric Dilution of Precision (GDOP) for a constellation of observers.
//!
//! GDOP quantifies how observer geometry amplifies measurement error into
//! position-estimate error. Build the geometry matrix `H` of unit
//! line-of-sight (LOS) vectors from each observer to the target, form the
//! normal matrix `HᵀH`, invert it, and take `GDOP = sqrt(trace((HᵀH)⁻¹))`.
//!
//! For the 2-D (x, y) localization case `H` is `N×2` and `HᵀH` is `2×2`, so a
//! closed-form 2×2 inverse suffices (no linear-algebra dependency needed).
//!
//! Lower GDOP = better geometry: observers spread ~120° apart around the target
//! give low GDOP; (near-)collinear observers give a singular/ill-conditioned
//! `HᵀH` → GDOP → ∞.

use crate::types::Position3D;

/// Geometric Dilution of Precision (2-D) for `observers` viewing a `target`.
///
/// Lower = better geometry. A ~120° constellation → low GDOP; collinear → very
/// large (→∞). Returns `None` if fewer than two observers, if any observer is
/// coincident with the target (undefined LOS), or if the geometry is singular
/// / degenerate (collinear) so `HᵀH` is not invertible.
pub fn gdop(observers: &[Position3D], target: &Position3D) -> Option<f64> {
    if observers.len() < 2 {
        return None;
    }

    // Accumulate HᵀH directly (2×2 symmetric) from unit LOS vectors.
    // Row i of H is the unit vector from target → observer i in (x, y).
    let mut a = 0.0; // sum ux*ux
    let mut b = 0.0; // sum ux*uy
    let mut d = 0.0; // sum uy*uy

    for obs in observers {
        let dx = obs.x - target.x;
        let dy = obs.y - target.y;
        let range = (dx * dx + dy * dy).sqrt();
        if range < 1e-9 {
            // Observer on top of the target → LOS undefined.
            return None;
        }
        let ux = dx / range;
        let uy = dy / range;
        a += ux * ux;
        b += ux * uy;
        d += uy * uy;
    }

    // Determinant of HᵀH = [[a, b], [b, d]].
    let det = a * d - b * b;
    if det.abs() < 1e-12 {
        // Singular: observers are (near-)collinear with the target.
        return None;
    }

    // (HᵀH)⁻¹ = 1/det * [[d, -b], [-b, a]]; trace = (d + a) / det.
    let trace_inv = (a + d) / det;
    if trace_inv <= 0.0 || !trace_inv.is_finite() {
        return None;
    }
    Some(trace_inv.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Position3D {
        Position3D { x, y, z: 0.0 }
    }

    #[test]
    fn test_triangle_lower_than_collinear() {
        let target = p(0.0, 0.0);
        // Three observers at 120° around the target, radius 10.
        let r = 10.0;
        let triangle = [
            p(r * 0.0_f64.cos(), r * 0.0_f64.sin()),
            p(
                r * (2.0 * std::f64::consts::PI / 3.0).cos(),
                r * (2.0 * std::f64::consts::PI / 3.0).sin(),
            ),
            p(
                r * (4.0 * std::f64::consts::PI / 3.0).cos(),
                r * (4.0 * std::f64::consts::PI / 3.0).sin(),
            ),
        ];
        // Three nearly-collinear observers (tiny y perturbation to stay invertible).
        let near_collinear = [p(5.0, 0.01), p(10.0, 0.0), p(15.0, 0.01)];

        let tri = gdop(&triangle, &target).expect("triangle finite GDOP");
        let col = gdop(&near_collinear, &target).expect("near-collinear finite GDOP");
        assert!(tri.is_finite(), "triangle GDOP must be finite: {tri}");
        assert!(
            tri < col,
            "120° constellation should have lower GDOP than near-collinear: tri={tri}, col={col}"
        );
    }

    #[test]
    fn test_collinear_degenerate() {
        let target = p(0.0, 0.0);
        // Perfectly collinear observers along +x → singular HᵀH.
        let collinear = [p(5.0, 0.0), p(10.0, 0.0), p(20.0, 0.0)];
        let g = gdop(&collinear, &target);
        assert!(
            g.is_none() || g.unwrap() > 1e6,
            "perfectly collinear geometry must be None or huge, got {g:?}"
        );
    }

    #[test]
    fn test_single_observer_none() {
        let target = p(0.0, 0.0);
        assert!(gdop(&[p(5.0, 5.0)], &target).is_none());
        assert!(gdop(&[], &target).is_none());
    }
}
