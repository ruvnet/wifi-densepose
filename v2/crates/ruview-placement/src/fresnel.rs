//! Fresnel-zone geometry for link observability (ADR-308 §2).
//!
//! **SYNTHETIC / L0.** WiFi sensing perturbs a link when the target sits inside
//! the link's first Fresnel zone. This module implements that geometry as a
//! deliberately simple, documented analytic model — the first Fresnel radius and
//! a clearance factor for a point relative to a link line — not real RF and not a
//! measurement. Everything is deterministic and allocation-free.

/// First Fresnel-zone radius (metres) at a point that splits the path into
/// longitudinal legs `d1` and `d2`:
///
/// `F1 = sqrt(λ · d1 · d2 / (d1 + d2))`.
///
/// Returns `0.0` for non-finite or non-physical inputs (never `NaN`/`inf`); the
/// caller treats a zero radius as "no clearance information", not a divide-by-zero.
/// At the midpoint (`d1 == d2 == L/2`) this reduces to `0.5·sqrt(λ·L)`, the
/// known analytic maximum used in tests.
#[must_use]
pub fn fresnel_radius(wavelength_m: f64, d1: f64, d2: f64) -> f64 {
    if !(wavelength_m.is_finite() && d1.is_finite() && d2.is_finite()) {
        return 0.0;
    }
    let sum = d1 + d2;
    if wavelength_m <= 0.0 || d1 < 0.0 || d2 < 0.0 || sum <= 0.0 {
        return 0.0;
    }
    let r = wavelength_m * d1 * d2 / sum;
    if r.is_finite() && r >= 0.0 {
        r.sqrt()
    } else {
        0.0
    }
}

/// Clearance factor in `[0, 1]` for point `p` relative to the link line `a → b`,
/// at wavelength `λ`.
///
/// - `1.0` on the link line, falling linearly to `0.0` at the first Fresnel-zone
///   boundary and `0.0` beyond it.
/// - `0.0` when `p` does not project *between* the endpoints (a target off the
///   ends of a link is not in its sensing corridor).
/// - `0.0` for a degenerate (coincident-endpoint) link.
///
/// SYNTHETIC geometry, not an RF measurement. Deterministic.
#[must_use]
pub fn link_clearance(a: (f64, f64), b: (f64, f64), p: (f64, f64), wavelength_m: f64) -> f64 {
    let abx = b.0 - a.0;
    let aby = b.1 - a.1;
    let len2 = abx * abx + aby * aby;
    if !len2.is_finite() || len2 <= 1e-12 {
        return 0.0;
    }
    let apx = p.0 - a.0;
    let apy = p.1 - a.1;
    let t = (apx * abx + apy * aby) / len2;
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return 0.0;
    }
    let len = len2.sqrt();
    let d1 = t * len;
    let d2 = (1.0 - t) * len;

    // Perpendicular distance from p to the foot of the projection on the line.
    let foot_x = a.0 + t * abx;
    let foot_y = a.1 + t * aby;
    let h = ((p.0 - foot_x).powi(2) + (p.1 - foot_y).powi(2)).sqrt();

    let f1 = fresnel_radius(wavelength_m, d1, d2);
    if !f1.is_finite() || f1 <= 0.0 || !h.is_finite() {
        return 0.0;
    }
    let clearance = 1.0 - h / f1;
    if clearance.is_finite() {
        clearance.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_radius_matches_analytic_midpoint() {
        // Midpoint of a path of length L: F1 = 0.5·sqrt(λ·L).
        let wavelength = 0.125_f64; // ~2.4 GHz
        let l = 4.0_f64;
        let (d1, d2) = (l / 2.0, l / 2.0);
        let expected = 0.5 * (wavelength * l).sqrt();
        assert!((fresnel_radius(wavelength, d1, d2) - expected).abs() < 1e-12);
    }

    #[test]
    fn fresnel_radius_is_zero_for_non_physical_input() {
        assert_eq!(fresnel_radius(f64::NAN, 1.0, 1.0), 0.0);
        assert_eq!(fresnel_radius(-1.0, 1.0, 1.0), 0.0);
        assert_eq!(fresnel_radius(0.125, 0.0, 0.0), 0.0);
        assert_eq!(fresnel_radius(0.125, -1.0, 2.0), 0.0);
    }

    #[test]
    fn clearance_is_one_on_the_line_and_zero_off_the_ends() {
        let a = (0.0, 0.0);
        let b = (4.0, 0.0);
        // On the line at the midpoint.
        assert!((link_clearance(a, b, (2.0, 0.0), 0.125) - 1.0).abs() < 1e-12);
        // Off the end of the segment: no clearance.
        assert_eq!(link_clearance(a, b, (5.0, 0.0), 0.125), 0.0);
        // Far off the line (perpendicular ≫ Fresnel radius): no clearance.
        assert_eq!(link_clearance(a, b, (2.0, 3.0), 0.125), 0.0);
    }

    #[test]
    fn degenerate_link_has_zero_clearance() {
        assert_eq!(link_clearance((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), 0.125), 0.0);
    }
}
