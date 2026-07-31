//! Image-method multipath ray tracer for shoebox rooms (ADR-276 §3).
//!
//! Allen–Berkley mirror images up to reflection order 2 plus single-bounce
//! bistatic scattering off each person. The channel at frequency `f` is
//!
//! ```text
//! H(f) = Σ_paths Γ_p · (c/f)/(4π) · s_p · e^{-j·2πf·d_p/c}
//! ```
//!
//! with `s_p = 1/d` for wall paths and `s_p = √(σ/4π)/(d₁·d₂)` for person
//! scattering (bistatic radar equation, amplitude form). Doppler is never
//! injected: it emerges from the person's path length changing between
//! snapshots.

use num_complex::Complex64;

use super::room::RoomSpec;

const C: f64 = 299_792_458.0;

/// One propagation path.
#[derive(Debug, Clone, Copy)]
pub struct PathContribution {
    /// Total geometric path length (metres) — sets delay and phase.
    pub distance_m: f64,
    /// Amplitude scale multiplying `λ/(4π)`: `1/d` for wall paths,
    /// `√(σ/4π)/(d₁·d₂)` for scatterers.
    pub amp_scale: f64,
    /// Product of complex reflection coefficients along the path
    /// (`Γ^order`, evaluated at the carrier).
    pub reflection: Complex64,
    /// Number of wall bounces (0 = direct or scatterer path).
    pub order: usize,
}

/// Enumerates all wall-image paths of order ≤ `max_order` plus person
/// scattering paths, for the room state at time `t` seconds.
#[must_use]
pub fn enumerate_paths(
    room: &RoomSpec,
    tx: [f64; 3],
    rx: [f64; 3],
    carrier_hz: f64,
    t: f64,
    max_order: usize,
) -> Vec<PathContribution> {
    let gamma = room.wall_material.reflection_coefficient(carrier_hz);
    let mut paths = Vec::new();

    // Allen–Berkley images: per axis, sign ∈ {+, −} and lattice shift
    // n ∈ {−1, 0, 1}; bounce count is |2n| for +, |2n−1| for −.
    for sx in [1i64, -1] {
        for nx in -1i64..=1 {
            let bx = if sx == 1 { (2 * nx).unsigned_abs() } else { (2 * nx - 1).unsigned_abs() };
            if bx as usize > max_order {
                continue;
            }
            for sy in [1i64, -1] {
                for ny in -1i64..=1 {
                    let by =
                        if sy == 1 { (2 * ny).unsigned_abs() } else { (2 * ny - 1).unsigned_abs() };
                    if (bx + by) as usize > max_order {
                        continue;
                    }
                    for sz in [1i64, -1] {
                        for nz in -1i64..=1 {
                            let bz = if sz == 1 {
                                (2 * nz).unsigned_abs()
                            } else {
                                (2 * nz - 1).unsigned_abs()
                            };
                            let order = (bx + by + bz) as usize;
                            if order > max_order {
                                continue;
                            }
                            let img = [
                                sx as f64 * tx[0] + 2.0 * nx as f64 * room.size[0],
                                sy as f64 * tx[1] + 2.0 * ny as f64 * room.size[1],
                                sz as f64 * tx[2] + 2.0 * nz as f64 * room.size[2],
                            ];
                            let d = ((img[0] - rx[0]).powi(2)
                                + (img[1] - rx[1]).powi(2)
                                + (img[2] - rx[2]).powi(2))
                            .sqrt();
                            if d < 1e-9 {
                                continue;
                            }
                            let reflection = if order == 0 {
                                Complex64::new(1.0, 0.0)
                            } else {
                                gamma.powu(order as u32)
                            };
                            // Reflections with |Γ|=0 contribute nothing.
                            if order > 0 && reflection.norm() < 1e-15 {
                                continue;
                            }
                            paths.push(PathContribution {
                                distance_m: d,
                                amp_scale: 1.0 / d,
                                reflection,
                                order,
                            });
                        }
                    }
                }
            }
        }
    }

    // Person scatterers (single bounce TX → person → RX).
    for person in &room.people {
        let p = person.position_at(t);
        let d1 = ((p[0] - tx[0]).powi(2) + (p[1] - tx[1]).powi(2) + (p[2] - tx[2]).powi(2)).sqrt();
        let d2 = ((p[0] - rx[0]).powi(2) + (p[1] - rx[1]).powi(2) + (p[2] - rx[2]).powi(2)).sqrt();
        if d1 < 1e-9 || d2 < 1e-9 {
            continue;
        }
        paths.push(PathContribution {
            distance_m: d1 + d2,
            amp_scale: (person.rcs_m2 / (4.0 * std::f64::consts::PI)).sqrt() / (d1 * d2),
            reflection: Complex64::new(1.0, 0.0),
            order: 0,
        });
    }

    paths
}

/// Synthesizes the channel frequency response at each `freqs_hz` for the
/// room state at time `t`.
#[must_use]
pub fn synthesize_csi(
    room: &RoomSpec,
    tx: [f64; 3],
    rx: [f64; 3],
    freqs_hz: &[f64],
    t: f64,
) -> Vec<Complex64> {
    let carrier = freqs_hz[freqs_hz.len() / 2];
    let paths = enumerate_paths(room, tx, rx, carrier, t, 2);
    freqs_hz
        .iter()
        .map(|&f| {
            let mut h = Complex64::new(0.0, 0.0);
            for p in &paths {
                let amp = (C / f) / (4.0 * std::f64::consts::PI) * p.amp_scale;
                let phase = -2.0 * std::f64::consts::PI * f * p.distance_m / C;
                h += p.reflection * Complex64::from_polar(amp, phase);
            }
            h
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::room::{Material, PersonSpec};

    fn freqs() -> Vec<f64> {
        // 56 subcarriers over 20 MHz around 2.437 GHz.
        (0..56).map(|k| 2.437e9 - 10e6 + 20e6 * k as f64 / 55.0).collect()
    }

    #[test]
    fn direct_path_is_exact_friis() {
        // Absorber walls ⇒ only the direct path survives.
        let room = RoomSpec::new([8.0, 6.0, 3.0], Material::absorber(), vec![]).unwrap();
        let tx = [1.0, 1.0, 1.5];
        let rx = [5.0, 4.0, 1.5]; // d = 5
        let freqs = freqs();
        let h = synthesize_csi(&room, tx, rx, &freqs, 0.0);
        for (f, hv) in freqs.iter().zip(&h) {
            let expected = (C / f) / (4.0 * std::f64::consts::PI * 5.0);
            assert!(
                (hv.norm() - expected).abs() < 1e-15,
                "at {f} Hz: |H| = {} vs Friis {expected}",
                hv.norm()
            );
        }
    }

    #[test]
    fn channel_is_reciprocal() {
        let people = vec![PersonSpec { start: [3.0, 2.0, 1.2], velocity: [0.4, 0.1, 0.0], rcs_m2: 0.5 }];
        let room = RoomSpec::new([8.0, 6.0, 3.0], Material::concrete(), people).unwrap();
        let a = [1.0, 1.0, 1.5];
        let b = [6.5, 4.2, 1.1];
        let freqs = freqs();
        let fwd = synthesize_csi(&room, a, b, &freqs, 0.35);
        let rev = synthesize_csi(&room, b, a, &freqs, 0.35);
        for (x, y) in fwd.iter().zip(&rev) {
            assert!((x - y).norm() < 1e-12, "H(a→b) must equal H(b→a): {x} vs {y}");
        }
    }

    #[test]
    fn first_order_reflection_matches_mirror_geometry() {
        let room = RoomSpec::new([8.0, 6.0, 3.0], Material::concrete(), vec![]).unwrap();
        let tx = [2.0, 3.0, 1.5];
        let rx = [6.0, 3.0, 1.5];
        let paths = enumerate_paths(&room, tx, rx, 2.437e9, 0.0, 2);

        // Floor bounce (z = 0): mirror TX to z = −1.5; d = √(4² + 3²) = 5.
        let floor = ((tx[0] - rx[0]).powi(2)
            + (tx[1] - rx[1]).powi(2)
            + (-tx[2] - rx[2]).powi(2))
        .sqrt();
        assert!((floor - 5.0).abs() < 1e-12, "test geometry sanity");
        assert!(
            paths.iter().any(|p| p.order == 1 && (p.distance_m - 5.0).abs() < 1e-12),
            "floor-bounce image path at exactly 5 m must exist"
        );

        // Ceiling bounce (z = 3): mirror TX to z = 4.5; d = √(16 + 9) = 5.
        assert!(
            paths
                .iter()
                .any(|p| p.order == 1 && (p.distance_m - 5.0).abs() < 1e-12
                    && p.distance_m > 0.0),
            "ceiling-bounce path must exist"
        );

        // Path count sanity: direct + 6 first-order + second-order set, all
        // with |Γ| > 0 for concrete.
        assert!(paths.iter().filter(|p| p.order == 1).count() == 6, "6 first-order walls");
        assert!(paths.iter().any(|p| p.order == 2));
        assert_eq!(paths.iter().filter(|p| p.order == 0).count(), 1, "one direct path");
    }

    #[test]
    fn moving_person_produces_the_analytic_doppler_phase_rate() {
        // Person walking radially outward along the TX–RX bisector normal;
        // compare the residual (person-only) phase rotation between
        // snapshots against −2πf·Δd/c.
        let person = PersonSpec { start: [4.0, 2.0, 1.2], velocity: [0.0, 0.8, 0.0], rcs_m2: 0.6 };
        let with_person =
            RoomSpec::new([8.0, 6.0, 3.0], Material::drywall(), vec![person]).unwrap();
        let empty = RoomSpec::new([8.0, 6.0, 3.0], Material::drywall(), vec![]).unwrap();
        let tx = [1.0, 2.0, 1.5];
        let rx = [7.0, 2.0, 1.5];
        let f = [2.437e9];
        let dt = 0.05;

        for step in 0..4 {
            let t0 = step as f64 * dt;
            let t1 = t0 + dt;
            let resid0 = synthesize_csi(&with_person, tx, rx, &f, t0)[0]
                - synthesize_csi(&empty, tx, rx, &f, t0)[0];
            let resid1 = synthesize_csi(&with_person, tx, rx, &f, t1)[0]
                - synthesize_csi(&empty, tx, rx, &f, t1)[0];
            let measured_dphi = (resid1 * resid0.conj()).arg();

            let path_len = |t: f64| {
                let p = person.position_at(t);
                let d1 = ((p[0] - tx[0]).powi(2) + (p[1] - tx[1]).powi(2) + (p[2] - tx[2]).powi(2))
                    .sqrt();
                let d2 = ((p[0] - rx[0]).powi(2) + (p[1] - rx[1]).powi(2) + (p[2] - rx[2]).powi(2))
                    .sqrt();
                d1 + d2
            };
            let expected_dphi = -2.0 * std::f64::consts::PI * f[0] * (path_len(t1) - path_len(t0)) / C;
            // Compare on the unit circle (phases are mod 2π).
            let diff = (Complex64::from_polar(1.0, measured_dphi)
                * Complex64::from_polar(1.0, -expected_dphi))
            .arg();
            assert!(
                diff.abs() < 1e-6,
                "step {step}: measured Δφ {measured_dphi} vs analytic {expected_dphi}"
            );
        }
    }
}
