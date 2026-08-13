//! Channel-gain queries against the Gaussian map, and the inverse update
//! that makes the map *learn* from measured links (ADR-275 §4).
//!
//! Physics model: free-space Friis amplitude with Beer–Lambert extinction
//! through the Gaussians,
//!
//! ```text
//! H(tx,rx,f) = (λ / 4πd) · e^{-j·2πd/λ} · exp(-Σ_g occ_g · I_g)
//! ```
//!
//! where `I_g = ∫₀ᴸ exp(-½·q_g(o + t·u)) dt` is the closed-form line
//! integral of Gaussian `g`'s unnormalized density along the TX→RX segment
//! (a 1-D Gaussian integral, evaluated with `erf`). Three anchors make this
//! testable, each scoped to what it actually exercises: an **empty map
//! returns exact Friis** (baseline only — with no Gaussians the
//! Beer–Lambert path never runs), the **closed-form line integral matches
//! numeric quadrature**, and the **occluded gain through [`channel_gain`]
//! matches Friis × Beer–Lambert with quadrature integrals** (the erf path
//! validated end-to-end through the public API).

use num_complex::Complex64;

use super::map::GaussianMap;
use super::primitive::{Provenance, RfGaussian};
use crate::math::erf;

const C: f64 = 299_792_458.0;

/// Closed-form line integral of `exp(-½·(x-μ)ᵀΣ⁻¹(x-μ))` along the segment
/// `o → o + L·u` (`u` unit).
///
/// With `a = uᵀΣ⁻¹u`, `b = uᵀΣ⁻¹(μ−o)`, `c = (μ−o)ᵀΣ⁻¹(μ−o)`:
/// `q(t) = a·(t − b/a)² + (c − b²/a)`, so
/// `I = e^{-(c−b²/a)/2} · √(π/2a) · [erf(√(a/2)(L−t₀)) + erf(√(a/2)·t₀)]`.
#[must_use]
pub fn line_integral(g: &RfGaussian, o: [f64; 3], u: [f64; 3], len: f64) -> f64 {
    let si = g.inv_covariance();
    let mo = [g.position[0] - o[0], g.position[1] - o[1], g.position[2] - o[2]];
    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            a += u[i] * si[i][j] * u[j];
            b += u[i] * si[i][j] * mo[j];
            c += mo[i] * si[i][j] * mo[j];
        }
    }
    if a <= 0.0 {
        return 0.0;
    }
    let t0 = b / a;
    let peak = (-0.5 * (c - b * b / a)).exp();
    let s = (a / 2.0).sqrt();
    peak * (std::f64::consts::PI / (2.0 * a)).sqrt() * (erf(s * (len - t0)) + erf(s * t0))
}

/// Total optical depth (nepers) of the map along a TX→RX segment, plus the
/// per-Gaussian integrals for gradient use: `(τ, [(idx, I_g)])`.
#[must_use]
pub fn optical_depth(map: &GaussianMap, tx: [f64; 3], rx: [f64; 3]) -> (f64, Vec<(usize, f64)>) {
    let d = [rx[0] - tx[0], rx[1] - tx[1], rx[2] - tx[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len < 1e-9 {
        return (0.0, Vec::new());
    }
    let u = [d[0] / len, d[1] / len, d[2] / len];
    // Candidate set: Gaussians within a 3 m corridor of the segment (3σ for
    // σ ≤ 1 m primitives), gathered by walking only the hash cells along
    // the link instead of a ball around its midpoint — see
    // `GaussianMap::query_near_segment` for the measured effect.
    let candidates = map.query_near_segment(tx, rx, 3.0);
    let mut tau = 0.0;
    let mut parts = Vec::new();
    for i in candidates {
        let g = &map.gaussians()[i];
        let integral = line_integral(g, tx, u, len);
        if integral > 1e-12 {
            tau += g.occupancy * integral;
            parts.push((i, integral));
        }
    }
    (tau, parts)
}

/// Complex channel gain between two points at carrier `freq_hz`.
///
/// Empty map ⇒ exact free-space Friis amplitude `λ/(4πd)` with propagation
/// phase `e^{-j2πd/λ}`.
#[must_use]
pub fn channel_gain(map: &GaussianMap, tx: [f64; 3], rx: [f64; 3], freq_hz: f64) -> Complex64 {
    let d = ((rx[0] - tx[0]).powi(2) + (rx[1] - tx[1]).powi(2) + (rx[2] - tx[2]).powi(2)).sqrt();
    if d < 1e-9 {
        return Complex64::new(1.0, 0.0);
    }
    let lambda = C / freq_hz;
    let amp = lambda / (4.0 * std::f64::consts::PI * d);
    let phase = -2.0 * std::f64::consts::PI * d / lambda;
    let (tau, _) = optical_depth(map, tx, rx);
    Complex64::from_polar(amp * (-tau).exp(), phase)
}

/// Power gain in dB.
#[must_use]
pub fn gain_db(map: &GaussianMap, tx: [f64; 3], rx: [f64; 3], freq_hz: f64) -> f64 {
    20.0 * channel_gain(map, tx, rx, freq_hz).norm().log10()
}

/// One inverse-update step from a measured link amplitude (ADR-275 §4.2 —
/// the physics-informed incremental mapping move: when the environment
/// changes, adjust or add a *compact* set of Gaussians instead of remapping).
///
/// The measured optical depth is `τ* = ln(friis_amp / measured_amp)`; the
/// map's depth is `τ = Σ occ_g·I_g`. A projected-gradient step moves the
/// intersected occupancies toward the residual `r = τ* − τ`:
///
/// `occ_g += lr · r · I_g / (Σ I_g² + ε)`, clamped at 0 —
///
/// which is exact Newton along this link's direction at `lr = 1`. When no
/// Gaussian meaningfully intersects the segment and the residual demands
/// attenuation, a new isotropic Gaussian is spawned at the segment midpoint
/// sized to explain the residual. Returns the post-update residual (nepers).
pub fn observe_link(
    map: &mut GaussianMap,
    tx: [f64; 3],
    rx: [f64; 3],
    freq_hz: f64,
    measured_amp: f64,
    lr: f64,
    timestamp_ns: u64,
) -> f64 {
    assert!(measured_amp > 0.0, "measured amplitude must be positive");
    let d = ((rx[0] - tx[0]).powi(2) + (rx[1] - tx[1]).powi(2) + (rx[2] - tx[2]).powi(2)).sqrt();
    let lambda = C / freq_hz;
    let friis = lambda / (4.0 * std::f64::consts::PI * d);
    let tau_target = (friis / measured_amp).ln();

    let (tau, parts) = optical_depth(map, tx, rx);
    let residual = tau_target - tau;

    let sum_i2: f64 = parts.iter().map(|(_, i)| i * i).sum();
    if sum_i2 > 1e-9 {
        for (idx, integral) in &parts {
            let g = &mut map.gaussians_mut()[*idx];
            g.occupancy = (g.occupancy + lr * residual * integral / sum_i2).max(0.0);
            g.timestamp_ns = g.timestamp_ns.max(timestamp_ns);
        }
        map.rebuild_grid();
    } else if residual > 0.05 {
        // Nothing on the path can explain the attenuation: spawn a compact
        // absorber at the midpoint sized to close the residual exactly.
        let mid = [(tx[0] + rx[0]) / 2.0, (tx[1] + rx[1]) / 2.0, (tx[2] + rx[2]) / 2.0];
        let mut g = RfGaussian::new(
            mid,
            [0.3, 0.3, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            0.0,
            0.6,
            timestamp_ns,
            300.0,
            Provenance { device_id: "gain-inverse".into(), model_version: 1, synthetic: false },
        )
        .expect("spawn parameters are statically valid");
        let dvec = [rx[0] - tx[0], rx[1] - tx[1], rx[2] - tx[2]];
        let u = [dvec[0] / d, dvec[1] / d, dvec[2] / d];
        let unit_integral = line_integral(&g, tx, u, d);
        if unit_integral > 1e-9 {
            g.occupancy = residual / unit_integral;
            map.insert(g);
        }
    }

    let (tau_after, _) = optical_depth(map, tx, rx);
    tau_target - tau_after
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::primitive::Provenance;

    fn prov() -> Provenance {
        Provenance { device_id: "gain-test".into(), model_version: 1, synthetic: true }
    }

    /// Scope: free-space baseline ONLY. With no Gaussians, `optical_depth`
    /// short-circuits to `τ = 0` before `line_integral`/`erf` ever runs, so
    /// this pins the Friis amplitude and propagation phase and nothing else.
    /// The Beer–Lambert/erf path is validated end-to-end by
    /// `occluded_channel_gain_matches_beer_lambert_quadrature` below.
    #[test]
    fn empty_map_returns_exact_friis() {
        let map = GaussianMap::new(1.0);
        let f = 2.437e9;
        let tx = [0.0, 0.0, 1.0];
        let rx = [4.0, 3.0, 1.0]; // d = 5
        let h = channel_gain(&map, tx, rx, f);
        let lambda = C / f;
        let expected_amp = lambda / (4.0 * std::f64::consts::PI * 5.0);
        assert!((h.norm() - expected_amp).abs() < 1e-15, "amplitude must be exact Friis");
        let expected_phase = -2.0 * std::f64::consts::PI * 5.0 / lambda;
        // Compare phasors (phase is only defined mod 2π).
        let diff = (h / Complex64::from_polar(expected_amp, expected_phase)) - 1.0;
        assert!(diff.norm() < 1e-9, "propagation phase must match");
    }

    /// The erf/Beer–Lambert claim, validated through the public API: with
    /// matter on the path, `|H|` from [`channel_gain`] must equal
    /// `Friis · exp(-Σ_g occ_g·I_g)` where every `I_g` is recomputed here by
    /// trapezoid quadrature — never by the closed form under test. Uses two
    /// anisotropic, rotated Gaussians at different perpendicular offsets so
    /// the exercised integrals are not degenerate symmetric cases.
    #[test]
    fn occluded_channel_gain_matches_beer_lambert_quadrature() {
        let f = 5.18e9;
        let tx = [0.0, 0.0, 1.0];
        let rx = [6.0, 0.0, 1.0];
        let gaussians = [
            // On the line of sight, anisotropic and rotated.
            RfGaussian::new(
                [2.5, 0.0, 1.0],
                [0.5, 0.8, 0.3],
                [0.9, 0.1, 0.2, 0.1],
                1.3,
                0.9,
                0,
                300.0,
                prov(),
            )
            .expect("valid"),
            // Half a metre off the line of sight: partial occlusion.
            RfGaussian::new(
                [4.0, 0.5, 1.2],
                [0.4, 0.6, 0.5],
                [0.8, -0.3, 0.1, 0.2],
                0.7,
                0.9,
                0,
                300.0,
                prov(),
            )
            .expect("valid"),
        ];

        // Ground-truth optical depth by quadrature over each density,
        // computed before the map ever sees the Gaussians.
        let len = 6.0;
        let u = [1.0, 0.0, 0.0];
        let n = 6000;
        let mut tau_quad = 0.0;
        for g in &gaussians {
            let mut acc = 0.0;
            for i in 0..=n {
                let t = len * i as f64 / n as f64;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                acc += w * g.density_at([tx[0] + t * u[0], tx[1] + t * u[1], tx[2] + t * u[2]]);
            }
            tau_quad += g.occupancy * acc * len / n as f64;
        }

        let mut map = GaussianMap::new(1.0);
        for g in gaussians {
            map.insert(g);
        }
        assert!(tau_quad > 0.3, "test must exercise real attenuation, τ = {tau_quad}");

        let lambda = C / f;
        let friis = lambda / (4.0 * std::f64::consts::PI * len);
        let expected = friis * (-tau_quad).exp();
        let h = channel_gain(&map, tx, rx, f);
        // Tolerance is set by the 1 mm trapezoid truncation error of the
        // reference integrals (same budget as
        // `line_integral_matches_numeric_quadrature`), not by the closed
        // form: agreement to 1 ppm through the public API.
        assert!(
            (h.norm() - expected).abs() < 1e-6 * expected,
            "|H| = {} must match Friis·exp(-τ_quad) = {expected}",
            h.norm()
        );
    }

    #[test]
    fn absorber_on_path_reduces_gain_monotonically() {
        let f = 5.18e9;
        let tx = [0.0, 0.0, 1.0];
        let rx = [6.0, 0.0, 1.0];
        let mut last = f64::INFINITY;
        for occ in [0.0, 0.2, 0.5, 1.0, 2.0] {
            let mut map = GaussianMap::new(1.0);
            let g = RfGaussian::new(
                [3.0, 0.0, 1.0],
                [0.4, 0.4, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                occ,
                0.9,
                0,
                300.0,
                prov(),
            )
            .expect("valid");
            map.insert(g);
            let db = gain_db(&map, tx, rx, f);
            assert!(db < last + 1e-12, "gain must fall as occupancy rises");
            last = db;
        }
    }

    #[test]
    fn off_path_absorber_barely_matters() {
        let f = 5.18e9;
        let tx = [0.0, 0.0, 1.0];
        let rx = [6.0, 0.0, 1.0];
        let mut map = GaussianMap::new(1.0);
        map.insert(
            RfGaussian::new(
                [3.0, 4.0, 1.0], // 4 m off the LoS, σ = 0.4 ⇒ 10σ away
                [0.4, 0.4, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                2.0,
                0.9,
                0,
                300.0,
                prov(),
            )
            .expect("valid"),
        );
        let clean = gain_db(&GaussianMap::new(1.0), tx, rx, f);
        let with = gain_db(&map, tx, rx, f);
        assert!((clean - with).abs() < 1e-6, "off-path matter must not attenuate LoS");
    }

    #[test]
    fn line_integral_matches_numeric_quadrature() {
        let g = RfGaussian::new(
            [2.0, 0.5, 1.0],
            [0.5, 0.8, 0.3],
            [0.9, 0.1, 0.2, 0.1], // arbitrary rotation
            1.0,
            0.9,
            0,
            300.0,
            prov(),
        )
        .expect("valid");
        let o = [0.0, 0.0, 1.0];
        let u = [1.0, 0.0, 0.0];
        let len = 6.0;
        // Trapezoid quadrature at 1 mm steps as ground truth.
        let n = 6000;
        let mut acc = 0.0;
        for i in 0..=n {
            let t = len * i as f64 / n as f64;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            acc += w * g.density_at([o[0] + t * u[0], o[1] + t * u[1], o[2] + t * u[2]]);
        }
        acc *= len / n as f64;
        let closed = line_integral(&g, o, u, len);
        assert!(
            (closed - acc).abs() < 1e-6,
            "closed form {closed} vs quadrature {acc}"
        );
    }

    /// Scope: single-scalar convergence ONLY. The 20 iterations are 20
    /// repeats of the *same* measurement on the *same* link, and
    /// `τ = occ·I` is exactly linear in the one spawned Gaussian's
    /// occupancy, so this verifies that the damped-Newton `observe_link`
    /// update converges one scalar unknown geometrically — nothing about
    /// learning from diverse observations. That stronger scenario is
    /// covered by `interleaved_links_resolve_distinct_absorbers` below.
    #[test]
    fn inverse_update_converges_one_scalar_from_a_repeated_link() {
        let f = 2.437e9;
        let tx = [0.0, 0.0, 1.0];
        let rx = [6.0, 0.0, 1.0];
        let lambda = C / f;
        let friis = lambda / (4.0 * std::f64::consts::PI * 6.0);
        // Ground truth: an unseen absorber imposes τ = 0.7 nepers (≈ 6.1 dB).
        let measured = friis * (-0.7f64).exp();

        let mut map = GaussianMap::new(1.0);
        let mut residual = f64::INFINITY;
        for step in 0..20 {
            residual = observe_link(&mut map, tx, rx, f, measured, 0.7, step);
        }
        assert!(!map.is_empty(), "an absorber must have been spawned");
        assert!(
            residual.abs() < 0.06,
            "residual after convergence must be < 0.06 nepers (≈0.5 dB), got {residual}"
        );
        let predicted_db = gain_db(&map, tx, rx, f);
        let measured_db = 20.0 * measured.log10();
        assert!(
            (predicted_db - measured_db).abs() < 0.5,
            "map prediction {predicted_db:.2} dB must match measurement {measured_db:.2} dB"
        );
    }

    /// Learning from genuinely diverse observations: two hidden absorbers
    /// with different optical depths, observed through three interleaved
    /// links — one crossing only the first absorber, one crossing only the
    /// second, and a coupled link crossing *both*. The coupled link is
    /// observed FIRST, so its greedy midpoint spawn creates a structurally
    /// wrong third absorber between the true two; the interleaved updates
    /// must then jointly drain that blob to zero occupancy while the two
    /// real unknowns converge to the hidden truth. This is the multi-
    /// measurement, multi-unknown scenario the single-link test above
    /// deliberately does not claim.
    #[test]
    fn interleaved_links_resolve_distinct_absorbers() {
        let f = 2.437e9;
        // Truth absorbers sit exactly at the midpoints of links A and B with
        // the same σ = 0.3 the spawner uses, so the truth is representable
        // and the joint system is consistent — what is under test is whether
        // interleaved coupled updates actually find it from a wrong start.
        let g1_pos = [3.0, 0.0, 1.0];
        let g2_pos = [3.0, 3.0, 1.0];
        let (occ1, occ2) = (1.2, 0.45); // genuinely different unknowns
        let mut truth = GaussianMap::new(1.0);
        for (pos, occ) in [(g1_pos, occ1), (g2_pos, occ2)] {
            truth.insert(
                RfGaussian::new(pos, [0.3, 0.3, 0.3], [1.0, 0.0, 0.0, 0.0], occ, 0.9, 0, 300.0, prov())
                    .expect("valid"),
            );
        }

        // Link C runs through both centres (the x = 3 vertical), coupling
        // the two unknowns; link A crosses only G1, link B only G2.
        let link_c = ([3.0, -1.5, 1.0], [3.0, 4.5, 1.0]);
        let link_a = ([0.0, 0.0, 1.0], [6.0, 0.0, 1.0]);
        let link_b = ([0.0, 3.0, 1.0], [6.0, 3.0, 1.0]);
        let links = [link_c, link_a, link_b];
        let measured: Vec<f64> =
            links.iter().map(|(tx, rx)| channel_gain(&truth, *tx, *rx, f).norm()).collect();

        // The two single-absorber links must present distinct attenuations,
        // otherwise "different unknowns" is not actually being tested.
        let db = |amp: f64| 20.0 * amp.log10();
        let clean_db = |tx: [f64; 3], rx: [f64; 3]| gain_db(&GaussianMap::new(1.0), tx, rx, f);
        let atten_a = clean_db(link_a.0, link_a.1) - db(measured[1]);
        let atten_b = clean_db(link_b.0, link_b.1) - db(measured[2]);
        assert!(
            atten_a > 2.0 * atten_b,
            "truth attenuations must differ: A {atten_a:.2} dB vs B {atten_b:.2} dB"
        );

        let mut learner = GaussianMap::new(1.0);
        let mut residuals = [f64::INFINITY; 3];
        for round in 0..40u64 {
            for (i, (tx, rx)) in links.iter().enumerate() {
                residuals[i] = observe_link(
                    &mut learner,
                    *tx,
                    *rx,
                    f,
                    measured[i],
                    0.7,
                    round * links.len() as u64 + i as u64,
                );
            }
            if round == 0 {
                // The wrong structure must actually have been created —
                // otherwise this test degenerates to three independent
                // single-scalar problems.
                assert!(
                    learner
                        .gaussians()
                        .iter()
                        .any(|g| (g.position[1] - 1.5).abs() < 1e-9 && g.occupancy > 0.5),
                    "coupled link must first spawn a (wrong) midpoint absorber"
                );
            }
        }

        // Every link's measurement must be explained simultaneously.
        for (i, r) in residuals.iter().enumerate() {
            assert!(
                r.abs() < 0.06,
                "link {i} residual must converge below 0.06 nepers, got {r}"
            );
        }
        for (i, (tx, rx)) in links.iter().enumerate() {
            let predicted_db = gain_db(&learner, *tx, *rx, f);
            let measured_db = db(measured[i]);
            assert!(
                (predicted_db - measured_db).abs() < 0.5,
                "link {i}: prediction {predicted_db:.2} dB vs measurement {measured_db:.2} dB"
            );
        }

        // Structure, not just per-link fit: the wrong blob at C's midpoint
        // must have been drained to (near) zero, and the two real absorbers
        // must carry occupancies close to the hidden truth — distinct
        // unknowns genuinely resolved, not one smeared explanation.
        assert_eq!(learner.gaussians().len(), 3, "C-blob + two real absorbers expected");
        let occupancy_near = |pos: [f64; 3]| {
            learner
                .gaussians()
                .iter()
                .find(|g| {
                    (0..3).map(|k| (g.position[k] - pos[k]).powi(2)).sum::<f64>().sqrt() < 1e-9
                })
                .map(|g| g.occupancy)
        };
        let blob = occupancy_near([3.0, 1.5, 1.0]).expect("blob spawned at C midpoint");
        assert!(blob < 0.05, "wrong midpoint blob must drain to ~0, still at {blob}");
        for (pos, occ) in [(g1_pos, occ1), (g2_pos, occ2)] {
            let recovered = occupancy_near(pos).expect("absorber spawned at truth position");
            assert!(
                (recovered - occ).abs() < 0.15 * occ,
                "recovered occupancy {recovered} must be within 15 % of truth {occ}"
            );
        }

        // Locality: a link far from both absorbers stays at clean Friis.
        let (tx_d, rx_d) = ([0.0, -3.0, 1.0], [6.0, -3.0, 1.0]);
        let clean = clean_db(tx_d, rx_d);
        let with_learned = gain_db(&learner, tx_d, rx_d, f);
        assert!(
            (clean - with_learned).abs() < 1e-6,
            "learning must not smear absorption onto untouched links: {clean} vs {with_learned}"
        );
    }
}
