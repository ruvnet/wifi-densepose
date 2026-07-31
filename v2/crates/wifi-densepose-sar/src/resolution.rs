//! Closed-form resolution and coherence-budget formulas (ADR-287 §3).
//!
//! These are textbook radar-imaging identities (see e.g. Skolnik, *Radar
//! Handbook*, and the standard stripmap-SAR cross-range formula). They are
//! implemented here so the crate's own reconstruction behavior can be
//! checked against them in [`tests/physics_validation.rs`] rather than
//! merely asserted in documentation.

/// Speed of light in vacuum, m/s.
pub const SPEED_OF_LIGHT_M_PER_S: f64 = 299_792_458.0;

/// Wavelength (meters) of a signal at `freq_hz`.
pub fn wavelength_m(freq_hz: f64) -> f64 {
    SPEED_OF_LIGHT_M_PER_S / freq_hz
}

/// Range resolution (meters) of a stepped-frequency / wideband radar with
/// total swept bandwidth `bandwidth_hz`: `ΔR = c / (2B)`.
///
/// This is the Rayleigh-style minimum range separation at which two
/// point targets on the same bearing become distinguishable after pulse
/// compression / coherent range processing. It does **not** depend on
/// carrier frequency, antenna count, or synthetic-aperture length --
/// only on how much spectrum was actually swept.
pub fn range_resolution_m(bandwidth_hz: f64) -> f64 {
    SPEED_OF_LIGHT_M_PER_S / (2.0 * bandwidth_hz)
}

/// Cross-range (azimuth) resolution (meters) of a synthetic aperture of
/// physical length `aperture_length_m`, imaging a target at `range_m`,
/// at carrier frequency `center_freq_hz`: `δ_CR ≈ λ·R / (2·L)`.
///
/// This is the classic stripmap-SAR angular-resolution identity: doubling
/// the aperture (or halving the wavelength) halves the achievable
/// cross-range spot size at a fixed range. It is undefined (returns
/// `f64::INFINITY`) for a degenerate (zero-length) aperture -- a single
/// antenna position carries no cross-range information at all, which is
/// exactly the point of building a synthetic aperture in the first place.
pub fn cross_range_resolution_m(center_freq_hz: f64, aperture_length_m: f64, range_m: f64) -> f64 {
    if aperture_length_m <= 0.0 {
        return f64::INFINITY;
    }
    wavelength_m(center_freq_hz) * range_m / (2.0 * aperture_length_m)
}

/// Maximum antenna-position error (meters) that keeps a coherent
/// (phase-focused) reconstruction inside the classical quarter-wave
/// budget, at carrier frequency `center_freq_hz`.
///
/// Derivation: moving an antenna's phase center by `Δp` while looking
/// (worst case) directly along boresight at the target changes the
/// round-trip path length by up to `2·Δp` (both the outbound and return
/// leg shift by `Δp`). Keeping that two-way path error under the
/// standard quarter-wavelength coherence budget (`λ/4` -- the same
/// criterion used for reflector-antenna and optical-surface tolerancing)
/// requires `2·Δp ≤ λ/4`, i.e. `Δp ≤ λ/8`.
///
/// Position error beyond this does not make reconstruction impossible --
/// it degrades the coherent sum smoothly (see
/// `phase_error_degrades_focus_beyond_pose_budget` in
/// `tests/physics_validation.rs`) -- but it is the standard rule-of-thumb
/// budget for "still well focused."
pub fn max_coherent_pose_error_m(center_freq_hz: f64) -> f64 {
    wavelength_m(center_freq_hz) / 8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_resolution_matches_hand_computed_value() {
        // 3 GHz of swept bandwidth: c/(2*3e9) = 4.9965...cm.
        let r = range_resolution_m(3.0e9);
        assert!((r - 0.049_965_409_666_666_66).abs() < 1e-9);
    }

    #[test]
    fn eight_ghz_pose_budget_is_about_5mm() {
        // At 8 GHz, lambda = c/f ~= 37.47mm, so lambda/8 ~= 4.68mm -- close
        // to the ~5mm rule-of-thumb quoted in the motivating design note.
        let budget = max_coherent_pose_error_m(8.0e9);
        assert!((budget - wavelength_m(8.0e9) / 8.0).abs() < 1e-12);
        assert!(budget < 0.005 && budget > 0.004);
    }

    #[test]
    fn cross_range_resolution_improves_with_longer_aperture() {
        let short = cross_range_resolution_m(5.0e9, 0.1, 2.0);
        let long = cross_range_resolution_m(5.0e9, 1.0, 2.0);
        assert!(long < short, "10x longer aperture must give finer cross-range resolution");
        // Exactly linear in 1/L.
        assert!((short / long - 10.0).abs() < 1e-9);
    }

    #[test]
    fn zero_length_aperture_has_no_cross_range_resolution() {
        assert_eq!(cross_range_resolution_m(5.0e9, 0.0, 2.0), f64::INFINITY);
    }

    #[test]
    fn wider_bandwidth_gives_finer_range_resolution() {
        assert!(range_resolution_m(4.0e9) < range_resolution_m(1.0e9));
    }
}
