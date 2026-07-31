//! Room, material, and person specifications for the synthetic world
//! (ADR-276 §2).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{Result, UnifiedError};

/// Vacuum permittivity (F/m).
const EPS0: f64 = 8.854_187_812_8e-12;

/// Wall material with complex permittivity — the quantity domain
/// randomization varies (randomize physics, not textures).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Relative permittivity ε_r (> 1 for solids).
    pub rel_permittivity: f64,
    /// Conductivity σ in S/m (loss term).
    pub conductivity_s_m: f64,
}

impl Material {
    /// Typical poured concrete (ITU-R P.2040 ballpark).
    #[must_use]
    pub fn concrete() -> Self {
        Self { rel_permittivity: 5.3, conductivity_s_m: 0.073 }
    }

    /// Gypsum drywall.
    #[must_use]
    pub fn drywall() -> Self {
        Self { rel_permittivity: 2.9, conductivity_s_m: 0.016 }
    }

    /// Window glass.
    #[must_use]
    pub fn glass() -> Self {
        Self { rel_permittivity: 6.3, conductivity_s_m: 0.004 }
    }

    /// Perfect absorber (anechoic) — kills all reflections; used by tests to
    /// isolate the direct path.
    #[must_use]
    pub fn absorber() -> Self {
        Self { rel_permittivity: 1.0, conductivity_s_m: 0.0 }
    }

    /// Complex relative permittivity at frequency `f`:
    /// `ε = ε_r − j·σ/(ω·ε₀)`.
    #[must_use]
    pub fn complex_permittivity(&self, freq_hz: f64) -> Complex64 {
        let omega = 2.0 * std::f64::consts::PI * freq_hz;
        Complex64::new(self.rel_permittivity, -self.conductivity_s_m / (omega * EPS0))
    }

    /// Normal-incidence Fresnel amplitude reflection coefficient
    /// `Γ = (1 − √ε)/(1 + √ε)` against air. (Normal incidence is the
    /// standard image-method simplification; angle dependence is a
    /// randomizable refinement, not a correctness requirement.)
    #[must_use]
    pub fn reflection_coefficient(&self, freq_hz: f64) -> Complex64 {
        let sqrt_eps = self.complex_permittivity(freq_hz).sqrt();
        (Complex64::new(1.0, 0.0) - sqrt_eps) / (Complex64::new(1.0, 0.0) + sqrt_eps)
    }
}

/// A person modeled as a moving bistatic point scatterer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PersonSpec {
    /// Position at t = 0, metres.
    pub start: [f64; 3],
    /// Constant velocity, m/s.
    pub velocity: [f64; 3],
    /// Radar cross-section, m² (torso ≈ 0.3–1.0 at WiFi bands).
    pub rcs_m2: f64,
}

impl PersonSpec {
    /// Position at time `t` seconds.
    #[must_use]
    pub fn position_at(&self, t: f64) -> [f64; 3] {
        [
            self.start[0] + self.velocity[0] * t,
            self.start[1] + self.velocity[1] * t,
            self.start[2] + self.velocity[2] * t,
        ]
    }
}

/// A shoebox room: `[0, Lx] × [0, Ly] × [0, Lz]` with one wall material and
/// zero or more people.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoomSpec {
    /// Interior dimensions `[Lx, Ly, Lz]`, metres.
    pub size: [f64; 3],
    /// Wall/floor/ceiling material.
    pub wall_material: Material,
    /// Occupants.
    pub people: Vec<PersonSpec>,
}

impl RoomSpec {
    /// Validated constructor: positive dimensions, finite fields, people
    /// starting inside the room.
    pub fn new(size: [f64; 3], wall_material: Material, people: Vec<PersonSpec>) -> Result<Self> {
        if size.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(UnifiedError::InvalidInput(format!(
                "room dimensions must be finite and positive, got {size:?}"
            )));
        }
        for p in &people {
            for (axis, v) in p.start.iter().enumerate() {
                if !v.is_finite() || *v < 0.0 || *v > size[axis] {
                    return Err(UnifiedError::InvalidInput(format!(
                        "person start {:?} outside room {size:?}",
                        p.start
                    )));
                }
            }
            if p.rcs_m2 <= 0.0 || !p.rcs_m2.is_finite() {
                return Err(UnifiedError::InvalidInput(format!(
                    "rcs_m2 must be positive, got {}",
                    p.rcs_m2
                )));
            }
        }
        Ok(Self { size, wall_material, people })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_coefficient_sane_for_concrete() {
        let g = Material::concrete().reflection_coefficient(2.437e9);
        // Concrete at 2.4 GHz: |Γ| ≈ 0.39–0.45, negative real part
        // (denser medium ⇒ phase inversion).
        assert!(g.norm() > 0.3 && g.norm() < 0.5, "|Γ| = {}", g.norm());
        assert!(g.re < 0.0);
        // Physical bound: |Γ| < 1 for any passive material.
        for m in [Material::drywall(), Material::glass(), Material::concrete()] {
            assert!(m.reflection_coefficient(5.18e9).norm() < 1.0);
        }
    }

    #[test]
    fn absorber_reflects_nothing() {
        let g = Material::absorber().reflection_coefficient(2.437e9);
        assert!(g.norm() < 1e-12, "ε_r = 1, σ = 0 must give Γ = 0, got {g}");
    }

    #[test]
    fn room_validation_rejects_bad_specs() {
        assert!(RoomSpec::new([4.0, -3.0, 2.5], Material::drywall(), vec![]).is_err());
        let outside = PersonSpec { start: [9.0, 1.0, 1.0], velocity: [0.0; 3], rcs_m2: 0.5 };
        assert!(RoomSpec::new([4.0, 3.0, 2.5], Material::drywall(), vec![outside]).is_err());
        let ok = PersonSpec { start: [2.0, 1.0, 1.0], velocity: [0.5, 0.0, 0.0], rcs_m2: 0.5 };
        let room = RoomSpec::new([4.0, 3.0, 2.5], Material::drywall(), vec![ok]).expect("valid");
        assert_eq!(room.people.len(), 1);
        assert_eq!(room.people[0].position_at(2.0), [3.0, 1.0, 1.0]);
    }
}
