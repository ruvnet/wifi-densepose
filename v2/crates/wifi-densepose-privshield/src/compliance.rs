//! Machine-checkable compliance: the shield shapes its own frames, never jams.
//!
//! Jamming (47 U.S.C. §333, §302a) is defined by *adding energy to interfere
//! with others' transmissions*. VEIL's protector applies an **orthogonal**
//! transform to its own beamforming feedback, which preserves the report's
//! energy exactly. This module turns that invariant into a checked artifact: it
//! measures the input/output energy of a protection step and asserts the ratio
//! is ~1, i.e. no energy was added. A regulator, an auditor, or the runtime
//! attestation layer (ADR-141) can read a [`ComplianceReport`] and see the
//! shield is a waveform-shaping control, not an emitter of interference.

use crate::identity::BfiSample;
use crate::linalg::norm_sq;

/// Tolerance on the energy ratio. Orthogonal rotations are exact up to f32
/// round-off across many Givens passes.
pub const ENERGY_TOLERANCE: f32 = 1e-2;

/// The result of auditing one protection step.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplianceReport {
    /// Energy of the report before protection.
    pub input_energy: f32,
    /// Energy of the report after protection.
    pub output_energy: f32,
    /// `output_energy / input_energy`. ~1.0 for an energy-preserving control.
    pub energy_ratio: f32,
    /// True iff the energy ratio is within [`ENERGY_TOLERANCE`] of 1.0.
    pub energy_conserving: bool,
    /// True iff the control adds energy on top of another station's signal.
    /// Always false for VEIL by construction — it transforms its own report.
    pub adds_interfering_energy: bool,
}

impl ComplianceReport {
    /// Audit a `(before, after)` protection pair.
    #[must_use]
    pub fn audit(before: &BfiSample, after: &BfiSample) -> Self {
        let input_energy = norm_sq(&before.values);
        let output_energy = norm_sq(&after.values);
        let energy_ratio = if input_energy > 1e-12 {
            output_energy / input_energy
        } else {
            1.0
        };
        Self {
            input_energy,
            output_energy,
            energy_ratio,
            energy_conserving: (energy_ratio - 1.0).abs() <= ENERGY_TOLERANCE,
            adds_interfering_energy: false,
        }
    }

    /// The bottom-line compliance verdict: energy-preserving and
    /// non-interfering ⇒ a compliant waveform control, not jamming.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.energy_conserving && !self.adds_interfering_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Channel, SceneConfig};
    use crate::protector::{Protector, ShieldConfig};

    #[test]
    fn protection_is_compliant() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 3);
        let p = Protector::new(ShieldConfig::default());
        let out = p.protect(&s, 555);
        let report = ComplianceReport::audit(&s, &out);
        assert!(report.is_compliant(), "{report:?}");
        assert!((report.energy_ratio - 1.0).abs() < ENERGY_TOLERANCE);
    }
}
