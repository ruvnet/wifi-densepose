//! WiFi Veil advisory integration (ADR-291).
//!
//! Bridges BFLD's privacy layer to the [`wifi-veil`](https://github.com/ruvnet/wifi-veil)
//! countermeasure crate: a deterministic, dependency-free attacker-vs-protector
//! model of BFI identity leakage and keyed emission-shaping ("shield")
//! configurations.
//!
//! # Evidence discipline
//!
//! Everything this module produces is **`SYNTHETIC` / evidence level L0** by
//! construction: `wifi-veil` models compliant waveform controls on synthetic
//! scenes and never touches a radio. Assessments quantify the *modeled*
//! re-identification risk of unprotected beamforming feedback and the *modeled*
//! effect of a shield; they are advisory inputs to privacy posture, never
//! measured hardware claims. See ADR-291 and the wifi-veil README.
//!
//! # Relationship to BFLD invariants
//!
//! BFLD's structural invariants (I1–I3, see the crate README) govern data that
//! *enters* this node. WiFi Veil addresses the complementary surface: what this
//! node's own *outgoing* feedback leaks to passive third parties. The
//! integration is advisory-only — nothing here emits RF, alters frames, or
//! relaxes a BFLD gate.

use wifi_veil::{experiment, ExperimentConfig};

/// Evidence label attached to every veil-derived figure.
///
/// Matches the repository-wide claim taxonomy (CLAUDE.md): synthetic model
/// output, reproduced by `cargo test`, not measured on hardware.
pub const VEIL_EVIDENCE: &str = "SYNTHETIC/L0";

/// Summary of one deterministic attacker-vs-protector experiment.
///
/// A thin, stable projection of [`wifi_veil::ExperimentReport`] carrying only
/// the figures BFLD consumers need, plus the mandatory evidence label.
#[derive(Debug, Clone, PartialEq)]
pub struct ShieldAssessment {
    /// Number of candidate identities in the synthetic scene.
    pub identities: usize,
    /// Ideal chance-level re-identification accuracy (`1 / identities`).
    pub chance_level: f32,
    /// Modeled passive re-identification accuracy with the shield **off**.
    pub reid_accuracy_off: f32,
    /// Modeled passive re-identification accuracy with the shield **on**.
    pub reid_accuracy_on: f32,
    /// Modeled protected-link throughput as a fraction of baseline.
    pub throughput_ratio: f64,
    /// `output_energy / input_energy` of a representative protected frame.
    /// ~1.0 means the control is energy-preserving (compliant, not jamming).
    pub energy_ratio: f32,
    /// True iff the energy ratio is within tolerance of 1.0.
    pub energy_conserving: bool,
    /// True iff the shield drove re-identification into the accepted
    /// chance band.
    pub drives_to_chance: bool,
    /// Evidence label; always [`VEIL_EVIDENCE`].
    pub evidence: &'static str,
}

impl ShieldAssessment {
    /// Residual re-identification margin above chance with the shield on.
    ///
    /// `0.0` (or below) means the modeled attacker is at or below chance;
    /// larger values mean residual identity leakage in the model.
    #[must_use]
    pub fn residual_reid_margin(&self) -> f32 {
        self.reid_accuracy_on - self.chance_level
    }

    /// One-line human-readable summary, evidence-tagged.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "[{}] re-ID {:.1}% -> {:.1}% (chance {:.1}%, {} identities), \
             throughput {:.1}%, energy ratio {:.6} ({})",
            self.evidence,
            self.reid_accuracy_off * 100.0,
            self.reid_accuracy_on * 100.0,
            self.chance_level * 100.0,
            self.identities,
            self.throughput_ratio * 100.0,
            self.energy_ratio,
            if self.energy_conserving {
                "energy-conserving"
            } else {
                "NOT energy-conserving"
            },
        )
    }
}

impl From<wifi_veil::ExperimentReport> for ShieldAssessment {
    fn from(r: wifi_veil::ExperimentReport) -> Self {
        Self {
            identities: r.identities,
            chance_level: r.chance_level,
            reid_accuracy_off: r.accuracy_shield_off,
            reid_accuracy_on: r.accuracy_shield_on,
            throughput_ratio: r.throughput_ratio,
            energy_ratio: r.compliance.energy_ratio,
            energy_conserving: r.compliance.energy_conserving,
            drives_to_chance: r.drives_to_chance(),
            evidence: VEIL_EVIDENCE,
        }
    }
}

/// Run the deterministic attacker-vs-protector experiment for `cfg`.
///
/// Fully deterministic: identical configs produce identical assessments.
#[must_use]
pub fn assess(cfg: &ExperimentConfig) -> ShieldAssessment {
    experiment::run(cfg).into()
}

/// Run the experiment with wifi-veil's shipped default scene and shield.
#[must_use]
pub fn assess_default() -> ShieldAssessment {
    assess(&ExperimentConfig::default())
}

/// Derive the optimizer-shipped shield configuration and its verifying
/// assessment for `base`.
///
/// Wraps [`wifi_veil::hyper_optimize`]: the returned shield uses the
/// spec-allowed throughput-optimal feedback resolution and a Givens-pass
/// count grown by the privacy margin factor.
#[must_use]
pub fn optimized_shield(base: &ExperimentConfig) -> (wifi_veil::ShieldConfig, ShieldAssessment) {
    let hyper = wifi_veil::hyper_optimize(base);
    let assessment = hyper.report.into();
    (hyper.shield, assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_assessment_is_deterministic() {
        let a = assess_default();
        let b = assess_default();
        assert_eq!(a, b);
    }

    #[test]
    fn shield_reduces_modeled_reid_accuracy() {
        let a = assess_default();
        assert!(
            a.reid_accuracy_on < a.reid_accuracy_off,
            "shield-on accuracy {} must be below shield-off {}",
            a.reid_accuracy_on,
            a.reid_accuracy_off
        );
    }

    #[test]
    fn default_shield_is_compliant_and_at_chance() {
        let a = assess_default();
        assert!(a.energy_conserving, "veil must be energy-preserving");
        assert!(a.drives_to_chance, "shipped default must reach chance band");
        assert!(a.throughput_ratio > 0.9, "throughput ratio {} too low", a.throughput_ratio);
    }

    #[test]
    fn evidence_label_is_synthetic_l0() {
        let a = assess_default();
        assert_eq!(a.evidence, VEIL_EVIDENCE);
        assert!(a.summary().starts_with("[SYNTHETIC/L0]"));
    }

    #[test]
    fn residual_margin_matches_fields() {
        let a = assess_default();
        let m = a.residual_reid_margin();
        assert!((m - (a.reid_accuracy_on - a.chance_level)).abs() < f32::EPSILON);
    }

    #[test]
    fn optimized_shield_verifies() {
        let (shield, assessment) = optimized_shield(&ExperimentConfig::default());
        assert!(shield.enabled);
        assert!(assessment.drives_to_chance);
        assert!(assessment.energy_conserving);
    }
}
