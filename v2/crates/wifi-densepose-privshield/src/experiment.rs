//! The attacker-vs-protector head-to-head.
//!
//! This is the "one node is the attacker, one node is the protector" experiment
//! from the project brief, in deterministic synthetic form. It runs the passive
//! re-identification attacker ([`crate::attacker`]) twice — once against
//! unprotected traffic and once against traffic shaped by the protector
//! ([`crate::protector`]) — and reports both accuracies against the chance
//! floor, alongside the modeled link throughput ([`crate::throughput`]) and a
//! compliance audit ([`crate::compliance`]).
//!
//! Success criteria (the brief's own bar):
//! 1. protection drives re-identification toward chance (`1/identities`);
//! 2. throughput stays above 95% of the unshielded baseline;
//! 3. the control is compliant (energy-preserving, non-jamming).

use crate::attacker::{
    AdaptivePoolingAttacker, AttackerKind, Metric, NearestCentroidAttacker, ReconstructionAttacker,
};
use crate::compliance::ComplianceReport;
use crate::identity::{Channel, SceneConfig};
use crate::prng::derive_key;
use crate::protector::{ObfMode, Protector, ShieldConfig};
use crate::throughput::LinkModel;

/// Configuration for a full experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Synthetic scene.
    pub scene: SceneConfig,
    /// Protector configuration.
    pub shield: ShieldConfig,
    /// Link model for the throughput estimate.
    pub link: LinkModel,
    /// Enrollment sessions per identity.
    pub enroll_sessions: u64,
    /// Test sessions per identity.
    pub test_sessions: u64,
    /// Accept re-ID as "at chance" if it is at or below
    /// `chance × chance_multiple + chance_margin`.
    pub chance_multiple: f32,
    /// Additive slack on the chance band.
    pub chance_margin: f32,
    /// Minimum acceptable throughput ratio.
    pub min_throughput_ratio: f64,
    /// Metric the passive attacker uses (for the nearest-centroid kind).
    pub attacker_metric: Metric,
    /// Which adversary shape to run.
    pub attacker_kind: AttackerKind,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            scene: SceneConfig::default(),
            shield: ShieldConfig::default(),
            link: LinkModel::default(),
            enroll_sessions: 12,
            test_sessions: 12,
            chance_multiple: 2.0,
            chance_margin: 0.03,
            min_throughput_ratio: 0.95,
            attacker_metric: Metric::Euclidean,
            attacker_kind: AttackerKind::NearestCentroid,
        }
    }
}

/// The outcome of an experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentReport {
    /// Number of candidate identities.
    pub identities: usize,
    /// Ideal chance-level accuracy (`1/identities`).
    pub chance_level: f32,
    /// Re-identification accuracy with the shield off.
    pub accuracy_shield_off: f32,
    /// Re-identification accuracy with the shield on.
    pub accuracy_shield_on: f32,
    /// Modeled throughput ratio of the protected link vs baseline.
    pub throughput_ratio: f64,
    /// Compliance audit of a representative protected frame.
    pub compliance: ComplianceReport,
    /// Upper edge of the accepted "at chance" band.
    pub chance_band: f32,
}

impl ExperimentReport {
    /// Did protection drive re-identification into the chance band?
    #[must_use]
    pub fn drives_to_chance(&self) -> bool {
        self.accuracy_shield_on <= self.chance_band
    }

    /// Is the shield-off attacker meaningfully better than chance (i.e. the
    /// threat is real in this scene, so the collapse is meaningful)?
    #[must_use]
    pub fn attack_is_effective_without_shield(&self) -> bool {
        self.accuracy_shield_off >= 0.5
    }

    /// Did throughput stay above the required floor?
    #[must_use]
    pub fn preserves_throughput(&self) -> bool {
        self.throughput_ratio >= 0.95
    }

    /// Overall pass: real threat, collapsed to chance, throughput preserved,
    /// and compliant.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.attack_is_effective_without_shield()
            && self.drives_to_chance()
            && self.preserves_throughput()
            && self.compliance.is_compliant()
    }
}

/// Build the enroll/test capture sets for a given shield, then measure attacker
/// accuracy. `shield_on` selects whether the protector is applied to every
/// captured frame (the attacker only ever sees what is transmitted).
fn measure_accuracy(
    cfg: &ExperimentConfig,
    ch: &Channel,
    protector: &Protector,
    shield_on: bool,
) -> f32 {
    let mut enroll = Vec::new();
    let mut test = Vec::new();

    for id in 0..cfg.scene.identities {
        for s in 0..cfg.enroll_sessions {
            let raw = ch.observe(id, b"enroll", s);
            let seen = if shield_on {
                protector.protect(&raw, rotation_key(cfg, b"enroll", s, id))
            } else {
                raw
            };
            enroll.push((id, seen));
        }
        for s in 0..cfg.test_sessions {
            let raw = ch.observe(id, b"test", s);
            let seen = if shield_on {
                protector.protect(&raw, rotation_key(cfg, b"test", s, id))
            } else {
                raw
            };
            test.push((id, seen));
        }
    }

    // Dispatch on the adversary shape (SOTA sweep, ADR-288 §sota).
    match cfg.attacker_kind {
        AttackerKind::NearestCentroid => {
            let mut a = NearestCentroidAttacker::with_metric(cfg.attacker_metric);
            a.enroll(&enroll);
            a.accuracy(&test)
        }
        AttackerKind::Reconstruction => {
            let mut a = ReconstructionAttacker::new();
            a.enroll(&enroll);
            a.accuracy(&test)
        }
        AttackerKind::AdaptivePooling => {
            let mut a = AdaptivePoolingAttacker::new();
            a.enroll(&enroll);
            a.accuracy(&test)
        }
    }
}

/// Derive the rotation key for a capture. In [`ObfMode::KeyedRotation`] the key
/// is per **session** (same rotation for every identity present in that sounding
/// interval — the AP rotates its precoder per interval, not per person; this is
/// what a legitimate receiver inverts and what makes cross-session averaging
/// collapse). In [`ObfMode::PerPacketUnitary`] it is per **packet** (unique per
/// capture), modeling the AP-side, client-transparent fresh-unitary defense.
/// The `KeyedRotation` labels are unchanged from the original so the reference
/// witness is stable.
fn rotation_key(cfg: &ExperimentConfig, phase: &[u8], session: u64, id: usize) -> u64 {
    match cfg.shield.mode {
        ObfMode::KeyedRotation => {
            let label: &[u8] = if phase == b"enroll" {
                b"rot-enroll"
            } else {
                b"rot-test"
            };
            derive_key(cfg.scene.seed, label, session, 0)
        }
        ObfMode::PerPacketUnitary => {
            let label: &[u8] = if phase == b"enroll" {
                b"rot-enroll-pkt"
            } else {
                b"rot-test-pkt"
            };
            derive_key(cfg.scene.seed, label, session, id as u64)
        }
    }
}

/// Run the full attacker-vs-protector experiment.
#[must_use]
pub fn run(cfg: &ExperimentConfig) -> ExperimentReport {
    let protector = Protector::new(cfg.shield.clone());
    let ch = Channel::new(cfg.scene.clone());

    let accuracy_shield_off = measure_accuracy(cfg, &ch, &protector, false);
    let accuracy_shield_on = measure_accuracy(cfg, &ch, &protector, true);

    let throughput_ratio = cfg.link.throughput_ratio(&cfg.shield);

    // Representative compliance audit: one protected frame vs its clean form.
    let clean = ch.observe(0, b"test", 0);
    let protected = protector.protect(&clean, derive_key(cfg.scene.seed, b"rot-test", 0, 0));
    let compliance = ComplianceReport::audit(&clean, &protected);

    let chance_level = cfg.scene.chance_level();
    let chance_band = chance_level * cfg.chance_multiple + cfg.chance_margin;

    ExperimentReport {
        identities: cfg.scene.identities,
        chance_level,
        accuracy_shield_off,
        accuracy_shield_on,
        throughput_ratio,
        compliance,
        chance_band,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shield_off_attack_succeeds() {
        let report = run(&ExperimentConfig::default());
        assert!(
            report.attack_is_effective_without_shield(),
            "shield-off accuracy {} should be well above chance {}",
            report.accuracy_shield_off,
            report.chance_level
        );
    }

    #[test]
    fn shield_on_drives_to_chance() {
        let report = run(&ExperimentConfig::default());
        assert!(
            report.drives_to_chance(),
            "shield-on accuracy {} should be within chance band {}",
            report.accuracy_shield_on,
            report.chance_band
        );
    }

    #[test]
    fn shield_preserves_throughput() {
        let report = run(&ExperimentConfig::default());
        assert!(
            report.preserves_throughput(),
            "throughput ratio {} below 0.95",
            report.throughput_ratio
        );
    }

    #[test]
    fn overall_experiment_passes() {
        let report = run(&ExperimentConfig::default());
        assert!(report.passed(), "{report:#?}");
    }

    #[test]
    fn experiment_is_deterministic() {
        assert_eq!(
            run(&ExperimentConfig::default()),
            run(&ExperimentConfig::default())
        );
    }

    // ---- SOTA-driven adversaries and modes (ADR-288 §sota) ----

    #[test]
    fn reconstruction_attacker_collapses() {
        // BFIAttack-style: reconstruction recovers the *rotated* CSI direction,
        // so a secret orthogonal rotation still drives it to chance — but it
        // works fine on unprotected traffic (sanity that the attacker is real).
        let cfg = ExperimentConfig {
            attacker_kind: AttackerKind::Reconstruction,
            ..ExperimentConfig::default()
        };
        let r = run(&cfg);
        assert!(
            r.accuracy_shield_off >= 0.5,
            "recon off {}",
            r.accuracy_shield_off
        );
        assert!(r.drives_to_chance(), "recon on {}", r.accuracy_shield_on);
    }

    #[test]
    fn adaptive_pooling_attacker_collapses() {
        let cfg = ExperimentConfig {
            attacker_kind: AttackerKind::AdaptivePooling,
            ..ExperimentConfig::default()
        };
        let r = run(&cfg);
        assert!(
            r.accuracy_shield_off >= 0.5,
            "pool off {}",
            r.accuracy_shield_off
        );
        assert!(r.drives_to_chance(), "pool on {}", r.accuracy_shield_on);
    }

    #[test]
    fn per_packet_unitary_mode_collapses_and_is_compliant() {
        let cfg = ExperimentConfig {
            shield: ShieldConfig {
                mode: ObfMode::PerPacketUnitary,
                ..ShieldConfig::default()
            },
            ..ExperimentConfig::default()
        };
        let r = run(&cfg);
        assert!(
            r.drives_to_chance(),
            "per-packet on {}",
            r.accuracy_shield_on
        );
        assert!(r.compliance.is_compliant());
    }

    #[test]
    fn dp_epsilon_still_collapses_and_stays_compliant() {
        // Layering the ε-DP dither on the rotation keeps the collapse and, thanks
        // to renormalization, keeps the emission energy-preserving (not jamming).
        let cfg = ExperimentConfig {
            shield: ShieldConfig {
                dp_epsilon: Some(1.0),
                ..ShieldConfig::default()
            },
            ..ExperimentConfig::default()
        };
        let r = run(&cfg);
        assert!(r.drives_to_chance());
        assert!(
            r.compliance.is_compliant(),
            "energy {}",
            r.compliance.energy_ratio
        );
    }
}
