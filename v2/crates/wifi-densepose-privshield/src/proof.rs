//! Deterministic proof bundle — the byte-stable witness for VEIL.
//!
//! Mirrors the `nvsim` / `archive/v1` proof pattern: run a fixed reference
//! experiment, fold its salient outputs into a single FNV-1a witness, and pin
//! that witness as a constant. If any constant drifts — the PRNG stream, the
//! rotation schedule, the throughput formula, the scene geometry — the witness
//! changes and the test fails loudly.
//!
//! The witness is derived from **quantized** outputs (accuracies to 1e-4,
//! throughput to 1e-6) so that legitimate cross-platform f32 round-off in the
//! last bits does not spuriously break the proof, while any real change to the
//! experiment's behavior still does.

use crate::experiment::{run, ExperimentConfig, ExperimentReport};
use crate::prng::fnv1a_64;

/// Deterministic-proof harness.
pub struct Proof;

impl Proof {
    /// Pinned witness over the reference experiment. Re-derived by
    /// [`Proof::witness`]; asserted by the test below.
    pub const EXPECTED_WITNESS: u64 = 0xD098_C38D_B7C6_BCA9;

    /// The reference configuration. Uses every default so the proof tracks the
    /// shipped behavior of the crate.
    #[must_use]
    pub fn reference_config() -> ExperimentConfig {
        ExperimentConfig::default()
    }

    /// Run the reference experiment.
    #[must_use]
    pub fn run_reference() -> ExperimentReport {
        run(&Self::reference_config())
    }

    /// Fold a report's salient outputs into a stable witness.
    #[must_use]
    pub fn witness(report: &ExperimentReport) -> u64 {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(report.identities as u64).to_le_bytes());
        // Quantize floats before folding so last-bit round-off is not part of
        // the witness.
        let q4 = |x: f32| (f64::from(x) * 10_000.0).round() as i64;
        let q6 = |x: f64| (x * 1_000_000.0).round() as i64;
        buf.extend_from_slice(&q4(report.chance_level).to_le_bytes());
        buf.extend_from_slice(&q4(report.accuracy_shield_off).to_le_bytes());
        buf.extend_from_slice(&q4(report.accuracy_shield_on).to_le_bytes());
        buf.extend_from_slice(&q6(report.throughput_ratio).to_le_bytes());
        buf.extend_from_slice(&q4(report.compliance.energy_ratio).to_le_bytes());
        fnv1a_64(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_experiment_passes() {
        assert!(Proof::run_reference().passed());
    }

    #[test]
    fn witness_is_stable() {
        let a = Proof::witness(&Proof::run_reference());
        let b = Proof::witness(&Proof::run_reference());
        assert_eq!(a, b, "witness must be reproducible");
    }

    #[test]
    fn witness_matches_pinned() {
        let w = Proof::witness(&Proof::run_reference());
        assert_eq!(
            w,
            Proof::EXPECTED_WITNESS,
            "witness drifted to {w:#018x}; update EXPECTED_WITNESS only if the \
             change to the reference experiment is intentional"
        );
    }
}
