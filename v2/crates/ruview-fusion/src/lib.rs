//! # `ruview-fusion` — uncertainty-aware sensor fusion (ADR-308, ADR-297 §11)
//!
//! **Many observations resolve to one probabilistic world state, not many feeds
//! into a visualization.** This is the defining invariant of ADR-308: a
//! dashboard that shows a WiFi layer, a mmWave layer, and a BLE layer side by
//! side is not fusion — it pushes reconciliation onto the human. Real fusion
//! produces *one* uncertainty-aware [`WorldState`] that every downstream
//! consumer (ADR-309 spatial memory, ADR-310 counterfactual, ADR-312 RF twin)
//! reads, with each contributing observation's provenance and confidence still
//! recoverable.
//!
//! [`FusionEngine`] ingests a set of [`PresenceObservation`]s — canonical
//! ADR-303 [`HalObservation`](ruview_hal::HalObservation)s paired with a
//! per-source occupancy [`Claim`] — that may span modalities (WiFi/CSI, BLE,
//! UWB, mmWave, …) and may conflict, and emits a single [`WorldState`]: a fused
//! per-container occupancy probability with a fused variance, the set of
//! contributing observations as recoverable provenance, and an aggregate
//! evidence level.
//!
//! ## How sources are combined
//!
//! Presence estimates combine by **inverse-variance weighting** (see
//! [`estimate::combine`]), the optimal linear combination of independent
//! Gaussian estimates. Concretely, for sources with probabilities `pᵢ` and
//! variances `vᵢ`, with precisions `wᵢ = 1/vᵢ`:
//!
//! ```text
//! fused mean     = Σ wᵢ pᵢ / Σ wᵢ
//! fused variance = 1 / Σ wᵢ
//! ```
//!
//! This is deliberately **not** a naive average:
//!
//! - **Agreement sharpens.** Two agreeing sources yield a fused variance
//!   *smaller* than either input — the belief gets more certain, which a mean
//!   can never do.
//! - **Uncertainty is respected.** A high-variance source gets a small weight
//!   and barely moves the fused value; it is down-weighted, not averaged in as
//!   if trustworthy.
//!
//! ## When the answer is UNKNOWN (ADR-297 rule 1)
//!
//! UNKNOWN is a first-class world-state value, never an error or a panic:
//!
//! - **Irreconcilable conflict.** When sources disagree by more than their
//!   stated uncertainty allows — measured by a reduced chi-square statistic
//!   against a configured threshold — the zone resolves to
//!   [`UnknownReason::IrreconcilableConflict`] instead of a confident average
//!   near the midpoint of two contradictory claims.
//! - **Insufficient coverage.** When too few usable observations cover a zone
//!   (all degraded/abstaining, or below the configured minimum), the zone
//!   resolves to [`UnknownReason::InsufficientCoverage`].
//!
//! ## Evidence and honesty discipline
//!
//! The fused evidence level is the **minimum** over contributing observations —
//! never lifted above the weakest necessary input (ADR-308). This crate asserts
//! **no accuracy number and makes no camera-grade claim** (CLAUDE.md, ADR-282);
//! its tests use synthetic in-code fixtures only (SYNTHETIC / L0..L2). It is a
//! pure, deterministic function of its inputs: no I/O, no clock, no randomness,
//! and malformed input abstains rather than panicking.
//!
//! ## Example
//!
//! ```
//! use ruview_fusion::{FusionEngine, PresenceObservation, Presence};
//! use ruview_hal::{HalObservation, Modality, Uncertainty};
//! use ruview_ontology::{
//!     Container, EvidenceLevel, Observation, ObservationId, SemanticProvenance, SensorId, SpaceId,
//! };
//!
//! fn hal(id: &str, sensor: &str, modality: Modality) -> HalObservation {
//!     HalObservation {
//!         modality,
//!         uncertainty: Uncertainty::known(0.9),
//!         observation: Observation {
//!             id: ObservationId::new(id).unwrap(),
//!             sensor: SensorId::new(sensor).unwrap(),
//!             located_in: Container::Space { id: SpaceId::new("kitchen").unwrap() },
//!             at_unix_ms: 1_000,
//!             evidence_level: EvidenceLevel::L2,
//!             provenance: SemanticProvenance::declared("fusion@1"),
//!         },
//!     }
//! }
//!
//! let engine = FusionEngine::default();
//! // WiFi and mmWave agree the kitchen is occupied — the belief sharpens.
//! let world = engine.fuse(&[
//!     PresenceObservation::estimated(hal("o1", "csi-1", Modality::Csi), 0.90, 0.04),
//!     PresenceObservation::estimated(hal("o2", "mm-1", Modality::Mmwave), 0.88, 0.04),
//! ]);
//!
//! let kitchen = Container::Space { id: SpaceId::new("kitchen").unwrap() };
//! let zone = world.zone(&kitchen).unwrap();
//! match zone.presence {
//!     Presence::Estimated { probability, variance } => {
//!         assert!(probability > 0.85 && probability < 0.92);
//!         assert!(variance < 0.04); // sharper than either input
//!     }
//!     Presence::Unknown { .. } => unreachable!(),
//! }
//! // Both observations' provenance is recoverable.
//! assert_eq!(zone.contributions.len(), 2);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod estimate;
mod engine;
mod observation;
mod world;

pub use engine::{FusionConfig, FusionEngine};
pub use estimate::Estimate;
pub use observation::{Claim, PresenceObservation};
pub use world::{Contribution, Presence, UnknownReason, WorldState, ZoneState};

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_hal::{HalObservation, Modality, Uncertainty};
    use ruview_ontology::{
        Container, EvidenceLevel, Observation, ObservationId, SemanticProvenance, SensorId, SpaceId,
    };

    const KITCHEN: &str = "kitchen";

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn container(space: &str) -> Container {
        Container::Space {
            id: SpaceId::new(space).unwrap(),
        }
    }

    /// A synthetic, non-degraded HAL observation in the given space.
    fn hal(id: &str, sensor: &str, modality: Modality, space: &str, ev: EvidenceLevel) -> HalObservation {
        HalObservation {
            modality,
            uncertainty: Uncertainty::known(0.9),
            observation: Observation {
                id: ObservationId::new(id).unwrap(),
                sensor: SensorId::new(sensor).unwrap(),
                located_in: container(space),
                at_unix_ms: 1_700_000_000_000,
                evidence_level: ev,
                provenance: SemanticProvenance::declared("synthetic-fusion@0"),
            },
        }
    }

    /// A degraded (malformed-input) HAL observation, as the HAL emits for bad
    /// raw frames: UNKNOWN/degraded uncertainty.
    fn degraded_hal(id: &str, sensor: &str, space: &str) -> HalObservation {
        HalObservation {
            modality: Modality::Csi,
            uncertainty: Uncertainty::degraded(),
            observation: Observation {
                id: ObservationId::new(id).unwrap(),
                sensor: SensorId::new(sensor).unwrap(),
                located_in: container(space),
                at_unix_ms: 1_700_000_000_000,
                evidence_level: EvidenceLevel::L0,
                provenance: SemanticProvenance::declared("synthetic-fusion@0"),
            },
        }
    }

    fn est_probability(p: &Presence) -> f64 {
        match *p {
            Presence::Estimated { probability, .. } => probability,
            Presence::Unknown { .. } => panic!("expected Estimated"),
        }
    }

    fn est_variance(p: &Presence) -> f64 {
        match *p {
            Presence::Estimated { variance, .. } => variance,
            Presence::Unknown { .. } => panic!("expected Estimated"),
        }
    }

    // Two agreeing observations SHARPEN the estimate: the fused variance is
    // strictly smaller than either contributing variance.
    #[test]
    fn agreeing_observations_sharpen() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            PresenceObservation::estimated(
                hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.90,
                0.04,
            ),
            PresenceObservation::estimated(
                hal("o2", "mm-1", Modality::Mmwave, KITCHEN, EvidenceLevel::L2),
                0.88,
                0.04,
            ),
        ]);

        assert_eq!(world.zones.len(), 1, "one world state, one zone — not two feeds");
        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(!zone.is_unknown());
        // Inverse-variance of two equal variances: 1/(1/0.04 + 1/0.04) = 0.02.
        assert!(approx(est_variance(&zone.presence), 0.02));
        assert!(est_variance(&zone.presence) < 0.04);
        // Mean lies between the two agreeing inputs.
        let p = est_probability(&zone.presence);
        assert!(p > 0.88 && p < 0.90);
    }

    // A high-uncertainty observation is DOWN-WEIGHTED: it barely moves the fused
    // value away from the precise source, and its recorded weight is tiny.
    #[test]
    fn high_uncertainty_observation_is_down_weighted() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            // Precise: p=0.9, v=0.01 (precision 100).
            PresenceObservation::estimated(
                hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.90,
                0.01,
            ),
            // Very uncertain: p=0.2, v=1.0 (precision 1).
            PresenceObservation::estimated(
                hal("o2", "ble-1", Modality::Ble, KITCHEN, EvidenceLevel::L2),
                0.20,
                1.0,
            ),
        ]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(!zone.is_unknown());
        // The uncertain source pulls the fused value only slightly off 0.9.
        let p = est_probability(&zone.presence);
        assert!((p - 0.9).abs() < 0.02, "fused {p} should stay near the precise 0.9");

        // The precise source carries almost all the weight.
        let precise = zone
            .contributions
            .iter()
            .find(|c| c.observation.as_str() == "o1")
            .unwrap();
        let uncertain = zone
            .contributions
            .iter()
            .find(|c| c.observation.as_str() == "o2")
            .unwrap();
        assert!(precise.weight > 0.98);
        assert!(uncertain.weight < 0.02);
        assert!(uncertain.weight < precise.weight);
    }

    // Irreconcilable conflict yields UNKNOWN, NOT a confident average near 0.5.
    #[test]
    fn irreconcilable_conflict_yields_unknown() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            // Confident "occupied".
            PresenceObservation::estimated(
                hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.95,
                0.01,
            ),
            // Confident "empty" — directly contradicts, both low-variance.
            PresenceObservation::estimated(
                hal("o2", "mm-1", Modality::Mmwave, KITCHEN, EvidenceLevel::L2),
                0.05,
                0.01,
            ),
        ]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(zone.is_unknown(), "conflict must not collapse to a confident average");
        match zone.presence {
            Presence::Unknown {
                reason: UnknownReason::IrreconcilableConflict { reduced_chi_square },
            } => {
                assert!(reduced_chi_square > 9.0);
            }
            other => panic!("expected IrreconcilableConflict, got {other:?}"),
        }
        // Both contradictory observations are still recorded as provenance.
        assert_eq!(zone.contributions.len(), 2);
        assert_eq!(zone.contributing_observations().count(), 2);
    }

    // A single observation PASSES THROUGH with its own probability and
    // uncertainty (no artificial sharpening, no conflict).
    #[test]
    fn single_observation_passes_through() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[PresenceObservation::estimated(
            hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
            0.70,
            0.05,
        )]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(!zone.is_unknown());
        assert!(approx(est_probability(&zone.presence), 0.70));
        assert!(approx(est_variance(&zone.presence), 0.05));
        assert_eq!(zone.contributions.len(), 1);
        // The lone source carries all the weight.
        assert!(approx(zone.contributions[0].weight, 1.0));
        assert_eq!(zone.evidence_level, EvidenceLevel::L2);
    }

    // Provenance is preserved: contributing observation ids and modalities are
    // recoverable from the fused state.
    #[test]
    fn provenance_is_preserved() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            PresenceObservation::estimated(
                hal("wifi-obs", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.80,
                0.05,
            ),
            PresenceObservation::estimated(
                hal("mm-obs", "mm-1", Modality::Mmwave, KITCHEN, EvidenceLevel::L3),
                0.82,
                0.05,
            ),
        ]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        let ids: Vec<&str> = zone.contributions.iter().map(|c| c.observation.as_str()).collect();
        assert!(ids.contains(&"wifi-obs"));
        assert!(ids.contains(&"mm-obs"));
        let modalities: Vec<&Modality> = zone.contributions.iter().map(|c| &c.modality).collect();
        assert!(modalities.contains(&&Modality::Csi));
        assert!(modalities.contains(&&Modality::Mmwave));
        // Aggregate evidence is the minimum (weakest) contributing level.
        assert_eq!(zone.evidence_level, EvidenceLevel::L2);
    }

    // A degraded / abstaining observation is not counted as coverage: a zone
    // with no usable estimate resolves to UNKNOWN (insufficient coverage), and
    // the abstaining observation is still recorded (weight 0, no estimate).
    #[test]
    fn insufficient_coverage_yields_unknown() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[PresenceObservation::estimated(
            degraded_hal("bad-obs", "csi-1", KITCHEN),
            0.9,
            0.01,
        )]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(zone.is_unknown());
        assert!(matches!(
            zone.presence,
            Presence::Unknown {
                reason: UnknownReason::InsufficientCoverage
            }
        ));
        // Provenance still records the abstaining observation.
        assert_eq!(zone.contributions.len(), 1);
        assert!(!zone.contributions[0].contributed());
        assert!(approx(zone.contributions[0].weight, 0.0));
        assert_eq!(zone.contributing_observations().count(), 0);
        assert_eq!(zone.evidence_level, EvidenceLevel::L0);
    }

    // An explicitly abstaining source (Claim::Unknown) is uncertainty-first-class
    // and does not error.
    #[test]
    fn explicit_unknown_claim_abstains() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            PresenceObservation::unknown(hal("abstain", "ble-1", Modality::Ble, KITCHEN, EvidenceLevel::L1)),
            PresenceObservation::estimated(
                hal("real", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.75,
                0.05,
            ),
        ]);

        let zone = world.zone(&container(KITCHEN)).unwrap();
        // The one real source resolves the zone; the abstainer only adds provenance.
        assert!(!zone.is_unknown());
        assert!(approx(est_probability(&zone.presence), 0.75));
        assert_eq!(zone.contributions.len(), 2);
        assert_eq!(zone.contributing_observations().count(), 1);
    }

    // Distinct containers fuse independently into one world state.
    #[test]
    fn distinct_zones_fuse_independently() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            PresenceObservation::estimated(
                hal("k1", "csi-1", Modality::Csi, "kitchen", EvidenceLevel::L2),
                0.9,
                0.04,
            ),
            PresenceObservation::estimated(
                hal("b1", "csi-2", Modality::Csi, "bedroom", EvidenceLevel::L2),
                0.1,
                0.04,
            ),
        ]);

        assert_eq!(world.zones.len(), 2);
        assert!(approx(est_probability(&world.zone(&container("kitchen")).unwrap().presence), 0.9));
        assert!(approx(est_probability(&world.zone(&container("bedroom")).unwrap().presence), 0.1));
    }

    // Determinism: identical inputs (in any order) fuse to the identical world
    // state.
    #[test]
    fn fusion_is_deterministic_and_order_independent() {
        let engine = FusionEngine::default();
        let a = PresenceObservation::estimated(
            hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
            0.90,
            0.03,
        );
        let b = PresenceObservation::estimated(
            hal("o2", "mm-1", Modality::Mmwave, KITCHEN, EvidenceLevel::L3),
            0.86,
            0.07,
        );

        let world1 = engine.fuse(&[a.clone(), b.clone()]);
        let world2 = engine.fuse(&[a.clone(), b.clone()]);
        assert_eq!(world1, world2);

        // Reordering the inputs does not change the fused state.
        let world3 = engine.fuse(&[b, a]);
        assert_eq!(world1, world3);
    }

    // The world state serde round-trips losslessly (one canonical semantics).
    #[test]
    fn world_state_serde_round_trip() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[
            PresenceObservation::estimated(
                hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
                0.9,
                0.04,
            ),
            PresenceObservation::estimated(
                degraded_hal("o2", "csi-2", KITCHEN),
                0.0,
                0.0,
            ),
        ]);
        let json = serde_json::to_string(&world).unwrap();
        let back: WorldState = serde_json::from_str(&json).unwrap();
        assert_eq!(world, back);
    }

    // A configured higher minimum coverage forces UNKNOWN when too few sources
    // cover a zone, even if the single source is confident.
    #[test]
    fn min_observations_gate() {
        let engine = FusionEngine::new(FusionConfig {
            conflict_reduced_chi_square: 9.0,
            min_observations: 2,
        });
        let world = engine.fuse(&[PresenceObservation::estimated(
            hal("o1", "csi-1", Modality::Csi, KITCHEN, EvidenceLevel::L2),
            0.95,
            0.01,
        )]);
        let zone = world.zone(&container(KITCHEN)).unwrap();
        assert!(matches!(
            zone.presence,
            Presence::Unknown {
                reason: UnknownReason::InsufficientCoverage
            }
        ));
    }

    #[test]
    fn empty_input_is_empty_world_not_panic() {
        let engine = FusionEngine::default();
        let world = engine.fuse(&[]);
        assert_eq!(world.zones.len(), 0);
        assert_eq!(world.at_unix_ms, 0);
    }
}
