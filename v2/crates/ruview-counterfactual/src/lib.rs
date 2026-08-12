//! # `ruview-counterfactual` — counterfactual spatial inference (ADR-313, ADR-300 phase 3)
//!
//! **SYNTHETIC / L0 — a research-forward generative-scoring scaffold, not a
//! measurement system.**
//!
//! This crate is a *phase-3, research-forward primitive*: a step beyond
//! discriminative classifiers toward a **generative spatial model**. A
//! classifier maps measurements to a label; it cannot say *"the observation is
//! better explained by absence"* or *"one occupant explains this better than
//! two,"* because it has no model of what a measurement *should* look like under
//! a hypothesized world state. This layer supplies that missing piece: given an
//! observed link-measurement set and the ADR-315 digital RF twin as the
//! generative forward model, it scores a small set of scene
//! [`Hypothesis`](Hypothesis) — including the **null hypothesis** (nobody
//! present) — and returns the maximum-likelihood explanation with a **margin**,
//! or a first-class `UNKNOWN` when the hypotheses are near-indistinguishable.
//!
//! It is a **consumer** of the twin, never a second simulator (ADR-313 option 3,
//! rejecting option 2): the ADR-315 twin supplies each link's baseline expected
//! distribution (geometry + propagation), and this layer applies a **documented
//! SYNTHETIC occupant-attenuation model** — a hypothesized occupant attenuates
//! any link whose line of sight passes near it. Hypotheses are drawn from the
//! ADR-311 fused world state and its neighbourhood and are expressed over the
//! canonical ADR-306 [`SpaceId`](ruview_ontology::SpaceId), so a counterfactual
//! result is a governed spatial statement, not an opaque score (ADR-300 rule 3).
//!
//! ## Honesty and evidence discipline (CLAUDE.md, ADR-313)
//!
//! Every likelihood is a **model-relative** score under a twin whose
//! distributions are a simulation at evidence level `L0`, labelled `SYNTHETIC`.
//! A twin *predicts*; it does not *measure*. A counterfactual verdict inherits
//! the `L0` level of its weakest input and is **never** presented as
//! camera-grade ground truth. This crate makes **no** hardware, `MEASURED`, or
//! accuracy claim, and asserts **no** discrimination-accuracy number (e.g. it
//! does not claim to "distinguish one occupant from two"); any such number would
//! require the mean-pose-style baseline discipline, a leakage-free held-out
//! split, and a reproducer before it could be tagged `MEASURED`.
//!
//! ## UNKNOWN is a first-class output (ADR-300 rule 1, ADR-313 §3)
//!
//! The layer never forces a label. The best explanation resolves to a
//! first-class [`BestExplanation::Unknown`] when the top two hypotheses are
//! near-indistinguishable (margin below threshold), when **no** hypothesis
//! explains the observation well (best mean per-link log-likelihood below a
//! floor — routed to the ADR-302 `UNKNOWN` verdict), when no hypotheses are
//! supplied, or when no observed link is evaluable against the twin. UNKNOWN is
//! a value, never an error, a panic, or a confident default.
//!
//! ## Determinism
//!
//! Everything is a pure, deterministic function of its inputs: no I/O, no clock,
//! and no randomness. Synthetic scenes vary only by the twin's explicit
//! [`seed`](ruview_twin::DeploymentDescription::seed); malformed input abstains
//! (first-class UNKNOWN or a typed error) rather than panicking, and allocation
//! is bounded by [`MAX_OCCUPANTS`] and [`MAX_HYPOTHESES`].
//!
//! ```
//! use ruview_counterfactual::*;
//! use ruview_twin::{synthetic_deployment, RfTwin};
//! use ruview_ontology::SpaceId;
//!
//! let twin = RfTwin::build(synthetic_deployment(7)).unwrap();
//! let space = SpaceId::new("space-7").unwrap();
//! let model = OccupantModel::default_body();
//!
//! // An empty-room observation set (nobody present) is best explained by the
//! // null hypothesis over a one-occupant alternative.
//! let empty = Hypothesis::empty("empty", space.clone());
//! let one = Hypothesis::new("one", space, vec![Occupant::new(2.5, 2.0)]).unwrap();
//! let observed = synthesize_observations(&twin, &empty, &model);
//!
//! let result = evaluate(&twin, &observed, &[empty, one]);
//! match result.best {
//!     BestExplanation::Explained { hypothesis_id, .. } => assert_eq!(hypothesis_id, "empty"),
//!     BestExplanation::Unknown { .. } => {} // also honest if indistinguishable
//! }
//! assert_eq!(result.evidence_level, ruview_ontology::EvidenceLevel::L0);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod hypothesis;
mod infer;

pub use hypothesis::{Hypothesis, HypothesisError, Occupant, MAX_HYPOTHESES, MAX_OCCUPANTS};
pub use infer::{
    evaluate, evaluate_with, score_hypothesis, synthesize_observations, BestExplanation,
    CounterfactualResult, HypothesisScore, OccupantModel, ScoringConfig, UnknownReason,
};

// Re-export the canonical ontology and twin vocabulary this crate consumes, so
// downstream speaks one semantics (ADR-300 rule 3, ADR-306).
pub use ruview_ontology::{EvidenceLevel, SemanticProvenance, SpaceId};
pub use ruview_twin::{LinkId, LinkObservation, ObservationSet, RfTwin};

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_twin::{synthetic_deployment, DeploymentDescription, RfTwin};

    fn space(seed: u64) -> SpaceId {
        SpaceId::new(format!("space-{seed}")).unwrap()
    }

    fn twin(seed: u64) -> RfTwin {
        RfTwin::build(synthetic_deployment(seed)).unwrap()
    }

    // The centre of the synthetic 5m×4m room lies on both diagonals, so a
    // centre occupant blocks the two diagonal links; edge occupants block an
    // edge link. Positions are explicit — no randomness.
    fn centre() -> Occupant {
        Occupant::new(2.5, 2.0)
    }
    fn edge() -> Occupant {
        Occupant::new(2.5, 0.5)
    }

    // Empty-room observations favour the null (empty) hypothesis over a
    // one-occupant hypothesis.
    #[test]
    fn empty_room_observations_favour_the_empty_hypothesis() {
        let t = twin(7);
        let model = OccupantModel::default_body();
        let empty = Hypothesis::empty("empty", space(7));
        let one = Hypothesis::new("one", space(7), vec![centre()]).unwrap();

        // Nobody present ⇒ observations are the twin's own predictions.
        let observed = synthesize_observations(&t, &empty, &model);

        let result = evaluate(&t, &observed, &[empty, one]);
        match &result.best {
            BestExplanation::Explained { hypothesis_id, occupant_count, margin } => {
                assert_eq!(hypothesis_id, "empty");
                assert_eq!(*occupant_count, 0);
                assert!(*margin > 0.0);
            }
            other => panic!("expected the empty hypothesis to win, got {other:?}"),
        }
        // The empty hypothesis ranks first and out-scores the occupant one.
        assert_eq!(result.ranked[0].hypothesis.id, "empty");
        assert!(result.ranked[0].log_likelihood > result.ranked[1].log_likelihood);
        // A twin verdict is always SYNTHETIC / L0.
        assert_eq!(result.evidence_level, EvidenceLevel::L0);
    }

    // A clear single-person scene favours one occupant over both zero and two.
    #[test]
    fn single_person_scene_favours_one_over_two_and_empty() {
        let t = twin(7);
        let model = OccupantModel::default_body();

        let empty = Hypothesis::empty("empty", space(7));
        let one = Hypothesis::new("one", space(7), vec![centre()]).unwrap();
        let two = Hypothesis::new("two", space(7), vec![centre(), edge()]).unwrap();

        // A scene with exactly one occupant at the room centre.
        let observed = synthesize_observations(&t, &one, &model);

        let result = evaluate(&t, &observed, &[empty.clone(), one, two]);
        match &result.best {
            BestExplanation::Explained { hypothesis_id, occupant_count, margin } => {
                assert_eq!(hypothesis_id, "one");
                assert_eq!(*occupant_count, 1);
                assert!(*margin > 0.0);
            }
            other => panic!("expected the one-occupant hypothesis to win, got {other:?}"),
        }

        // One out-scores both two and empty explicitly.
        let ll = |id: &str| {
            result
                .ranked
                .iter()
                .find(|s| s.hypothesis.id == id)
                .unwrap()
                .log_likelihood
        };
        assert!(ll("one") > ll("two"));
        assert!(ll("one") > ll("empty"));
    }

    // An occupant far outside the room blocks no link, so the one-occupant and
    // empty hypotheses are near-indistinguishable ⇒ first-class UNKNOWN.
    #[test]
    fn ambiguous_scene_returns_unknown_low_margin() {
        let t = twin(7);
        let model = OccupantModel::default_body();

        let empty = Hypothesis::empty("empty", space(7));
        // Far outside the 5m×4m room ⇒ blocks nothing.
        let ghost = Hypothesis::new("ghost", space(7), vec![Occupant::new(50.0, 50.0)]).unwrap();

        let observed = synthesize_observations(&t, &empty, &model);

        let result = evaluate(&t, &observed, &[empty, ghost]);
        match result.best {
            BestExplanation::Unknown {
                reason: UnknownReason::NearIndistinguishable { margin, threshold },
            } => {
                assert!(margin < threshold);
                assert!(margin.abs() < 1e-9, "blocking nothing ⇒ identical scores");
            }
            other => panic!("expected near-indistinguishable UNKNOWN, got {other:?}"),
        }
    }

    // Out-of-model measurements (gross deviations the twin cannot account for)
    // route to UNKNOWN rather than a forced occupancy label (ADR-313 §3).
    #[test]
    fn out_of_model_observation_routes_to_unknown() {
        let t = twin(7);
        let model = OccupantModel::default_body();
        let empty = Hypothesis::empty("empty", space(7));
        let one = Hypothesis::new("one", space(7), vec![centre()]).unwrap();

        // Take the empty-room set and shove every value 100 dB off — nothing the
        // twin or any hypothesis can explain.
        let base = synthesize_observations(&t, &empty, &model);
        let mut scattered = ObservationSet::new();
        for obs in &base.observations {
            scattered = scattered.with(obs.link.clone(), obs.value + 100.0);
        }

        let result = evaluate(&t, &scattered, &[empty, one]);
        assert!(matches!(
            result.best,
            BestExplanation::Unknown {
                reason: UnknownReason::NoHypothesisExplains { .. }
            }
        ));
    }

    // Ranking is deterministic: identical inputs (in any hypothesis order)
    // produce an identical ranked result.
    #[test]
    fn ranking_is_deterministic_and_order_independent() {
        let t = twin(7);
        let model = OccupantModel::default_body();

        let empty = Hypothesis::empty("empty", space(7));
        let one = Hypothesis::new("one", space(7), vec![centre()]).unwrap();
        let two = Hypothesis::new("two", space(7), vec![centre(), edge()]).unwrap();

        let observed = synthesize_observations(&t, &one, &model);

        let r1 = evaluate(&t, &observed, &[empty.clone(), one.clone(), two.clone()]);
        let r2 = evaluate(&t, &observed, &[empty.clone(), one.clone(), two.clone()]);
        assert_eq!(r1, r2);

        // Reordering the hypotheses does not change the ranked result or verdict.
        let r3 = evaluate(&t, &observed, &[two, empty, one]);
        assert_eq!(r1.ranked, r3.ranked);
        assert_eq!(r1.best, r3.best);
    }

    // Boundary validation: malformed input abstains, never panics.
    #[test]
    fn boundary_validation_does_not_panic() {
        let t = twin(9);
        let model = OccupantModel::default_body();

        // No hypotheses ⇒ first-class UNKNOWN, not an error.
        let none: Vec<Hypothesis> = Vec::new();
        let empty_observed = ObservationSet::new();
        let r = evaluate(&t, &empty_observed, &none);
        assert!(matches!(
            r.best,
            BestExplanation::Unknown { reason: UnknownReason::NoHypotheses }
        ));
        assert!(r.ranked.is_empty());

        // Hypotheses present but nothing evaluable ⇒ NoEvaluableLinks.
        let empty = Hypothesis::empty("empty", space(9));
        let r = evaluate(&t, &empty_observed, std::slice::from_ref(&empty));
        assert!(matches!(
            r.best,
            BestExplanation::Unknown { reason: UnknownReason::NoEvaluableLinks }
        ));

        // Non-finite occupant position is rejected at construction.
        assert!(matches!(
            Hypothesis::new("bad", space(9), vec![Occupant::new(f64::NAN, 0.0)]),
            Err(HypothesisError::NonFinitePosition { index: 0 })
        ));

        // Too many occupants is rejected at construction (bounded allocation).
        let many = vec![Occupant::new(0.0, 0.0); MAX_OCCUPANTS + 1];
        assert!(matches!(
            Hypothesis::new("big", space(9), many),
            Err(HypothesisError::TooManyOccupants { .. })
        ));

        // An observation for a link outside the twin is counted UNKNOWN, not a
        // panic. Build a set referencing a ghost node.
        let ghost_link = LinkId::new(
            ruview_ontology::SensorId::new("node-0").unwrap(),
            ruview_ontology::SensorId::new("ghost").unwrap(),
        );
        let observed = ObservationSet::new().with(ghost_link, -50.0);
        let r = evaluate(&t, &observed, std::slice::from_ref(&empty));
        assert_eq!(r.ranked[0].unknown_links, 1);
        assert_eq!(r.ranked[0].evaluated_links, 0);
        assert!(matches!(
            r.best,
            BestExplanation::Unknown { reason: UnknownReason::NoEvaluableLinks }
        ));

        // A malformed occupant injected via the struct literal (bypassing the
        // validating constructor) is treated as non-blocking, not a NaN score.
        let malformed = Hypothesis {
            id: "malformed".into(),
            space: space(9),
            occupants: vec![Occupant::new(f64::INFINITY, 0.0)],
        };
        let good_observed = synthesize_observations(&t, &empty, &model);
        let score = score_hypothesis(&t, &good_observed, &malformed, &model);
        assert!(score.log_likelihood.is_finite());
    }

    // A single hypothesis that fits well is Explained with an infinite margin
    // (no competitor), still gated by the fit floor.
    #[test]
    fn single_hypothesis_has_infinite_margin_when_it_fits() {
        let t = twin(3);
        let model = OccupantModel::default_body();
        let empty = Hypothesis::empty("empty", space(3));
        let observed = synthesize_observations(&t, &empty, &model);

        let result = evaluate(&t, &observed, std::slice::from_ref(&empty));
        match result.best {
            BestExplanation::Explained { margin, occupant_count, .. } => {
                assert!(margin.is_infinite());
                assert_eq!(occupant_count, 0);
            }
            other => panic!("expected Explained with infinite margin, got {other:?}"),
        }
    }

    // The result serde round-trips losslessly (one canonical semantics), and the
    // SYNTHETIC / L0 discipline is on the wire.
    #[test]
    fn result_serde_round_trip_is_lossless() {
        let t = twin(2);
        let model = OccupantModel::default_body();
        let empty = Hypothesis::empty("empty", space(2));
        let one = Hypothesis::new("one", space(2), vec![centre()]).unwrap();
        let observed = synthesize_observations(&t, &one, &model);

        let result = evaluate(&t, &observed, &[empty, one]);
        let json = serde_json::to_string_pretty(&result).unwrap();
        let back: CounterfactualResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
        assert!(json.contains("\"L0\""));
    }

    // Distinct twin seeds give distinct-but-reproducible synthetic scenes; the
    // scene variation is driven only by the explicit seed (no wall clock).
    #[test]
    fn scenes_vary_only_by_explicit_seed() {
        let model = OccupantModel::default_body();
        let a: DeploymentDescription = synthetic_deployment(1);
        let b: DeploymentDescription = synthetic_deployment(1);
        assert_eq!(a, b);

        let ta = RfTwin::build(a).unwrap();
        let tb = twin(1);
        let one_a = Hypothesis::new("one", space(1), vec![centre()]).unwrap();
        let one_b = Hypothesis::new("one", space(1), vec![centre()]).unwrap();
        let obs_a = synthesize_observations(&ta, &one_a, &model);
        let obs_b = synthesize_observations(&tb, &one_b, &model);
        assert_eq!(obs_a, obs_b);
    }
}
