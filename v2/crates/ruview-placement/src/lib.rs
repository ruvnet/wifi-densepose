//! # `ruview-placement` — sensor placement optimizer (ADR-308, ADR-300 phase 3)
//!
//! **SYNTHETIC / L0 — a planning scaffold, not a measurement system.**
//!
//! This crate is a *research-forward primitive*: given a coarse floor plan
//! ([`FloorPlan`], an ADR-306 scene abstraction) and a hardware [`Inventory`], it
//! recommends radio-node positions by scoring candidate placements against the
//! ADR-315 RF twin's **SYNTHETIC** propagation model. Every coverage and
//! observability value it produces is a *simulation* at evidence level `L0`
//! (ADR-282), labelled `SYNTHETIC`: a **recommendation**, never a sensing claim.
//! Nothing here is a hardware, `MEASURED`, or accuracy claim, and the crate
//! asserts **no** coverage or accuracy number — a twin/optimizer *predicts*, it
//! does not *measure* (ADR-308 evidence discipline).
//!
//! Consistent with ADR-300 rule 1, *insufficient information* is a first-class
//! value ([`Observability::Unknown`]), never an error and never a confident
//! default. Consistent with rule 3, the crate reuses the canonical ontology
//! vocabulary ([`SpaceId`], [`ZoneId`], [`SensorId`], [`Container`],
//! [`EvidenceLevel`], [`SemanticProvenance`]) rather than inventing its own.
//!
//! ## What the optimizer does
//!
//! - **Predict** ([`score_placement`]): for each objective ([`Objective`]) it
//!   samples the target region and scores modelled observability from
//!   Fresnel-zone clearance ([`crate::fresnel`]) over well-predicted twin links,
//!   reporting both a score **and** its uncertainty per target, and flagging
//!   blind spots.
//! - **Search** ([`optimize`]): greedy forward selection over a **seeded**
//!   candidate grid recommends a [`PlacementPlan`]; adding a radio can only
//!   maintain or raise the modelled score, so the plan's score trace is
//!   monotonically non-decreasing and plateaus at saturation.
//! - **Rank** ([`rank_placements`]): score and order supplied candidate
//!   placements, highest modelled observability first.
//! - **Post-install compare** ([`compare_post_install`]): compare the optimizer's
//!   `L0` prediction against **caller-supplied** measured observability (this
//!   crate never measures) and recommend adjustments plus a coarse twin-parameter
//!   residual to feed back into ADR-315.
//!
//! ## Determinism
//!
//! Everything is deterministic. Synthetic scenes are varied by an explicit
//! [`seed`](PlacementParams::seed); there is no wall-clock, no unseeded
//! randomness, and no I/O anywhere in the crate. Allocation is bounded at every
//! boundary ([`MAX_PLAN_WALLS`], [`MAX_ZONES`], [`MAX_INVENTORY`],
//! [`PlacementParams::max_sample_points`], [`PlacementParams::max_candidates`]),
//! and malformed input yields a typed [`PlacementError`] or a first-class
//! `Unknown`, never a panic.
//!
//! ```
//! use ruview_placement::*;
//!
//! // A reproducible SYNTHETIC scene, an inventory, and one objective.
//! let plan = synthetic_floorplan(7);
//! let inventory = Inventory::homogeneous("esp32-s3", 20.0, 4);
//! let objectives = vec![synthetic_objective(&plan)];
//! let params = PlacementParams::default_synthetic();
//!
//! plan.validate().unwrap();
//! inventory.validate().unwrap();
//!
//! let recommended = optimize(&plan, &inventory, &objectives, &params);
//! assert_eq!(recommended.evidence_level, EvidenceLevel::L0); // SYNTHETIC
//! // The greedy score trace never decreases.
//! for w in recommended.score_trace.windows(2) {
//!     assert!(w[1] + 1e-9 >= w[0]);
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod compare;
mod coverage;
mod fresnel;
mod geometry;
mod inventory;
mod plan;

pub use compare::{
    compare_post_install, compare_post_install_with_tolerance, Adjustment, AdjustmentAction,
    AdjustmentReport, CompareVerdict, MeasuredObservability, MeasuredTarget, ResidualKind,
    TargetComparison, TwinResidual, DEFAULT_COMPARE_TOLERANCE,
};
pub use coverage::{
    rank_placements, score_placement, Objective, Observability, ObservabilityUnknown, Phenomenon,
    PlacedRadio, Placement, PlacementParams, PlacementScore, RankedPlacement, TargetCoverage,
};
pub use fresnel::{fresnel_radius, link_clearance};
pub use geometry::{
    sample_points, FloorPlan, PlacementError, Rect, ZoneGeometry, MAX_PLAN_WALLS, MAX_ZONES,
};
pub use inventory::{Inventory, RadioSpec, MAX_INVENTORY};
pub use plan::{candidate_positions, optimize, PlacementPlan};

// Re-export the canonical ontology and twin vocabulary consumers need, so they
// speak one semantics (ADR-300 rule 3).
pub use ruview_ontology::{
    Container, EvidenceLevel, SemanticProvenance, SensorId, SpaceId, ZoneId,
};
pub use ruview_twin::{Point3, PropagationParams, Wall};

/// Build a deterministic **SYNTHETIC** floor plan from an explicit `seed`.
///
/// A `5 m × 4 m` space with one interior wall and two zones (a central "core"
/// zone straddling the room and a "corner" zone in the far top-right). Zone
/// footprints are jittered by a seeded `splitmix64` stream so distinct seeds give
/// distinct-but-reproducible scenes; the same seed always yields the same scene.
/// This is a simulation fixture, not a model of any real room.
#[must_use]
pub fn synthetic_floorplan(seed: u64) -> FloorPlan {
    let mut state = seed;
    // Deterministic jitter helper in [-0.25, 0.25] metres.
    let mut jitter = || (splitmix64_unit(&mut state) - 0.5) * 0.5;

    let jx = jitter();
    let jy = jitter();

    let space = SpaceId::new(format!("space-{seed}")).expect("static id is valid");
    let bounds = Rect::new(0.0, 0.0, 5.0, 4.0);

    let walls = vec![Wall {
        id: "interior-wall".to_string(),
        a: (2.5, 3.0),
        b: (2.5, 4.0),
        attenuation_db: 6.0,
    }];

    let core = ZoneGeometry {
        id: ZoneId::new(format!("core-{seed}")).expect("static id is valid"),
        // A band across the middle of the room where links crisscross.
        bounds: Rect::new(
            (1.5 + jx).clamp(0.5, 2.0),
            (1.5 + jy).clamp(0.5, 2.0),
            3.5,
            2.5,
        ),
    };
    let corner = ZoneGeometry {
        id: ZoneId::new(format!("corner-{seed}")).expect("static id is valid"),
        // A far top-right pocket, easy to leave as a blind spot.
        bounds: Rect::new(4.0, 3.2, 4.9, 3.9),
    };

    FloorPlan {
        space,
        bounds,
        walls,
        zones: vec![core, corner],
    }
}

/// A default presence objective on the synthetic plan's central "core" zone.
#[must_use]
pub fn synthetic_objective(plan: &FloorPlan) -> Objective {
    let target = plan
        .zones
        .first()
        .map(|z| Container::Zone { id: z.id.clone() })
        .unwrap_or(Container::Space { id: plan.space.clone() });
    Objective::new(target, Phenomenon::Presence)
}

/// One `splitmix64` step mapped to a unit `f64` in `[0, 1)`. Deterministic; the
/// only source of scene variation in the fixtures (varied by explicit seed).
fn splitmix64_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PlacementParams {
        PlacementParams::default_synthetic()
    }

    /// Two radios placed on the bottom wall of a 4×4 room; links run along `y≈0`.
    fn bottom_pair_plan() -> FloorPlan {
        FloorPlan {
            space: SpaceId::new("room").unwrap(),
            bounds: Rect::new(0.0, 0.0, 4.0, 4.0),
            walls: Vec::new(),
            zones: vec![
                // A zone straddling the link line — should be covered.
                ZoneGeometry {
                    id: ZoneId::new("on-line").unwrap(),
                    bounds: Rect::new(1.0, 0.0, 3.0, 0.3),
                },
                // A far top zone away from the link line — a blind spot.
                ZoneGeometry {
                    id: ZoneId::new("far-top").unwrap(),
                    bounds: Rect::new(1.0, 3.0, 3.0, 4.0),
                },
            ],
        }
    }

    fn bottom_pair_placement() -> Placement {
        Placement {
            radios: vec![
                PlacedRadio::new(Point3::new(0.2, 0.1, 1.0), 20.0),
                PlacedRadio::new(Point3::new(3.8, 0.1, 1.0), 20.0),
            ],
        }
    }

    #[test]
    fn better_covering_placement_ranks_higher() {
        let plan = bottom_pair_plan();
        let objectives = vec![Objective::new(
            Container::Zone { id: ZoneId::new("on-line").unwrap() },
            Phenomenon::Presence,
        )];
        let p = params();

        // Good: radios straddle the zone so the link's Fresnel zone crosses it.
        let good = bottom_pair_placement();
        // Bad: both radios clustered in the far corner, link far from the zone.
        let bad = Placement {
            radios: vec![
                PlacedRadio::new(Point3::new(0.1, 3.8, 1.0), 20.0),
                PlacedRadio::new(Point3::new(0.4, 3.9, 1.0), 20.0),
            ],
        };

        let ranked = rank_placements(&plan, &[bad.clone(), good.clone()], &objectives, &p);
        // The good placement (input index 1) ranks first.
        assert_eq!(ranked[0].index, 1);
        assert!(ranked[0].score.total_score > ranked[1].score.total_score);

        // And its observability is genuinely higher.
        let sg = score_placement(&plan, &good, &objectives, &p);
        let sb = score_placement(&plan, &bad, &objectives, &p);
        assert!(sg.total_score > sb.total_score);
    }

    #[test]
    fn blind_spot_zone_is_flagged() {
        let plan = bottom_pair_plan();
        let placement = bottom_pair_placement();
        let objectives = vec![
            Objective::new(
                Container::Zone { id: ZoneId::new("on-line").unwrap() },
                Phenomenon::Presence,
            ),
            Objective::new(
                Container::Zone { id: ZoneId::new("far-top").unwrap() },
                Phenomenon::Presence,
            ),
        ];
        let score = score_placement(&plan, &placement, &objectives, &params());

        assert!(score.has_blind_spot());
        let far = Container::Zone { id: ZoneId::new("far-top").unwrap() };
        assert!(score.blind_spots.contains(&far));

        // The far-top zone is a blind spot; the on-line zone is not.
        let on_line = score
            .per_target
            .iter()
            .find(|t| t.target == Container::Zone { id: ZoneId::new("on-line").unwrap() })
            .unwrap();
        let far_top = score
            .per_target
            .iter()
            .find(|t| t.target == far)
            .unwrap();
        assert!(!on_line.blind_spot);
        assert!(far_top.blind_spot);
        assert!(far_top.covered_fraction < on_line.covered_fraction);
    }

    #[test]
    fn adding_a_node_improves_score_monotonically_until_saturation() {
        let plan = bottom_pair_plan();
        let objectives = vec![Objective::new(
            Container::Zone { id: ZoneId::new("on-line").unwrap() },
            Phenomenon::Presence,
        )];
        let p = params();

        // Incrementally add radios along the bottom wall.
        let positions = [
            Point3::new(0.2, 0.1, 1.0),
            Point3::new(3.8, 0.1, 1.0),
            Point3::new(2.0, 0.1, 1.0),
            Point3::new(1.0, 0.1, 1.0),
            Point3::new(3.0, 0.1, 1.0),
        ];
        let mut radios = Vec::new();
        let mut prev = -1.0_f64;
        let mut scores = Vec::new();
        for pos in positions {
            radios.push(PlacedRadio::new(pos, 20.0));
            let s = score_placement(&plan, &Placement { radios: radios.clone() }, &objectives, &p)
                .total_score;
            // Monotone non-decreasing at every step.
            assert!(s + 1e-9 >= prev, "score decreased: {prev} -> {s}");
            prev = s;
            scores.push(s);
        }

        // It strictly improved at least once early on...
        assert!(scores[1] > scores[0]);
        // ...and saturates: a later step adds (near-)nothing.
        let last = scores.len() - 1;
        assert!((scores[last] - scores[last - 1]).abs() < 1e-6);

        // The greedy optimizer's own trace is also non-decreasing.
        let inv = Inventory::homogeneous("esp32-s3", 20.0, 5);
        let recommended = optimize(&plan, &inv, &objectives, &p);
        for w in recommended.score_trace.windows(2) {
            assert!(w[1] + 1e-9 >= w[0]);
        }
    }

    #[test]
    fn optimize_reports_uncertainty_and_never_a_bare_number() {
        let plan = synthetic_floorplan(3);
        let inv = Inventory::homogeneous("esp32-s3", 20.0, 4);
        let objectives = vec![synthetic_objective(&plan)];
        let recommended = optimize(&plan, &inv, &objectives, &params());

        // Every known target carries BOTH a score and an uncertainty (ADR-308 §2).
        let mut saw_known = false;
        for t in &recommended.score.per_target {
            if let Observability::Known { score, uncertainty } = t.observability {
                saw_known = true;
                assert!((0.0..=1.0).contains(&score));
                assert!((0.0..=1.0).contains(&uncertainty));
            }
        }
        assert!(saw_known);
        assert_eq!(recommended.evidence_level, EvidenceLevel::L0);
    }

    #[test]
    fn predicted_vs_observed_delta_yields_adjustment_suggestion() {
        let plan = bottom_pair_plan();
        let placement = bottom_pair_placement();
        let objectives = vec![Objective::new(
            Container::Zone { id: ZoneId::new("on-line").unwrap() },
            Phenomenon::Presence,
        )];
        let predicted = score_placement(&plan, &placement, &objectives, &params());

        let target = Container::Zone { id: ZoneId::new("on-line").unwrap() };
        let (pred_score, _) = predicted
            .per_target
            .iter()
            .find(|t| t.target == target)
            .unwrap()
            .observability
            .known()
            .expect("predicted score known");

        // Caller supplies a much *lower* measured observability than predicted.
        let measured = MeasuredObservability::new().with(target.clone(), (pred_score - 0.6).max(0.0));
        let report = compare_post_install(&predicted, &measured);

        assert_eq!(report.predicted_evidence_level, EvidenceLevel::L0);
        assert!(report.needs_adjustment());
        let adj = report.adjustments.iter().find(|a| a.target == target).unwrap();
        assert!(matches!(
            adj.action,
            AdjustmentAction::AddNode | AdjustmentAction::MoveNode | AdjustmentAction::ReAim
        ));
        // The residual points the twin at higher effective attenuation.
        let residual = adj.twin_residual.expect("residual suggested");
        assert_eq!(residual.kind, ResidualKind::EffectiveAttenuationHigher);
        assert!(residual.magnitude_db > 0.0);

        let cmp = report.comparisons.iter().find(|c| c.target == target).unwrap();
        assert_eq!(cmp.verdict, CompareVerdict::Underperforming);

        // A missing measurement is first-class UNKNOWN, not an error.
        let empty = MeasuredObservability::new();
        let report2 = compare_post_install(&predicted, &empty);
        let cmp2 = report2.comparisons.iter().find(|c| c.target == target).unwrap();
        assert_eq!(cmp2.verdict, CompareVerdict::Unknown);
        assert!(!report2.needs_adjustment());
    }

    #[test]
    fn matching_observation_needs_no_adjustment() {
        let plan = bottom_pair_plan();
        let placement = bottom_pair_placement();
        let objectives = vec![Objective::new(
            Container::Zone { id: ZoneId::new("on-line").unwrap() },
            Phenomenon::Presence,
        )];
        let predicted = score_placement(&plan, &placement, &objectives, &params());
        let target = Container::Zone { id: ZoneId::new("on-line").unwrap() };
        let (pred_score, _) = predicted.per_target[0].observability.known().unwrap();

        // Measured equals predicted: verdict Match, no corrective action.
        let measured = MeasuredObservability::new().with(target.clone(), pred_score);
        let report = compare_post_install(&predicted, &measured);
        let cmp = report.comparisons.iter().find(|c| c.target == target).unwrap();
        assert_eq!(cmp.verdict, CompareVerdict::Match);
        assert!(!report.needs_adjustment());
    }

    #[test]
    fn optimize_is_deterministic_and_seed_varies_the_scene() {
        let inv = Inventory::homogeneous("esp32-s3", 20.0, 4);
        let p = params();

        // Same seed ⇒ identical plan (bit-for-bit via serde).
        let plan_a = synthetic_floorplan(11);
        let objectives_a = vec![synthetic_objective(&plan_a)];
        let r1 = optimize(&plan_a, &inv, &objectives_a, &p);
        let r2 = optimize(&plan_a, &inv, &objectives_a, &p);
        assert_eq!(r1, r2);
        assert_eq!(
            serde_json::to_string(&r1).unwrap(),
            serde_json::to_string(&r2).unwrap()
        );

        // Distinct seeds give distinct-but-reproducible scenes.
        let plan_b = synthetic_floorplan(12);
        assert_ne!(plan_a, plan_b);

        // Candidate generation is seeded and deterministic.
        let c1 = candidate_positions(&plan_a, &p);
        let c2 = candidate_positions(&plan_a, &p);
        assert_eq!(c1, c2);
        let mut p_seeded = p;
        p_seeded.seed = p.seed.wrapping_add(1);
        let c3 = candidate_positions(&plan_a, &p_seeded);
        assert_ne!(c1, c3); // a different seed shifts the grid
    }

    #[test]
    fn serde_round_trip_is_lossless_and_labels_evidence() {
        let plan = synthetic_floorplan(1);
        let inv = Inventory::homogeneous("esp32-s3", 20.0, 3);
        let objectives = vec![synthetic_objective(&plan)];
        let recommended = optimize(&plan, &inv, &objectives, &params());

        let json = serde_json::to_string_pretty(&recommended).unwrap();
        let back: PlacementPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(recommended, back);
        // Evidence discipline is on the wire: L0 / SYNTHETIC.
        assert!(json.contains("\"evidence_level\": \"L0\""));
    }

    #[test]
    fn boundary_validation_rejects_malformed_input_without_panic() {
        // Inverted rectangle.
        let mut plan = synthetic_floorplan(1);
        plan.bounds = Rect::new(5.0, 4.0, 0.0, 0.0);
        assert!(matches!(plan.validate(), Err(PlacementError::InvalidRect { .. })));

        // Non-finite wall coordinate.
        let mut plan = synthetic_floorplan(1);
        plan.walls[0].a.0 = f64::NAN;
        assert!(matches!(plan.validate(), Err(PlacementError::NonFinite { .. })));

        // Too many radios.
        let over = Inventory::homogeneous("x", 20.0, MAX_INVENTORY + 1);
        assert!(matches!(over.validate(), Err(PlacementError::TooManyRadios { .. })));

        // Non-finite tx power.
        let bad_inv = Inventory { radios: vec![RadioSpec::new("x", f64::INFINITY)] };
        assert!(matches!(bad_inv.validate(), Err(PlacementError::NonFinite { .. })));

        // Invalid parameter.
        let mut bad_params = params();
        bad_params.wavelength_m = 0.0;
        assert!(matches!(bad_params.validate(), Err(PlacementError::InvalidParameter { .. })));

        // Objective targeting a container not in the plan ⇒ first-class UNKNOWN,
        // never a panic or error.
        let plan = synthetic_floorplan(1);
        let placement = Placement {
            radios: vec![
                PlacedRadio::new(Point3::new(0.5, 0.5, 1.0), 20.0),
                PlacedRadio::new(Point3::new(4.5, 3.5, 1.0), 20.0),
            ],
        };
        let ghost = Objective::new(
            Container::Zone { id: ZoneId::new("ghost-zone").unwrap() },
            Phenomenon::Presence,
        );
        let score = score_placement(&plan, &placement, &[ghost], &params());
        assert!(matches!(
            score.per_target[0].observability,
            Observability::Unknown { reason: ObservabilityUnknown::TargetNotInPlan }
        ));

        // A non-finite placement position is skipped, never panics: with fewer
        // than two finite nodes there are no links, so the target is UNKNOWN.
        let nan_placement = Placement {
            radios: vec![PlacedRadio::new(Point3::new(f64::NAN, 0.0, 1.0), 20.0)],
        };
        let objectives = vec![synthetic_objective(&plan)];
        let score = score_placement(&plan, &nan_placement, &objectives, &params());
        assert!(matches!(
            score.per_target[0].observability,
            Observability::Unknown { reason: ObservabilityUnknown::NoLinks }
        ));
    }

    #[test]
    fn marginal_case_reports_higher_uncertainty() {
        // A zone squarely on the link line vs. a marginal one at the Fresnel edge:
        // the marginal case is reported with higher uncertainty (ADR-308 §2, the
        // model reports uncertainty rather than overstating a coarse result).
        let plan = FloorPlan {
            space: SpaceId::new("room").unwrap(),
            bounds: Rect::new(0.0, 0.0, 4.0, 4.0),
            walls: Vec::new(),
            zones: vec![
                ZoneGeometry {
                    id: ZoneId::new("on-line").unwrap(),
                    bounds: Rect::new(1.0, 0.0, 3.0, 0.2),
                },
                ZoneGeometry {
                    id: ZoneId::new("marginal").unwrap(),
                    bounds: Rect::new(1.0, 1.5, 3.0, 1.9),
                },
            ],
        };
        let placement = bottom_pair_placement();
        let p = params();
        let on_line = score_placement(
            &plan,
            &placement,
            &[Objective::new(
                Container::Zone { id: ZoneId::new("on-line").unwrap() },
                Phenomenon::Presence,
            )],
            &p,
        );
        let marginal = score_placement(
            &plan,
            &placement,
            &[Objective::new(
                Container::Zone { id: ZoneId::new("marginal").unwrap() },
                Phenomenon::Presence,
            )],
            &p,
        );
        let (_, u_on) = on_line.per_target[0].observability.known().unwrap();
        let (_, u_marg) = marginal.per_target[0].observability.known().unwrap();
        assert!(u_marg > u_on, "marginal uncertainty {u_marg} should exceed on-line {u_on}");
    }
}
