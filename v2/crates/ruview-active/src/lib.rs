//! # `ruview-active` — closed-loop RF experiment control (ADR-306, ADR-297 primitive 9)
//!
//! **SYNTHETIC / L0 research-forward model scaffold (ADR-282, ADR-297 phase 3).**
//! This crate turns sensing from *RF-happens → observe* into
//! *RuView-controls-RF → observe the response → optimize the next measurement*.
//! It models the **control loop** ADR-280 deferred (ADR-280 built the governed
//! actuation surface but left information-gain-driven closed-loop control as a
//! roadmap item): read the current per-zone uncertainty and the last response,
//! and propose the controllable measurement configuration expected to reduce
//! that uncertainty most.
//!
//! It is a **simulation / planning model**, not a driver. Constructing a
//! [`ControlAction`] or a [`MeasurementPlan`] drives **no** radio, changes no
//! pairing state, and emits **no** RF — the loop *emits a plan*, and a fielded
//! caller submits every proposal through the ADR-280 fail-closed
//! admission/actuation surface. A twin predicts; it does not measure: no
//! `MEASURED`, accuracy, or traffic-reduction claim is made or implied. Every
//! exploration figure this crate produces is `SYNTHETIC` (CLAUDE.md honesty
//! discipline; ADR-306 §3).
//!
//! ## The four ADR-297 non-negotiable rules, as they bind this crate
//!
//! 1. **UNKNOWN is first-class, never an error.** A [`LastResponse::Unknown`]
//!    is not zero uncertainty and not an error: the policy *widens* exploration
//!    ([`ControllerConfig::unknown_widen`]) rather than committing to a narrow
//!    configuration on a target it cannot currently resolve. A non-finite
//!    uncertainty becomes maximal uncertainty, never a silent zero.
//!    [`ClosedLoopController::step`] is total — no input panics.
//! 2. **Certificates bind cryptographically.** Out of scope here; a proposal
//!    names an already-authenticated ADR-303
//!    [`ZoneId`](ruview_ontology::ZoneId), and a fielded loop step is admitted
//!    through the ADR-280 governed path before any actuation.
//! 3. **One canonical semantics downstream.** The loop reuses the canonical
//!    [`ZoneId`](ruview_ontology::ZoneId),
//!    [`EvidenceLevel`](ruview_ontology::EvidenceLevel), and
//!    [`SemanticProvenance`](ruview_ontology::SemanticProvenance) rather than
//!    reinventing per-crate identity/evidence shapes. It shares the *notion* of
//!    expected gain with ADR-311 but defines its own [`ControlAction`]
//!    vocabulary and does **not** depend on `ruview-infogain`, so the two crates
//!    build in parallel.
//! 4. **Honest evidence.** Every proposal is stamped [`EvidenceLevel::L1`]
//!    (heuristic/synthetic) with an explicit synthetic provenance; the
//!    exploration scalar is a modelled magnitude, not a measured gain. When no
//!    axis is controllable the controller returns [`PassiveReason::NoControllableAxes`]
//!    rather than fabricating a gain estimate.
//!
//! ## The loop, in one call
//!
//! ```
//! use ruview_active::*;
//! use ruview_ontology::ZoneId;
//!
//! // A deployment that can vary channel width and antenna aperture.
//! let cap = ControlCapability::new(
//!     vec![Channel::new(Band::Ghz5, 36).unwrap()],
//!     vec![Bandwidth::Bw20, Bandwidth::Bw160],
//!     vec![],
//!     vec![
//!         AntennaSelection::new([0], 4).unwrap(),
//!         AntennaSelection::new([0, 1, 2, 3], 4).unwrap(),
//!     ],
//! )
//! .unwrap();
//! let ctrl = ClosedLoopController::new(cap, ControllerConfig::default());
//!
//! // A poorly-known zone drives an exploratory (widest) measurement.
//! let uncertain = ZoneBelief::new(
//!     ZoneId::new("kitchen").unwrap(),
//!     Uncertainty::new(0.95),
//!     LastResponse::None,
//! );
//! let decision = ctrl.step(&uncertain);
//! let p = decision.proposal().unwrap();
//! assert_eq!(p.intent, ControlIntent::Explore);
//! assert_eq!(p.action.bandwidth, Some(Bandwidth::Bw160)); // widest
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod control;
mod policy;

pub use control::{
    AntennaSelection, Band, Bandwidth, Cadence, Channel, ControlAction, ControlCapability,
    ControlError, MAX_AXIS_VALUES, MAX_CADENCE_MS, MAX_CHAINS, MIN_CADENCE_MS,
};
pub use policy::{
    ClosedLoopController, ControlDecision, ControlIntent, ControlProposal, ControllerConfig,
    LastResponse, MeasurementPlan, PassiveReason, Uncertainty, ZoneBelief, MODEL_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_ontology::{EvidenceLevel, ZoneId};

    fn zid(s: &str) -> ZoneId {
        ZoneId::new(s).unwrap()
    }

    /// A deployment controlling channel (sweep set), bandwidth, cadence, and
    /// antenna aperture — a full controllable surface for the loop tests.
    fn full_capability() -> ControlCapability {
        ControlCapability::new(
            vec![
                Channel::new(Band::Ghz5, 36).unwrap(),
                Channel::new(Band::Ghz5, 40).unwrap(),
                Channel::new(Band::Ghz5, 44).unwrap(),
            ],
            vec![Bandwidth::Bw20, Bandwidth::Bw80, Bandwidth::Bw160],
            vec![
                Cadence::from_interval_ms(1000).unwrap(), // slow
                Cadence::from_interval_ms(100).unwrap(),  // fast
            ],
            vec![
                AntennaSelection::new([0], 4).unwrap(),
                AntennaSelection::new([0, 1], 4).unwrap(),
                AntennaSelection::new([0, 1, 2, 3], 4).unwrap(),
            ],
        )
        .unwrap()
    }

    fn controller() -> ClosedLoopController {
        ClosedLoopController::new(full_capability(), ControllerConfig::default())
    }

    // ADR-306 §2: a high-uncertainty zone drives an exploratory control action
    // — widest bandwidth, fastest cadence, widest aperture, Explore intent.
    #[test]
    fn high_uncertainty_drives_exploratory_action() {
        let belief = ZoneBelief::new(zid("kitchen"), Uncertainty::new(1.0), LastResponse::None);
        let p = controller().step(&belief).proposal().cloned().unwrap();

        assert_eq!(p.intent, ControlIntent::Explore);
        assert_eq!(p.action.bandwidth, Some(Bandwidth::Bw160)); // widest available
        assert_eq!(p.action.cadence.unwrap().interval_ms(), 100); // fastest
        assert_eq!(p.action.antenna.as_ref().unwrap().chain_count(), 4); // widest aperture
        assert_eq!(p.evidence_level, EvidenceLevel::L1); // honest synthetic label
        assert_eq!(p.provenance.model_version, MODEL_VERSION);
    }

    // ADR-306 validation: convergence (falling uncertainty) reduces exploration
    // — the proposal narrows and the intent flips to Exploit.
    #[test]
    fn convergence_reduces_exploration() {
        let ctrl = controller();
        let high = ZoneBelief::new(zid("z"), Uncertainty::new(0.95), LastResponse::None);
        let low = ZoneBelief::new(zid("z"), Uncertainty::new(0.05), LastResponse::None);

        let ph = ctrl.step(&high).proposal().cloned().unwrap();
        let pl = ctrl.step(&low).proposal().cloned().unwrap();

        // Exploration strictly falls as the zone converges.
        assert!(pl.exploration < ph.exploration);
        assert_eq!(ph.intent, ControlIntent::Explore);
        assert_eq!(pl.intent, ControlIntent::Exploit);

        // The converged proposal is narrower/slower on every graded axis.
        assert!(pl.action.bandwidth.unwrap().mhz() < ph.action.bandwidth.unwrap().mhz());
        assert!(
            pl.action.cadence.unwrap().interval_ms() > ph.action.cadence.unwrap().interval_ms()
        );
        assert!(
            pl.action.antenna.as_ref().unwrap().chain_count()
                < ph.action.antenna.as_ref().unwrap().chain_count()
        );
        assert_eq!(pl.action.bandwidth, Some(Bandwidth::Bw20)); // narrowest
    }

    // ADR-297 rule 1: an UNKNOWN last response widens exploration relative to
    // the same uncertainty with an observed response.
    #[test]
    fn unknown_last_response_widens_exploration() {
        let ctrl = controller();
        let u = Uncertainty::new(0.4); // below the 0.5 explore threshold on its own

        let observed = ZoneBelief::new(
            zid("z"),
            u,
            LastResponse::Observed {
                evidence_level: EvidenceLevel::L2,
                residual: Uncertainty::new(0.4),
            },
        );
        let unknown = ZoneBelief::new(zid("z"), u, LastResponse::Unknown);

        let po = ctrl.step(&observed).proposal().cloned().unwrap();
        let pu = ctrl.step(&unknown).proposal().cloned().unwrap();

        // UNKNOWN pushes exploration strictly higher...
        assert!(pu.exploration > po.exploration);
        // ...enough to cross from Exploit into Explore (0.4 + 0.3 = 0.7 >= 0.5).
        assert_eq!(po.intent, ControlIntent::Exploit);
        assert_eq!(pu.intent, ControlIntent::Explore);
    }

    // ADR-306 §2 degradation: an empty controllable set (ESP32-only) falls back
    // to the passive planner with no error and no fabricated gain.
    #[test]
    fn empty_capability_degrades_to_passive() {
        let ctrl = ClosedLoopController::new(ControlCapability::none(), ControllerConfig::default());
        let belief = ZoneBelief::new(zid("z"), Uncertainty::new(1.0), LastResponse::None);
        match ctrl.step(&belief) {
            ControlDecision::Passive { zone, reason } => {
                assert_eq!(zone, zid("z"));
                assert_eq!(reason, PassiveReason::NoControllableAxes);
            }
            other => panic!("expected passive fallback, got {other:?}"),
        }
    }

    // A cadence-only deployment (ESP32 that can vary sounding rate) still closes
    // the loop on its one controllable axis; the others stay uncontrolled.
    #[test]
    fn cadence_only_capability_controls_only_cadence() {
        let cap = ControlCapability::new(
            vec![],
            vec![],
            vec![
                Cadence::from_interval_ms(2000).unwrap(),
                Cadence::from_interval_ms(50).unwrap(),
            ],
            vec![],
        )
        .unwrap();
        let ctrl = ClosedLoopController::new(cap, ControllerConfig::default());
        let belief = ZoneBelief::new(zid("z"), Uncertainty::new(1.0), LastResponse::None);
        let p = ctrl.step(&belief).proposal().cloned().unwrap();

        assert!(p.action.channel.is_none());
        assert!(p.action.bandwidth.is_none());
        assert!(p.action.antenna.is_none());
        assert_eq!(p.action.cadence.unwrap().interval_ms(), 50); // fastest, exploring
        assert!(!p.action.is_noop());
    }

    // Control ranges are validated: an invalid channel/bandwidth/cadence/antenna
    // is rejected at the boundary and can never enter an action.
    #[test]
    fn invalid_control_values_are_rejected() {
        // 2.4 GHz has no channel 15.
        assert!(matches!(
            Channel::new(Band::Ghz24, 15),
            Err(ControlError::InvalidChannel { .. })
        ));
        // 5 GHz channel 37 is not a standard channel.
        assert!(matches!(
            Channel::new(Band::Ghz5, 37),
            Err(ControlError::InvalidChannel { .. })
        ));
        // A valid 5 GHz channel is accepted.
        assert!(Channel::new(Band::Ghz5, 36).is_ok());

        // 33 MHz is not a recognised channel width.
        assert!(matches!(
            Bandwidth::from_mhz(33),
            Err(ControlError::InvalidBandwidth { mhz: 33 })
        ));
        assert_eq!(Bandwidth::from_mhz(80).unwrap(), Bandwidth::Bw80);

        // Cadence outside the modelled range is rejected on both ends.
        assert!(matches!(
            Cadence::from_interval_ms(0),
            Err(ControlError::InvalidCadence { .. })
        ));
        assert!(matches!(
            Cadence::from_interval_ms(MAX_CADENCE_MS + 1),
            Err(ControlError::InvalidCadence { .. })
        ));

        // Antenna selection: empty and out-of-range indices are rejected.
        assert!(matches!(
            AntennaSelection::new(Vec::<u8>::new(), 4),
            Err(ControlError::EmptyAntennaSelection)
        ));
        assert!(matches!(
            AntennaSelection::new([4], 4),
            Err(ControlError::AntennaChainOutOfRange { index: 4, .. })
        ));
        // A zero-chain aperture is rejected.
        assert!(matches!(
            AntennaSelection::new([0], 0),
            Err(ControlError::AntennaChainOutOfRange { .. })
        ));
    }

    // The capability set bounds allocation: an axis longer than MAX_AXIS_VALUES
    // is rejected rather than accepted unbounded.
    #[test]
    fn oversized_capability_axis_is_rejected() {
        let cadences: Vec<Cadence> = (1..=(MAX_AXIS_VALUES as u32 + 1))
            .map(|ms| Cadence::from_interval_ms(ms).unwrap())
            .collect();
        assert!(matches!(
            ControlCapability::new(vec![], vec![], cadences, vec![]),
            Err(ControlError::AxisTooLarge { .. })
        ));
    }

    // Non-finite uncertainty is treated as maximal uncertainty, never a silent
    // zero (ADR-297 rule 1), and never panics.
    #[test]
    fn non_finite_uncertainty_is_maximal_not_zero() {
        assert_eq!(Uncertainty::new(f64::NAN).value(), 1.0);
        assert_eq!(Uncertainty::new(f64::INFINITY).value(), 1.0);
        assert_eq!(Uncertainty::new(-5.0).value(), 0.0);
        assert_eq!(Uncertainty::new(2.0).value(), 1.0);

        let belief = ZoneBelief::new(zid("z"), Uncertainty::new(f64::NAN), LastResponse::None);
        let p = controller().step(&belief).proposal().cloned().unwrap();
        assert_eq!(p.intent, ControlIntent::Explore); // maximal → explore
    }

    // Channel sweep: exploring across cycles rotates deterministically through
    // the controllable channels; exploiting anchors to the first channel.
    #[test]
    fn channel_sweeps_when_exploring_and_anchors_when_exploiting() {
        let ctrl = controller();
        let mk = |cycle: u64| {
            ZoneBelief::new(zid("z"), Uncertainty::new(1.0), LastResponse::None).with_cycle(cycle)
        };
        let c0 = ctrl.step(&mk(0)).proposal().unwrap().action.channel.unwrap();
        let c1 = ctrl.step(&mk(1)).proposal().unwrap().action.channel.unwrap();
        let c2 = ctrl.step(&mk(2)).proposal().unwrap().action.channel.unwrap();
        let c3 = ctrl.step(&mk(3)).proposal().unwrap().action.channel.unwrap();
        assert_eq!(c0.number(), 36);
        assert_eq!(c1.number(), 40);
        assert_eq!(c2.number(), 44);
        assert_eq!(c3.number(), 36); // wraps deterministically

        // Exploiting (low uncertainty) anchors to the first channel regardless
        // of cycle.
        let exploit = ZoneBelief::new(zid("z"), Uncertainty::new(0.0), LastResponse::None)
            .with_cycle(2);
        let ce = ctrl.step(&exploit).proposal().unwrap().action.channel.unwrap();
        assert_eq!(ce.number(), 36);
    }

    // plan() orders proposals most-exploratory-first and separates passive
    // zones; the ordering is a deterministic function of the inputs.
    #[test]
    fn plan_orders_by_exploration_and_collects_passive() {
        let ctrl = controller();
        let beliefs = vec![
            ZoneBelief::new(zid("low"), Uncertainty::new(0.1), LastResponse::None),
            ZoneBelief::new(zid("high"), Uncertainty::new(0.9), LastResponse::None),
            ZoneBelief::new(zid("mid"), Uncertainty::new(0.5), LastResponse::None),
        ];
        let plan = ctrl.plan(&beliefs);
        let order: Vec<&str> = plan.proposals.iter().map(|p| p.zone.as_str()).collect();
        assert_eq!(order, vec!["high", "mid", "low"]);
        assert!(plan.passive.is_empty());

        // With an empty capability every zone degrades to passive.
        let passive_ctrl =
            ClosedLoopController::new(ControlCapability::none(), ControllerConfig::default());
        let plan2 = passive_ctrl.plan(&beliefs);
        assert!(plan2.proposals.is_empty());
        assert_eq!(plan2.passive, vec![zid("high"), zid("low"), zid("mid")]);
    }

    // Determinism: identical inputs yield identical decisions and plans.
    #[test]
    fn controller_is_deterministic() {
        let ctrl = controller();
        let beliefs = vec![
            ZoneBelief::new(zid("a"), Uncertainty::new(0.7), LastResponse::Unknown).with_cycle(3),
            ZoneBelief::new(
                zid("b"),
                Uncertainty::new(0.2),
                LastResponse::Observed {
                    evidence_level: EvidenceLevel::L3,
                    residual: Uncertainty::new(0.2),
                },
            ),
        ];
        assert_eq!(ctrl.plan(&beliefs), ctrl.plan(&beliefs));
        assert_eq!(ctrl.step(&beliefs[0]), ctrl.step(&beliefs[0]));
    }

    // The whole plan round-trips losslessly through serde (canonical output).
    #[test]
    fn plan_serde_round_trips() {
        let ctrl = controller();
        let beliefs = vec![
            ZoneBelief::new(zid("a"), Uncertainty::new(0.9), LastResponse::None),
            ZoneBelief::new(zid("b"), Uncertainty::new(0.1), LastResponse::Unknown),
        ];
        let plan = ctrl.plan(&beliefs);
        let json = serde_json::to_string(&plan).unwrap();
        let back: MeasurementPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }

    // An empty belief set yields an empty plan, no panic.
    #[test]
    fn empty_beliefs_yield_empty_plan() {
        let plan = controller().plan(&[]);
        assert!(plan.is_empty());
        assert_eq!(plan, MeasurementPlan::empty());
    }
}
