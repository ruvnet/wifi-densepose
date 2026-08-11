//! # `ruview-infogain` — information-gain scheduler (ADR-311, ADR-297 primitive 14)
//!
//! **SYNTHETIC / L0 research-forward scaffold (ADR-282, ADR-297 phase 3).**
//! This crate models *which radios/modalities to spend the next sampling budget
//! on* by value of information. It is a **simulation/model scaffold**: every
//! informativeness estimate is a model prediction and every cost is a modelled
//! magnitude. Nothing here measures hardware, and **no** `MEASURED`, accuracy,
//! or energy/latency/throughput claim is made or implied — a twin predicts, it
//! does not measure (CLAUDE.md honesty discipline; ADR-311 asserts no
//! efficiency number). Any figure produced by this crate is `SYNTHETIC`.
//!
//! ## What it does
//!
//! With many sensors, processing every stream at full rate wastes the three
//! scarce edge resources — compute, energy, bandwidth. This scheduler assigns
//! each candidate action a value
//!
//! ```text
//! Value(sensor) ≈ expected_uncertainty_reduction / (compute + energy + bandwidth)
//! ```
//!
//! and spends the budget on the highest-value actions, so the edge samples the
//! most informative radios first. In a fielded system the numerator comes from
//! the ADR-312 RF-twin forward model against the ADR-308 fused covariance and
//! the denominator from ADR-317 HAL cost descriptors; this crate takes both as
//! caller-supplied inputs and stays a pure allocator.
//!
//! ## The four ADR-297 non-negotiable rules, as they bind this crate
//!
//! 1. **UNKNOWN is first-class, never an error.** A candidate whose
//!    informativeness the model cannot predict is
//!    [`ExpectedReduction::Unknown`] — handled by an explicit
//!    [`UnknownPolicy`] (probe or defer), **never** silently treated as zero. A
//!    malformed cost is UNKNOWN cost and defers the candidate rather than
//!    panicking or guessing. [`Scheduler::plan`] is total: no input panics.
//! 2. **Certificates bind cryptographically.** Out of scope here; a candidate
//!    names an already-authenticated ADR-302 [`SensorId`](ruview_ontology::SensorId).
//! 3. **One canonical semantics downstream.** Candidates reuse the canonical
//!    [`SensorId`](ruview_ontology::SensorId) and HAL [`Modality`](ruview_hal::Modality)
//!    rather than reinventing per-crate identity/modality shapes.
//! 4. **Honest evidence.** A scheduling decision is a resource choice, not a
//!    sensing claim; the plan records which sensors were skipped so ADR-299 can
//!    raise `UNKNOWN` for an under-sampled zone rather than reporting a stale
//!    estimate as current.
//!
//! ## Purity
//!
//! [`Scheduler::plan`] has **no** scheduling side effects: it starts no
//! sampling and touches no hardware — it returns a [`SchedulePlan`]. ADR-306
//! active sensing chooses the probe on each selected sensor; ADR-308 fusion
//! incorporates the result. It is deterministic (no wall clock, no randomness;
//! synthetic scenes vary only by explicit caller-supplied parameters) and
//! bounded in allocation.
//!
//! ## Example
//!
//! ```
//! use ruview_infogain::*;
//! use ruview_hal::Modality;
//! use ruview_ontology::SensorId;
//!
//! let candidates = vec![
//!     SensorAction::new(
//!         SensorId::new("wifi-1").unwrap(),
//!         Modality::Csi,
//!         ExpectedReduction::known(0.9), // high modelled information...
//!         Cost::new(1.0, 1.0, 1.0),      // ...at low cost → high value
//!     ),
//!     SensorAction::new(
//!         SensorId::new("mmwave-occluded").unwrap(),
//!         Modality::Mmwave,
//!         ExpectedReduction::known(0.1), // low information...
//!         Cost::new(5.0, 5.0, 5.0),      // ...at high cost → low value
//!     ),
//! ];
//!
//! let sched = Scheduler::new(SchedulerConfig::default());
//! let plan = sched.plan(&candidates, &Budget::new(2.0, 2.0, 2.0));
//!
//! // Only the informative-per-cost WiFi link fits the budget.
//! assert_eq!(plan.sampled_sensors(), vec![&SensorId::new("wifi-1").unwrap()]);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod candidate;
mod cost;
mod scheduler;

pub use candidate::{ExpectedReduction, SensorAction};
pub use cost::{Cost, CostPolicy};
pub use scheduler::{
    Budget, DeferReason, DeferredAction, ScheduledAction, SchedulePlan, Scheduler, SchedulerConfig,
    SelectReason, UnknownPolicy,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_hal::Modality;
    use ruview_ontology::SensorId;

    fn sid(s: &str) -> SensorId {
        SensorId::new(s).unwrap()
    }

    fn action(name: &str, reduction: ExpectedReduction, cost: Cost) -> SensorAction {
        SensorAction::new(sid(name), Modality::Csi, reduction, cost)
    }

    fn known(name: &str, r: f64, c: f64) -> SensorAction {
        action(name, ExpectedReduction::known(r), Cost::new(c, c, c))
    }

    fn plan(candidates: &[SensorAction], budget: Budget) -> SchedulePlan {
        Scheduler::new(SchedulerConfig::default()).plan(candidates, &budget)
    }

    // ADR-311 §2: the highest value/cost candidate is ranked and selected first.
    #[test]
    fn highest_value_per_cost_selected_first() {
        let candidates = vec![
            known("low", 0.2, 1.0),  // density 0.2 / 3.0
            known("high", 0.9, 1.0), // density 0.9 / 3.0
            known("mid", 0.5, 1.0),  // density 0.5 / 3.0
        ];
        // Ample budget: all fit, but order must be by descending value density.
        let p = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        let order: Vec<&str> = p.selected.iter().map(|a| a.sensor.as_str()).collect();
        assert_eq!(order, vec!["high", "mid", "low"]);
        assert!(p.selected.iter().all(|a| a.reason == SelectReason::Value));
        assert!(p.deferred.is_empty());
    }

    // ADR-311 §1: a high-cost low-gain sensor is deferred when the budget cannot
    // hold both it and the more valuable action.
    #[test]
    fn high_cost_low_gain_deferred_under_budget() {
        let candidates = vec![
            known("cheap-informative", 0.9, 1.0),  // density 0.30
            known("costly-uninformative", 0.1, 5.0), // density ~0.0067
        ];
        // Budget fits the cheap action but not both.
        let p = plan(&candidates, Budget::new(3.0, 3.0, 3.0));
        assert_eq!(p.sampled_sensors(), vec![&sid("cheap-informative")]);
        assert_eq!(p.deferred.len(), 1);
        assert_eq!(p.deferred[0].sensor.as_str(), "costly-uninformative");
        assert_eq!(p.deferred[0].reason, DeferReason::Budget);
    }

    // The cumulative selected cost never exceeds the budget in any dimension.
    #[test]
    fn budget_is_respected_in_every_dimension() {
        let candidates = vec![
            known("a", 0.9, 2.0),
            known("b", 0.8, 2.0),
            known("c", 0.7, 2.0),
            known("d", 0.6, 2.0),
        ];
        let budget = Budget::new(5.0, 5.0, 5.0);
        let p = plan(&candidates, budget);
        assert!(p.spent.compute <= budget.compute + 1e-9);
        assert!(p.spent.energy <= budget.energy + 1e-9);
        assert!(p.spent.bandwidth <= budget.bandwidth + 1e-9);
        // Two of the cost-2 actions fit under a budget of 5; the third does not.
        assert_eq!(p.selected.len(), 2);
    }

    // A candidate that exactly fills the remaining budget is admitted.
    #[test]
    fn exact_fit_is_admitted() {
        let candidates = vec![known("exact", 0.5, 2.0)];
        let p = plan(&candidates, Budget::new(2.0, 2.0, 2.0));
        assert_eq!(p.selected.len(), 1);
        assert_eq!(p.deferred.len(), 0);
    }

    // Deterministic tie-break: equal value density resolves by cheapest weighted
    // cost, then by sensor id — same inputs always give the same plan.
    #[test]
    fn tie_break_is_deterministic() {
        // Equal density (0.5 / 2.0), so tie-break falls to sensor id.
        let candidates = vec![
            known("zebra", 0.5, 2.0),
            known("alpha", 0.5, 2.0),
            known("mike", 0.5, 2.0),
        ];
        let p1 = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        let p2 = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        assert_eq!(p1, p2);
        let order: Vec<&str> = p1.selected.iter().map(|a| a.sensor.as_str()).collect();
        assert_eq!(order, vec!["alpha", "mike", "zebra"]);

        // Cost tie-break wins over id: cheaper same-density action ranks first.
        let mixed = vec![
            action("expensive", ExpectedReduction::known(1.0), Cost::new(2.0, 2.0, 2.0)), // 1/6
            action("cheap", ExpectedReduction::known(0.5), Cost::new(1.0, 1.0, 1.0)),      // 0.5/3 = 1/6
        ];
        let pm = plan(&mixed, Budget::new(99.0, 99.0, 99.0));
        let order: Vec<&str> = pm.selected.iter().map(|a| a.sensor.as_str()).collect();
        assert_eq!(order, vec!["cheap", "expensive"]);
    }

    // ADR-297 rule 1: an unknown-value candidate is NOT treated as zero. Under
    // the default Defer policy it is deferred with an explicit reason.
    #[test]
    fn unknown_value_defer_policy_defers_explicitly() {
        let candidates = vec![
            action("unknown", ExpectedReduction::Unknown, Cost::new(1.0, 1.0, 1.0)),
            known("known", 0.5, 1.0),
        ];
        let p = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        assert_eq!(p.sampled_sensors(), vec![&sid("known")]);
        assert_eq!(p.deferred.len(), 1);
        assert_eq!(p.deferred[0].sensor.as_str(), "unknown");
        assert_eq!(p.deferred[0].reason, DeferReason::UnknownDeferred);
    }

    // ADR-311: the Probe policy spends budget to LEARN an unknown candidate's
    // informativeness, with an explicit synthetic probe value (not zero).
    #[test]
    fn unknown_value_probe_policy_selects_to_learn() {
        let config = SchedulerConfig {
            policy: CostPolicy::UNIFORM,
            unknown: UnknownPolicy::Probe { probe_value: 1.0 },
            sampling_floor: None,
        };
        let candidates = vec![
            action("unknown", ExpectedReduction::Unknown, Cost::new(1.0, 1.0, 1.0)),
            known("weak", 0.1, 1.0),
        ];
        let p = Scheduler::new(config).plan(&candidates, &Budget::new(1.0, 1.0, 1.0));
        // Probe value (1.0) beats the weak known (0.1), so the unknown is probed.
        assert_eq!(p.selected.len(), 1);
        assert_eq!(p.selected[0].sensor.as_str(), "unknown");
        assert_eq!(p.selected[0].reason, SelectReason::Probe);
        // No honest value figure is reported for an unknown reduction.
        assert_eq!(p.selected[0].value_density, None);
    }

    // ADR-311 §2: the sampling floor force-includes a starved low-value sensor
    // so it is re-evaluated rather than permanently blinded.
    #[test]
    fn sampling_floor_forces_starved_low_value_sensor() {
        let config = SchedulerConfig {
            policy: CostPolicy::UNIFORM,
            unknown: UnknownPolicy::Defer,
            sampling_floor: Some(3),
        };
        let mut starved = known("starved", 0.0, 1.0); // zero value: would be deferred
        starved.cycles_since_sampled = 5; // ... but it is past the floor
        let fresh = known("fresh", 0.9, 1.0);
        let candidates = vec![starved, fresh];
        let p = Scheduler::new(config).plan(&candidates, &Budget::new(99.0, 99.0, 99.0));
        // Both selected; the starved one is force-included with the Floor reason.
        assert_eq!(p.selected.len(), 2);
        let starved_sel = p
            .selected
            .iter()
            .find(|a| a.sensor.as_str() == "starved")
            .unwrap();
        assert_eq!(starved_sel.reason, SelectReason::Floor);
        // Floor is best-effort under a hard budget: it cannot fit → deferred.
        let tight = Scheduler::new(config).plan(&candidates, &Budget::new(0.0, 0.0, 0.0));
        assert!(tight.selected.is_empty());
        assert!(tight.deferred.iter().all(|d| d.reason == DeferReason::Budget));
    }

    // Empty candidate set → empty plan, nothing spent, no panic.
    #[test]
    fn empty_candidate_set_yields_empty_plan() {
        let p = plan(&[], Budget::new(10.0, 10.0, 10.0));
        assert!(p.is_empty());
        assert!(p.selected.is_empty());
        assert!(p.deferred.is_empty());
        assert_eq!(p.spent, Cost::ZERO);
        assert_eq!(p, SchedulePlan::empty());
    }

    // Malformed input never panics: non-finite/negative cost defers the
    // candidate (UNKNOWN cost); non-finite reduction becomes Unknown; negative
    // reduction clamps to zero.
    #[test]
    fn malformed_input_never_panics() {
        // Non-finite reduction → Unknown.
        assert_eq!(ExpectedReduction::known(f64::NAN), ExpectedReduction::Unknown);
        assert_eq!(
            ExpectedReduction::known(f64::INFINITY),
            ExpectedReduction::Unknown
        );
        // Negative reduction clamps to zero.
        assert_eq!(ExpectedReduction::known(-3.0), ExpectedReduction::Known(0.0));

        let candidates = vec![
            action("nan-cost", ExpectedReduction::known(0.9), Cost::new(f64::NAN, 1.0, 1.0)),
            action("neg-cost", ExpectedReduction::known(0.9), Cost::new(-1.0, 1.0, 1.0)),
            action("inf-cost", ExpectedReduction::known(0.9), Cost::new(f64::INFINITY, 1.0, 1.0)),
            known("good", 0.9, 1.0),
        ];
        let p = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        // Only the well-formed candidate is sampled; the rest defer as malformed.
        assert_eq!(p.sampled_sensors(), vec![&sid("good")]);
        let malformed: Vec<&str> = p
            .deferred
            .iter()
            .filter(|d| d.reason == DeferReason::MalformedCost)
            .map(|d| d.sensor.as_str())
            .collect();
        assert_eq!(malformed, vec!["inf-cost", "nan-cost", "neg-cost"]);
    }

    // A zero-cost (free) action gets a bounded, finite value density and is not
    // rejected by a division by zero.
    #[test]
    fn zero_cost_action_is_bounded_not_infinite() {
        let candidates = vec![action(
            "free",
            ExpectedReduction::known(1.0),
            Cost::new(0.0, 0.0, 0.0),
        )];
        let p = plan(&candidates, Budget::new(1.0, 1.0, 1.0));
        assert_eq!(p.selected.len(), 1);
        let d = p.selected[0].value_density.unwrap();
        assert!(d.is_finite());
    }

    // A known-zero-reduction candidate is deferred as NoGain (honest: no
    // modelled information), distinct from UNKNOWN.
    #[test]
    fn known_zero_reduction_defers_as_no_gain() {
        let candidates = vec![known("nogain", 0.0, 1.0)];
        let p = plan(&candidates, Budget::new(99.0, 99.0, 99.0));
        assert!(p.selected.is_empty());
        assert_eq!(p.deferred.len(), 1);
        assert_eq!(p.deferred[0].reason, DeferReason::NoGain);
    }

    // The cost policy is a deployment choice: weighting a resource heavily can
    // flip which sensor is more valuable per unit cost.
    #[test]
    fn cost_policy_weighting_changes_ranking() {
        // "a" is cheap on compute but expensive on energy; "b" the reverse.
        let a = action("a", ExpectedReduction::known(1.0), Cost::new(1.0, 10.0, 1.0));
        let b = action("b", ExpectedReduction::known(1.0), Cost::new(10.0, 1.0, 1.0));
        let candidates = vec![a, b];

        // Energy-heavy policy (battery node): "a" is costlier → "b" ranks first.
        let energy_heavy = SchedulerConfig {
            policy: CostPolicy::new(1.0, 100.0, 1.0),
            unknown: UnknownPolicy::Defer,
            sampling_floor: None,
        };
        let p = Scheduler::new(energy_heavy).plan(&candidates, &Budget::new(99.0, 99.0, 99.0));
        assert_eq!(p.selected[0].sensor.as_str(), "b");

        // Compute-heavy policy (wired gateway): the ranking flips to "a".
        let compute_heavy = SchedulerConfig {
            policy: CostPolicy::new(100.0, 1.0, 1.0),
            unknown: UnknownPolicy::Defer,
            sampling_floor: None,
        };
        let p = Scheduler::new(compute_heavy).plan(&candidates, &Budget::new(99.0, 99.0, 99.0));
        assert_eq!(p.selected[0].sensor.as_str(), "a");
    }

    // Determinism: identical inputs yield byte-identical plans across runs.
    #[test]
    fn plan_is_deterministic() {
        let candidates = vec![
            known("a", 0.7, 2.0),
            known("b", 0.3, 1.0),
            action("c", ExpectedReduction::Unknown, Cost::new(1.0, 1.0, 1.0)),
        ];
        let budget = Budget::new(3.0, 3.0, 3.0);
        let sched = Scheduler::new(SchedulerConfig::default());
        assert_eq!(sched.plan(&candidates, &budget), sched.plan(&candidates, &budget));
    }

    // The whole plan round-trips losslessly through serde (canonical output).
    #[test]
    fn plan_serde_round_trips() {
        let candidates = vec![
            known("a", 0.9, 1.0),
            known("b", 0.1, 5.0),
            action("c", ExpectedReduction::Unknown, Cost::new(1.0, 1.0, 1.0)),
        ];
        let p = plan(&candidates, Budget::new(2.0, 2.0, 2.0));
        let json = serde_json::to_string(&p).unwrap();
        let back: SchedulePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
