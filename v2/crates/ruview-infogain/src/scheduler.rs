//! The information-gain scheduler: rank candidates by value of information and
//! select the most informative subset under a resource budget (ADR-311 §2).
//!
//! **SYNTHETIC / L0 scaffold (ADR-282).** The scheduler emits an *allocation*
//! (a [`SchedulePlan`]), never a measurement and never a sensing claim. It has
//! **no** side effects: it starts no sampling, touches no hardware, and asserts
//! no efficiency figure — ADR-306 active sensing chooses the probe on each
//! selected sensor and ADR-308 fusion incorporates the result. A scheduling
//! decision is a resource choice, not evidence.
//!
//! ## Selection algorithm (documented)
//!
//! The value function is
//!
//! ```text
//! Value(action) = expected_uncertainty_reduction / weighted_cost
//! ```
//!
//! where `weighted_cost` collapses the compute/energy/bandwidth triple under the
//! configured [`CostPolicy`](crate::CostPolicy). Maximising total expected
//! reduction under a multi-resource budget is a knapsack; this scheduler uses a
//! **bounded greedy** heuristic — sort candidates by value density (reduction
//! per unit weighted cost) and take each that still fits the remaining budget.
//! It is `O(n log n)`, allocates one bounded working vector, and is fully
//! deterministic. A candidate that does not fit is deferred, not dropped, and
//! the scheduler keeps scanning lower-density candidates that may still fit —
//! so a small cheap action can be picked after a large one is skipped.
//!
//! Two policies sit on top of the greedy core:
//! - **Sampling floor**: a candidate whose `cycles_since_sampled` has reached
//!   the configured floor is *force-included* (subject only to the hard budget)
//!   so a low-value sensor is re-evaluated as the scene changes rather than
//!   being starved permanently.
//! - **Unknown-value handling**: a candidate with an
//!   [`Unknown`](crate::ExpectedReduction::Unknown) reduction is never treated
//!   as zero — the [`UnknownPolicy`] either probes it (assigns an explicit probe
//!   value so budget is spent to *learn* its informativeness) or defers it.

use serde::{Deserialize, Serialize};

use ruview_hal::Modality;
use ruview_ontology::SensorId;

use crate::candidate::{ExpectedReduction, SensorAction};
use crate::cost::{Cost, CostPolicy};

/// Tolerance for the budget fit comparison, absorbing float round-off so a
/// candidate that exactly fills the budget is not spuriously rejected.
const BUDGET_EPSILON: f64 = 1e-9;

/// How the scheduler treats a candidate with an
/// [`Unknown`](crate::ExpectedReduction::Unknown) expected reduction (ADR-311:
/// unknown value is not zero value).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum UnknownPolicy {
    /// Defer unknown-value candidates: they are not ranked by value (an unknown
    /// is not asserted to be worthless), but they remain eligible for the
    /// sampling floor so they are eventually re-evaluated.
    Defer,
    /// Probe unknown-value candidates: assign them an explicit optimistic
    /// `probe_value` so the scheduler may spend budget to *learn* their
    /// informativeness. The value is synthetic exploration pressure, not a
    /// prediction; it is clamped to a finite non-negative number.
    Probe {
        /// The synthetic value density weight given to an unknown candidate.
        probe_value: f64,
    },
}

/// Scheduler configuration: the cost policy, the unknown-value policy, and the
/// optional sampling floor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// How the cost triple is weighted into the value-function denominator.
    pub policy: CostPolicy,
    /// How unknown-value candidates are handled.
    pub unknown: UnknownPolicy,
    /// Force-sample a candidate once `cycles_since_sampled >=` this value.
    /// `None` disables the floor. The floor is best-effort under the hard
    /// budget — a forced candidate that cannot fit any resource is still
    /// deferred rather than violating the budget.
    #[serde(default)]
    pub sampling_floor: Option<u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: CostPolicy::UNIFORM,
            unknown: UnknownPolicy::Defer,
            sampling_floor: None,
        }
    }
}

/// The multi-resource budget for one scheduling cycle. Each selected action
/// consumes its [`Cost`] triple; the cumulative spend may not exceed the budget
/// in any single dimension.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Compute budget for the cycle.
    pub compute: f64,
    /// Energy budget for the cycle.
    pub energy: f64,
    /// Bandwidth budget for the cycle.
    pub bandwidth: f64,
}

impl Budget {
    /// Construct a budget, clamping any non-finite or negative dimension to
    /// `0.0` (an unusable dimension admits nothing, rather than erroring).
    #[must_use]
    pub fn new(compute: f64, energy: f64, bandwidth: f64) -> Self {
        Self {
            compute: clamp_budget(compute),
            energy: clamp_budget(energy),
            bandwidth: clamp_budget(bandwidth),
        }
    }

    /// True when `spent + cost` stays within every dimension of this budget.
    fn admits(&self, spent: &Cost, cost: &Cost) -> bool {
        spent.compute + cost.compute <= self.compute + BUDGET_EPSILON
            && spent.energy + cost.energy <= self.energy + BUDGET_EPSILON
            && spent.bandwidth + cost.bandwidth <= self.bandwidth + BUDGET_EPSILON
    }
}

fn clamp_budget(v: f64) -> f64 {
    if v.is_finite() && v >= 0.0 {
        v
    } else {
        0.0
    }
}

/// Why a candidate was selected into the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectReason {
    /// Selected by information-gain value density (the ordinary path).
    Value,
    /// Force-included by the sampling floor, not by its current value.
    Floor,
    /// Selected to probe an unknown-value candidate and learn its informativeness.
    Probe,
}

/// Why a candidate was deferred (skipped this cycle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferReason {
    /// No remaining budget in at least one resource dimension.
    Budget,
    /// Unknown expected reduction under [`UnknownPolicy::Defer`] — deferred
    /// explicitly, *not* treated as zero value.
    UnknownDeferred,
    /// A known, non-positive expected reduction: no modelled information to gain.
    NoGain,
    /// The cost triple was malformed (non-finite/negative); cost is UNKNOWN, so
    /// the candidate is deferred rather than guessed at.
    MalformedCost,
}

/// One selected action in the emitted plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScheduledAction {
    /// The sensor to sample.
    pub sensor: SensorId,
    /// Its modality.
    pub modality: Modality,
    /// The value density that ranked it, when defined (`None` for an
    /// unknown-value candidate forced in by the floor).
    pub value_density: Option<f64>,
    /// Why it was selected.
    pub reason: SelectReason,
}

/// One deferred (skipped) action in the emitted plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeferredAction {
    /// The sensor that was not sampled this cycle.
    pub sensor: SensorId,
    /// Its modality.
    pub modality: Modality,
    /// Why it was deferred.
    pub reason: DeferReason,
}

/// The scheduler's output: a pure allocation for one cycle.
///
/// Recording both `selected` and `deferred` is the ADR-311 §3 honesty
/// requirement — skipping a sensor is a *deliberate* reduction in coverage, so
/// downstream observability (ADR-299) can raise `UNKNOWN` for an under-sampled
/// zone rather than reporting a stale estimate as current.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchedulePlan {
    /// Actions to sample this cycle, in selection order (floor-forced first,
    /// then descending value density).
    pub selected: Vec<ScheduledAction>,
    /// Actions skipped this cycle, each with its reason, sorted by sensor id.
    pub deferred: Vec<DeferredAction>,
    /// Total modelled cost the plan commits (component-wise, ≤ budget).
    pub spent: Cost,
}

impl SchedulePlan {
    /// The empty plan (no candidates, nothing spent).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            selected: Vec::new(),
            deferred: Vec::new(),
            spent: Cost::ZERO,
        }
    }

    /// True when nothing was selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// The sensors actually sampled by this plan, for the ADR-311 §3 sampling
    /// record consumed downstream.
    #[must_use]
    pub fn sampled_sensors(&self) -> Vec<&SensorId> {
        self.selected.iter().map(|a| &a.sensor).collect()
    }
}

/// The information-gain scheduler.
#[derive(Clone, Copy, Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
}

/// Internal classification of a candidate before selection.
struct Ranked<'a> {
    action: &'a SensorAction,
    density: f64,
    reason: SelectReason,
    /// The value density to report, `None` when the reduction is unknown.
    reported_density: Option<f64>,
}

impl Scheduler {
    /// Construct a scheduler with the given configuration.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Produce an allocation for one cycle. Pure and deterministic: identical
    /// candidates + budget always yield an identical plan, with no side effects.
    #[must_use]
    pub fn plan(&self, candidates: &[SensorAction], budget: &Budget) -> SchedulePlan {
        let mut forced: Vec<Ranked<'_>> = Vec::new();
        let mut ranked: Vec<Ranked<'_>> = Vec::new();
        let mut deferred: Vec<DeferredAction> = Vec::new();

        for action in candidates {
            // Malformed cost is UNKNOWN cost — defer, never guess a number.
            if !action.cost.is_well_formed() {
                deferred.push(defer(action, DeferReason::MalformedCost));
                continue;
            }

            let floor_forced = self
                .config
                .sampling_floor
                .is_some_and(|n| action.cycles_since_sampled >= n);

            // Determine the value density and the "ordinary" (non-floor) reason.
            let (density, reported, ordinary_reason, defer_reason) =
                self.classify(action);

            if floor_forced {
                // Force-included regardless of value; report the floor reason
                // but keep the density we could compute (may be None).
                forced.push(Ranked {
                    action,
                    density,
                    reason: SelectReason::Floor,
                    reported_density: reported,
                });
                continue;
            }

            match ordinary_reason {
                Some(reason) => ranked.push(Ranked {
                    action,
                    density,
                    reason,
                    reported_density: reported,
                }),
                // Not force-forced and no positive value: defer with the honest
                // reason (NoGain or UnknownDeferred).
                None => deferred.push(defer(action, defer_reason)),
            }
        }

        // Forced candidates go first, in a deterministic (sensor-id) order.
        forced.sort_by(|a, b| a.action.sensor.as_str().cmp(b.action.sensor.as_str()));

        // Value-ranked candidates: highest density first, then cheapest, then
        // sensor id — a fully deterministic total order (no NaN, all clamped).
        ranked.sort_by(|a, b| {
            b.density
                .total_cmp(&a.density)
                .then_with(|| {
                    let ca = self.config.policy.scalar_cost(&a.action.cost);
                    let cb = self.config.policy.scalar_cost(&b.action.cost);
                    ca.total_cmp(&cb)
                })
                .then_with(|| a.action.sensor.as_str().cmp(b.action.sensor.as_str()))
        });

        let mut selected: Vec<ScheduledAction> = Vec::new();
        let mut spent = Cost::ZERO;

        for r in forced.into_iter().chain(ranked.into_iter()) {
            if budget.admits(&spent, &r.action.cost) {
                spent = spent.plus(&r.action.cost);
                selected.push(ScheduledAction {
                    sensor: r.action.sensor.clone(),
                    modality: r.action.modality.clone(),
                    value_density: r.reported_density,
                    reason: r.reason,
                });
            } else {
                deferred.push(defer(r.action, DeferReason::Budget));
            }
        }

        deferred.sort_by(|a, b| a.sensor.as_str().cmp(b.sensor.as_str()));

        SchedulePlan {
            selected,
            deferred,
            spent,
        }
    }

    /// Classify a well-formed candidate into `(density, reported_density,
    /// ordinary_reason, defer_reason_if_no_value)`.
    fn classify(
        &self,
        action: &SensorAction,
    ) -> (f64, Option<f64>, Option<SelectReason>, DeferReason) {
        match action.expected_reduction {
            ExpectedReduction::Known(v) if v > 0.0 => {
                let density = v / self.config.policy.scalar_cost(&action.cost);
                (density, Some(density), Some(SelectReason::Value), DeferReason::NoGain)
            }
            ExpectedReduction::Known(_) => {
                // Known zero reduction: no modelled information to gain.
                (0.0, Some(0.0), None, DeferReason::NoGain)
            }
            ExpectedReduction::Unknown => match self.config.unknown {
                UnknownPolicy::Probe { probe_value } => {
                    let pv = if probe_value.is_finite() && probe_value >= 0.0 {
                        probe_value
                    } else {
                        0.0
                    };
                    let density = pv / self.config.policy.scalar_cost(&action.cost);
                    // Reported density stays None: an unknown reduction has no
                    // honest value figure even when probed.
                    (density, None, Some(SelectReason::Probe), DeferReason::UnknownDeferred)
                }
                UnknownPolicy::Defer => (0.0, None, None, DeferReason::UnknownDeferred),
            },
        }
    }
}

fn defer(action: &SensorAction, reason: DeferReason) -> DeferredAction {
    DeferredAction {
        sensor: action.sensor.clone(),
        modality: action.modality.clone(),
        reason,
    }
}
