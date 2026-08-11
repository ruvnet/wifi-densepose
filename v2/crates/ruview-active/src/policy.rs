//! The closed-loop experiment controller (ADR-306 §2).
//!
//! **SYNTHETIC / L0 model scaffold.** This is the *loop* ADR-280 deferred: it
//! reads a modelled per-zone uncertainty and the last modelled response, and
//! proposes the next controllable measurement configuration expected to reduce
//! that uncertainty most. It is an information-driven **planning** policy — it
//! shares the *notion* of expected gain with ADR-311 but defines its own
//! control vocabulary and takes **no** dependency on `ruview-infogain`, so the
//! two crates build in parallel.
//!
//! No number here is `MEASURED`. Every exploration level is a modelled
//! magnitude, not a measured information gain (CLAUDE.md honesty discipline).
//! The controller **emits a plan; it never emits RF** and never bypasses the
//! ADR-280 governed admission/actuation surface — a fielded caller submits each
//! proposal through that fail-closed path.
//!
//! ## First-class UNKNOWN (ADR-297 rule 1)
//!
//! The last response is [`LastResponse::Unknown`] whenever the previous
//! solicited measurement returned nothing interpretable. UNKNOWN is **not**
//! zero uncertainty and **not** an error: the policy *widens* exploration by
//! [`ControllerConfig::unknown_widen`] rather than committing to a narrow,
//! exploitative configuration on a target it cannot currently resolve.

use serde::{Deserialize, Serialize};

use ruview_ontology::{EvidenceLevel, SemanticProvenance, ZoneId};

use crate::control::{ControlAction, ControlCapability};

/// A modelled uncertainty scalar in `[0, 1]`: `0.0` fully resolved, `1.0`
/// maximally uncertain. Construction clamps to range and maps a non-finite
/// input to maximal uncertainty (an unusable estimate is treated as "know
/// nothing", never silently as zero).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Uncertainty(f64);

impl Uncertainty {
    /// Clamp an arbitrary value into `[0, 1]`; a non-finite value becomes
    /// maximal uncertainty (`1.0`).
    #[must_use]
    pub fn new(value: f64) -> Self {
        if value.is_finite() {
            Self(value.clamp(0.0, 1.0))
        } else {
            Self(1.0)
        }
    }

    /// The clamped scalar value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// The outcome of the previous solicited measurement for a zone.
///
/// This reuses the canonical [`EvidenceLevel`] vocabulary rather than a
/// per-crate grade (ADR-297 rule 3). A fielded caller derives it from the
/// ADR-303 [`Observation`](ruview_ontology::Observation) the sounding produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LastResponse {
    /// An interpretable response arrived at the given evidence level, leaving a
    /// modelled residual uncertainty.
    Observed {
        /// The canonical evidence level of the response.
        evidence_level: EvidenceLevel,
        /// Modelled residual uncertainty left by the response.
        residual: Uncertainty,
    },
    /// The last solicited measurement returned nothing interpretable — a
    /// first-class UNKNOWN, not an error and not zero uncertainty.
    Unknown,
    /// No measurement has been solicited yet (loop start).
    None,
}

/// The modelled belief about a single controllable target zone, and the input
/// to one closed-loop [`ClosedLoopController::step`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneBelief {
    /// The target zone (canonical ontology id, ADR-303).
    pub zone: ZoneId,
    /// Current modelled uncertainty about the zone.
    pub uncertainty: Uncertainty,
    /// The outcome of the previous solicited measurement.
    pub last_response: LastResponse,
    /// A deterministic loop counter used only to sweep channels across cycles.
    /// Injected by the caller — never sampled from a clock (ADR-297 §rules).
    #[serde(default)]
    pub cycle: u64,
}

impl ZoneBelief {
    /// Construct a belief. `cycle` defaults to `0`.
    #[must_use]
    pub fn new(zone: ZoneId, uncertainty: Uncertainty, last_response: LastResponse) -> Self {
        Self {
            zone,
            uncertainty,
            last_response,
            cycle: 0,
        }
    }

    /// Builder-style setter for the deterministic sweep cycle.
    #[must_use]
    pub fn with_cycle(mut self, cycle: u64) -> Self {
        self.cycle = cycle;
        self
    }
}

/// Whether a proposal is exploratory (widen to resolve a poorly-known zone) or
/// exploitative (narrow, concentrate on a well-known zone).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlIntent {
    /// Widen the measurement to resolve high uncertainty.
    Explore,
    /// Narrow the measurement to exploit an already-resolved zone.
    Exploit,
}

/// Why the controller could not propose a controllable action and fell back to
/// the passive planner (ADR-306 §2 degradation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassiveReason {
    /// The deployment exposes no controllable axis (e.g. ESP32-only). The
    /// controller defers to the ADR-280 staleness planner rather than
    /// fabricating a gain estimate.
    NoControllableAxes,
}

/// A proposed governed measurement for one zone.
///
/// The `exploration` scalar is a **SYNTHETIC** modelled magnitude, never a
/// measured information gain, and the proposal carries L1 (heuristic/synthetic)
/// evidence with an explicit provenance so no projection can silently upgrade
/// it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlProposal {
    /// The target zone.
    pub zone: ZoneId,
    /// The controllable configuration to request (a plan, not an emission).
    pub action: ControlAction,
    /// Explore vs exploit.
    pub intent: ControlIntent,
    /// Modelled exploration level in `[0, 1]` (SYNTHETIC; not a measurement).
    pub exploration: f64,
    /// Honest evidence label for the proposal — always L1 (synthetic model).
    pub evidence_level: EvidenceLevel,
    /// Provenance tagging the proposal as a synthetic model output.
    pub provenance: SemanticProvenance,
}

/// The controller's decision for one zone: either a governed proposal or a
/// passive fallback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlDecision {
    /// Request a controllable measurement.
    Actuate(ControlProposal),
    /// No controllable axis — defer to the passive planner.
    Passive {
        /// The zone that could not be actively controlled.
        zone: ZoneId,
        /// Why the fallback occurred.
        reason: PassiveReason,
    },
}

impl ControlDecision {
    /// The proposal, if this decision is an actuation.
    #[must_use]
    pub fn proposal(&self) -> Option<&ControlProposal> {
        match self {
            Self::Actuate(p) => Some(p),
            Self::Passive { .. } => None,
        }
    }
}

/// A full measurement plan over several zones, ordered most-uncertain-first.
///
/// It is a pure planning artifact: it starts no sounding and touches no
/// hardware. Zones with no controllable axis are recorded in
/// [`MeasurementPlan::passive`] so the caller knows to route them to the
/// staleness planner instead.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MeasurementPlan {
    /// Governed proposals, ordered by descending exploration then by zone id.
    pub proposals: Vec<ControlProposal>,
    /// Zones that degraded to the passive planner.
    pub passive: Vec<ZoneId>,
}

impl MeasurementPlan {
    /// An empty plan.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when the plan contains neither a proposal nor a passive zone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty() && self.passive.is_empty()
    }
}

/// Configuration for the closed-loop controller. All knobs are deployment
/// choices; there is no wall clock and no randomness.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Exploration level at or above which a proposal is [`ControlIntent::Explore`]
    /// (below it, [`ControlIntent::Exploit`]). In `[0, 1]`.
    pub explore_threshold: f64,
    /// Additive widening applied to the exploration level when the last
    /// response was [`LastResponse::Unknown`]. Bounded into `[0, 1]` after
    /// application (ADR-297 rule 1: UNKNOWN widens rather than commits).
    pub unknown_widen: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            explore_threshold: 0.5,
            unknown_widen: 0.3,
        }
    }
}

/// The synthetic model version stamped onto every proposal's provenance.
pub const MODEL_VERSION: &str = "ruview-active@synthetic-l0";

/// The closed-loop RF experiment controller (ADR-306).
///
/// Holds the controllable [`ControlCapability`] of the deployment and the
/// policy [`ControllerConfig`]. [`ClosedLoopController::step`] is a total,
/// deterministic function of its inputs: identical inputs always yield an
/// identical decision, and no input panics.
#[derive(Clone, Debug)]
pub struct ClosedLoopController {
    capability: ControlCapability,
    config: ControllerConfig,
}

impl ClosedLoopController {
    /// Build a controller over a deployment's controllable capability set.
    #[must_use]
    pub fn new(capability: ControlCapability, config: ControllerConfig) -> Self {
        Self { capability, config }
    }

    /// The controllable capability set.
    #[must_use]
    pub fn capability(&self) -> &ControlCapability {
        &self.capability
    }

    /// One closed-loop step for a single zone: read the belief, compute the
    /// modelled exploration level, and propose the next controllable
    /// configuration — or fall back to the passive planner when nothing is
    /// controllable.
    #[must_use]
    pub fn step(&self, belief: &ZoneBelief) -> ControlDecision {
        if self.capability.is_empty() {
            return ControlDecision::Passive {
                zone: belief.zone.clone(),
                reason: PassiveReason::NoControllableAxes,
            };
        }

        let exploration = self.exploration_level(belief);
        let explore = exploration >= self.config.explore_threshold;
        let intent = if explore {
            ControlIntent::Explore
        } else {
            ControlIntent::Exploit
        };

        let action = self.select_action(belief, exploration, explore);

        ControlDecision::Actuate(ControlProposal {
            zone: belief.zone.clone(),
            action,
            intent,
            exploration,
            evidence_level: EvidenceLevel::L1,
            provenance: SemanticProvenance::declared(MODEL_VERSION),
        })
    }

    /// Plan across several zones. Each zone is stepped; proposals are ordered
    /// most-exploratory-first (tie-break by zone id) so the scarcest budget is
    /// spent where uncertainty is highest, and passive zones are collected
    /// separately.
    #[must_use]
    pub fn plan(&self, beliefs: &[ZoneBelief]) -> MeasurementPlan {
        let mut proposals = Vec::new();
        let mut passive = Vec::new();
        for belief in beliefs {
            match self.step(belief) {
                ControlDecision::Actuate(p) => proposals.push(p),
                ControlDecision::Passive { zone, .. } => passive.push(zone),
            }
        }
        // Deterministic ordering: descending exploration, then ascending zone id.
        proposals.sort_by(|a, b| {
            b.exploration
                .partial_cmp(&a.exploration)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.zone.as_str().cmp(b.zone.as_str()))
        });
        passive.sort();
        MeasurementPlan { proposals, passive }
    }

    /// The modelled exploration level for a belief: driven by current
    /// uncertainty, widened when the last response was UNKNOWN.
    fn exploration_level(&self, belief: &ZoneBelief) -> f64 {
        let base = belief.uncertainty.value();
        let e = match &belief.last_response {
            // UNKNOWN response: widen exploration rather than commit.
            LastResponse::Unknown => base + self.config.unknown_widen.max(0.0),
            LastResponse::Observed { .. } | LastResponse::None => base,
        };
        e.clamp(0.0, 1.0)
    }

    /// Map the exploration level onto a controllable action across the axes the
    /// deployment exposes. Uncontrollable axes stay `None`.
    fn select_action(&self, belief: &ZoneBelief, exploration: f64, explore: bool) -> ControlAction {
        let cap = &self.capability;

        // Channel: sweep across cycles when exploring; anchor to the first
        // channel when exploiting. Categorical axis, no exploration grading.
        let channel = if cap.channels.is_empty() {
            None
        } else if explore {
            let idx = (belief.cycle as usize) % cap.channels.len();
            Some(cap.channels[idx])
        } else {
            Some(cap.channels[0])
        };

        // Graded axes: capability vectors are sorted least-exploratory-first,
        // so a higher exploration level selects a wider / faster value.
        let bandwidth = graded_pick(&cap.bandwidths, exploration).copied();
        let cadence = graded_pick(&cap.cadences, exploration).copied();
        let antenna = graded_pick(&cap.antennas, exploration).cloned();

        ControlAction {
            channel,
            bandwidth,
            cadence,
            antenna,
        }
    }
}

/// Pick from a least-exploratory-first slice by mapping an exploration level in
/// `[0, 1]` onto an index. Returns `None` for an empty slice.
fn graded_pick<T>(values: &[T], exploration: f64) -> Option<&T> {
    if values.is_empty() {
        return None;
    }
    let e = exploration.clamp(0.0, 1.0);
    let last = values.len() - 1;
    let idx = (e * last as f64).round() as usize;
    values.get(idx.min(last))
}
