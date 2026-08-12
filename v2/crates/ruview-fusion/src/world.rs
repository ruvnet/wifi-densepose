//! The fused output: one probabilistic world state (ADR-311 §3).
//!
//! The invariant of ADR-311 is the *shape* of the output: many observations
//! resolve to **one** [`WorldState`], not many feeds into a visualization. A
//! [`WorldState`] holds a per-container [`ZoneState`], each carrying either a
//! fused [`Presence::Estimated`] belief or a first-class [`Presence::Unknown`]
//! when the evidence cannot support a confident single value. Every fused value
//! keeps recoverable per-observation provenance ([`Contribution`]s) and an
//! aggregate evidence level that is never lifted above the weakest contributing
//! input (ADR-311: "never upgraded above the weakest contributing L-level").

use serde::{Deserialize, Serialize};

use ruview_hal::Modality;
use ruview_ontology::{Container, EvidenceLevel, ObservationId, SensorId};

use crate::estimate::Estimate;

/// Why a zone resolved to UNKNOWN instead of a confident estimate. UNKNOWN is a
/// value, not an error (ADR-300 rule 1): the reason stays legible.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum UnknownReason {
    /// Fewer usable (non-degraded, quantified) observations covered the zone
    /// than the engine's minimum, so there is not enough evidence to resolve it.
    InsufficientCoverage,
    /// Contributing observations disagree by more than their stated uncertainty
    /// allows. The engine refuses to emit a confident average of irreconcilable
    /// sources and reports the disagreement instead.
    IrreconcilableConflict {
        /// The reduced chi-square disagreement statistic that crossed the
        /// configured threshold.
        reduced_chi_square: f64,
    },
}

/// The fused occupancy belief for one container.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "presence", rename_all = "snake_case")]
pub enum Presence {
    /// A fused probabilistic belief: probability of occupancy and its variance.
    Estimated {
        /// Fused probability of occupancy in `[0.0, 1.0]`.
        probability: f64,
        /// Fused variance — sharpened (smaller) when sources agree.
        variance: f64,
    },
    /// UNKNOWN — the evidence could not resolve to one confident estimate.
    Unknown {
        /// Why the zone is UNKNOWN.
        reason: UnknownReason,
    },
}

impl Presence {
    /// True when this is [`Presence::Unknown`].
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

/// One contributing observation's recoverable provenance in a fused zone.
///
/// Every observation grouped into a zone yields a `Contribution`, whether or not
/// it moved the fused value. `estimate` is `Some` for a contributing source and
/// `None` for an abstaining one (degraded / UNKNOWN); `weight` is its normalized
/// inverse-variance weight in `[0.0, 1.0]` (`0.0` when it did not contribute),
/// which makes down-weighting of uncertain sources auditable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contribution {
    /// The contributing observation's id.
    pub observation: ObservationId,
    /// The authenticated sensor that produced it.
    pub sensor: SensorId,
    /// The modality it was sensed through.
    pub modality: Modality,
    /// The observation's own evidence level.
    pub evidence_level: EvidenceLevel,
    /// The presence estimate it contributed, or `None` if it abstained.
    pub estimate: Option<Estimate>,
    /// Its normalized weight in the fused value, in `[0.0, 1.0]`.
    pub weight: f64,
}

impl Contribution {
    /// True when this observation contributed a weighted estimate (did not
    /// abstain).
    #[must_use]
    pub fn contributed(&self) -> bool {
        self.estimate.is_some()
    }
}

/// The fused belief and provenance for a single container.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneState {
    /// The container (space or zone) this state describes.
    pub container: Container,
    /// The fused occupancy belief, or UNKNOWN.
    pub presence: Presence,
    /// Aggregate evidence level — the minimum over contributing observations,
    /// never above the weakest necessary input; `L0` when nothing contributed.
    pub evidence_level: EvidenceLevel,
    /// Per-observation provenance for every observation grouped into this zone,
    /// in a deterministic (observation-id) order.
    pub contributions: Vec<Contribution>,
}

impl ZoneState {
    /// True when this zone resolved to UNKNOWN.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        self.presence.is_unknown()
    }

    /// The ids of the observations that contributed a weighted estimate.
    pub fn contributing_observations(&self) -> impl Iterator<Item = &ObservationId> {
        self.contributions
            .iter()
            .filter(|c| c.contributed())
            .map(|c| &c.observation)
    }
}

/// One probabilistic world state fused from many observations.
///
/// This is the single object every downstream consumer reads (ADR-312/310/312):
/// one probabilistic world, not a modality stack. Zones are held in a
/// deterministic order so the state is reproducible.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldState {
    /// The "as-of" time of the state (Unix ms) — the maximum contributing
    /// observation timestamp, injected via the observations, never sampled from
    /// a clock. `0` when there were no observations.
    pub at_unix_ms: i64,
    /// The fused per-container states, ordered deterministically by container.
    pub zones: Vec<ZoneState>,
}

impl WorldState {
    /// The fused state for a container, if present.
    #[must_use]
    pub fn zone(&self, container: &Container) -> Option<&ZoneState> {
        self.zones.iter().find(|z| &z.container == container)
    }
}
