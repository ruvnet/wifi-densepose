//! Scene hypotheses over the canonical ontology (ADR-313 §1).
//!
//! **SYNTHETIC / L0 — a research-forward model scaffold, not a measurement
//! system.** A [`Hypothesis`] is a *hypothesized* scene state — an occupant
//! count and their coarse positions — expressed over the ADR-306 canonical
//! [`SpaceId`] so a counterfactual result is a governed spatial statement, not
//! an opaque score. Nothing here is a hardware, `MEASURED`, or accuracy claim,
//! and this crate asserts **no** discrimination-accuracy number (ADR-313
//! evidence discipline).
//!
//! Hypotheses are drawn from (and score *relative to*) the ADR-311 fused world
//! state and its neighbourhood: the current estimate, the **null hypothesis**
//! (nobody present, [`Hypothesis::empty`]), and a bounded set of nearby
//! alternatives (±1 occupant, shifted position). Positions are a coarse metric
//! abstraction in the twin's local frame, not surveyed coordinates.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ruview_ontology::SpaceId;

/// Upper bound on occupants in a single hypothesis. Bounds allocation and the
/// per-link scoring cost on untrusted input; construction beyond this is
/// rejected, never truncated.
pub const MAX_OCCUPANTS: usize = 64;

/// Upper bound on hypotheses evaluated in one call. Bounds allocation on
/// untrusted input.
pub const MAX_HYPOTHESES: usize = 256;

/// One hypothesized occupant at a coarse metric position in the twin's frame.
///
/// **SYNTHETIC.** An occupant is modelled by the counterfactual layer as a body
/// that attenuates any link whose line of sight passes near it (see
/// [`crate::infer`]); it is a hypothesis element, never evidence of a person.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Occupant {
    /// Coarse `x` position, metres, in the twin's local frame.
    pub x: f64,
    /// Coarse `y` position, metres, in the twin's local frame.
    pub y: f64,
}

impl Occupant {
    /// Construct an occupant at a coarse position.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The horizontal-plane position `(x, y)` used for link-blocking geometry.
    #[must_use]
    pub const fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// True when both coordinates are finite (rejects `NaN`/`inf`).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// A hypothesized scene state: how many occupants are present and where.
///
/// **SYNTHETIC / L0.** The occupant count is `occupants.len()`; the **null
/// hypothesis** ([`Hypothesis::empty`]) is an empty occupant set — *nobody
/// present*. The [`id`](Self::id) is a stable label used both for reporting the
/// best explanation and as a deterministic tie-break in ranking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    /// Stable, caller-supplied label (e.g. `"empty"`, `"one"`, `"two"`). Used to
    /// report the best explanation and to break ranking ties deterministically.
    pub id: String,
    /// The ontology space this hypothesis is over (governed spatial statement).
    pub space: SpaceId,
    /// The hypothesized occupants; `occupants.len()` is the occupant count. An
    /// empty vector is the null (nobody-present) hypothesis.
    pub occupants: Vec<Occupant>,
}

impl Hypothesis {
    /// The **null hypothesis**: nobody present in `space`.
    #[must_use]
    pub fn empty(id: impl Into<String>, space: SpaceId) -> Self {
        Self {
            id: id.into(),
            space,
            occupants: Vec::new(),
        }
    }

    /// Construct and validate a hypothesis. Rejects non-finite occupant
    /// positions and occupant counts above [`MAX_OCCUPANTS`] at the boundary;
    /// never panics on malformed input.
    pub fn new(
        id: impl Into<String>,
        space: SpaceId,
        occupants: Vec<Occupant>,
    ) -> Result<Self, HypothesisError> {
        if occupants.len() > MAX_OCCUPANTS {
            return Err(HypothesisError::TooManyOccupants {
                len: occupants.len(),
                max: MAX_OCCUPANTS,
            });
        }
        for (i, occ) in occupants.iter().enumerate() {
            if !occ.is_finite() {
                return Err(HypothesisError::NonFinitePosition { index: i });
            }
        }
        Ok(Self {
            id: id.into(),
            space,
            occupants,
        })
    }

    /// The hypothesized occupant count.
    #[must_use]
    pub fn occupant_count(&self) -> usize {
        self.occupants.len()
    }

    /// True when this is the null (nobody-present) hypothesis.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.occupants.is_empty()
    }
}

/// Boundary errors from constructing a hypothesis. Malformed input yields one of
/// these; it never panics.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum HypothesisError {
    /// An occupant carried a non-finite coordinate.
    #[error("non-finite occupant position at index {index}")]
    NonFinitePosition {
        /// Offending occupant index.
        index: usize,
    },
    /// More occupants than [`MAX_OCCUPANTS`].
    #[error("too many occupants: {len} exceeds maximum {max}")]
    TooManyOccupants {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
}
