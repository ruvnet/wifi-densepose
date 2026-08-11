//! Twin model, geometry, and construction (ADR-312 §1).
//!
//! **SYNTHETIC / L0.** Every structure here is part of a *simulation scaffold*.
//! A [`RfTwin`] is a persistent, per-deployment *model* of an RF environment; it
//! **predicts** an expected observable, it does not **measure** one. No value it
//! holds or produces is a hardware, `MEASURED`, or accuracy claim (ADR-282 L0,
//! ADR-297 evidence discipline). Coordinates are a coarse metric abstraction,
//! and the propagation model in [`crate::predict`] is a deliberately simple
//! log-distance model, not real RF.
//!
//! Geometry and radio identity are *referenced* from the canonical
//! [`ruview_ontology`] vocabulary ([`SpaceId`], [`SensorId`], [`Container`],
//! [`SemanticProvenance`], [`EvidenceLevel`]) rather than reinvented — the twin
//! annotates the ADR-303 scene with RF state (ADR-312 "annotates and persists").

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ruview_ontology::{Container, EvidenceLevel, SemanticProvenance, SensorId, SpaceId};

/// Upper bound on radio nodes accepted in one deployment. Bounds allocation on
/// untrusted input; construction beyond this is rejected, never truncated.
pub const MAX_NODES: usize = 1024;

/// Upper bound on wall/reflector segments accepted in one deployment.
pub const MAX_WALLS: usize = 4096;

/// A point in metric coordinates (metres), in the deployment's local ENU-style
/// frame. This is a coarse abstraction, not a surveyed position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    /// East / x, metres.
    pub x: f64,
    /// North / y, metres.
    pub y: f64,
    /// Up / z, metres.
    pub z: f64,
}

impl Point3 {
    /// Construct a point.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// True when every coordinate is finite (rejects `NaN`/`inf` at the
    /// boundary).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Euclidean distance to another point, in metres.
    #[must_use]
    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Horizontal-plane endpoint `(x, y)` used for wall-crossing tests.
    #[must_use]
    pub fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

/// A wall or static reflector, modelled (SYNTHETIC) as a floor-plan segment that
/// adds a fixed attenuation to any link whose straight path crosses it. This is
/// a coarse stand-in for the worldgraph `Wall { rf_attenuation_db }` (ADR-303),
/// not a solved electromagnetic obstacle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wall {
    /// Stable, caller-supplied identifier for the wall/reflector.
    pub id: String,
    /// One floor-plan endpoint `(x, y)`, metres.
    pub a: (f64, f64),
    /// The other floor-plan endpoint `(x, y)`, metres.
    pub b: (f64, f64),
    /// Extra one-way attenuation added to a crossing link, in decibels.
    pub attenuation_db: f64,
}

impl Wall {
    /// True when both endpoints and the attenuation are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.a.0.is_finite()
            && self.a.1.is_finite()
            && self.b.0.is_finite()
            && self.b.1.is_finite()
            && self.attenuation_db.is_finite()
    }
}

/// A radio placed at a metric position. Identity and containment are ontology
/// references, not new vocabulary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadioNode {
    /// Ontology sensor identity (ADR-302 authenticated device / ADR-303 sensor).
    pub id: SensorId,
    /// Metric position in the deployment frame.
    pub position: Point3,
    /// Ontology container the radio is placed in (a `Space` or `Zone`).
    pub located_in: Container,
    /// Modelled transmit power, dBm. SYNTHETIC parameter of the forward model.
    pub tx_power_dbm: f64,
}

/// Parameters of the SYNTHETIC log-distance path-loss model (ADR-312 §1). These
/// describe a simple didactic propagation model, **not** a calibrated RF fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropagationParams {
    /// Path-loss exponent `n` (free space ≈ 2.0; indoor typically higher). Must
    /// be strictly positive.
    pub path_loss_exponent: f64,
    /// Reference distance `d0`, metres. Must be strictly positive.
    pub reference_distance_m: f64,
    /// Path loss at the reference distance `PL(d0)`, decibels.
    pub reference_loss_db: f64,
    /// Base shadowing standard deviation, decibels. Its square is the baseline
    /// variance of the expected distribution. Must be non-negative.
    pub shadowing_sigma_db: f64,
}

impl PropagationParams {
    /// A neutral indoor-ish default (`n = 3`, `d0 = 1 m`, `PL(d0) = 40 dB`,
    /// `sigma = 4 dB`). SYNTHETIC; asserts nothing about any real environment.
    #[must_use]
    pub fn default_indoor() -> Self {
        Self {
            path_loss_exponent: 3.0,
            reference_distance_m: 1.0,
            reference_loss_db: 40.0,
            shadowing_sigma_db: 4.0,
        }
    }

    /// Validate the parameters at the boundary.
    fn validate(&self) -> Result<(), TwinError> {
        if !(self.path_loss_exponent.is_finite() && self.path_loss_exponent > 0.0) {
            return Err(TwinError::InvalidParameter {
                what: "path_loss_exponent must be finite and > 0",
            });
        }
        if !(self.reference_distance_m.is_finite() && self.reference_distance_m > 0.0) {
            return Err(TwinError::InvalidParameter {
                what: "reference_distance_m must be finite and > 0",
            });
        }
        if !self.reference_loss_db.is_finite() {
            return Err(TwinError::InvalidParameter {
                what: "reference_loss_db must be finite",
            });
        }
        if !(self.shadowing_sigma_db.is_finite() && self.shadowing_sigma_db >= 0.0) {
            return Err(TwinError::InvalidParameter {
                what: "shadowing_sigma_db must be finite and >= 0",
            });
        }
        Ok(())
    }
}

/// An unordered radio-to-radio link. Endpoints are stored in a canonical order
/// (`a <= b`) so the same physical link has one key regardless of direction.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LinkId {
    /// The lexicographically smaller endpoint (the modelled transmitter).
    pub a: SensorId,
    /// The lexicographically larger endpoint (the modelled receiver).
    pub b: SensorId,
}

impl LinkId {
    /// Build a canonical link key from two endpoints (self-links are rejected by
    /// the caller/twin, not here).
    #[must_use]
    pub fn new(x: SensorId, y: SensorId) -> Self {
        if x <= y {
            Self { a: x, b: y }
        } else {
            Self { a: y, b: x }
        }
    }

    /// True when both endpoints refer to the same node (a degenerate self-link).
    #[must_use]
    pub fn is_self_link(&self) -> bool {
        self.a == self.b
    }
}

/// Recorded multipath / calibration state for one link: an extra variance
/// (dB²) folded into that link's expected distribution. A bounded temporal
/// summary in ADR-312 terms; here a single non-negative scalar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultipathRecord {
    /// The link this record applies to.
    pub link: LinkId,
    /// Extra variance added to the link's expected distribution, dB² (>= 0).
    pub extra_variance_db2: f64,
}

/// Reason a twin version was advanced (ADR-312 §2 versioning). Kept for audit;
/// the twin is always relative to a *named* baseline version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionEvent {
    /// A calibration event under ADR-298 advanced the baseline.
    Calibration,
    /// A deliberate geometry edit advanced the baseline.
    GeometryEdit,
    /// An operator-accepted physical change advanced the baseline.
    AcceptedChange,
}

/// The input description of a deployment, from which a [`RfTwin`] is built.
///
/// Scenes are varied deterministically by [`seed`](Self::seed): there is **no**
/// wall-clock or unseeded randomness anywhere in this crate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeploymentDescription {
    /// Ontology space this deployment annotates (geometry reference, not copy).
    pub space: SpaceId,
    /// Radio nodes and their metric positions.
    pub nodes: Vec<RadioNode>,
    /// Walls / reflectors modelled as attenuating floor-plan segments.
    pub walls: Vec<Wall>,
    /// Propagation model parameters.
    pub params: PropagationParams,
    /// Recorded per-link multipath / calibration variance state.
    #[serde(default)]
    pub multipath: Vec<MultipathRecord>,
    /// Referenced calibration baseline (ADR-298), as a version handle only.
    pub calibration_version: String,
    /// Explicit seed identifying the synthetic scene. Deterministic.
    pub seed: u64,
}

/// A persistent, versioned, per-deployment RF *model* (ADR-312).
///
/// **SYNTHETIC / L0.** The twin's expected distributions and any propagation
/// simulation are a model (ADR-282 L0), never evidence that a physical state is
/// the case. Its load-bearing output is a *delta and its significance against
/// its own modelled variance* (see [`crate::delta`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RfTwin {
    /// Monotonic baseline version. Advanced on calibration / geometry / accepted
    /// change so a delta is always relative to a named baseline.
    pub version: u64,
    /// Ontology space reference this twin annotates.
    pub space: SpaceId,
    /// Radio nodes keyed by ontology sensor id (deterministic ordering).
    pub nodes: BTreeMap<SensorId, RadioNode>,
    /// Walls / reflectors.
    pub walls: Vec<Wall>,
    /// Propagation model parameters.
    pub params: PropagationParams,
    /// Recorded per-link multipath / calibration variance state.
    pub multipath: Vec<MultipathRecord>,
    /// Referenced calibration baseline handle (ADR-298).
    pub calibration_version: String,
    /// Provenance travelling with the twin (ADR-303 §2).
    pub provenance: SemanticProvenance,
    /// Evidence level of everything the twin asserts. Always `L0` (SYNTHETIC):
    /// a twin predicts, it does not measure.
    pub evidence_level: EvidenceLevel,
    /// The synthetic scene seed this twin was built from.
    pub seed: u64,
}

impl RfTwin {
    /// Build a twin from a deployment description, validating every input at the
    /// boundary. Never panics on malformed input; returns a typed [`TwinError`].
    ///
    /// The built twin is always `EvidenceLevel::L0` (SYNTHETIC) and starts at
    /// baseline `version = 1`.
    pub fn build(desc: DeploymentDescription) -> Result<Self, TwinError> {
        if desc.nodes.len() > MAX_NODES {
            return Err(TwinError::TooManyNodes {
                len: desc.nodes.len(),
                max: MAX_NODES,
            });
        }
        if desc.walls.len() > MAX_WALLS {
            return Err(TwinError::TooManyWalls {
                len: desc.walls.len(),
                max: MAX_WALLS,
            });
        }
        desc.params.validate()?;

        let mut nodes: BTreeMap<SensorId, RadioNode> = BTreeMap::new();
        for node in desc.nodes {
            if !node.position.is_finite() {
                return Err(TwinError::NonFiniteCoordinate {
                    node: node.id.as_str().to_string(),
                });
            }
            if !node.tx_power_dbm.is_finite() {
                return Err(TwinError::InvalidParameter {
                    what: "tx_power_dbm must be finite",
                });
            }
            if nodes.insert(node.id.clone(), node.clone()).is_some() {
                return Err(TwinError::DuplicateNode {
                    node: node.id.as_str().to_string(),
                });
            }
        }

        for wall in &desc.walls {
            if !wall.is_finite() {
                return Err(TwinError::NonFiniteCoordinate {
                    node: format!("wall:{}", wall.id),
                });
            }
        }

        for rec in &desc.multipath {
            if !(rec.extra_variance_db2.is_finite() && rec.extra_variance_db2 >= 0.0) {
                return Err(TwinError::InvalidParameter {
                    what: "multipath extra_variance_db2 must be finite and >= 0",
                });
            }
            if rec.link.is_self_link() {
                return Err(TwinError::SelfLink {
                    node: rec.link.a.as_str().to_string(),
                });
            }
        }

        Ok(Self {
            version: 1,
            space: desc.space,
            nodes,
            walls: desc.walls,
            params: desc.params,
            multipath: desc.multipath,
            calibration_version: desc.calibration_version,
            provenance: SemanticProvenance::declared("ruview-twin@0 (SYNTHETIC/L0)"),
            evidence_level: EvidenceLevel::L0,
            seed: desc.seed,
        })
    }

    /// Every unordered link between distinct nodes, in deterministic order.
    #[must_use]
    pub fn links(&self) -> Vec<LinkId> {
        let ids: Vec<&SensorId> = self.nodes.keys().collect();
        let mut out = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                out.push(LinkId::new(ids[i].clone(), ids[j].clone()));
            }
        }
        out
    }

    /// Borrow a node by id.
    #[must_use]
    pub fn node(&self, id: &SensorId) -> Option<&RadioNode> {
        self.nodes.get(id)
    }

    /// The extra variance recorded for a link (0 when none is recorded).
    #[must_use]
    pub fn extra_variance(&self, link: &LinkId) -> f64 {
        self.multipath
            .iter()
            .find(|r| &r.link == link)
            .map_or(0.0, |r| r.extra_variance_db2)
    }

    /// Advance the baseline version on an auditable event and return the new
    /// version. History semantics (ADR-309) live outside this crate; here we
    /// simply move the named baseline forward.
    pub fn advance_version(&mut self, _event: VersionEvent) -> u64 {
        self.version = self.version.saturating_add(1);
        self.version
    }
}

/// Boundary errors from building or operating on a twin. Malformed input yields
/// one of these; it never panics.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TwinError {
    /// A node or wall carried a non-finite coordinate.
    #[error("non-finite coordinate on `{node}`")]
    NonFiniteCoordinate {
        /// Offending node id (or `wall:<id>`).
        node: String,
    },
    /// Two nodes shared the same id.
    #[error("duplicate node id `{node}`")]
    DuplicateNode {
        /// The duplicated node id.
        node: String,
    },
    /// A degenerate link whose endpoints are the same node.
    #[error("self-link on node `{node}`")]
    SelfLink {
        /// The node id.
        node: String,
    },
    /// A model parameter was out of its valid domain.
    #[error("invalid parameter: {what}")]
    InvalidParameter {
        /// Human-readable reason.
        what: &'static str,
    },
    /// More nodes than [`MAX_NODES`].
    #[error("too many nodes: {len} exceeds maximum {max}")]
    TooManyNodes {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// More walls than [`MAX_WALLS`].
    #[error("too many walls: {len} exceeds maximum {max}")]
    TooManyWalls {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
}
