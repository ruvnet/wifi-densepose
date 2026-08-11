//! Coverage / observability scoring of a candidate placement (ADR-305 §2).
//!
//! **SYNTHETIC / L0.** Everything here is a *recommendation derived from a
//! simulation*, never a sensing claim. A candidate placement is scored by
//! consuming the ADR-312 [`RfTwin`] forward model: for a grid of sample points in
//! a target region, a point is "observable" when it lies inside the first Fresnel
//! zone of a well-predicted link ([`crate::fresnel`]). Per target we report an
//! [`Observability`] carrying **both** a modelled score and its uncertainty —
//! never a single confident number for a simulated result (ADR-305 §2). Following
//! ADR-297 rule 1, a target the model cannot evaluate is [`Observability::Unknown`],
//! a first-class value, not an error.

use serde::{Deserialize, Serialize};

use ruview_ontology::{Container, EvidenceLevel, SemanticProvenance, SensorId};
use ruview_twin::{
    predict_link, Container as TwinContainer, DeploymentDescription, ExpectedDistribution, Point3,
    PropagationParams, RadioNode, RfTwin,
};

use crate::fresnel::link_clearance;
use crate::geometry::{sample_points, FloorPlan, PlacementError};

/// A single placed radio in a candidate placement. Position is reused twin
/// geometry ([`Point3`]); identity is assigned when the twin is built.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacedRadio {
    /// Metric position (metres) in the plan frame.
    pub position: Point3,
    /// Modelled transmit power, dBm.
    pub tx_power_dbm: f64,
}

impl PlacedRadio {
    /// Construct a placed radio.
    #[must_use]
    pub const fn new(position: Point3, tx_power_dbm: f64) -> Self {
        Self { position, tx_power_dbm }
    }
}

/// A candidate set of radio positions to score.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Placement {
    /// The placed radios.
    pub radios: Vec<PlacedRadio>,
}

impl Placement {
    /// An empty placement.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of placed radios.
    #[must_use]
    pub fn len(&self) -> usize {
        self.radios.len()
    }

    /// True when no radios are placed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.radios.is_empty()
    }
}

/// The phenomenon a sensing objective requires. Higher-order phenomena demand
/// stronger Fresnel clearance to count a point as observable ([`Self::demand`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phenomenon {
    /// Coarse presence / occupancy.
    Presence,
    /// Vital-sign sensing (stricter clearance demand).
    Vitals,
    /// Pose estimation (strictest clearance demand).
    Pose,
}

impl Phenomenon {
    /// Multiplier applied to the base coverage threshold: a stricter phenomenon
    /// needs a higher modelled sensing value at a point to count it as covered.
    #[must_use]
    pub fn demand(self) -> f64 {
        match self {
            Phenomenon::Presence => 1.0,
            Phenomenon::Vitals => 1.6,
            Phenomenon::Pose => 2.2,
        }
    }
}

/// A sensing objective: a phenomenon that must be observable in a target region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    /// The space or zone that must be observable.
    pub target: Container,
    /// The phenomenon required there.
    pub phenomenon: Phenomenon,
}

impl Objective {
    /// Construct an objective.
    #[must_use]
    pub fn new(target: Container, phenomenon: Phenomenon) -> Self {
        Self { target, phenomenon }
    }
}

/// Parameters of the SYNTHETIC coverage model. All defaults are didactic; they
/// assert nothing about any real environment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementParams {
    /// Modelled wavelength, metres (default ≈ 2.4 GHz).
    pub wavelength_m: f64,
    /// Modelled RSSI (dBm) at/above which a link has full sensing quality.
    pub good_rssi_dbm: f64,
    /// Modelled RSSI (dBm) at/below which a link has zero sensing quality.
    pub floor_rssi_dbm: f64,
    /// Point sensing value (`clearance × quality`) at/above which a sample point
    /// counts as covered, before the per-phenomenon demand multiplier.
    pub coverage_threshold: f64,
    /// Covered-fraction below which a target is flagged as a blind spot.
    pub blind_spot_fraction: f64,
    /// Sample-grid step, metres.
    pub grid_step_m: f64,
    /// Candidate-grid step, metres (used by the search).
    pub candidate_step_m: f64,
    /// Reference variance (dB²) mapping modelled link variance to `[0, 1]`
    /// uncertainty.
    pub uncertainty_variance_ref_db2: f64,
    /// Cap on sample points per target (bounded allocation).
    pub max_sample_points: usize,
    /// Cap on generated candidate positions (bounded search).
    pub max_candidates: usize,
    /// Explicit seed for deterministic candidate generation. No RNG anywhere.
    pub seed: u64,
    /// Twin propagation-model parameters.
    pub propagation: PropagationParams,
}

impl PlacementParams {
    /// A neutral SYNTHETIC default set.
    #[must_use]
    pub fn default_synthetic() -> Self {
        Self {
            wavelength_m: 0.1249,
            good_rssi_dbm: -50.0,
            floor_rssi_dbm: -85.0,
            coverage_threshold: 0.10,
            blind_spot_fraction: 0.15,
            grid_step_m: 0.5,
            candidate_step_m: 1.0,
            uncertainty_variance_ref_db2: 64.0,
            max_sample_points: 4096,
            max_candidates: 512,
            seed: 0,
            propagation: PropagationParams::default_indoor(),
        }
    }

    /// Validate the parameters at the boundary.
    pub fn validate(&self) -> Result<(), PlacementError> {
        let finite_pos = |v: f64| v.is_finite() && v > 0.0;
        if !finite_pos(self.wavelength_m) {
            return Err(PlacementError::InvalidParameter { what: "wavelength_m must be > 0" });
        }
        if !finite_pos(self.grid_step_m) {
            return Err(PlacementError::InvalidParameter { what: "grid_step_m must be > 0" });
        }
        if !finite_pos(self.candidate_step_m) {
            return Err(PlacementError::InvalidParameter { what: "candidate_step_m must be > 0" });
        }
        if !(self.good_rssi_dbm.is_finite()
            && self.floor_rssi_dbm.is_finite()
            && self.good_rssi_dbm > self.floor_rssi_dbm)
        {
            return Err(PlacementError::InvalidParameter {
                what: "good_rssi_dbm must be finite and > floor_rssi_dbm",
            });
        }
        if !(self.coverage_threshold.is_finite() && self.coverage_threshold > 0.0) {
            return Err(PlacementError::InvalidParameter { what: "coverage_threshold must be > 0" });
        }
        if !(self.uncertainty_variance_ref_db2.is_finite() && self.uncertainty_variance_ref_db2 > 0.0)
        {
            return Err(PlacementError::InvalidParameter {
                what: "uncertainty_variance_ref_db2 must be > 0",
            });
        }
        if self.max_sample_points == 0 || self.max_candidates == 0 {
            return Err(PlacementError::InvalidParameter {
                what: "sample/candidate caps must be > 0",
            });
        }
        // Mirror the twin's propagation-parameter domain (its own validator is
        // private); the twin re-checks these on build regardless.
        let p = &self.propagation;
        if !(p.path_loss_exponent.is_finite() && p.path_loss_exponent > 0.0)
            || !(p.reference_distance_m.is_finite() && p.reference_distance_m > 0.0)
            || !p.reference_loss_db.is_finite()
            || !(p.shadowing_sigma_db.is_finite() && p.shadowing_sigma_db >= 0.0)
        {
            return Err(PlacementError::InvalidParameter { what: "invalid propagation params" });
        }
        Ok(())
    }
}

/// Why a target's observability is unknown. UNKNOWN is a first-class output
/// (ADR-297 rule 1), not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityUnknown {
    /// The objective's target is not present in this floor plan.
    TargetNotInPlan,
    /// The target region produced no sample points (degenerate geometry).
    EmptyRegion,
    /// Fewer than two placed radios, so the twin has no links to evaluate.
    NoLinks,
    /// The modelled computation produced a non-finite value.
    NonFinite,
}

/// Modelled observability of a target region.
///
/// **SYNTHETIC / L0.** A model-relative statement carrying its own uncertainty,
/// never evidence that a region *is* being sensed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Observability {
    /// A modelled observability `score` in `[0, 1]` and its `uncertainty` in
    /// `[0, 1]`. Both are reported; neither is a confident single number.
    Known {
        /// Mean modelled sensing value over the region, `[0, 1]`.
        score: f64,
        /// Modelled uncertainty of that score, `[0, 1]` (higher = less certain).
        uncertainty: f64,
    },
    /// The model cannot evaluate this target; carries a first-class reason.
    Unknown {
        /// Why it is unknown.
        reason: ObservabilityUnknown,
    },
}

impl Observability {
    /// Borrow `(score, uncertainty)` when known.
    #[must_use]
    pub fn known(&self) -> Option<(f64, f64)> {
        match self {
            Observability::Known { score, uncertainty } => Some((*score, *uncertainty)),
            Observability::Unknown { .. } => None,
        }
    }
}

/// Per-target coverage detail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetCoverage {
    /// The target region.
    pub target: Container,
    /// The phenomenon required there.
    pub phenomenon: Phenomenon,
    /// Modelled observability (score + uncertainty, or UNKNOWN).
    pub observability: Observability,
    /// Fraction of sample points that met the (phenomenon-scaled) coverage
    /// threshold, `[0, 1]`.
    pub covered_fraction: f64,
    /// Number of sample points evaluated.
    pub sample_count: usize,
    /// True when this target is flagged as a blind spot.
    pub blind_spot: bool,
}

/// The score of a candidate placement across all objectives.
///
/// **SYNTHETIC / L0.** A recommendation, never a sensing claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementScore {
    /// Sample-weighted mean of the per-target modelled scores (Unknown targets
    /// contribute nothing), `[0, 1]`.
    pub total_score: f64,
    /// Per-target coverage detail.
    pub per_target: Vec<TargetCoverage>,
    /// Targets flagged as blind spots (a subset of `per_target`, by container).
    pub blind_spots: Vec<Container>,
    /// Number of radios in the scored placement.
    pub node_count: usize,
    /// Evidence level of this score. Always `L0` (SYNTHETIC).
    pub evidence_level: EvidenceLevel,
    /// Provenance travelling with the score.
    pub provenance: SemanticProvenance,
}

impl PlacementScore {
    /// True when at least one target is flagged as a blind spot.
    #[must_use]
    pub fn has_blind_spot(&self) -> bool {
        !self.blind_spots.is_empty()
    }
}

/// A precomputed link with the geometry and modelled quality the coverage model
/// needs. Internal.
struct LinkGeom {
    a: (f64, f64),
    b: (f64, f64),
    quality: f64,
    variance: f64,
}

/// Map a modelled RSSI mean to a sensing quality in `[0, 1]`.
fn quality_from_mean(mean: f64, params: &PlacementParams) -> f64 {
    if !mean.is_finite() {
        return 0.0;
    }
    let span = params.good_rssi_dbm - params.floor_rssi_dbm;
    ((mean - params.floor_rssi_dbm) / span).clamp(0.0, 1.0)
}

/// Build the twin from a placement and derive per-link geometry + quality. An
/// empty result means "no evaluable links" (fewer than two finite nodes, or the
/// twin rejected the scene) — surfaced as UNKNOWN by callers, never a panic.
fn build_links(plan: &FloorPlan, placement: &Placement, params: &PlacementParams) -> Vec<LinkGeom> {
    let space = plan.space.clone();
    let mut nodes: Vec<RadioNode> = Vec::new();
    for (i, radio) in placement.radios.iter().enumerate() {
        if !radio.position.is_finite() || !radio.tx_power_dbm.is_finite() {
            continue;
        }
        let id = match SensorId::new(format!("place-{i}")) {
            Ok(id) => id,
            Err(_) => continue,
        };
        nodes.push(RadioNode {
            id,
            position: radio.position,
            located_in: TwinContainer::Space { id: space.clone() },
            tx_power_dbm: radio.tx_power_dbm,
        });
    }
    if nodes.len() < 2 {
        return Vec::new();
    }
    let desc = DeploymentDescription {
        space,
        nodes,
        walls: plan.walls.clone(),
        params: params.propagation,
        multipath: Vec::new(),
        calibration_version: "synthetic-placement".to_string(),
        seed: 0,
    };
    let twin = match RfTwin::build(desc) {
        Ok(twin) => twin,
        Err(_) => return Vec::new(),
    };
    let mut links = Vec::new();
    for link in twin.links() {
        if let ExpectedDistribution::Known { mean, variance, .. } = predict_link(&twin, &link) {
            let (Some(na), Some(nb)) = (twin.node(&link.a), twin.node(&link.b)) else {
                continue;
            };
            links.push(LinkGeom {
                a: na.position.xy(),
                b: nb.position.xy(),
                quality: quality_from_mean(mean, params),
                variance,
            });
        }
    }
    links
}

/// The best modelled sensing value at a point and the variance of the link that
/// achieved it. Sensing is `max over links of clearance × quality`.
fn point_sensing(links: &[LinkGeom], p: (f64, f64), params: &PlacementParams) -> (f64, f64) {
    let mut best = 0.0_f64;
    let mut best_var = params.uncertainty_variance_ref_db2;
    for link in links {
        let s = link_clearance(link.a, link.b, p, params.wavelength_m) * link.quality;
        if s > best {
            best = s;
            best_var = link.variance;
        }
    }
    (best, best_var)
}

/// Score one target region against the precomputed links.
fn score_target(
    plan: &FloorPlan,
    links: &[LinkGeom],
    objective: &Objective,
    params: &PlacementParams,
) -> TargetCoverage {
    let phenomenon = objective.phenomenon;
    let target = objective.target.clone();

    let region = match plan.region_for(&target) {
        Some(r) => r,
        None => {
            return TargetCoverage {
                target,
                phenomenon,
                observability: Observability::Unknown {
                    reason: ObservabilityUnknown::TargetNotInPlan,
                },
                covered_fraction: 0.0,
                sample_count: 0,
                blind_spot: false,
            };
        }
    };

    let samples = sample_points(&region, params.grid_step_m, params.max_sample_points);
    if samples.is_empty() {
        return TargetCoverage {
            target,
            phenomenon,
            observability: Observability::Unknown { reason: ObservabilityUnknown::EmptyRegion },
            covered_fraction: 0.0,
            sample_count: 0,
            blind_spot: false,
        };
    }

    if links.is_empty() {
        // No links to evaluate: genuinely unknown, and a blind spot by definition.
        return TargetCoverage {
            target,
            phenomenon,
            observability: Observability::Unknown { reason: ObservabilityUnknown::NoLinks },
            covered_fraction: 0.0,
            sample_count: samples.len(),
            blind_spot: true,
        };
    }

    let threshold = (params.coverage_threshold * phenomenon.demand()).clamp(f64::MIN_POSITIVE, 1.0);
    let n = samples.len() as f64;
    let mut score_sum = 0.0_f64;
    let mut unc_sum = 0.0_f64;
    let mut covered = 0usize;

    for &p in &samples {
        let (sensing, var) = point_sensing(links, p, params);
        score_sum += sensing;
        if sensing >= threshold {
            covered += 1;
            unc_sum += (var / params.uncertainty_variance_ref_db2).clamp(0.0, 1.0);
        } else {
            // An uncovered point is maximally uncertain.
            unc_sum += 1.0;
        }
    }

    let score = (score_sum / n).clamp(0.0, 1.0);
    let uncertainty = (unc_sum / n).clamp(0.0, 1.0);
    let covered_fraction = covered as f64 / n;
    let blind_spot = covered_fraction < params.blind_spot_fraction;

    let observability = if score.is_finite() && uncertainty.is_finite() {
        Observability::Known { score, uncertainty }
    } else {
        Observability::Unknown { reason: ObservabilityUnknown::NonFinite }
    };

    TargetCoverage {
        target,
        phenomenon,
        observability,
        covered_fraction,
        sample_count: samples.len(),
        blind_spot,
    }
}

/// Provenance stamped on every placement result (SYNTHETIC/L0).
fn placement_provenance() -> SemanticProvenance {
    SemanticProvenance::declared("ruview-placement@0 (SYNTHETIC/L0)")
}

/// Score a candidate placement against a set of objectives over a floor plan.
///
/// **SYNTHETIC / L0.** Never panics: malformed geometry or an out-of-plan target
/// yields [`Observability::Unknown`] for that target, not an error. The returned
/// score is `EvidenceLevel::L0` — a recommendation, never a sensing claim.
#[must_use]
pub fn score_placement(
    plan: &FloorPlan,
    placement: &Placement,
    objectives: &[Objective],
    params: &PlacementParams,
) -> PlacementScore {
    let links = build_links(plan, placement, params);

    let mut per_target = Vec::with_capacity(objectives.len());
    let mut blind_spots = Vec::new();
    let mut weighted_score = 0.0_f64;
    let mut weight = 0.0_f64;

    for objective in objectives {
        let cov = score_target(plan, &links, objective, params);
        if let Observability::Known { score, .. } = cov.observability {
            let w = cov.sample_count as f64;
            weighted_score += score * w;
            weight += w;
        }
        if cov.blind_spot {
            blind_spots.push(cov.target.clone());
        }
        per_target.push(cov);
    }

    let total_score = if weight > 0.0 { weighted_score / weight } else { 0.0 };

    PlacementScore {
        total_score,
        per_target,
        blind_spots,
        node_count: placement.radios.len(),
        evidence_level: EvidenceLevel::L0,
        provenance: placement_provenance(),
    }
}

/// A placement paired with its computed score and its position in the input list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedPlacement {
    /// Index of this placement in the input slice.
    pub index: usize,
    /// The placement.
    pub placement: Placement,
    /// Its score.
    pub score: PlacementScore,
}

/// Rank candidate placements by modelled total score, highest first.
///
/// Deterministic: ties break by ascending input index, so the same inputs always
/// yield the same ranking. SYNTHETIC/L0.
#[must_use]
pub fn rank_placements(
    plan: &FloorPlan,
    placements: &[Placement],
    objectives: &[Objective],
    params: &PlacementParams,
) -> Vec<RankedPlacement> {
    let mut ranked: Vec<RankedPlacement> = placements
        .iter()
        .enumerate()
        .map(|(index, placement)| RankedPlacement {
            index,
            placement: placement.clone(),
            score: score_placement(plan, placement, objectives, params),
        })
        .collect();
    ranked.sort_by(|x, y| {
        y.score
            .total_score
            .partial_cmp(&x.score.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.index.cmp(&y.index))
    });
    ranked
}
