//! Counterfactual scoring and best-explanation selection (ADR-310 §2, §3).
//!
//! **SYNTHETIC / L0 — a research-forward generative-scoring scaffold, not a
//! measurement system.** This module scores a small set of scene
//! [`Hypothesis`](crate::Hypothesis) against an observed link-measurement set,
//! using the ADR-312 [`RfTwin`] as the generative forward model. It is a
//! *consumer* of the twin, not a second simulator: the twin supplies the
//! baseline expected distribution per link (geometry + propagation), and this
//! layer applies a **documented SYNTHETIC occupant-attenuation model** on top —
//! a hypothesized occupant attenuates any link whose line of sight passes near
//! it. Nothing here is a hardware, `MEASURED`, or accuracy claim; this crate
//! asserts **no** discrimination-accuracy number (ADR-310 evidence discipline).
//!
//! ## Likelihood (documented SYNTHETIC)
//!
//! For each observed link with a known base distribution `N(m0, v0)` from the
//! twin, the occupant-adjusted distribution under a hypothesis is
//! `N(m0 - att, v0 + extra)`, where `att`/`extra` accumulate a linear-falloff
//! body effect over occupants that block the link. The per-link explanatory
//! score is the Gaussian log-likelihood of the observed value under that
//! adjusted distribution; a hypothesis's score is the sum over evaluated links.
//! This is a deliberately simple, deterministic model — clearly a scaffold, not
//! real RF.
//!
//! ## UNKNOWN is first-class (ADR-297 rule 1, ADR-310 §3)
//!
//! The layer never forces a label. It returns [`BestExplanation::Unknown`] when
//! the top two hypotheses are near-indistinguishable (margin below threshold),
//! when **no** hypothesis explains the observation well (best mean per-link
//! log-likelihood below a floor — the observation is outside what the twin can
//! account for, routed to the ADR-299 UNKNOWN verdict rather than a forced
//! occupancy label), when no hypotheses are supplied, or when no observed link
//! is evaluable against the twin.

use serde::{Deserialize, Serialize};

use ruview_ontology::{EvidenceLevel, SemanticProvenance};
use ruview_twin::{
    ExpectedDistribution, LinkId, LinkObservation, ObservationSet, RfTwin,
};

use crate::hypothesis::Hypothesis;

/// `2π`, used in the Gaussian log-likelihood normaliser.
const TAU: f64 = std::f64::consts::TAU;

/// The SYNTHETIC occupant → link effect. A hypothesized occupant within
/// [`body_radius_m`](Self::body_radius_m) of a link's line of sight attenuates
/// it (and adds variance), with a linear falloff to zero at the radius edge.
///
/// **SYNTHETIC.** These are model parameters of a didactic occupant model, not a
/// calibrated RF body-shadowing fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccupantModel {
    /// Perpendicular distance (metres) within which an occupant blocks a link.
    /// Must be finite and `> 0`.
    pub body_radius_m: f64,
    /// Maximum attenuation (dB) added to a directly-blocked link (falloff → 0 at
    /// the radius edge). Must be finite and `>= 0`.
    pub attenuation_db: f64,
    /// Maximum extra variance (dB²) added to a directly-blocked link's expected
    /// distribution. Must be finite and `>= 0`.
    pub extra_variance_db2: f64,
}

impl OccupantModel {
    /// A neutral SYNTHETIC default: `0.9 m` body radius, `6 dB` peak
    /// attenuation, `4 dB²` peak extra variance. Asserts nothing about any real
    /// body or environment.
    #[must_use]
    pub fn default_body() -> Self {
        Self {
            body_radius_m: 0.9,
            attenuation_db: 6.0,
            extra_variance_db2: 4.0,
        }
    }

    /// True when every parameter is in its valid domain.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.body_radius_m.is_finite()
            && self.body_radius_m > 0.0
            && self.attenuation_db.is_finite()
            && self.attenuation_db >= 0.0
            && self.extra_variance_db2.is_finite()
            && self.extra_variance_db2 >= 0.0
    }
}

impl Default for OccupantModel {
    fn default() -> Self {
        Self::default_body()
    }
}

/// Thresholds that route a scored hypothesis set to a best explanation or to a
/// first-class UNKNOWN verdict (ADR-310 §3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Minimum log-likelihood-ratio margin (nats) between the top two
    /// hypotheses required to declare a best explanation; below this the two are
    /// near-indistinguishable and the result is UNKNOWN. Must be finite `>= 0`.
    pub min_margin_nats: f64,
    /// Minimum best *mean per-link* log-likelihood (nats) required for any
    /// hypothesis to count as explaining the observation; below this the
    /// observation is outside what the twin can account for and the result is
    /// UNKNOWN (routed to the ADR-299 verdict). Must be finite.
    pub min_mean_log_likelihood: f64,
}

impl ScoringConfig {
    /// Neutral SYNTHETIC defaults: a `0.5`-nat margin and a `-10.0`-nat mean
    /// per-link floor. These are model gates, not calibrated error rates.
    #[must_use]
    pub fn default_gates() -> Self {
        Self {
            min_margin_nats: 0.5,
            min_mean_log_likelihood: -10.0,
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self::default_gates()
    }
}

/// The explanatory score of one hypothesis against the observed measurements.
///
/// **SYNTHETIC / L0.** `log_likelihood` is a model-relative explanatory score
/// (summed Gaussian log-likelihood under the twin + occupant model), never a
/// detection or accuracy claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisScore {
    /// The scored hypothesis.
    pub hypothesis: Hypothesis,
    /// Summed per-link Gaussian log-likelihood over evaluated links (nats).
    /// Higher ⇒ this hypothesis better explains the observation.
    pub log_likelihood: f64,
    /// Number of observed links evaluated against a known twin distribution.
    pub evaluated_links: usize,
    /// Number of observed links that could not be evaluated (unknown under the
    /// twin, or a non-finite / undefined likelihood); first-class, never an
    /// error.
    pub unknown_links: usize,
}

impl HypothesisScore {
    /// Mean per-link log-likelihood, or `None` when no link was evaluable.
    #[must_use]
    pub fn mean_log_likelihood(&self) -> Option<f64> {
        (self.evaluated_links > 0).then(|| self.log_likelihood / self.evaluated_links as f64)
    }
}

/// The best-explanation outcome. Either one hypothesis explains the observation
/// with a positive margin, or the result is a first-class UNKNOWN.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum BestExplanation {
    /// One hypothesis best explains the observation, by `margin` nats over the
    /// runner-up (`f64::INFINITY` when it is the only hypothesis).
    Explained {
        /// The winning hypothesis's stable id.
        hypothesis_id: String,
        /// Its hypothesized occupant count.
        occupant_count: usize,
        /// Log-likelihood-ratio margin (nats) over the runner-up.
        margin: f64,
    },
    /// No confident best explanation; carries a first-class reason (ADR-297
    /// rule 1, ADR-310 §3).
    Unknown {
        /// Why the result is UNKNOWN.
        reason: UnknownReason,
    },
}

/// Why a counterfactual evaluation resolved to UNKNOWN instead of a confident
/// best explanation. UNKNOWN is a value, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum UnknownReason {
    /// No hypotheses were supplied to evaluate.
    NoHypotheses,
    /// No observed link was evaluable against the twin (nothing to score).
    NoEvaluableLinks,
    /// The top two hypotheses are near-indistinguishable: the margin fell below
    /// the configured threshold.
    NearIndistinguishable {
        /// The observed margin (nats).
        margin: f64,
        /// The configured minimum margin (nats).
        threshold: f64,
    },
    /// No hypothesis explains the observation well: the best mean per-link
    /// log-likelihood fell below the configured floor. The observation is
    /// outside what the twin can account for (routes to the ADR-299 verdict).
    NoHypothesisExplains {
        /// The best hypothesis's mean per-link log-likelihood (nats).
        best_mean_log_likelihood: f64,
        /// The configured floor (nats).
        floor: f64,
    },
}

/// The typed result of a counterfactual evaluation.
///
/// **SYNTHETIC / L0.** Every score and the verdict are model-relative and
/// inherit the twin's `L0` evidence level; nothing here is a camera-grade or
/// `MEASURED` claim (CLAUDE.md honesty rule, ADR-310 evidence discipline).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// Every hypothesis's score, ranked best-first: descending by
    /// `log_likelihood`, ties broken by ascending hypothesis id (deterministic).
    pub ranked: Vec<HypothesisScore>,
    /// The best explanation, or a first-class UNKNOWN.
    pub best: BestExplanation,
    /// Evidence level of the verdict — inherits the weakest input; the twin is
    /// `L0` (SYNTHETIC), so a counterfactual verdict is always `L0`.
    pub evidence_level: EvidenceLevel,
    /// Provenance travelling with the verdict.
    pub provenance: SemanticProvenance,
}

impl CounterfactualResult {
    /// The top-ranked hypothesis score, if any hypotheses were scored.
    #[must_use]
    pub fn top(&self) -> Option<&HypothesisScore> {
        self.ranked.first()
    }
}

/// Perpendicular distance from point `p` to segment `a`–`b` (metres). Clamps to
/// the endpoints for a point beyond the segment; handles a degenerate
/// (zero-length) segment as point distance. Deterministic and allocation-free.
fn point_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (px, py) = p;
    let (ax, ay) = a;
    let (bx, by) = b;
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 <= f64::EPSILON {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let cx = ax + t * dx;
    let cy = ay + t * dy;
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// Accumulated occupant effect (`attenuation_db`, `extra_variance_db2`) on one
/// link from a hypothesis's occupants under the model. Non-finite occupant
/// positions and links whose endpoints are absent from the twin contribute
/// nothing (defensive; never panics).
fn occupant_effect(
    twin: &RfTwin,
    link: &LinkId,
    hypothesis: &Hypothesis,
    model: &OccupantModel,
) -> (f64, f64) {
    let (a, b) = match (twin.node(&link.a), twin.node(&link.b)) {
        (Some(a), Some(b)) => (a.position.xy(), b.position.xy()),
        _ => return (0.0, 0.0),
    };
    let mut att = 0.0;
    let mut extra = 0.0;
    for occ in &hypothesis.occupants {
        if !occ.is_finite() {
            continue;
        }
        let d = point_segment_distance(occ.xy(), a, b);
        if d < model.body_radius_m {
            // Linear falloff to zero at the radius edge (SYNTHETIC).
            let shield = 1.0 - d / model.body_radius_m;
            att += model.attenuation_db * shield;
            extra += model.extra_variance_db2 * shield;
        }
    }
    (att, extra)
}

/// The occupant-adjusted expected distribution `(mean, variance)` for a link
/// under a hypothesis, or `None` when the twin cannot predict the link or the
/// adjusted distribution is degenerate (non-positive / non-finite variance).
#[must_use]
fn adjusted_distribution(
    twin: &RfTwin,
    link: &LinkId,
    hypothesis: &Hypothesis,
    model: &OccupantModel,
) -> Option<(f64, f64)> {
    let (m0, v0) = match twin.predict(link) {
        ExpectedDistribution::Known { mean, variance, .. } => (mean, variance),
        ExpectedDistribution::Unknown { .. } => return None,
    };
    let (att, extra) = occupant_effect(twin, link, hypothesis, model);
    let mean = m0 - att;
    let variance = v0 + extra;
    if mean.is_finite() && variance.is_finite() && variance > 0.0 {
        Some((mean, variance))
    } else {
        None
    }
}

/// Gaussian log-likelihood of `observed` under `N(mean, variance)` (nats).
/// `variance` is required `> 0` by the caller; returns `None` on a non-finite
/// result rather than propagating a poisoned score.
#[must_use]
fn gaussian_log_likelihood(observed: f64, mean: f64, variance: f64) -> Option<f64> {
    let dev = observed - mean;
    let ll = -0.5 * (TAU * variance).ln() - (dev * dev) / (2.0 * variance);
    ll.is_finite().then_some(ll)
}

/// Score one hypothesis against the observed set under the twin and occupant
/// model. Deterministic; allocation is bounded by the observation count.
#[must_use]
pub fn score_hypothesis(
    twin: &RfTwin,
    observed: &ObservationSet,
    hypothesis: &Hypothesis,
    model: &OccupantModel,
) -> HypothesisScore {
    let mut log_likelihood = 0.0;
    let mut evaluated_links = 0usize;
    let mut unknown_links = 0usize;

    for LinkObservation { link, value } in &observed.observations {
        match adjusted_distribution(twin, link, hypothesis, model) {
            Some((mean, variance)) => match gaussian_log_likelihood(*value, mean, variance) {
                Some(ll) => {
                    log_likelihood += ll;
                    evaluated_links += 1;
                }
                None => unknown_links += 1,
            },
            None => unknown_links += 1,
        }
    }

    HypothesisScore {
        hypothesis: hypothesis.clone(),
        log_likelihood,
        evaluated_links,
        unknown_links,
    }
}

/// Evaluate counterfactual hypotheses against an observed measurement set using
/// the [`OccupantModel::default_body`] and [`ScoringConfig::default_gates`].
#[must_use]
pub fn evaluate(
    twin: &RfTwin,
    observed: &ObservationSet,
    hypotheses: &[Hypothesis],
) -> CounterfactualResult {
    evaluate_with(
        twin,
        observed,
        hypotheses,
        &OccupantModel::default_body(),
        &ScoringConfig::default_gates(),
    )
}

/// Evaluate counterfactual hypotheses with explicit model and scoring gates.
///
/// Scores every hypothesis, ranks them best-first (deterministically), and
/// selects a best explanation with a margin — or a first-class UNKNOWN when the
/// top two are near-indistinguishable, when no hypothesis explains the
/// observation, when no hypotheses are supplied, or when no link is evaluable.
///
/// **SYNTHETIC / L0.** The verdict inherits the twin's `L0` evidence level and
/// is never a camera-grade or `MEASURED` claim.
#[must_use]
pub fn evaluate_with(
    twin: &RfTwin,
    observed: &ObservationSet,
    hypotheses: &[Hypothesis],
    model: &OccupantModel,
    config: &ScoringConfig,
) -> CounterfactualResult {
    let provenance = SemanticProvenance::declared("ruview-counterfactual@0 (SYNTHETIC/L0)");
    let evidence_level = EvidenceLevel::L0;

    // Score every hypothesis, then rank deterministically (bounded input).
    let capped = hypotheses.len().min(crate::hypothesis::MAX_HYPOTHESES);
    let mut ranked: Vec<HypothesisScore> = hypotheses[..capped]
        .iter()
        .map(|h| score_hypothesis(twin, observed, h, model))
        .collect();
    // Descending by log-likelihood; ties broken by ascending hypothesis id.
    ranked.sort_by(|a, b| {
        b.log_likelihood
            .total_cmp(&a.log_likelihood)
            .then_with(|| a.hypothesis.id.cmp(&b.hypothesis.id))
    });

    let best = decide(&ranked, config);

    CounterfactualResult {
        ranked,
        best,
        evidence_level,
        provenance,
    }
}

/// Select the best explanation (or UNKNOWN) from a ranked score list.
fn decide(ranked: &[HypothesisScore], config: &ScoringConfig) -> BestExplanation {
    let Some(top) = ranked.first() else {
        return BestExplanation::Unknown {
            reason: UnknownReason::NoHypotheses,
        };
    };

    // Nothing was evaluable against the twin ⇒ nothing to explain.
    let Some(mean_ll) = top.mean_log_likelihood() else {
        return BestExplanation::Unknown {
            reason: UnknownReason::NoEvaluableLinks,
        };
    };

    // No hypothesis explains the observation well ⇒ ADR-299 UNKNOWN verdict.
    if mean_ll < config.min_mean_log_likelihood {
        return BestExplanation::Unknown {
            reason: UnknownReason::NoHypothesisExplains {
                best_mean_log_likelihood: mean_ll,
                floor: config.min_mean_log_likelihood,
            },
        };
    }

    // Margin over the runner-up (infinite when the top is the only hypothesis).
    let margin = match ranked.get(1) {
        Some(second) => top.log_likelihood - second.log_likelihood,
        None => f64::INFINITY,
    };

    if margin < config.min_margin_nats {
        return BestExplanation::Unknown {
            reason: UnknownReason::NearIndistinguishable {
                margin,
                threshold: config.min_margin_nats,
            },
        };
    }

    BestExplanation::Explained {
        hypothesis_id: top.hypothesis.id.clone(),
        occupant_count: top.hypothesis.occupant_count(),
        margin,
    }
}

/// Synthesize the observation set a scene would produce under the twin and
/// occupant model — the occupant-adjusted mean of every twin link with a known
/// base distribution. The empty hypothesis reproduces the twin's own
/// predictions (the zero-occupant reference); a populated hypothesis attenuates
/// the blocked links.
///
/// **SYNTHETIC.** This is a deterministic simulation fixture for exploring and
/// testing counterfactual scoring — not a measurement and not a sampler (no
/// randomness). Links the twin cannot predict are omitted.
#[must_use]
pub fn synthesize_observations(
    twin: &RfTwin,
    hypothesis: &Hypothesis,
    model: &OccupantModel,
) -> ObservationSet {
    let mut observed = ObservationSet::new();
    for link in twin.links() {
        if let Some((mean, _variance)) = adjusted_distribution(twin, &link, hypothesis, model) {
            observed = observed.with(link, mean);
        }
    }
    observed
}
