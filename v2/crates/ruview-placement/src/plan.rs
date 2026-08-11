//! Deterministic placement search: floor plan + inventory → recommended plan
//! (ADR-308 §2).
//!
//! **SYNTHETIC / L0.** The search consumes the SYNTHETIC coverage model in
//! [`crate::coverage`] and recommends radio positions that maximise modelled
//! objective observability subject to the inventory count and the scene geometry.
//! It is a *recommendation*, never a guarantee that a room is sensed (ADR-308
//! consequences). Determinism is total: candidate positions come from a seeded
//! grid ([`PlacementParams::seed`]) with **no RNG and no wall-clock**; greedy
//! forward selection then adds the best candidate one radio at a time. Because
//! point observability is a *max over links*, adding a radio can only maintain or
//! raise the score — so the recorded [`PlacementPlan::score_trace`] is
//! monotonically non-decreasing and plateaus at saturation.

use serde::{Deserialize, Serialize};

use ruview_ontology::{EvidenceLevel, SemanticProvenance};
use ruview_twin::Point3;

use crate::coverage::{
    score_placement, Objective, Placement, PlacedRadio, PlacementParams, PlacementScore,
};
use crate::geometry::FloorPlan;
use crate::inventory::Inventory;

/// A recommended placement plan.
///
/// **SYNTHETIC / L0.** Carries the chosen [`Placement`], its [`PlacementScore`],
/// and the monotonic score trace of the greedy search (one entry per radio
/// added). A recommendation, never a sensing claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementPlan {
    /// The recommended radio positions.
    pub placement: Placement,
    /// The score of the recommended placement.
    pub score: PlacementScore,
    /// Total score after each radio was added, in order — non-decreasing.
    pub score_trace: Vec<f64>,
    /// Number of candidate positions the search considered.
    pub candidate_count: usize,
    /// Evidence level of this plan. Always `L0` (SYNTHETIC).
    pub evidence_level: EvidenceLevel,
    /// Provenance travelling with the plan.
    pub provenance: SemanticProvenance,
}

/// One `splitmix64` step mapped to `[0, 1)`. Deterministic; the only source of
/// candidate-grid variation in this crate (varied by an explicit seed, never RNG).
fn splitmix64_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Generate the deterministic candidate-position grid over the plan bounds.
///
/// A seeded sub-step offset varies the grid reproducibly between seeds; the same
/// seed always yields the same candidates. Bounded by `params.max_candidates`.
#[must_use]
pub fn candidate_positions(plan: &FloorPlan, params: &PlacementParams) -> Vec<Point3> {
    if !plan.bounds.is_valid() || !(params.candidate_step_m.is_finite() && params.candidate_step_m > 0.0)
    {
        return Vec::new();
    }
    let step = params.candidate_step_m;
    let mut state = params.seed;
    // Seeded offsets in [0, step) so distinct seeds shift the grid deterministically.
    let ox = splitmix64_unit(&mut state) * step;
    let oy = splitmix64_unit(&mut state) * step;

    let mut out = Vec::new();
    let mut y = plan.bounds.min_y + step / 2.0 + oy;
    // Keep the first row inside the room if the offset pushed it past the far edge.
    if y >= plan.bounds.max_y {
        y = plan.bounds.center().1;
    }
    while y < plan.bounds.max_y {
        let mut x = plan.bounds.min_x + step / 2.0 + ox;
        if x >= plan.bounds.max_x {
            x = plan.bounds.center().0;
        }
        while x < plan.bounds.max_x {
            if out.len() >= params.max_candidates {
                return out;
            }
            out.push(Point3::new(x, y, 1.0));
            x += step;
        }
        y += step;
    }
    if out.is_empty() {
        let (cx, cy) = plan.bounds.center();
        out.push(Point3::new(cx, cy, 1.0));
    }
    out
}

/// Improvement below this counts as no gain (tie), so ties break deterministically
/// to the first (lowest-index) candidate.
const IMPROVEMENT_EPS: f64 = 1e-9;

/// Optimise a placement: greedily add radios from the inventory to maximise
/// modelled objective observability over the floor plan.
///
/// **SYNTHETIC / L0.** Deterministic and never panics. The number of radios is
/// bounded by the inventory; positions come from the seeded candidate grid. The
/// returned [`PlacementPlan::score_trace`] is non-decreasing by construction.
#[must_use]
pub fn optimize(
    plan: &FloorPlan,
    inventory: &Inventory,
    objectives: &[Objective],
    params: &PlacementParams,
) -> PlacementPlan {
    let candidates = candidate_positions(plan, params);
    let mut chosen: Vec<PlacedRadio> = Vec::new();
    let mut trace: Vec<f64> = Vec::new();

    for spec in &inventory.radios {
        let tx = spec.tx_power_dbm;
        let mut best_index: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for (ci, cand) in candidates.iter().enumerate() {
            // Skip a position already chosen (a duplicate adds no link geometry).
            if chosen.iter().any(|r| r.position == *cand) {
                continue;
            }
            let mut trial = chosen.clone();
            trial.push(PlacedRadio::new(*cand, tx));
            let s = score_placement(plan, &Placement { radios: trial }, objectives, params)
                .total_score;
            if s > best_score + IMPROVEMENT_EPS {
                best_score = s;
                best_index = Some(ci);
            }
        }

        match best_index {
            Some(ci) => {
                chosen.push(PlacedRadio::new(candidates[ci], tx));
                trace.push(best_score.max(0.0));
            }
            // No usable candidate remained (e.g. all positions taken); stop.
            None => break,
        }
    }

    let placement = Placement { radios: chosen };
    let score = score_placement(plan, &placement, objectives, params);

    PlacementPlan {
        placement,
        score,
        score_trace: trace,
        candidate_count: candidates.len(),
        evidence_level: EvidenceLevel::L0,
        provenance: SemanticProvenance::declared("ruview-placement@0 (SYNTHETIC/L0)"),
    }
}
