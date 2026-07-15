//! Canonical pose-metric core (ADR-155 §Tier-1.1) — the single source of truth
//! for PCK and OKS, **available without the `tch-backend` feature**.
//!
//! # Why this module exists (ADR-155 Milestone-1, §8 backlog resolution)
//!
//! The full [`crate::metrics`] module is gated behind `tch-backend` (libtorch
//! FFI) because it also hosts the trainer accumulators, min-cut matchers, and
//! ndarray/petgraph machinery. But the *metric definition itself*
//! ([`pck_canonical`], [`oks_canonical`], [`canonical_torso_size`]) depends only
//! on `ndarray` — no tch. Hoisting those four functions here makes the canonical
//! definition reachable from the workspace test gate
//! (`cargo test --no-default-features`) so the integration test
//! (`tests/test_metrics.rs`) can validate the **production** function against
//! hand-computed fixtures, instead of testing an independent reimplementation
//! that could be wrong the same way (the §8 "reference kernels" finding).
//!
//! [`crate::metrics`] re-exports every item here, so all existing call sites and
//! the tch-gated trainer path are unchanged: there is still exactly **one**
//! implementation of each metric, now in one *un-gated* place.
//!
//! # CANONICAL METRIC (the only definitions valid for a *reported* number)
//!
//! - [`pck_canonical`] — **PCK\@k, torso-normalized.** A keypoint `j` is correct
//!   iff `‖pred_j − gt_j‖₂ ≤ k · torso`, where
//!   `torso = ‖left_hip(11) − right_hip(12)‖₂` in the keypoint coordinate space,
//!   with a bounding-box-diagonal fallback when the hips are not both visible.
//!   **Zero visible joints ⇒ `(0, 0, 0.0)`** — no evidence scores 0, never 1.
//! - [`oks_canonical`] — **COCO OKS** with `s = sqrt(area)` derived from the GT
//!   pose extent (never a fixed `1.0`); a degenerate pose returns 0.0.
//!
//! # No mock data
//!
//! All computations are grounded in real geometry following published metric
//! definitions. No random or synthetic values are introduced at runtime.

use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// COCO keypoint sigmas (17 joints)
// ---------------------------------------------------------------------------

/// Per-joint sigma values from the COCO keypoint evaluation standard.
///
/// These constants control the spread of the OKS Gaussian kernel for each
/// of the 17 COCO-defined body joints.
pub const COCO_KP_SIGMAS: [f32; 17] = [
    0.026, // 0  nose
    0.025, // 1  left_eye
    0.025, // 2  right_eye
    0.035, // 3  left_ear
    0.035, // 4  right_ear
    0.079, // 5  left_shoulder
    0.079, // 6  right_shoulder
    0.072, // 7  left_elbow
    0.072, // 8  right_elbow
    0.062, // 9  left_wrist
    0.062, // 10 right_wrist
    0.107, // 11 left_hip
    0.107, // 12 right_hip
    0.087, // 13 left_knee
    0.087, // 14 right_knee
    0.089, // 15 left_ankle
    0.089, // 16 right_ankle
];

// ===========================================================================
// CANONICAL METRIC — single source of truth (ADR-155 §Tier-1.1)
// ===========================================================================

/// COCO joint index of the left hip.
pub const CANON_LEFT_HIP: usize = 11;
/// COCO joint index of the right hip.
pub const CANON_RIGHT_HIP: usize = 12;

// --- Tuning constants (ADR-155 M2 §8: de-magicked from bare literals; values
// are bit-identical to the prior inline literals — documentation only, no
// behaviour change). ---

/// Visibility cutoff: a keypoint counts as *visible* iff `visibility[j] >= 0.5`.
///
/// This is the COCO convention (visibility flag 2 = "labelled and visible";
/// any soft confidence ≥ 0.5 is treated as present). Used identically in
/// [`bounding_box_diagonal`], [`canonical_torso_size`], [`pck_canonical`] and
/// [`oks_canonical`].
const VISIBILITY_THRESHOLD: f32 = 0.5;

/// Minimum positive extent for a usable reference scale (torso width or bbox
/// diagonal). Below this the sample has no measurable evidence and is reported
/// as unscoreable (PCK `(0,0,0.0)` / OKS `0.0`) rather than dividing by ≈0.
const MIN_REFERENCE_EXTENT: f32 = 1e-6;

/// Fallback per-joint OKS sigma for joint indices beyond the 17 COCO-defined
/// keypoints (defensive: the canonical path only ever scores `j < 17`). Mid-range
/// of the COCO sigma band — see [`COCO_KP_SIGMAS`].
const OKS_FALLBACK_SIGMA: f32 = 0.07;

/// Compute the Euclidean diagonal of the bounding box of visible keypoints.
///
/// The bounding box is defined by the axis-aligned extent of all keypoints
/// that have `visibility[j] >= 0.5`.  Returns 0.0 if there are no visible
/// keypoints or all are co-located.
pub(crate) fn bounding_box_diagonal(
    kp: &Array2<f32>,
    visibility: &Array1<f32>,
    num_joints: usize,
) -> f32 {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut any_visible = false;

    for j in 0..num_joints {
        if visibility[j] >= VISIBILITY_THRESHOLD {
            let x = kp[[j, 0]];
            let y = kp[[j, 1]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            any_visible = true;
        }
    }

    if !any_visible {
        return 0.0;
    }

    let w = (x_max - x_min).max(0.0);
    let h = (y_max - y_min).max(0.0);
    (w * w + h * h).sqrt()
}

/// Canonical torso normalizer used by [`pck_canonical`].
///
/// Returns `‖left_hip − right_hip‖₂` (COCO joints 11↔12) when both hips are
/// visible; otherwise the diagonal of the visible-keypoint bounding box. The
/// distance is computed in whatever coordinate space `gt_kpts` is expressed in
/// (the canonical PCK requires pred and gt to share that space).
///
/// Returns `None` when there is no positive-extent reference available (no
/// visible hips *and* a degenerate/empty visible bbox), signalling the caller
/// that the sample cannot be scored.
pub fn canonical_torso_size(gt_kpts: &Array2<f32>, visibility: &Array1<f32>) -> Option<f32> {
    let n = gt_kpts.shape()[0].min(visibility.len());
    if CANON_LEFT_HIP < n
        && CANON_RIGHT_HIP < n
        && visibility[CANON_LEFT_HIP] >= VISIBILITY_THRESHOLD
        && visibility[CANON_RIGHT_HIP] >= VISIBILITY_THRESHOLD
    {
        let dx = gt_kpts[[CANON_LEFT_HIP, 0]] - gt_kpts[[CANON_RIGHT_HIP, 0]];
        let dy = gt_kpts[[CANON_LEFT_HIP, 1]] - gt_kpts[[CANON_RIGHT_HIP, 1]];
        let torso = (dx * dx + dy * dy).sqrt();
        if torso > MIN_REFERENCE_EXTENT {
            return Some(torso);
        }
    }
    // Fallback: bounding-box diagonal of visible keypoints.
    let diag = bounding_box_diagonal(gt_kpts, visibility, n);
    if diag > MIN_REFERENCE_EXTENT {
        Some(diag)
    } else {
        None
    }
}

/// **CANONICAL PCK\@`threshold`** — the single definition used for every
/// reported number (ADR-155 §Tier-1.1).
///
/// A keypoint `j` with `visibility[j] >= 0.5` is *correct* iff
/// `‖pred_j − gt_j‖₂ ≤ threshold · torso`, where `torso` is
/// [`canonical_torso_size`] in the keypoint coordinate space.
///
/// # Returns
/// `(correct, total, pck)` where `pck ∈ [0,1]`. **`(0, 0, 0.0)` when no
/// keypoint is visible or the torso reference is degenerate** — a sample with
/// no measurable evidence scores 0, never 1 (closes the
/// `MetricsAccumulator` false-perfect bug).
///
/// # Normalization basis (vs other PCK definitions in the workspace)
/// This is **hip↔hip torso WIDTH** normalized in the keypoint coordinate space.
/// It is deliberately **distinct** from the live sensing-server's
/// `compute_pck_torso_height` (torso-HEIGHT nose→hip, pixel-space) — see ADR-155
/// §2.1 / §8. Those numbers must never be conflated.
pub fn pck_canonical(
    pred_kpts: &Array2<f32>,
    gt_kpts: &Array2<f32>,
    visibility: &Array1<f32>,
    threshold: f32,
) -> (usize, usize, f32) {
    let n = pred_kpts.shape()[0]
        .min(gt_kpts.shape()[0])
        .min(visibility.len());
    let torso = match canonical_torso_size(gt_kpts, visibility) {
        Some(t) => t,
        // No measurable reference scale ⇒ cannot score ⇒ 0.0 (NOT trivially 1.0).
        None => return (0, 0, 0.0),
    };
    let dist_threshold = threshold * torso;

    let mut correct = 0usize;
    let mut total = 0usize;
    for j in 0..n {
        if visibility[j] < VISIBILITY_THRESHOLD {
            continue;
        }
        total += 1;
        let dx = pred_kpts[[j, 0]] - gt_kpts[[j, 0]];
        let dy = pred_kpts[[j, 1]] - gt_kpts[[j, 1]];
        if (dx * dx + dy * dy).sqrt() <= dist_threshold {
            correct += 1;
        }
    }
    let pck = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };
    (correct, total, pck)
}

/// **CANONICAL OKS** — COCO Object Keypoint Similarity (ADR-155 §Tier-1.1).
///
/// `OKS = Σⱼ exp(−dⱼ² / (2 s² kⱼ²)) · δ(vⱼ≥0.5) / Σⱼ δ(vⱼ≥0.5)` with
/// `s = sqrt(area)` derived from the **GT keypoint bounding box in the
/// keypoint coordinate space** (via [`canonical_torso_size`]² as a robust,
/// always-positive proxy for area when an explicit bbox is unavailable).
///
/// Passing normalized [0,1] coordinates is fine *because the scale is derived
/// from the pose itself* — there is no `s = 1.0` escape hatch that would make
/// OKS ≈ 1.0 for any pose (the historical "fake Gold tier" bug).
///
/// Returns 0.0 when no keypoints are visible or the scale is degenerate.
pub fn oks_canonical(
    pred_kpts: &Array2<f32>,
    gt_kpts: &Array2<f32>,
    visibility: &Array1<f32>,
) -> f32 {
    let n = pred_kpts.shape()[0]
        .min(gt_kpts.shape()[0])
        .min(visibility.len());
    // Scale: area ≈ torso². Derived from the actual pose, never a fixed 1.0.
    let s = match canonical_torso_size(gt_kpts, visibility) {
        Some(t) => t,
        None => return 0.0,
    };
    let s_sq = s * s;
    if s_sq <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for j in 0..n {
        if visibility[j] < VISIBILITY_THRESHOLD {
            continue;
        }
        den += 1.0;
        let dx = pred_kpts[[j, 0]] - gt_kpts[[j, 0]];
        let dy = pred_kpts[[j, 1]] - gt_kpts[[j, 1]];
        let d_sq = dx * dx + dy * dy;
        let k = if j < COCO_KP_SIGMAS.len() {
            COCO_KP_SIGMAS[j]
        } else {
            OKS_FALLBACK_SIGMA
        };
        num += (-d_sq / (2.0 * s_sq * k * k)).exp();
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[cfg(test)]
mod consts_tests {
    use super::*;

    /// ADR-155 M2 §8: the de-magicked tuning consts must equal the prior inline
    /// literals exactly — this pins them so a future "tidy-up" cannot silently
    /// shift the metric definition (operating-value guard).
    #[test]
    fn metrics_core_consts_unchanged_from_literals() {
        assert_eq!(VISIBILITY_THRESHOLD, 0.5_f32);
        assert_eq!(MIN_REFERENCE_EXTENT, 1e-6_f32);
        assert_eq!(OKS_FALLBACK_SIGMA, 0.07_f32);
        assert_eq!(CANON_LEFT_HIP, 11);
        assert_eq!(CANON_RIGHT_HIP, 12);
    }

    /// Characterize the visibility-threshold boundary: a keypoint at exactly the
    /// cutoff (vis == 0.5) is INCLUDED (`>=`), just below (0.499) is EXCLUDED.
    /// Pins current `>=`-inclusive behaviour at the edge.
    #[test]
    fn visibility_threshold_boundary_is_inclusive() {
        // Two GT hips give a positive torso; vary the (single) scored joint's
        // visibility around the 0.5 cutoff and confirm it flips total in/out.
        let gt = Array2::from_shape_vec(
            (13, 2),
            (0..13).flat_map(|j| [j as f32, 0.0]).collect::<Vec<_>>(),
        )
        .unwrap();
        // hips at 11,12 give torso = |11-12| = 1.0 along x.
        let pred = gt.clone();
        let mk_vis = |v0: f32| {
            let mut vis = Array1::<f32>::zeros(13);
            vis[CANON_LEFT_HIP] = 1.0;
            vis[CANON_RIGHT_HIP] = 1.0;
            vis[0] = v0; // joint 0 is the one we toggle
            vis
        };
        // At exactly 0.5 → joint 0 is counted (total includes it: 3 visible).
        let (_, total_at, _) = pck_canonical(&pred, &gt, &mk_vis(0.5), 0.2);
        assert_eq!(total_at, 3, "vis == 0.5 must be INCLUDED (>=)");
        // Just below → joint 0 excluded (only the 2 hips visible).
        let (_, total_below, _) = pck_canonical(&pred, &gt, &mk_vis(0.499), 0.2);
        assert_eq!(total_below, 2, "vis < 0.5 must be EXCLUDED");
    }

    /// Characterize the reference-extent floor: a near-zero-extent GT pose (all
    /// keypoints coincident, hips coincident) is UNSCOREABLE → `(0,0,0.0)`,
    /// never a trivial perfect score. Pins the `MIN_REFERENCE_EXTENT` guard.
    #[test]
    fn degenerate_extent_below_floor_is_unscoreable() {
        // All 13 joints at the same point ⇒ torso ≈ 0, bbox diag ≈ 0 < 1e-6.
        let gt = Array2::<f32>::zeros((13, 2));
        let pred = gt.clone();
        let mut vis = Array1::<f32>::zeros(13);
        vis[CANON_LEFT_HIP] = 1.0;
        vis[CANON_RIGHT_HIP] = 1.0;
        assert!(canonical_torso_size(&gt, &vis).is_none());
        assert_eq!(pck_canonical(&pred, &gt, &vis, 0.2), (0, 0, 0.0));
        assert_eq!(oks_canonical(&pred, &gt, &vis), 0.0);
    }
}
