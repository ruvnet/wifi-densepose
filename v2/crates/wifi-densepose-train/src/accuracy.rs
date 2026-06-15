//! Metric-locked pose-accuracy harness (ADR-155 §Tier-1.2; needs ADR slot 173).
//!
//! # Why this module exists
//!
//! Three PCK\@20 numbers float around this project and **cannot be lined up**
//! because each silently uses a *different* PCK definition:
//!
//! | Number | Source | PCK normalization |
//! |--------|--------|-------------------|
//! | 96.09 %  | WiFlow-STD reproduction | image / bounding-box normalized (looser) |
//! | 81.63 %  | AetherArena MM-Fi (ADR-150) | torso-diameter (standard MM-Fi / GraphPose-Fi) |
//! | 61.1 %   | GraphPose-Fi (preprint) | torso-diameter, 3D, mm-scale (harder) |
//!
//! The project was burned **twice** by metric ambiguity (a now-retracted "92.9 %
//! PCK\@20" used *absolute* pixel thresholds, not torso normalization). The fix
//! is to make the normalizer **explicit, selectable, and carried with every
//! reported number** so an unlabeled PCK figure is structurally impossible.
//!
//! [`metrics_core`](crate::metrics_core) already pins the *canonical*
//! torso-normalized PCK ([`pck_canonical`](crate::metrics_core::pck_canonical)).
//! This module generalizes it to a [`PckNormalization`] enum covering all three
//! conventions the SOTA brief names, adds [`mpjpe`] (mm), and bundles results
//! into a self-describing [`PoseAccuracy`] struct. It **reuses** the
//! `metrics_core` primitives (hip distance, bounding-box diagonal) — there is
//! still exactly one implementation of each geometric reference.
//!
//! # This is measurement infrastructure, not an accuracy claim
//!
//! Nothing here asserts any project model is good. The unit tests prove the
//! *harness* is arithmetically correct against hand-computed fixtures (no GPU,
//! no datasets), including the key demonstration that the **same predictions
//! score different PCK under the three normalizations** — proof the ambiguity is
//! real and the definitions are genuinely distinct.
//!
//! # Literature
//!
//! - Torso-diameter PCK is the MM-Fi / GraphPose-Fi convention (Yang et al.,
//!   *GraphPose-Fi*, arXiv:2511.19105): a keypoint is correct iff its error is
//!   within `k · d_torso`, with `d_torso` the hip↔hip (or shoulder↔hip) span.
//! - Bounding-box / image-normalized PCK is the WiFlow-STD-style looser
//!   convention (arXiv:2602.08661) — normalize by the GT pose bbox diagonal.
//! - MPJPE (mean per-joint position error, mm) is reported by GraphPose-Fi and
//!   Person-in-WiFi-3D (Yan et al., CVPR 2024).

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};

use crate::metrics_core::{
    bounding_box_diagonal, CANON_LEFT_HIP, CANON_RIGHT_HIP,
};

/// Visibility cutoff: a keypoint counts as *visible* iff `visibility[j] >= 0.5`
/// (COCO convention; matches [`crate::metrics_core`]).
const VISIBILITY_THRESHOLD: f32 = 0.5;

/// Minimum positive normalizer extent. Below this the reference scale is
/// considered degenerate (zero torso, collapsed bbox) and the frame is reported
/// unscoreable rather than dividing by ≈0.
const MIN_REFERENCE_EXTENT: f32 = 1e-6;

// ===========================================================================
// PCK normalization — the explicit, selectable definition
// ===========================================================================

/// The PCK normalization basis — **the single knob that made three project
/// numbers non-comparable**, now explicit and carried with every result.
///
/// A keypoint `j` (with `visibility[j] >= 0.5`) is *correct* iff
/// `‖pred_j − gt_j‖₂ ≤ τ`, where the **distance tolerance `τ`** is derived from
/// the chosen normalization and the PCK threshold `k` (given as a percentage,
/// e.g. `20` for PCK\@20):
///
/// | Variant | `τ` (tolerance in coordinate units) |
/// |---------|--------------------------------------|
/// | [`TorsoDiameter`](Self::TorsoDiameter)        | `(k/100) · d_torso` |
/// | [`BoundingBoxDiagonal`](Self::BoundingBoxDiagonal) | `(k/100) · d_bbox`  |
/// | [`AbsolutePixels`](Self::AbsolutePixels)      | `threshold` (k ignored) |
///
/// `d_torso` is the hip↔hip span (COCO joints 11↔12), falling back to the bbox
/// diagonal when both hips are not visible — identical to
/// [`crate::metrics_core::canonical_torso_size`]. `d_bbox` is the diagonal of
/// the axis-aligned bounding box of all visible GT keypoints.
///
/// These yield **different** PCK on the *same* predictions whenever
/// `d_torso ≠ d_bbox` (always true for a real pose: the bbox is larger than the
/// hip span), which is exactly why the 96 / 81.6 / 61 numbers cannot be lined
/// up without declaring this enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PckNormalization {
    /// **Torso-diameter** (hip↔hip span). The standard MM-Fi / GraphPose-Fi
    /// convention and the *stricter* of the two relative normalizers. This is
    /// the canonical default ([`crate::metrics_core::pck_canonical`]).
    TorsoDiameter,
    /// **Bounding-box diagonal** (a.k.a. image-normalized). The looser
    /// WiFlow-STD-style convention: normalize by the GT pose bbox diagonal,
    /// which is larger than the torso span ⇒ a more forgiving threshold ⇒ a
    /// higher PCK on identical predictions.
    BoundingBoxDiagonal,
    /// **Absolute pixel/coordinate threshold** — no pose-relative
    /// normalization. The PCK `k` percentage is ignored; the held `threshold`
    /// is the raw distance tolerance directly. Included so historical
    /// retracted-style numbers are reproducible, and **clearly labeled as
    /// non-comparable** to the relative variants (it does not scale with body
    /// size or camera distance).
    AbsolutePixels(f32),
}

impl PckNormalization {
    /// Human-readable, *self-documenting* label for a reported number — so a
    /// `PoseAccuracy` printed anywhere always carries its definition.
    pub fn label(&self) -> String {
        match self {
            PckNormalization::TorsoDiameter => "torso-diameter".to_string(),
            PckNormalization::BoundingBoxDiagonal => "bbox-diagonal".to_string(),
            PckNormalization::AbsolutePixels(t) => format!("absolute-px({t})"),
        }
    }

    /// Compute the per-frame distance tolerance `τ` for PCK threshold `k`
    /// (percentage). Returns `None` when the (relative) normalizer is degenerate
    /// — the frame cannot be scored.
    ///
    /// `gt_kpts` is `[n, 2]` (or `[n, ≥2]`, only x/y used); `visibility` is `[n]`.
    fn tolerance(&self, gt_kpts: &Array2<f32>, visibility: &Array1<f32>, k: u8) -> Option<f32> {
        let n = gt_kpts.shape()[0].min(visibility.len());
        match self {
            PckNormalization::AbsolutePixels(threshold) => {
                // Raw tolerance, independent of pose scale and of `k`.
                if *threshold > 0.0 {
                    Some(*threshold)
                } else {
                    None
                }
            }
            PckNormalization::TorsoDiameter => {
                let d = torso_diameter(gt_kpts, visibility, n)?;
                Some((k as f32 / 100.0) * d)
            }
            PckNormalization::BoundingBoxDiagonal => {
                let d = bounding_box_diagonal(gt_kpts, visibility, n);
                if d > MIN_REFERENCE_EXTENT {
                    Some((k as f32 / 100.0) * d)
                } else {
                    None
                }
            }
        }
    }
}

/// Hip↔hip torso diameter with a bbox-diagonal fallback — the relative
/// normalizer shared by `TorsoDiameter` PCK and
/// [`crate::metrics_core::canonical_torso_size`]. Returns `None` when no
/// positive-extent reference exists.
fn torso_diameter(gt_kpts: &Array2<f32>, visibility: &Array1<f32>, n: usize) -> Option<f32> {
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
    let diag = bounding_box_diagonal(gt_kpts, visibility, n);
    if diag > MIN_REFERENCE_EXTENT {
        Some(diag)
    } else {
        None
    }
}

// ===========================================================================
// Single-frame PCK / MPJPE
// ===========================================================================

/// Per-frame **PCK\@`k`** under the selected `normalization`.
///
/// A keypoint `j` with `visibility[j] >= 0.5` is correct iff
/// `‖pred_j − gt_j‖₂ ≤ τ`, with `τ` from
/// [`PckNormalization::tolerance`]. Only x/y are used (2D PCK is the standard
/// keypoint-PCK definition; pass 2-column arrays).
///
/// # Returns
/// `(correct, total, pck)` with `pck ∈ [0,1]`. **`(0, 0, 0.0)`** when no
/// keypoint is visible, or (for the relative normalizers) the reference scale is
/// degenerate — a frame with no measurable evidence scores 0, never 1.
/// NaN-valued coordinates make a keypoint *incorrect* (the `<=` comparison is
/// false for NaN) rather than panicking.
pub fn pck_at(
    pred_kpts: &Array2<f32>,
    gt_kpts: &Array2<f32>,
    visibility: &Array1<f32>,
    k: u8,
    normalization: PckNormalization,
) -> (usize, usize, f32) {
    let n = pred_kpts.shape()[0]
        .min(gt_kpts.shape()[0])
        .min(visibility.len());
    let tol = match normalization.tolerance(gt_kpts, visibility, k) {
        Some(t) => t,
        None => return (0, 0, 0.0),
    };

    let mut correct = 0usize;
    let mut total = 0usize;
    for j in 0..n {
        if visibility[j] < VISIBILITY_THRESHOLD {
            continue;
        }
        total += 1;
        let dx = pred_kpts[[j, 0]] - gt_kpts[[j, 0]];
        let dy = pred_kpts[[j, 1]] - gt_kpts[[j, 1]];
        let dist = (dx * dx + dy * dy).sqrt();
        // NaN-safe: `NaN <= tol` is false, so a NaN coordinate counts as wrong.
        if dist <= tol {
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

/// Per-frame **MPJPE** (mean per-joint position error) over visible keypoints,
/// in the coordinate units of the inputs (report as mm when inputs are mm).
///
/// `pred`/`gt` are `[n, D]` with `D ∈ {2, 3}` (2D or 3D pose); all `D` columns
/// are used. Joints with `visibility[j] < 0.5` are excluded.
///
/// Returns `0.0` when no keypoint is visible (no evidence). A NaN coordinate
/// propagates into the returned mean (callers filter NaN frames upstream); it
/// does not panic.
pub fn mpjpe(pred: &Array2<f32>, gt: &Array2<f32>, visibility: &Array1<f32>) -> f32 {
    let n = pred.shape()[0].min(gt.shape()[0]).min(visibility.len());
    let d = pred.shape()[1].min(gt.shape()[1]);
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for j in 0..n {
        if visibility[j] < VISIBILITY_THRESHOLD {
            continue;
        }
        let mut sq = 0.0f32;
        for c in 0..d {
            let diff = pred[[j, c]] - gt[[j, c]];
            sq += diff * diff;
        }
        sum += sq.sqrt();
        count += 1;
    }
    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

// ===========================================================================
// Self-describing result struct + batch report
// ===========================================================================

/// A pose-accuracy result that **always carries the definition it was computed
/// under** — making an unlabeled PCK number structurally impossible.
///
/// Built by [`accuracy_report`] over a set of frames. `pck_at` maps each
/// requested threshold `k` (percentage, e.g. `20`) to its PCK in `[0,1]`. The
/// `normalization` field records *which* PCK definition produced those numbers,
/// so two `PoseAccuracy` values can only be compared when their `normalization`
/// matches (the comparability check the project lacked).
#[derive(Debug, Clone, PartialEq)]
pub struct PoseAccuracy {
    /// PCK\@k for each requested threshold percentage `k`, in `[0,1]`.
    pub pck_at: BTreeMap<u8, f32>,
    /// Mean per-joint position error in coordinate units (mm for mm inputs).
    pub mpjpe: f32,
    /// The normalization basis under which `pck_at` was computed — the label a
    /// reported number must always carry.
    pub normalization: PckNormalization,
    /// Number of keypoints per frame (the pose convention, e.g. 17 for COCO).
    pub n_keypoints: usize,
    /// Number of frames aggregated into this result.
    pub n_frames: usize,
}

impl PoseAccuracy {
    /// Convenience accessor for a single threshold, returning `None` when that
    /// `k` was not requested.
    pub fn pck(&self, k: u8) -> Option<f32> {
        self.pck_at.get(&k).copied()
    }

    /// A one-line, self-documenting summary suitable for logs / RESULTS.md, e.g.
    /// `PCK@20=0.750 (torso-diameter, 17kp, 1 frames) MPJPE=0.030`.
    pub fn summary(&self) -> String {
        let pcks: Vec<String> = self
            .pck_at
            .iter()
            .map(|(k, v)| format!("PCK@{k}={v:.3}"))
            .collect();
        format!(
            "{} ({}, {}kp, {} frames) MPJPE={:.4}",
            pcks.join(" "),
            self.normalization.label(),
            self.n_keypoints,
            self.n_frames,
            self.mpjpe
        )
    }
}

/// One frame's prediction + ground truth + visibility for batch scoring.
///
/// All three arrays share row count `n_keypoints`; `pred`/`gt` are `[n, D]`
/// (`D ∈ {2,3}`), `visibility` is `[n]`.
#[derive(Debug, Clone)]
pub struct PoseFrame {
    /// Predicted keypoints `[n, D]`.
    pub pred: Array2<f32>,
    /// Ground-truth keypoints `[n, D]`.
    pub gt: Array2<f32>,
    /// Per-keypoint visibility `[n]` (`>= 0.5` ⇒ visible).
    pub visibility: Array1<f32>,
}

/// Aggregate [`PoseAccuracy`] over a batch of frames under **one** explicit
/// `normalization`, for the requested PCK thresholds `ks` (percentages).
///
/// PCK is micro-averaged over keypoints (sum of correct ÷ sum of visible across
/// all frames — the standard keypoint-PCK aggregation), so frames with more
/// visible joints contribute proportionally. MPJPE is micro-averaged over
/// visible joints likewise. Unscoreable frames (no visible joints, degenerate
/// relative normalizer) contribute `(0, 0)` and so are excluded from the
/// denominator rather than scored as perfect.
///
/// An **empty** `frames` slice yields all-zero PCK and `0.0` MPJPE — never a
/// panic or NaN.
pub fn accuracy_report(
    frames: &[PoseFrame],
    ks: &[u8],
    normalization: PckNormalization,
) -> PoseAccuracy {
    let n_keypoints = frames.first().map(|f| f.gt.shape()[0]).unwrap_or(0);

    // PCK: per-threshold (correct, total) accumulators across frames.
    let mut pck_acc: BTreeMap<u8, (usize, usize)> = ks.iter().map(|&k| (k, (0, 0))).collect();
    // MPJPE: sum of per-joint distances and visible-joint count.
    let mut mpjpe_sum = 0.0f32;
    let mut mpjpe_count = 0usize;

    for frame in frames {
        for &k in ks {
            let (c, t, _) = pck_at(&frame.pred, &frame.gt, &frame.visibility, k, normalization);
            let entry = pck_acc.entry(k).or_insert((0, 0));
            entry.0 += c;
            entry.1 += t;
        }
        // Per-frame MPJPE re-derived as a (sum, count) contribution so the
        // batch value is a true micro-average over joints.
        let n = frame.pred.shape()[0].min(frame.gt.shape()[0]).min(frame.visibility.len());
        let d = frame.pred.shape()[1].min(frame.gt.shape()[1]);
        for j in 0..n {
            if frame.visibility[j] < VISIBILITY_THRESHOLD {
                continue;
            }
            let mut sq = 0.0f32;
            for c in 0..d {
                let diff = frame.pred[[j, c]] - frame.gt[[j, c]];
                sq += diff * diff;
            }
            mpjpe_sum += sq.sqrt();
            mpjpe_count += 1;
        }
    }

    let pck_at: BTreeMap<u8, f32> = pck_acc
        .into_iter()
        .map(|(k, (c, t))| {
            let v = if t > 0 { c as f32 / t as f32 } else { 0.0 };
            (k, v)
        })
        .collect();

    let mpjpe = if mpjpe_count > 0 {
        mpjpe_sum / mpjpe_count as f32
    } else {
        0.0
    };

    PoseAccuracy {
        pck_at,
        mpjpe,
        normalization,
        n_keypoints,
        n_frames: frames.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 17-joint `[17, 2]` pose from `(joint, x, y)` triples.
    fn pose17(joints: &[(usize, f32, f32)]) -> Array2<f32> {
        let mut a = Array2::<f32>::zeros((17, 2));
        for &(j, x, y) in joints {
            a[[j, 0]] = x;
            a[[j, 1]] = y;
        }
        a
    }

    fn vis17(visible: &[usize]) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(17);
        for &j in visible {
            v[j] = 2.0;
        }
        v
    }

    // -------- consts pinned (no silent metric drift) --------
    #[test]
    fn accuracy_consts_unchanged() {
        assert_eq!(VISIBILITY_THRESHOLD, 0.5_f32);
        assert_eq!(MIN_REFERENCE_EXTENT, 1e-6_f32);
    }

    // -------- perfect prediction ⇒ PCK = 1.0, MPJPE = 0 --------
    #[test]
    fn perfect_prediction_pck_one_mpjpe_zero() {
        let gt = pose17(&[
            (5, 0.35, 0.35),
            (CANON_LEFT_HIP, 0.40, 0.50),
            (CANON_RIGHT_HIP, 0.60, 0.50),
        ]);
        let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        for norm in [
            PckNormalization::TorsoDiameter,
            PckNormalization::BoundingBoxDiagonal,
            PckNormalization::AbsolutePixels(0.01),
        ] {
            let (c, t, pck) = pck_at(&gt, &gt, &vis, 20, norm);
            assert_eq!((c, t), (3, 3), "{norm:?}");
            assert!((pck - 1.0).abs() < 1e-6, "{norm:?} perfect PCK must be 1.0");
        }
        assert_eq!(mpjpe(&gt, &gt, &vis), 0.0);
    }

    // -------- all keypoints just OUTSIDE threshold ⇒ PCK = 0.0 --------
    //
    // Hand calc (torso): hips at (0.40,0.50)/(0.60,0.50) ⇒ torso = 0.20.
    // threshold k=20 ⇒ τ = 0.20·0.20 = 0.04. Push every scored joint to an
    // error of 0.05 (> 0.04) ⇒ all wrong. To avoid the hips themselves being
    // "correct", we displace the hips too (their displaced positions still
    // define the torso from GT, which is unchanged).
    #[test]
    fn all_just_outside_threshold_pck_zero() {
        let gt = pose17(&[
            (5, 0.50, 0.50),
            (CANON_LEFT_HIP, 0.40, 0.50),
            (CANON_RIGHT_HIP, 0.60, 0.50),
        ]);
        // GT torso = 0.20, τ@20 = 0.04. Displace each scored joint by dx=0.05.
        let pred = pose17(&[
            (5, 0.55, 0.50),
            (CANON_LEFT_HIP, 0.45, 0.50),
            (CANON_RIGHT_HIP, 0.65, 0.50),
        ]);
        let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        let (c, t, pck) = pck_at(&pred, &gt, &vis, 20, PckNormalization::TorsoDiameter);
        assert_eq!(t, 3);
        assert_eq!(c, 0, "all errors 0.05 > τ 0.04 ⇒ none correct");
        assert_eq!(pck, 0.0);
    }

    // -------- half-in / half-out ⇒ PCK = 0.5 --------
    //
    // Hand calc (torso): torso = 0.20, τ@20 = 0.04. Four visible joints; two
    // exact (dist 0 ≤ 0.04, correct), two displaced 0.05 (> 0.04, wrong)
    // ⇒ 2/4 = 0.5.
    #[test]
    fn half_in_half_out_pck_half() {
        let gt = pose17(&[
            (0, 0.50, 0.20),
            (5, 0.50, 0.50),
            (CANON_LEFT_HIP, 0.40, 0.50),
            (CANON_RIGHT_HIP, 0.60, 0.50),
        ]);
        let pred = pose17(&[
            (0, 0.50, 0.20),          // exact ⇒ correct
            (5, 0.55, 0.50),          // err 0.05 ⇒ wrong
            (CANON_LEFT_HIP, 0.40, 0.50),  // exact ⇒ correct
            (CANON_RIGHT_HIP, 0.65, 0.50), // err 0.05 ⇒ wrong
        ]);
        let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        let (c, t, pck) = pck_at(&pred, &gt, &vis, 20, PckNormalization::TorsoDiameter);
        assert_eq!((c, t), (2, 4));
        assert!((pck - 0.5).abs() < 1e-6, "expected 0.5, got {pck}");
    }

    // -------- THE KEY PROOF: same predictions, three normalizations, three PCK --------
    //
    // One construction scored three ways. Hand calc:
    //   GT: nose(0)=(0.50,0.10), l_sh(5)=(0.50,0.30),
    //       l_hip(11)=(0.40,0.90), r_hip(12)=(0.60,0.90).
    //   Visible = {0,5,11,12}, all four.
    //   torso  = |0.60-0.40| = 0.20  (hips, y equal).
    //   bbox: x∈[0.40,0.60] (w=0.20), y∈[0.10,0.90] (h=0.80)
    //         ⇒ diag = sqrt(0.20² + 0.80²) = sqrt(0.04+0.64)=sqrt(0.68)=0.8246…
    //
    //   Pred errors (pure dx): nose 0.00, l_sh 0.10, l_hip 0.00, r_hip 0.00.
    //   (Only joint 5 is displaced, by 0.10.)
    //
    //   k = 20:
    //   • Torso  τ = 0.20·0.20 = 0.040 → joint5 err 0.10 > 0.040 ⇒ WRONG
    //       ⇒ 3 correct / 4 = 0.75
    //   • Bbox   τ = 0.20·0.8246 = 0.16492 → joint5 err 0.10 ≤ 0.16492 ⇒ CORRECT
    //       ⇒ 4 correct / 4 = 1.00
    //   • Abs(0.05) τ = 0.05 → joint5 err 0.10 > 0.05 ⇒ WRONG
    //       ⇒ 3 correct / 4 = 0.75   (same count as torso HERE by coincidence)
    //
    //   To make ALL THREE differ, also test Abs(0.08): τ=0.08, joint5 0.10>0.08
    //   ⇒ still 0.75. So we additionally displace nose by 0.06 (between 0.05 and
    //   0.08) to separate the two absolute thresholds — see below.
    #[test]
    fn three_normalizations_give_different_pck_on_identical_input() {
        let gt = pose17(&[
            (0, 0.50, 0.10),  // nose
            (5, 0.50, 0.30),  // left_shoulder
            (CANON_LEFT_HIP, 0.40, 0.90),
            (CANON_RIGHT_HIP, 0.60, 0.90),
        ]);
        // nose displaced 0.06, shoulder displaced 0.10, hips exact.
        let pred = pose17(&[
            (0, 0.56, 0.10),  // err 0.06
            (5, 0.60, 0.30),  // err 0.10
            (CANON_LEFT_HIP, 0.40, 0.90),  // exact
            (CANON_RIGHT_HIP, 0.60, 0.90), // exact
        ]);
        let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);

        // Torso τ@20 = 0.04: nose 0.06>0.04 wrong, sh 0.10>0.04 wrong,
        //   hips exact ⇒ 2/4 = 0.5.
        let (_, _, torso) = pck_at(&pred, &gt, &vis, 20, PckNormalization::TorsoDiameter);
        // Bbox diag = sqrt(0.68)=0.82462; τ@20 = 0.164924:
        //   nose 0.06 ≤ τ correct, sh 0.10 ≤ τ correct, hips exact ⇒ 4/4 = 1.0.
        let (_, _, bbox) = pck_at(&pred, &gt, &vis, 20, PckNormalization::BoundingBoxDiagonal);
        // Abs(0.08): nose 0.06 ≤ 0.08 correct, sh 0.10 > 0.08 wrong, hips exact
        //   ⇒ 3/4 = 0.75.
        let (_, _, abs) = pck_at(&pred, &gt, &vis, 20, PckNormalization::AbsolutePixels(0.08));

        assert!((torso - 0.5).abs() < 1e-6, "torso PCK expected 0.5, got {torso}");
        assert!((bbox - 1.0).abs() < 1e-6, "bbox PCK expected 1.0, got {bbox}");
        assert!((abs - 0.75).abs() < 1e-6, "abs(0.08) PCK expected 0.75, got {abs}");

        // The whole point: identical predictions, three DISTINCT PCK values.
        assert!(torso != bbox && bbox != abs && torso != abs,
            "normalizations must give distinct PCK: torso={torso}, bbox={bbox}, abs={abs}");
    }

    // -------- AbsolutePixels ignores k (raw threshold) --------
    #[test]
    fn absolute_pixels_ignores_threshold_percentage() {
        let gt = pose17(&[(5, 0.50, 0.50), (CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);
        let pred = pose17(&[(5, 0.53, 0.50), (CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);
        let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        // τ = 0.05 raw; joint5 err 0.03 ≤ 0.05 correct. k=5 and k=99 must agree.
        let (_, _, p5) = pck_at(&pred, &gt, &vis, 5, PckNormalization::AbsolutePixels(0.05));
        let (_, _, p99) = pck_at(&pred, &gt, &vis, 99, PckNormalization::AbsolutePixels(0.05));
        assert_eq!(p5, p99, "AbsolutePixels must ignore the k percentage");
        assert!((p5 - 1.0).abs() < 1e-6, "all three within 0.05, got {p5}");
    }

    // -------- MPJPE hand-computed (2D and 3D) --------
    #[test]
    fn mpjpe_hand_computed_2d() {
        // joint0 err (3,4)->5, joint1 exact->0 ⇒ mean (5+0)/2 = 2.5.
        let gt = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let pred = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 1.0, 1.0]).unwrap();
        let vis = Array1::from(vec![2.0, 2.0]);
        assert!((mpjpe(&pred, &gt, &vis) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn mpjpe_hand_computed_3d() {
        // single joint err (1,2,2) -> sqrt(1+4+4)=3.0.
        let gt = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let pred = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 2.0]).unwrap();
        let vis = Array1::from(vec![2.0]);
        assert!((mpjpe(&pred, &gt, &vis) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mpjpe_excludes_invisible_joints() {
        // joint0 visible err 5, joint1 INVISIBLE err 100 ⇒ mean = 5 (joint1 dropped).
        let gt = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let pred = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 100.0, 0.0]).unwrap();
        let vis = Array1::from(vec![2.0, 0.0]);
        assert!((mpjpe(&pred, &gt, &vis) - 5.0).abs() < 1e-6);
    }

    // -------- degenerate inputs: no panic --------
    #[test]
    fn zero_torso_is_unscoreable_not_perfect() {
        // Both hips coincident ⇒ torso ≈ 0; bbox also collapses ⇒ None.
        let gt = pose17(&[(CANON_LEFT_HIP, 0.5, 0.5), (CANON_RIGHT_HIP, 0.5, 0.5)]);
        let vis = vis17(&[CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        assert_eq!(pck_at(&gt, &gt, &vis, 20, PckNormalization::TorsoDiameter), (0, 0, 0.0));
        assert_eq!(pck_at(&gt, &gt, &vis, 20, PckNormalization::BoundingBoxDiagonal), (0, 0, 0.0));
    }

    #[test]
    fn no_visible_keypoints_scores_zero() {
        let gt = pose17(&[(CANON_LEFT_HIP, 0.4, 0.5), (CANON_RIGHT_HIP, 0.6, 0.5)]);
        let vis = vis17(&[]); // nothing visible
        let (c, t, pck) = pck_at(&gt, &gt, &vis, 20, PckNormalization::TorsoDiameter);
        assert_eq!((c, t, pck), (0, 0, 0.0));
        assert_eq!(mpjpe(&gt, &gt, &vis), 0.0);
    }

    #[test]
    fn nan_coords_do_not_panic_and_count_wrong() {
        let gt = pose17(&[(5, 0.5, 0.5), (CANON_LEFT_HIP, 0.4, 0.5), (CANON_RIGHT_HIP, 0.6, 0.5)]);
        let mut pred = gt.clone();
        pred[[5, 0]] = f32::NAN; // joint 5 prediction is NaN
        let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        let (c, t, pck) = pck_at(&pred, &gt, &vis, 20, PckNormalization::TorsoDiameter);
        assert_eq!(t, 3);
        assert_eq!(c, 2, "NaN joint must count as wrong, hips correct ⇒ 2/3");
        assert!((pck - 2.0 / 3.0).abs() < 1e-6);
        // mpjpe with a NaN joint yields NaN (caller filters) but must not panic.
        assert!(mpjpe(&pred, &gt, &vis).is_nan());
    }

    // -------- batch report: micro-average + self-describing struct --------
    #[test]
    fn accuracy_report_micro_averages_and_carries_definition() {
        // Frame A: 2 visible, both correct (2/2). Frame B: 2 visible, both wrong (0/2).
        // Micro-average over joints: 2 correct / 4 = 0.5 (NOT mean-of-frame-PCK,
        // which would be (1.0+0.0)/2 = 0.5 here too, but the accumulator is the
        // joint-level one).
        let gt = pose17(&[(CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);
        let vis = vis17(&[CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        let frame_a = PoseFrame { pred: gt.clone(), gt: gt.clone(), visibility: vis.clone() };
        // Frame B: displace both hips by 0.05 (> τ 0.04) ⇒ both wrong.
        let pred_b = pose17(&[(CANON_LEFT_HIP, 0.45, 0.50), (CANON_RIGHT_HIP, 0.65, 0.50)]);
        let frame_b = PoseFrame { pred: pred_b, gt: gt.clone(), visibility: vis.clone() };

        let report = accuracy_report(
            &[frame_a, frame_b],
            &[20, 50],
            PckNormalization::TorsoDiameter,
        );
        assert_eq!(report.n_frames, 2);
        assert_eq!(report.n_keypoints, 17);
        assert_eq!(report.normalization, PckNormalization::TorsoDiameter);
        // PCK@20: 2 correct / 4 visible = 0.5.
        assert!((report.pck(20).unwrap() - 0.5).abs() < 1e-6);
        // PCK@50: τ = 0.5·0.20 = 0.10, frame B err 0.05 ≤ 0.10 ⇒ all correct
        //   ⇒ 4/4 = 1.0.
        assert!((report.pck(50).unwrap() - 1.0).abs() < 1e-6);
        // A reported number always carries its definition in the summary.
        assert!(report.summary().contains("torso-diameter"));
    }

    #[test]
    fn accuracy_report_empty_is_zero_not_nan() {
        let report = accuracy_report(&[], &[20], PckNormalization::BoundingBoxDiagonal);
        assert_eq!(report.n_frames, 0);
        assert_eq!(report.pck(20), Some(0.0));
        assert_eq!(report.mpjpe, 0.0);
        assert!(!report.mpjpe.is_nan());
    }

    // -------- bbox-norm is looser than torso-norm (sanity, on a batch) --------
    #[test]
    fn bbox_norm_scores_at_least_torso_norm() {
        // bbox diagonal >= torso span always (bbox encloses the hips), so for the
        // SAME frames bbox-PCK >= torso-PCK at the same k. Pin this ordering.
        let gt = pose17(&[
            (0, 0.50, 0.10),
            (5, 0.50, 0.40),
            (CANON_LEFT_HIP, 0.40, 0.90),
            (CANON_RIGHT_HIP, 0.60, 0.90),
        ]);
        let pred = pose17(&[
            (0, 0.55, 0.10),
            (5, 0.58, 0.40),
            (CANON_LEFT_HIP, 0.42, 0.90),
            (CANON_RIGHT_HIP, 0.62, 0.90),
        ]);
        let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
        let frame = PoseFrame { pred, gt, visibility: vis };
        let torso = accuracy_report(std::slice::from_ref(&frame), &[20], PckNormalization::TorsoDiameter);
        let bbox = accuracy_report(std::slice::from_ref(&frame), &[20], PckNormalization::BoundingBoxDiagonal);
        assert!(
            bbox.pck(20).unwrap() >= torso.pck(20).unwrap(),
            "bbox-norm (looser) must be >= torso-norm: bbox={:?} torso={:?}",
            bbox.pck(20), torso.pck(20)
        );
    }
}
