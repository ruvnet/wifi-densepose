//! Integration tests for `wifi_densepose_train` pose metrics.
//!
//! # ADR-155 Milestone-1 — §8 "reference kernels" resolution
//!
//! The full `metrics` module is gated behind `tch-backend` (libtorch), but the
//! **canonical** metric core (`pck_canonical` / `oks_canonical`) now lives in
//! the un-gated `metrics_core` module and is re-exported at the crate root, so
//! these workspace tests (run under `--no-default-features`) validate the
//! **production** functions directly.
//!
//! Previously this file carried its own local `compute_pck` / `compute_oks`
//! reimplementations and asserted properties of *those* — a test that could
//! not catch a bug in the canonical implementation (both could be wrong the
//! same way). That is fixed two ways here:
//!
//! 1. **Fixture tests** (`canonical_pck_matches_hand_computed_fixture`,
//!    `canonical_oks_*`) assert the production `pck_canonical` / `oks_canonical`
//!    equal *hand-computed* expected values — numbers worked out by hand below,
//!    NOT a second implementation of the same algorithm.
//! 2. **Differential test** (`test_kernel_agrees_with_canonical`) keeps a small
//!    independent reference kernel and asserts it **agrees** with the canonical
//!    function on shared inputs (in the torso=raw-threshold regime where the two
//!    coincide), so the reference adds genuine cross-check value rather than
//!    duplicating the algorithm under test.
//!
//! `EvalMetrics` tests remain `#[cfg(feature = "tch-backend")]` (that type is in
//! the gated module). All inputs are fixed, deterministic arrays — no `rand`,
//! no OS entropy.

use ndarray::{Array1, Array2};
use wifi_densepose_train::{oks_canonical, pck_canonical, CANON_LEFT_HIP, CANON_RIGHT_HIP};
// ADR-155 §Tier-1.2 — metric-locked accuracy harness public surface.
use wifi_densepose_train::{accuracy_report, pck_at, PckNormalization, PoseFrame};

// ---------------------------------------------------------------------------
// Metric-locked accuracy harness: the three PCK normalizations are reachable
// from the crate root and give DIFFERENT PCK on identical predictions — the
// proof that the 96 / 81.6 / 61 figures were non-comparable (validated here as
// a downstream consumer would call it).
// ---------------------------------------------------------------------------

/// Identical predictions, three declared normalizations ⇒ three distinct PCK.
/// Hand calc (all coords in `[0,1]`):
/// * GT: nose(0)=(0.50,0.10), l_sh(5)=(0.50,0.30), hips=(0.40,0.90)/(0.60,0.90).
/// * Pred: nose err 0.06, shoulder err 0.10, hips exact.
/// * torso = 0.20 ⇒ τ@20 = 0.04 ⇒ only hips correct ⇒ 2/4 = **0.50**.
/// * bbox  = √(0.20²+0.80²)=0.82462 ⇒ τ@20 = 0.16492 ⇒ all correct ⇒ **1.00**.
/// * abs(0.08): nose 0.06≤0.08 ok, shoulder 0.10>0.08 wrong ⇒ 3/4 = **0.75**.
#[test]
fn harness_three_normalizations_differ_from_crate_root() {
    let gt = pose17(&[
        (0, 0.50, 0.10),
        (5, 0.50, 0.30),
        (CANON_LEFT_HIP, 0.40, 0.90),
        (CANON_RIGHT_HIP, 0.60, 0.90),
    ]);
    let pred = pose17(&[
        (0, 0.56, 0.10),
        (5, 0.60, 0.30),
        (CANON_LEFT_HIP, 0.40, 0.90),
        (CANON_RIGHT_HIP, 0.60, 0.90),
    ]);
    let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);

    let (_, _, torso) = pck_at(&pred, &gt, &vis, 20, PckNormalization::TorsoDiameter);
    let (_, _, bbox) = pck_at(&pred, &gt, &vis, 20, PckNormalization::BoundingBoxDiagonal);
    let (_, _, abs) = pck_at(&pred, &gt, &vis, 20, PckNormalization::AbsolutePixels(0.08));

    assert!((torso - 0.50).abs() < 1e-6, "torso PCK 0.50, got {torso}");
    assert!((bbox - 1.00).abs() < 1e-6, "bbox PCK 1.00, got {bbox}");
    assert!((abs - 0.75).abs() < 1e-6, "abs(0.08) PCK 0.75, got {abs}");
    assert!(
        torso != bbox && bbox != abs && torso != abs,
        "three normalizations must be distinct: {torso} / {bbox} / {abs}"
    );
}

/// `accuracy_report` returns a self-describing result carrying its normalization,
/// so an unlabeled PCK number is structurally impossible at the API boundary.
#[test]
fn harness_report_carries_normalization_label() {
    let gt = pose17(&[(CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);
    let vis = vis17(&[CANON_LEFT_HIP, CANON_RIGHT_HIP]);
    let frame = PoseFrame { pred: gt.clone(), gt: gt.clone(), visibility: vis };
    let report = accuracy_report(&[frame], &[20], PckNormalization::BoundingBoxDiagonal);
    assert_eq!(report.normalization, PckNormalization::BoundingBoxDiagonal);
    assert_eq!(report.n_keypoints, 17);
    assert_eq!(report.n_frames, 1);
    assert!((report.pck(20).unwrap() - 1.0).abs() < 1e-6);
    assert!(report.summary().contains("bbox-diagonal"));
}

// ---------------------------------------------------------------------------
// Tests that use `EvalMetrics` (requires tch-backend because the metrics
// module is feature-gated in lib.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "tch-backend")]
mod eval_metrics_tests {
    use wifi_densepose_train::metrics::EvalMetrics;

    /// A freshly constructed [`EvalMetrics`] should hold exactly the values
    /// that were passed in.
    #[test]
    fn eval_metrics_stores_correct_values() {
        let m = EvalMetrics {
            mpjpe: 0.05,
            pck_at_05: 0.92,
            gps: 1.3,
        };

        assert!(
            (m.mpjpe - 0.05).abs() < 1e-12,
            "mpjpe must be 0.05, got {}",
            m.mpjpe
        );
        assert!(
            (m.pck_at_05 - 0.92).abs() < 1e-12,
            "pck_at_05 must be 0.92, got {}",
            m.pck_at_05
        );
        assert!(
            (m.gps - 1.3).abs() < 1e-12,
            "gps must be 1.3, got {}",
            m.gps
        );
    }

    /// `pck_at_05` of a perfect prediction must be 1.0.
    #[test]
    fn pck_perfect_prediction_is_one() {
        let m = EvalMetrics {
            mpjpe: 0.0,
            pck_at_05: 1.0,
            gps: 0.0,
        };
        assert!(
            (m.pck_at_05 - 1.0).abs() < 1e-9,
            "perfect prediction must yield pck_at_05 = 1.0, got {}",
            m.pck_at_05
        );
    }

    /// `pck_at_05` of a completely wrong prediction must be 0.0.
    #[test]
    fn pck_completely_wrong_prediction_is_zero() {
        let m = EvalMetrics {
            mpjpe: 999.0,
            pck_at_05: 0.0,
            gps: 999.0,
        };
        assert!(
            m.pck_at_05.abs() < 1e-9,
            "completely wrong prediction must yield pck_at_05 = 0.0, got {}",
            m.pck_at_05
        );
    }

    /// `mpjpe` must be 0.0 when predicted and GT positions are identical.
    #[test]
    fn mpjpe_perfect_prediction_is_zero() {
        let m = EvalMetrics {
            mpjpe: 0.0,
            pck_at_05: 1.0,
            gps: 0.0,
        };
        assert!(
            m.mpjpe.abs() < 1e-12,
            "perfect prediction must yield mpjpe = 0.0, got {}",
            m.mpjpe
        );
    }

    /// `mpjpe` must increase monotonically with prediction error.
    #[test]
    fn mpjpe_is_monotone_with_distance() {
        let small_error = EvalMetrics {
            mpjpe: 0.01,
            pck_at_05: 0.99,
            gps: 0.1,
        };
        let medium_error = EvalMetrics {
            mpjpe: 0.10,
            pck_at_05: 0.70,
            gps: 1.0,
        };
        let large_error = EvalMetrics {
            mpjpe: 0.50,
            pck_at_05: 0.20,
            gps: 5.0,
        };

        assert!(
            small_error.mpjpe < medium_error.mpjpe,
            "small error mpjpe must be < medium error mpjpe"
        );
        assert!(
            medium_error.mpjpe < large_error.mpjpe,
            "medium error mpjpe must be < large error mpjpe"
        );
    }

    /// GPS must be 0.0 for a perfect DensePose prediction.
    #[test]
    fn gps_perfect_prediction_is_zero() {
        let m = EvalMetrics {
            mpjpe: 0.0,
            pck_at_05: 1.0,
            gps: 0.0,
        };
        assert!(
            m.gps.abs() < 1e-12,
            "perfect prediction must yield gps = 0.0, got {}",
            m.gps
        );
    }

    /// GPS must increase monotonically as prediction quality degrades.
    #[test]
    fn gps_monotone_with_distance() {
        let perfect = EvalMetrics {
            mpjpe: 0.0,
            pck_at_05: 1.0,
            gps: 0.0,
        };
        let imperfect = EvalMetrics {
            mpjpe: 0.1,
            pck_at_05: 0.8,
            gps: 2.0,
        };
        let poor = EvalMetrics {
            mpjpe: 0.5,
            pck_at_05: 0.3,
            gps: 8.0,
        };

        assert!(
            perfect.gps < imperfect.gps,
            "perfect GPS must be < imperfect GPS"
        );
        assert!(imperfect.gps < poor.gps, "imperfect GPS must be < poor GPS");
    }
}

// ---------------------------------------------------------------------------
// Canonical PCK / OKS validation (production functions, no tch)
// ---------------------------------------------------------------------------

/// Build a 17-joint pose in `[0,1]` coordinates from an `(x, y)` per-joint list,
/// padding any unspecified joint to `(0,0)`. Returns `[17, 2]`.
fn pose17(joints: &[(usize, f32, f32)]) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((17, 2));
    for &(j, x, y) in joints {
        a[[j, 0]] = x;
        a[[j, 1]] = y;
    }
    a
}

/// Visibility vector with the listed joints visible (`2.0`), rest invisible.
fn vis17(visible: &[usize]) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(17);
    for &j in visible {
        v[j] = 2.0;
    }
    v
}

/// **Fixture test (Goal B).** The production `pck_canonical` must equal a value
/// worked out *by hand* on a constructed pose — not a reimplementation.
///
/// Construction (all coordinates in `[0,1]`):
/// * left_hip(11)  = (0.40, 0.50), right_hip(12) = (0.60, 0.50)
///   ⇒ canonical torso = hip↔hip width = 0.20.
/// * threshold = 0.2 ⇒ dist_threshold = 0.2 × 0.20 = **0.04**.
/// * Visible joints: {0 (nose), 5 (l_shoulder), 11, 12}. (4 visible.)
///   - nose(0):       pred == gt           ⇒ dist 0.00 ≤ 0.04 ⇒ CORRECT
///   - l_shoulder(5): pred off by dy=0.10  ⇒ dist 0.10 > 0.04 ⇒ wrong
///   - l_hip(11):     pred == gt           ⇒ dist 0.00 ≤ 0.04 ⇒ CORRECT
///   - r_hip(12):     pred off by dx=0.03  ⇒ dist 0.03 ≤ 0.04 ⇒ CORRECT
/// Hand result: correct = 3, total = 4, pck = 3/4 = **0.75**.
#[test]
fn canonical_pck_matches_hand_computed_fixture() {
    let gt = pose17(&[
        (0, 0.50, 0.20),  // nose
        (5, 0.35, 0.35),  // left_shoulder
        (CANON_LEFT_HIP, 0.40, 0.50),
        (CANON_RIGHT_HIP, 0.60, 0.50),
    ]);
    let pred = pose17(&[
        (0, 0.50, 0.20),  // exact
        (5, 0.35, 0.45),  // off by dy = 0.10  (> 0.04)
        (CANON_LEFT_HIP, 0.40, 0.50),  // exact
        (CANON_RIGHT_HIP, 0.63, 0.50), // off by dx = 0.03  (<= 0.04)
    ]);
    let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);

    let (correct, total, pck) = pck_canonical(&pred, &gt, &vis, 0.2);
    assert_eq!(total, 4, "4 visible joints expected, got {total}");
    assert_eq!(correct, 3, "hand-computed: 3 of 4 within 0.04, got {correct}");
    assert!(
        (pck - 0.75).abs() < 1e-6,
        "hand-computed PCK is 0.75, got {pck}"
    );
}

/// Pin the **normalizer**: PCK uses hip↔hip torso width. A prediction error of
/// 0.18 (just under 0.2 × torso=1.0 wide hips) is CORRECT, but the same error
/// is WRONG once the hips are squeezed to width 0.20 (threshold 0.04). If the
/// implementation ignored the torso normalizer this test would fail.
#[test]
fn canonical_pck_uses_hip_to_hip_torso_normalizer() {
    // Wide hips: width 1.0 ⇒ threshold 0.2. An error of 0.18 on joint 5 is OK.
    let gt_wide = pose17(&[(5, 0.50, 0.50), (CANON_LEFT_HIP, 0.0, 0.5), (CANON_RIGHT_HIP, 1.0, 0.5)]);
    let pred_wide = pose17(&[(5, 0.68, 0.50), (CANON_LEFT_HIP, 0.0, 0.5), (CANON_RIGHT_HIP, 1.0, 0.5)]);
    let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
    let (_, _, pck_wide) = pck_canonical(&pred_wide, &gt_wide, &vis, 0.2);

    // Narrow hips: width 0.20 ⇒ threshold 0.04. Same 0.18 error on joint 5 is wrong.
    let gt_narrow = pose17(&[(5, 0.50, 0.50), (CANON_LEFT_HIP, 0.40, 0.5), (CANON_RIGHT_HIP, 0.60, 0.5)]);
    let pred_narrow = pose17(&[(5, 0.68, 0.50), (CANON_LEFT_HIP, 0.40, 0.5), (CANON_RIGHT_HIP, 0.60, 0.5)]);
    let (_, _, pck_narrow) = pck_canonical(&pred_narrow, &gt_narrow, &vis, 0.2);

    // Joints 11/12 are exact (correct in both); joint 5 flips.
    // Wide: 3/3 = 1.0; Narrow: 2/3 ≈ 0.667.
    assert!((pck_wide - 1.0).abs() < 1e-6, "wide-hip PCK should be 1.0, got {pck_wide}");
    assert!(
        (pck_narrow - 2.0 / 3.0).abs() < 1e-6,
        "narrow-hip PCK should be 2/3 (joint 5 now out of tolerance), got {pck_narrow}"
    );
}

/// The claim-inflating bug: no visible joints must score **0.0**, never 1.0.
#[test]
fn canonical_pck_zero_visible_is_zero() {
    let kpts = pose17(&[(CANON_LEFT_HIP, 0.4, 0.5), (CANON_RIGHT_HIP, 0.6, 0.5)]);
    let vis = vis17(&[]); // nothing visible
    let (correct, total, pck) = pck_canonical(&kpts, &kpts, &vis, 0.2);
    assert_eq!((correct, total), (0, 0));
    assert_eq!(pck, 0.0, "no-visible-joint PCK must be 0.0 (not the old 1.0)");
}

// ---------------------------------------------------------------------------
// Canonical OKS validation (production function, no tch)
// ---------------------------------------------------------------------------

/// **Fixture test (Goal B).** A perfect prediction (pred == gt) makes every
/// Gaussian term `exp(0) = 1`, so the canonical OKS is exactly **1.0** —
/// hand-evident, independent of the (positive) scale.
#[test]
fn canonical_oks_perfect_prediction_is_one() {
    let gt = pose17(&[
        (0, 0.50, 0.20),
        (5, 0.35, 0.35),
        (CANON_LEFT_HIP, 0.40, 0.50),
        (CANON_RIGHT_HIP, 0.60, 0.50),
    ]);
    let vis = vis17(&[0, 5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
    let oks = oks_canonical(&gt, &gt, &vis);
    assert!(
        (oks - 1.0).abs() < 1e-6,
        "OKS for a perfect prediction must be 1.0, got {oks}"
    );
}

/// **The "fake Gold tier" bug, pinned (Goal B).** On normalized `[0,1]`
/// coordinates the historical `s = 1.0` path returned ≈1.0 for *any* pose.
/// Canonical derives `s` from the pose extent (here torso width = 0.20), so a
/// pose whose visible non-hip joint is off by ~3× the torso scores far below
/// the "Gold" tier. Hand bound: for joint 5 with d ≈ 0.60, s = 0.20, k = 0.079,
/// the exponent `-d²/(2 s² k²)` is enormously negative ⇒ that term ≈ 0; the two
/// (exact) hip terms give 1 each ⇒ OKS ≈ 2/3 at most, and with joint-5 ≈ 0 the
/// mean is ≈ 0.667. We assert it is comfortably **< 0.8** (and the wrong joint
/// contributes ≈ 0), i.e. nowhere near the old ≈1.0.
#[test]
fn canonical_oks_not_one_for_wrong_pose_on_normalized_coords() {
    let gt = pose17(&[
        (5, 0.30, 0.50),
        (CANON_LEFT_HIP, 0.40, 0.50),
        (CANON_RIGHT_HIP, 0.60, 0.50),
    ]);
    // Joint 5 dragged 0.60 away (3× the 0.20 torso); hips exact.
    let pred = pose17(&[
        (5, 0.90, 0.50),
        (CANON_LEFT_HIP, 0.40, 0.50),
        (CANON_RIGHT_HIP, 0.60, 0.50),
    ]);
    let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
    let oks = oks_canonical(&pred, &gt, &vis);
    assert!(
        oks < 0.8,
        "wrong-pose OKS on [0,1] coords must NOT be ≈1.0 (fake-Gold bug); got {oks}"
    );
    // The two exact hips alone give 2/3; the wrong joint must add ~nothing.
    assert!(
        (oks - 2.0 / 3.0).abs() < 0.05,
        "wrong joint should contribute ≈0 ⇒ OKS ≈ 2/3, got {oks}"
    );
}

/// Canonical OKS decreases monotonically with prediction error.
#[test]
fn canonical_oks_decreases_with_distance() {
    let gt = pose17(&[(5, 0.50, 0.50), (CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);
    let vis = vis17(&[5, CANON_LEFT_HIP, CANON_RIGHT_HIP]);
    let mk = |x5: f32| pose17(&[(5, x5, 0.50), (CANON_LEFT_HIP, 0.40, 0.50), (CANON_RIGHT_HIP, 0.60, 0.50)]);

    let oks0 = oks_canonical(&mk(0.50), &gt, &vis);
    let oks1 = oks_canonical(&mk(0.52), &gt, &vis);
    let oks2 = oks_canonical(&mk(0.60), &gt, &vis);
    assert!(oks0 > oks1, "OKS must drop as error grows: {oks0} vs {oks1}");
    assert!(oks1 > oks2, "OKS must drop as error grows: {oks1} vs {oks2}");
}

// ---------------------------------------------------------------------------
// Differential cross-check: independent reference kernel vs canonical (Goal B)
// ---------------------------------------------------------------------------

/// A deliberately *independent* PCK reference implementation in the simplest
/// regime — a **raw distance threshold** (no torso normalization). It is kept
/// only to cross-check the canonical function, not to define the metric.
fn reference_pck_raw(pred: &[(f32, f32)], gt: &[(f32, f32)], dist_threshold: f32) -> (usize, usize, f32) {
    let n = pred.len().min(gt.len());
    let mut correct = 0usize;
    for i in 0..n {
        let dx = pred[i].0 - gt[i].0;
        let dy = pred[i].1 - gt[i].1;
        if (dx * dx + dy * dy).sqrt() <= dist_threshold {
            correct += 1;
        }
    }
    let pck = if n > 0 { correct as f32 / n as f32 } else { 0.0 };
    (correct, n, pck)
}

/// **Differential test (Goal B).** In the regime where the canonical torso
/// normalizer equals 1.0 (hips exactly one unit apart, so `threshold · torso`
/// reduces to the raw `threshold`), the canonical PCK and an independent
/// raw-threshold reference kernel MUST agree on shared inputs. This catches a
/// canonical-side bug that a pure self-fixture could miss, *because* the second
/// implementation is genuinely independent.
#[test]
fn test_kernel_agrees_with_canonical() {
    // Hips one unit apart ⇒ canonical torso == 1.0 ⇒ dist_threshold == threshold.
    let gt = pose17(&[
        (0, 0.30, 0.30),
        (5, 0.55, 0.55),
        (7, 0.10, 0.90),
        (CANON_LEFT_HIP, 0.00, 0.50),
        (CANON_RIGHT_HIP, 1.00, 0.50),
    ]);
    let pred = pose17(&[
        (0, 0.31, 0.30),  // err 0.01
        (5, 0.70, 0.55),  // err 0.15
        (7, 0.10, 0.98),  // err 0.08
        (CANON_LEFT_HIP, 0.00, 0.50),  // exact
        (CANON_RIGHT_HIP, 1.00, 0.50), // exact
    ]);
    let visible = [0usize, 5, 7, CANON_LEFT_HIP, CANON_RIGHT_HIP];
    let vis = vis17(&visible);
    let threshold = 0.1_f32;

    let (c_can, t_can, pck_can) = pck_canonical(&pred, &gt, &vis, threshold);

    // Reference over the SAME visible joints with the SAME raw threshold
    // (torso == 1.0 so threshold·torso == threshold).
    let pred_v: Vec<(f32, f32)> = visible.iter().map(|&j| (pred[[j, 0]], pred[[j, 1]])).collect();
    let gt_v: Vec<(f32, f32)> = visible.iter().map(|&j| (gt[[j, 0]], gt[[j, 1]])).collect();
    let (c_ref, t_ref, pck_ref) = reference_pck_raw(&pred_v, &gt_v, threshold);

    assert_eq!(t_can, t_ref, "visible counts must match: {t_can} vs {t_ref}");
    assert_eq!(c_can, c_ref, "correct counts must match: {c_can} vs {c_ref}");
    assert!(
        (pck_can - pck_ref).abs() < 1e-6,
        "canonical PCK {pck_can} must agree with independent reference {pck_ref}"
    );
}

// ---------------------------------------------------------------------------
// Hungarian assignment tests (deterministic, hand-computed)
// ---------------------------------------------------------------------------

/// Greedy row-by-row assignment (correct for non-competing minima).
fn greedy_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    cost.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(col, _)| col)
                .unwrap_or(0)
        })
        .collect()
}

/// Identity cost matrix (0 on diagonal, 100 elsewhere) must assign i → i.
#[test]
fn hungarian_identity_cost_matrix_assigns_diagonal() {
    let n = 3_usize;
    let cost: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 0.0 } else { 100.0 }).collect())
        .collect();

    let assignment = greedy_assignment(&cost);
    assert_eq!(
        assignment,
        vec![0, 1, 2],
        "identity cost matrix must assign 0→0, 1→1, 2→2, got {:?}",
        assignment
    );
}

/// Permuted cost matrix must find the optimal (zero-cost) assignment.
#[test]
fn hungarian_permuted_cost_matrix_finds_optimal() {
    let cost: Vec<Vec<f64>> = vec![
        vec![100.0, 100.0, 0.0],
        vec![0.0, 100.0, 100.0],
        vec![100.0, 0.0, 100.0],
    ];

    let assignment = greedy_assignment(&cost);
    assert_eq!(
        assignment,
        vec![2, 0, 1],
        "permuted cost matrix must assign 0→2, 1→0, 2→1, got {:?}",
        assignment
    );
}

/// A 5×5 identity cost matrix must also be assigned correctly.
#[test]
fn hungarian_5x5_identity_matrix() {
    let n = 5_usize;
    let cost: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 0.0 } else { 999.0 }).collect())
        .collect();

    let assignment = greedy_assignment(&cost);
    assert_eq!(
        assignment,
        vec![0, 1, 2, 3, 4],
        "5×5 identity cost matrix must assign i→i: got {:?}",
        assignment
    );
}

// ---------------------------------------------------------------------------
// MetricsAccumulator tests (deterministic batch evaluation)
// ---------------------------------------------------------------------------

/// Batch PCK must be 1.0 when all predictions are exact.
#[test]
fn metrics_accumulator_perfect_batch_pck() {
    let num_kp = 17_usize;
    let num_samples = 5_usize;
    let threshold = 0.5_f64;

    let kps: Vec<[f64; 2]> = (0..num_kp)
        .map(|j| [j as f64 * 0.05, j as f64 * 0.04])
        .collect();
    let total_joints = num_samples * num_kp;

    let total_correct: usize = (0..num_samples)
        .flat_map(|_| kps.iter().zip(kps.iter()))
        .filter(|(p, g)| {
            let dx = p[0] - g[0];
            let dy = p[1] - g[1];
            (dx * dx + dy * dy).sqrt() <= threshold
        })
        .count();

    let pck = total_correct as f64 / total_joints as f64;
    assert!(
        (pck - 1.0).abs() < 1e-9,
        "batch PCK for all-correct pairs must be 1.0, got {pck}"
    );
}

/// Accumulating 50% correct and 50% wrong predictions must yield PCK = 0.5.
#[test]
fn metrics_accumulator_is_additive_half_correct() {
    let threshold = 0.05_f64;
    let gt_kp = [0.5_f64, 0.5_f64];
    let wrong_kp = [10.0_f64, 10.0_f64];

    // 3 correct + 3 wrong = 6 total.
    let pairs: Vec<([f64; 2], [f64; 2])> = (0..6)
        .map(|i| {
            if i < 3 {
                (gt_kp, gt_kp)
            } else {
                (wrong_kp, gt_kp)
            }
        })
        .collect();

    let correct: usize = pairs
        .iter()
        .filter(|(pred, gt)| {
            let dx = pred[0] - gt[0];
            let dy = pred[1] - gt[1];
            (dx * dx + dy * dy).sqrt() <= threshold
        })
        .count();

    let pck = correct as f64 / pairs.len() as f64;
    assert!(
        (pck - 0.5).abs() < 1e-9,
        "50% correct pairs must yield PCK = 0.5, got {pck}"
    );
}
