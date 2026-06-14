//! Subcarrier interpolation and selection utilities.
//!
//! This module provides functions to resample CSI subcarrier arrays between
//! different subcarrier counts using linear interpolation, and to select
//! the most informative subcarriers based on signal variance.
//!
//! # Example
//!
//! ```rust
//! use wifi_densepose_train::subcarrier::interpolate_subcarriers;
//! use ndarray::Array4;
//!
//! // Resample from 114 → 56 subcarriers
//! let arr = Array4::<f32>::zeros((100, 3, 3, 114));
//! let resampled = interpolate_subcarriers(&arr, 56);
//! assert_eq!(resampled.shape(), &[100, 3, 3, 56]);
//! ```

use ndarray::{s, Array4};
use ruvector_solver::neumann::NeumannSolver;
use ruvector_solver::types::CsrMatrix;

// --- Sparse-interpolation tuning constants (ADR-155 M2 §8: de-magicked from
// bare literals in `interpolate_subcarriers_sparse`; values bit-identical to the
// prior inline literals — documentation only, no behaviour change). ---

/// Gaussian-basis width (in the normalised `[0,1]` subcarrier position space)
/// for the sparse-interpolation kernel `exp(-Δ²/σ²)`. Wider σ ⇒ smoother fit.
const SPARSE_BASIS_SIGMA: f32 = 0.15;

/// Sparsity cutoff: basis entries below this magnitude are dropped from the
/// normal-equations assembly, keeping `AᵀA` sparse.
const SPARSE_BASIS_THRESHOLD: f32 = 1e-4;

/// Tikhonov regularisation strength `λ` added to the `AᵀA` diagonal for
/// numerical stability of the (possibly ill-conditioned) normal equations.
const SPARSE_REGULARIZATION_LAMBDA: f32 = 0.1;

/// Magnitude below which an assembled `AᵀA` entry is treated as structurally
/// zero and omitted from the COO triplet list.
const SPARSE_COO_PRUNE_EPS: f32 = 1e-8;

/// Convergence tolerance for the Neumann-series sparse solver (`f64` to match
/// [`NeumannSolver::new`]).
const SPARSE_SOLVER_TOL: f64 = 1e-5;

/// Maximum Neumann-series iterations before the solver returns (falls back to
/// linear interpolation on non-convergence).
const SPARSE_SOLVER_MAX_ITERS: usize = 500;

// ---------------------------------------------------------------------------
// interpolate_subcarriers
// ---------------------------------------------------------------------------

/// Resample a 4-D CSI array along the subcarrier axis (last dimension) to
/// `target_sc` subcarriers using linear interpolation.
///
/// # Arguments
///
/// - `arr`: Input array with shape `[T, n_tx, n_rx, n_sc]`.
/// - `target_sc`: Number of output subcarriers.
///
/// # Returns
///
/// A new array with shape `[T, n_tx, n_rx, target_sc]`.
///
/// # Panics
///
/// Panics if `target_sc == 0` or the input has no subcarrier dimension.
///
/// Non-contiguous inputs (e.g. a transposed or strided view) are handled
/// gracefully: the subcarrier lane is copied into a contiguous scratch buffer
/// when the underlying storage is not contiguous, so this function never
/// panics on layout (ADR-155 §Tier-2).
pub fn interpolate_subcarriers(arr: &Array4<f32>, target_sc: usize) -> Array4<f32> {
    assert!(target_sc > 0, "target_sc must be > 0");

    let shape = arr.shape();
    let (n_t, n_tx, n_rx, n_sc) = (shape[0], shape[1], shape[2], shape[3]);

    if n_sc == target_sc {
        return arr.clone();
    }

    let mut out = Array4::<f32>::zeros((n_t, n_tx, n_rx, target_sc));

    // Precompute interpolation weights once.
    let weights = compute_interp_weights(n_sc, target_sc);

    // Reusable scratch buffer for the non-contiguous fallback path.
    let mut scratch: Vec<f32> = Vec::new();

    for t in 0..n_t {
        for tx in 0..n_tx {
            for rx in 0..n_rx {
                let src = arr.slice(s![t, tx, rx, ..]);
                // Prefer the contiguous fast path; fall back to an owned copy
                // for non-contiguous layouts instead of panicking.
                let src_slice: &[f32] = match src.as_slice() {
                    Some(s) => s,
                    None => {
                        scratch.clear();
                        scratch.extend(src.iter().copied());
                        &scratch
                    }
                };

                for (k, &(i0, i1, w)) in weights.iter().enumerate() {
                    let v = src_slice[i0] * (1.0 - w) + src_slice[i1] * w;
                    out[[t, tx, rx, k]] = v;
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// compute_interp_weights
// ---------------------------------------------------------------------------

/// Compute linear interpolation indices and fractional weights for resampling
/// from `src_sc` to `target_sc` subcarriers.
///
/// Returns a `Vec` of `(i0, i1, frac)` tuples where each output subcarrier `k`
/// is computed as `src[i0] * (1 - frac) + src[i1] * frac`.
///
/// # Arguments
///
/// - `src_sc`: Number of subcarriers in the source array.
/// - `target_sc`: Number of subcarriers in the output array.
///
/// # Panics
///
/// Panics if `src_sc == 0` or `target_sc == 0`.
pub fn compute_interp_weights(src_sc: usize, target_sc: usize) -> Vec<(usize, usize, f32)> {
    assert!(src_sc > 0, "src_sc must be > 0");
    assert!(target_sc > 0, "target_sc must be > 0");

    let mut weights = Vec::with_capacity(target_sc);

    for k in 0..target_sc {
        // Map output index k to a continuous position in the source array.
        // Scale so that index 0 maps to 0 and index (target_sc-1) maps to
        // (src_sc-1) — i.e., endpoints are preserved.
        let pos = if target_sc == 1 {
            0.0f32
        } else {
            k as f32 * (src_sc - 1) as f32 / (target_sc - 1) as f32
        };

        let i0 = (pos.floor() as usize).min(src_sc - 1);
        let i1 = (pos.ceil() as usize).min(src_sc - 1);
        let frac = pos - pos.floor();

        weights.push((i0, i1, frac));
    }

    weights
}

// ---------------------------------------------------------------------------
// interpolate_subcarriers_sparse
// ---------------------------------------------------------------------------

/// Resample CSI subcarriers using sparse regularized least-squares (ruvector-solver).
///
/// Models the CSI spectrum as a sparse combination of Gaussian basis functions
/// evaluated at source-subcarrier positions, physically motivated by multipath
/// propagation (each received component corresponds to a sparse set of delays).
///
/// The interpolation solves: `A·x ≈ b`
/// - `b`: CSI amplitude at source subcarrier positions `[src_sc]`
/// - `A`: Gaussian basis matrix `[src_sc, target_sc]` — each row j is the
///   Gaussian kernel `exp(-||target_k - src_j||^2 / sigma^2)` for each k
/// - `x`: target subcarrier values (to be solved)
///
/// A regularization term `λI` is added to A^T·A for numerical stability.
///
/// Falls back to linear interpolation on solver error.
///
/// # Performance
///
/// O(√n_sc) iterations for n_sc subcarriers via Neumann series solver.
pub fn interpolate_subcarriers_sparse(arr: &Array4<f32>, target_sc: usize) -> Array4<f32> {
    assert!(target_sc > 0, "target_sc must be > 0");

    let shape = arr.shape();
    let (n_t, n_tx, n_rx, n_sc) = (shape[0], shape[1], shape[2], shape[3]);

    if n_sc == target_sc {
        return arr.clone();
    }

    // Build the Gaussian basis matrix A: [src_sc, target_sc]
    // A[j, k] = exp(-((j/(n_sc-1) - k/(target_sc-1))^2) / sigma^2)
    let sigma = SPARSE_BASIS_SIGMA;
    let sigma_sq = sigma * sigma;

    // Source and target normalized positions in [0, 1]
    let src_pos: Vec<f32> = (0..n_sc)
        .map(|j| {
            if n_sc == 1 {
                0.0
            } else {
                j as f32 / (n_sc - 1) as f32
            }
        })
        .collect();
    let tgt_pos: Vec<f32> = (0..target_sc)
        .map(|k| {
            if target_sc == 1 {
                0.0
            } else {
                k as f32 / (target_sc - 1) as f32
            }
        })
        .collect();

    // Only include entries above a sparsity threshold
    let threshold = SPARSE_BASIS_THRESHOLD;

    // Build A^T A + λI regularized system for normal equations
    // We solve: (A^T A + λI) x = A^T b
    // A^T A is [target_sc × target_sc]
    let lambda = SPARSE_REGULARIZATION_LAMBDA;
    let mut ata_coo: Vec<(usize, usize, f32)> = Vec::new();

    // Compute A^T A
    // (A^T A)[k1, k2] = sum_j A[j,k1] * A[j,k2]
    // This is dense but small (target_sc × target_sc, typically 56×56)
    let mut ata = vec![vec![0.0_f32; target_sc]; target_sc];
    #[allow(clippy::needless_range_loop)]
    for j in 0..n_sc {
        for k1 in 0..target_sc {
            let diff1 = src_pos[j] - tgt_pos[k1];
            let a_jk1 = (-diff1 * diff1 / sigma_sq).exp();
            if a_jk1 < threshold {
                continue;
            }
            for k2 in 0..target_sc {
                let diff2 = src_pos[j] - tgt_pos[k2];
                let a_jk2 = (-diff2 * diff2 / sigma_sq).exp();
                if a_jk2 < threshold {
                    continue;
                }
                ata[k1][k2] += a_jk1 * a_jk2;
            }
        }
    }

    // Add λI regularization and convert to COO
    for (k, row) in ata.iter().enumerate() {
        for (k2, &cell) in row.iter().enumerate() {
            let val = cell + if k == k2 { lambda } else { 0.0 };
            if val.abs() > SPARSE_COO_PRUNE_EPS {
                ata_coo.push((k, k2, val));
            }
        }
    }

    // Build CsrMatrix for the normal equations system (A^T A + λI)
    let normal_matrix = CsrMatrix::<f32>::from_coo(target_sc, target_sc, ata_coo);
    let solver = NeumannSolver::new(SPARSE_SOLVER_TOL, SPARSE_SOLVER_MAX_ITERS);

    let mut out = Array4::<f32>::zeros((n_t, n_tx, n_rx, target_sc));

    for t in 0..n_t {
        for tx in 0..n_tx {
            for rx in 0..n_rx {
                let src_slice: Vec<f32> = (0..n_sc).map(|s| arr[[t, tx, rx, s]]).collect();

                // Compute A^T b [target_sc]
                let mut atb = vec![0.0_f32; target_sc];
                for j in 0..n_sc {
                    let b_j = src_slice[j];
                    for k in 0..target_sc {
                        let diff = src_pos[j] - tgt_pos[k];
                        let a_jk = (-diff * diff / sigma_sq).exp();
                        if a_jk > threshold {
                            atb[k] += a_jk * b_j;
                        }
                    }
                }

                // Solve (A^T A + λI) x = A^T b
                match solver.solve(&normal_matrix, &atb) {
                    Ok(result) => {
                        for k in 0..target_sc {
                            out[[t, tx, rx, k]] = result.solution[k];
                        }
                    }
                    Err(_) => {
                        // Fallback to linear interpolation
                        let weights = compute_interp_weights(n_sc, target_sc);
                        for (k, &(i0, i1, w)) in weights.iter().enumerate() {
                            out[[t, tx, rx, k]] = src_slice[i0] * (1.0 - w) + src_slice[i1] * w;
                        }
                    }
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// select_subcarriers_by_variance
// ---------------------------------------------------------------------------

/// Select the `k` most informative subcarrier indices based on temporal variance.
///
/// Computes the variance of each subcarrier across the time and antenna
/// dimensions, then returns the indices of the `k` subcarriers with the
/// highest variance, sorted in ascending order.
///
/// # Arguments
///
/// - `arr`: Input array with shape `[T, n_tx, n_rx, n_sc]`.
/// - `k`: Number of subcarriers to select.
///
/// # Returns
///
/// A `Vec<usize>` of length `k` with the selected subcarrier indices (ascending).
///
/// # Panics
///
/// Panics if `k == 0` or `k > n_sc`.
pub fn select_subcarriers_by_variance(arr: &Array4<f32>, k: usize) -> Vec<usize> {
    let shape = arr.shape();
    let n_sc = shape[3];

    assert!(k > 0, "k must be > 0");
    assert!(k <= n_sc, "k ({k}) must be <= n_sc ({n_sc})");

    let total_elems = shape[0] * shape[1] * shape[2];

    // Compute mean per subcarrier.
    let mut means = vec![0.0f64; n_sc];
    for (sc, mean_sc) in means.iter_mut().enumerate() {
        let col = arr.slice(s![.., .., .., sc]);
        let sum: f64 = col.iter().map(|&v| v as f64).sum();
        *mean_sc = sum / total_elems as f64;
    }

    // Compute variance per subcarrier.
    let mut variances = vec![0.0f64; n_sc];
    for sc in 0..n_sc {
        let col = arr.slice(s![.., .., .., sc]);
        let mean = means[sc];
        let var: f64 =
            col.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / total_elems as f64;
        variances[sc] = var;
    }

    // Rank subcarriers by descending variance.
    let mut ranked: Vec<usize> = (0..n_sc).collect();
    ranked.sort_by(|&a, &b| {
        variances[b]
            .partial_cmp(&variances[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top-k and sort ascending for a canonical representation.
    let mut selected: Vec<usize> = ranked[..k].to_vec();
    selected.sort_unstable();
    selected
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// ADR-155 M2 §8: the de-magicked sparse-interpolation consts must equal the
    /// prior inline literals exactly (operating-value guard).
    #[test]
    fn sparse_interp_consts_unchanged_from_literals() {
        assert_eq!(SPARSE_BASIS_SIGMA, 0.15_f32);
        assert_eq!(SPARSE_BASIS_THRESHOLD, 1e-4_f32);
        assert_eq!(SPARSE_REGULARIZATION_LAMBDA, 0.1_f32);
        assert_eq!(SPARSE_COO_PRUNE_EPS, 1e-8_f32);
        assert_eq!(SPARSE_SOLVER_TOL, 1e-5_f64);
        assert_eq!(SPARSE_SOLVER_MAX_ITERS, 500);
    }

    /// Characterize the `target_sc == 1` boundary of `compute_interp_weights`:
    /// the single output maps to source index 0 with zero fraction (the special
    /// branch that avoids dividing by `target_sc - 1 == 0`).
    #[test]
    fn compute_interp_weights_single_target_is_index_zero() {
        let w = compute_interp_weights(7, 1);
        assert_eq!(w.len(), 1);
        let (i0, i1, frac) = w[0];
        assert_eq!(i0, 0);
        assert_eq!(i1, 0);
        assert_abs_diff_eq!(frac, 0.0_f32, epsilon = 1e-6);
    }

    /// Characterize sparse interpolation to a single subcarrier: must produce
    /// the right shape and a finite value (exercises the `target_sc == 1`
    /// normalized-position branch).
    #[test]
    fn sparse_interp_single_target_is_finite() {
        let arr = Array4::<f32>::from_shape_fn((2, 1, 1, 8), |(_, _, _, k)| k as f32);
        let out = interpolate_subcarriers_sparse(&arr, 1);
        assert_eq!(out.shape(), &[2, 1, 1, 1]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn identity_resample() {
        let arr =
            Array4::<f32>::from_shape_fn((4, 3, 3, 56), |(t, tx, rx, k)| (t + tx + rx + k) as f32);
        let out = interpolate_subcarriers(&arr, 56);
        assert_eq!(out.shape(), arr.shape());
        // Identity resample must preserve all values exactly.
        for v in arr.iter().zip(out.iter()) {
            assert_abs_diff_eq!(v.0, v.1, epsilon = 1e-6);
        }
    }

    #[test]
    fn upsample_endpoints_preserved() {
        // When resampling from 4 → 8 the first and last values are exact.
        let arr = Array4::<f32>::from_shape_fn((1, 1, 1, 4), |(_, _, _, k)| k as f32);
        let out = interpolate_subcarriers(&arr, 8);
        assert_eq!(out.shape(), &[1, 1, 1, 8]);
        assert_abs_diff_eq!(out[[0, 0, 0, 0]], 0.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 0, 0, 7]], 3.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn downsample_endpoints_preserved() {
        // Downsample from 8 → 4.
        let arr = Array4::<f32>::from_shape_fn((1, 1, 1, 8), |(_, _, _, k)| k as f32 * 2.0);
        let out = interpolate_subcarriers(&arr, 4);
        assert_eq!(out.shape(), &[1, 1, 1, 4]);
        // First value: 0.0, last value: 14.0
        assert_abs_diff_eq!(out[[0, 0, 0, 0]], 0.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, 0, 0, 3]], 14.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn compute_interp_weights_identity() {
        let w = compute_interp_weights(5, 5);
        assert_eq!(w.len(), 5);
        for (k, &(i0, i1, frac)) in w.iter().enumerate() {
            assert_eq!(i0, k);
            assert_eq!(i1, k);
            assert_abs_diff_eq!(frac, 0.0_f32, epsilon = 1e-6);
        }
    }

    #[test]
    fn select_subcarriers_returns_correct_count() {
        let arr = Array4::<f32>::from_shape_fn((10, 3, 3, 56), |(t, _, _, k)| (t * k) as f32);
        let selected = select_subcarriers_by_variance(&arr, 8);
        assert_eq!(selected.len(), 8);
    }

    #[test]
    fn select_subcarriers_sorted_ascending() {
        let arr = Array4::<f32>::from_shape_fn((10, 3, 3, 56), |(t, _, _, k)| (t * k) as f32);
        let selected = select_subcarriers_by_variance(&arr, 10);
        for w in selected.windows(2) {
            assert!(w[0] < w[1], "Indices must be sorted ascending");
        }
    }

    #[test]
    fn select_subcarriers_all_same_returns_all() {
        // When all subcarriers have zero variance, the function should still
        // return k valid indices.
        let arr = Array4::<f32>::ones((5, 2, 2, 20));
        let selected = select_subcarriers_by_variance(&arr, 5);
        assert_eq!(selected.len(), 5);
        // All selected indices must be in [0, 19]
        for &idx in &selected {
            assert!(idx < 20);
        }
    }

    #[test]
    fn sparse_interpolation_114_to_56_shape() {
        let arr = Array4::<f32>::from_shape_fn((4, 1, 3, 114), |(t, _, rx, k)| {
            ((t + rx + k) as f32).sin()
        });
        let out = interpolate_subcarriers_sparse(&arr, 56);
        assert_eq!(out.shape(), &[4, 1, 3, 56]);
    }

    // ADR-155 §Tier-2: a non-contiguous input (subcarrier axis strided after an
    // axis permutation) must NOT panic — the old `.as_slice().unwrap_or_else(||
    // panic!(...))` path crashed on any non-contiguous layout.
    #[test]
    fn non_contiguous_input_does_not_panic() {
        // Build a [t, sc, tx, rx] array, then permute so subcarriers land in the
        // last axis. The resulting owned Array4 has non-standard strides, so its
        // last-axis lanes are non-contiguous in memory.
        let base =
            Array4::<f32>::from_shape_fn((4, 8, 3, 3), |(t, sc, tx, rx)| (t + sc + tx + rx) as f32);
        // permuted_axes consumes the owned array and returns an owned Array4
        // with swapped strides: logical shape [t, tx, rx, sc], sc axis strided.
        let strided: Array4<f32> = base.permuted_axes([0, 2, 3, 1]);
        // Sanity: a last-axis lane really is non-contiguous.
        assert!(strided.slice(s![0, 0, 0, ..]).as_slice().is_none());

        let out = interpolate_subcarriers(&strided, 4);
        assert_eq!(out.shape(), &[4, 3, 3, 4]);
        // Endpoints preserved exactly even via the fallback copy path.
        for tx in 0..3 {
            for rx in 0..3 {
                let first = strided[[0, tx, rx, 0]];
                let last = strided[[0, tx, rx, 7]];
                assert_abs_diff_eq!(out[[0, tx, rx, 0]], first, epsilon = 1e-5);
                assert_abs_diff_eq!(out[[0, tx, rx, 3]], last, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn sparse_interpolation_identity() {
        // For same source and target count, should return same array
        let arr = Array4::<f32>::from_shape_fn((2, 1, 1, 20), |(_, _, _, k)| k as f32);
        let out = interpolate_subcarriers_sparse(&arr, 20);
        assert_eq!(out.shape(), &[2, 1, 1, 20]);
    }
}
