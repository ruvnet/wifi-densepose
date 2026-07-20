//! Masked-autoencoder (MAE) pretraining recipe for the ADR-150 RF foundation
//! encoder — ADR-152 §2.3 (amends ADR-150 §2.3).
//!
//! Implements the *measured* tokenization recipe from the UNSW MAE pretraining
//! study (arXiv [2511.18792](https://arxiv.org/abs/2511.18792), Nov 2025), the
//! largest heterogeneous CSI pretraining run to date (1,320,892 samples, 14
//! public datasets, 4 devices, 2.4/5/6 GHz, 20–160 MHz):
//!
//! - **80% masking ratio** over the patch grid.
//! - **Small (30, 3) patches** — 30 time steps × 3 subcarriers — measured
//!   **+4.7%** over (40, 5) patches by preserving fine temporal dynamics.
//! - Encoder capacity stays **ViT-Small-class (~15M params)**: ViT-Base adds
//!   only +0.4–0.9% over ViT-Small in-study, corroborating ADR-150's own
//!   finding that capacity hurts cross-subject transfer.
//! - Unseen-domain performance scales **log-linearly with pretraining data,
//!   unsaturated at 1.3M samples** — data aggregation outranks architecture
//!   work (ADR-152 §2.3).
//!
//! This module provides the GPU-free half of the recipe: configuration,
//! patchification, and deterministic random masking. The (future, ADR-150)
//! encoder consumes [`PatchGrid`] + [`MaskIndices`] to compute the masked
//! reconstruction loss (`L_masked_csi` in ADR-150 §2.3's loss stack).
//!
//! ## Axis convention
//!
//! A CSI window is `time × subcarriers`, row-major (`index = t * subc + sc`),
//! matching the crate's `[T, …, n_sc]` dataset layout (time first, subcarriers
//! last) and the UNSW "(30 time steps, 3 subcarriers)" patch framing. Patches
//! are indexed row-major over the patch grid (`p = pt * n_patches_subc + ps`),
//! and values within a patch are row-major time-major
//! (`local = lt * patch_subc + lsc`).
//!
//! ## Divisibility policy: error, never truncate
//!
//! Window dimensions **must** be exact multiples of the patch dimensions.
//! Non-divisible shapes return [`MaeError::NotDivisible`] instead of silently
//! truncating trailing samples (this crate never silently drops data). The
//! error names the largest divisible crop; use
//! [`MaePretrainConfig::cropped_window_shape`] to compute it and crop
//! explicitly before calling [`patchify`].
//!
//! ## Example
//!
//! ```rust
//! use wifi_densepose_train::mae::MaePretrainConfig;
//!
//! let cfg = MaePretrainConfig::default(); // 0.80 masking, (30, 3) patches
//! cfg.validate().expect("default recipe is valid");
//!
//! // 90 frames × 54 subcarriers → a 3 × 18 grid of (30, 3) patches.
//! let window = vec![0.25_f32; 90 * 54];
//! let (grid, mask) = cfg.mask_window(&window, 90, 54).unwrap();
//! assert_eq!(grid.n_patches(), 54);
//! assert_eq!(mask.masked.len(), 43); // round(0.80 * 54)
//! assert_eq!(mask.visible.len(), 11);
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, MaeError};
use crate::virtual_aug::Xorshift64;

// ---------------------------------------------------------------------------
// MaePretrainConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters for masked-CSI pretraining (ADR-152 §2.3).
///
/// Defaults are the measured-optimal UNSW recipe (arXiv 2511.18792); change
/// them only with benchmark evidence. Serializable so the recipe is recorded
/// in checkpoint metadata alongside [`crate::config::TrainingConfig`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaePretrainConfig {
    /// Fraction of patches hidden from the encoder, in `(0, 1)`.
    ///
    /// Default: **0.80** (UNSW measured optimum).
    pub mask_ratio: f64,

    /// Patch extent along the time axis, in frames. Default: **30**.
    pub patch_time: usize,

    /// Patch extent along the subcarrier axis. Default: **3**.
    pub patch_subc: usize,

    /// Base seed for the deterministic mask sampler. Default: **42**.
    ///
    /// For per-sample masks derive a child seed (e.g.
    /// `seed ^ sample_idx as u64`) and pass it to [`random_mask`]; reusing one
    /// seed yields the identical mask for every sample.
    pub seed: u64,
}

impl Default for MaePretrainConfig {
    fn default() -> Self {
        MaePretrainConfig {
            mask_ratio: 0.80,
            patch_time: 30,
            patch_subc: 3,
            seed: 42,
        }
    }
}

impl MaePretrainConfig {
    /// Validate the shape-independent fields.
    ///
    /// # Validated invariants
    ///
    /// - `mask_ratio` must be strictly inside `(0, 1)` and finite.
    /// - `patch_time` and `patch_subc` must be at least 1.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.mask_ratio.is_finite() || self.mask_ratio <= 0.0 || self.mask_ratio >= 1.0 {
            return Err(ConfigError::invalid_value(
                "mask_ratio",
                format!("must be in (0.0, 1.0), got {}", self.mask_ratio),
            ));
        }
        if self.patch_time == 0 {
            return Err(ConfigError::invalid_value("patch_time", "must be >= 1"));
        }
        if self.patch_subc == 0 {
            return Err(ConfigError::invalid_value("patch_subc", "must be >= 1"));
        }
        Ok(())
    }

    /// Check this recipe against a concrete `time × subc` window shape.
    ///
    /// Errors if a patch dimension exceeds the window or if either axis is
    /// not an exact multiple of the patch extent (divisibility policy above).
    pub fn validate_for_window(&self, time: usize, subc: usize) -> Result<(), MaeError> {
        check_axis("time", time, self.patch_time)?;
        check_axis("subcarrier", subc, self.patch_subc)?;
        Ok(())
    }

    /// Largest `(time, subc)` crop of the given window that is exactly
    /// divisible by the patch dimensions. Either component may be 0 when the
    /// window is smaller than one patch.
    #[must_use]
    pub fn cropped_window_shape(&self, time: usize, subc: usize) -> (usize, usize) {
        (
            (time / self.patch_time) * self.patch_time,
            (subc / self.patch_subc) * self.patch_subc,
        )
    }

    /// Number of patches a `time × subc` window yields under this recipe.
    pub fn num_patches(&self, time: usize, subc: usize) -> Result<usize, MaeError> {
        self.validate_for_window(time, subc)?;
        Ok((time / self.patch_time) * (subc / self.patch_subc))
    }

    /// Exact number of masked patches for a grid of `n_patches`:
    /// `round(mask_ratio * n_patches)`, clamped to `[0, n_patches]`.
    #[must_use]
    pub fn num_masked(&self, n_patches: usize) -> usize {
        ((self.mask_ratio * n_patches as f64).round() as usize).min(n_patches)
    }

    /// Patchify `window` and draw the deterministic random mask in one step,
    /// using `self.seed`. See [`patchify`] and [`random_mask`].
    ///
    /// # Errors
    ///
    /// Everything [`patchify`] rejects, plus [`MaeError::InvalidMaskRatio`]
    /// if `self.mask_ratio` is not finite or outside `(0, 1)` (the
    /// [`Self::validate`] rule) — a NaN ratio must never silently mask zero
    /// patches.
    pub fn mask_window(
        &self,
        window: &[f32],
        time: usize,
        subc: usize,
    ) -> Result<(PatchGrid, MaskIndices), MaeError> {
        let grid = patchify(window, time, subc, self)?;
        let mask = random_mask(grid.n_patches(), self.mask_ratio, self.seed)?;
        Ok((grid, mask))
    }
}

// ---------------------------------------------------------------------------
// PatchGrid / MaskIndices
// ---------------------------------------------------------------------------

/// A CSI window decomposed into non-overlapping `patch_time × patch_subc`
/// patches (see the module-level axis convention).
#[derive(Debug, Clone, PartialEq)]
pub struct PatchGrid {
    /// Patch extent along the time axis.
    pub patch_time: usize,
    /// Patch extent along the subcarrier axis.
    pub patch_subc: usize,
    /// Number of patch rows (`time / patch_time`).
    pub n_patches_time: usize,
    /// Number of patch columns (`subc / patch_subc`).
    pub n_patches_subc: usize,
    /// Flattened patches, row-major over the grid; each inner `Vec` is one
    /// patch of length `patch_time * patch_subc`, row-major time-major.
    pub patches: Vec<Vec<f32>>,
}

impl PatchGrid {
    /// Total number of patches in the grid.
    #[must_use]
    pub fn n_patches(&self) -> usize {
        self.n_patches_time * self.n_patches_subc
    }

    /// Number of scalar values per patch.
    #[must_use]
    pub fn patch_len(&self) -> usize {
        self.patch_time * self.patch_subc
    }

    /// Window shape `(time, subc)` this grid reconstructs to.
    #[must_use]
    pub fn window_shape(&self) -> (usize, usize) {
        (
            self.n_patches_time * self.patch_time,
            self.n_patches_subc * self.patch_subc,
        )
    }
}

/// Sorted, disjoint patch-index sets produced by [`random_mask`]. Together
/// they cover `0..n_patches` exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskIndices {
    /// Indices of patches hidden from the encoder (`round(ratio * n)` of them).
    pub masked: Vec<usize>,
    /// Indices of patches the encoder sees.
    pub visible: Vec<usize>,
}

// ---------------------------------------------------------------------------
// patchify / unpatchify
// ---------------------------------------------------------------------------

/// Decompose a row-major `time × subc` CSI window into the patch grid defined
/// by `cfg`.
///
/// # Errors
///
/// - [`MaeError::WindowShapeMismatch`] if `window.len() != time * subc`.
/// - [`MaeError::PatchExceedsWindow`] / [`MaeError::NotDivisible`] per the
///   module-level divisibility policy.
/// - [`MaeError::NonFiniteValue`] on the first NaN/±inf encountered —
///   corrupted CSI must be cleaned upstream, never masked over (cf. the
///   WiFlow-STD NaN-poisoning incident, ADR-152 §2.2).
pub fn patchify(
    window: &[f32],
    time: usize,
    subc: usize,
    cfg: &MaePretrainConfig,
) -> Result<PatchGrid, MaeError> {
    let expected = time * subc;
    if window.len() != expected {
        return Err(MaeError::WindowShapeMismatch {
            time,
            subc,
            expected,
            actual: window.len(),
        });
    }
    cfg.validate_for_window(time, subc)?;
    if let Some(idx) = window.iter().position(|v| !v.is_finite()) {
        return Err(MaeError::NonFiniteValue {
            row: idx / subc,
            col: idx % subc,
            value: window[idx],
        });
    }

    let n_patches_time = time / cfg.patch_time;
    let n_patches_subc = subc / cfg.patch_subc;
    let mut patches = Vec::with_capacity(n_patches_time * n_patches_subc);
    for pt in 0..n_patches_time {
        for ps in 0..n_patches_subc {
            let mut patch = Vec::with_capacity(cfg.patch_time * cfg.patch_subc);
            for lt in 0..cfg.patch_time {
                let t = pt * cfg.patch_time + lt;
                let row_start = t * subc + ps * cfg.patch_subc;
                patch.extend_from_slice(&window[row_start..row_start + cfg.patch_subc]);
            }
            patches.push(patch);
        }
    }

    Ok(PatchGrid {
        patch_time: cfg.patch_time,
        patch_subc: cfg.patch_subc,
        n_patches_time,
        n_patches_subc,
        patches,
    })
}

/// Reassemble the full row-major `time × subc` window from a [`PatchGrid`].
/// Exact inverse of [`patchify`].
#[must_use]
pub fn unpatchify(grid: &PatchGrid) -> Vec<f32> {
    unpatchify_select(grid, None, 0.0)
}

/// Reassemble the window keeping only the patches listed in `visible`;
/// every other patch's region is filled with `fill` (the standard MAE
/// "visible tokens + mask token" view of the input).
#[must_use]
pub fn unpatchify_visible(grid: &PatchGrid, visible: &[usize], fill: f32) -> Vec<f32> {
    unpatchify_select(grid, Some(visible), fill)
}

fn unpatchify_select(grid: &PatchGrid, keep: Option<&[usize]>, fill: f32) -> Vec<f32> {
    let (time, subc) = grid.window_shape();
    let mut window = vec![fill; time * subc];
    for (p, patch) in grid.patches.iter().enumerate() {
        if let Some(keep) = keep {
            if !keep.contains(&p) {
                continue;
            }
        }
        let pt = p / grid.n_patches_subc;
        let ps = p % grid.n_patches_subc;
        for lt in 0..grid.patch_time {
            let t = pt * grid.patch_time + lt;
            let row_start = t * subc + ps * grid.patch_subc;
            let local_start = lt * grid.patch_subc;
            window[row_start..row_start + grid.patch_subc]
                .copy_from_slice(&patch[local_start..local_start + grid.patch_subc]);
        }
    }
    window
}

// ---------------------------------------------------------------------------
// random_mask
// ---------------------------------------------------------------------------

/// Draw a deterministic random mask over `n_patches` patches.
///
/// Exactly `round(mask_ratio * n_patches)` patches (clamped to
/// `[0, n_patches]`) are masked, chosen by a seeded Fisher–Yates shuffle
/// ([`Xorshift64`]), so the same `(n_patches, mask_ratio, seed)` triple always
/// yields the same mask. Both index lists are sorted ascending, disjoint, and
/// together cover `0..n_patches`.
///
/// # Errors
///
/// [`MaeError::InvalidMaskRatio`] if `mask_ratio` is not finite or outside
/// the open interval `(0, 1)` — the same rule as
/// [`MaePretrainConfig::validate`]. Erroring (never clamping) keeps the
/// module's error-not-silent policy: a NaN ratio would otherwise silently
/// mask zero patches and a ratio ≥ 1 would mask everything.
pub fn random_mask(n_patches: usize, mask_ratio: f64, seed: u64) -> Result<MaskIndices, MaeError> {
    if !mask_ratio.is_finite() || mask_ratio <= 0.0 || mask_ratio >= 1.0 {
        return Err(MaeError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_masked = ((mask_ratio * n_patches as f64).round() as usize).min(n_patches);
    let mut order: Vec<usize> = (0..n_patches).collect();
    let mut rng = Xorshift64::new(seed);
    for i in (1..n_patches).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut masked: Vec<usize> = order[..n_masked].to_vec();
    let mut visible: Vec<usize> = order[n_masked..].to_vec();
    masked.sort_unstable();
    visible.sort_unstable();
    Ok(MaskIndices { masked, visible })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn check_axis(axis: &'static str, window: usize, patch: usize) -> Result<(), MaeError> {
    if patch > window {
        return Err(MaeError::PatchExceedsWindow {
            axis,
            patch,
            window,
        });
    }
    let remainder = window % patch;
    if remainder != 0 {
        return Err(MaeError::NotDivisible {
            axis,
            window,
            patch,
            remainder,
            crop: window - remainder,
        });
    }
    Ok(())
}
