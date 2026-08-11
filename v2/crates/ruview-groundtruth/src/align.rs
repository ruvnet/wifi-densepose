//! Deterministic time alignment (ADR-300 §2, generalizing ADR-290).
//!
//! Estimate and reference series rarely share a clock. This module recovers a
//! **constant offset** by resampling both series onto a common grid
//! (nearest-sample, never bridging gaps larger than a configured limit) and
//! searching a bounded lag window for the offset that best aligns them:
//! normalized cross-correlation for continuous measurands, label-agreement
//! fraction for categorical ones. Every step is deterministic — no wall clock,
//! no randomness — and the chosen offset is *reported*, never silently applied.

use serde::{Deserialize, Serialize};

use crate::error::GroundTruthError;
use crate::model::Reading;
use crate::series::{EstimateSeries, ReferenceObservation, ReferenceSeries};

/// The largest common grid, in points, bounding allocation.
pub const MAX_GRID_POINTS: usize = 2_000_000;
/// The largest lag search window, in candidate steps, bounding work.
pub const MAX_LAG_STEPS: usize = 200_000;

/// Floating-point tie margin for selecting the best-scoring offset.
const SCORE_EPS: f64 = 1e-9;

/// Configuration for the alignment search. All fields are in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Common resampling grid step (must be positive).
    pub grid_ms: i64,
    /// Half-width of the lag search window; offsets in `[-max_lag, +max_lag]`
    /// are considered (must be non-negative).
    pub max_lag_ms: i64,
    /// Largest gap bridged when resampling: a grid point with no sample within
    /// this distance is left empty rather than interpolated (must be
    /// non-negative).
    pub max_gap_ms: i64,
}

impl Default for AlignmentConfig {
    /// ADR-290 defaults: 1 s grid, ±30 s lag window, 2 s max gap.
    fn default() -> Self {
        Self {
            grid_ms: 1_000,
            max_lag_ms: 30_000,
            max_gap_ms: 2_000,
        }
    }
}

impl AlignmentConfig {
    /// Validate the configuration and the bounded work it implies for the given
    /// series time spans.
    ///
    /// # Errors
    /// [`GroundTruthError::InvalidConfig`] for non-positive/negative fields,
    /// [`GroundTruthError::GridTooLarge`], or
    /// [`GroundTruthError::LagWindowTooLarge`].
    fn validate(&self, est_span_ms: i64) -> Result<(), GroundTruthError> {
        if self.grid_ms <= 0 {
            return Err(GroundTruthError::InvalidConfig {
                reason: "grid_ms must be positive",
            });
        }
        if self.max_lag_ms < 0 {
            return Err(GroundTruthError::InvalidConfig {
                reason: "max_lag_ms must be non-negative",
            });
        }
        if self.max_gap_ms < 0 {
            return Err(GroundTruthError::InvalidConfig {
                reason: "max_gap_ms must be non-negative",
            });
        }
        // The estimate span bounds the widest possible grid (overlap ⊆ estimate
        // range), so this caps every per-lag resample.
        let grid_points = (est_span_ms / self.grid_ms) as usize + 1;
        if grid_points > MAX_GRID_POINTS {
            return Err(GroundTruthError::GridTooLarge {
                max: MAX_GRID_POINTS,
            });
        }
        let lag_steps = (self.max_lag_ms / self.grid_ms) as usize * 2 + 1;
        if lag_steps > MAX_LAG_STEPS {
            return Err(GroundTruthError::LagWindowTooLarge {
                max: MAX_LAG_STEPS,
            });
        }
        Ok(())
    }
}

/// The recovered constant offset and the quality of the alignment at it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    /// Recovered constant offset, milliseconds: the reference is sampled at
    /// `grid_time + offset_ms` to align with the estimate.
    pub offset_ms: i64,
    /// The grid step used.
    pub grid_ms: i64,
    /// Alignment quality at the chosen offset: normalized cross-correlation for
    /// continuous measurands, label-agreement fraction for categorical ones.
    /// `None` when it could not be computed (too few overlapping points, or a
    /// constant/zero-variance continuous signal) — a first-class UNKNOWN, not
    /// an error.
    pub score: Option<f64>,
    /// Total grid points spanning the overlap at the chosen offset.
    pub grid_points: usize,
    /// Grid points where both series had a sample within `max_gap_ms`.
    pub paired_points: usize,
}

/// Resample `samples` (sorted by time) onto `grid` by nearest sample within
/// `max_gap_ms`; a grid point with no sample in range yields `None` (no
/// bridging). `sample_times` must correspond 1:1 to `samples`.
fn resample(
    samples: &[ReferenceObservation],
    sample_times: &[i64],
    grid: &[i64],
    max_gap_ms: i64,
) -> Vec<Option<Reading>> {
    let mut out = Vec::with_capacity(grid.len());
    for &t in grid {
        // Nearest neighbour by binary search over the sorted timestamps.
        let idx = sample_times.partition_point(|&x| x < t);
        let mut best: Option<(i64, usize)> = None;
        for cand in [idx.wrapping_sub(1), idx] {
            if cand < samples.len() {
                let dt = (sample_times[cand] - t).abs();
                let better = match best {
                    None => true,
                    Some((bd, _)) => dt < bd,
                };
                if better {
                    best = Some((dt, cand));
                }
            }
        }
        match best {
            Some((dt, ci)) if dt <= max_gap_ms => out.push(Some(samples[ci].reading.clone())),
            _ => out.push(None),
        }
    }
    out
}

/// Score a set of aligned readings: NCC for scalars, agreement fraction for
/// labels. `None` when not computable (fewer than two paired scalars, zero
/// variance, or no paired labels).
fn score_pairs(pairs: &[(Reading, Reading)]) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    match &pairs[0].0 {
        Reading::Scalar(_) => {
            let xs: Vec<f64> = pairs.iter().filter_map(|(e, _)| e.as_scalar()).collect();
            let ys: Vec<f64> = pairs.iter().filter_map(|(_, r)| r.as_scalar()).collect();
            if xs.len() < 2 || xs.len() != ys.len() {
                return None;
            }
            normalized_cross_correlation(&xs, &ys)
        }
        Reading::Label(_) => {
            let n = pairs.len();
            let agree = pairs
                .iter()
                .filter(|(e, r)| e.as_label() == r.as_label())
                .count();
            Some(agree as f64 / n as f64)
        }
    }
}

/// Normalized cross-correlation of two equal-length vectors; `None` if either
/// has zero variance.
fn normalized_cross_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let a = x - mx;
        let b = y - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    let denom = (dx * dy).sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }
    Some(num / denom)
}

/// Build the grid over the overlap of the estimate and offset reference ranges,
/// on the estimate timeline. Returns an empty vector when there is no overlap.
fn overlap_grid(
    est_lo: i64,
    est_hi: i64,
    ref_lo: i64,
    ref_hi: i64,
    offset: i64,
    grid_ms: i64,
) -> Vec<i64> {
    // Reference is sampled at grid_time + offset, so the reference range maps to
    // [ref_lo - offset, ref_hi - offset] on the estimate timeline.
    let lo = est_lo.max(ref_lo.saturating_sub(offset));
    let hi = est_hi.min(ref_hi.saturating_sub(offset));
    if lo > hi {
        return Vec::new();
    }
    let mut grid = Vec::new();
    let mut t = lo;
    while t <= hi {
        grid.push(t);
        // grid_ms > 0 guaranteed by config validation.
        match t.checked_add(grid_ms) {
            Some(next) => t = next,
            None => break,
        }
    }
    grid
}

/// Produce the aligned reading pairs at a given offset, plus the total grid
/// point count over the overlap (used for coverage).
pub(crate) fn paired_at(
    estimate: &EstimateSeries,
    reference: &ReferenceSeries,
    offset: i64,
    cfg: &AlignmentConfig,
) -> (usize, Vec<(Reading, Reading)>) {
    let est = estimate.samples();
    let refs = reference.samples();
    let est_times: Vec<i64> = est.iter().map(|o| o.at_unix_ms).collect();
    let ref_times: Vec<i64> = refs.iter().map(|o| o.at_unix_ms).collect();
    let (est_lo, est_hi) = (est_times[0], est_times[est_times.len() - 1]);
    let (ref_lo, ref_hi) = (ref_times[0], ref_times[ref_times.len() - 1]);

    let grid = overlap_grid(est_lo, est_hi, ref_lo, ref_hi, offset, cfg.grid_ms);
    let total = grid.len();
    if total == 0 {
        return (0, Vec::new());
    }
    // Estimate sampled on the grid; reference sampled at grid + offset.
    let ref_grid: Vec<i64> = grid
        .iter()
        .map(|&g| g.saturating_add(offset))
        .collect();
    let est_r = resample(est, &est_times, &grid, cfg.max_gap_ms);
    let ref_r = resample(refs, &ref_times, &ref_grid, cfg.max_gap_ms);

    let mut pairs = Vec::new();
    for (e, r) in est_r.into_iter().zip(ref_r.into_iter()) {
        if let (Some(e), Some(r)) = (e, r) {
            pairs.push((e, r));
        }
    }
    (total, pairs)
}

/// Estimate the constant offset that best aligns `estimate` to `reference`.
///
/// Searches offsets in `[-max_lag_ms, +max_lag_ms]` stepped by `grid_ms`,
/// scoring each by NCC (continuous) or agreement (categorical). Ties are broken
/// deterministically toward the smallest absolute offset, then the smallest
/// signed offset. When no offset yields any paired points the result reports
/// offset `0` with a `None` score — a first-class UNKNOWN.
///
/// # Errors
/// [`GroundTruthError::MeasurandMismatch`] if the two series describe different
/// measurands, or a configuration error from [`AlignmentConfig::validate`].
pub fn estimate_alignment(
    estimate: &EstimateSeries,
    reference: &ReferenceSeries,
    cfg: &AlignmentConfig,
) -> Result<Alignment, GroundTruthError> {
    if estimate.measurand != reference.measurand {
        return Err(GroundTruthError::MeasurandMismatch {
            estimate: estimate.measurand.label(),
            reference: reference.measurand.label(),
        });
    }
    let est_times = estimate.samples();
    let span = est_times[est_times.len() - 1].at_unix_ms - est_times[0].at_unix_ms;
    cfg.validate(span.max(0))?;

    let mut best_offset: i64 = 0;
    let mut best_score: Option<f64> = None;

    let mut offset = -cfg.max_lag_ms;
    while offset <= cfg.max_lag_ms {
        let (_, pairs) = paired_at(estimate, reference, offset, cfg);
        let score = score_pairs(&pairs);
        if let Some(s) = score {
            let replace = match best_score {
                None => true,
                Some(b) => {
                    s > b + SCORE_EPS
                        || ((s - b).abs() <= SCORE_EPS && offset.abs() < best_offset.abs())
                }
            };
            if replace {
                best_score = Some(s);
                best_offset = offset;
            }
        }
        match offset.checked_add(cfg.grid_ms) {
            Some(next) => offset = next,
            None => break,
        }
    }

    let (total, pairs) = paired_at(estimate, reference, best_offset, cfg);
    Ok(Alignment {
        offset_ms: best_offset,
        grid_ms: cfg.grid_ms,
        score: best_score,
        grid_points: total,
        paired_points: pairs.len(),
    })
}
