//! Per-episode and aggregate SAR + MARL metrics (ADR-149 Stage 1).

use crate::evals::stats::{stratified_bootstrap_ci, ConfidenceInterval};

/// Per-episode SAR metrics (Stage 1 kinematic).
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// Fraction of the mission area scanned at least once, in [0, 1].
    pub coverage_pct: f64,
    /// Localization error (m) of the fused victim estimate; `None` if no detection.
    pub localization_error_m: Option<f64>,
    /// GDOP of the contributing-drone constellation at detection; `None` if none.
    pub gdop_at_detection: Option<f64>,
    /// Mission-elapsed seconds to first detection; `None` if no detection.
    pub time_to_first_detection_s: Option<f64>,
    /// Whether at least one victim was detected this episode.
    pub detected: bool,
    /// Count of inter-drone proximity violations (kinematic proxy for collisions).
    pub collisions: u32,
    /// Fraction of scanned area covered by more than one drone, in [0, 1].
    pub overlap_ratio: f64,
    /// Scalar episodic return (reward-like coverage/detection objective).
    pub episodic_return: f64,
}

/// Aggregate over a seed × episode matrix with IQM + 95% bootstrap CIs.
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub coverage_iqm: ConfidenceInterval,
    /// IQM over detected episodes only (undetected episodes carry no error).
    pub localization_iqm: ConfidenceInterval,
    pub detection_rate: f64,
    pub mean_gdop: f64,
    pub return_iqm: ConfidenceInterval,
    pub n_episodes: usize,
}

impl AggregateMetrics {
    /// Aggregate a seed-stratified matrix of episodes. Each inner `Vec` is one
    /// seed's episodes; bootstrap resampling is stratified by seed so the CI
    /// reflects between-seed variance (the dominant source per ADR-149).
    pub fn from_strata(per_seed: &[Vec<EpisodeMetrics>], boot_seed: u64) -> Self {
        const N_BOOT: usize = 1000;

        let coverage_strata: Vec<Vec<f64>> = per_seed
            .iter()
            .map(|s| s.iter().map(|e| e.coverage_pct).collect())
            .collect();
        let return_strata: Vec<Vec<f64>> = per_seed
            .iter()
            .map(|s| s.iter().map(|e| e.episodic_return).collect())
            .collect();
        // Localization: only detected episodes contribute. Keep stratification
        // by seed but drop empty strata so the bootstrap doesn't degenerate.
        let loc_strata: Vec<Vec<f64>> = per_seed
            .iter()
            .map(|s| {
                s.iter()
                    .filter_map(|e| e.localization_error_m)
                    .collect::<Vec<f64>>()
            })
            .filter(|v: &Vec<f64>| !v.is_empty())
            .collect();

        let mut detected = 0usize;
        let mut total = 0usize;
        let mut gdop_sum = 0.0;
        let mut gdop_n = 0usize;
        for seed in per_seed {
            for e in seed {
                total += 1;
                if e.detected {
                    detected += 1;
                }
                if let Some(g) = e.gdop_at_detection {
                    if g.is_finite() {
                        gdop_sum += g;
                        gdop_n += 1;
                    }
                }
            }
        }

        let detection_rate = if total == 0 {
            0.0
        } else {
            detected as f64 / total as f64
        };
        let mean_gdop = if gdop_n == 0 {
            0.0
        } else {
            gdop_sum / gdop_n as f64
        };

        AggregateMetrics {
            coverage_iqm: stratified_bootstrap_ci(&coverage_strata, N_BOOT, boot_seed),
            localization_iqm: stratified_bootstrap_ci(
                &loc_strata,
                N_BOOT,
                boot_seed.wrapping_add(1),
            ),
            detection_rate,
            mean_gdop,
            return_iqm: stratified_bootstrap_ci(
                &return_strata,
                N_BOOT,
                boot_seed.wrapping_add(2),
            ),
            n_episodes: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(cov: f64, loc: Option<f64>, ret: f64, detected: bool) -> EpisodeMetrics {
        EpisodeMetrics {
            coverage_pct: cov,
            localization_error_m: loc,
            gdop_at_detection: if detected { Some(2.0) } else { None },
            time_to_first_detection_s: if detected { Some(10.0) } else { None },
            detected,
            collisions: 0,
            overlap_ratio: 0.1,
            episodic_return: ret,
        }
    }

    #[test]
    fn test_aggregate_detection_rate_and_shape() {
        let per_seed = vec![
            vec![
                ep(0.8, Some(1.5), 80.0, true),
                ep(0.7, None, 70.0, false),
            ],
            vec![
                ep(0.9, Some(2.0), 90.0, true),
                ep(0.85, Some(1.0), 85.0, true),
            ],
        ];
        let agg = AggregateMetrics::from_strata(&per_seed, 7);
        assert_eq!(agg.n_episodes, 4);
        assert!((agg.detection_rate - 0.75).abs() < 1e-9);
        assert!(agg.coverage_iqm.lo <= agg.coverage_iqm.point);
        assert!(agg.coverage_iqm.point <= agg.coverage_iqm.hi);
        assert!(agg.mean_gdop > 0.0);
    }
}
