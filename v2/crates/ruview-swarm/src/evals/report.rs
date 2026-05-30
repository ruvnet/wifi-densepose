//! RESULTS.md leaderboard generator (ADR-149 Stage 1).

use crate::evals::metrics::AggregateMetrics;
use crate::evals::stats::ConfidenceInterval;

/// Wi2SAR published localization baseline (paper-to-paper), metres.
const WI2SAR_LOCALIZATION_M: f64 = 5.0;

/// Format a CI as `point [lo, hi]` with two decimals.
fn fmt_ci(ci: &ConfidenceInterval) -> String {
    format!("{:.3} [{:.3}, {:.3}]", ci.point, ci.lo, ci.hi)
}

/// Render a markdown leaderboard: one row per flight pattern with coverage
/// IQM±CI, localization IQM±CI, detection rate, and mean GDOP — plus the
/// Wi2SAR paper baseline row clearly labelled paper-to-paper.
///
/// `rows` is `(pattern_name, aggregate)`; rows are emitted in the order given,
/// so callers should pre-sort (e.g. by descending coverage point estimate).
pub fn render_results_md(rows: &[(String, AggregateMetrics)]) -> String {
    let mut s = String::new();
    s.push_str("# ruview-swarm Evaluation Results (ADR-149 Stage 1, kinematic)\n\n");
    s.push_str(
        "Statistically-rigorous evaluation harness: seeded multi-run rollouts with \
         IQM + 95% stratified-bootstrap confidence intervals (Agarwal et al., \
         NeurIPS 2021).\n\n",
    );

    // Run configuration header.
    let (n_episodes, n_seeds) = rows
        .first()
        .map(|(_, a)| {
            let n = a.n_episodes;
            // Episodes-per-seed isn't stored; report total + leave seed split to caller note.
            (n, 0usize)
        })
        .unwrap_or((0, 0));
    s.push_str("## Run configuration\n\n");
    s.push_str(&format!(
        "- **Stage**: 1 (kinematic, self-contained, deterministic per seed)\n\
         - **Episodes per pattern**: {n_episodes} (seed × episode matrix)\n\
         - **CI method**: 95% stratified bootstrap of the IQM, stratified by seed\n\
         - **GDOP**: 2-D geometric dilution of precision at first detection\n"
    ));
    let _ = n_seeds;
    s.push_str(
        "\n> **Stage 2 pending**: high-fidelity Gazebo/PX4 SITL evaluation \
         (false-alarm rate, real collision rate on the median seeds) is a \
         follow-on — see ADR-149 §6.1. The collision figures below are a \
         kinematic min-separation proxy, not SITL physics.\n\n",
    );

    // Leaderboard table.
    s.push_str("## Flight-pattern leaderboard\n\n");
    s.push_str(
        "| Flight pattern | Coverage IQM [95% CI] | Localization (m) IQM [95% CI] | \
         Detection rate | Mean GDOP |\n",
    );
    s.push_str(
        "|----------------|-----------------------|-------------------------------|\
         ----------------|-----------|\n",
    );
    for (name, agg) in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.1}% | {:.3} |\n",
            name,
            fmt_ci(&agg.coverage_iqm),
            fmt_ci(&agg.localization_iqm),
            agg.detection_rate * 100.0,
            agg.mean_gdop,
        ));
    }
    // Wi2SAR paper baseline row (paper-to-paper, no kinematic re-run).
    s.push_str(&format!(
        "| _Wi2SAR (paper baseline)_ | _n/a_ | _{:.1} (paper)_ | _n/a_ | _n/a_ |\n",
        WI2SAR_LOCALIZATION_M,
    ));

    s.push_str(
        "\n_Wi2SAR row is the published single-drone localization figure \
         (arxiv 2604.09115), shown paper-to-paper for reference only — it was \
         not re-run through this kinematic harness._\n",
    );

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evals::stats::ConfidenceInterval;

    fn agg(cov: f64, det: f64) -> AggregateMetrics {
        let ci = |p: f64| ConfidenceInterval { point: p, lo: p - 0.05, hi: p + 0.05 };
        AggregateMetrics {
            coverage_iqm: ci(cov),
            localization_iqm: ci(1.5),
            detection_rate: det,
            mean_gdop: 2.1,
            return_iqm: ci(80.0),
            n_episodes: 100,
        }
    }

    #[test]
    fn test_render_contains_rows_and_baseline() {
        let rows = vec![
            ("partitioned_lawnmower".to_string(), agg(0.92, 0.95)),
            ("levy_flight".to_string(), agg(0.40, 0.50)),
        ];
        let md = render_results_md(&rows);
        assert!(md.contains("partitioned_lawnmower"));
        assert!(md.contains("levy_flight"));
        assert!(md.contains("Wi2SAR"));
        assert!(md.contains("Stage 2 pending"));
        assert!(md.contains("95% stratified bootstrap"));
        // Coverage point estimate appears.
        assert!(md.contains("0.920"));
    }
}
