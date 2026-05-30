//! ADR-149 Stage-1 evaluation CLI.
//!
//! Runs the kinematic eval matrix over every flight pattern (default) and
//! writes a ranked `RESULTS.md` leaderboard. Pure Rust — no special feature
//! flag required, so it builds and runs in default CI.
//!
//! Defaults are intentionally small (10 seeds × 10 episodes) so the run is fast.
//! The full ADR-149 reporting configuration is 10 seeds × 50 episodes — pass
//! `--seeds 10 --episodes 50` for the publication run.
//!
//! ```text
//! cargo run -p ruview-swarm --bin eval_swarm -- \
//!   --seeds 10 --episodes 10 --out crates/ruview-swarm/evals/RESULTS.md
//! ```

use std::path::PathBuf;

use ruview_swarm::evals::metrics::AggregateMetrics;
use ruview_swarm::evals::report::render_results_md;
use ruview_swarm::evals::runner::{run_matrix, EvalConfig};
use ruview_swarm::planning::patterns::FlightPattern;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seeds = 10usize;
    let mut episodes = 10usize;
    let mut out = PathBuf::from("crates/ruview-swarm/evals/RESULTS.md");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                i += 1;
                seeds = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seeds);
            }
            "--episodes" => {
                i += 1;
                episodes = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(episodes);
            }
            "--out" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    out = PathBuf::from(p);
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "eval_swarm — ADR-149 Stage-1 kinematic evaluator\n\
                     Usage: eval_swarm [--seeds N] [--episodes M] [--out PATH]\n\
                     Defaults: --seeds 10 --episodes 10 --out crates/ruview-swarm/evals/RESULTS.md"
                );
                return;
            }
            other => {
                eprintln!("warning: ignoring unknown argument '{other}'");
            }
        }
        i += 1;
    }

    eprintln!(
        "Running ADR-149 Stage-1 eval: {seeds} seeds × {episodes} episodes \
         over {} flight patterns...",
        FlightPattern::all().len()
    );

    let mut rows: Vec<(String, AggregateMetrics)> = Vec::new();
    for (idx, pattern) in FlightPattern::all().into_iter().enumerate() {
        let mut cfg = EvalConfig::sar_small(pattern);
        cfg.seeds = seeds;
        cfg.episodes_per_seed = episodes;
        let matrix = run_matrix(&cfg);
        let agg = AggregateMetrics::from_strata(&matrix, 0x0149 ^ idx as u64);
        eprintln!(
            "  {}: coverage IQM {:.3}, detection {:.0}%",
            pattern.name(),
            agg.coverage_iqm.point,
            agg.detection_rate * 100.0
        );
        rows.push((pattern.name().to_string(), agg));
    }

    // Rank by descending coverage point estimate.
    rows.sort_by(|a, b| {
        b.1.coverage_iqm
            .point
            .partial_cmp(&a.1.coverage_iqm.point)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let md = render_results_md(&rows);

    if let Some(parent) = out.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("error: could not create {}: {e}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(e) = std::fs::write(&out, &md) {
        eprintln!("error: could not write {}: {e}", out.display());
        std::process::exit(1);
    }
    eprintln!("Wrote {} ({} bytes).", out.display(), md.len());
}
