//! ADR-149 statistically-rigorous evaluation harness (Stage 1, kinematic).
//!
//! Produces SAR + MARL metrics over a seeded N-seed × M-episode matrix with
//! IQM + 95% stratified-bootstrap CIs, a (sigma, kappa) CSI-noise sweep, and
//! GDOP-stratified localization error. Generates evals/RESULTS.md.
//!
//! Stage 2 (Gazebo/PX4 SITL high-fidelity, false-alarm + collision rate on the
//! median seeds) is a follow-on — see ADR-149 §6.1.
pub mod gdop;
pub mod stats;
pub mod metrics;
pub mod runner;
pub mod report;

pub use gdop::gdop;
pub use stats::{iqm, stratified_bootstrap_ci, ConfidenceInterval};
pub use metrics::{EpisodeMetrics, AggregateMetrics};
pub use runner::{EvalConfig, NoiseLevel, run_matrix};
pub use report::render_results_md;
