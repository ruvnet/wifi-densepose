//! `cog-person-count` — learned multi-person counter (ADR-103).
//!
//! Replaces the PR #491 slot heuristic with:
//!  * a small Candle network (encoder + count head + confidence head),
//!  * Stoer-Wagner-bounded multi-node fusion,
//!  * `{count, confidence, count_p95_low, count_p95_high}` output.
//!
//! Design lives in `docs/adr/ADR-103-learned-multi-person-counter.md`.

pub mod fusion;
pub mod inference;
pub mod publisher;
pub mod runtime;

pub const COG_ID: &str = "person-count";
pub const COG_VERSION: &str = env!("CARGO_PKG_VERSION");
