//! Task allocation: auction-based and FNN-scored bid evaluation.
//!
// NOTE: Task allocation is ITAR-controlled (USML Category VIII(h)(12)).
// Only available when the `itar-unrestricted` feature is enabled.

#[cfg(feature = "itar-unrestricted")]
pub mod auction;
#[cfg(feature = "itar-unrestricted")]
pub mod fnn;

#[cfg(feature = "itar-unrestricted")]
pub use auction::{AuctionAllocator, Bid};
#[cfg(feature = "itar-unrestricted")]
pub use fnn::FnnScorer;

/// Stub: task allocation is export-controlled. Enable `itar-unrestricted` feature.
#[cfg(not(feature = "itar-unrestricted"))]
pub fn allocate_stub() -> crate::SwarmResult<()> {
    Err(crate::SwarmError::Security(
        "Task allocation requires itar-unrestricted feature (USML VIII(h)(12))".into(),
    ))
}
