//! Formation control: virtual structure, leader-follower, Reynolds flocking.
//!
// NOTE: Formation control is ITAR-controlled (USML Category VIII(h)(12)).
// Only available when the `itar-unrestricted` feature is enabled.

#[cfg(feature = "itar-unrestricted")]
pub mod virtual_structure;
#[cfg(feature = "itar-unrestricted")]
pub mod leader_follower;
#[cfg(feature = "itar-unrestricted")]
pub mod reynolds;

#[cfg(feature = "itar-unrestricted")]
pub use virtual_structure::VirtualStructure;
#[cfg(feature = "itar-unrestricted")]
pub use leader_follower::LeaderFollower;
#[cfg(feature = "itar-unrestricted")]
pub use reynolds::ReynoldsParams;

/// Stub: formation control is export-controlled. Enable `itar-unrestricted` feature.
#[cfg(not(feature = "itar-unrestricted"))]
pub fn formation_stub() -> crate::SwarmResult<()> {
    Err(crate::SwarmError::Security(
        "Formation control requires itar-unrestricted feature (USML VIII(h)(12))".into(),
    ))
}
