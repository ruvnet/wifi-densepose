//! Privacy-mode filter for outbound MQTT (and Matter) state messages.
//!
//! Implements the ADR-106 primitive-isolation contract at the integration
//! boundary, gated by [`crate::cli::Args::privacy_mode`]. When the flag is
//! set, biometric channels (HR, BR, raw pose keypoints) are stripped
//! from every outbound message *and* their entities are never discovered
//! by Home Assistant — `discovery.rs::DiscoveryBuilder::enabled_entities`
//! returns the filtered set.
//!
//! Semantic primitives (someone-sleeping, possible-distress, etc) stay
//! enabled in privacy mode because they're inferred *states*, not raw
//! biometric values. The inference runs server-side and only the boolean
//! / numeric state crosses the wire. This is the key design choice that
//! makes ADR-115 §3.12 enterprise- and healthcare-deployable.

use super::discovery::EntityKind;

/// Decision for one outbound publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishDecision {
    /// Send as-is.
    Publish,
    /// Drop silently (entity is suppressed by privacy mode).
    Suppress,
}

/// Decide whether an entity may be published given a privacy-mode flag.
///
/// Discovery and state share the same filter so an HA controller can't
/// learn from the absence of state that the entity might exist with
/// different filters in place — if it's stripped, it's stripped at every
/// layer.
pub fn decide(entity: EntityKind, privacy_mode: bool) -> PublishDecision {
    if privacy_mode && entity.is_biometric() {
        PublishDecision::Suppress
    } else {
        PublishDecision::Publish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn privacy_off_publishes_everything() {
        for e in [
            EntityKind::Presence,
            EntityKind::HeartRate,
            EntityKind::BreathingRate,
            EntityKind::PoseKeypoints,
            EntityKind::SomeoneSleeping,
            EntityKind::PossibleDistress,
            EntityKind::FallDetected,
        ] {
            assert_eq!(decide(e, false), PublishDecision::Publish);
        }
    }

    #[test]
    fn privacy_on_suppresses_biometrics_only() {
        // HR / BR / pose keypoints → suppressed.
        assert_eq!(
            decide(EntityKind::HeartRate, true),
            PublishDecision::Suppress
        );
        assert_eq!(
            decide(EntityKind::BreathingRate, true),
            PublishDecision::Suppress
        );
        assert_eq!(
            decide(EntityKind::PoseKeypoints, true),
            PublishDecision::Suppress
        );
    }

    #[test]
    fn privacy_on_keeps_non_biometric_signals() {
        for e in [
            EntityKind::Presence,
            EntityKind::PersonCount,
            EntityKind::MotionLevel,
            EntityKind::Rssi,
            EntityKind::ZoneOccupancy,
            EntityKind::FallDetected,
            EntityKind::PresenceScore,
        ] {
            assert_eq!(
                decide(e, true),
                PublishDecision::Publish,
                "{:?} should not be suppressed",
                e
            );
        }
    }

    #[test]
    fn privacy_on_keeps_semantic_primitives() {
        // Per ADR-115 §3.12.3 — semantic primitives are *inferred* states,
        // not raw biometrics, so they remain available in privacy mode.
        // This is the core privacy win of HA-MIND.
        for e in [
            EntityKind::SomeoneSleeping,
            EntityKind::PossibleDistress,
            EntityKind::RoomActive,
            EntityKind::ElderlyInactivityAnomaly,
            EntityKind::MeetingInProgress,
            EntityKind::BathroomOccupied,
            EntityKind::FallRiskElevated,
            EntityKind::BedExit,
            EntityKind::NoMovement,
            EntityKind::MultiRoomTransition,
        ] {
            assert_eq!(
                decide(e, true),
                PublishDecision::Publish,
                "{:?} should not be suppressed",
                e
            );
        }
    }
}
