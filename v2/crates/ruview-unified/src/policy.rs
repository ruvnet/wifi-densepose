//! Edge sensing control plane (ADR-277) — purposes, zones, retention,
//! identity gating, and the export trust boundary.
//!
//! Aligned with the sensing-service vocabulary of IEEE 802.11bf-2025 and
//! the ETSI ISAC architecture (sensing purpose + sensing zone as first-class
//! authorization objects; the ETSI security report's issue classes motivate
//! the fail-closed defaults). Three hard rules, all enforced structurally:
//!
//! 1. **Raw RF never leaves the trust boundary.** The only exportable type
//!    is [`BoundedEvent`] — it cannot carry a tensor, and
//!    [`TrustBoundary::export`] is the only egress. There is deliberately
//!    no API that serializes an [`crate::tensor::RfTensor`] outward.
//! 2. **Fail closed.** Unknown zone ⇒ deny. Purpose not granted ⇒ deny.
//!    Identity inference ⇒ deny unless the zone *explicitly* enables it in
//!    addition to granting the purpose.
//! 3. **Every output is accountable.** A [`BoundedEvent`] cannot be built
//!    without uncertainty, provenance, model version, and purpose
//!    (ADR-273 acceptance item 8).

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::gaussian::primitive::Provenance;
use crate::{Result, UnifiedError};

/// Sensing purposes (ETSI ISAC sensing-service classes, WLAN-sensing
/// aligned). Ordering matters only for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SensingPurpose {
    /// Someone is / is not present.
    Presence,
    /// Coarse activity class.
    Activity,
    /// Respiration / heart-rate class vitals.
    Vitals,
    /// Position estimation.
    Localization,
    /// Skeletal pose tracking.
    PoseTracking,
    /// Identity recognition — the high-risk purpose; doubly gated.
    IdentityRecognition,
    /// RF channel diagnostics (no human inference).
    ChannelDiagnostics,
}

/// A spatial sensing zone and what it permits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyZone {
    /// Zone identifier (maps to rooms/regions in the scene graph).
    pub id: String,
    /// Purposes granted in this zone.
    pub allowed_purposes: BTreeSet<SensingPurpose>,
    /// Maximum event age at export, seconds (retention bound).
    pub retention_s: u64,
    /// Second factor for identity: even if `IdentityRecognition` is in
    /// `allowed_purposes`, it is denied unless this is also true.
    pub identity_explicitly_enabled: bool,
}

/// Payload of a bounded event — semantically typed results only; no
/// variant can carry raw RF samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventValue {
    /// Presence verdict.
    Presence(bool),
    /// Activity class index.
    ActivityClass(u8),
    /// Respiration rate, breaths/minute.
    RespirationBpm(f64),
    /// Position estimate, metres, room frame.
    Location([f64; 3]),
    /// Anomaly z-score.
    AnomalyScore(f64),
}

/// The only type allowed across the trust boundary. Construction validates
/// that the accountability fields are present and sane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedEvent {
    /// Purpose under which this event was produced.
    pub purpose: SensingPurpose,
    /// Typed result.
    pub value: EventValue,
    /// Mandatory uncertainty in `[0, 1]` (1 = no information).
    pub uncertainty: f64,
    /// Evidence provenance (device, model, synthetic flag).
    pub provenance: Provenance,
    /// Model version that produced the inference.
    pub model_version: u32,
    /// Event timestamp, ns since epoch.
    pub timestamp_ns: u64,
    /// Zone the event was sensed in.
    pub zone_id: String,
}

impl BoundedEvent {
    /// Validated constructor — the only way to build an exportable event.
    pub fn new(
        purpose: SensingPurpose,
        value: EventValue,
        uncertainty: f64,
        provenance: Provenance,
        model_version: u32,
        timestamp_ns: u64,
        zone_id: impl Into<String>,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&uncertainty) {
            return Err(UnifiedError::InvalidInput(format!(
                "uncertainty must be in [0,1], got {uncertainty}"
            )));
        }
        if model_version == 0 {
            return Err(UnifiedError::InvalidInput(
                "model_version 0 (unassigned) is not exportable".into(),
            ));
        }
        Ok(Self {
            purpose,
            value,
            uncertainty,
            provenance,
            model_version,
            timestamp_ns,
            zone_id: zone_id.into(),
        })
    }
}

/// The policy engine: zone registry + authorization checks.
#[derive(Debug, Default)]
pub struct PolicyEngine {
    zones: HashMap<String, PrivacyZone>,
}

impl PolicyEngine {
    /// Empty engine (denies everything until zones are configured).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers or replaces a zone.
    pub fn upsert_zone(&mut self, zone: PrivacyZone) {
        self.zones.insert(zone.id.clone(), zone);
    }

    /// Authorizes sensing for `purpose` in `zone_id`. Fail-closed on every
    /// branch: unknown zone, ungranted purpose, and the identity double
    /// gate all deny.
    pub fn authorize(&self, zone_id: &str, purpose: SensingPurpose) -> Result<()> {
        let zone = self
            .zones
            .get(zone_id)
            .ok_or_else(|| UnifiedError::PolicyDenied(format!("unknown zone {zone_id:?}")))?;
        if !zone.allowed_purposes.contains(&purpose) {
            return Err(UnifiedError::PolicyDenied(format!(
                "purpose {purpose:?} not granted in zone {zone_id:?}"
            )));
        }
        if purpose == SensingPurpose::IdentityRecognition && !zone.identity_explicitly_enabled {
            return Err(UnifiedError::PolicyDenied(format!(
                "identity recognition requires explicit enablement in zone {zone_id:?}"
            )));
        }
        Ok(())
    }
}

/// The egress point. Holds the policy engine and a monotonically supplied
/// "now"; the **only** public method emits [`BoundedEvent`]s — raw RF has no
/// path through here by construction.
#[derive(Debug, Default)]
pub struct TrustBoundary {
    engine: PolicyEngine,
}

impl TrustBoundary {
    /// New boundary over a configured engine.
    #[must_use]
    pub fn new(engine: PolicyEngine) -> Self {
        Self { engine }
    }

    /// Zone-config passthrough.
    pub fn engine_mut(&mut self) -> &mut PolicyEngine {
        &mut self.engine
    }

    /// Exports an event if — and only if — the zone grants its purpose and
    /// the event is inside the zone's retention window at `now_ns`.
    /// Returns the event back on success so callers can hand it to a
    /// transport; on denial the event is dropped with a typed error.
    pub fn export(&self, event: BoundedEvent, now_ns: u64) -> Result<BoundedEvent> {
        self.engine.authorize(&event.zone_id, event.purpose)?;
        let zone = self
            .engine
            .zones
            .get(&event.zone_id)
            .expect("authorize verified the zone exists");
        let age_s = now_ns.saturating_sub(event.timestamp_ns) / 1_000_000_000;
        if age_s > zone.retention_s {
            return Err(UnifiedError::PolicyDenied(format!(
                "event age {age_s}s exceeds zone retention {}s",
                zone.retention_s
            )));
        }
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prov() -> Provenance {
        Provenance { device_id: "esp32s3-a1".into(), model_version: 3, synthetic: false }
    }

    fn zone(purposes: &[SensingPurpose], identity: bool) -> PrivacyZone {
        PrivacyZone {
            id: "living-room".into(),
            allowed_purposes: purposes.iter().copied().collect(),
            retention_s: 3600,
            identity_explicitly_enabled: identity,
        }
    }

    fn event(purpose: SensingPurpose, ts: u64) -> BoundedEvent {
        BoundedEvent::new(
            purpose,
            EventValue::Presence(true),
            0.12,
            prov(),
            3,
            ts,
            "living-room",
        )
        .expect("valid event")
    }

    #[test]
    fn unknown_zone_denies() {
        let boundary = TrustBoundary::new(PolicyEngine::new());
        let err = boundary.export(event(SensingPurpose::Presence, 0), 0).unwrap_err();
        assert!(matches!(err, UnifiedError::PolicyDenied(_)));
    }

    #[test]
    fn ungranted_purpose_denies() {
        let mut engine = PolicyEngine::new();
        engine.upsert_zone(zone(&[SensingPurpose::Presence], false));
        let boundary = TrustBoundary::new(engine);
        assert!(boundary.export(event(SensingPurpose::Presence, 0), 0).is_ok());
        assert!(matches!(
            boundary.export(event(SensingPurpose::Localization, 0), 0),
            Err(UnifiedError::PolicyDenied(_))
        ));
    }

    #[test]
    fn identity_needs_both_grant_and_explicit_enable() {
        // Granted in purposes but NOT explicitly enabled ⇒ deny.
        let mut engine = PolicyEngine::new();
        engine.upsert_zone(zone(
            &[SensingPurpose::Presence, SensingPurpose::IdentityRecognition],
            false,
        ));
        assert!(matches!(
            engine.authorize("living-room", SensingPurpose::IdentityRecognition),
            Err(UnifiedError::PolicyDenied(_))
        ));
        // Both factors present ⇒ allow.
        engine.upsert_zone(zone(
            &[SensingPurpose::Presence, SensingPurpose::IdentityRecognition],
            true,
        ));
        assert!(engine.authorize("living-room", SensingPurpose::IdentityRecognition).is_ok());
        // Explicit flag alone (purpose not granted) ⇒ still deny.
        engine.upsert_zone(zone(&[SensingPurpose::Presence], true));
        assert!(engine.authorize("living-room", SensingPurpose::IdentityRecognition).is_err());
    }

    #[test]
    fn retention_bound_is_enforced() {
        let mut engine = PolicyEngine::new();
        engine.upsert_zone(zone(&[SensingPurpose::Presence], false));
        let boundary = TrustBoundary::new(engine);
        let e = event(SensingPurpose::Presence, 0);
        // Within retention (1 h): fine.
        assert!(boundary.export(e.clone(), 3_500 * 1_000_000_000).is_ok());
        // Beyond retention: denied.
        assert!(matches!(
            boundary.export(e, 3_700 * 1_000_000_000),
            Err(UnifiedError::PolicyDenied(_))
        ));
    }

    #[test]
    fn accountability_fields_are_mandatory() {
        // Out-of-range uncertainty refuses construction.
        assert!(BoundedEvent::new(
            SensingPurpose::Presence,
            EventValue::Presence(true),
            1.5,
            prov(),
            3,
            0,
            "z"
        )
        .is_err());
        // Unassigned model version refuses construction.
        assert!(BoundedEvent::new(
            SensingPurpose::Presence,
            EventValue::Presence(true),
            0.1,
            prov(),
            0,
            0,
            "z"
        )
        .is_err());
    }
}
