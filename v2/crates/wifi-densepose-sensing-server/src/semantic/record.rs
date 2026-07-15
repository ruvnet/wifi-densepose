//! ADR-140 — `SemanticStateRecord`: the auditable, versioned, privacy-gated
//! wire form of a semantic belief, plus the Ruflo agent-bridge routing that
//! fires on *multi-signal agreement* (e.g. fall-risk + elderly-anomaly →
//! caregiver escalation).
//!
//! This extends the existing [`SemanticEvent`](super::bus::SemanticEvent)
//! (kind/state/node/timestamp) with the provenance the house rule mandates:
//! model version, calibration version, privacy action, expiry, confidence,
//! room, and evidence refs. Each record is the wire form of an ADR-139
//! `WorldNode::SemanticState`.

use super::bus::{SemanticEvent, SemanticKind};
use super::common::PrimitiveState;

/// Privacy action enforced at the semantic layer (ADR-140 §2 → ADR-141).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyAction {
    /// Emit the full record (RawResearch / CareWithConsent style).
    Allow,
    /// Drop person-identifying detail, keep room-level occupancy.
    AnonymizeByRoom,
    /// Strip biometric scalars (HR/BR) from the emitted record.
    StripBiometrics,
}

/// Per-deployment context used to enrich a [`SemanticEvent`] into a
/// [`SemanticStateRecord`]. Loaded from the model/calibration manifest.
#[derive(Debug, Clone)]
pub struct RecordContext {
    /// Model version that produced the underlying belief (ADR-136 `model_id`).
    pub model_version: String,
    /// Calibration version (ADR-135 baseline id) in effect.
    pub calibration_version: String,
    /// Active privacy action (ADR-141 mode → action).
    pub privacy_action: PrivacyAction,
    /// Record time-to-live (ms); `expiry_at = timestamp_ms + default_ttl_ms`.
    pub default_ttl_ms: i64,
}

impl Default for RecordContext {
    fn default() -> Self {
        Self {
            model_version: "unassigned".into(),
            calibration_version: "uncalibrated".into(),
            privacy_action: PrivacyAction::Allow,
            default_ttl_ms: 30_000,
        }
    }
}

/// Auditable, versioned semantic state record (ADR-140 §2.1).
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticStateRecord {
    /// Which primitive produced this belief.
    pub kind: SemanticKind,
    /// Room/area this belief is scoped to (None = whole installation).
    pub room: Option<String>,
    /// Source sensing node id.
    pub node_id: String,
    /// Capture time (Unix ms).
    pub timestamp_ms: i64,
    /// Belief expiry (Unix ms); after this the record is stale.
    pub expiry_at_ms: i64,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Model version (ADR-136).
    pub model_version: String,
    /// Calibration version (ADR-135).
    pub calibration_version: String,
    /// Privacy action under which it was derived (ADR-141).
    pub privacy_action: PrivacyAction,
    /// Evidence refs (ADR-137) — here, the human-readable reason tags.
    pub evidence_refs: Vec<String>,
    /// Whether the underlying primitive is currently "active"/firing.
    pub active: bool,
}

impl SemanticStateRecord {
    /// Enrich a [`SemanticEvent`] into a record using deployment context and a
    /// room mapping for the event's node.
    #[must_use]
    pub fn from_event(event: &SemanticEvent, room: Option<String>, ctx: &RecordContext) -> Self {
        let (confidence, active, mut evidence_refs) = match &event.state {
            PrimitiveState::Boolean { active, reason, .. } => {
                (if *active { 0.9 } else { 0.1 }, *active, reason.tags.clone())
            }
            PrimitiveState::Scalar { value, reason } => {
                ((*value as f32 / 100.0).clamp(0.0, 1.0), *value > 0.0, reason.tags.clone())
            }
            PrimitiveState::Event { event_type, reason } => {
                let mut t = reason.tags.clone();
                t.push(format!("event={event_type}"));
                (1.0, true, t)
            }
            PrimitiveState::Idle => (0.0, false, Vec::new()),
        };

        // Privacy enforcement at the record boundary.
        if ctx.privacy_action == PrivacyAction::StripBiometrics {
            evidence_refs.retain(|t| !is_biometric_tag(t));
        }

        Self {
            kind: event.kind,
            room,
            node_id: event.node_id.clone(),
            timestamp_ms: event.timestamp_ms,
            expiry_at_ms: event.timestamp_ms + ctx.default_ttl_ms,
            confidence,
            model_version: ctx.model_version.clone(),
            calibration_version: ctx.calibration_version.clone(),
            privacy_action: ctx.privacy_action,
            evidence_refs,
            active,
        }
    }

    /// Whether this record is still valid at `now_ms`.
    #[must_use]
    pub fn is_fresh(&self, now_ms: i64) -> bool {
        now_ms < self.expiry_at_ms
    }
}

fn is_biometric_tag(tag: &str) -> bool {
    let t = tag.to_ascii_lowercase();
    t.contains("hr=") || t.contains("br=") || t.contains("bpm")
}

// ---- ADR-140 §2.3 Ruflo agent bridge: multi-signal agreement routing ----

/// A routing decision handed to the ADR-133 HOMECORE-ASSIST / Ruflo layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRoute {
    /// Stable route identifier (e.g. "caregiver_escalation").
    pub route_id: &'static str,
    /// Severity 0..=3 (info, notice, warning, critical).
    pub severity: u8,
}

/// A rule that fires a route when *all* required kinds are simultaneously
/// active and fresh in the candidate record set (multi-signal agreement).
#[derive(Debug, Clone)]
pub struct MultiSignalRule {
    /// Kinds that must all be active+fresh for the route to fire.
    pub required_kinds: Vec<SemanticKind>,
    /// Minimum confidence each required record must meet.
    pub min_confidence: f32,
    /// Route emitted when the rule matches.
    pub route: AgentRoute,
}

impl MultiSignalRule {
    /// Evaluate the rule against fresh, active records at `now_ms`. Returns the
    /// route iff every required kind has at least one active record meeting
    /// `min_confidence`. Routing on agreement (not a single signal) is what
    /// suppresses single-primitive false positives for high-impact actions.
    #[must_use]
    pub fn evaluate(&self, records: &[SemanticStateRecord], now_ms: i64) -> Option<AgentRoute> {
        let all_present = self.required_kinds.iter().all(|k| {
            records.iter().any(|r| {
                r.kind == *k && r.active && r.is_fresh(now_ms) && r.confidence >= self.min_confidence
            })
        });
        all_present.then(|| self.route.clone())
    }
}

/// Evaluate every rule, returning the matched routes (deduped by route_id,
/// highest severity first).
#[must_use]
pub fn route_all(
    rules: &[MultiSignalRule],
    records: &[SemanticStateRecord],
    now_ms: i64,
) -> Vec<AgentRoute> {
    let mut routes: Vec<AgentRoute> =
        rules.iter().filter_map(|r| r.evaluate(records, now_ms)).collect();
    routes.sort_by(|a, b| b.severity.cmp(&a.severity).then(a.route_id.cmp(b.route_id)));
    routes.dedup();
    routes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::common::Reason;

    fn event(kind: SemanticKind, state: PrimitiveState, ts: i64) -> SemanticEvent {
        SemanticEvent { kind, state, node_id: "node-1".into(), timestamp_ms: ts }
    }

    #[test]
    fn record_from_scalar_event_carries_provenance() {
        let ctx = RecordContext {
            model_version: "rfenc-1.2".into(),
            calibration_version: "cal:abc".into(),
            privacy_action: PrivacyAction::Allow,
            default_ttl_ms: 30_000,
        };
        let ev = event(
            SemanticKind::FallRisk,
            PrimitiveState::Scalar { value: 80.0, reason: Reason::new(&["accel_spike", "hr=110bpm"]) },
            1_000,
        );
        let r = SemanticStateRecord::from_event(&ev, Some("living_room".into()), &ctx);
        assert_eq!(r.model_version, "rfenc-1.2");
        assert_eq!(r.calibration_version, "cal:abc");
        assert_eq!(r.expiry_at_ms, 31_000);
        assert!((r.confidence - 0.8).abs() < 1e-6);
        assert!(r.active);
        assert_eq!(r.room.as_deref(), Some("living_room"));
        assert!(r.is_fresh(20_000) && !r.is_fresh(31_000));
    }

    #[test]
    fn strip_biometrics_removes_hr_br_tags() {
        let ctx = RecordContext { privacy_action: PrivacyAction::StripBiometrics, ..Default::default() };
        let ev = event(
            SemanticKind::PossibleDistress,
            PrimitiveState::Scalar { value: 50.0, reason: Reason::new(&["motion<5%", "hr=130bpm", "br=22"]) },
            0,
        );
        let r = SemanticStateRecord::from_event(&ev, None, &ctx);
        assert_eq!(r.evidence_refs, vec!["motion<5%".to_string()]);
    }

    #[test]
    fn multi_signal_rule_fires_only_on_agreement() {
        let now = 1_000;
        let ctx = RecordContext::default();
        let fall = SemanticStateRecord::from_event(
            &event(SemanticKind::FallRisk, PrimitiveState::Scalar { value: 90.0, reason: Reason::empty() }, now),
            Some("bedroom".into()), &ctx,
        );
        let elderly = SemanticStateRecord::from_event(
            &event(SemanticKind::ElderlyAnomaly, PrimitiveState::Boolean { active: true, changed: true, reason: Reason::empty() }, now),
            Some("bedroom".into()), &ctx,
        );
        let rule = MultiSignalRule {
            required_kinds: vec![SemanticKind::FallRisk, SemanticKind::ElderlyAnomaly],
            min_confidence: 0.5,
            route: AgentRoute { route_id: "caregiver_escalation", severity: 3 },
        };

        // Only fall present → no route (no agreement).
        assert_eq!(rule.evaluate(&[fall.clone()], now), None);
        // Both present + active + fresh → route fires.
        assert_eq!(
            rule.evaluate(&[fall.clone(), elderly.clone()], now),
            Some(AgentRoute { route_id: "caregiver_escalation", severity: 3 })
        );
        // Stale records do not fire.
        assert_eq!(rule.evaluate(&[fall.clone(), elderly.clone()], now + 60_000), None);
    }

    /// ADR-140 acceptance (the credibility path):
    /// `raw snapshot -> semantic primitive -> SemanticStateRecord ->
    ///  (HOMECORE state) -> Ruflo agreement rule -> expired record rejected`.
    #[test]
    fn acceptance_raw_snapshot_to_expired_rejection() {
        use crate::semantic::bus::SemanticBus;
        use crate::semantic::common::{PrimitiveConfig, RawSnapshot};
        use std::time::Duration;

        // raw snapshot (past the warmup window) with a fall detected.
        let mut bus = SemanticBus::new(PrimitiveConfig::default());
        let snap = RawSnapshot {
            node_id: "living_room".into(),
            since_start: Duration::from_secs(61),
            timestamp_ms: 1_000,
            fall_detected: true,
            motion: 0.5,
            ..Default::default()
        };

        // raw snapshot -> semantic primitive (real SemanticBus FSM tick).
        let events = bus.tick(&snap);
        let fall = events
            .iter()
            .find(|e| e.kind == SemanticKind::FallRisk)
            .expect("fall_detected past warmup must emit a FallRisk primitive");

        // semantic primitive -> SemanticStateRecord (provenance from real context).
        let ctx = RecordContext {
            model_version: "rfenc-v1".into(),
            calibration_version: "cal:abc".into(),
            privacy_action: PrivacyAction::Allow,
            default_ttl_ms: 30_000,
        };
        let rec = SemanticStateRecord::from_event(fall, Some("living_room".into()), &ctx);
        // -> HOMECORE state: the record IS the operational state (room + provenance).
        assert!(rec.active && rec.confidence > 0.0);
        assert_eq!(rec.room.as_deref(), Some("living_room"));
        assert_eq!(rec.model_version, "rfenc-v1");
        assert_eq!(rec.calibration_version, "cal:abc");

        let now = snap.timestamp_ms;
        // -> Ruflo agreement rule. A single-signal rule fires on the fresh record;
        //    a genuine multi-signal rule does NOT (agreement required → no false alarm).
        let single = MultiSignalRule {
            required_kinds: vec![SemanticKind::FallRisk],
            min_confidence: 0.1,
            route: AgentRoute { route_id: "fall_notice", severity: 2 },
        };
        assert!(single.evaluate(std::slice::from_ref(&rec), now).is_some());
        let agreement = MultiSignalRule {
            required_kinds: vec![SemanticKind::FallRisk, SemanticKind::ElderlyAnomaly],
            min_confidence: 0.1,
            route: AgentRoute { route_id: "caregiver_escalation", severity: 3 },
        };
        assert!(
            agreement.evaluate(std::slice::from_ref(&rec), now).is_none(),
            "no caregiver escalation without multi-signal agreement"
        );

        // -> expired record rejected (stale belief must not become fake truth).
        let after_expiry = rec.expiry_at_ms + 1;
        assert!(!rec.is_fresh(after_expiry));
        assert!(
            single.evaluate(std::slice::from_ref(&rec), after_expiry).is_none(),
            "an expired record fires no route"
        );
    }

    #[test]
    fn route_all_sorts_by_severity_and_dedups() {
        let now = 0;
        let ctx = RecordContext::default();
        let active = |k| SemanticStateRecord::from_event(
            &event(k, PrimitiveState::Boolean { active: true, changed: true, reason: Reason::empty() }, now),
            None, &ctx,
        );
        let records = vec![active(SemanticKind::FallRisk), active(SemanticKind::NoMovement)];
        let rules = vec![
            MultiSignalRule { required_kinds: vec![SemanticKind::FallRisk], min_confidence: 0.5, route: AgentRoute { route_id: "fall_notice", severity: 2 } },
            MultiSignalRule { required_kinds: vec![SemanticKind::NoMovement, SemanticKind::FallRisk], min_confidence: 0.5, route: AgentRoute { route_id: "safety_critical", severity: 3 } },
        ];
        let routes = route_all(&rules, &records, now);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].route_id, "safety_critical"); // higher severity first
    }
}
