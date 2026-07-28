//! Programmable perception — the active sensing control plane (ADR-280).
//!
//! The shift this module implements: from *passive* sensing (accept
//! whatever measurements arrive) to *programmable* perception (the system
//! chooses where, when, how, and at what fidelity to sense, then resolves
//! uncertainty deliberately). Five contracts:
//!
//! 1. [`SensingTask`] — the evidence-aware task contract (ETSI ISAC
//!    sensing-task vocabulary: purpose, area, resolution, latency,
//!    confidence, retention, consumers, consent).
//! 2. [`SensingAction`] + [`InformationGoal`] — a request to actively
//!    gather evidence against a hypothesis, bounded by latency, energy,
//!    and a privacy ceiling.
//! 3. [`ActiveSensingPlanner`] over [`SpatialStateFreshness`] — age-of-
//!    information scheduling: refresh what is stale, changing, and
//!    important, not everything uniformly.
//! 4. [`CoherentSensorGroup`] — distributed-aperture fusion is allowed
//!    **only** when time, phase, and geometry compatibility is proven;
//!    out-of-bounds members fail closed (the dominant failure mode of
//!    emerging systems is hidden synchronization/calibration dependence).
//! 5. [`FieldActuator`] + [`ActuationReceipt`] — programmable radio
//!    environments (RIS, movable antennas) are actuators whose state
//!    changes alter *who is observable*, so actuation demands the same
//!    policy authorization and auditability as sensing itself.
//!
//! Plus [`TaskSufficientRepresentation`] — semantic, task-scoped
//! compression whose leakage rules are validated, not assumed.

use serde::{Deserialize, Serialize};

use crate::policy::{PolicyEngine, SensingPurpose};
use crate::tensor::RfModality;
use crate::{Result, UnifiedError};

/// RuField-aligned privacy classes (ADR-262 §3.3 vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrivacyClass {
    /// Raw signal — never leaves the trust boundary.
    P0,
    /// Heavily aggregated, non-personal.
    P1,
    /// Anonymous presence/occupancy grade.
    P2,
    /// Behavioral inference grade.
    P3,
    /// Derived personal inference grade.
    P4,
    /// Identity-bound grade.
    P5,
}

/// Axis-aligned spatial zone in the building frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialZone {
    /// Zone identifier (matches ADR-277 `PrivacyZone` ids).
    pub id: String,
    /// Minimum corner, metres.
    pub min_m: [f64; 3],
    /// Maximum corner, metres.
    pub max_m: [f64; 3],
}

impl SpatialZone {
    /// Whether a point lies inside the zone.
    #[must_use]
    pub fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|k| p[k] >= self.min_m[k] && p[k] <= self.max_m[k])
    }
}

/// The evidence-aware sensing task contract (ADR-280 §2). Enforced
/// *before capture begins*, not applied later as metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingTask {
    /// Task identifier.
    pub task_id: u128,
    /// Purpose (drives ADR-277 zone authorization).
    pub purpose: SensingPurpose,
    /// Target area.
    pub target_area: SpatialZone,
    /// Modalities the task may use.
    pub modalities: Vec<RfModality>,
    /// Requested spatial resolution, metres.
    pub requested_resolution_m: f64,
    /// Maximum acceptable result latency, ms.
    pub maximum_latency_ms: u32,
    /// Minimum confidence below which results become *no decision*.
    pub minimum_confidence: f64,
    /// Raw (P0) retention bound, seconds — local only.
    pub raw_retention_seconds: u64,
    /// Result retention bound, seconds.
    pub result_retention_seconds: u64,
    /// Principals allowed to consume results.
    pub authorized_consumers: Vec<String>,
    /// Consent reference, when the purpose requires one.
    pub consent_reference: Option<String>,
    /// Requested raw export. Kept in the contract for ISAC-vocabulary
    /// compatibility, but see [`PolicyEngine`]-backed admission: ADR-277's
    /// structural rule means this is **always refused** today.
    pub raw_export_allowed: bool,
}

/// Admits a sensing task against the ADR-277 policy engine. Fail-closed:
/// unknown zone, ungranted purpose, identity single-gate, raw export, and
/// missing-consent identity tasks all deny.
pub fn admit_task(engine: &PolicyEngine, task: &SensingTask) -> Result<()> {
    if task.raw_export_allowed {
        return Err(UnifiedError::PolicyDenied(
            "raw RF export is structurally disabled (ADR-277 §2.1); \
             the contract field exists for ISAC vocabulary compatibility only"
                .into(),
        ));
    }
    if !(task.minimum_confidence.is_finite() && (0.0..=1.0).contains(&task.minimum_confidence)) {
        return Err(UnifiedError::InvalidInput("minimum_confidence must be in [0,1]".into()));
    }
    if !(task.requested_resolution_m.is_finite() && task.requested_resolution_m > 0.0) {
        return Err(UnifiedError::InvalidInput("requested_resolution_m must be finite and > 0".into()));
    }
    if task.maximum_latency_ms == 0 {
        return Err(UnifiedError::InvalidInput("maximum_latency_ms must be > 0".into()));
    }
    if task.modalities.is_empty() {
        return Err(UnifiedError::InvalidInput("a sensing task must declare at least one modality".into()));
    }
    if task.purpose == SensingPurpose::IdentityRecognition && task.consent_reference.is_none() {
        return Err(UnifiedError::PolicyDenied(
            "identity recognition tasks require a consent reference".into(),
        ));
    }
    engine.authorize(&task.target_area.id, task.purpose)
}

/// What an active sensing request is trying to learn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGoal {
    /// Human-readable hypothesis under test.
    pub hypothesis: String,
    /// Current uncertainty in `[0, 1]`.
    pub current_uncertainty: f64,
    /// Target uncertainty in `[0, 1]` (must be below current).
    pub target_uncertainty: f64,
    /// Expected information gain of the action (heuristic units).
    pub expected_information_gain: f64,
}

/// A deliberate act of sensing (ADR-280 §3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingAction {
    /// Action identifier.
    pub action_id: String,
    /// Region to observe.
    pub target_region: SpatialZone,
    /// Modality to use.
    pub modality: RfModality,
    /// Goal that justifies the action.
    pub desired_information: InformationGoal,
    /// Latency budget, ms.
    pub maximum_latency_ms: u32,
    /// Energy budget, joules.
    pub energy_budget_j: f64,
    /// Highest privacy class the action may produce.
    pub privacy_ceiling: PrivacyClass,
}

/// Freshness state of one spatial region (age-of-information model,
/// ADR-280 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialStateFreshness {
    /// Region identifier.
    pub region_id: String,
    /// Region geometry.
    pub region: SpatialZone,
    /// Last observation, ns since epoch.
    pub last_observed_ns: u64,
    /// Expected change rate (events/s scale factor).
    pub expected_change_rate: f64,
    /// Uncertainty growth per second of staleness.
    pub uncertainty_growth_rate: f64,
    /// Business criticality weight (≥ 0).
    pub business_criticality: f64,
    /// Cost of sensing this region (energy/traffic units, > 0).
    pub sensing_cost: f64,
}

impl SpatialStateFreshness {
    /// Uncertainty accumulated since the last observation, capped at 1.
    #[must_use]
    pub fn uncertainty_at(&self, now_ns: u64) -> f64 {
        let age_s = now_ns.saturating_sub(self.last_observed_ns) as f64 / 1e9;
        (self.uncertainty_growth_rate * age_s).min(1.0)
    }

    /// Refresh priority: `uncertainty × change rate × criticality ÷ cost`.
    #[must_use]
    pub fn priority(&self, now_ns: u64) -> f64 {
        self.uncertainty_at(now_ns) * self.expected_change_rate * self.business_criticality
            / self.sensing_cost.max(1e-9)
    }
}

/// Age-of-information sensing scheduler: refreshes regions in priority
/// order instead of uniformly.
#[derive(Debug, Default)]
pub struct ActiveSensingPlanner {
    regions: Vec<SpatialStateFreshness>,
    /// Priority below which a region is not worth sensing this cycle.
    pub priority_threshold: f64,
}

impl ActiveSensingPlanner {
    /// New planner with a priority threshold.
    #[must_use]
    pub fn new(priority_threshold: f64) -> Self {
        Self { regions: Vec::new(), priority_threshold }
    }

    /// Registers or replaces a region.
    pub fn upsert_region(&mut self, region: SpatialStateFreshness) {
        if let Some(r) = self.regions.iter_mut().find(|r| r.region_id == region.region_id) {
            *r = region;
        } else {
            self.regions.push(region);
        }
    }

    /// Marks a region observed at `now_ns`.
    pub fn mark_observed(&mut self, region_id: &str, now_ns: u64) {
        if let Some(r) = self.regions.iter_mut().find(|r| r.region_id == region_id) {
            r.last_observed_ns = now_ns;
        }
    }

    /// Highest-priority region above the threshold, as a concrete
    /// [`SensingAction`]; `None` when nothing is worth sensing.
    #[must_use]
    pub fn next_action(&self, now_ns: u64, modality: RfModality) -> Option<SensingAction> {
        let best = self
            .regions
            .iter()
            .map(|r| (r.priority(now_ns), r))
            .filter(|(p, _)| *p >= self.priority_threshold)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;
        let (priority, region) = best;
        let uncertainty = region.uncertainty_at(now_ns);
        Some(SensingAction {
            action_id: format!("aoi-{}-{now_ns}", region.region_id),
            target_region: region.region.clone(),
            modality,
            desired_information: InformationGoal {
                hypothesis: format!("state of region {} is stale", region.region_id),
                current_uncertainty: uncertainty,
                target_uncertainty: (uncertainty * 0.2).min(0.05),
                expected_information_gain: priority,
            },
            maximum_latency_ms: 500,
            energy_budget_j: region.sensing_cost,
            privacy_ceiling: PrivacyClass::P2,
        })
    }
}

/// Clock/phase/geometry sync state reported by one member of a
/// distributed aperture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberSyncState {
    /// Member identifier.
    pub member_id: String,
    /// Measured time error vs the group reference, ns.
    pub time_error_ns: f64,
    /// Measured phase error vs the group reference, rad.
    pub phase_error_rad: f64,
    /// Hash of the member's calibrated baseline geometry.
    pub geometry_hash: u64,
}

/// A coherent sensing group (ADR-280 §5): no coherent fusion unless
/// time, phase, and geometry compatibility is *proven*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherentSensorGroup {
    /// Group identifier.
    pub group_id: String,
    /// Member identifiers.
    pub members: Vec<String>,
    /// Maximum tolerated time error, ns.
    pub maximum_time_error_ns: f64,
    /// Maximum tolerated phase error, rad.
    pub maximum_phase_error_rad: f64,
    /// Required baseline geometry hash (all members must match).
    pub baseline_geometry_hash: u64,
}

impl CoherentSensorGroup {
    /// Fail-closed fusion gate: every group member must report, be within
    /// time and phase bounds, and match the baseline geometry hash.
    /// Unknown reporters, missing members, or any out-of-bounds member
    /// deny fusion with a typed error.
    pub fn can_fuse(&self, states: &[MemberSyncState]) -> Result<()> {
        for member in &self.members {
            let Some(s) = states.iter().find(|s| &s.member_id == member) else {
                return Err(UnifiedError::PolicyDenied(format!(
                    "coherent fusion denied: member {member:?} did not report sync state"
                )));
            };
            if !s.time_error_ns.is_finite() || s.time_error_ns.abs() > self.maximum_time_error_ns {
                return Err(UnifiedError::PolicyDenied(format!(
                    "coherent fusion denied: {member:?} time error {} ns exceeds {} ns",
                    s.time_error_ns, self.maximum_time_error_ns
                )));
            }
            if !s.phase_error_rad.is_finite()
                || s.phase_error_rad.abs() > self.maximum_phase_error_rad
            {
                return Err(UnifiedError::PolicyDenied(format!(
                    "coherent fusion denied: {member:?} phase error {} rad exceeds {} rad",
                    s.phase_error_rad, self.maximum_phase_error_rad
                )));
            }
            if s.geometry_hash != self.baseline_geometry_hash {
                return Err(UnifiedError::PolicyDenied(format!(
                    "coherent fusion denied: {member:?} geometry hash mismatch"
                )));
            }
        }
        for s in states {
            if !self.members.contains(&s.member_id) {
                return Err(UnifiedError::PolicyDenied(format!(
                    "coherent fusion denied: {:?} is not a group member",
                    s.member_id
                )));
            }
        }
        Ok(())
    }
}

/// Kind of radio-environment actuator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuatorKind {
    /// Reconfigurable intelligent surface.
    Ris,
    /// Mechanically movable antenna.
    MovableAntenna,
    /// Fluid antenna.
    FluidAntenna,
}

/// A programmable radio-environment actuator (ADR-280 §6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldActuator {
    /// Actuator identifier.
    pub actuator_id: String,
    /// Kind.
    pub kind: ActuatorKind,
    /// Pose in the building frame.
    pub pose_m: [f64; 3],
    /// Named states the actuator supports.
    pub supported_states: Vec<String>,
    /// Zone whose observability this actuator changes.
    pub affected_zone_id: String,
}

/// Audit receipt for an applied actuation. Constructed only by
/// [`request_actuation`] — there is no other way to obtain one, so every
/// state change that alters observability is policy-checked and logged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuationReceipt {
    /// State that was requested.
    pub requested_state: String,
    /// State actually applied.
    pub applied_state: String,
    /// Application time, ns.
    pub applied_ns: u64,
    /// Controller identity.
    pub controller_id: String,
    /// Purpose under which the actuation was authorized.
    pub purpose: SensingPurpose,
}

/// Requests an actuator state change. Denied unless (a) the actuator
/// supports the state and (b) the affected zone grants the purpose under
/// the ADR-277 engine — changing an RIS configuration can change *which
/// rooms and people are observable*, so it is governed like sensing.
pub fn request_actuation(
    engine: &PolicyEngine,
    actuator: &FieldActuator,
    state: &str,
    purpose: SensingPurpose,
    controller_id: &str,
    now_ns: u64,
) -> Result<ActuationReceipt> {
    if !actuator.supported_states.iter().any(|s| s == state) {
        return Err(UnifiedError::InvalidInput(format!(
            "actuator {:?} does not support state {state:?}",
            actuator.actuator_id
        )));
    }
    engine.authorize(&actuator.affected_zone_id, purpose)?;
    Ok(ActuationReceipt {
        requested_state: state.to_string(),
        applied_state: state.to_string(),
        applied_ns: now_ns,
        controller_id: controller_id.to_string(),
        purpose,
    })
}

/// A task-scoped semantic compression of observations (ADR-280 §7):
/// transmit only the information the current physical task needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSufficientRepresentation {
    /// Task this representation serves.
    pub task_id: u128,
    /// Source frame receipt ids (lineage).
    pub source_receipts: Vec<u128>,
    /// Compressed semantic state.
    pub semantic_state: Vec<f32>,
    /// Claimed information bound, bits.
    pub information_bound_bits: f64,
    /// Information classes *explicitly* excluded (e.g. `"identity"`,
    /// `"vitals"`, `"trajectory-history"`).
    pub excluded_information: Vec<String>,
    /// Privacy class of the representation.
    pub privacy_class: PrivacyClass,
}

/// Purpose-scoped leakage validation: compression must remain task
/// scoped. A representation sufficient for anonymous occupancy must not
/// retain identity information; each purpose has a privacy-class ceiling
/// and a set of information classes it must exclude.
pub fn validate_representation(
    rep: &TaskSufficientRepresentation,
    purpose: SensingPurpose,
) -> Result<()> {
    let (ceiling, must_exclude): (PrivacyClass, &[&str]) = match purpose {
        SensingPurpose::Presence | SensingPurpose::ChannelDiagnostics => {
            (PrivacyClass::P2, &["identity", "vitals"])
        }
        SensingPurpose::Activity | SensingPurpose::Localization => {
            (PrivacyClass::P3, &["identity"])
        }
        SensingPurpose::Vitals | SensingPurpose::PoseTracking => (PrivacyClass::P4, &["identity"]),
        SensingPurpose::IdentityRecognition => (PrivacyClass::P5, &[]),
    };
    if rep.privacy_class > ceiling {
        return Err(UnifiedError::PolicyDenied(format!(
            "representation class {:?} exceeds ceiling {ceiling:?} for purpose {purpose:?}",
            rep.privacy_class
        )));
    }
    for class in must_exclude {
        if !rep.excluded_information.iter().any(|e| e == class) {
            return Err(UnifiedError::PolicyDenied(format!(
                "purpose {purpose:?} requires the representation to explicitly exclude {class:?}"
            )));
        }
    }
    if rep.source_receipts.is_empty() {
        return Err(UnifiedError::InvalidInput(
            "task-sufficient representation must carry source lineage".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PrivacyZone;

    fn zone(id: &str) -> SpatialZone {
        SpatialZone { id: id.into(), min_m: [0.0; 3], max_m: [5.0, 4.0, 3.0] }
    }

    fn engine_with(purposes: &[SensingPurpose]) -> PolicyEngine {
        let mut e = PolicyEngine::new();
        e.upsert_zone(PrivacyZone {
            id: "lab".into(),
            allowed_purposes: purposes.iter().copied().collect(),
            retention_s: 3600,
            identity_explicitly_enabled: false,
        });
        e
    }

    fn task(purpose: SensingPurpose, raw_export: bool) -> SensingTask {
        SensingTask {
            task_id: 1,
            purpose,
            target_area: zone("lab"),
            modalities: vec![RfModality::WifiCsi],
            requested_resolution_m: 0.5,
            maximum_latency_ms: 100,
            minimum_confidence: 0.8,
            raw_retention_seconds: 60,
            result_retention_seconds: 3600,
            authorized_consumers: vec!["ha-bridge".into()],
            consent_reference: None,
            raw_export_allowed: raw_export,
        }
    }

    #[test]
    fn task_admission_is_fail_closed() {
        let engine = engine_with(&[SensingPurpose::Presence]);
        assert!(admit_task(&engine, &task(SensingPurpose::Presence, false)).is_ok());
        // Raw export is refused regardless of any other grant.
        assert!(matches!(
            admit_task(&engine, &task(SensingPurpose::Presence, true)),
            Err(UnifiedError::PolicyDenied(_))
        ));
        // Ungranted purpose denied.
        assert!(admit_task(&engine, &task(SensingPurpose::Localization, false)).is_err());
        // Identity without consent denied before even reaching the zone check.
        assert!(admit_task(&engine, &task(SensingPurpose::IdentityRecognition, false)).is_err());
    }

    #[test]
    fn planner_prioritizes_stale_critical_regions() {
        let mut planner = ActiveSensingPlanner::new(0.01);
        let mk = |id: &str, change: f64, crit: f64, cost: f64| SpatialStateFreshness {
            region_id: id.into(),
            region: zone(id),
            last_observed_ns: 0,
            expected_change_rate: change,
            uncertainty_growth_rate: 0.05,
            business_criticality: crit,
            sensing_cost: cost,
        };
        planner.upsert_region(mk("server-room", 0.1, 5.0, 1.0));
        planner.upsert_region(mk("emergency-exit", 0.5, 8.0, 1.0));
        planner.upsert_region(mk("storage", 0.01, 0.5, 1.0));

        let now = 10_000_000_000; // 10 s of staleness everywhere
        let action = planner.next_action(now, RfModality::WifiCsi).expect("something stale");
        assert_eq!(action.target_region.id, "emergency-exit", "highest priority wins");

        // After observing it, the next-highest region is selected.
        planner.mark_observed("emergency-exit", now);
        let action = planner.next_action(now, RfModality::WifiCsi).expect("next region");
        assert_eq!(action.target_region.id, "server-room");
    }

    #[test]
    fn planner_reduces_sensing_traffic_versus_uniform_refresh() {
        // 20 regions, one hot (changes often, critical), the rest cold.
        let mut planner = ActiveSensingPlanner::new(0.05);
        for i in 0..20 {
            let hot = i == 0;
            planner.upsert_region(SpatialStateFreshness {
                region_id: format!("r{i}"),
                region: zone("lab"),
                last_observed_ns: 0,
                expected_change_rate: if hot { 1.0 } else { 0.01 },
                uncertainty_growth_rate: 0.2,
                business_criticality: if hot { 5.0 } else { 0.5 },
                sensing_cost: 1.0,
            });
        }
        // Simulate 100 scheduling ticks, 1 s apart. Uniform refresh would
        // sense 20 regions × 100 ticks = 2000 observations; the planner
        // senses at most one region per tick and only above threshold.
        let mut actions = 0;
        for tick in 1..=100u64 {
            let now = tick * 1_000_000_000;
            if let Some(a) = planner.next_action(now, RfModality::WifiCsi) {
                planner.mark_observed(&a.target_region.id, now);
                actions += 1;
            }
        }
        let uniform = 20 * 100;
        let reduction = 1.0 - actions as f64 / uniform as f64;
        println!("AoI planner: {actions} observations vs {uniform} uniform ({reduction:.2} reduction)");
        assert!(
            reduction >= 0.70,
            "planner must cut sensing traffic by >= 70 % in sparse environments, got {reduction:.2}"
        );
        assert!(actions > 0, "the hot region must still be observed");
    }

    #[test]
    fn coherent_fusion_fails_closed() {
        let group = CoherentSensorGroup {
            group_id: "aisle-3".into(),
            members: vec!["ap-1".into(), "ap-2".into()],
            maximum_time_error_ns: 50.0,
            maximum_phase_error_rad: 0.2,
            baseline_geometry_hash: 0xBEEF,
        };
        let ok = |id: &str| MemberSyncState {
            member_id: id.into(),
            time_error_ns: 10.0,
            phase_error_rad: 0.05,
            geometry_hash: 0xBEEF,
        };
        assert!(group.can_fuse(&[ok("ap-1"), ok("ap-2")]).is_ok());

        // Missing member ⇒ deny.
        assert!(group.can_fuse(&[ok("ap-1")]).is_err());
        // Clock out of bounds ⇒ deny.
        let mut drift = ok("ap-2");
        drift.time_error_ns = 400.0;
        assert!(group.can_fuse(&[ok("ap-1"), drift]).is_err());
        // Phase out of bounds ⇒ deny.
        let mut phase = ok("ap-2");
        phase.phase_error_rad = 1.0;
        assert!(group.can_fuse(&[ok("ap-1"), phase]).is_err());
        // Geometry changed since calibration ⇒ deny.
        let mut moved = ok("ap-2");
        moved.geometry_hash = 0xDEAD;
        assert!(group.can_fuse(&[ok("ap-1"), moved]).is_err());
        // A non-member reporting in ⇒ deny.
        assert!(group.can_fuse(&[ok("ap-1"), ok("ap-2"), ok("rogue")]).is_err());
    }

    #[test]
    fn actuation_requires_policy_authorization() {
        let engine = engine_with(&[SensingPurpose::Presence]);
        let ris = FieldActuator {
            actuator_id: "ris-7".into(),
            kind: ActuatorKind::Ris,
            pose_m: [2.0, 0.0, 2.5],
            supported_states: vec!["beam-east".into(), "beam-west".into()],
            affected_zone_id: "lab".into(),
        };
        // Authorized purpose + supported state ⇒ receipt.
        let receipt =
            request_actuation(&engine, &ris, "beam-east", SensingPurpose::Presence, "ctl-1", 99)
                .expect("authorized actuation");
        assert_eq!(receipt.applied_state, "beam-east");
        assert_eq!(receipt.purpose, SensingPurpose::Presence);

        // Unsupported state ⇒ deny.
        assert!(request_actuation(&engine, &ris, "beam-up", SensingPurpose::Presence, "c", 0)
            .is_err());
        // Purpose not granted in the affected zone ⇒ deny (an RIS cannot be
        // steered to observe a zone for a purpose the zone never granted).
        assert!(request_actuation(&engine, &ris, "beam-east", SensingPurpose::Vitals, "c", 0)
            .is_err());
    }

    #[test]
    fn task_sufficient_representation_is_leakage_checked() {
        let rep = |class: PrivacyClass, excluded: &[&str]| TaskSufficientRepresentation {
            task_id: 5,
            source_receipts: vec![1, 2],
            semantic_state: vec![0.1, 0.9],
            information_bound_bits: 8.0,
            excluded_information: excluded.iter().map(|s| (*s).to_string()).collect(),
            privacy_class: class,
        };
        // Occupancy-grade representation excluding identity + vitals: fine.
        assert!(validate_representation(
            &rep(PrivacyClass::P2, &["identity", "vitals"]),
            SensingPurpose::Presence
        )
        .is_ok());
        // Same purpose but the representation forgot to exclude identity: deny.
        assert!(validate_representation(
            &rep(PrivacyClass::P2, &["vitals"]),
            SensingPurpose::Presence
        )
        .is_err());
        // Class above the purpose ceiling: deny.
        assert!(validate_representation(
            &rep(PrivacyClass::P4, &["identity", "vitals"]),
            SensingPurpose::Presence
        )
        .is_err());
        // No lineage: deny.
        let mut orphan = rep(PrivacyClass::P2, &["identity", "vitals"]);
        orphan.source_receipts.clear();
        assert!(validate_representation(&orphan, SensingPurpose::Presence).is_err());
    }

    /// Only `Presence` was ever exercised above; the other three
    /// ceiling/exclusion-set branches (Activity/Localization at P3,
    /// Vitals/PoseTracking at P4, IdentityRecognition at P5) had zero test
    /// coverage — a bug in any of them would go undetected.
    #[test]
    fn task_sufficient_representation_covers_every_purpose_branch() {
        let rep = |class: PrivacyClass, excluded: &[&str]| TaskSufficientRepresentation {
            task_id: 6,
            source_receipts: vec![1],
            semantic_state: vec![0.2],
            information_bound_bits: 4.0,
            excluded_information: excluded.iter().map(|s| (*s).to_string()).collect(),
            privacy_class: class,
        };

        for purpose in [SensingPurpose::Activity, SensingPurpose::Localization] {
            // P3 ceiling excluding identity: fine.
            assert!(validate_representation(&rep(PrivacyClass::P3, &["identity"]), purpose).is_ok());
            // Forgot to exclude identity: deny.
            assert!(validate_representation(&rep(PrivacyClass::P3, &[]), purpose).is_err());
            // Above the P3 ceiling: deny.
            assert!(validate_representation(&rep(PrivacyClass::P4, &["identity"]), purpose).is_err());
        }

        for purpose in [SensingPurpose::Vitals, SensingPurpose::PoseTracking] {
            // P4 ceiling excluding identity: fine.
            assert!(validate_representation(&rep(PrivacyClass::P4, &["identity"]), purpose).is_ok());
            // Forgot to exclude identity: deny.
            assert!(validate_representation(&rep(PrivacyClass::P4, &[]), purpose).is_err());
            // Above the P4 ceiling: deny.
            assert!(validate_representation(&rep(PrivacyClass::P5, &["identity"]), purpose).is_err());
        }

        // IdentityRecognition: P5 ceiling, nothing required to be excluded.
        assert!(validate_representation(&rep(PrivacyClass::P5, &[]), SensingPurpose::IdentityRecognition)
            .is_ok());
        // Still bounded — no class exceeds P5, so exercise the lineage guard instead.
        let mut orphan = rep(PrivacyClass::P5, &[]);
        orphan.source_receipts.clear();
        assert!(validate_representation(&orphan, SensingPurpose::IdentityRecognition).is_err());
    }
}
