//! Tuning for the association / lifecycle / decay policy (ADR-304).
//!
//! All thresholds are explicit and deterministic; nothing here reads a clock or
//! draws randomness. The manager injects every timestamp.

use ruview_ontology::{EvidenceLevel, SemanticProvenance};

/// Bounded-cost association, lifecycle, and decay policy.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Maximum Euclidean position distance for a value-gate pass (same units as
    /// [`Detection::position`](crate::Detection)).
    pub gate_position: f64,
    /// Maximum coarse-feature L1 distance for a value-gate pass.
    pub gate_feature: f64,
    /// Minimum cost separation between the best and second-best candidate track
    /// for an assignment to be *unambiguous*. If two tracks are within this
    /// margin the detection is left tentative rather than risk a swap.
    pub ambiguity_margin: f64,
    /// Associated detections required to promote a tentative track to active.
    pub confirm_after: u32,
    /// Idle gap (ms) after which an active track is marked lost (still
    /// re-identifiable within [`max_coast_ms`](Self::max_coast_ms)).
    pub lost_after_ms: i64,
    /// Association horizon (ms). Beyond this idle gap a track is expired and a
    /// fresh pseudonym is minted rather than forcing a join — under-linking is
    /// the privacy-safe failure mode.
    pub max_coast_ms: i64,
    /// Relative weight of the position term in the association cost.
    pub w_pos: f64,
    /// Relative weight of the feature term in the association cost.
    pub w_feat: f64,
    /// Evidence level stamped on emitted [`Track`](ruview_ontology::Track) /
    /// [`Person`](ruview_ontology::Person) nodes. Defaults to `L1`
    /// (heuristic/synthetic); this crate asserts no accuracy number.
    pub emit_evidence_level: EvidenceLevel,
    /// Provenance stamped on emitted nodes. Carries the pseudonymous privacy
    /// decision; never a civil identifier.
    pub provenance: SemanticProvenance,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            gate_position: 2.0,
            gate_feature: 6.0,
            ambiguity_margin: 0.15,
            confirm_after: 2,
            lost_after_ms: 1_000,
            max_coast_ms: 5_000,
            w_pos: 1.0,
            w_feat: 1.0,
            emit_evidence_level: EvidenceLevel::L1,
            provenance: SemanticProvenance {
                evidence: Vec::new(),
                model_version: "ruview-track".to_string(),
                calibration_version: "none".to_string(),
                privacy_decision: "pseudonymous".to_string(),
            },
        }
    }
}

impl TrackerConfig {
    /// Normalizing denominator for the association cost (`w_pos + w_feat`).
    /// Guarded to a positive value so confidence math never divides by zero.
    #[must_use]
    pub(crate) fn weight_sum(&self) -> f64 {
        let s = self.w_pos + self.w_feat;
        if s > 0.0 {
            s
        } else {
            1.0
        }
    }
}
