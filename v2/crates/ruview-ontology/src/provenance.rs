//! Evidence ladder and provenance carried by every fact (ADR-303 §2, ADR-282).
//!
//! The ontology mandates that a fact cannot cross a surface boundary and lose
//! its lineage: every leaf entity carries exactly one [`EvidenceLevel`] plus a
//! [`SemanticProvenance`] record, so no projection can silently upgrade or drop
//! the evidence level.

use serde::{Deserialize, Serialize};

/// The ADR-282 evidence ladder, L0–L5. Exactly one level travels with each
/// fact. Ordering is meaningful: `L0 < L1 < … < L5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceLevel {
    /// L0 — declared/assumed, no signal evidence.
    L0,
    /// L1 — heuristic/synthetic.
    L1,
    /// L2 — single-surface signal evidence.
    L2,
    /// L3 — corroborated across surfaces.
    L3,
    /// L4 — calibrated and held-out validated.
    L4,
    /// L5 — witnessed / certified (ADR-316).
    L5,
}

/// Mandatory provenance for every fact (mirrors the `worldgraph`
/// `SemanticProvenance` house rule so the two can map losslessly). Every field
/// is a bounded string handle, not embedded data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticProvenance {
    /// Evidence content-address handle(s) (ADR-137 `EvidenceRef`).
    #[serde(default)]
    pub evidence: Vec<String>,
    /// Model version that produced the fact (ADR-136).
    pub model_version: String,
    /// Calibration baseline in effect (ADR-135/ADR-298).
    pub calibration_version: String,
    /// Privacy decision the fact was derived under (ADR-141).
    pub privacy_decision: String,
}

impl SemanticProvenance {
    /// A minimal declared-provenance record for L0/L1 structural facts that
    /// have no signal evidence yet. Deterministic; no I/O.
    #[must_use]
    pub fn declared(model_version: impl Into<String>) -> Self {
        Self {
            evidence: Vec::new(),
            model_version: model_version.into(),
            calibration_version: "none".to_string(),
            privacy_decision: "none".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_level_orders_ascending() {
        assert!(EvidenceLevel::L0 < EvidenceLevel::L5);
        assert!(EvidenceLevel::L3 > EvidenceLevel::L2);
    }
}
