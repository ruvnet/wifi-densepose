//! Bound lifecycle metadata for the sensing server's in-memory field model.
//!
//! The numerical [`FieldModel`](wifi_densepose_signal::ruvsense::field_model::FieldModel)
//! remains owned by the binary's shared state.  This module owns the small,
//! auditable protocol boundary that prevents a model collected for one room or
//! node set from being relabelled as another installation.

use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_CALIBRATION_SOURCE_NODES: usize = 16;
pub const CALIBRATED_PRESENCE_EVIDENCE_SCHEMA: &str =
    "ruview.calibration.calibrated-presence-evidence.v2";
const SHA256_HEX_LEN: usize = 64;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationStartRequest {
    pub binding_digest: String,
    pub source_node_ids: Vec<u8>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationBoundRequest {
    pub boot_epoch: String,
    pub session_id: String,
    pub binding_digest: String,
    pub source_node_ids: Vec<u8>,
}

/// Reset normally carries the complete bound identity.  The all-`None` shape
/// exists only so an administrator can clear a legacy `--calibrate` model that
/// predates session binding; a bound session never accepts missing identity.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationResetRequest {
    pub boot_epoch: String,
    pub session_id: Option<String>,
    pub binding_digest: Option<String>,
    pub source_node_ids: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CalibrationSessionIdentity {
    pub boot_epoch: String,
    pub session_id: String,
    pub model_id: String,
    pub binding_digest: String,
    pub source_node_ids: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CalibrationModelReceipt {
    pub schema: &'static str,
    pub boot_epoch: String,
    pub session_id: String,
    pub model_id: String,
    pub binding_digest: String,
    pub source_node_ids: Vec<u8>,
    pub frame_count: u64,
    pub variance_explained: f64,
    pub baseline_eigenvalue_count: usize,
    pub completed_at_unix_ms: u64,
}

/// A per-frame result that can be joined back to one exact immutable field
/// model receipt. It is emitted only when the binary has produced a strict
/// calibrated occupancy result; heuristic fallbacks never receive this shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibratedPresenceEvidence {
    pub schema: String,
    pub boot_epoch: String,
    pub session_id: String,
    pub model_id: String,
    pub binding_digest: String,
    pub source_node_ids: Vec<u8>,
    pub model_completed_at_unix_ms: u64,
    pub inference_node_id: u8,
    pub source_tick: u64,
    pub observed_at_unix_ms: u64,
    pub inference_method: String,
    pub presence: bool,
    pub person_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_energy: Option<CalibratedResidualEnergyEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibratedNodeResidualEnergy {
    pub node_id: u8,
    pub energy: f64,
    pub null_median: f64,
    pub null_p95: f64,
    pub null_p99: f64,
    pub normalized_energy: f64,
    pub decision_threshold: f64,
    pub above_threshold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibratedResidualEnergyEvidence {
    pub nodes: Vec<CalibratedNodeResidualEnergy>,
    pub aggregate_mean_energy: f64,
    pub aggregate_mean_normalized_energy: f64,
    pub nodes_above_threshold: usize,
    pub node_quorum: usize,
    pub hysteresis_present: bool,
    pub hysteresis_candidate_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCalibrationBinding {
    binding_digest: String,
    source_node_ids: Vec<u8>,
}

impl ValidatedCalibrationBinding {
    pub fn binding_digest(&self) -> &str {
        &self.binding_digest
    }

    pub fn source_node_ids(&self) -> &[u8] {
        &self.source_node_ids
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationSessionError {
    #[error("binding_digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidBindingDigest,
    #[error("source_node_ids must contain 1 to 16 unique IDs in strictly ascending order")]
    InvalidSourceNodeIds,
    #[error("a calibration session is already active")]
    SessionAlreadyActive,
    #[error("no bound calibration session is active")]
    NoBoundSession,
    #[error("the calibration boot epoch does not match this server process")]
    BootEpochMismatch,
    #[error("the calibration session_id does not match the active session")]
    SessionMismatch,
    #[error("the calibration binding_digest does not match the active session")]
    BindingMismatch,
    #[error("the calibration source_node_ids do not match the active session")]
    SourceNodeMismatch,
    #[error("every bound source node must contribute before the model can be finalized")]
    UnobservedSourceNodes,
    #[error("the field model returned invalid completion metrics")]
    InvalidModelMetrics,
}

impl CalibrationSessionError {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidBindingDigest => "invalid_binding_digest",
            Self::InvalidSourceNodeIds => "invalid_source_node_ids",
            Self::SessionAlreadyActive => "calibration_session_active",
            Self::NoBoundSession => "no_bound_calibration_session",
            Self::BootEpochMismatch => "calibration_boot_epoch_mismatch",
            Self::SessionMismatch => "calibration_session_mismatch",
            Self::BindingMismatch => "calibration_binding_mismatch",
            Self::SourceNodeMismatch => "calibration_source_nodes_mismatch",
            Self::UnobservedSourceNodes => "calibration_source_nodes_unobserved",
            Self::InvalidModelMetrics => "invalid_calibration_model_metrics",
        }
    }
}

pub fn validate_start_request(
    request: CalibrationStartRequest,
) -> Result<ValidatedCalibrationBinding, CalibrationSessionError> {
    validate_binding_digest(&request.binding_digest)?;
    validate_source_node_ids(&request.source_node_ids)?;
    Ok(ValidatedCalibrationBinding {
        binding_digest: request.binding_digest,
        source_node_ids: request.source_node_ids,
    })
}

fn validate_binding_digest(value: &str) -> Result<(), CalibrationSessionError> {
    if value.len() == SHA256_HEX_LEN
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        Ok(())
    } else {
        Err(CalibrationSessionError::InvalidBindingDigest)
    }
}

fn validate_source_node_ids(value: &[u8]) -> Result<(), CalibrationSessionError> {
    if value.is_empty()
        || value.len() > MAX_CALIBRATION_SOURCE_NODES
        || value.windows(2).any(|pair| pair[0] >= pair[1])
    {
        Err(CalibrationSessionError::InvalidSourceNodeIds)
    } else {
        Ok(())
    }
}

fn opaque_id(prefix: &str) -> String {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    let mut value = String::with_capacity(prefix.len() + 1 + bytes.len() * 2);
    value.push_str(prefix);
    value.push('-');
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

#[derive(Debug)]
pub struct CalibrationSessionContract {
    boot_epoch: String,
    session: Option<CalibrationSessionIdentity>,
    model_receipt: Option<CalibrationModelReceipt>,
    observed_source_node_ids: Vec<u8>,
}

impl Default for CalibrationSessionContract {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationSessionContract {
    pub fn new() -> Self {
        Self {
            boot_epoch: opaque_id("cal-boot"),
            session: None,
            model_receipt: None,
            observed_source_node_ids: Vec::new(),
        }
    }

    pub fn boot_epoch(&self) -> &str {
        &self.boot_epoch
    }

    pub fn session(&self) -> Option<&CalibrationSessionIdentity> {
        self.session.as_ref()
    }

    pub fn model_receipt(&self) -> Option<&CalibrationModelReceipt> {
        self.model_receipt.as_ref()
    }

    /// Bind a strict calibrated occupancy result to the current immutable
    /// receipt. Returning `None` keeps callers fail-closed when the receipt,
    /// source node, timestamp, count, or method cannot be proven.
    pub fn calibrated_presence_evidence(
        &self,
        inference_node_id: u8,
        source_tick: u64,
        observed_at_unix_ms: u64,
        person_count: usize,
        inference_method: &'static str,
    ) -> Option<CalibratedPresenceEvidence> {
        self.calibrated_presence_evidence_with_residual(
            inference_node_id,
            source_tick,
            observed_at_unix_ms,
            person_count,
            inference_method,
            None,
        )
    }

    pub fn calibrated_presence_evidence_with_residual(
        &self,
        inference_node_id: u8,
        source_tick: u64,
        observed_at_unix_ms: u64,
        person_count: usize,
        inference_method: &'static str,
        residual_energy: Option<CalibratedResidualEnergyEvidence>,
    ) -> Option<CalibratedPresenceEvidence> {
        let receipt = self.model_receipt.as_ref()?;
        if receipt
            .source_node_ids
            .binary_search(&inference_node_id)
            .is_err()
            || observed_at_unix_ms < receipt.completed_at_unix_ms
            || person_count > 3
            || !matches!(
                inference_method,
                "field_model_eigenvalue_v1"
                    | "field_model_perturbation_energy_v1"
                    | "field_model_null_normalized_v2"
            )
        {
            return None;
        }
        Some(CalibratedPresenceEvidence {
            schema: CALIBRATED_PRESENCE_EVIDENCE_SCHEMA.to_string(),
            boot_epoch: receipt.boot_epoch.clone(),
            session_id: receipt.session_id.clone(),
            model_id: receipt.model_id.clone(),
            binding_digest: receipt.binding_digest.clone(),
            source_node_ids: receipt.source_node_ids.clone(),
            model_completed_at_unix_ms: receipt.completed_at_unix_ms,
            inference_node_id,
            source_tick,
            observed_at_unix_ms,
            inference_method: inference_method.to_string(),
            presence: person_count > 0,
            person_count,
            residual_energy,
        })
    }

    /// Legacy unbound startup calibration preserves its historical all-node
    /// behavior. Every API-started session admits only its snapshotted source
    /// node set, so unrelated room traffic cannot enter the fitted model.
    pub fn accepts_source_node(&self, node_id: u8) -> bool {
        self.session.as_ref().map_or(true, |session| {
            session.source_node_ids.binary_search(&node_id).is_ok()
        })
    }

    /// Record contribution only after the ingest path has accepted a real
    /// frame/history for this node. Returns `false` for a node outside the
    /// bound set so callers can skip numerical ingestion as well.
    pub fn record_source_node(&mut self, node_id: u8) -> bool {
        if !self.accepts_source_node(node_id) {
            return false;
        }
        if self.session.is_some() {
            match self.observed_source_node_ids.binary_search(&node_id) {
                Ok(_) => {}
                Err(index) => self.observed_source_node_ids.insert(index, node_id),
            }
        }
        true
    }

    pub fn observed_source_node_ids(&self) -> &[u8] {
        &self.observed_source_node_ids
    }

    pub fn missing_source_node_ids(&self) -> Vec<u8> {
        self.session.as_ref().map_or_else(Vec::new, |session| {
            session
                .source_node_ids
                .iter()
                .copied()
                .filter(|node_id| {
                    self.observed_source_node_ids
                        .binary_search(node_id)
                        .is_err()
                })
                .collect()
        })
    }

    pub fn begin(
        &mut self,
        binding: ValidatedCalibrationBinding,
    ) -> Result<CalibrationSessionIdentity, CalibrationSessionError> {
        if self.session.is_some() {
            return Err(CalibrationSessionError::SessionAlreadyActive);
        }
        let identity = CalibrationSessionIdentity {
            boot_epoch: self.boot_epoch.clone(),
            session_id: opaque_id("cal-session"),
            model_id: opaque_id("cal-model"),
            binding_digest: binding.binding_digest,
            source_node_ids: binding.source_node_ids,
        };
        self.model_receipt = None;
        self.observed_source_node_ids.clear();
        self.session = Some(identity.clone());
        Ok(identity)
    }

    pub fn validate_bound_request(
        &self,
        request: &CalibrationBoundRequest,
    ) -> Result<&CalibrationSessionIdentity, CalibrationSessionError> {
        validate_binding_digest(&request.binding_digest)?;
        validate_source_node_ids(&request.source_node_ids)?;
        if request.boot_epoch != self.boot_epoch {
            return Err(CalibrationSessionError::BootEpochMismatch);
        }
        let session = self
            .session
            .as_ref()
            .ok_or(CalibrationSessionError::NoBoundSession)?;
        if request.session_id != session.session_id {
            return Err(CalibrationSessionError::SessionMismatch);
        }
        if request.binding_digest != session.binding_digest {
            return Err(CalibrationSessionError::BindingMismatch);
        }
        if request.source_node_ids != session.source_node_ids {
            return Err(CalibrationSessionError::SourceNodeMismatch);
        }
        Ok(session)
    }

    pub fn finalize(
        &mut self,
        request: &CalibrationBoundRequest,
        frame_count: u64,
        variance_explained: f64,
        baseline_eigenvalue_count: usize,
        completed_at_unix_ms: u64,
    ) -> Result<CalibrationModelReceipt, CalibrationSessionError> {
        self.validate_bound_request(request)?;
        if let Some(receipt) = self.model_receipt.as_ref() {
            return Ok(receipt.clone());
        }
        if !self.missing_source_node_ids().is_empty() {
            return Err(CalibrationSessionError::UnobservedSourceNodes);
        }
        if frame_count == 0
            || !variance_explained.is_finite()
            || !(0.0..=1.001).contains(&variance_explained)
            || completed_at_unix_ms == 0
        {
            return Err(CalibrationSessionError::InvalidModelMetrics);
        }
        let session = self
            .session
            .as_ref()
            .ok_or(CalibrationSessionError::NoBoundSession)?;
        let receipt = CalibrationModelReceipt {
            schema: "ruview.calibration.field-model-receipt.v1",
            boot_epoch: session.boot_epoch.clone(),
            session_id: session.session_id.clone(),
            model_id: session.model_id.clone(),
            binding_digest: session.binding_digest.clone(),
            source_node_ids: session.source_node_ids.clone(),
            frame_count,
            variance_explained,
            baseline_eigenvalue_count,
            completed_at_unix_ms,
        };
        self.model_receipt = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn reset(
        &mut self,
        request: &CalibrationResetRequest,
    ) -> Result<Option<CalibrationSessionIdentity>, CalibrationSessionError> {
        if request.boot_epoch != self.boot_epoch {
            return Err(CalibrationSessionError::BootEpochMismatch);
        }
        match self.session.as_ref() {
            Some(session) => {
                let session_id = request
                    .session_id
                    .as_deref()
                    .ok_or(CalibrationSessionError::SessionMismatch)?;
                let binding_digest = request
                    .binding_digest
                    .as_deref()
                    .ok_or(CalibrationSessionError::BindingMismatch)?;
                let source_node_ids = request
                    .source_node_ids
                    .as_deref()
                    .ok_or(CalibrationSessionError::SourceNodeMismatch)?;
                validate_binding_digest(binding_digest)?;
                validate_source_node_ids(source_node_ids)?;
                if session_id != session.session_id {
                    return Err(CalibrationSessionError::SessionMismatch);
                }
                if binding_digest != session.binding_digest {
                    return Err(CalibrationSessionError::BindingMismatch);
                }
                if source_node_ids != session.source_node_ids {
                    return Err(CalibrationSessionError::SourceNodeMismatch);
                }
            }
            None => {
                if request.session_id.is_some()
                    || request.binding_digest.is_some()
                    || request.source_node_ids.is_some()
                {
                    return Err(CalibrationSessionError::NoBoundSession);
                }
            }
        }
        let previous = self.session.take();
        self.model_receipt = None;
        self.observed_source_node_ids.clear();
        Ok(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn contract() -> CalibrationSessionContract {
        CalibrationSessionContract {
            boot_epoch: "cal-boot-test".to_string(),
            session: None,
            model_receipt: None,
            observed_source_node_ids: Vec::new(),
        }
    }

    fn start(digest: &str, nodes: Vec<u8>) -> CalibrationStartRequest {
        CalibrationStartRequest {
            binding_digest: digest.to_string(),
            source_node_ids: nodes,
        }
    }

    fn bound(identity: &CalibrationSessionIdentity) -> CalibrationBoundRequest {
        CalibrationBoundRequest {
            boot_epoch: identity.boot_epoch.clone(),
            session_id: identity.session_id.clone(),
            binding_digest: identity.binding_digest.clone(),
            source_node_ids: identity.source_node_ids.clone(),
        }
    }

    #[test]
    fn start_binding_requires_lowercase_sha256_and_sorted_unique_nodes() {
        assert_eq!(
            validate_start_request(start("ABC", vec![1])).unwrap_err(),
            CalibrationSessionError::InvalidBindingDigest
        );
        assert_eq!(
            validate_start_request(start(DIGEST_A, vec![])).unwrap_err(),
            CalibrationSessionError::InvalidSourceNodeIds
        );
        assert_eq!(
            validate_start_request(start(DIGEST_A, vec![7, 1])).unwrap_err(),
            CalibrationSessionError::InvalidSourceNodeIds
        );
        assert_eq!(
            validate_start_request(start(DIGEST_A, vec![1, 1])).unwrap_err(),
            CalibrationSessionError::InvalidSourceNodeIds
        );
        assert!(validate_start_request(start(DIGEST_A, vec![1, 7])).is_ok());
    }

    #[test]
    fn session_snapshots_boot_binding_and_nodes() {
        let mut contract = contract();
        let binding = validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap();
        let identity = contract.begin(binding).unwrap();
        assert_eq!(identity.boot_epoch, "cal-boot-test");
        assert_eq!(identity.binding_digest, DIGEST_A);
        assert_eq!(identity.source_node_ids, vec![1, 7]);
        assert!(identity.session_id.starts_with("cal-session-"));
        assert!(identity.model_id.starts_with("cal-model-"));
        assert_eq!(contract.session(), Some(&identity));
        assert!(contract.accepts_source_node(1));
        assert!(contract.accepts_source_node(7));
        assert!(!contract.accepts_source_node(2));
        assert_eq!(contract.missing_source_node_ids(), vec![1, 7]);
        assert!(contract.record_source_node(7));
        assert!(!contract.record_source_node(2));
        assert_eq!(contract.observed_source_node_ids(), &[7]);
        assert_eq!(contract.missing_source_node_ids(), vec![1]);
    }

    #[test]
    fn bound_operations_reject_every_identity_mismatch() {
        let mut contract = contract();
        let identity = contract
            .begin(validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap())
            .unwrap();
        let valid = bound(&identity);
        assert!(contract.validate_bound_request(&valid).is_ok());

        let mut wrong = valid.clone();
        wrong.boot_epoch = "cal-boot-old".to_string();
        assert_eq!(
            contract.validate_bound_request(&wrong).unwrap_err(),
            CalibrationSessionError::BootEpochMismatch
        );
        let mut wrong = valid.clone();
        wrong.session_id = "cal-session-other".to_string();
        assert_eq!(
            contract.validate_bound_request(&wrong).unwrap_err(),
            CalibrationSessionError::SessionMismatch
        );
        let mut wrong = valid.clone();
        wrong.binding_digest = DIGEST_B.to_string();
        assert_eq!(
            contract.validate_bound_request(&wrong).unwrap_err(),
            CalibrationSessionError::BindingMismatch
        );
        let mut wrong = valid;
        wrong.source_node_ids = vec![1, 8];
        assert_eq!(
            contract.validate_bound_request(&wrong).unwrap_err(),
            CalibrationSessionError::SourceNodeMismatch
        );
    }

    #[test]
    fn finalized_receipt_is_immutable_and_idempotent() {
        let mut contract = contract();
        let identity = contract
            .begin(validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap())
            .unwrap();
        let request = bound(&identity);
        assert!(contract.record_source_node(1));
        assert!(contract.record_source_node(7));
        let first = contract
            .finalize(&request, 12_000, 0.42, 2, 123_456)
            .unwrap();
        let second = contract
            .finalize(&request, 99_999, 0.99, 9, 999_999)
            .unwrap();
        assert_eq!(first, second);
        assert!(first.model_id.starts_with("cal-model-"));
        assert_eq!(first.model_id, identity.model_id);
        assert_eq!(contract.model_receipt(), Some(&first));
    }

    #[test]
    fn calibrated_presence_evidence_is_exactly_receipt_and_frame_bound() {
        let mut contract = contract();
        let identity = contract
            .begin(validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap())
            .unwrap();
        assert!(contract.record_source_node(1));
        assert!(contract.record_source_node(7));
        let receipt = contract
            .finalize(&bound(&identity), 12_000, 0.42, 2, 123_456)
            .unwrap();

        let evidence = contract
            .calibrated_presence_evidence(7, 88, 123_999, 1, "field_model_perturbation_energy_v1")
            .expect("bound evidence");
        assert_eq!(evidence.schema, CALIBRATED_PRESENCE_EVIDENCE_SCHEMA);
        assert_eq!(evidence.boot_epoch, receipt.boot_epoch);
        assert_eq!(evidence.session_id, receipt.session_id);
        assert_eq!(evidence.model_id, receipt.model_id);
        assert_eq!(evidence.binding_digest, receipt.binding_digest);
        assert_eq!(evidence.source_node_ids, vec![1, 7]);
        assert_eq!(evidence.inference_node_id, 7);
        assert_eq!(evidence.source_tick, 88);
        assert_eq!(evidence.observed_at_unix_ms, 123_999);
        assert!(evidence.presence);
        assert_eq!(evidence.person_count, 1);
        assert!(evidence.residual_energy.is_none());

        let residual = CalibratedResidualEnergyEvidence {
            nodes: vec![CalibratedNodeResidualEnergy {
                node_id: 7,
                energy: 2.0,
                null_median: 0.5,
                null_p95: 0.8,
                null_p99: 1.0,
                normalized_energy: 2.0,
                decision_threshold: 1.0,
                above_threshold: true,
            }],
            aggregate_mean_energy: 2.0,
            aggregate_mean_normalized_energy: 2.0,
            nodes_above_threshold: 1,
            node_quorum: 1,
            hysteresis_present: true,
            hysteresis_candidate_frames: 0,
        };
        let diagnostic = contract
            .calibrated_presence_evidence_with_residual(
                7,
                89,
                124_000,
                1,
                "field_model_null_normalized_v2",
                Some(residual.clone()),
            )
            .expect("null-normalized diagnostic evidence");
        assert_eq!(diagnostic.residual_energy, Some(residual));

        assert!(contract
            .calibrated_presence_evidence(2, 89, 124_000, 0, "field_model_perturbation_energy_v1",)
            .is_none());
        assert!(contract
            .calibrated_presence_evidence(7, 89, 123_000, 0, "field_model_perturbation_energy_v1",)
            .is_none());
        assert!(contract
            .calibrated_presence_evidence(7, 89, 124_000, 4, "field_model_eigenvalue_v1")
            .is_none());
        assert!(contract
            .calibrated_presence_evidence(7, 89, 124_000, 0, "heuristic")
            .is_none());
    }

    #[test]
    fn finalize_rejects_a_bound_node_that_never_contributed() {
        let mut contract = contract();
        let identity = contract
            .begin(validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap())
            .unwrap();
        assert!(contract.record_source_node(1));
        assert_eq!(
            contract
                .finalize(&bound(&identity), 12_000, 0.42, 2, 123_456)
                .unwrap_err(),
            CalibrationSessionError::UnobservedSourceNodes
        );
        assert_eq!(contract.missing_source_node_ids(), vec![7]);
    }

    #[test]
    fn reset_requires_the_complete_current_binding() {
        let mut contract = contract();
        let identity = contract
            .begin(validate_start_request(start(DIGEST_A, vec![1, 7])).unwrap())
            .unwrap();
        let partial = CalibrationResetRequest {
            boot_epoch: identity.boot_epoch.clone(),
            session_id: Some(identity.session_id.clone()),
            binding_digest: None,
            source_node_ids: Some(identity.source_node_ids.clone()),
        };
        assert_eq!(
            contract.reset(&partial).unwrap_err(),
            CalibrationSessionError::BindingMismatch
        );
        let request = CalibrationResetRequest {
            boot_epoch: identity.boot_epoch.clone(),
            session_id: Some(identity.session_id.clone()),
            binding_digest: Some(identity.binding_digest.clone()),
            source_node_ids: Some(identity.source_node_ids.clone()),
        };
        assert_eq!(contract.reset(&request).unwrap(), Some(identity));
        assert!(contract.session().is_none());
        assert!(contract.model_receipt().is_none());
    }

    #[test]
    fn legacy_unbound_reset_requires_boot_epoch_and_null_identity() {
        let mut contract = contract();
        let request = CalibrationResetRequest {
            boot_epoch: "cal-boot-test".to_string(),
            session_id: None,
            binding_digest: None,
            source_node_ids: None,
        };
        assert_eq!(contract.reset(&request).unwrap(), None);
    }

    #[test]
    fn request_deserialization_rejects_unknown_fields() {
        let value = serde_json::json!({
            "binding_digest": DIGEST_A,
            "source_node_ids": [1],
            "room_name": "must not cross the boundary"
        });
        assert!(serde_json::from_value::<CalibrationStartRequest>(value).is_err());
    }
}
