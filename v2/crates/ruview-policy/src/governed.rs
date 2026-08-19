//! Governed action intents and witnessed authorization receipts (ADR-327).
//!
//! This module never touches an actuator. Its strongest outcome is an
//! `Authorized` receipt that a separate, explicitly configured adapter may
//! consume. Observe and recommend are the default modes; execute fails closed
//! unless a registered policy, live assurance, and signed approvals all pass.

use crate::{authorize, ActionClass, AssuranceInputs, Authorization, FailedCondition};
use ruview_attest::{Signature, Signer, Verifier};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const INTENT_DOMAIN: &[u8] = b"ruview.governed-intent.v1\0";
const APPROVAL_DOMAIN: &[u8] = b"ruview.governed-approval.v1\0";
const RECEIPT_DOMAIN: &[u8] = b"ruview.governed-receipt.v1\0";
const MAX_ID_BYTES: usize = 128;
const MAX_APPROVALS: usize = 16;
const MAX_TARGET_PREFIXES: usize = 32;
const MAX_INTENT_LIFETIME_MS: i64 = 86_400_000;
const MAX_RECEIPTS: usize = 10_000;

/// Requested governance mode. Automation should default to `Recommend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IntentMode {
    /// Record a governed observation without proposing a consequence.
    Observe,
    /// Produce a recommendation for human/policy review.
    Recommend,
    /// Request an authorization receipt for a separately configured adapter.
    Execute,
}

impl Default for IntentMode {
    fn default() -> Self {
        Self::Recommend
    }
}

/// A typed, bounded request. Parameters are represented only by a digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionIntent {
    /// Idempotency key for this exact attempt.
    pub intent_id: String,
    /// Authenticated tenant identifier.
    pub tenant_id: String,
    /// Authenticated workspace identifier.
    pub workspace_id: String,
    /// Registered action kind, such as `alert.raise`.
    pub action_kind: String,
    /// Exact registered policy version requested by this intent.
    pub policy_version: String,
    /// Bounded target identifier.
    pub target_id: String,
    /// Consequence/assurance class.
    pub class: ActionClass,
    /// Observe, recommend, or explicitly request authorization.
    #[serde(default)]
    pub mode: IntentMode,
    /// Authenticated requesting principal or agent.
    pub requested_by: String,
    /// Intent creation time in Unix milliseconds.
    pub issued_at_ms: i64,
    /// Hard expiry in Unix milliseconds.
    pub expires_at_ms: i64,
    /// Caller-generated replay nonce. All zeroes are invalid.
    pub nonce: [u8; 16],
    /// Digest of canonical adapter parameters; raw values are not logged here.
    pub parameters_digest: [u8; 32],
    /// Digest of the governed perception/evidence input.
    pub evidence_digest: [u8; 32],
}

impl ActionIntent {
    /// Deterministic bytes bound into approvals and receipts.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(INTENT_DOMAIN);
        for value in [
            self.intent_id.as_str(),
            self.tenant_id.as_str(),
            self.workspace_id.as_str(),
            self.action_kind.as_str(),
            self.policy_version.as_str(),
            self.target_id.as_str(),
            self.requested_by.as_str(),
        ] {
            push_field(&mut out, value.as_bytes());
        }
        out.push(self.class as u8);
        out.push(self.mode as u8);
        out.extend_from_slice(&self.issued_at_ms.to_le_bytes());
        out.extend_from_slice(&self.expires_at_ms.to_le_bytes());
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.parameters_digest);
        out.extend_from_slice(&self.evidence_digest);
        out
    }

    /// Digest used as the immutable idempotency fingerprint.
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }
}

/// A versioned, locally registered execution rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRule {
    /// Exact action kind this rule governs.
    pub action_kind: String,
    /// Monotonic/configuration version included in approvals and receipts.
    pub policy_version: String,
    /// Required assurance class. Intent class must match exactly.
    pub class: ActionClass,
    /// Number of distinct valid human/service approvals (at least one).
    pub minimum_approvals: usize,
    /// Trusted execution grant required in addition to perception assurance.
    pub required_grant: String,
    /// At least one prefix must match the target identifier.
    pub target_prefixes: Vec<String>,
}

/// Local allow-list of action rules. Absence is a deny.
#[derive(Clone, Debug, Default)]
pub struct PolicyRegistry {
    rules: BTreeMap<String, ActionRule>,
}

impl PolicyRegistry {
    /// Register one valid rule; duplicate action kinds are refused.
    pub fn register(&mut self, rule: ActionRule) -> Result<(), GovernanceError> {
        validate_id(&rule.action_kind)?;
        validate_id(&rule.policy_version)?;
        validate_id(&rule.required_grant)?;
        if rule.minimum_approvals == 0 || rule.minimum_approvals > MAX_APPROVALS {
            return Err(GovernanceError::InvalidInput(
                "approval threshold is out of bounds",
            ));
        }
        if rule.target_prefixes.is_empty() || rule.target_prefixes.len() > MAX_TARGET_PREFIXES {
            return Err(GovernanceError::InvalidInput(
                "target prefix list is out of bounds",
            ));
        }
        for prefix in &rule.target_prefixes {
            validate_id(prefix)?;
        }
        if self.rules.contains_key(&rule.action_kind) {
            return Err(GovernanceError::PolicyConflict);
        }
        self.rules.insert(rule.action_kind.clone(), rule);
        Ok(())
    }

    fn get(&self, action_kind: &str) -> Option<&ActionRule> {
        self.rules.get(action_kind)
    }
}

/// Trusted grants established by the host authorization adapter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthorityContext {
    grants: BTreeSet<String>,
}

impl AuthorityContext {
    /// Build a bounded set of authenticated grants. Strings are exact-match.
    pub fn from_authenticated_grants<I, S>(grants: I) -> Result<Self, GovernanceError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut values = BTreeSet::new();
        for (index, grant) in grants.into_iter().enumerate() {
            if index >= MAX_APPROVALS {
                return Err(GovernanceError::InvalidInput(
                    "authority grant set is out of bounds",
                ));
            }
            let grant = grant.into();
            validate_id(&grant)?;
            values.insert(grant);
        }
        Ok(Self { grants: values })
    }

    fn contains(&self, grant: &str) -> bool {
        self.grants.contains(grant)
    }
}

/// Human or service approval decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalDecision {
    /// Explicit approval.
    Approve,
    /// Explicit rejection; any valid rejection denies this attempt.
    Reject,
}

/// Content signed by one registered approver.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalContent {
    /// Intent digest prevents approval substitution.
    pub intent_digest: [u8; 32],
    /// Exact policy version reviewed by the approver.
    pub policy_version: String,
    /// Registered approver identity.
    pub approver_id: String,
    /// Explicit approve/reject decision.
    pub decision: ApprovalDecision,
    /// Approval timestamp in Unix milliseconds.
    pub approved_at_ms: i64,
}

impl ApprovalContent {
    /// Deterministic signing bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(APPROVAL_DOMAIN);
        out.extend_from_slice(&self.intent_digest);
        push_field(&mut out, self.policy_version.as_bytes());
        push_field(&mut out, self.approver_id.as_bytes());
        out.push(self.decision as u8);
        out.extend_from_slice(&self.approved_at_ms.to_le_bytes());
        out
    }
}

/// Signed approval envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedApproval {
    /// Signed approval content.
    pub content: ApprovalContent,
    /// Attestation signature/MAC.
    pub signature: Signature,
}

impl SignedApproval {
    /// Sign approval content with an enrolled signer.
    pub fn sign<S: Signer + ?Sized>(content: ApprovalContent, signer: &S) -> Self {
        let signature = signer.sign(&content.canonical_bytes());
        Self { content, signature }
    }
}

/// Resolves approver identities to enrolled verification keys.
pub trait ApprovalVerifier {
    /// Return true only for a registered identity and valid signature.
    fn verify(&self, approver_id: &str, message: &[u8], signature: &Signature) -> bool;
}

/// Stable terminal reason for a non-authorized receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenialReason {
    /// Intent was expired or not yet valid.
    IntentExpired,
    /// Action kind has no registered policy.
    NoPolicy,
    /// Intent class does not match the registered policy.
    ClassMismatch,
    /// Intent names a policy version other than the registered version.
    PolicyVersionMismatch,
    /// Trusted host authority lacks the exact policy grant.
    MissingAuthority,
    /// Target is outside the registered allow-list.
    TargetNotAllowed,
    /// Too few distinct, valid, explicit approvals.
    InsufficientApprovals,
    /// An approval was malformed, rejected, duplicated, or unauthenticated.
    InvalidApproval,
    /// Existing assurance policy denied the requested class.
    AssuranceDenied(FailedCondition),
}

/// Terminal governance decision. `Authorized` is not proof of actuation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GovernedDecision {
    /// Observation was witnessed only.
    Observed,
    /// Recommendation was witnessed and awaits a new execute intent.
    Recommended,
    /// A separate configured adapter may execute this exact intent.
    Authorized,
    /// Authorization failed closed.
    Denied(DenialReason),
}

/// Signed, hash-chained receipt content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiptContent {
    /// Monotonic sequence within this engine instance.
    pub sequence: u64,
    /// Engine/service identity issuing the receipt.
    pub issuer_id: String,
    /// Exact intent digest.
    pub intent_digest: [u8; 32],
    /// Intent idempotency key for lookup.
    pub intent_id: String,
    /// Authenticated tenant/workspace copied from the intent.
    pub tenant_id: String,
    /// Authenticated tenant/workspace copied from the intent.
    pub workspace_id: String,
    /// Registered policy version, if a policy was found.
    pub policy_version: Option<String>,
    /// Terminal governance decision.
    pub decision: GovernedDecision,
    /// Number of distinct verified approvals used.
    pub verified_approvals: usize,
    /// Decision timestamp supplied by the caller.
    pub decided_at_ms: i64,
    /// Intent expiry copied into the receipt for adapter-side checks.
    pub expires_at_ms: i64,
    /// Non-secret replay nonce copied into the signed receipt.
    pub nonce: [u8; 16],
    /// Previous receipt digest; zeroes start a chain.
    pub previous_receipt_digest: [u8; 32],
}

impl ReceiptContent {
    /// Deterministic bytes for signing and chain hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(RECEIPT_DOMAIN);
        out.extend_from_slice(&self.sequence.to_le_bytes());
        for value in [
            self.issuer_id.as_str(),
            self.intent_id.as_str(),
            self.tenant_id.as_str(),
            self.workspace_id.as_str(),
        ] {
            push_field(&mut out, value.as_bytes());
        }
        out.extend_from_slice(&self.intent_digest);
        match &self.policy_version {
            Some(version) => {
                out.push(1);
                push_field(&mut out, version.as_bytes());
            }
            None => out.push(0),
        }
        push_decision(&mut out, &self.decision);
        out.extend_from_slice(&(self.verified_approvals as u64).to_le_bytes());
        out.extend_from_slice(&self.decided_at_ms.to_le_bytes());
        out.extend_from_slice(&self.expires_at_ms.to_le_bytes());
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.previous_receipt_digest);
        out
    }
}

/// Signed receipt. It authorizes at most; it never asserts physical execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionReceipt {
    /// Signed content.
    pub content: ReceiptContent,
    /// Signature over canonical content bytes.
    pub signature: Signature,
}

impl ActionReceipt {
    /// Verify the issuer signature.
    pub fn verify<V: Verifier + ?Sized>(&self, verifier: &V) -> bool {
        verifier.verify(&self.content.canonical_bytes(), &self.signature)
    }

    /// Digest used by the next receipt's chain link.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.content.canonical_bytes());
        hasher.update(&self.signature.0);
        *hasher.finalize().as_bytes()
    }
}

#[derive(Clone, Debug)]
struct StoredReceipt {
    intent_digest: [u8; 32],
    receipt: ActionReceipt,
}

/// Stateful governance boundary providing idempotency and receipt chaining.
#[derive(Clone, Debug)]
pub struct GovernanceEngine {
    issuer_id: String,
    policies: PolicyRegistry,
    receipts: BTreeMap<String, StoredReceipt>,
    nonces: BTreeMap<(String, String, [u8; 16]), [u8; 32]>,
    next_sequence: u64,
    previous_receipt_digest: [u8; 32],
}

impl GovernanceEngine {
    /// Create an engine with an explicit local policy registry.
    pub fn new(issuer_id: String, policies: PolicyRegistry) -> Result<Self, GovernanceError> {
        validate_id(&issuer_id)?;
        Ok(Self {
            issuer_id,
            policies,
            receipts: BTreeMap::new(),
            nonces: BTreeMap::new(),
            next_sequence: 1,
            previous_receipt_digest: [0; 32],
        })
    }

    /// Evaluate and witness one intent. A repeated identical intent returns the
    /// exact prior receipt; changed reuse of its idempotency key is rejected.
    pub fn evaluate<S: Signer + ?Sized, V: ApprovalVerifier + ?Sized>(
        &mut self,
        intent: &ActionIntent,
        authority: &AuthorityContext,
        assurance: &AssuranceInputs,
        approvals: &[SignedApproval],
        approval_verifier: &V,
        receipt_signer: &S,
        now_ms: i64,
    ) -> Result<ActionReceipt, GovernanceError> {
        validate_intent(intent)?;
        let intent_digest = intent.digest();
        if let Some(stored) = self.receipts.get(&intent.intent_id) {
            return if stored.intent_digest == intent_digest {
                Ok(stored.receipt.clone())
            } else {
                Err(GovernanceError::IdempotencyConflict)
            };
        }
        if self.receipts.len() >= MAX_RECEIPTS {
            return Err(GovernanceError::CapacityReached);
        }
        let nonce_key = (
            intent.tenant_id.clone(),
            intent.workspace_id.clone(),
            intent.nonce,
        );
        if self.nonces.contains_key(&nonce_key) {
            return Err(GovernanceError::NonceReplay);
        }

        let rule = self.policies.get(&intent.action_kind);
        let (decision, verified_approvals) =
            if now_ms < intent.issued_at_ms || now_ms >= intent.expires_at_ms {
                (GovernedDecision::Denied(DenialReason::IntentExpired), 0)
            } else {
                match intent.mode {
                    IntentMode::Observe => (GovernedDecision::Observed, 0),
                    IntentMode::Recommend => (GovernedDecision::Recommended, 0),
                    IntentMode::Execute => evaluate_execution(
                        intent,
                        intent_digest,
                        authority,
                        assurance,
                        approvals,
                        approval_verifier,
                        rule,
                        now_ms,
                    ),
                }
            };
        let policy_version = rule.map(|value| value.policy_version.clone());
        let content = ReceiptContent {
            sequence: self.next_sequence,
            issuer_id: self.issuer_id.clone(),
            intent_digest,
            intent_id: intent.intent_id.clone(),
            tenant_id: intent.tenant_id.clone(),
            workspace_id: intent.workspace_id.clone(),
            policy_version,
            decision,
            verified_approvals,
            decided_at_ms: now_ms,
            expires_at_ms: intent.expires_at_ms,
            nonce: intent.nonce,
            previous_receipt_digest: self.previous_receipt_digest,
        };
        let receipt = ActionReceipt {
            signature: receipt_signer.sign(&content.canonical_bytes()),
            content,
        };
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(GovernanceError::SequenceExhausted)?;
        self.previous_receipt_digest = receipt.digest();
        self.receipts.insert(
            intent.intent_id.clone(),
            StoredReceipt {
                intent_digest,
                receipt: receipt.clone(),
            },
        );
        self.nonces.insert(nonce_key, intent_digest);
        Ok(receipt)
    }
}

fn evaluate_execution<V: ApprovalVerifier + ?Sized>(
    intent: &ActionIntent,
    intent_digest: [u8; 32],
    authority: &AuthorityContext,
    assurance: &AssuranceInputs,
    approvals: &[SignedApproval],
    verifier: &V,
    rule: Option<&ActionRule>,
    now_ms: i64,
) -> (GovernedDecision, usize) {
    if now_ms < intent.issued_at_ms || now_ms >= intent.expires_at_ms {
        return (GovernedDecision::Denied(DenialReason::IntentExpired), 0);
    }
    let Some(rule) = rule else {
        return (GovernedDecision::Denied(DenialReason::NoPolicy), 0);
    };
    if intent.class != rule.class {
        return (GovernedDecision::Denied(DenialReason::ClassMismatch), 0);
    }
    if intent.policy_version != rule.policy_version {
        return (
            GovernedDecision::Denied(DenialReason::PolicyVersionMismatch),
            0,
        );
    }
    if !authority.contains(&rule.required_grant) {
        return (GovernedDecision::Denied(DenialReason::MissingAuthority), 0);
    }
    if !rule
        .target_prefixes
        .iter()
        .any(|prefix| intent.target_id.starts_with(prefix))
    {
        return (GovernedDecision::Denied(DenialReason::TargetNotAllowed), 0);
    }
    if approvals.len() > MAX_APPROVALS {
        return (GovernedDecision::Denied(DenialReason::InvalidApproval), 0);
    }
    let mut distinct = BTreeSet::new();
    for approval in approvals {
        let content = &approval.content;
        if validate_id(&content.approver_id).is_err()
            || content.intent_digest != intent_digest
            || content.policy_version != rule.policy_version
            || content.approved_at_ms < intent.issued_at_ms
            || content.approved_at_ms >= intent.expires_at_ms
            || content.approved_at_ms > now_ms
            || content.decision != ApprovalDecision::Approve
            || !distinct.insert(content.approver_id.as_str())
            || !verifier.verify(
                &content.approver_id,
                &content.canonical_bytes(),
                &approval.signature,
            )
        {
            return (
                GovernedDecision::Denied(DenialReason::InvalidApproval),
                distinct.len(),
            );
        }
    }
    if distinct.len() < rule.minimum_approvals {
        return (
            GovernedDecision::Denied(DenialReason::InsufficientApprovals),
            distinct.len(),
        );
    }
    match authorize(intent.class, assurance) {
        Authorization::Allow { .. } => (GovernedDecision::Authorized, distinct.len()),
        Authorization::Deny { failed_condition } => (
            GovernedDecision::Denied(DenialReason::AssuranceDenied(failed_condition)),
            distinct.len(),
        ),
    }
}

/// Engine/configuration errors. Policy denials are signed receipts, not errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GovernanceError {
    /// Malformed caller/configuration input.
    #[error("invalid governed-action input: {0}")]
    InvalidInput(&'static str),
    /// Duplicate action rule.
    #[error("action policy already registered")]
    PolicyConflict,
    /// An intent idempotency key was reused with different content.
    #[error("intent idempotency conflict")]
    IdempotencyConflict,
    /// A nonce was already bound to a different intent id.
    #[error("intent nonce replay")]
    NonceReplay,
    /// The bounded in-memory replay store reached capacity.
    #[error("governance receipt capacity reached")]
    CapacityReached,
    /// Receipt sequence exhausted.
    #[error("receipt sequence exhausted")]
    SequenceExhausted,
}

fn validate_intent(intent: &ActionIntent) -> Result<(), GovernanceError> {
    for value in [
        intent.intent_id.as_str(),
        intent.tenant_id.as_str(),
        intent.workspace_id.as_str(),
        intent.action_kind.as_str(),
        intent.policy_version.as_str(),
        intent.target_id.as_str(),
        intent.requested_by.as_str(),
    ] {
        validate_id(value)?;
    }
    if intent.issued_at_ms >= intent.expires_at_ms
        || intent.expires_at_ms - intent.issued_at_ms > MAX_INTENT_LIFETIME_MS
        || intent.nonce == [0; 16]
    {
        return Err(GovernanceError::InvalidInput("intent lifetime is invalid"));
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), GovernanceError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(GovernanceError::InvalidInput("identifier is invalid"));
    }
    Ok(())
}

fn push_field(output: &mut Vec<u8>, field: &[u8]) {
    output.extend_from_slice(&(field.len() as u32).to_le_bytes());
    output.extend_from_slice(field);
}

fn push_decision(output: &mut Vec<u8>, decision: &GovernedDecision) {
    match decision {
        GovernedDecision::Observed => output.push(0),
        GovernedDecision::Recommended => output.push(1),
        GovernedDecision::Authorized => output.push(2),
        GovernedDecision::Denied(reason) => {
            output.push(3);
            push_denial(output, reason);
        }
    }
}

fn push_denial(output: &mut Vec<u8>, reason: &DenialReason) {
    match reason {
        DenialReason::IntentExpired => output.push(0),
        DenialReason::NoPolicy => output.push(1),
        DenialReason::ClassMismatch => output.push(2),
        DenialReason::PolicyVersionMismatch => output.push(3),
        DenialReason::MissingAuthority => output.push(4),
        DenialReason::TargetNotAllowed => output.push(5),
        DenialReason::InsufficientApprovals => output.push(6),
        DenialReason::InvalidApproval => output.push(7),
        DenialReason::AssuranceDenied(condition) => {
            output.push(8);
            match condition {
                FailedCondition::NoPolicy => output.push(0),
                FailedCondition::CertificateInvalid => output.push(1),
                FailedCondition::CertificateClassTooLow { required, actual } => {
                    output.extend_from_slice(&[2, *required as u8, *actual as u8]);
                }
                FailedCondition::CertificateStale { age_secs, max_secs } => {
                    output.push(3);
                    output.extend_from_slice(&age_secs.to_le_bytes());
                    output.extend_from_slice(&max_secs.to_le_bytes());
                }
                FailedCondition::DomainDegraded => output.push(4),
                FailedCondition::DomainNotKnown => output.push(5),
                FailedCondition::UncertaintyOverCeiling { max_uncertainty } => {
                    output.push(6);
                    output.extend_from_slice(&max_uncertainty.to_bits().to_le_bytes());
                }
                FailedCondition::EvidenceBelowFloor { required, actual } => {
                    output.extend_from_slice(&[7, *required as u8, *actual as u8]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CertificateClass, DomainState};
    use ruview_attest::Blake3MacSigner;
    use ruview_evidence::EvidenceLevel;

    const NOW: i64 = 10_000;

    struct Approvers(BTreeMap<String, Blake3MacSigner>);

    impl ApprovalVerifier for Approvers {
        fn verify(&self, approver_id: &str, message: &[u8], signature: &Signature) -> bool {
            self.0
                .get(approver_id)
                .is_some_and(|key| Verifier::verify(key, message, signature))
        }
    }

    fn registry() -> PolicyRegistry {
        let mut registry = PolicyRegistry::default();
        registry
            .register(ActionRule {
                action_kind: "alert.raise".into(),
                policy_version: "v1".into(),
                class: ActionClass::Security,
                minimum_approvals: 1,
                required_grant: "alerts:execute".into(),
                target_prefixes: vec!["alert:".into()],
            })
            .unwrap();
        registry
    }

    fn intent(mode: IntentMode) -> ActionIntent {
        ActionIntent {
            intent_id: "intent-1".into(),
            tenant_id: "tenant-1".into(),
            workspace_id: "workspace-1".into(),
            action_kind: "alert.raise".into(),
            policy_version: "v1".into(),
            target_id: "alert:room-1".into(),
            class: ActionClass::Security,
            mode,
            requested_by: "agent-1".into(),
            issued_at_ms: NOW - 100,
            expires_at_ms: NOW + 100,
            nonce: [1; 16],
            parameters_digest: [1; 32],
            evidence_digest: [2; 32],
        }
    }

    fn assurance() -> AssuranceInputs {
        AssuranceInputs {
            certificate_class: CertificateClass::Standard,
            certificate_valid: true,
            certificate_age_secs: 1,
            domain_state: DomainState::Known,
            uncertainty: 0.1,
            evidence_level: EvidenceLevel::L2,
        }
    }

    fn approvers() -> Approvers {
        Approvers(BTreeMap::from([(
            "human-1".into(),
            Blake3MacSigner::new([3; 32]),
        )]))
    }

    fn authority() -> AuthorityContext {
        AuthorityContext::from_authenticated_grants(["alerts:execute"]).unwrap()
    }

    fn approval(intent: &ActionIntent) -> SignedApproval {
        SignedApproval::sign(
            ApprovalContent {
                intent_digest: intent.digest(),
                policy_version: "v1".into(),
                approver_id: "human-1".into(),
                decision: ApprovalDecision::Approve,
                approved_at_ms: NOW - 1,
            },
            &Blake3MacSigner::new([3; 32]),
        )
    }

    #[test]
    fn observe_and_recommend_are_non_executing_defaults() {
        let signer = Blake3MacSigner::new([9; 32]);
        for (mode, expected) in [
            (IntentMode::Observe, GovernedDecision::Observed),
            (IntentMode::Recommend, GovernedDecision::Recommended),
        ] {
            let mut engine =
                GovernanceEngine::new("issuer".into(), PolicyRegistry::default()).unwrap();
            let receipt = engine
                .evaluate(
                    &intent(mode),
                    &authority(),
                    &assurance(),
                    &[],
                    &approvers(),
                    &signer,
                    NOW,
                )
                .unwrap();
            assert_eq!(receipt.content.decision, expected);
            assert!(receipt.verify(&signer));
        }
    }

    #[test]
    fn expired_observation_and_recommendation_intents_are_denied() {
        let signer = Blake3MacSigner::new([9; 32]);
        for mode in [IntentMode::Observe, IntentMode::Recommend] {
            let mut request = intent(mode);
            request.issued_at_ms = NOW - 200;
            request.expires_at_ms = NOW - 1;
            let mut engine =
                GovernanceEngine::new("issuer".into(), PolicyRegistry::default()).unwrap();
            let receipt = engine
                .evaluate(
                    &request,
                    &authority(),
                    &assurance(),
                    &[],
                    &approvers(),
                    &signer,
                    NOW,
                )
                .unwrap();
            assert_eq!(
                receipt.content.decision,
                GovernedDecision::Denied(DenialReason::IntentExpired)
            );
            assert!(receipt.verify(&signer));
        }
    }

    #[test]
    fn execute_requires_policy_signed_approval_and_assurance() {
        let receipt_signer = Blake3MacSigner::new([9; 32]);
        let request = intent(IntentMode::Execute);
        let mut engine = GovernanceEngine::new("issuer".into(), registry()).unwrap();
        let denied = engine
            .evaluate(
                &request,
                &authority(),
                &assurance(),
                &[],
                &approvers(),
                &receipt_signer,
                NOW,
            )
            .unwrap();
        assert_eq!(
            denied.content.decision,
            GovernedDecision::Denied(DenialReason::InsufficientApprovals)
        );

        let mut second = request.clone();
        second.intent_id = "intent-2".into();
        second.nonce = [2; 16];
        let authorized = engine
            .evaluate(
                &second,
                &authority(),
                &assurance(),
                &[approval(&second)],
                &approvers(),
                &receipt_signer,
                NOW,
            )
            .unwrap();
        assert_eq!(authorized.content.decision, GovernedDecision::Authorized);
        assert_eq!(authorized.content.previous_receipt_digest, denied.digest());
        assert!(authorized.verify(&receipt_signer));
    }

    #[test]
    fn invalid_approval_and_unknown_domain_fail_closed() {
        let signer = Blake3MacSigner::new([9; 32]);
        let request = intent(IntentMode::Execute);
        let mut bad = approval(&request);
        bad.signature.0[0] ^= 1;
        let mut engine = GovernanceEngine::new("issuer".into(), registry()).unwrap();
        let receipt = engine
            .evaluate(
                &request,
                &authority(),
                &assurance(),
                &[bad],
                &approvers(),
                &signer,
                NOW,
            )
            .unwrap();
        assert_eq!(
            receipt.content.decision,
            GovernedDecision::Denied(DenialReason::InvalidApproval)
        );

        let mut second = request.clone();
        second.intent_id = "intent-2".into();
        second.nonce = [2; 16];
        let mut weak = assurance();
        weak.domain_state = DomainState::Unknown;
        let receipt = engine
            .evaluate(
                &second,
                &authority(),
                &weak,
                &[approval(&second)],
                &approvers(),
                &signer,
                NOW,
            )
            .unwrap();
        assert_eq!(
            receipt.content.decision,
            GovernedDecision::Denied(DenialReason::AssuranceDenied(
                FailedCondition::DomainNotKnown
            ))
        );
    }

    #[test]
    fn spaces_read_is_not_execution_authority_and_nonce_reuse_is_rejected() {
        let signer = Blake3MacSigner::new([9; 32]);
        let request = intent(IntentMode::Execute);
        let read_only = AuthorityContext::from_authenticated_grants(["spaces:read"]).unwrap();
        let mut engine = GovernanceEngine::new("issuer".into(), registry()).unwrap();
        let receipt = engine
            .evaluate(
                &request,
                &read_only,
                &assurance(),
                &[approval(&request)],
                &approvers(),
                &signer,
                NOW,
            )
            .unwrap();
        assert_eq!(
            receipt.content.decision,
            GovernedDecision::Denied(DenialReason::MissingAuthority)
        );

        let mut changed_id = request;
        changed_id.intent_id = "intent-other".into();
        assert_eq!(
            engine.evaluate(
                &changed_id,
                &authority(),
                &assurance(),
                &[approval(&changed_id)],
                &approvers(),
                &signer,
                NOW,
            ),
            Err(GovernanceError::NonceReplay)
        );
    }

    #[test]
    fn idempotency_is_exact_and_changed_reuse_is_rejected() {
        let signer = Blake3MacSigner::new([9; 32]);
        let request = intent(IntentMode::Recommend);
        let mut engine = GovernanceEngine::new("issuer".into(), registry()).unwrap();
        let first = engine
            .evaluate(
                &request,
                &authority(),
                &assurance(),
                &[],
                &approvers(),
                &signer,
                NOW,
            )
            .unwrap();
        let replay = engine
            .evaluate(
                &request,
                &authority(),
                &assurance(),
                &[],
                &approvers(),
                &signer,
                NOW + 1,
            )
            .unwrap();
        assert_eq!(first, replay);
        let mut changed = request;
        changed.parameters_digest = [0xAA; 32];
        assert_eq!(
            engine.evaluate(
                &changed,
                &authority(),
                &assurance(),
                &[],
                &approvers(),
                &signer,
                NOW,
            ),
            Err(GovernanceError::IdempotencyConflict)
        );
    }
}
