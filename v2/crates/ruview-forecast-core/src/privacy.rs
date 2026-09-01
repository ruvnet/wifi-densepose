//! Privacy classification and explicit hosted-export authorization receipts.

use crate::digest::CanonicalWriter;
use crate::series::{validate_text, MAX_SOURCE_REFERENCE_LEN};
use crate::{CanonicalDigest, ForecastError, SourceKind, SourceState};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Deserializer, Serialize};

const MAX_SECURITY_ID_LEN: usize = 128;
const MAX_GOVERNANCE_VALIDITY_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
const MAX_CLOCK_SKEW_MS: u64 = 5 * 60 * 1_000;

/// ADR-260 semantic privacy class.
///
/// Evidence and privacy are independent: a measured series can still be P0 or
/// P5, and a synthetic series can model P4 content. This enum intentionally has
/// no total ordering because raw P0 and identity-linked P5 are both restricted
/// for different reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrivacyClass {
    /// Raw waveform or sensor frame.
    P0,
    /// Derived non-identity features.
    P1,
    /// Occupancy and motion only.
    P2,
    /// Anonymous aggregate state.
    P3,
    /// Biometric or health inference.
    P4,
    /// Identity-linked inference.
    P5,
}

/// Immutable tenant, purpose, consent, DPA, export, and retention policy.
///
/// Optional receipts remain explicit `None`; callers must not infer approval
/// from their absence. Only [`FalGovernanceVerifier::verify`] can mint the
/// non-deserializable hosted-training authority handle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DataPolicy {
    privacy_class: PrivacyClass,
    tenant_id: String,
    account_id: String,
    workspace_id: String,
    purpose: String,
    policy_receipt: CanonicalDigest,
    consent_receipt: Option<CanonicalDigest>,
    dpa_receipt: Option<CanonicalDigest>,
    export_receipt: Option<CanonicalDigest>,
    retention_until_ms: u64,
    deidentified: bool,
}

impl DataPolicy {
    /// Construct a validated governance binding.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        privacy_class: PrivacyClass,
        tenant_id: impl Into<String>,
        account_id: impl Into<String>,
        workspace_id: impl Into<String>,
        purpose: impl Into<String>,
        policy_receipt: CanonicalDigest,
        consent_receipt: Option<CanonicalDigest>,
        dpa_receipt: Option<CanonicalDigest>,
        export_receipt: Option<CanonicalDigest>,
        retention_until_ms: u64,
        deidentified: bool,
    ) -> Result<Self, ForecastError> {
        Self::from_parts(
            privacy_class,
            tenant_id.into(),
            account_id.into(),
            workspace_id.into(),
            purpose.into(),
            policy_receipt,
            consent_receipt,
            dpa_receipt,
            export_receipt,
            retention_until_ms,
            deidentified,
        )
    }

    /// Semantic privacy class.
    #[must_use]
    pub const fn privacy_class(&self) -> PrivacyClass {
        self.privacy_class
    }

    /// Tenant binding from an authenticated authority.
    #[must_use]
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Account binding from an authenticated authority.
    #[must_use]
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Workspace binding from an authenticated authority.
    #[must_use]
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    /// Approved processing purpose.
    #[must_use]
    pub fn purpose(&self) -> &str {
        &self.purpose
    }

    /// Governance decision receipt.
    #[must_use]
    pub const fn policy_receipt(&self) -> CanonicalDigest {
        self.policy_receipt
    }

    /// Explicit consent receipt, when required.
    #[must_use]
    pub const fn consent_receipt(&self) -> Option<CanonicalDigest> {
        self.consent_receipt
    }

    /// Data-processing agreement receipt, when export is approved.
    #[must_use]
    pub const fn dpa_receipt(&self) -> Option<CanonicalDigest> {
        self.dpa_receipt
    }

    /// Destination/export approval receipt, when export is approved.
    #[must_use]
    pub const fn export_receipt(&self) -> Option<CanonicalDigest> {
        self.export_receipt
    }

    /// Absolute retention expiry injected by the policy authority.
    #[must_use]
    pub const fn retention_until_ms(&self) -> u64 {
        self.retention_until_ms
    }

    /// Whether a reviewed de-identification transform was applied.
    #[must_use]
    pub const fn is_deidentified(&self) -> bool {
        self.deidentified
    }

    /// Deterministic policy digest bound into series, requests, and receipts.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"data-policy-v1");
        writer.tag(match self.privacy_class {
            PrivacyClass::P0 => 0,
            PrivacyClass::P1 => 1,
            PrivacyClass::P2 => 2,
            PrivacyClass::P3 => 3,
            PrivacyClass::P4 => 4,
            PrivacyClass::P5 => 5,
        });
        writer.string(&self.tenant_id);
        writer.string(&self.account_id);
        writer.string(&self.workspace_id);
        writer.string(&self.purpose);
        writer.digest(self.policy_receipt);
        write_optional_digest(&mut writer, self.consent_receipt);
        write_optional_digest(&mut writer, self.dpa_receipt);
        write_optional_digest(&mut writer, self.export_receipt);
        writer.u64(self.retention_until_ms);
        writer.bool(self.deidentified);
        writer.finish()
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        privacy_class: PrivacyClass,
        tenant_id: String,
        account_id: String,
        workspace_id: String,
        purpose: String,
        policy_receipt: CanonicalDigest,
        consent_receipt: Option<CanonicalDigest>,
        dpa_receipt: Option<CanonicalDigest>,
        export_receipt: Option<CanonicalDigest>,
        retention_until_ms: u64,
        deidentified: bool,
    ) -> Result<Self, ForecastError> {
        validate_text("tenant_id", &tenant_id, MAX_SECURITY_ID_LEN, false)?;
        validate_text("account_id", &account_id, MAX_SECURITY_ID_LEN, false)?;
        validate_text("workspace_id", &workspace_id, MAX_SECURITY_ID_LEN, false)?;
        validate_text("purpose", &purpose, MAX_SOURCE_REFERENCE_LEN, false)?;
        check_digest("policy_receipt", policy_receipt)?;
        check_optional_digest("consent_receipt", consent_receipt)?;
        check_optional_digest("dpa_receipt", dpa_receipt)?;
        check_optional_digest("export_receipt", export_receipt)?;
        if retention_until_ms == 0 {
            return Err(ForecastError::ZeroValue {
                field: "retention_until_ms",
            });
        }
        if matches!(privacy_class, PrivacyClass::P4 | PrivacyClass::P5) && consent_receipt.is_none()
        {
            return Err(ForecastError::MissingReceipt { field: "consent" });
        }
        Ok(Self {
            privacy_class,
            tenant_id,
            account_id,
            workspace_id,
            purpose,
            policy_receipt,
            consent_receipt,
            dpa_receipt,
            export_receipt,
            retention_until_ms,
            deidentified,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DataPolicyWire {
    privacy_class: PrivacyClass,
    tenant_id: String,
    account_id: String,
    workspace_id: String,
    purpose: String,
    policy_receipt: CanonicalDigest,
    consent_receipt: Option<CanonicalDigest>,
    dpa_receipt: Option<CanonicalDigest>,
    export_receipt: Option<CanonicalDigest>,
    retention_until_ms: u64,
    deidentified: bool,
}

impl<'de> Deserialize<'de> for DataPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DataPolicyWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.privacy_class,
            wire.tenant_id,
            wire.account_id,
            wire.workspace_id,
            wire.purpose,
            wire.policy_receipt,
            wire.consent_receipt,
            wire.dpa_receipt,
            wire.export_receipt,
            wire.retention_until_ms,
            wire.deidentified,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Canonical claims signed by an operator governance authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FalGovernanceClaims {
    receipt_id: String,
    dataset_digest: CanonicalDigest,
    schema_digest: CanonicalDigest,
    policy_digest: CanonicalDigest,
    source_kind: SourceKind,
    tenant_id: String,
    account_id: String,
    workspace_id: String,
    purpose: String,
    generator_recipe_digest: CanonicalDigest,
    generator_seed: u64,
    issued_at_ms: u64,
    expires_at_ms: u64,
}

impl FalGovernanceClaims {
    /// Construct claims for an operator-side signing tool.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        receipt_id: impl Into<String>,
        dataset_digest: CanonicalDigest,
        schema_digest: CanonicalDigest,
        source: &SourceState,
        policy: &DataPolicy,
        generator_recipe_digest: CanonicalDigest,
        generator_seed: u64,
        issued_at_ms: u64,
        expires_at_ms: u64,
    ) -> Result<Self, ForecastError> {
        let receipt_id = receipt_id.into();
        validate_text("fal_receipt_id", &receipt_id, MAX_SECURITY_ID_LEN, false)?;
        check_digest("fal_dataset_digest", dataset_digest)?;
        check_digest("fal_schema_digest", schema_digest)?;
        check_digest("generator_recipe_digest", generator_recipe_digest)?;
        if issued_at_ms >= expires_at_ms {
            return Err(ForecastError::InvalidTimeRange);
        }
        if expires_at_ms - issued_at_ms > MAX_GOVERNANCE_VALIDITY_MS {
            return Err(ForecastError::DurationLimitExceeded {
                field: "fal_governance_validity",
                actual_ms: expires_at_ms - issued_at_ms,
                max_ms: MAX_GOVERNANCE_VALIDITY_MS,
            });
        }
        if expires_at_ms > policy.retention_until_ms {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal governance signing",
                reason: "approval exceeds policy retention",
            });
        }
        validate_fal_policy(source, policy)?;
        Ok(Self {
            receipt_id,
            dataset_digest,
            schema_digest,
            policy_digest: policy.canonical_digest(),
            source_kind: source.kind(),
            tenant_id: policy.tenant_id.clone(),
            account_id: policy.account_id.clone(),
            workspace_id: policy.workspace_id.clone(),
            purpose: policy.purpose.clone(),
            generator_recipe_digest,
            generator_seed,
            issued_at_ms,
            expires_at_ms,
        })
    }

    /// Unique bounded governance receipt identifier.
    #[must_use]
    pub fn receipt_id(&self) -> &str {
        &self.receipt_id
    }

    /// Exact approved dataset digest.
    #[must_use]
    pub const fn dataset_digest(&self) -> CanonicalDigest {
        self.dataset_digest
    }

    /// Exact exported feature schema digest.
    #[must_use]
    pub const fn schema_digest(&self) -> CanonicalDigest {
        self.schema_digest
    }

    /// Exact governance policy digest.
    #[must_use]
    pub const fn policy_digest(&self) -> CanonicalDigest {
        self.policy_digest
    }

    /// Evidence source kind at approval time.
    #[must_use]
    pub const fn source_kind(&self) -> SourceKind {
        self.source_kind
    }

    /// Approval expiry in milliseconds.
    #[must_use]
    pub const fn expires_at_ms(&self) -> u64 {
        self.expires_at_ms
    }

    /// Domain-separated digest signed by the operator key.
    #[must_use]
    pub fn signing_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"fal-governance-claims-v1");
        writer.string(&self.receipt_id);
        writer.digest(self.dataset_digest);
        writer.digest(self.schema_digest);
        writer.digest(self.policy_digest);
        writer.tag(match self.source_kind {
            SourceKind::Synthetic => 0,
            SourceKind::Claimed => 1,
            SourceKind::Measured => 2,
        });
        writer.string(&self.tenant_id);
        writer.string(&self.account_id);
        writer.string(&self.workspace_id);
        writer.string(&self.purpose);
        writer.digest(self.generator_recipe_digest);
        writer.u64(self.generator_seed);
        writer.u64(self.issued_at_ms);
        writer.u64(self.expires_at_ms);
        writer.finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FalGovernanceClaimsWire {
    receipt_id: String,
    dataset_digest: CanonicalDigest,
    schema_digest: CanonicalDigest,
    policy_digest: CanonicalDigest,
    source_kind: SourceKind,
    tenant_id: String,
    account_id: String,
    workspace_id: String,
    purpose: String,
    generator_recipe_digest: CanonicalDigest,
    generator_seed: u64,
    issued_at_ms: u64,
    expires_at_ms: u64,
}

impl<'de> Deserialize<'de> for FalGovernanceClaims {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FalGovernanceClaimsWire::deserialize(deserializer)?;
        validate_text(
            "fal_receipt_id",
            &wire.receipt_id,
            MAX_SECURITY_ID_LEN,
            false,
        )
        .and_then(|_| validate_text("tenant_id", &wire.tenant_id, MAX_SECURITY_ID_LEN, false))
        .and_then(|_| validate_text("account_id", &wire.account_id, MAX_SECURITY_ID_LEN, false))
        .and_then(|_| {
            validate_text(
                "workspace_id",
                &wire.workspace_id,
                MAX_SECURITY_ID_LEN,
                false,
            )
        })
        .and_then(|_| validate_text("purpose", &wire.purpose, MAX_SOURCE_REFERENCE_LEN, false))
        .and_then(|_| check_digest("fal_dataset_digest", wire.dataset_digest))
        .and_then(|_| check_digest("fal_schema_digest", wire.schema_digest))
        .and_then(|_| check_digest("fal_policy_digest", wire.policy_digest))
        .and_then(|_| check_digest("generator_recipe_digest", wire.generator_recipe_digest))
        .map_err(serde::de::Error::custom)?;
        if wire.issued_at_ms >= wire.expires_at_ms {
            return Err(serde::de::Error::custom(
                "invalid governance validity range",
            ));
        }
        Ok(Self {
            receipt_id: wire.receipt_id,
            dataset_digest: wire.dataset_digest,
            schema_digest: wire.schema_digest,
            policy_digest: wire.policy_digest,
            source_kind: wire.source_kind,
            tenant_id: wire.tenant_id,
            account_id: wire.account_id,
            workspace_id: wire.workspace_id,
            purpose: wire.purpose,
            generator_recipe_digest: wire.generator_recipe_digest,
            generator_seed: wire.generator_seed,
            issued_at_ms: wire.issued_at_ms,
            expires_at_ms: wire.expires_at_ms,
        })
    }
}

/// Untrusted signed governance envelope. Verification is mandatory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SignedFalGovernanceReceipt {
    claims: FalGovernanceClaims,
    signer_public_key: Vec<u8>,
    signature: Vec<u8>,
}

impl SignedFalGovernanceReceipt {
    /// Construct an untrusted envelope from exact Ed25519 bytes.
    pub fn new(
        claims: FalGovernanceClaims,
        signer_public_key: Vec<u8>,
        signature: Vec<u8>,
    ) -> Result<Self, ForecastError> {
        if signer_public_key.len() != 32 || signature.len() != 64 {
            return Err(ForecastError::GovernanceSignatureInvalid);
        }
        Ok(Self {
            claims,
            signer_public_key,
            signature,
        })
    }

    /// Signed claims.
    #[must_use]
    pub fn claims(&self) -> &FalGovernanceClaims {
        &self.claims
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignedFalGovernanceReceiptWire {
    claims: FalGovernanceClaims,
    signer_public_key: Vec<u8>,
    signature: Vec<u8>,
}

impl<'de> Deserialize<'de> for SignedFalGovernanceReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SignedFalGovernanceReceiptWire::deserialize(deserializer)?;
        Self::new(wire.claims, wire.signer_public_key, wire.signature)
            .map_err(serde::de::Error::custom)
    }
}

/// Operator-key verifier. Keys are validated and fixed at construction.
pub struct FalGovernanceVerifier {
    allowed_public_keys: Vec<[u8; 32]>,
}

impl FalGovernanceVerifier {
    /// Configure a non-empty operator signer allowlist.
    pub fn new(allowed_public_keys: Vec<[u8; 32]>) -> Result<Self, ForecastError> {
        if allowed_public_keys.is_empty() {
            return Err(ForecastError::EmptySignerAllowlist);
        }
        if allowed_public_keys.len() > 32 {
            return Err(ForecastError::LimitExceeded {
                field: "governance_signers",
                actual: allowed_public_keys.len(),
                max: 32,
            });
        }
        for (index, key) in allowed_public_keys.iter().enumerate() {
            if allowed_public_keys[..index].contains(key) {
                return Err(ForecastError::GovernanceSignerNotAllowed);
            }
            VerifyingKey::from_bytes(key).map_err(|_| ForecastError::GovernanceSignatureInvalid)?;
        }
        Ok(Self {
            allowed_public_keys,
        })
    }

    /// Verify signature, authority, exact data/policy, principal, purpose, and
    /// expiry, producing the only type accepted by remote training.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        signed: &SignedFalGovernanceReceipt,
        dataset_digest: CanonicalDigest,
        schema_digest: CanonicalDigest,
        source: &SourceState,
        policy: &DataPolicy,
        principal_tenant_id: &str,
        principal_account_id: &str,
        principal_workspace_id: &str,
        now_ms: u64,
    ) -> Result<VerifiedFalDataset, ForecastError> {
        let signer: [u8; 32] = signed
            .signer_public_key
            .as_slice()
            .try_into()
            .map_err(|_| ForecastError::GovernanceSignatureInvalid)?;
        if !self.allowed_public_keys.contains(&signer) {
            return Err(ForecastError::GovernanceSignerNotAllowed);
        }
        let key = VerifyingKey::from_bytes(&signer)
            .map_err(|_| ForecastError::GovernanceSignatureInvalid)?;
        let signature_bytes: [u8; 64] = signed
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| ForecastError::GovernanceSignatureInvalid)?;
        let signature = Signature::from_bytes(&signature_bytes);
        key.verify_strict(signed.claims.signing_digest().as_bytes(), &signature)
            .map_err(|_| ForecastError::GovernanceSignatureInvalid)?;

        validate_fal_policy(source, policy)?;
        let claims = &signed.claims;
        if claims.dataset_digest != dataset_digest
            || claims.schema_digest != schema_digest
            || claims.policy_digest != policy.canonical_digest()
            || claims.source_kind != source.kind()
            || claims.tenant_id != policy.tenant_id
            || claims.account_id != policy.account_id
            || claims.workspace_id != policy.workspace_id
            || claims.purpose != policy.purpose
        {
            return Err(ForecastError::DigestMismatch {
                field: "fal_governance_claims",
            });
        }
        if claims.tenant_id != principal_tenant_id
            || claims.account_id != principal_account_id
            || claims.workspace_id != principal_workspace_id
        {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal governance verification",
                reason: "authenticated principal does not match signed claims",
            });
        }
        let latest_acceptable_issue =
            now_ms
                .checked_add(MAX_CLOCK_SKEW_MS)
                .ok_or(ForecastError::SizeOverflow {
                    field: "governance_clock_skew",
                })?;
        if claims.issued_at_ms > latest_acceptable_issue
            || now_ms >= claims.expires_at_ms
            || claims.expires_at_ms > policy.retention_until_ms
        {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal governance verification",
                reason: "signed authorization is not currently valid",
            });
        }
        let mut receipt_writer = CanonicalWriter::new(b"signed-fal-governance-receipt-v1");
        receipt_writer.digest(claims.signing_digest());
        receipt_writer.bytes(&signer);
        receipt_writer.bytes(&signed.signature);
        Ok(VerifiedFalDataset {
            claims: claims.clone(),
            signed_receipt_digest: receipt_writer.finish(),
        })
    }
}

/// Nonconstructible proof that an operator-authorized dataset is currently
/// eligible for fal. It cannot be deserialized from caller input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedFalDataset {
    claims: FalGovernanceClaims,
    signed_receipt_digest: CanonicalDigest,
}

impl VerifiedFalDataset {
    /// Exact dataset digest.
    #[must_use]
    pub const fn dataset_digest(&self) -> CanonicalDigest {
        self.claims.dataset_digest
    }

    /// Exact schema digest.
    #[must_use]
    pub const fn schema_digest(&self) -> CanonicalDigest {
        self.claims.schema_digest
    }

    /// Exact policy digest.
    #[must_use]
    pub const fn policy_digest(&self) -> CanonicalDigest {
        self.claims.policy_digest
    }

    /// Signed authorization expiry.
    #[must_use]
    pub const fn expires_at_ms(&self) -> u64 {
        self.claims.expires_at_ms
    }

    /// Exact synthetic generator recipe digest.
    #[must_use]
    pub const fn generator_recipe_digest(&self) -> CanonicalDigest {
        self.claims.generator_recipe_digest
    }

    /// Deterministic synthetic generator seed.
    #[must_use]
    pub const fn generator_seed(&self) -> u64 {
        self.claims.generator_seed
    }

    /// Recheck authenticated principal and time immediately before upload.
    pub fn reverify_submission_context(
        &self,
        principal_tenant_id: &str,
        principal_account_id: &str,
        principal_workspace_id: &str,
        now_ms: u64,
    ) -> Result<(), ForecastError> {
        if self.claims.tenant_id != principal_tenant_id
            || self.claims.account_id != principal_account_id
            || self.claims.workspace_id != principal_workspace_id
        {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal submission",
                reason: "authenticated principal does not match verified dataset",
            });
        }
        let latest_acceptable_issue =
            now_ms
                .checked_add(MAX_CLOCK_SKEW_MS)
                .ok_or(ForecastError::SizeOverflow {
                    field: "governance_clock_skew",
                })?;
        if self.claims.issued_at_ms > latest_acceptable_issue || now_ms >= self.claims.expires_at_ms
        {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal submission",
                reason: "verified export authorization is expired",
            });
        }
        Ok(())
    }

    /// Deterministic digest bound into [`crate::TrainSpec`].
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"verified-fal-dataset-v1");
        writer.digest(self.claims.signing_digest());
        writer.digest(self.signed_receipt_digest);
        writer.finish()
    }
}

fn validate_fal_policy(source: &SourceState, policy: &DataPolicy) -> Result<(), ForecastError> {
    if policy.dpa_receipt.is_none() {
        return Err(ForecastError::MissingReceipt { field: "DPA" });
    }
    if policy.export_receipt.is_none() {
        return Err(ForecastError::MissingReceipt { field: "export" });
    }
    if source.kind() != SourceKind::Synthetic {
        return Err(ForecastError::PrivacyDenied {
            operation: "fal export",
            reason: "v1 hosted training accepts synthetic datasets only",
        });
    }
    Ok(())
}

fn check_digest(field: &'static str, digest: CanonicalDigest) -> Result<(), ForecastError> {
    if digest.is_zero() {
        return Err(ForecastError::ZeroDigest { field });
    }
    Ok(())
}

fn check_optional_digest(
    field: &'static str,
    digest: Option<CanonicalDigest>,
) -> Result<(), ForecastError> {
    if digest.is_some_and(|value| value.is_zero()) {
        return Err(ForecastError::ZeroDigest { field });
    }
    Ok(())
}

fn write_optional_digest(writer: &mut CanonicalWriter, digest: Option<CanonicalDigest>) {
    match digest {
        Some(value) => {
            writer.bool(true);
            writer.digest(value);
        }
        None => writer.bool(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"privacy-test", value)
    }

    fn policy(class: PrivacyClass, export: bool, consent: bool) -> DataPolicy {
        DataPolicy::new(
            class,
            "tenant",
            "account",
            "workspace",
            "forecast training",
            digest(b"policy"),
            consent.then(|| digest(b"consent")),
            export.then(|| digest(b"dpa")),
            export.then(|| digest(b"export")),
            9_999_999,
            true,
        )
        .unwrap()
    }

    fn signed_receipt(
        source: &SourceState,
        policy: &DataPolicy,
    ) -> (SigningKey, SignedFalGovernanceReceipt) {
        let claims = FalGovernanceClaims::new(
            "approval-1",
            digest(b"data"),
            digest(b"schema"),
            source,
            policy,
            digest(b"generator-recipe"),
            42,
            100,
            1_000,
        )
        .unwrap();
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        let signature = signing_key.sign(claims.signing_digest().as_bytes());
        let signed = SignedFalGovernanceReceipt::new(
            claims,
            signing_key.verifying_key().to_bytes().to_vec(),
            signature.to_bytes().to_vec(),
        )
        .unwrap();
        (signing_key, signed)
    }

    #[test]
    fn real_restricted_data_is_never_fal_eligible() {
        let source = SourceState::claimed("capture").unwrap();
        for class in [
            PrivacyClass::P0,
            PrivacyClass::P1,
            PrivacyClass::P2,
            PrivacyClass::P3,
            PrivacyClass::P4,
            PrivacyClass::P5,
        ] {
            let policy = policy(class, true, true);
            assert!(matches!(
                FalGovernanceClaims::new(
                    "denied",
                    digest(b"data"),
                    digest(b"schema"),
                    &source,
                    &policy,
                    digest(b"recipe"),
                    1,
                    1,
                    2,
                ),
                Err(ForecastError::PrivacyDenied { .. })
            ));
        }
    }

    #[test]
    fn synthetic_data_still_requires_governance_receipts() {
        let source = SourceState::synthetic("generator").unwrap();
        let missing = policy(PrivacyClass::P1, false, true);
        assert!(matches!(
            FalGovernanceClaims::new(
                "missing",
                digest(b"data"),
                digest(b"schema"),
                &source,
                &missing,
                digest(b"recipe"),
                1,
                1,
                2,
            ),
            Err(ForecastError::MissingReceipt { .. })
        ));
    }

    #[test]
    fn verified_handle_requires_signature_and_exact_current_context() {
        let source = SourceState::synthetic("generator").unwrap();
        let policy = policy(PrivacyClass::P1, true, false);
        let (key, signed) = signed_receipt(&source, &policy);
        let verifier = FalGovernanceVerifier::new(vec![key.verifying_key().to_bytes()]).unwrap();
        let verified = verifier
            .verify(
                &signed,
                digest(b"data"),
                digest(b"schema"),
                &source,
                &policy,
                "tenant",
                "account",
                "workspace",
                500,
            )
            .unwrap();
        assert_eq!(verified.generator_seed(), 42);
        assert!(verified
            .reverify_submission_context("tenant", "account", "workspace", 999)
            .is_ok());

        assert!(matches!(
            verifier.verify(
                &signed,
                digest(b"data"),
                digest(b"wrong-schema"),
                &source,
                &policy,
                "tenant",
                "account",
                "workspace",
                500,
            ),
            Err(ForecastError::DigestMismatch { .. })
        ));
        assert!(matches!(
            verifier.verify(
                &signed,
                digest(b"data"),
                digest(b"schema"),
                &SourceState::claimed("real").unwrap(),
                &policy,
                "tenant",
                "account",
                "workspace",
                500,
            ),
            Err(ForecastError::PrivacyDenied { .. })
        ));
        assert!(matches!(
            verifier.verify(
                &signed,
                digest(b"data"),
                digest(b"schema"),
                &source,
                &policy,
                "wrong",
                "account",
                "workspace",
                500,
            ),
            Err(ForecastError::PrivacyDenied { .. })
        ));
        assert!(matches!(
            verified.reverify_submission_context("tenant", "account", "workspace", 1_000),
            Err(ForecastError::PrivacyDenied { .. })
        ));
    }

    #[test]
    fn policy_deserialization_is_strict() {
        let value = policy(PrivacyClass::P2, true, true);
        let mut json = serde_json::to_value(&value).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::json!(1));
        assert!(serde_json::from_value::<DataPolicy>(json).is_err());
    }

    #[test]
    fn governance_verifier_rejects_duplicate_signers() {
        let key = SigningKey::from_bytes(&[9_u8; 32])
            .verifying_key()
            .to_bytes();
        assert!(matches!(
            FalGovernanceVerifier::new(vec![key, key]),
            Err(ForecastError::GovernanceSignerNotAllowed)
        ));
    }
}
