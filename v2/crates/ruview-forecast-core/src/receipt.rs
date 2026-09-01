//! Immutable provenance and content receipts.

use crate::digest::CanonicalWriter;
use crate::series::{validate_text, MAX_SOURCE_REFERENCE_LEN};
use crate::{CanonicalDigest, ForecastError};
use serde::{Deserialize, Deserializer, Serialize};

const RECEIPT_SCHEMA_VERSION: u16 = 1;
const MAX_MODEL_ID_LEN: usize = 256;
const MAX_MODEL_VERSION_LEN: usize = 128;

/// Evidence class ordered from weakest to strongest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SourceKind {
    /// Generated or simulated data only.
    Synthetic,
    /// A real or derived result without a ground-truth reproducer.
    Claimed,
    /// Ground-truth-backed data with a dataset digest and reproducer.
    Measured,
}

/// Validated provenance claim attached to inputs, artifacts, and outputs.
///
/// Fields are private and deserialization re-runs the same constructor rules,
/// preventing a synthetic source from being represented as measured without a
/// dataset digest and reproducer. This structure is not an authority signature;
/// an evidence service must verify referenced receipts before relying on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SourceState {
    kind: SourceKind,
    reference: String,
    dataset_digest: Option<CanonicalDigest>,
    reproducer: Option<String>,
}

impl SourceState {
    /// Record a bounded synthetic generator or fixture reference.
    pub fn synthetic(reference: impl Into<String>) -> Result<Self, ForecastError> {
        Self::from_parts(SourceKind::Synthetic, reference.into(), None, None)
    }

    /// Record a bounded claim or acquisition reference without ground truth.
    pub fn claimed(reference: impl Into<String>) -> Result<Self, ForecastError> {
        Self::from_parts(SourceKind::Claimed, reference.into(), None, None)
    }

    /// Record measured evidence backed by an immutable dataset and reproducer.
    pub fn measured(
        reference: impl Into<String>,
        dataset_digest: CanonicalDigest,
        reproducer: impl Into<String>,
    ) -> Result<Self, ForecastError> {
        Self::from_parts(
            SourceKind::Measured,
            reference.into(),
            Some(dataset_digest),
            Some(reproducer.into()),
        )
    }

    /// Evidence class.
    #[must_use]
    pub const fn kind(&self) -> SourceKind {
        self.kind
    }

    /// Bounded source reference.
    #[must_use]
    pub fn reference(&self) -> &str {
        &self.reference
    }

    /// Measured dataset digest, present only for measured evidence.
    #[must_use]
    pub const fn dataset_digest(&self) -> Option<CanonicalDigest> {
        self.dataset_digest
    }

    /// Ground-truth reproducer, present only for measured evidence.
    #[must_use]
    pub fn reproducer(&self) -> Option<&str> {
        self.reproducer.as_deref()
    }

    /// The strongest evidence kind derivation may retain from two inputs.
    #[must_use]
    pub fn evidence_floor(left: &Self, right: &Self) -> SourceKind {
        left.kind.min(right.kind)
    }

    /// Create an honest default state for a derived forecast.
    ///
    /// Synthetic input remains synthetic. Any non-synthetic result defaults to
    /// claimed because producing a value is not itself ground-truth evaluation.
    pub fn derived_forecast(
        reference: impl Into<String>,
        input: &Self,
        artifact: &Self,
    ) -> Result<Self, ForecastError> {
        let reference = reference.into();
        match Self::evidence_floor(input, artifact) {
            SourceKind::Synthetic => Self::synthetic(reference),
            SourceKind::Claimed | SourceKind::Measured => Self::claimed(reference),
        }
    }

    /// Deterministic digest of this structured provenance claim.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"source-state-v1");
        self.write_canonical(&mut writer);
        writer.finish()
    }

    pub(crate) fn validate(&self) -> Result<(), ForecastError> {
        validate_text(
            "source_reference",
            &self.reference,
            MAX_SOURCE_REFERENCE_LEN,
            false,
        )?;
        match self.kind {
            SourceKind::Synthetic | SourceKind::Claimed => {
                if self.dataset_digest.is_some() || self.reproducer.is_some() {
                    return Err(ForecastError::InvalidSourceState {
                        reason: "only measured sources may carry dataset evidence",
                    });
                }
            }
            SourceKind::Measured => {
                let digest = self
                    .dataset_digest
                    .ok_or(ForecastError::InvalidSourceState {
                        reason: "measured source requires a dataset digest",
                    })?;
                if digest.is_zero() {
                    return Err(ForecastError::ZeroDigest {
                        field: "source_dataset_digest",
                    });
                }
                let reproducer =
                    self.reproducer
                        .as_deref()
                        .ok_or(ForecastError::InvalidSourceState {
                            reason: "measured source requires a reproducer",
                        })?;
                validate_text(
                    "source_reproducer",
                    reproducer,
                    MAX_SOURCE_REFERENCE_LEN,
                    false,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.tag(match self.kind {
            SourceKind::Synthetic => 0,
            SourceKind::Claimed => 1,
            SourceKind::Measured => 2,
        });
        writer.string(&self.reference);
        match self.dataset_digest {
            Some(digest) => {
                writer.bool(true);
                writer.digest(digest);
            }
            None => writer.bool(false),
        }
        match &self.reproducer {
            Some(reproducer) => {
                writer.bool(true);
                writer.string(reproducer);
            }
            None => writer.bool(false),
        }
    }

    fn from_parts(
        kind: SourceKind,
        reference: String,
        dataset_digest: Option<CanonicalDigest>,
        reproducer: Option<String>,
    ) -> Result<Self, ForecastError> {
        let state = Self {
            kind,
            reference,
            dataset_digest,
            reproducer,
        };
        state.validate()?;
        Ok(state)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceStateWire {
    kind: SourceKind,
    reference: String,
    dataset_digest: Option<CanonicalDigest>,
    reproducer: Option<String>,
}

impl<'de> Deserialize<'de> for SourceState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SourceStateWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.kind,
            wire.reference,
            wire.dataset_digest,
            wire.reproducer,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Immutable identity and provenance receipt for an activated model artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArtifactReceipt {
    schema_version: u16,
    model_id: String,
    model_version: String,
    artifact_digest: CanonicalDigest,
    config_digest: CanonicalDigest,
    policy_digest: CanonicalDigest,
    source: SourceState,
}

impl ArtifactReceipt {
    /// Construct a validated artifact receipt.
    pub fn new(
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        artifact_digest: CanonicalDigest,
        config_digest: CanonicalDigest,
        policy_digest: CanonicalDigest,
        source: SourceState,
    ) -> Result<Self, ForecastError> {
        Self::from_parts(
            RECEIPT_SCHEMA_VERSION,
            model_id.into(),
            model_version.into(),
            artifact_digest,
            config_digest,
            policy_digest,
            source,
        )
    }

    /// Stable model family identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Stable artifact version.
    #[must_use]
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Digest of exact activated artifact bytes.
    #[must_use]
    pub const fn artifact_digest(&self) -> CanonicalDigest {
        self.artifact_digest
    }

    /// Digest of exact model configuration.
    #[must_use]
    pub const fn config_digest(&self) -> CanonicalDigest {
        self.config_digest
    }

    /// Digest of the training-data governance policy for this artifact.
    #[must_use]
    pub const fn policy_digest(&self) -> CanonicalDigest {
        self.policy_digest
    }

    /// Artifact evidence provenance.
    #[must_use]
    pub fn source(&self) -> &SourceState {
        &self.source
    }

    /// Deterministic digest of the full receipt.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"artifact-receipt-v1");
        writer.u64(u64::from(self.schema_version));
        writer.string(&self.model_id);
        writer.string(&self.model_version);
        writer.digest(self.artifact_digest);
        writer.digest(self.config_digest);
        writer.digest(self.policy_digest);
        self.source.write_canonical(&mut writer);
        writer.finish()
    }

    fn from_parts(
        schema_version: u16,
        model_id: String,
        model_version: String,
        artifact_digest: CanonicalDigest,
        config_digest: CanonicalDigest,
        policy_digest: CanonicalDigest,
        source: SourceState,
    ) -> Result<Self, ForecastError> {
        if schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ForecastError::LimitExceeded {
                field: "receipt_schema_version",
                actual: usize::from(schema_version),
                max: usize::from(RECEIPT_SCHEMA_VERSION),
            });
        }
        validate_text("model_id", &model_id, MAX_MODEL_ID_LEN, false)?;
        validate_text(
            "model_version",
            &model_version,
            MAX_MODEL_VERSION_LEN,
            false,
        )?;
        if artifact_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "artifact_digest",
            });
        }
        if config_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "config_digest",
            });
        }
        if policy_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "artifact_policy_digest",
            });
        }
        source.validate()?;
        Ok(Self {
            schema_version,
            model_id,
            model_version,
            artifact_digest,
            config_digest,
            policy_digest,
            source,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactReceiptWire {
    schema_version: u16,
    model_id: String,
    model_version: String,
    artifact_digest: CanonicalDigest,
    config_digest: CanonicalDigest,
    policy_digest: CanonicalDigest,
    source: SourceState,
}

impl<'de> Deserialize<'de> for ArtifactReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ArtifactReceiptWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.schema_version,
            wire.model_id,
            wire.model_version,
            wire.artifact_digest,
            wire.config_digest,
            wire.policy_digest,
            wire.source,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Immutable link from a forecast to its request, output, and model artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ForecastReceipt {
    schema_version: u16,
    artifact: ArtifactReceipt,
    request_digest: CanonicalDigest,
    output_digest: CanonicalDigest,
    input_source: SourceState,
    input_policy_digest: CanonicalDigest,
    source: SourceState,
}

impl ForecastReceipt {
    /// Construct a validated content receipt.
    pub fn new(
        artifact: ArtifactReceipt,
        request_digest: CanonicalDigest,
        output_digest: CanonicalDigest,
        input_source: SourceState,
        input_policy_digest: CanonicalDigest,
        source: SourceState,
    ) -> Result<Self, ForecastError> {
        Self::from_parts(
            RECEIPT_SCHEMA_VERSION,
            artifact,
            request_digest,
            output_digest,
            input_source,
            input_policy_digest,
            source,
        )
    }

    /// Activated artifact receipt.
    #[must_use]
    pub fn artifact(&self) -> &ArtifactReceipt {
        &self.artifact
    }

    /// Exact canonical request digest.
    #[must_use]
    pub const fn request_digest(&self) -> CanonicalDigest {
        self.request_digest
    }

    /// Exact canonical output-payload digest.
    #[must_use]
    pub const fn output_digest(&self) -> CanonicalDigest {
        self.output_digest
    }

    /// Evidence state of the exact input series.
    #[must_use]
    pub fn input_source(&self) -> &SourceState {
        &self.input_source
    }

    /// Governance policy digest of the exact input series.
    #[must_use]
    pub const fn input_policy_digest(&self) -> CanonicalDigest {
        self.input_policy_digest
    }

    /// Evidence state of the derived output.
    #[must_use]
    pub fn source(&self) -> &SourceState {
        &self.source
    }

    /// Deterministic digest of the complete receipt.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"forecast-receipt-v1");
        writer.u64(u64::from(self.schema_version));
        writer.digest(self.artifact.canonical_digest());
        writer.digest(self.request_digest);
        writer.digest(self.output_digest);
        self.input_source.write_canonical(&mut writer);
        writer.digest(self.input_policy_digest);
        self.source.write_canonical(&mut writer);
        writer.finish()
    }

    fn from_parts(
        schema_version: u16,
        artifact: ArtifactReceipt,
        request_digest: CanonicalDigest,
        output_digest: CanonicalDigest,
        input_source: SourceState,
        input_policy_digest: CanonicalDigest,
        source: SourceState,
    ) -> Result<Self, ForecastError> {
        if schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ForecastError::LimitExceeded {
                field: "receipt_schema_version",
                actual: usize::from(schema_version),
                max: usize::from(RECEIPT_SCHEMA_VERSION),
            });
        }
        if request_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "request_digest",
            });
        }
        if output_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "output_digest",
            });
        }
        if input_policy_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "input_policy_digest",
            });
        }
        input_source.validate()?;
        source.validate()?;
        let allowed =
            SourceState::evidence_floor(&input_source, artifact.source()).min(SourceKind::Claimed);
        if source.kind() > allowed {
            return Err(ForecastError::EvidenceEscalation {
                output: source.kind(),
                allowed,
            });
        }
        Ok(Self {
            schema_version,
            artifact,
            request_digest,
            output_digest,
            input_source,
            input_policy_digest,
            source,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastReceiptWire {
    schema_version: u16,
    artifact: ArtifactReceipt,
    request_digest: CanonicalDigest,
    output_digest: CanonicalDigest,
    input_source: SourceState,
    input_policy_digest: CanonicalDigest,
    source: SourceState,
}

impl<'de> Deserialize<'de> for ForecastReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ForecastReceiptWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.schema_version,
            wire.artifact,
            wire.request_digest,
            wire.output_digest,
            wire.input_source,
            wire.input_policy_digest,
            wire.source,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"test", label)
    }

    #[test]
    fn measured_requires_real_digest_and_reproducer() {
        let invalid =
            SourceState::measured("dataset", CanonicalDigest::default(), "cargo test --locked");
        assert!(matches!(invalid, Err(ForecastError::ZeroDigest { .. })));
        let valid = SourceState::measured("dataset", digest(b"data"), "reproducer").unwrap();
        assert_eq!(valid.kind(), SourceKind::Measured);
    }

    #[test]
    fn source_deserialization_cannot_forge_measured_state() {
        let json = r#"{"kind":"MEASURED","reference":"x","dataset_digest":null,"reproducer":null}"#;
        assert!(serde_json::from_str::<SourceState>(json).is_err());
    }

    #[test]
    fn artifact_receipt_round_trip_is_stable() {
        let receipt = ArtifactReceipt::new(
            "last-value",
            "1",
            digest(b"artifact"),
            digest(b"config"),
            digest(b"policy"),
            SourceState::claimed("built-in").unwrap(),
        )
        .unwrap();
        let before = receipt.canonical_digest();
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: ArtifactReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.canonical_digest(), before);
    }
}
