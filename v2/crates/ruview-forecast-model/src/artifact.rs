//! Canonical candidate artifacts and signed activation envelopes.
//!
//! [`ModelArtifact`] is deliberately a **candidate-only** value. A candidate
//! can be produced by an untrusted trainer, including a remote trainer, but it
//! cannot be passed to the Burn decoder. Activation requires
//! [`VerifiedModelArtifact`], which can only be obtained by verifying an
//! Ed25519 signature from a locally configured [`TrustedSignerSet`].

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ConfigError, ForecastModelConfig, QUANTILE_COUNT};

/// Binary prefix for an independently trained `RuView` forecast candidate.
pub const ARTIFACT_MAGIC: [u8; 8] = *b"RVFM\0\0\0\x01";
/// Binary prefix for an Ed25519-signed activation envelope.
pub const SIGNED_ARTIFACT_MAGIC: [u8; 8] = *b"RVFS\0\0\0\x01";
/// Hard upper bound for any artifact envelope accepted by this crate.
pub const MAX_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;

const MANIFEST_MAGIC: [u8; 8] = *b"RVMM\0\0\0\x01";
const MAX_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_BUILD_ID_BYTES: usize = 128;
const MAX_TRUSTED_SIGNERS: usize = 32;
const SIGNED_HEADER_BYTES: usize = 8 + 8 + 32 + 64;
const RECORD_OVERHEAD_BYTES: usize = 4 * 1024 * 1024;
const SIGNING_DOMAIN: &[u8] = b"ruview.forecast.signed-artifact.v1\0";
/// Runtime compatibility level understood by this crate release.
pub const RUNTIME_COMPATIBILITY_VERSION: u32 = 1;

/// Artifact validation, signature, or codec failure.
#[derive(Debug, Error)]
pub enum ModelError {
    /// Architecture configuration is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Envelope is malformed or exceeds a declared bound.
    #[error("malformed artifact: {0}")]
    Malformed(&'static str),
    /// Manifest and payload disagree.
    #[error("artifact digest mismatch")]
    DigestMismatch,
    /// The envelope was not signed by a locally allowlisted key.
    #[error("artifact signer is not trusted")]
    UntrustedSigner,
    /// The allowlisted Ed25519 signature did not verify.
    #[error("artifact signature verification failed")]
    InvalidSignature,
    /// A validly signed artifact violated local activation policy.
    #[error("artifact activation policy rejected it: {0}")]
    ActivationPolicy(&'static str),
    /// Runtime tensor shape or value constraints were violated.
    #[error("invalid model input: {0}")]
    Shape(String),
    /// Clean-room declaration is absent or contradictory.
    #[error("artifact clean-room declaration is not acceptable")]
    CleanRoom,
    /// A backend record could not be encoded or decoded.
    #[cfg(feature = "model")]
    #[error("model record failure: {0}")]
    Record(String),
}

/// Auditable manifest carried beside, but not inside, executable code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    /// Envelope schema version.
    pub schema_version: u16,
    /// Stable architecture identifier.
    pub architecture: String,
    /// Exact architecture configuration.
    pub config: ForecastModelConfig,
    /// Declared number of learned scalars.
    pub parameter_count: usize,
    /// Digest of the feature schema used for training and activation.
    pub feature_schema_digest: [u8; 32],
    /// Digest of the immutable split/data manifest, never the raw data.
    pub training_manifest_digest: [u8; 32],
    /// BLAKE3 digest of the model record bytes.
    pub weights_digest: [u8; 32],
    /// Reproducible training seed.
    pub seed: u64,
    /// Monotonic publisher release counter used to reject rollback.
    pub release_epoch: u64,
    /// Oldest `RuView` forecast runtime compatibility level accepted.
    pub minimum_runtime_version: u32,
    /// Newest `RuView` forecast runtime compatibility level accepted.
    pub maximum_runtime_version: u32,
    /// Optional absolute UTC expiry in Unix milliseconds.
    pub expires_at_unix_ms: Option<u64>,
    /// Build identity supplied by the trusted training pipeline.
    pub build_id: String,
    /// Must remain false: no third-party forecast outputs were teacher labels.
    pub teacher_outputs_used: bool,
    /// Must remain true for activation under the clean-room contract.
    pub independently_implemented: bool,
}

impl ArtifactManifest {
    /// Validate the manifest without allocating model tensors.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.config.validate()?;
        if self.schema_version != 1
            || self.architecture != "ruview-factorized-forecast-mixer-v1"
            || self.parameter_count != self.config.parameter_count()?
        {
            return Err(ModelError::Malformed("manifest/config disagreement"));
        }
        if self.teacher_outputs_used || !self.independently_implemented {
            return Err(ModelError::CleanRoom);
        }
        if self.release_epoch == 0
            || self.minimum_runtime_version == 0
            || self.minimum_runtime_version > self.maximum_runtime_version
            || self.expires_at_unix_ms == Some(0)
        {
            return Err(ModelError::Malformed(
                "invalid release compatibility policy",
            ));
        }
        let build = self.build_id.as_bytes();
        if build.is_empty()
            || build.len() > MAX_BUILD_ID_BYTES
            || build.first() == Some(&b' ')
            || build.last() == Some(&b' ')
            || !build.iter().all(|byte| (0x20..=0x7e).contains(byte))
        {
            return Err(ModelError::Malformed("invalid build_id"));
        }
        if self.feature_schema_digest == [0; 32]
            || self.training_manifest_digest == [0; 32]
            || self.weights_digest == [0; 32]
        {
            return Err(ModelError::Malformed("zero digest is not an identity"));
        }
        Ok(())
    }

    fn encode_canonical(&self) -> Result<Vec<u8>, ModelError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(512);
        bytes.extend_from_slice(&MANIFEST_MAGIC);
        push_u16(&mut bytes, self.schema_version);
        push_text(&mut bytes, &self.architecture)?;
        encode_config(&mut bytes, &self.config)?;
        push_usize(&mut bytes, self.parameter_count)?;
        bytes.extend_from_slice(&self.feature_schema_digest);
        bytes.extend_from_slice(&self.training_manifest_digest);
        bytes.extend_from_slice(&self.weights_digest);
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.extend_from_slice(&self.release_epoch.to_le_bytes());
        bytes.extend_from_slice(&self.minimum_runtime_version.to_le_bytes());
        bytes.extend_from_slice(&self.maximum_runtime_version.to_le_bytes());
        match self.expires_at_unix_ms {
            Some(value) => {
                bytes.push(1);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            None => bytes.push(0),
        }
        push_text(&mut bytes, &self.build_id)?;
        bytes.push(u8::from(self.teacher_outputs_used));
        bytes.push(u8::from(self.independently_implemented));
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(ModelError::Malformed("manifest exceeds limit"));
        }
        Ok(bytes)
    }

    fn decode_canonical(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
            return Err(ModelError::Malformed("invalid manifest length"));
        }
        let mut reader = Reader::new(bytes);
        if reader.take::<8>()? != MANIFEST_MAGIC {
            return Err(ModelError::Malformed("bad manifest magic"));
        }
        let schema_version = reader.u16()?;
        let architecture = reader.text(64)?;
        let config = decode_config(&mut reader)?;
        let parameter_count = reader.usize()?;
        let feature_schema_digest = reader.take::<32>()?;
        let training_manifest_digest = reader.take::<32>()?;
        let weights_digest = reader.take::<32>()?;
        let seed = reader.u64()?;
        let release_epoch = reader.u64()?;
        let minimum_runtime_version = reader.u32()?;
        let maximum_runtime_version = reader.u32()?;
        let expires_at_unix_ms = match reader.take::<1>()?[0] {
            0 => None,
            1 => Some(reader.u64()?),
            _ => return Err(ModelError::Malformed("invalid optional expiry tag")),
        };
        let build_id = reader.text(MAX_BUILD_ID_BYTES)?;
        let teacher_outputs_used = reader.boolean()?;
        let independently_implemented = reader.boolean()?;
        if !reader.is_empty() {
            return Err(ModelError::Malformed("trailing manifest bytes"));
        }
        let manifest = Self {
            schema_version,
            architecture,
            config,
            parameter_count,
            feature_schema_digest,
            training_manifest_digest,
            weights_digest,
            seed,
            release_epoch,
            minimum_runtime_version,
            maximum_runtime_version,
            expires_at_unix_ms,
            build_id,
            teacher_outputs_used,
            independently_implemented,
        };
        manifest.validate()?;
        if manifest.encode_canonical()? != bytes {
            return Err(ModelError::Malformed("non-canonical manifest"));
        }
        Ok(manifest)
    }
}

/// Candidate model record produced by a trainer.
///
/// This type proves bounds, clean-room declarations, and a payload digest. It
/// does **not** prove publisher authenticity and therefore cannot activate a
/// runtime.
#[derive(Debug, Clone)]
pub struct ModelArtifact {
    manifest: ArtifactManifest,
    weights: Vec<u8>,
}

impl ModelArtifact {
    /// Construct and validate a candidate from trainer outputs.
    pub fn new(manifest: ArtifactManifest, weights: Vec<u8>) -> Result<Self, ModelError> {
        manifest.validate()?;
        let record_limit = manifest
            .parameter_count
            .checked_mul(5)
            .and_then(|value| value.checked_add(RECORD_OVERHEAD_BYTES))
            .ok_or(ModelError::Malformed("record limit overflow"))?
            .min(MAX_ARTIFACT_BYTES - SIGNED_HEADER_BYTES);
        if weights.is_empty() || weights.len() > record_limit {
            return Err(ModelError::Malformed("invalid model record length"));
        }
        if *blake3::hash(&weights).as_bytes() != manifest.weights_digest {
            return Err(ModelError::DigestMismatch);
        }
        Ok(Self { manifest, weights })
    }

    /// Validated candidate manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Digest-checked candidate record bytes.
    #[must_use]
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    /// Encode the canonical candidate envelope.
    pub fn encode(&self) -> Result<Vec<u8>, ModelError> {
        let manifest = self.manifest.encode_canonical()?;
        let total = 20usize
            .checked_add(manifest.len())
            .and_then(|value| value.checked_add(self.weights.len()))
            .ok_or(ModelError::Malformed("artifact length overflow"))?;
        if total > MAX_ARTIFACT_BYTES - SIGNED_HEADER_BYTES {
            return Err(ModelError::Malformed("artifact exceeds limit"));
        }
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&ARTIFACT_MAGIC);
        push_u32(&mut bytes, manifest.len())?;
        push_u64(&mut bytes, self.weights.len())?;
        bytes.extend_from_slice(&manifest);
        bytes.extend_from_slice(&self.weights);
        Ok(bytes)
    }

    /// Decode and digest-check a candidate without granting activation rights.
    pub fn decode(bytes: &[u8]) -> Result<Self, ModelError> {
        let sections = candidate_sections(bytes)?;
        let manifest = ArtifactManifest::decode_canonical(sections.manifest)?;
        let artifact = Self::new(manifest, sections.weights.to_vec())?;
        if artifact.encode()? != bytes {
            return Err(ModelError::Malformed("non-canonical candidate"));
        }
        Ok(artifact)
    }

    /// Domain-separated message for an offline release signer.
    ///
    /// The message commits to the exact canonical candidate bytes and to the
    /// BLAKE3 digest recomputed from the actual record bytes.
    pub fn signing_message(&self) -> Result<Vec<u8>, ModelError> {
        signing_message(&self.encode()?)
    }
}

/// Locally configured, bounded allowlist of release-signing keys.
#[derive(Debug, Clone)]
pub struct TrustedSignerSet {
    keys: Vec<[u8; 32]>,
}

/// Local anti-rollback and compatibility requirements for activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactActivationPolicy {
    minimum_release_epoch: u64,
    runtime_version: u32,
    now_unix_ms: u64,
    expected_feature_schema_digest: [u8; 32],
    accepted_digest_at_minimum_epoch: Option<[u8; 32]>,
}

impl ArtifactActivationPolicy {
    /// Construct a local policy. Time and the accepted epoch must come from a
    /// trusted local source, never from the remote training job.
    pub fn new(
        minimum_release_epoch: u64,
        runtime_version: u32,
        now_unix_ms: u64,
        expected_feature_schema_digest: [u8; 32],
    ) -> Result<Self, ModelError> {
        if minimum_release_epoch == 0 || runtime_version == 0 || now_unix_ms == 0 {
            return Err(ModelError::Malformed("invalid local activation policy"));
        }
        if expected_feature_schema_digest == [0; 32] {
            return Err(ModelError::Malformed("zero local feature schema digest"));
        }
        Ok(Self {
            minimum_release_epoch,
            runtime_version,
            now_unix_ms,
            expected_feature_schema_digest,
            accepted_digest_at_minimum_epoch: None,
        })
    }

    /// Bind the previously persisted artifact digest at the minimum accepted
    /// epoch. Reusing that epoch for different bytes is rejected.
    pub fn with_previous_artifact_digest(mut self, digest: [u8; 32]) -> Result<Self, ModelError> {
        if digest == [0; 32] {
            return Err(ModelError::Malformed("zero previous artifact digest"));
        }
        self.accepted_digest_at_minimum_epoch = Some(digest);
        Ok(self)
    }
}

impl TrustedSignerSet {
    /// Validate and construct a non-empty release-key allowlist.
    pub fn new(keys: Vec<[u8; 32]>) -> Result<Self, ModelError> {
        if keys.is_empty() || keys.len() > MAX_TRUSTED_SIGNERS {
            return Err(ModelError::Malformed("invalid trusted signer count"));
        }
        for (index, key) in keys.iter().enumerate() {
            VerifyingKey::from_bytes(key)
                .map_err(|_| ModelError::Malformed("invalid trusted signer key"))?;
            if keys[..index].contains(key) {
                return Err(ModelError::Malformed("duplicate trusted signer key"));
            }
        }
        Ok(Self { keys })
    }

    fn contains(&self, key: &[u8; 32]) -> bool {
        self.keys.iter().any(|trusted| trusted == key)
    }
}

/// Structurally valid signed envelope awaiting local trust verification.
#[derive(Debug, Clone)]
pub struct SignedModelArtifact {
    candidate: Vec<u8>,
    signer_public_key: [u8; 32],
    signature: [u8; 64],
}

impl SignedModelArtifact {
    /// Assemble a signed envelope from a canonical candidate and an externally
    /// produced signature. This constructor does not grant activation rights.
    pub fn new(
        candidate: &ModelArtifact,
        signer_public_key: [u8; 32],
        signature: [u8; 64],
    ) -> Result<Self, ModelError> {
        VerifyingKey::from_bytes(&signer_public_key)
            .map_err(|_| ModelError::Malformed("invalid signer key"))?;
        Ok(Self {
            candidate: candidate.encode()?,
            signer_public_key,
            signature,
        })
    }

    /// Encode the transport envelope.
    pub fn encode(&self) -> Result<Vec<u8>, ModelError> {
        let total = SIGNED_HEADER_BYTES
            .checked_add(self.candidate.len())
            .ok_or(ModelError::Malformed("signed length overflow"))?;
        if total > MAX_ARTIFACT_BYTES {
            return Err(ModelError::Malformed("signed artifact exceeds limit"));
        }
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&SIGNED_ARTIFACT_MAGIC);
        push_u64(&mut bytes, self.candidate.len())?;
        bytes.extend_from_slice(&self.signer_public_key);
        bytes.extend_from_slice(&self.signature);
        bytes.extend_from_slice(&self.candidate);
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<BorrowedSignedArtifact<'_>, ModelError> {
        if bytes.len() <= SIGNED_HEADER_BYTES || bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(ModelError::Malformed("invalid signed envelope length"));
        }
        if bytes[..8] != SIGNED_ARTIFACT_MAGIC {
            return Err(ModelError::Malformed("bad signed artifact magic"));
        }
        let candidate_len = usize::try_from(u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .map_err(|_| ModelError::Malformed("candidate length"))?,
        ))
        .map_err(|_| ModelError::Malformed("candidate length overflow"))?;
        if candidate_len == 0 || SIGNED_HEADER_BYTES.checked_add(candidate_len) != Some(bytes.len())
        {
            return Err(ModelError::Malformed("trailing or truncated signed bytes"));
        }
        let signer_public_key = bytes[16..48]
            .try_into()
            .map_err(|_| ModelError::Malformed("signer key"))?;
        let signature = bytes[48..SIGNED_HEADER_BYTES]
            .try_into()
            .map_err(|_| ModelError::Malformed("signature"))?;
        Ok(BorrowedSignedArtifact {
            candidate: &bytes[SIGNED_HEADER_BYTES..],
            signer_public_key,
            signature,
        })
    }
}

/// Borrowed envelope used so an unauthenticated candidate is not copied into
/// a second, potentially 256 MiB allocation before its signature is checked.
struct BorrowedSignedArtifact<'a> {
    candidate: &'a [u8],
    signer_public_key: [u8; 32],
    signature: [u8; 64],
}

/// Signature-authenticated artifact permitted to reach a backend decoder.
///
/// Its fields are private and the only constructor verifies both local signer
/// policy and a domain-separated Ed25519 signature before parsing the manifest.
#[derive(Debug, Clone)]
pub struct VerifiedModelArtifact {
    artifact: ModelArtifact,
    signer_public_key: [u8; 32],
    envelope_digest: [u8; 32],
}

impl VerifiedModelArtifact {
    /// Verify a signed envelope against a local release-key allowlist.
    pub fn decode_and_verify(
        encoded: &[u8],
        trusted: &TrustedSignerSet,
    ) -> Result<Self, ModelError> {
        let signed = SignedModelArtifact::decode(encoded)?;
        if !trusted.contains(&signed.signer_public_key) {
            return Err(ModelError::UntrustedSigner);
        }
        let key = VerifyingKey::from_bytes(&signed.signer_public_key)
            .map_err(|_| ModelError::InvalidSignature)?;
        let signature = Signature::from_bytes(&signed.signature);
        let message = signing_message(signed.candidate)?;
        key.verify_strict(&message, &signature)
            .map_err(|_| ModelError::InvalidSignature)?;

        // Manifest parsing and backend-record exposure occur only after the
        // cryptographic gate above has succeeded.
        let artifact = ModelArtifact::decode(signed.candidate)?;
        Ok(Self {
            artifact,
            signer_public_key: signed.signer_public_key,
            envelope_digest: *blake3::hash(encoded).as_bytes(),
        })
    }

    /// Authenticated, validated manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        self.artifact.manifest()
    }

    /// Key that authenticated this envelope.
    #[must_use]
    pub const fn signer_public_key(&self) -> [u8; 32] {
        self.signer_public_key
    }

    /// BLAKE3 identity of the complete signed envelope.
    #[must_use]
    pub const fn envelope_digest(&self) -> [u8; 32] {
        self.envelope_digest
    }

    /// Apply local policy and return the sole backend-decodable wrapper.
    pub fn activate(
        self,
        policy: &ArtifactActivationPolicy,
    ) -> Result<ActivatedModelArtifact, ModelError> {
        let manifest = self.manifest();
        if manifest.release_epoch < policy.minimum_release_epoch {
            return Err(ModelError::ActivationPolicy("release rollback"));
        }
        if manifest.release_epoch == policy.minimum_release_epoch
            && policy
                .accepted_digest_at_minimum_epoch
                .is_some_and(|digest| digest != self.envelope_digest)
        {
            return Err(ModelError::ActivationPolicy(
                "release epoch digest conflict",
            ));
        }
        if policy.runtime_version < manifest.minimum_runtime_version
            || policy.runtime_version > manifest.maximum_runtime_version
        {
            return Err(ModelError::ActivationPolicy("runtime incompatibility"));
        }
        if manifest
            .expires_at_unix_ms
            .is_some_and(|expires| policy.now_unix_ms >= expires)
        {
            return Err(ModelError::ActivationPolicy("artifact expired"));
        }
        if manifest.feature_schema_digest != policy.expected_feature_schema_digest {
            return Err(ModelError::ActivationPolicy("feature schema mismatch"));
        }
        Ok(ActivatedModelArtifact { verified: self })
    }
}

/// Signature- and policy-verified artifact permitted to reach Burn.
///
/// The field is private; construction requires both Ed25519 verification and
/// local activation-policy validation.
#[derive(Debug, Clone)]
pub struct ActivatedModelArtifact {
    verified: VerifiedModelArtifact,
}

impl ActivatedModelArtifact {
    /// Authenticated and policy-compatible manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        self.verified.manifest()
    }

    /// Authenticated backend record bytes.
    #[cfg(feature = "model")]
    pub(crate) fn weights(&self) -> &[u8] {
        self.verified.artifact.weights()
    }

    /// BLAKE3 identity of the complete signed envelope.
    #[must_use]
    pub const fn envelope_digest(&self) -> [u8; 32] {
        self.verified.envelope_digest()
    }
}

struct CandidateSections<'a> {
    manifest: &'a [u8],
    weights: &'a [u8],
}

fn candidate_sections(bytes: &[u8]) -> Result<CandidateSections<'_>, ModelError> {
    if bytes.len() < 21 || bytes.len() > MAX_ARTIFACT_BYTES - SIGNED_HEADER_BYTES {
        return Err(ModelError::Malformed("invalid candidate envelope length"));
    }
    if bytes[..8] != ARTIFACT_MAGIC {
        return Err(ModelError::Malformed("bad candidate magic"));
    }
    let manifest_len = usize::try_from(u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| ModelError::Malformed("manifest length"))?,
    ))
    .map_err(|_| ModelError::Malformed("manifest length overflow"))?;
    let weights_len = usize::try_from(u64::from_le_bytes(
        bytes[12..20]
            .try_into()
            .map_err(|_| ModelError::Malformed("weights length"))?,
    ))
    .map_err(|_| ModelError::Malformed("weights length overflow"))?;
    if manifest_len == 0 || manifest_len > MAX_MANIFEST_BYTES || weights_len == 0 {
        return Err(ModelError::Malformed("invalid candidate section length"));
    }
    let manifest_end = 20usize
        .checked_add(manifest_len)
        .ok_or(ModelError::Malformed("manifest end overflow"))?;
    let expected_end = manifest_end
        .checked_add(weights_len)
        .ok_or(ModelError::Malformed("weights end overflow"))?;
    if expected_end != bytes.len() {
        return Err(ModelError::Malformed(
            "trailing or truncated candidate bytes",
        ));
    }
    Ok(CandidateSections {
        manifest: &bytes[20..manifest_end],
        weights: &bytes[manifest_end..],
    })
}

fn signing_message(candidate: &[u8]) -> Result<Vec<u8>, ModelError> {
    let sections = candidate_sections(candidate)?;
    let candidate_digest = blake3::hash(candidate);
    let actual_weights_digest = blake3::hash(sections.weights);
    let mut message = Vec::with_capacity(SIGNING_DOMAIN.len() + 64);
    message.extend_from_slice(SIGNING_DOMAIN);
    message.extend_from_slice(candidate_digest.as_bytes());
    message.extend_from_slice(actual_weights_digest.as_bytes());
    Ok(message)
}

fn encode_config(bytes: &mut Vec<u8>, config: &ForecastModelConfig) -> Result<(), ModelError> {
    for value in [
        config.context_len,
        config.horizon,
        config.patch_len,
        config.patch_stride,
        config.d_model,
        config.layers,
        config.temporal_kernel,
        config.variate_heads,
        config.ff_width,
        config.horizon_rank,
        config.max_variates,
    ] {
        push_usize(bytes, value)?;
    }
    bytes.extend_from_slice(&config.dropout.to_bits().to_le_bytes());
    for quantile in config.quantiles {
        bytes.extend_from_slice(&quantile.to_bits().to_le_bytes());
    }
    push_usize(bytes, config.descriptor_width)?;
    push_usize(bytes, config.time_width)?;
    Ok(())
}

fn decode_config(reader: &mut Reader<'_>) -> Result<ForecastModelConfig, ModelError> {
    let context_len = reader.usize()?;
    let horizon = reader.usize()?;
    let patch_len = reader.usize()?;
    let patch_stride = reader.usize()?;
    let d_model = reader.usize()?;
    let layers = reader.usize()?;
    let temporal_kernel = reader.usize()?;
    let variate_heads = reader.usize()?;
    let ff_width = reader.usize()?;
    let horizon_rank = reader.usize()?;
    let max_variates = reader.usize()?;
    let dropout = f64::from_bits(reader.u64()?);
    let mut quantiles = [0.0; QUANTILE_COUNT];
    for quantile in &mut quantiles {
        *quantile = f32::from_bits(reader.u32()?);
    }
    let descriptor_width = reader.usize()?;
    let time_width = reader.usize()?;
    Ok(ForecastModelConfig {
        context_len,
        horizon,
        patch_len,
        patch_stride,
        d_model,
        layers,
        temporal_kernel,
        variate_heads,
        ff_width,
        horizon_rank,
        max_variates,
        dropout,
        quantiles,
        descriptor_width,
        time_width,
    })
}

fn push_text(bytes: &mut Vec<u8>, text: &str) -> Result<(), ModelError> {
    let len = u16::try_from(text.len()).map_err(|_| ModelError::Malformed("text too long"))?;
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
    Ok(())
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) -> Result<(), ModelError> {
    let value = u32::try_from(value).map_err(|_| ModelError::Malformed("u32 length overflow"))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_u64(bytes: &mut Vec<u8>, value: usize) -> Result<(), ModelError> {
    let value = u64::try_from(value).map_err(|_| ModelError::Malformed("u64 length overflow"))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) -> Result<(), ModelError> {
    push_u64(bytes, value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(ModelError::Malformed("manifest cursor overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ModelError::Malformed("truncated manifest"))?
            .try_into()
            .map_err(|_| ModelError::Malformed("manifest field width"))?;
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn usize(&mut self) -> Result<usize, ModelError> {
        usize::try_from(self.u64()?).map_err(|_| ModelError::Malformed("usize overflow"))
    }

    fn text(&mut self, max: usize) -> Result<String, ModelError> {
        let len = usize::from(self.u16()?);
        if len == 0 || len > max {
            return Err(ModelError::Malformed("invalid text length"));
        }
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ModelError::Malformed("text end overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ModelError::Malformed("truncated text"))?;
        self.offset = end;
        std::str::from_utf8(value)
            .map(str::to_owned)
            .map_err(|_| ModelError::Malformed("text is not UTF-8"))
    }

    fn boolean(&mut self) -> Result<bool, ModelError> {
        match self.take::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ModelError::Malformed("invalid boolean")),
        }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer, SigningKey};

    use super::*;

    fn artifact() -> ModelArtifact {
        let weights = b"independent-test-record".to_vec();
        let manifest = ArtifactManifest {
            schema_version: 1,
            architecture: "ruview-factorized-forecast-mixer-v1".to_owned(),
            parameter_count: crate::TINY_PARAMETER_COUNT,
            config: ForecastModelConfig::tiny_ci(),
            feature_schema_digest: [1; 32],
            training_manifest_digest: [2; 32],
            weights_digest: *blake3::hash(&weights).as_bytes(),
            seed: 7,
            release_epoch: 4,
            minimum_runtime_version: 1,
            maximum_runtime_version: 1,
            expires_at_unix_ms: Some(2_000_000),
            build_id: "unit-test".to_owned(),
            teacher_outputs_used: false,
            independently_implemented: true,
        };
        ModelArtifact::new(manifest, weights).unwrap()
    }

    fn signed() -> (Vec<u8>, TrustedSignerSet) {
        let artifact = artifact();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let signature = signing_key
            .sign(&artifact.signing_message().unwrap())
            .to_bytes();
        let public_key = signing_key.verifying_key().to_bytes();
        let envelope = SignedModelArtifact::new(&artifact, public_key, signature)
            .unwrap()
            .encode()
            .unwrap();
        (envelope, TrustedSignerSet::new(vec![public_key]).unwrap())
    }

    #[test]
    fn candidate_round_trip_is_canonical() {
        let artifact = artifact();
        let bytes = artifact.encode().unwrap();
        let decoded = ModelArtifact::decode(&bytes).unwrap();
        assert_eq!(decoded.manifest(), artifact.manifest());
        assert_eq!(decoded.weights(), artifact.weights());
        assert_eq!(decoded.encode().unwrap(), bytes);
    }

    #[test]
    fn signed_envelope_requires_an_allowlisted_valid_signature() {
        let (bytes, trusted) = signed();
        let verified = VerifiedModelArtifact::decode_and_verify(&bytes, &trusted).unwrap();
        assert_eq!(verified.manifest(), artifact().manifest());

        let stranger = SigningKey::from_bytes(&[9; 32]).verifying_key().to_bytes();
        let untrusted = TrustedSignerSet::new(vec![stranger]).unwrap();
        assert!(matches!(
            VerifiedModelArtifact::decode_and_verify(&bytes, &untrusted),
            Err(ModelError::UntrustedSigner)
        ));
    }

    #[test]
    fn signed_tampering_fails_before_candidate_activation() {
        let (mut bytes, trusted) = signed();
        *bytes.last_mut().unwrap() ^= 0x55;
        assert!(matches!(
            VerifiedModelArtifact::decode_and_verify(&bytes, &trusted),
            Err(ModelError::InvalidSignature)
        ));
    }

    #[test]
    fn zero_identity_and_teacher_declaration_are_rejected() {
        let mut manifest = artifact().manifest().clone();
        manifest.feature_schema_digest = [0; 32];
        assert!(matches!(manifest.validate(), Err(ModelError::Malformed(_))));

        let mut manifest = artifact().manifest().clone();
        manifest.teacher_outputs_used = true;
        assert!(matches!(manifest.validate(), Err(ModelError::CleanRoom)));
    }

    #[test]
    fn malformed_lengths_fail_closed() {
        let (mut bytes, trusted) = signed();
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            VerifiedModelArtifact::decode_and_verify(&bytes, &trusted),
            Err(ModelError::Malformed(_))
        ));
    }

    #[test]
    fn verified_artifact_still_obeys_local_rollback_and_expiry_policy() {
        let (bytes, trusted) = signed();
        let verified = VerifiedModelArtifact::decode_and_verify(&bytes, &trusted).unwrap();
        let rollback = ArtifactActivationPolicy::new(5, 1, 1_000_000, [1; 32]).unwrap();
        assert!(matches!(
            verified.clone().activate(&rollback),
            Err(ModelError::ActivationPolicy("release rollback"))
        ));
        let expired = ArtifactActivationPolicy::new(4, 1, 2_000_000, [1; 32]).unwrap();
        assert!(matches!(
            verified.clone().activate(&expired),
            Err(ModelError::ActivationPolicy("artifact expired"))
        ));
        let conflict = ArtifactActivationPolicy::new(4, 1, 1_000_000, [1; 32])
            .unwrap()
            .with_previous_artifact_digest([9; 32])
            .unwrap();
        assert!(matches!(
            verified.activate(&conflict),
            Err(ModelError::ActivationPolicy(
                "release epoch digest conflict"
            ))
        ));
    }
}
