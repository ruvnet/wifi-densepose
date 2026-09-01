//! Closed-origin fal.ai queue client for synthetic-only pretraining.

use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE},
    Client, Request, StatusCode,
};
use ruview_forecast_core::{CanonicalDigest, TrainSpec};
use serde::{
    de::{DeserializeOwned, Error as _, IgnoredAny},
    Deserialize, Deserializer, Serialize,
};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    time::{Duration, Instant},
};
use thiserror::Error;
use url::Url;

use crate::{
    artifact::{ArtifactDescriptor, ArtifactError, ArtifactKind, ArtifactStore, FAL_ARTIFACT_ROOT},
    config::{
        JobId, ModelProfile, OptimizerSpec, Sha256Digest, SyntheticDatasetSpec,
        MAX_OPTIMIZER_STEPS, MAX_TRAINING_REQUEST_BYTES, MAX_WALL_TIME_SECONDS,
    },
};

const QUEUE_ORIGIN: &str = "https://queue.fal.run";
const API_ORIGIN: &str = "https://api.fal.ai";
const MAX_RESPONSE: usize = 1024 * 1024;
const MAX_BODY: usize = 64 * 1024;
const MAX_HOSTED_LIFETIME_MS: u64 = 24 * 60 * 60 * 1_000;
const QUEUE_START_ALLOWANCE_SECONDS: u64 = 30 * 60;
const QUEUE_START_ALLOWANCE_MS: u64 = QUEUE_START_ALLOWANCE_SECONDS * 1_000;
const RESULT_DOWNLOAD_GRACE_MS: u64 = 15 * 60 * 1_000;
// The Direct Server request timeout is 3,600 seconds. Keep five minutes outside
// the job's own wall/billable reservation for response handoff and worker
// cleanup instead of letting application work consume the provider deadline.
const MAX_FAL_JOB_BUDGET_SECONDS: u64 = 55 * 60;
const COMPILED_WORKER_BUILD_ID: Option<&str> = option_env!("RUVIEW_WORKER_BUILD_ID");
const COMPILED_BUILD_MANIFEST_SHA256: Option<&str> = option_env!("RUVIEW_BUILD_MANIFEST_SHA256");

/// Protocol error. Provider bodies are never retained.
#[derive(Debug, Error)]
pub enum FalError {
    /// Invalid API key.
    #[error("invalid fal authentication key")]
    InvalidKey,
    /// Invalid trusted app configuration.
    #[error("invalid fal app")]
    InvalidApp,
    /// Invalid provider request id.
    #[error("invalid fal request id")]
    InvalidRequestId,
    /// Provider URL escaped the expected route.
    #[error("unexpected fal operation URL")]
    UnexpectedOperationUrl,
    /// Hosted recipe or reservation failed validation.
    #[error("hosted synthetic plan rejected: {0}")]
    InvalidHostedPlan(&'static str),
    /// Idempotent transport failed.
    #[error("fal transport failed")]
    Transport(#[source] reqwest::Error),
    /// Submission outcome is ambiguous and must be reconciled, never retried.
    #[error(
        "fal submission is remote-unknown; reconcile request {request_digest}, job {job_digest}"
    )]
    RemoteUnknown {
        /// Digest of the exact redacted request.
        request_digest: Sha256Digest,
        /// Fresh opaque hosted job namespace.
        job_digest: Sha256Digest,
    },
    /// Provider HTTP failure.
    #[error("fal returned HTTP {0}")]
    Http(StatusCode),
    /// Bounded response/body limit exceeded.
    #[error("fal payload exceeded its byte cap")]
    TooLarge,
    /// Malformed bounded response.
    #[error("invalid fal response")]
    InvalidResponse,
    /// Poll deadline elapsed.
    #[error("fal polling deadline exceeded")]
    PollTimeout,
    /// The digest-bound provider artifact handoff window elapsed.
    #[error("fal artifact handoff window expired")]
    ArtifactsExpired,
    /// Artifact verification failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// Secret key whose debug form is always redacted.
pub struct FalKey(HeaderValue);
impl FalKey {
    /// Builds `Authorization: Key …`.
    pub fn new(value: impl AsRef<str>) -> Result<Self, FalError> {
        let value = value.as_ref();
        if value.is_empty() || value.len() > 4096 || !value.contains(':') {
            return Err(FalError::InvalidKey);
        }
        Ok(Self(
            HeaderValue::from_str(&format!("Key {value}")).map_err(|_| FalError::InvalidKey)?,
        ))
    }
}
impl fmt::Debug for FalKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FalKey([REDACTED])")
    }
}

/// Process-configured approved app. It is not a request DTO.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FalApp {
    owner: String,
    app: String,
}
impl FalApp {
    /// Parses exactly `owner/app` using safe segments.
    pub fn new(value: impl AsRef<str>) -> Result<Self, FalError> {
        let parts: Vec<_> = value.as_ref().split('/').collect();
        if parts.len() != 2 || parts.iter().any(|part| !safe_app_segment(part)) {
            return Err(FalError::InvalidApp);
        }
        Ok(Self {
            owner: parts[0].into(),
            app: parts[1].into(),
        })
    }
    fn route(&self, origin: &Url, tail: &[&str]) -> Url {
        let mut url = origin.clone();
        url.path_segments_mut()
            .expect("HTTP URL")
            .extend([self.owner.as_str(), self.app.as_str()])
            .extend(tail.iter().copied());
        url
    }
}
impl fmt::Display for FalApp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.owner, self.app)
    }
}

/// Allowlisted worker-side generator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostedGeneratorProfile {
    /// RuView coupled seasonal recipe v1.
    RuViewCoupledV1,
}

/// Bounded optimizer subset sent to fal.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedOptimizer {
    /// Epoch count.
    pub epochs: u16,
    /// Batch size.
    pub batch_size: u16,
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Gradient norm cap.
    pub gradient_clip_norm: f64,
    /// Model seed.
    pub seed: u64,
}
impl From<&OptimizerSpec> for HostedOptimizer {
    fn from(v: &OptimizerSpec) -> Self {
        Self {
            epochs: v.epochs,
            batch_size: v.batch_size,
            learning_rate: v.learning_rate,
            weight_decay: v.weight_decay,
            gradient_clip_norm: v.gradient_clip_norm,
            seed: v.seed,
        }
    }
}

/// Cost provenance; submission never calls an operator cap “measured”.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HostedCostBasis {
    /// Explicit operator ceiling, not a provider estimate.
    UnmeasuredOperatorCap,
}

/// Operator-approved compute and spend reservation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedBudget {
    /// Maximum optimizer steps.
    pub max_optimizer_steps: u64,
    /// Worker deadline.
    pub max_wall_time_seconds: u64,
    /// Maximum provider billable seconds.
    pub max_billable_seconds: u64,
    /// Explicit spend ceiling in micro-USD.
    pub max_micro_usd: u64,
    /// Maximum candidate bytes.
    pub max_artifact_bytes: u64,
    /// Conservative peak worker memory reservation.
    pub max_memory_bytes: u64,
    /// Honest price provenance.
    pub cost_basis: HostedCostBasis,
}

/// Synthetic-only fal wire contract. Its closed fields cannot represent a
/// customer dataset, path, policy object, or RuView identity.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedSyntheticPayload {
    /// Protocol version.
    pub version: u16,
    /// Fresh provider-only random namespace digest.
    pub job_digest: Sha256Digest,
    /// Hash of this redacted request.
    pub request_digest: Sha256Digest,
    /// Fixed model profile.
    pub model_profile: ModelProfile,
    /// Fixed generator profile.
    pub generator_profile: HostedGeneratorProfile,
    /// Exact recipe digest.
    pub generator_recipe_digest: CanonicalDigest,
    /// Generated windows.
    pub windows: u32,
    /// Coupled variates.
    pub variates: u16,
    /// Missing rate, permille.
    pub missing_per_mille: u16,
    /// Generator seed.
    pub generator_seed: u64,
    /// Optimizer settings.
    pub optimizer: HostedOptimizer,
    /// Reserved limits.
    pub budget: HostedBudget,
    /// Expected immutable worker build.
    pub worker_build_id: String,
    /// Cargo.lock/container provenance digest.
    pub build_manifest_digest: Sha256Digest,
    /// Absolute UTC expiry enforced by both submitter and worker.
    pub expires_at_ms: u64,
}
impl HostedSyntheticPayload {
    /// Revalidates decoded bytes and expiry before device allocation.
    pub fn validate_at(&self, now_ms: u64) -> Result<(), FalError> {
        self.validate_integrity()?;
        self.validate_expiry(now_ms, QUEUE_START_ALLOWANCE_MS)
    }

    fn validate_expiry(&self, now_ms: u64, queue_allowance_ms: u64) -> Result<(), FalError> {
        let required_lifetime_ms = self
            .budget
            .max_billable_seconds
            .checked_mul(1_000)
            .and_then(|value| value.checked_add(queue_allowance_ms))
            .and_then(|value| value.checked_add(RESULT_DOWNLOAD_GRACE_MS))
            .ok_or(FalError::InvalidHostedPlan("expiry overflow"))?;
        let minimum_expiry = now_ms
            .checked_add(required_lifetime_ms)
            .ok_or(FalError::InvalidHostedPlan("expiry overflow"))?;
        let maximum_expiry = now_ms
            .checked_add(MAX_HOSTED_LIFETIME_MS)
            .ok_or(FalError::InvalidHostedPlan("expiry overflow"))?;
        if now_ms == 0 || self.expires_at_ms < minimum_expiry || self.expires_at_ms > maximum_expiry
        {
            return Err(FalError::InvalidHostedPlan("expired or excessive lifetime"));
        }
        Ok(())
    }

    /// Revalidates the payload against the immutable identity compiled into
    /// the deployed worker image.
    pub fn validate_for_worker(&self, now_ms: u64) -> Result<(), FalError> {
        self.validate_integrity()?;
        self.validate_expiry(now_ms, 0)?;
        let worker_build_id = COMPILED_WORKER_BUILD_ID.ok_or(FalError::InvalidHostedPlan(
            "worker build identity not compiled",
        ))?;
        let manifest = COMPILED_BUILD_MANIFEST_SHA256.ok_or(FalError::InvalidHostedPlan(
            "build manifest identity not compiled",
        ))?;
        let manifest = Sha256Digest::from_hex(manifest)
            .map_err(|_| FalError::InvalidHostedPlan("compiled build digest"))?;
        self.validate_worker_identity(worker_build_id, manifest)
    }

    fn validate_integrity(&self) -> Result<(), FalError> {
        if self.version != 1
            || self.worker_build_id.is_empty()
            || self.worker_build_id.len() > 128
            || !self
                .worker_build_id
                .bytes()
                .all(|b| (0x21..=0x7e).contains(&b))
        {
            return Err(FalError::InvalidHostedPlan("version/build"));
        }
        let generator = SyntheticDatasetSpec {
            windows: self.windows,
            variates: self.variates,
            missing_per_mille: self.missing_per_mille,
            seed: self.generator_seed,
        };
        generator
            .validate()
            .map_err(|_| FalError::InvalidHostedPlan("generator"))?;
        if generator.canonical_digest() != self.generator_recipe_digest {
            return Err(FalError::InvalidHostedPlan("recipe digest"));
        }
        let optimizer = OptimizerSpec {
            epochs: self.optimizer.epochs,
            batch_size: self.optimizer.batch_size,
            learning_rate: self.optimizer.learning_rate,
            weight_decay: self.optimizer.weight_decay,
            gradient_clip_norm: self.optimizer.gradient_clip_norm,
            checkpoint_every_epochs: self.optimizer.epochs,
            seed: self.optimizer.seed,
        };
        optimizer
            .validate()
            .map_err(|_| FalError::InvalidHostedPlan("optimizer"))?;
        let planned = u64::from(self.windows)
            .div_ceil(u64::from(self.optimizer.batch_size))
            .checked_mul(u64::from(self.optimizer.epochs))
            .ok_or(FalError::InvalidHostedPlan("steps overflow"))?;
        let b = &self.budget;
        if b.max_optimizer_steps == 0
            || b.max_optimizer_steps > MAX_OPTIMIZER_STEPS
            || planned > b.max_optimizer_steps
            || b.max_wall_time_seconds == 0
            || b.max_wall_time_seconds > MAX_WALL_TIME_SECONDS
            || b.max_wall_time_seconds > MAX_FAL_JOB_BUDGET_SECONDS
            || b.max_billable_seconds < b.max_wall_time_seconds
            || b.max_billable_seconds > MAX_WALL_TIME_SECONDS
            || b.max_billable_seconds > MAX_FAL_JOB_BUDGET_SECONDS
            || b.max_micro_usd == 0
            || b.max_artifact_bytes == 0
            || b.max_artifact_bytes > ruview_forecast_model::MAX_ARTIFACT_BYTES as u64
            || !(256 * 1024 * 1024..=96 * 1024 * 1024 * 1024).contains(&b.max_memory_bytes)
        {
            return Err(FalError::InvalidHostedPlan("budget"));
        }
        validate_memory_budget(self)?;
        if payload_digest(self) != self.request_digest {
            return Err(FalError::InvalidHostedPlan("request digest"));
        }
        Ok(())
    }

    fn validate_worker_identity(
        &self,
        expected_worker_build_id: &str,
        expected_build_manifest_digest: Sha256Digest,
    ) -> Result<(), FalError> {
        if self.worker_build_id != expected_worker_build_id
            || self.build_manifest_digest != expected_build_manifest_digest
        {
            return Err(FalError::InvalidHostedPlan(
                "worker build identity mismatch",
            ));
        }
        Ok(())
    }
}

/// Non-serializable just-in-time reservation.
#[derive(Debug)]
pub struct ReservedSyntheticSubmission {
    payload: HostedSyntheticPayload,
}
impl ReservedSyntheticSubmission {
    /// Builds a redacted plan from a core-validated synthetic fal TrainSpec.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        train: &TrainSpec,
        generator: &SyntheticDatasetSpec,
        model_profile: ModelProfile,
        optimizer: &OptimizerSpec,
        budget: HostedBudget,
        worker_build_id: String,
        build_manifest_digest: Sha256Digest,
        expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<Self, FalError> {
        let contract = train
            .require_fal_synthetic_contract()
            .map_err(|_| FalError::InvalidHostedPlan("local-only TrainSpec"))?;
        if contract.generator_recipe_digest() != generator.canonical_digest()
            || contract.generator_seed() != generator.seed
            || train.dataset_digest() != generator.canonical_digest()
        {
            return Err(FalError::InvalidHostedPlan("TrainSpec recipe"));
        }
        let config = model_profile.config();
        if train.context_length() != config.context_len
            || train.horizon() != config.horizon
            || train.quantiles().values() != config.quantiles
            || usize::from(generator.variates) > config.max_variates
        {
            return Err(FalError::InvalidHostedPlan("model profile"));
        }
        if now_ms == 0 || expires_at_ms <= now_ms {
            return Err(FalError::InvalidHostedPlan("expired reservation"));
        }
        if expires_at_ms > train.policy().retention_until_ms() {
            return Err(FalError::InvalidHostedPlan(
                "reservation exceeds TrainSpec retention",
            ));
        }
        let mut payload = HostedSyntheticPayload {
            version: 1,
            // A fresh random hosted namespace prevents low-entropy local job
            // identifiers from crossing the provider boundary.
            job_digest: Sha256Digest::of_bytes(uuid::Uuid::new_v4().as_bytes()),
            request_digest: Sha256Digest::of_bytes(b"placeholder"),
            model_profile,
            generator_profile: HostedGeneratorProfile::RuViewCoupledV1,
            generator_recipe_digest: generator.canonical_digest(),
            windows: generator.windows,
            variates: generator.variates,
            missing_per_mille: generator.missing_per_mille,
            generator_seed: generator.seed,
            optimizer: HostedOptimizer::from(optimizer),
            budget,
            worker_build_id,
            build_manifest_digest,
            expires_at_ms,
        };
        payload.request_digest = payload_digest(&payload);
        payload.validate_at(now_ms)?;
        Ok(Self { payload })
    }
    fn reverify(&self, now_ms: u64) -> Result<&HostedSyntheticPayload, FalError> {
        self.payload.validate_at(now_ms)?;
        Ok(&self.payload)
    }
}
fn payload_digest(payload: &HostedSyntheticPayload) -> Sha256Digest {
    let mut clone = payload.clone();
    clone.request_digest = Sha256Digest::of_bytes(b"placeholder");
    Sha256Digest::of_bytes(&serde_json::to_vec(&clone).expect("wire serialization"))
}

fn validate_memory_budget(payload: &HostedSyntheticPayload) -> Result<(), FalError> {
    let model = payload.model_profile.config();
    let activation_cells = model
        .activation_cells()
        .map_err(|_| FalError::InvalidHostedPlan("model activation estimate"))?;
    let forward_cells = activation_cells
        .checked_mul(usize::from(payload.optimizer.batch_size))
        .ok_or(FalError::InvalidHostedPlan("activation estimate overflow"))?;
    if forward_cells > ruview_forecast_model::MAX_CONFIG_ACTIVATION_CELLS {
        return Err(FalError::InvalidHostedPlan(
            "batch exceeds model activation limit",
        ));
    }
    let forward_multiply_adds = model
        .forward_multiply_adds()
        .map_err(|_| FalError::InvalidHostedPlan("model forward work estimate"))?
        .checked_mul(u64::from(payload.optimizer.batch_size))
        .ok_or(FalError::InvalidHostedPlan(
            "forward work estimate overflow",
        ))?;
    if forward_multiply_adds > ruview_forecast_model::MAX_FORWARD_MULTIPLY_ADDS {
        return Err(FalError::InvalidHostedPlan(
            "batch exceeds model forward work limit",
        ));
    }
    let activation_bytes = u64::try_from(forward_cells)
        .ok()
        .and_then(|cells| cells.checked_mul(4))
        // Forward values, saved backward graph, gradients, and backend
        // bookkeeping. This mirrors the conservative local preflight.
        .and_then(|bytes| bytes.checked_mul(10))
        .ok_or(FalError::InvalidHostedPlan("activation bytes overflow"))?;
    let parameter_bytes = u64::try_from(
        model
            .parameter_count()
            .map_err(|_| FalError::InvalidHostedPlan("model parameter estimate"))?,
    )
    .ok()
    // Parameters, gradients, Adam moments, serialization copy, and backend
    // bookkeeping.
    .and_then(|parameters| parameters.checked_mul(24))
    .ok_or(FalError::InvalidHostedPlan("parameter bytes overflow"))?;
    let required = activation_bytes
        .checked_add(parameter_bytes)
        .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
        .ok_or(FalError::InvalidHostedPlan("memory estimate overflow"))?;
    if required > payload.budget.max_memory_bytes {
        return Err(FalError::InvalidHostedPlan(
            "model exceeds hosted memory reservation",
        ));
    }
    Ok(())
}

/// Validated request id.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(try_from = "String", into = "String")]
pub struct FalRequestId(String);
impl FalRequestId {
    /// Validates one safe URL segment.
    pub fn new(v: impl Into<String>) -> Result<Self, FalError> {
        let v = v.into();
        if safe_segment(&v, 160) {
            Ok(Self(v))
        } else {
            Err(FalError::InvalidRequestId)
        }
    }
    /// String value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for FalRequestId {
    type Error = FalError;
    fn try_from(v: String) -> Result<Self, Self::Error> {
        Self::new(v)
    }
}
impl From<FalRequestId> for String {
    fn from(v: FalRequestId) -> Self {
        v.0
    }
}

/// Reconciliation handle; the app remains client-bound.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FalRequestHandle {
    /// Provider id.
    pub request_id: FalRequestId,
    /// Exact redacted request digest.
    pub request_digest: Sha256Digest,
    /// Fresh opaque hosted job namespace.
    pub job_digest: Sha256Digest,
    /// Expected immutable worker build identity.
    pub worker_build_id: String,
    /// Expected immutable build-manifest digest.
    pub build_manifest_digest: Sha256Digest,
    /// Expected candidate handoff expiry.
    pub artifacts_expire_at_ms: u64,
    /// Immutable cumulative artifact-byte cap from the submitted request.
    pub max_artifact_bytes: u64,
}

impl FalRequestHandle {
    fn require_artifact_handoff_at(&self, now_ms: u64) -> Result<(), FalError> {
        if now_ms == 0 || self.artifacts_expire_at_ms == 0 || now_ms >= self.artifacts_expire_at_ms
        {
            return Err(FalError::ArtifactsExpired);
        }
        Ok(())
    }

    fn require_artifact_budget(&self, outcome: &HostedTrainingOutcome) -> Result<(), FalError> {
        if self.max_artifact_bytes == 0
            || self.max_artifact_bytes > ruview_forecast_model::MAX_ARTIFACT_BYTES as u64
        {
            return Err(FalError::InvalidResponse);
        }
        let total = outcome
            .descriptors()
            .into_iter()
            .try_fold(0_u64, |total, descriptor| {
                total.checked_add(descriptor.size_bytes)
            })
            .ok_or(FalError::InvalidResponse)?;
        if total > self.max_artifact_bytes {
            return Err(FalError::InvalidResponse);
        }
        Ok(())
    }
}

/// Strict untrusted candidate response returned by the hosted worker.
///
/// Exactly four descriptors are accepted. This proves transport integrity and
/// request binding only; release signing still happens locally.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedTrainingOutcome {
    request_digest: Sha256Digest,
    job_digest: Sha256Digest,
    worker_build_id: String,
    build_manifest_digest: Sha256Digest,
    artifacts_expire_at_ms: u64,
    candidate: ArtifactDescriptor,
    manifest: ArtifactDescriptor,
    receipt: ArtifactDescriptor,
    checkpoint: ArtifactDescriptor,
    production_signed: bool,
}

impl HostedTrainingOutcome {
    /// Constructs a strict unsigned hosted result.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_digest: Sha256Digest,
        job_digest: Sha256Digest,
        worker_build_id: String,
        build_manifest_digest: Sha256Digest,
        artifacts_expire_at_ms: u64,
        candidate: ArtifactDescriptor,
        manifest: ArtifactDescriptor,
        receipt: ArtifactDescriptor,
        checkpoint: ArtifactDescriptor,
    ) -> Result<Self, FalError> {
        let outcome = Self {
            request_digest,
            job_digest,
            worker_build_id,
            build_manifest_digest,
            artifacts_expire_at_ms,
            candidate,
            manifest,
            receipt,
            checkpoint,
            production_signed: false,
        };
        outcome.validate()?;
        Ok(outcome)
    }

    /// Exact request digest.
    pub fn request_digest(&self) -> Sha256Digest {
        self.request_digest
    }

    /// Fresh opaque hosted job digest.
    pub fn job_digest(&self) -> Sha256Digest {
        self.job_digest
    }

    /// Immutable worker build identity.
    pub fn worker_build_id(&self) -> &str {
        &self.worker_build_id
    }

    /// Immutable build manifest digest.
    pub fn build_manifest_digest(&self) -> Sha256Digest {
        self.build_manifest_digest
    }

    /// Best-effort worker cleanup deadline for provider-side candidate files.
    pub fn artifacts_expire_at_ms(&self) -> u64 {
        self.artifacts_expire_at_ms
    }

    /// Candidate descriptor.
    pub fn candidate(&self) -> &ArtifactDescriptor {
        &self.candidate
    }

    /// Manifest descriptor.
    pub fn manifest(&self) -> &ArtifactDescriptor {
        &self.manifest
    }

    /// Receipt descriptor.
    pub fn receipt(&self) -> &ArtifactDescriptor {
        &self.receipt
    }

    /// Checkpoint descriptor.
    pub fn checkpoint(&self) -> &ArtifactDescriptor {
        &self.checkpoint
    }

    /// Validates strict kinds, one opaque job namespace, and unsigned status.
    pub fn validate(&self) -> Result<(), FalError> {
        if self.production_signed
            || self.artifacts_expire_at_ms == 0
            || self.worker_build_id.is_empty()
            || self.worker_build_id.len() > 128
            || !self
                .worker_build_id
                .bytes()
                .all(|byte| (0x21..=0x7e).contains(&byte))
        {
            return Err(FalError::InvalidResponse);
        }
        let expected_job = self.job_digest.to_hex();
        for (descriptor, kind) in [
            (&self.candidate, ArtifactKind::Model),
            (&self.manifest, ArtifactKind::Manifest),
            (&self.receipt, ArtifactKind::Receipt),
            (&self.checkpoint, ArtifactKind::Checkpoint),
        ] {
            descriptor.validate()?;
            if descriptor.id.kind != kind || descriptor.id.job_id.as_str() != expected_job {
                return Err(FalError::InvalidResponse);
            }
        }
        Ok(())
    }

    /// Checks this response against the local reconciliation handle.
    pub fn validate_expected(&self, handle: &FalRequestHandle) -> Result<(), FalError> {
        self.validate()?;
        if self.request_digest != handle.request_digest
            || self.job_digest != handle.job_digest
            || self.worker_build_id != handle.worker_build_id
            || self.build_manifest_digest != handle.build_manifest_digest
            || self.artifacts_expire_at_ms != handle.artifacts_expire_at_ms
        {
            return Err(FalError::InvalidResponse);
        }
        handle.require_artifact_budget(self)?;
        Ok(())
    }

    fn descriptors(&self) -> [&ArtifactDescriptor; 4] {
        [
            &self.candidate,
            &self.manifest,
            &self.receipt,
            &self.checkpoint,
        ]
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostedTrainingOutcomeWire {
    request_digest: Sha256Digest,
    job_digest: Sha256Digest,
    worker_build_id: String,
    build_manifest_digest: Sha256Digest,
    artifacts_expire_at_ms: u64,
    candidate: ArtifactDescriptor,
    manifest: ArtifactDescriptor,
    receipt: ArtifactDescriptor,
    checkpoint: ArtifactDescriptor,
    production_signed: bool,
}

impl<'de> Deserialize<'de> for HostedTrainingOutcome {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = HostedTrainingOutcomeWire::deserialize(deserializer)?;
        if wire.production_signed {
            return Err(D::Error::custom(
                "fal candidate cannot be production signed",
            ));
        }
        Self::new(
            wire.request_digest,
            wire.job_digest,
            wire.worker_build_id,
            wire.build_manifest_digest,
            wire.artifacts_expire_at_ms,
            wire.candidate,
            wire.manifest,
            wire.receipt,
            wire.checkpoint,
        )
        .map_err(D::Error::custom)
    }
}

/// Queue lifecycle.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum QueueState {
    /// Queued.
    #[serde(rename = "IN_QUEUE")]
    InQueue,
    /// Running.
    #[serde(rename = "IN_PROGRESS")]
    InProgress,
    /// Terminal.
    #[serde(rename = "COMPLETED")]
    Completed,
}

/// Minimal status; raw provider errors are discarded during decoding.
#[derive(Clone, Debug, Serialize)]
pub struct QueueStatus {
    /// State.
    pub status: QueueState,
    /// Queue position.
    pub queue_position: Option<u64>,
    /// Whether an error field existed.
    pub has_error: bool,
}
#[derive(Deserialize)]
struct StatusWire {
    status: QueueState,
    #[serde(default)]
    queue_position: Option<u64>,
    #[serde(default)]
    error: Option<IgnoredAny>,
}
impl<'de> Deserialize<'de> for QueueStatus {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = StatusWire::deserialize(d)?;
        Ok(Self {
            status: w.status,
            queue_position: w.queue_position,
            has_error: w.error.is_some(),
        })
    }
}
#[derive(Deserialize)]
struct SubmitResponse {
    request_id: String,
    response_url: String,
    status_url: String,
    cancel_url: String,
}
#[derive(Clone, Copy)]
enum Operation {
    Result,
    Status,
    Cancel,
}

/// Retry policy for idempotent calls only.
#[derive(Clone, Copy, Debug)]
pub struct IdempotentRetryPolicy {
    /// Attempts.
    pub max_attempts: u8,
    /// Initial delay.
    pub initial_delay: Duration,
    /// Max delay.
    pub maximum_delay: Duration,
}
impl Default for IdempotentRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_delay: Duration::from_millis(200),
            maximum_delay: Duration::from_secs(2),
        }
    }
}

/// Authenticated client bound to one approved app.
pub struct FalQueueClient {
    client: Client,
    key: FalKey,
    app: FalApp,
    queue: Url,
    api: Url,
    retries: IdempotentRetryPolicy,
}
impl FalQueueClient {
    /// Creates an HTTPS-only redirect-free client.
    pub fn new(key: FalKey, app: FalApp) -> Result<Self, FalError> {
        Self::new_inner(
            key,
            app,
            Url::parse(QUEUE_ORIGIN).unwrap(),
            Url::parse(API_ORIGIN).unwrap(),
            true,
        )
    }
    fn new_inner(
        key: FalKey,
        app: FalApp,
        queue: Url,
        api: Url,
        https_only: bool,
    ) -> Result<Self, FalError> {
        let client = Client::builder()
            .https_only(https_only)
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(FalError::Transport)?;
        Ok(Self {
            client,
            key,
            app,
            queue,
            api,
            retries: Default::default(),
        })
    }
    /// Overrides idempotent retry bounds.
    pub fn with_idempotent_retries(mut self, r: IdempotentRetryPolicy) -> Self {
        self.retries = r;
        self
    }
    /// One-send submission after immediate reservation recheck.
    pub async fn submit(
        &self,
        plan: &ReservedSyntheticSubmission,
    ) -> Result<FalRequestHandle, FalError> {
        let p = plan.reverify(unix_time_millis())?;
        let request = self.build_submit(p)?;
        let response = self
            .client
            .execute(request)
            .await
            .map_err(|_| FalError::RemoteUnknown {
                request_digest: p.request_digest,
                job_digest: p.job_digest,
            })?;
        if !response.status().is_success() {
            return Err(FalError::Http(response.status()));
        }
        let parsed = async {
            let r: SubmitResponse = read_json_limited(response).await?;
            let h = FalRequestHandle {
                request_id: FalRequestId::new(r.request_id)?,
                request_digest: p.request_digest,
                job_digest: p.job_digest,
                worker_build_id: p.worker_build_id.clone(),
                build_manifest_digest: p.build_manifest_digest,
                artifacts_expire_at_ms: p.expires_at_ms,
                max_artifact_bytes: p.budget.max_artifact_bytes,
            };
            validate_result_url(&r.response_url, &self.url(&h, Operation::Result))?;
            validate_url(&r.status_url, &self.url(&h, Operation::Status))?;
            validate_url(&r.cancel_url, &self.url(&h, Operation::Cancel))?;
            Ok::<_, FalError>(h)
        }
        .await;
        parsed.map_err(|_| FalError::RemoteUnknown {
            request_digest: p.request_digest,
            job_digest: p.job_digest,
        })
    }
    /// Retrieves status without logs. Status remains available after artifact
    /// expiry for reconciliation; it grants no result/download authority.
    pub async fn status(&self, h: &FalRequestHandle) -> Result<QueueStatus, FalError> {
        read_json_limited(
            self.send_idempotent(reqwest::Method::GET, self.url(h, Operation::Status))
                .await?,
        )
        .await
    }
    /// Polls until terminal or deadline.
    pub async fn poll_until_complete(
        &self,
        h: &FalRequestHandle,
        interval: Duration,
        deadline: Duration,
    ) -> Result<QueueStatus, FalError> {
        let start = Instant::now();
        loop {
            let s = self.status(h).await?;
            if s.status == QueueState::Completed {
                return Ok(s);
            }
            if start.elapsed() >= deadline {
                return Err(FalError::PollTimeout);
            }
            tokio::time::sleep(interval).await
        }
    }
    /// Retrieves and strictly validates the hosted candidate result.
    pub async fn result(&self, h: &FalRequestHandle) -> Result<HostedTrainingOutcome, FalError> {
        h.require_artifact_handoff_at(unix_time_millis())?;
        let outcome: HostedTrainingOutcome = read_json_limited(
            self.send_idempotent(reqwest::Method::GET, self.url(h, Operation::Result))
                .await?,
        )
        .await?;
        h.require_artifact_handoff_at(unix_time_millis())?;
        outcome.validate_expected(h)?;
        Ok(outcome)
    }
    /// Requests cancellation; ambiguous failure becomes remote-unknown.
    /// Cancellation remains available after artifact expiry for cleanup and
    /// reconciliation, but does not re-open the result handoff window.
    pub async fn cancel(&self, h: &FalRequestHandle) -> Result<(), FalError> {
        match self
            .send_idempotent(reqwest::Method::PUT, self.url(h, Operation::Cancel))
            .await
        {
            Ok(_) => Ok(()),
            Err(FalError::Transport(_)) => Err(FalError::RemoteUnknown {
                request_digest: h.request_digest,
                job_digest: h.job_digest,
            }),
            Err(e) => Err(e),
        }
    }
    /// Downloads to a capability-confined quarantine store and verifies digest.
    async fn download_artifact(
        &self,
        h: &FalRequestHandle,
        d: &ArtifactDescriptor,
        store: &ArtifactStore,
    ) -> Result<ArtifactDescriptor, FalError> {
        h.require_artifact_handoff_at(unix_time_millis())?;
        d.validate()?;
        if d.id.job_id.as_str() != h.job_digest.to_hex() {
            return Err(FalError::InvalidResponse);
        }
        let mut r = self
            .get_artifact_with_consistency_retry(artifact_url(&self.api, d))
            .await?;
        if r.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .is_some_and(|n| n != d.size_bytes)
        {
            return Err(ArtifactError::SizeMismatch {
                expected: d.size_bytes,
                actual: r.content_length().unwrap_or(0),
            }
            .into());
        }
        let cap = usize::try_from(d.size_bytes).map_err(|_| FalError::TooLarge)?;
        let mut bytes = Vec::with_capacity(cap);
        let mut hash = Sha256::new();
        while let Some(c) = r.chunk().await.map_err(FalError::Transport)? {
            h.require_artifact_handoff_at(unix_time_millis())?;
            if bytes.len().checked_add(c.len()).is_none_or(|n| n > cap) {
                return Err(FalError::TooLarge);
            }
            hash.update(&c);
            bytes.extend_from_slice(&c)
        }
        if bytes.len() != cap {
            return Err(ArtifactError::SizeMismatch {
                expected: d.size_bytes,
                actual: bytes.len() as u64,
            }
            .into());
        }
        let actual = Sha256Digest::from_hex(&hex::encode(hash.finalize()))
            .map_err(|_| FalError::InvalidResponse)?;
        if actual != d.sha256 {
            return Err(ArtifactError::DigestMismatch {
                expected: d.sha256,
                actual,
            }
            .into());
        }
        h.require_artifact_handoff_at(unix_time_millis())?;
        let out = store.commit_bytes(&d.id.job_id, d.id.kind, &bytes)?;
        h.require_artifact_handoff_at(unix_time_millis())?;
        if out != *d {
            return Err(ArtifactError::Conflict.into());
        }
        store.verify(&out)?;
        Ok(out)
    }

    /// Downloads and verifies exactly the four descriptors bound to a result.
    pub async fn download_outcome(
        &self,
        handle: &FalRequestHandle,
        outcome: &HostedTrainingOutcome,
        quarantine: &ArtifactStore,
    ) -> Result<Vec<ArtifactDescriptor>, FalError> {
        handle.require_artifact_handoff_at(unix_time_millis())?;
        outcome.validate_expected(handle)?;
        let mut downloaded = Vec::with_capacity(4);
        let result = async {
            for descriptor in outcome.descriptors() {
                downloaded.push(
                    self.download_artifact(handle, descriptor, quarantine)
                        .await?,
                );
            }
            let receipt = downloaded
                .iter()
                .find(|descriptor| descriptor.id.kind == ArtifactKind::Receipt)
                .ok_or(FalError::InvalidResponse)?;
            let receipt_bytes = quarantine.read_bytes(receipt)?;
            let binding: HostedReceiptBinding =
                serde_json::from_slice(&receipt_bytes).map_err(|_| FalError::InvalidResponse)?;
            if binding.upstream_request_digest.as_deref() != Some(&handle.request_digest.to_hex())
                || !binding.candidate_is_untrusted
                || binding.production_signed
            {
                return Err(FalError::InvalidResponse);
            }
            handle.require_artifact_handoff_at(unix_time_millis())?;
            Ok(downloaded)
        }
        .await;
        if result.is_err() {
            let job_id =
                JobId::new(handle.job_digest.to_hex()).map_err(|_| FalError::InvalidResponse)?;
            match quarantine.remove_job_outputs(&job_id) {
                Ok(()) => {}
                Err(ArtifactError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        result
    }
    async fn get_artifact_with_consistency_retry(
        &self,
        url: Url,
    ) -> Result<reqwest::Response, FalError> {
        let attempts = self.retries.max_attempts.max(1);
        let mut delay = self.retries.initial_delay;
        for attempt in 1..=attempts {
            match self
                .client
                .get(url.clone())
                .header(AUTHORIZATION, self.key.0.clone())
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response)
                    if attempt < attempts
                        && (response.status() == StatusCode::NOT_FOUND
                            || retryable(response.status())) => {}
                Ok(response) => return Err(FalError::Http(response.status())),
                Err(error) if attempt < attempts && (error.is_connect() || error.is_timeout()) => {}
                Err(error) => return Err(FalError::Transport(error)),
            }
            tokio::time::sleep(delay).await;
            delay = delay.saturating_mul(2).min(self.retries.maximum_delay);
        }
        unreachable!()
    }
    fn url(&self, h: &FalRequestHandle, op: Operation) -> Url {
        let mut tail = vec!["train", "requests", h.request_id.as_str()];
        match op {
            Operation::Result => {}
            Operation::Status => tail.push("status"),
            Operation::Cancel => tail.push("cancel"),
        }
        self.app.route(&self.queue, &tail)
    }
    fn build_submit(&self, p: &HostedSyntheticPayload) -> Result<Request, FalError> {
        p.validate_integrity()?;
        let body = serde_json::to_vec(p).map_err(|_| FalError::InvalidResponse)?;
        if body.len() > MAX_BODY || body.len() > MAX_TRAINING_REQUEST_BYTES {
            return Err(FalError::TooLarge);
        }
        self.client
            .post(self.app.route(&self.queue, &["train"]))
            .headers(self.headers())
            .body(body)
            .build()
            .map_err(FalError::Transport)
    }
    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, self.key.0.clone());
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert("x-fal-no-retry", HeaderValue::from_static("1"));
        h.insert("x-fal-request-timeout", HeaderValue::from_static("1800"));
        h
    }
    async fn send_idempotent(
        &self,
        method: reqwest::Method,
        url: Url,
    ) -> Result<reqwest::Response, FalError> {
        let attempts = self.retries.max_attempts.max(1);
        let mut delay = self.retries.initial_delay;
        for attempt in 1..=attempts {
            match self
                .client
                .request(method.clone(), url.clone())
                .header(AUTHORIZATION, self.key.0.clone())
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) if retryable(r.status()) && attempt < attempts => {}
                Ok(r) => return Err(FalError::Http(r.status())),
                Err(e) if attempt < attempts && (e.is_connect() || e.is_timeout()) => {}
                Err(e) => return Err(FalError::Transport(e)),
            }
            tokio::time::sleep(delay).await;
            delay = delay.saturating_mul(2).min(self.retries.maximum_delay)
        }
        unreachable!()
    }
}

#[derive(Deserialize)]
struct HostedReceiptBinding {
    upstream_request_digest: Option<String>,
    candidate_is_untrusted: bool,
    production_signed: bool,
}
impl fmt::Debug for FalQueueClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FalQueueClient")
            .field("key", &self.key)
            .field("app", &self.app)
            .finish_non_exhaustive()
    }
}

fn safe_segment(v: &str, max: usize) -> bool {
    !v.is_empty()
        && v.len() <= max
        && v != "."
        && v != ".."
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

fn safe_app_segment(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}
fn retryable(s: StatusCode) -> bool {
    s == StatusCode::REQUEST_TIMEOUT || s == StatusCode::TOO_MANY_REQUESTS || s.is_server_error()
}
fn unix_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
fn validate_url(raw: &str, expected: &Url) -> Result<(), FalError> {
    let p = Url::parse(raw).map_err(|_| FalError::UnexpectedOperationUrl)?;
    if p != *expected
        || !p.username().is_empty()
        || p.password().is_some()
        || p.query().is_some()
        || p.fragment().is_some()
    {
        Err(FalError::UnexpectedOperationUrl)
    } else {
        Ok(())
    }
}
fn validate_result_url(raw: &str, expected: &Url) -> Result<(), FalError> {
    if validate_url(raw, expected).is_ok() {
        return Ok(());
    }
    let mut response = expected.clone();
    response.path_segments_mut().unwrap().push("response");
    validate_url(raw, &response)
}
fn artifact_url(origin: &Url, d: &ArtifactDescriptor) -> Url {
    let mut u = origin.clone();
    let rel = d.id.relative_path();
    u.path_segments_mut()
        .unwrap()
        .extend(["v1", "serverless", "files", "file"])
        .extend(FAL_ARTIFACT_ROOT.trim_start_matches("/data/").split('/'))
        .extend(rel.split('/'));
    u
}
async fn read_json_limited<T: DeserializeOwned>(mut r: reqwest::Response) -> Result<T, FalError> {
    if r.content_length().is_some_and(|n| n > MAX_RESPONSE as u64) {
        return Err(FalError::TooLarge);
    }
    let mut b = Vec::new();
    while let Some(c) = r.chunk().await.map_err(FalError::Transport)? {
        let n = b.len().checked_add(c.len()).ok_or(FalError::TooLarge)?;
        if n > MAX_RESPONSE {
            return Err(FalError::TooLarge);
        }
        b.extend_from_slice(&c)
    }
    serde_json::from_slice(&b).map_err(|_| FalError::InvalidResponse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ArtifactId, ArtifactKind};
    use crate::config::{synthetic_train_spec, JobId};

    fn hosted_budget() -> HostedBudget {
        HostedBudget {
            max_optimizer_steps: 1,
            max_wall_time_seconds: 60,
            max_billable_seconds: 60,
            max_micro_usd: 1,
            max_artifact_bytes: ruview_forecast_model::MAX_ARTIFACT_BYTES as u64,
            max_memory_bytes: 4 * 1024 * 1024 * 1024,
            cost_basis: HostedCostBasis::UnmeasuredOperatorCap,
        }
    }

    fn reserved(now: u64) -> ReservedSyntheticSubmission {
        let generator = SyntheticDatasetSpec {
            windows: 1,
            variates: 2,
            missing_per_mille: 0,
            seed: 7,
        };
        let train = synthetic_train_spec(ModelProfile::TinyCi, &generator, true, now + 3_000_000)
            .expect("synthetic fal spec");
        let optimizer = OptimizerSpec {
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.001,
            weight_decay: 0.0,
            gradient_clip_norm: 1.0,
            checkpoint_every_epochs: 1,
            seed: 9,
        };
        ReservedSyntheticSubmission::reserve(
            &train,
            &generator,
            ModelProfile::TinyCi,
            &optimizer,
            hosted_budget(),
            "worker-test".to_string(),
            Sha256Digest::of_bytes(b"build"),
            now + 3_000_000,
            now,
        )
        .expect("reservation")
    }
    #[test]
    fn fal_app_rejects_path_and_host_confusion() {
        for v in [
            "https://queue.fal.run/o/a",
            "o/../../x",
            "o/a/x",
            "o%2fa/x",
            "o@evil/a",
            "127.0.0.1/a",
            "o\\a",
            "o//a",
        ] {
            assert!(FalApp::new(v).is_err())
        }
        assert_eq!(
            FalApp::new("ruvnet/forecast").unwrap().to_string(),
            "ruvnet/forecast"
        )
    }
    #[test]
    fn fal_urls_require_exact_origin_path_and_no_query() {
        let e = Url::parse("https://queue.fal.run/o/a/train/requests/r/status").unwrap();
        assert!(validate_url(e.as_str(), &e).is_ok());
        for b in [
            "http://queue.fal.run/o/a/train/requests/r/status",
            "https://queue.fal.run.evil/o/a/train/requests/r/status",
            "https://u@queue.fal.run/o/a/train/requests/r/status",
            "https://queue.fal.run/o/a/train/requests/r/status?x=1",
        ] {
            assert!(validate_url(b, &e).is_err())
        }
    }
    #[test]
    fn fal_submit_headers_disable_retry_and_omit_store_io() {
        let c = FalQueueClient::new(
            FalKey::new("id:secret").unwrap(),
            FalApp::new("o/a").unwrap(),
        )
        .unwrap();
        assert!(!c.headers().contains_key("x-fal-store-io"));
        assert_eq!(c.headers()["x-fal-no-retry"], "1");
        assert_eq!(c.headers()["x-fal-request-timeout"], "1800")
    }
    #[test]
    fn hosted_payload_dto_has_no_customer_fields() {
        let s = include_str!("fal.rs");
        let p = s
            .split("pub struct HostedSyntheticPayload")
            .nth(1)
            .unwrap()
            .split("impl HostedSyntheticPayload")
            .next()
            .unwrap();
        for bad in [
            "DataPolicy",
            "dataset_path",
            "dataset_bytes",
            "tenant",
            "account",
            "workspace",
            "site",
            "room",
            "device",
            "session",
            "split_plan",
        ] {
            assert!(!p.contains(bad), "{bad}")
        }
    }
    #[test]
    fn queue_status_discards_provider_error_body() {
        let s: QueueStatus =
            serde_json::from_str(r#"{"status":"COMPLETED","error":{"secret":"never"}}"#).unwrap();
        assert!(s.has_error);
        assert!(!format!("{s:?}").contains("never"))
    }
    #[test]
    fn result_url_accepts_exact_response_suffix() {
        let e = Url::parse("https://queue.fal.run/o/a/train/requests/r").unwrap();
        assert!(
            validate_result_url("https://queue.fal.run/o/a/train/requests/r/response", &e).is_ok()
        )
    }
    #[test]
    fn artifact_file_url_uses_expected_api_path() {
        let d = ArtifactDescriptor {
            id: ArtifactId {
                job_id: JobId::new("job-1").unwrap(),
                kind: ArtifactKind::Model,
            },
            size_bytes: 7,
            sha256: Sha256Digest::of_bytes(b"payload"),
        };
        assert_eq!(
            artifact_url(&Url::parse(API_ORIGIN).unwrap(), &d).as_str(),
            "https://api.fal.ai/v1/serverless/files/file/ruview-forecast/artifacts/job-1/model.mpk"
        )
    }
    #[test]
    fn fal_key_debug_is_redacted() {
        assert_eq!(
            format!("{:?}", FalKey::new("id:secret").unwrap()),
            "FalKey([REDACTED])"
        )
    }
    #[test]
    fn hosted_payload_rejects_app_field() {
        assert!(serde_json::from_str::<HostedSyntheticPayload>(r#"{"app":"evil/app"}"#).is_err())
    }
    #[test]
    fn sha256_digest_changes_with_input() {
        assert_ne!(Sha256Digest::of_bytes(b"x"), Sha256Digest::of_bytes(b"y"))
    }

    #[test]
    fn hosted_expiry_is_on_wire_and_digest_bound() {
        let now = 1_000_000;
        let mut plan = reserved(now);
        let encoded = serde_json::to_string(&plan.payload).unwrap();
        assert!(encoded.contains("expires_at_ms"));
        assert!(plan.payload.validate_at(now).is_ok());
        assert!(plan
            .payload
            .validate_at(plan.payload.expires_at_ms)
            .is_err());
        plan.payload.expires_at_ms += 1;
        assert!(matches!(
            plan.payload.validate_at(now),
            Err(FalError::InvalidHostedPlan("request digest"))
        ));
    }

    #[test]
    fn hosted_reservation_cannot_exceed_source_retention() {
        let now = 1_000_000;
        let retention_until_ms = now + 3_000_000;
        let generator = SyntheticDatasetSpec {
            windows: 1,
            variates: 2,
            missing_per_mille: 0,
            seed: 7,
        };
        let train =
            synthetic_train_spec(ModelProfile::TinyCi, &generator, true, retention_until_ms)
                .expect("synthetic fal spec");
        let optimizer = OptimizerSpec {
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.001,
            weight_decay: 0.0,
            gradient_clip_norm: 1.0,
            checkpoint_every_epochs: 1,
            seed: 9,
        };
        let reserve = |expires_at_ms| {
            ReservedSyntheticSubmission::reserve(
                &train,
                &generator,
                ModelProfile::TinyCi,
                &optimizer,
                hosted_budget(),
                "worker-test".to_string(),
                Sha256Digest::of_bytes(b"build"),
                expires_at_ms,
                now,
            )
        };

        assert!(reserve(retention_until_ms).is_ok());
        assert!(matches!(
            reserve(retention_until_ms + 1),
            Err(FalError::InvalidHostedPlan(
                "reservation exceeds TrainSpec retention"
            ))
        ));
    }

    #[test]
    fn delayed_worker_preserves_download_grace() {
        let now = 4_000_000;
        let plan = reserved(now);
        assert!(plan.payload.validate_at(now).is_ok());
        assert!(plan
            .payload
            .validate_expiry(now + QUEUE_START_ALLOWANCE_MS, 0)
            .is_ok());
        assert!(plan
            .payload
            .validate_expiry(plan.payload.expires_at_ms - RESULT_DOWNLOAD_GRACE_MS + 1, 0,)
            .is_err());
    }

    #[test]
    fn hosted_job_budget_preserves_provider_handoff_margin() {
        let mut plan = reserved(4_000_000);
        plan.payload.budget.max_wall_time_seconds = MAX_FAL_JOB_BUDGET_SECONDS;
        plan.payload.budget.max_billable_seconds = MAX_FAL_JOB_BUDGET_SECONDS;
        plan.payload.request_digest = payload_digest(&plan.payload);
        assert!(plan.payload.validate_integrity().is_ok());

        plan.payload.budget.max_wall_time_seconds = MAX_FAL_JOB_BUDGET_SECONDS + 1;
        plan.payload.budget.max_billable_seconds = MAX_FAL_JOB_BUDGET_SECONDS + 1;
        plan.payload.request_digest = payload_digest(&plan.payload);
        assert!(matches!(
            plan.payload.validate_integrity(),
            Err(FalError::InvalidHostedPlan("budget"))
        ));
    }

    #[test]
    fn artifact_handoff_expiry_boundary_is_fail_closed() {
        let plan = reserved(4_000_000);
        let handle = FalRequestHandle {
            request_id: FalRequestId::new("provider-request").unwrap(),
            request_digest: plan.payload.request_digest,
            job_digest: plan.payload.job_digest,
            worker_build_id: plan.payload.worker_build_id.clone(),
            build_manifest_digest: plan.payload.build_manifest_digest,
            artifacts_expire_at_ms: plan.payload.expires_at_ms,
            max_artifact_bytes: plan.payload.budget.max_artifact_bytes,
        };
        assert!(handle
            .require_artifact_handoff_at(handle.artifacts_expire_at_ms - 1)
            .is_ok());
        assert!(matches!(
            handle.require_artifact_handoff_at(handle.artifacts_expire_at_ms),
            Err(FalError::ArtifactsExpired)
        ));
        assert!(matches!(
            handle.require_artifact_handoff_at(handle.artifacts_expire_at_ms + 1),
            Err(FalError::ArtifactsExpired)
        ));
    }

    #[test]
    fn privacy_hosted_job_namespace_is_random_and_not_local_job_text() {
        let now = 2_000_000;
        let first = reserved(now);
        let second = reserved(now);
        assert_ne!(first.payload.job_digest, second.payload.job_digest);
        let wire = serde_json::to_string(&first.payload).unwrap();
        assert!(!wire.contains("local-job"));
        let reservation_source = include_str!("fal.rs")
            .split("pub fn reserve(")
            .nth(1)
            .unwrap()
            .split("fn reverify")
            .next()
            .unwrap();
        assert!(!reservation_source.contains("JobId"));
    }

    #[test]
    fn auth_worker_identity_must_match_exactly() {
        let plan = reserved(3_000_000);
        assert!(plan
            .payload
            .validate_worker_identity("worker-test", Sha256Digest::of_bytes(b"build"))
            .is_ok());
        assert!(plan
            .payload
            .validate_worker_identity("other", Sha256Digest::of_bytes(b"build"))
            .is_err());
    }

    #[test]
    fn fal_result_rejects_provider_signed_claim_and_wrong_kind() {
        let job_digest = Sha256Digest::of_bytes(b"opaque-job");
        let job = JobId::new(job_digest.to_hex()).unwrap();
        let descriptor = |kind| ArtifactDescriptor {
            id: ArtifactId {
                job_id: job.clone(),
                kind,
            },
            size_bytes: 1,
            sha256: Sha256Digest::of_bytes(kind.filename().as_bytes()),
        };
        let outcome = HostedTrainingOutcome::new(
            Sha256Digest::of_bytes(b"request"),
            job_digest,
            "worker-test".to_string(),
            Sha256Digest::of_bytes(b"build"),
            9_999_999,
            descriptor(ArtifactKind::Model),
            descriptor(ArtifactKind::Manifest),
            descriptor(ArtifactKind::Receipt),
            descriptor(ArtifactKind::Checkpoint),
        )
        .unwrap();
        let handle = FalRequestHandle {
            request_id: FalRequestId::new("request-id").unwrap(),
            request_digest: Sha256Digest::of_bytes(b"request"),
            job_digest,
            worker_build_id: "worker-test".to_string(),
            build_manifest_digest: Sha256Digest::of_bytes(b"build"),
            artifacts_expire_at_ms: 9_999_999,
            max_artifact_bytes: 4,
        };
        assert!(outcome.validate_expected(&handle).is_ok());
        let mut wrong_build = handle.clone();
        wrong_build.worker_build_id = "other-worker".to_string();
        assert!(outcome.validate_expected(&wrong_build).is_err());
        let mut wrong_expiry = handle;
        wrong_expiry.artifacts_expire_at_ms += 1;
        assert!(outcome.validate_expected(&wrong_expiry).is_err());
        let mut wire = serde_json::to_value(outcome).unwrap();
        wire["production_signed"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<HostedTrainingOutcome>(wire).is_err());
    }

    #[test]
    fn fal_result_enforces_cumulative_artifact_budget_boundary() {
        let job_digest = Sha256Digest::of_bytes(b"budgeted-job");
        let job = JobId::new(job_digest.to_hex()).unwrap();
        let descriptor = |kind| ArtifactDescriptor {
            id: ArtifactId {
                job_id: job.clone(),
                kind,
            },
            size_bytes: 1,
            sha256: Sha256Digest::of_bytes(kind.filename().as_bytes()),
        };
        let mut outcome = HostedTrainingOutcome::new(
            Sha256Digest::of_bytes(b"budgeted-request"),
            job_digest,
            "worker-test".to_string(),
            Sha256Digest::of_bytes(b"build"),
            u64::MAX,
            descriptor(ArtifactKind::Model),
            descriptor(ArtifactKind::Manifest),
            descriptor(ArtifactKind::Receipt),
            descriptor(ArtifactKind::Checkpoint),
        )
        .unwrap();
        let handle = FalRequestHandle {
            request_id: FalRequestId::new("request-id").unwrap(),
            request_digest: Sha256Digest::of_bytes(b"budgeted-request"),
            job_digest,
            worker_build_id: "worker-test".to_string(),
            build_manifest_digest: Sha256Digest::of_bytes(b"build"),
            artifacts_expire_at_ms: u64::MAX,
            max_artifact_bytes: 4,
        };

        assert!(outcome.validate_expected(&handle).is_ok());
        outcome.candidate.size_bytes += 1;
        assert!(matches!(
            outcome.validate_expected(&handle),
            Err(FalError::InvalidResponse)
        ));
    }

    #[tokio::test]
    async fn fal_download_rejects_over_budget_without_writing() {
        let job_digest = Sha256Digest::of_bytes(b"over-budget-job");
        let job = JobId::new(job_digest.to_hex()).unwrap();
        let descriptor = |kind, size_bytes| ArtifactDescriptor {
            id: ArtifactId {
                job_id: job.clone(),
                kind,
            },
            size_bytes,
            sha256: Sha256Digest::of_bytes(kind.filename().as_bytes()),
        };
        let outcome = HostedTrainingOutcome::new(
            Sha256Digest::of_bytes(b"over-budget-request"),
            job_digest,
            "worker-test".to_string(),
            Sha256Digest::of_bytes(b"build"),
            u64::MAX,
            descriptor(ArtifactKind::Model, 2),
            descriptor(ArtifactKind::Manifest, 1),
            descriptor(ArtifactKind::Receipt, 1),
            descriptor(ArtifactKind::Checkpoint, 1),
        )
        .unwrap();
        let handle = FalRequestHandle {
            request_id: FalRequestId::new("request-id").unwrap(),
            request_digest: Sha256Digest::of_bytes(b"over-budget-request"),
            job_digest,
            worker_build_id: "worker-test".to_string(),
            build_manifest_digest: Sha256Digest::of_bytes(b"build"),
            artifacts_expire_at_ms: u64::MAX,
            max_artifact_bytes: 4,
        };
        let client = FalQueueClient::new_inner(
            FalKey::new("id:secret").unwrap(),
            FalApp::new("o/a").unwrap(),
            Url::parse("http://127.0.0.1:1/").unwrap(),
            Url::parse("http://127.0.0.1:1/").unwrap(),
            false,
        )
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        let quarantine = ArtifactStore::new(root.path()).unwrap();

        assert!(matches!(
            client
                .download_outcome(&handle, &outcome, &quarantine)
                .await,
            Err(FalError::InvalidResponse)
        ));
        assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
    }

    #[test]
    fn deploy_context_is_git_archive_and_allowlisted() {
        let wrapper = include_str!("../deploy/fal/deploy.py");
        assert!(wrapper.contains("\"archive\""));
        assert!(wrapper.contains("FORECAST_SCOPE"));
        assert!(wrapper.contains("ARCHIVE_SCOPE"));
        assert!(wrapper.contains("ALLOWED_SUFFIXES"));
        assert!(wrapper.contains("deploy/fal/Dockerfile"));
        assert!(wrapper.contains("def self_test()"));
        assert!(wrapper.contains("forecast deployment scope is dirty"));
        assert!(!wrapper.contains("copytree"));
    }
}
