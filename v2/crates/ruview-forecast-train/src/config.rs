//! Typed, bounded configuration accepted by the local and hosted runners.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use ruview_forecast_core::{
    CanonicalDigest, DataPolicy, HoldoutKey, NormalizationPolicy, PrivacyClass, QuantileSet,
    SeriesKey, SourceState, SplitMember, SplitStrategy, TemporalSplitPlan, TimeRange, TrainSpec,
    TrainingDestinationKind,
};
use ruview_forecast_model::ForecastModelConfig;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Maximum accepted serialized training request size.
pub const MAX_TRAINING_REQUEST_BYTES: usize = 1024 * 1024;
/// Maximum accepted corpus size for one invocation. Larger governed corpora
/// must be pre-sharded into independently hashed runs.
pub const MAX_DATASET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Hard ceiling on optimizer updates in one invocation.
pub const MAX_OPTIMIZER_STEPS: u64 = 2_000_000;
/// Hard wall-clock ceiling on one invocation (24 hours).
pub const MAX_WALL_TIME_SECONDS: u64 = 24 * 60 * 60;
/// Model v1's hard batch ceiling. Kept here so configuration-only builds do
/// not need to compile Burn.
pub const MAX_TRAINING_BATCH: u16 = 64;
/// Conservative ceiling for one decoded JSONL window.
pub const MAX_WINDOW_CELLS: usize = 2_000_000;
/// Maximum bytes in one JSONL window record.
pub const MAX_JSONL_LINE_BYTES: usize = 8 * 1024 * 1024;
/// Maximum declared windows in one streaming local shard.
pub const MAX_LOCAL_WINDOWS: u32 = 1_000_000;
/// Maximum coupled variates in the v1 local shard contract.
pub const MAX_LOCAL_VARIATES: u16 = 128;
/// Preferred deterministic shuffle reservoir when the memory reservation can
/// accommodate it.
pub const DEFAULT_SHUFFLE_WINDOWS: usize = 64;

/// Configuration validation failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// An identifier contains disallowed bytes or has an invalid length.
    #[error("invalid job id; expected 1..=64 ASCII letters, digits, '_' or '-'")]
    InvalidJobId,
    /// A relative dataset path escaped its configured root or used unsupported
    /// path syntax.
    #[error("invalid dataset path; expected a bounded relative path of safe ASCII segments")]
    InvalidDatasetPath,
    /// A SHA-256 digest was not exactly 64 hexadecimal characters.
    #[error("invalid SHA-256 digest")]
    InvalidDigest,
    /// The declared dataset size is out of bounds.
    #[error("dataset size must be in 1..={MAX_DATASET_BYTES} bytes")]
    InvalidDatasetSize,
    /// The declared local shard schema is incomplete or out of bounds.
    #[error("invalid local dataset contract: {0}")]
    InvalidDatasetContract(&'static str),
    /// The deterministic synthetic generator was out of bounds.
    #[error("invalid synthetic corpus: {0}")]
    InvalidSynthetic(&'static str),
    /// An optimizer value is out of bounds or non-finite.
    #[error("invalid optimizer configuration: {0}")]
    InvalidOptimizer(&'static str),
    /// A requested compute budget is invalid.
    #[error("invalid training budget: {0}")]
    InvalidBudget(&'static str),
    /// Training device is unavailable in this build.
    #[error("training device is not enabled in this build: {0}")]
    DeviceUnavailable(&'static str),
    /// The core training contract rejected the specification.
    #[error("invalid forecast training specification: {0}")]
    TrainSpec(String),
    /// A request file was too large or malformed.
    #[error("invalid request file: {0}")]
    RequestFile(String),
}

/// Stable, user-supplied idempotency key for one training run.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct JobId(String);

impl JobId {
    /// Creates a bounded job identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, ConfigError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(ConfigError::InvalidJobId);
        }
        Ok(Self(value))
    }

    /// Returns the validated identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for JobId {
    type Error = ConfigError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<JobId> for String {
    fn from(value: JobId) -> Self {
        value.0
    }
}

/// A path below the runner's configured dataset root.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RelativeDataPath(String);

impl RelativeDataPath {
    /// Validates a portable, relative, non-traversing path.
    pub fn new(value: impl Into<String>) -> Result<Self, ConfigError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 512
            || value.contains('\\')
            || value.contains('\0')
            || value.bytes().any(|byte| {
                !(byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'_' | b'-' | b'.'))
            })
        {
            return Err(ConfigError::InvalidDatasetPath);
        }

        if value
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
        {
            return Err(ConfigError::InvalidDatasetPath);
        }
        let path = Path::new(&value);
        if path.is_absolute()
            || path.components().any(|component| {
                !matches!(component, Component::Normal(_))
                    || component.as_os_str().to_string_lossy() == "."
                    || component.as_os_str().to_string_lossy() == ".."
            })
        {
            return Err(ConfigError::InvalidDatasetPath);
        }

        Ok(Self(value))
    }

    /// Returns the portable path value.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Converts the validated path to a platform path.
    pub fn to_path_buf(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }
}

impl TryFrom<String> for RelativeDataPath {
    type Error = ConfigError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RelativeDataPath> for String {
    fn from(value: RelativeDataPath) -> Self {
        value.0
    }
}

/// A fixed-size SHA-256 digest with canonical lower-case hex serialization.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Parses exactly one SHA-256 hexadecimal digest.
    pub fn from_hex(value: &str) -> Result<Self, ConfigError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ConfigError::InvalidDigest);
        }
        let bytes = hex::decode(value).map_err(|_| ConfigError::InvalidDigest)?;
        let bytes: [u8; 32] = bytes.try_into().map_err(|_| ConfigError::InvalidDigest)?;
        Ok(Self(bytes))
    }

    /// Hashes an in-memory payload.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Returns canonical lower-case hexadecimal text.
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }

    /// Returns the raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Sha256Digest")
            .field(&self.to_hex())
            .finish()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(D::Error::custom)
    }
}

/// A dataset file whose identity is verified before deserialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetInput {
    /// Path below the configured dataset root.
    pub path: RelativeDataPath,
    /// Exact expected bytes.
    pub size_bytes: u64,
    /// Exact expected SHA-256 digest.
    pub sha256: Sha256Digest,
    /// Exact number of JSONL windows in the shard.
    pub window_count: u32,
    /// Exact coupled variate count in every window.
    pub variates: u16,
    /// Canonical identity of feature order, units, transforms, and semantics.
    pub feature_schema_digest: CanonicalDigest,
}

impl DatasetInput {
    /// Validates declared resource bounds.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.size_bytes == 0 || self.size_bytes > MAX_DATASET_BYTES {
            return Err(ConfigError::InvalidDatasetSize);
        }
        if !(1..=MAX_LOCAL_WINDOWS).contains(&self.window_count) {
            return Err(ConfigError::InvalidDatasetContract(
                "window_count must be in 1..=1000000",
            ));
        }
        if !(1..=MAX_LOCAL_VARIATES).contains(&self.variates) {
            return Err(ConfigError::InvalidDatasetContract(
                "variates must be in 1..=128",
            ));
        }
        if self.feature_schema_digest.as_bytes() == &[0; 32] {
            return Err(ConfigError::InvalidDatasetContract(
                "feature_schema_digest must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Bounded, deterministic data generation used for smoke tests and the
/// separately compiled synthetic fal path. It never accepts an external
/// source/provenance claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDatasetSpec {
    /// Number of generated training windows.
    pub windows: u32,
    /// Number of coupled variates.
    pub variates: u16,
    /// Missing-cell rate in permille.
    pub missing_per_mille: u16,
    /// Generator seed, independent from model initialization.
    pub seed: u64,
}

impl SyntheticDatasetSpec {
    /// Validates generator resource bounds.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=1_000_000).contains(&self.windows) {
            return Err(ConfigError::InvalidSynthetic(
                "windows must be in 1..=1000000",
            ));
        }
        if !(1..=128).contains(&self.variates) {
            return Err(ConfigError::InvalidSynthetic("variates must be in 1..=128"));
        }
        if self.missing_per_mille > 500 {
            return Err(ConfigError::InvalidSynthetic(
                "missing_per_mille must be in 0..=500",
            ));
        }
        Ok(())
    }

    /// Canonical identity of the exact deterministic generator recipe.
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&1_u16.to_be_bytes());
        bytes.extend_from_slice(&self.windows.to_be_bytes());
        bytes.extend_from_slice(&self.variates.to_be_bytes());
        bytes.extend_from_slice(&self.missing_per_mille.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        CanonicalDigest::of_bytes(b"ruview-synthetic-generator-v1", &bytes)
    }
}

/// Typed data source. The fal API does not accept this enum; hosted execution
/// uses a distinct minimal payload plus a non-serializable verified governance
/// handle.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatasetSource {
    /// Local, hash-addressed corpus manifest.
    Manifest(DatasetInput),
    /// In-process deterministic smoke/training corpus.
    Synthetic(SyntheticDatasetSpec),
}

impl DatasetSource {
    fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Manifest(input) => input.validate(),
            Self::Synthetic(spec) => spec.validate(),
        }
    }
}

/// Independently specified model capacity profiles. They are names rather
/// than arbitrary layer dimensions so hosted and local receipts stay
/// comparable.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    /// Small deterministic profile used by CI and smoke tests.
    TinyCi,
    /// Larger profile intended for the user's Linux or hosted accelerator.
    LargeLinux,
}

impl ModelProfile {
    /// Resolves the fixed model architecture.
    pub fn config(self) -> ForecastModelConfig {
        match self {
            Self::TinyCi => ForecastModelConfig::tiny_ci(),
            Self::LargeLinux => ForecastModelConfig::large_linux(),
        }
    }
}

/// A typed device choice. No executable, shell, image, or free-form argument
/// is accepted anywhere in the training request.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TrainingDevice {
    /// Burn ndarray CPU backend; also works on macOS for smoke tests.
    Cpu,
    /// Burn CUDA backend on Linux or a fal GPU worker.
    Cuda {
        /// Zero-based CUDA device ordinal.
        ordinal: u8,
    },
}

impl TrainingDevice {
    fn validate(self) -> Result<(), ConfigError> {
        match self {
            Self::Cpu if cfg!(feature = "cpu") => Ok(()),
            Self::Cuda { .. } if cfg!(feature = "cuda") => Ok(()),
            Self::Cpu => Err(ConfigError::DeviceUnavailable("cpu")),
            Self::Cuda { .. } => Err(ConfigError::DeviceUnavailable("cuda")),
        }
    }
}

/// Bounded AdamW training parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerSpec {
    /// Number of full passes over the training windows.
    pub epochs: u16,
    /// Number of windows per optimizer step.
    pub batch_size: u16,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// L2-style decoupled weight decay.
    pub weight_decay: f64,
    /// Maximum gradient norm.
    pub gradient_clip_norm: f64,
    /// Emit a recoverable checkpoint after this many epochs.
    pub checkpoint_every_epochs: u16,
    /// Seed bound into the training receipt.
    pub seed: u64,
}

impl OptimizerSpec {
    /// Validates training parameters before any allocation or device work.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=512).contains(&self.epochs) {
            return Err(ConfigError::InvalidOptimizer("epochs must be 1..=512"));
        }
        if !(1..=MAX_TRAINING_BATCH).contains(&self.batch_size) {
            return Err(ConfigError::InvalidOptimizer("batch_size must be 1..=64"));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.learning_rate > 1.0
        {
            return Err(ConfigError::InvalidOptimizer(
                "learning_rate must be finite and in (0, 1]",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 || self.weight_decay > 1.0 {
            return Err(ConfigError::InvalidOptimizer(
                "weight_decay must be finite and in [0, 1]",
            ));
        }
        if !self.gradient_clip_norm.is_finite()
            || self.gradient_clip_norm <= 0.0
            || self.gradient_clip_norm > 1_000.0
        {
            return Err(ConfigError::InvalidOptimizer(
                "gradient_clip_norm must be finite and in (0, 1000]",
            ));
        }
        // V1 emits one model-only checkpoint at successful completion or
        // cooperative cancellation. Optimizer/cursor resume is deliberately
        // not claimed by this contract.
        if self.checkpoint_every_epochs != self.epochs {
            return Err(ConfigError::InvalidOptimizer(
                "v1 requires checkpoint_every_epochs to equal epochs",
            ));
        }
        Ok(())
    }
}

/// Provider-independent compute and spend guardrails. These are request-side
/// ceilings; the server may impose lower deployment limits.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingBudget {
    /// Maximum successful optimizer updates.
    pub max_optimizer_steps: u64,
    /// Maximum elapsed wall time, including checkpoints.
    pub max_wall_time_seconds: u64,
    /// Conservative peak bytes permitted for model activations, gradients,
    /// optimizer state, and the current streaming batch.
    pub max_memory_bytes: u64,
    /// Maximum bytes that this invocation may publish across all artifacts.
    pub max_artifact_bytes: u64,
    /// Maximum atomic checkpoints, including the successful final checkpoint.
    pub max_checkpoints: u16,
}

impl TrainingBudget {
    /// Validates hard operational limits.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_optimizer_steps == 0 || self.max_optimizer_steps > MAX_OPTIMIZER_STEPS {
            return Err(ConfigError::InvalidBudget(
                "max_optimizer_steps must be in 1..=2000000",
            ));
        }
        if self.max_wall_time_seconds == 0 || self.max_wall_time_seconds > MAX_WALL_TIME_SECONDS {
            return Err(ConfigError::InvalidBudget(
                "max_wall_time_seconds must be in 1..=86400",
            ));
        }
        if !(256 * 1024 * 1024..=96 * 1024 * 1024 * 1024).contains(&self.max_memory_bytes) {
            return Err(ConfigError::InvalidBudget(
                "max_memory_bytes must be between 256 MiB and 96 GiB",
            ));
        }
        if !(1024 * 1024..=ruview_forecast_model::MAX_ARTIFACT_BYTES as u64 * 4)
            .contains(&self.max_artifact_bytes)
        {
            return Err(ConfigError::InvalidBudget("invalid max_artifact_bytes"));
        }
        if self.max_checkpoints != 1 {
            return Err(ConfigError::InvalidBudget(
                "v1 requires max_checkpoints to equal 1",
            ));
        }
        Ok(())
    }
}

/// Trusted local request. This type is deliberately not serializable: local
/// config is first decoded into [`LocalTrainingRequestWire`] and the core
/// [`TrainSpec`] is reconstructed through its validating constructor. Hosted
/// execution uses a separate redacted DTO in the fal module.
#[derive(Clone, Debug)]
pub struct TrainingRequest {
    /// Stable logical job identity.
    pub job_id: JobId,
    /// Validated backend-neutral split, horizon, and normalization contract.
    pub train: TrainSpec,
    /// Hash-addressed input corpus.
    pub dataset: DatasetSource,
    /// Fixed architecture capacity profile.
    pub model: ModelProfile,
    /// Device selected explicitly by the caller.
    pub device: TrainingDevice,
    /// Bounded optimizer configuration.
    pub optimizer: OptimizerSpec,
    /// Hard compute/time ceilings enforced cooperatively by the runner.
    pub budget: TrainingBudget,
    /// Optional upstream synthetic-host request identity. Local request files
    /// cannot populate it; the validated worker adapter binds it in-process.
    execution_binding: Option<CanonicalDigest>,
}

impl TrainingRequest {
    /// Constructs a local-only typed request.
    #[allow(clippy::too_many_arguments)]
    pub fn new_local(
        job_id: JobId,
        train: TrainSpec,
        dataset: DatasetSource,
        model: ModelProfile,
        device: TrainingDevice,
        optimizer: OptimizerSpec,
        budget: TrainingBudget,
    ) -> Result<Self, ConfigError> {
        let request = Self {
            job_id,
            train,
            dataset,
            model,
            device,
            optimizer,
            budget,
            execution_binding: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Cross-validates every request component.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.train.destination() != TrainingDestinationKind::Local {
            return Err(ConfigError::TrainSpec(
                "local runner requires a local TrainSpec; fal has a distinct synthetic DTO"
                    .to_string(),
            ));
        }
        self.dataset.validate()?;
        self.device.validate()?;
        self.optimizer.validate()?;
        self.budget.validate()?;

        let model = self.model.config();
        model
            .validate()
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
        if self.train.context_length() != model.context_len
            || self.train.horizon() != model.horizon
            || self.train.quantiles().values() != model.quantiles
        {
            return Err(ConfigError::TrainSpec(
                "TrainSpec context, horizon, and quantiles must exactly match model profile"
                    .to_string(),
            ));
        }
        let source_digest = match &self.dataset {
            DatasetSource::Manifest(input) => {
                if self.train.normalization() != NormalizationPolicy::None {
                    return Err(ConfigError::TrainSpec(
                        "JSONL v1 requires an explicit no-normalization contract; a future two-pass adapter will implement StandardizeTrainOnly"
                            .to_string(),
                    ));
                }
                CanonicalDigest::of_bytes(b"ruview-jsonl-window-shard-v1", input.sha256.as_bytes())
            }
            DatasetSource::Synthetic(spec) => {
                if usize::from(spec.variates) > model.max_variates {
                    return Err(ConfigError::InvalidSynthetic(
                        "variates exceed selected model profile",
                    ));
                }
                let cells = usize::try_from(spec.windows)
                    .ok()
                    .and_then(|windows| {
                        model
                            .context_len
                            .checked_add(model.horizon)
                            .and_then(|rows| rows.checked_mul(usize::from(spec.variates)))
                            .and_then(|per_window| per_window.checked_mul(windows))
                    })
                    .ok_or(ConfigError::InvalidSynthetic(
                        "generated cell count overflow",
                    ))?;
                if cells > 2_000_000_000 {
                    return Err(ConfigError::InvalidSynthetic(
                        "generated corpus exceeds two billion cells",
                    ));
                }
                spec.canonical_digest()
            }
        };
        if source_digest != self.train.dataset_digest() {
            return Err(ConfigError::TrainSpec(
                "TrainSpec dataset digest does not bind the selected source".to_string(),
            ));
        }
        let batches = match &self.dataset {
            DatasetSource::Synthetic(spec) => {
                u64::from(spec.windows).div_ceil(u64::from(self.optimizer.batch_size))
            }
            DatasetSource::Manifest(input) => {
                if usize::from(input.variates) > model.max_variates {
                    return Err(ConfigError::InvalidDatasetContract(
                        "variates exceed selected model profile",
                    ));
                }
                u64::from(input.window_count).div_ceil(u64::from(self.optimizer.batch_size))
            }
        };
        let planned = batches
            .checked_mul(u64::from(self.optimizer.epochs))
            .ok_or(ConfigError::InvalidBudget("planned steps overflow"))?;
        if planned > self.budget.max_optimizer_steps {
            return Err(ConfigError::InvalidBudget(
                "planned optimizer steps exceed max_optimizer_steps",
            ));
        }
        let activation_cells_per_example = model
            .activation_cells()
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
        let forward_cells = activation_cells_per_example
            .checked_mul(usize::from(self.optimizer.batch_size))
            .ok_or(ConfigError::InvalidBudget(
                "activation cell estimate overflow",
            ))?;
        if forward_cells > 64 * 1024 * 1024 {
            return Err(ConfigError::InvalidBudget(
                "batch exceeds the model forward activation cell limit",
            ));
        }
        let forward_multiply_adds = model
            .forward_multiply_adds()
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?
            .checked_mul(u64::from(self.optimizer.batch_size))
            .ok_or(ConfigError::InvalidBudget(
                "forward multiply-add estimate overflow",
            ))?;
        if forward_multiply_adds > ruview_forecast_model::MAX_FORWARD_MULTIPLY_ADDS {
            return Err(ConfigError::InvalidBudget(
                "batch exceeds the model forward multiply-add limit",
            ));
        }
        let activation_bytes = u64::try_from(activation_cells_per_example)
            .ok()
            .and_then(|cells| cells.checked_mul(u64::from(self.optimizer.batch_size)))
            .and_then(|cells| cells.checked_mul(4))
            // Forward values, backward graph/gradients, and optimizer state.
            .and_then(|bytes| bytes.checked_mul(10))
            .ok_or(ConfigError::InvalidBudget(
                "activation memory estimate overflow",
            ))?;
        let parameter_bytes = u64::try_from(
            model
                .parameter_count()
                .map_err(|error| ConfigError::TrainSpec(error.to_string()))?,
        )
        .ok()
        // Parameters, gradients, Adam moments, serialization copy, and
        // backend bookkeeping. This is intentionally conservative.
        .and_then(|parameters| parameters.checked_mul(24))
        .ok_or(ConfigError::InvalidBudget(
            "parameter memory estimate overflow",
        ))?;
        let fixed_memory = activation_bytes
            .checked_add(parameter_bytes)
            .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
            .ok_or(ConfigError::InvalidBudget("memory estimate overflow"))?;
        let required_memory = if matches!(self.dataset, DatasetSource::Manifest(_)) {
            // One bounded decoder line plus at least one decoded shuffle
            // window. The runner chooses a larger reservoir only from the
            // remaining reservation.
            fixed_memory.checked_add(2 * MAX_JSONL_LINE_BYTES as u64)
        } else {
            Some(fixed_memory)
        }
        .ok_or(ConfigError::InvalidBudget("memory estimate overflow"))?;
        if required_memory > self.budget.max_memory_bytes {
            return Err(ConfigError::InvalidBudget(
                "conservative training estimate exceeds max_memory_bytes",
            ));
        }
        Ok(())
    }

    /// Selects a deterministic local shuffle capacity from the validated
    /// reservation. Synthetic runs do not call this method.
    #[cfg(feature = "training")]
    pub(crate) fn local_shuffle_capacity(&self) -> usize {
        let model = self.model.config();
        let activation = model
            .activation_cells()
            .unwrap_or(usize::MAX)
            .saturating_mul(usize::from(self.optimizer.batch_size))
            .saturating_mul(4)
            .saturating_mul(10);
        let parameters = model
            .parameter_count()
            .unwrap_or(usize::MAX)
            .saturating_mul(24);
        let fixed = activation
            .saturating_add(parameters)
            .saturating_add(16 * 1024 * 1024)
            .saturating_add(MAX_JSONL_LINE_BYTES);
        let available = usize::try_from(self.budget.max_memory_bytes)
            .unwrap_or(usize::MAX)
            .saturating_sub(fixed);
        (available / MAX_JSONL_LINE_BYTES).clamp(1, DEFAULT_SHUFFLE_WINDOWS)
    }

    /// Binds a validated hosted synthetic payload into the candidate and
    /// training receipt. This cannot turn a manifest/local-data request into a
    /// hosted request and it grants no release authority.
    pub fn bind_hosted_synthetic_execution(
        mut self,
        binding: CanonicalDigest,
    ) -> Result<Self, ConfigError> {
        if binding.is_zero() || !matches!(self.dataset, DatasetSource::Synthetic(_)) {
            return Err(ConfigError::TrainSpec(
                "hosted execution binding requires a non-zero synthetic request digest".to_string(),
            ));
        }
        self.execution_binding = Some(binding);
        self.validate()?;
        Ok(self)
    }

    /// Optional hosted payload identity included in run idempotency and
    /// receipt verification.
    #[must_use]
    pub const fn execution_binding(&self) -> Option<CanonicalDigest> {
        self.execution_binding
    }

    /// Converts a request into a wrapper that can only exist after validation.
    pub fn into_validated(self) -> Result<ValidatedTrainingRequest, ConfigError> {
        self.validate()?;
        Ok(ValidatedTrainingRequest(self))
    }
}

/// Untrusted local file representation. It is converted into a non-
/// deserializable core `TrainSpec` after every constituent validates.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalTrainingRequestWire {
    /// Stable logical job identity.
    pub job_id: JobId,
    /// Local training contract fields.
    pub train: LocalTrainSpecWire,
    /// Hash-addressed local source or deterministic synthetic recipe.
    pub dataset: DatasetSource,
    /// Fixed model profile.
    pub model: ModelProfile,
    /// Local backend.
    pub device: TrainingDevice,
    /// Bounded optimizer settings.
    pub optimizer: OptimizerSpec,
    /// Operational resource budget.
    pub budget: TrainingBudget,
}

/// Serializable local-only fields used to reconstruct a trusted TrainSpec.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalTrainSpecWire {
    /// Required history rows.
    pub context_length: usize,
    /// Forecast rows.
    pub horizon: usize,
    /// Cadence in milliseconds.
    pub step_ms: u64,
    /// Exact output quantiles.
    pub quantiles: QuantileSet,
    /// Leakage-safe split plan.
    pub split_plan: TemporalSplitPlan,
    /// Train-only normalization policy.
    pub normalization: NormalizationPolicy,
    /// Digest of the JSONL shard or synthetic recipe.
    pub dataset_digest: CanonicalDigest,
    /// Local governance/audit binding. This object never crosses fal.
    pub policy: DataPolicy,
}

impl LocalTrainingRequestWire {
    /// Builds the trusted local request through core constructors.
    pub fn into_request(self) -> Result<TrainingRequest, ConfigError> {
        let train = TrainSpec::new_local(
            self.train.context_length,
            self.train.horizon,
            self.train.step_ms,
            self.train.quantiles,
            self.train.split_plan,
            self.train.normalization,
            self.train.dataset_digest,
            self.train.policy,
        )
        .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
        TrainingRequest::new_local(
            self.job_id,
            train,
            self.dataset,
            self.model,
            self.device,
            self.optimizer,
            self.budget,
        )
    }
}

/// Builds the deterministic synthetic training contract used by smoke tests
/// and by the hosted worker after decoding the redacted recipe. All identity
/// strings are fixed synthetic namespaces and never originate from RuView.
pub fn synthetic_train_spec(
    model_profile: ModelProfile,
    generator: &SyntheticDatasetSpec,
    fal_destination: bool,
    retention_until_ms: u64,
) -> Result<TrainSpec, ConfigError> {
    generator.validate()?;
    let model = model_profile.config();
    let span = u64::try_from(model.context_len + model.horizon + 1)
        .ok()
        .and_then(|rows| rows.checked_mul(1_000))
        .ok_or(ConfigError::TrainSpec(
            "synthetic range overflow".to_string(),
        ))?;
    let train_member = SplitMember::new(
        SeriesKey::new("synthetic-train", "generator-v1", "train")
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?,
        TimeRange::new(1, span).map_err(|error| ConfigError::TrainSpec(error.to_string()))?,
    );
    let test_member = SplitMember::new(
        SeriesKey::new("synthetic-test", "generator-v2", "test")
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?,
        TimeRange::new(1, span).map_err(|error| ConfigError::TrainSpec(error.to_string()))?,
    );
    let split = TemporalSplitPlan::new(
        SplitStrategy::EntityHoldout(HoldoutKey::Strict),
        vec![train_member],
        vec![],
        vec![test_member],
        model.horizon,
        1_000,
        0,
    )
    .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
    let policy = DataPolicy::new(
        PrivacyClass::P3,
        "synthetic",
        "synthetic",
        "synthetic",
        "forecast-foundation-pretraining",
        CanonicalDigest::of_bytes(b"synthetic-policy-v1", b"approved"),
        None,
        None,
        None,
        retention_until_ms,
        true,
    )
    .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
    let quantiles = QuantileSet::new(model.quantiles.to_vec())
        .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
    if fal_destination {
        let source = SourceState::synthetic("ruview-coupled-generator-v1")
            .map_err(|error| ConfigError::TrainSpec(error.to_string()))?;
        TrainSpec::new_fal_synthetic(
            model.context_len,
            model.horizon,
            1_000,
            quantiles,
            split,
            NormalizationPolicy::None,
            generator.canonical_digest(),
            policy,
            &source,
            generator.canonical_digest(),
            generator.seed,
        )
        .map_err(|error| ConfigError::TrainSpec(error.to_string()))
    } else {
        TrainSpec::new_local(
            model.context_len,
            model.horizon,
            1_000,
            quantiles,
            split,
            NormalizationPolicy::None,
            generator.canonical_digest(),
            policy,
        )
        .map_err(|error| ConfigError::TrainSpec(error.to_string()))
    }
}

/// A training request whose complete validation has succeeded.
#[derive(Clone, Debug)]
pub struct ValidatedTrainingRequest(TrainingRequest);

impl ValidatedTrainingRequest {
    /// Returns the checked request.
    pub fn get(&self) -> &TrainingRequest {
        &self.0
    }

    /// Consumes the validation wrapper.
    pub fn into_inner(self) -> TrainingRequest {
        self.0
    }
}

/// Loads and validates a JSON or TOML request without unbounded allocation.
pub fn load_request(path: &Path) -> Result<ValidatedTrainingRequest, ConfigError> {
    use std::io::Read as _;

    #[cfg(unix)]
    let mut file = {
        use rustix::fs::{Mode, OFlags};
        std::fs::File::from(
            rustix::fs::open(
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .map_err(std::io::Error::from)
            .map_err(|error| ConfigError::RequestFile(error.to_string()))?,
        )
    };
    #[cfg(not(unix))]
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|error| ConfigError::RequestFile(error.to_string()))?;
    let metadata = file
        .metadata()
        .map_err(|error| ConfigError::RequestFile(error.to_string()))?;
    if !metadata.is_file() {
        return Err(ConfigError::RequestFile(
            "request must be a regular file".to_string(),
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_TRAINING_REQUEST_BYTES as u64 {
        return Err(ConfigError::RequestFile(format!(
            "size must be in 1..={MAX_TRAINING_REQUEST_BYTES} bytes"
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.by_ref()
        .take(MAX_TRAINING_REQUEST_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| ConfigError::RequestFile(error.to_string()))?;
    if bytes.len() > MAX_TRAINING_REQUEST_BYTES {
        return Err(ConfigError::RequestFile(format!(
            "size must be in 1..={MAX_TRAINING_REQUEST_BYTES} bytes"
        )));
    }
    let request: LocalTrainingRequestWire = match path.extension().and_then(|value| value.to_str())
    {
        Some("toml") => toml::from_str(
            std::str::from_utf8(&bytes)
                .map_err(|error| ConfigError::RequestFile(error.to_string()))?,
        )
        .map_err(|error| ConfigError::RequestFile(error.to_string()))?,
        Some("json") => serde_json::from_slice(&bytes)
            .map_err(|error| ConfigError::RequestFile(error.to_string()))?,
        _ => {
            return Err(ConfigError::RequestFile(
                "extension must be .json or .toml".to_string(),
            ))
        }
    };
    request.into_request()?.into_validated()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_id_rejects_path_and_header_syntax() {
        for bad in ["", "../job", "job/name", "job value", "job\nheader"] {
            assert!(JobId::new(bad).is_err(), "accepted {bad:?}");
        }
        assert!(JobId::new("run_2026-09-01").is_ok());
    }

    #[test]
    fn relative_data_path_rejects_traversal_and_absolute_paths() {
        for bad in ["", "../data.json", "/data/data.json", "a/../b", "a\\b"] {
            assert!(RelativeDataPath::new(bad).is_err(), "accepted {bad:?}");
        }
        assert!(RelativeDataPath::new("approved/site-a/corpus.json").is_ok());
    }

    #[test]
    fn digest_serialization_is_canonical() {
        let digest = Sha256Digest::of_bytes(b"ruview");
        let json = serde_json::to_string(&digest).expect("serialize digest");
        assert_eq!(json.len(), 66);
        let decoded: Sha256Digest = serde_json::from_str(&json).expect("deserialize digest");
        assert_eq!(decoded, digest);
        assert!(Sha256Digest::from_hex("00").is_err());
    }

    #[test]
    fn synthetic_smoke_contract_has_a_strictly_disjoint_holdout() {
        let generator = SyntheticDatasetSpec {
            windows: 4,
            variates: 3,
            missing_per_mille: 50,
            seed: 7,
        };
        let spec = synthetic_train_spec(ModelProfile::TinyCi, &generator, false, 86_400_000)
            .expect("synthetic smoke contract");
        assert_eq!(spec.destination(), TrainingDestinationKind::Local);
    }

    #[test]
    fn local_dataset_contract_requires_shape_and_feature_identity() {
        let mut input = DatasetInput {
            path: RelativeDataPath::new("shards/train.jsonl").expect("path"),
            size_bytes: 1,
            sha256: Sha256Digest::of_bytes(b"dataset"),
            window_count: 4,
            variates: 3,
            feature_schema_digest: CanonicalDigest::of_bytes(
                b"test-feature-schema-v1",
                b"amplitude-db,phase-rad",
            ),
        };
        assert!(input.validate().is_ok());
        input.window_count = 0;
        assert!(matches!(
            input.validate(),
            Err(ConfigError::InvalidDatasetContract(_))
        ));
    }

    #[test]
    fn v1_checkpoint_contract_rejects_unimplemented_resume_claims() {
        let mut optimizer = OptimizerSpec {
            epochs: 4,
            batch_size: 2,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            gradient_clip_norm: 1.0,
            checkpoint_every_epochs: 2,
            seed: 7,
        };
        assert!(matches!(
            optimizer.validate(),
            Err(ConfigError::InvalidOptimizer(_))
        ));
        optimizer.checkpoint_every_epochs = optimizer.epochs;
        assert!(optimizer.validate().is_ok());

        let mut budget = TrainingBudget {
            max_optimizer_steps: 8,
            max_wall_time_seconds: 60,
            max_memory_bytes: 1024 * 1024 * 1024,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_checkpoints: 2,
        };
        assert!(matches!(
            budget.validate(),
            Err(ConfigError::InvalidBudget(_))
        ));
        budget.max_checkpoints = 1;
        assert!(budget.validate().is_ok());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn large_profile_batch_two_is_the_reviewed_maximum() {
        let generator = SyntheticDatasetSpec {
            windows: 2,
            variates: 8,
            missing_per_mille: 0,
            seed: 7,
        };
        let make = |batch_size| {
            TrainingRequest::new_local(
                JobId::new(format!("large-batch-{batch_size}")).expect("job"),
                synthetic_train_spec(ModelProfile::LargeLinux, &generator, false, u64::MAX)
                    .expect("train spec"),
                DatasetSource::Synthetic(generator.clone()),
                ModelProfile::LargeLinux,
                TrainingDevice::Cpu,
                OptimizerSpec {
                    epochs: 1,
                    batch_size,
                    learning_rate: 1e-3,
                    weight_decay: 1e-4,
                    gradient_clip_norm: 1.0,
                    checkpoint_every_epochs: 1,
                    seed: 11,
                },
                TrainingBudget {
                    max_optimizer_steps: 2,
                    max_wall_time_seconds: 3_600,
                    max_memory_bytes: 8 * 1024 * 1024 * 1024,
                    max_artifact_bytes: 512 * 1024 * 1024,
                    max_checkpoints: 1,
                },
            )
        };
        assert!(make(2).is_ok());
        assert!(matches!(make(3), Err(ConfigError::InvalidBudget(_))));
    }
}
