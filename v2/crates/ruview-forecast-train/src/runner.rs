//! Real Burn optimizer loop shared by the CLI and Direct Server.

use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use burn_core::module::AutodiffModule;
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::tensor::{ElementConversion, Tensor, TensorData};
use burn_optim::grad_clipping::GradientClippingConfig;
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};
use ruview_forecast_core::CanonicalDigest;
use ruview_forecast_model::{
    masked_pinball_loss, record_to_bytes, ArtifactManifest, ForecastModelConfig, ModelArtifact,
    ModelError, ModelInput, RuForecastMixer, TrainingBatch,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact::{
    open_verified_dataset, ArtifactDescriptor, ArtifactError, ArtifactKind, ArtifactStore,
    VerifiedDataset,
};
use crate::cancel::Cancellation;
use crate::config::{
    DatasetSource, ModelProfile, SyntheticDatasetSpec, TrainingDevice, ValidatedTrainingRequest,
};
use crate::corpus::{JsonlWindow, ShuffledWindows};

enum PendingBatch {
    Synthetic { offset: u32, size: usize },
    Manifest(Vec<JsonlWindow>),
}

/// Training execution failure.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// The request referenced a manifest corpus whose on-disk window adapter
    /// could not validate it.
    #[error("manifest corpus failed validation: {0}")]
    Manifest(String),
    /// The model or Burn backend rejected an operation.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Fixed model profile configuration was invalid.
    #[error(transparent)]
    ModelConfig(#[from] ruview_forecast_model::ConfigError),
    /// Artifact publication failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// A tensor backend operation failed.
    #[error("tensor execution failed: {0}")]
    Tensor(String),
    /// An arithmetic or compute-bound invariant failed.
    #[error("training budget rejected the run: {0}")]
    Budget(&'static str),
    /// The governance retention window elapsed before execution.
    #[error("training policy retention window has expired")]
    RetentionExpired,
    /// Training produced a non-finite loss.
    #[error("training produced a non-finite loss")]
    NonFiniteLoss,
    /// The request was cancelled; a verified checkpoint was published.
    #[error("training cancelled")]
    Cancelled {
        /// Last atomic checkpoint.
        checkpoint: ArtifactDescriptor,
    },
    /// Cooperative cancellation was observed after optimization completed but
    /// before the fixed artifact set reached its commit boundary. The runner
    /// converts this internal state into [`Self::Cancelled`] after rollback
    /// and checkpoint publication.
    #[error("cancellation observed at artifact publication boundary")]
    PublicationCancelled,
    /// This binary did not include the selected backend.
    #[error("selected training backend is not compiled in")]
    BackendUnavailable,
    /// Existing files were partial, malformed, or belonged to another
    /// request that reused the job id.
    #[error("existing training outcome failed closed: {0}")]
    Recovery(&'static str),
}

/// Candidate artifacts and bounded metrics from a successful run.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingOutcome {
    /// Unsigned candidate envelope. It cannot activate a production runtime.
    pub candidate: ArtifactDescriptor,
    /// Human-readable copy of the candidate manifest.
    pub manifest: ArtifactDescriptor,
    /// Metrics and provenance receipt.
    pub receipt: ArtifactDescriptor,
    /// Final checkpoint, identical to the candidate at successful completion.
    pub checkpoint: ArtifactDescriptor,
    /// Optimizer updates executed.
    pub optimizer_steps: u64,
    /// Epochs completed.
    pub epochs_completed: u16,
    /// Final masked pinball loss.
    pub final_loss: f32,
    /// Elapsed optimization time before candidate serialization/publication.
    /// Whole-invocation wall and retention limits are enforced separately at
    /// every durable publication boundary.
    pub optimizer_elapsed_millis: u64,
    /// Always false. Release signing happens only after local verification.
    production_signed: bool,
}

impl TrainingOutcome {
    /// Returns false for every trainer-produced candidate. Only the separate
    /// activation API can create a production-authorized model.
    #[must_use]
    pub const fn is_production_signed(&self) -> bool {
        self.production_signed
    }

    /// Candidate descriptor.
    #[must_use]
    pub const fn candidate(&self) -> &ArtifactDescriptor {
        &self.candidate
    }

    /// Sidecar manifest descriptor.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactDescriptor {
        &self.manifest
    }

    /// Training receipt descriptor.
    #[must_use]
    pub const fn receipt(&self) -> &ArtifactDescriptor {
        &self.receipt
    }

    /// Final model-only checkpoint descriptor.
    #[must_use]
    pub const fn checkpoint(&self) -> &ArtifactDescriptor {
        &self.checkpoint
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrainingReceipt {
    schema_version: u16,
    job_id: String,
    model_profile: ModelProfile,
    device: TrainingDevice,
    generator: Option<SyntheticDatasetSpec>,
    local_manifest: Option<crate::config::DatasetInput>,
    request_digest: String,
    upstream_request_digest: Option<String>,
    train_spec_digest: String,
    split_digest: String,
    source_digest: String,
    build_id: String,
    rust_toolchain: String,
    target_triple: String,
    cargo_lock_sha256: String,
    source_commit: String,
    container_digest: String,
    optimizer_steps: u64,
    epochs_completed: u16,
    final_loss: f32,
    optimizer_elapsed_millis: u64,
    candidate_sha256: String,
    candidate_is_untrusted: bool,
    production_signed: bool,
    clean_room_exposure: String,
}

/// Local/hosted trainer with an atomic artifact store.
#[derive(Clone, Debug)]
pub struct BurnTrainer {
    artifacts: ArtifactStore,
    dataset_root: Option<PathBuf>,
}

impl BurnTrainer {
    /// Binds the trainer to one capability-confined artifact root.
    pub fn new(artifacts: ArtifactStore) -> Self {
        Self {
            artifacts,
            dataset_root: None,
        }
    }

    /// Binds an additional capability root for local hash-addressed JSONL
    /// shards. Hosted workers never call this constructor.
    pub fn with_dataset_root(artifacts: ArtifactStore, dataset_root: PathBuf) -> Self {
        Self {
            artifacts,
            dataset_root: Some(dataset_root),
        }
    }

    fn recover_existing(
        &self,
        request: &ValidatedTrainingRequest,
    ) -> Result<Option<TrainingOutcome>, TrainingError> {
        let job_id = &request.get().job_id;
        let candidate = self
            .artifacts
            .existing_descriptor(job_id, ArtifactKind::Model)?;
        let manifest = self
            .artifacts
            .existing_descriptor(job_id, ArtifactKind::Manifest)?;
        let receipt = self
            .artifacts
            .existing_descriptor(job_id, ArtifactKind::Receipt)?;
        let checkpoint = self
            .artifacts
            .existing_descriptor(job_id, ArtifactKind::Checkpoint)?;
        let present = [
            candidate.is_some(),
            manifest.is_some(),
            receipt.is_some(),
            checkpoint.is_some(),
        ]
        .into_iter()
        .filter(|value| *value)
        .count();
        if present == 0 {
            return Ok(None);
        }
        if present != 4 {
            return Err(TrainingError::Recovery("partial artifact set"));
        }
        let candidate = candidate.ok_or(TrainingError::Recovery("missing candidate"))?;
        let manifest = manifest.ok_or(TrainingError::Recovery("missing manifest"))?;
        let receipt = receipt.ok_or(TrainingError::Recovery("missing receipt"))?;
        let checkpoint = checkpoint.ok_or(TrainingError::Recovery("missing checkpoint"))?;
        let candidate_bytes = self.artifacts.read_bytes(&candidate)?;
        let manifest_bytes = self.artifacts.read_bytes(&manifest)?;
        let receipt_bytes = self.artifacts.read_bytes(&receipt)?;
        let embedded = ModelArtifact::decode(&candidate_bytes)
            .map_err(|_| TrainingError::Recovery("malformed candidate"))?;
        let sidecar: ArtifactManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| TrainingError::Recovery("malformed manifest"))?;
        let value: TrainingReceipt = serde_json::from_slice(&receipt_bytes)
            .map_err(|_| TrainingError::Recovery("malformed receipt"))?;
        if value.schema_version != 1
            || value.job_id != job_id.as_str()
            || value.request_digest != request_digest(request).to_hex()
            || value.upstream_request_digest
                != request
                    .get()
                    .execution_binding()
                    .map(CanonicalDigest::to_hex)
            || value.candidate_sha256 != candidate.sha256.to_hex()
            || !value.candidate_is_untrusted
            || value.production_signed
            || value.optimizer_steps == 0
            || value.optimizer_steps > request.get().budget.max_optimizer_steps
            || value.epochs_completed != request.get().optimizer.epochs
            || !value.final_loss.is_finite()
            || candidate.size_bytes != checkpoint.size_bytes
            || candidate.sha256 != checkpoint.sha256
            || embedded.manifest() != &sidecar
        {
            return Err(TrainingError::Recovery("receipt or artifact mismatch"));
        }
        for descriptor in [&candidate, &manifest, &receipt, &checkpoint] {
            self.artifacts.verify(descriptor)?;
        }
        Ok(Some(TrainingOutcome {
            candidate,
            manifest,
            receipt,
            checkpoint,
            optimizer_steps: value.optimizer_steps,
            epochs_completed: value.epochs_completed,
            final_loss: value.final_loss,
            optimizer_elapsed_millis: value.optimizer_elapsed_millis,
            production_signed: false,
        }))
    }

    /// Runs the explicit backend selected by the typed request.
    pub fn train(
        &self,
        request: &ValidatedTrainingRequest,
        cancellation: &dyn Cancellation,
    ) -> Result<TrainingOutcome, TrainingError> {
        let request_value = request.get();
        let _lease = self.artifacts.lock_job(&request_value.job_id)?;
        let now_ms = unix_time_millis();
        if now_ms == 0 || now_ms >= request_value.train.policy().retention_until_ms() {
            return Err(TrainingError::RetentionExpired);
        }
        if let Some(existing) = self.recover_existing(request)? {
            return Ok(existing);
        }
        let wall_limit = Duration::from_secs(request_value.budget.max_wall_time_seconds);
        let wall_limit_ms = request_value
            .budget
            .max_wall_time_seconds
            .checked_mul(1_000)
            .ok_or(TrainingError::Budget("wall-time budget overflow"))?;
        let latest_completion_ms = now_ms
            .checked_add(wall_limit_ms)
            .ok_or(TrainingError::Budget("wall-time deadline overflow"))?;
        if latest_completion_ms >= request_value.train.policy().retention_until_ms() {
            return Err(TrainingError::RetentionExpired);
        }
        let started = Instant::now();
        let verified_dataset = match &request_value.dataset {
            DatasetSource::Synthetic(_) => None,
            DatasetSource::Manifest(input) => {
                let root = self.dataset_root.as_ref().ok_or_else(|| {
                    TrainingError::Manifest(
                        "manifest request requires a configured dataset root".to_string(),
                    )
                })?;
                Some(open_verified_dataset(root, input)?)
            }
        };
        ensure_execution_allowed(request, started, wall_limit)?;
        match request_value.device {
            TrainingDevice::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    use burn_autodiff::Autodiff;
                    use burn_ndarray::{NdArray, NdArrayDevice};
                    type Backend = Autodiff<NdArray<f32>>;
                    self.train_backend::<Backend>(
                        request,
                        verified_dataset.as_ref(),
                        &NdArrayDevice::default(),
                        cancellation,
                        started,
                        wall_limit,
                    )
                }
                #[cfg(not(feature = "cpu"))]
                Err(TrainingError::BackendUnavailable)
            }
            TrainingDevice::Cuda { ordinal } => {
                #[cfg(feature = "cuda")]
                {
                    use burn_autodiff::Autodiff;
                    use burn_cuda::{Cuda, CudaDevice};
                    type Backend = Autodiff<Cuda<f32, i32>>;
                    self.train_backend::<Backend>(
                        request,
                        verified_dataset.as_ref(),
                        &CudaDevice::new(usize::from(ordinal)),
                        cancellation,
                        started,
                        wall_limit,
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    Err(TrainingError::BackendUnavailable)
                }
            }
        }
    }

    fn train_backend<B: AutodiffBackend<FloatElem = f32>>(
        &self,
        request: &ValidatedTrainingRequest,
        verified_dataset: Option<&VerifiedDataset>,
        device: &B::Device,
        cancellation: &dyn Cancellation,
        started: Instant,
        wall_limit: Duration,
    ) -> Result<TrainingOutcome, TrainingError>
    where
        RuForecastMixer<B>: AutodiffModule<B, InnerModule = RuForecastMixer<B::InnerBackend>>,
    {
        let request_value = request.get();
        let config = request_value.model.config();
        config.validate()?;
        if let DatasetSource::Synthetic(synthetic) = &request_value.dataset {
            if usize::from(synthetic.variates) > config.max_variates {
                return Err(TrainingError::Budget(
                    "synthetic variates exceed the selected model profile",
                ));
            }
        }

        B::seed(device, request_value.optimizer.seed);
        let mut model = RuForecastMixer::<B>::init(&config, device)?;
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(request_value.optimizer.weight_decay as f32)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                request_value.optimizer.gradient_clip_norm as f32,
            )))
            .init();
        let mut steps = 0_u64;
        let mut final_loss = f32::NAN;
        let mut completed_epochs = 0_u16;

        for epoch in 0..request_value.optimizer.epochs {
            let mut synthetic_offset = 0_u32;
            let mut epoch_windows = 0_u32;
            let mut stream = match (&request_value.dataset, verified_dataset) {
                (DatasetSource::Manifest(_), Some(dataset)) => Some(
                    ShuffledWindows::new(
                        dataset,
                        &request_value.train,
                        &config,
                        request_value.local_shuffle_capacity(),
                        request_value.optimizer.seed ^ u64::from(epoch),
                    )
                    .map_err(|error| TrainingError::Manifest(error.to_string()))?,
                ),
                (DatasetSource::Synthetic(_), None) => None,
                _ => {
                    return Err(TrainingError::Manifest(
                        "source capability mismatch".to_string(),
                    ))
                }
            };
            loop {
                ensure_execution_allowed(request, started, wall_limit)?;
                if cancellation.is_cancelled() {
                    let checkpoint = self.checkpoint(
                        request,
                        &config,
                        model.valid(),
                        steps,
                        completed_epochs,
                        started,
                        wall_limit,
                    )?;
                    return Err(TrainingError::Cancelled { checkpoint });
                }
                // Discover source exhaustion before enforcing a next-step
                // budget. This lets an exact planned-step cap succeed while
                // still rejecting a real extra batch before Tensor creation.
                let pending = match &request_value.dataset {
                    DatasetSource::Synthetic(synthetic) => {
                        if synthetic_offset >= synthetic.windows {
                            break;
                        }
                        let remaining = synthetic.windows - synthetic_offset;
                        let size = remaining.min(u32::from(request_value.optimizer.batch_size));
                        PendingBatch::Synthetic {
                            offset: synthetic_offset,
                            size: size as usize,
                        }
                    }
                    DatasetSource::Manifest(_) => {
                        let stream = stream.as_mut().expect("manifest stream exists");
                        let mut windows =
                            Vec::with_capacity(usize::from(request_value.optimizer.batch_size));
                        while windows.len() < usize::from(request_value.optimizer.batch_size) {
                            match stream
                                .next_window()
                                .map_err(|error| TrainingError::Manifest(error.to_string()))?
                            {
                                Some(window) => windows.push(window),
                                None => break,
                            }
                        }
                        if windows.is_empty() {
                            break;
                        }
                        let DatasetSource::Manifest(input) = &request_value.dataset else {
                            return Err(TrainingError::Manifest(
                                "manifest source changed during training".to_string(),
                            ));
                        };
                        if windows
                            .iter()
                            .any(|window| window.variates != input.variates)
                        {
                            return Err(TrainingError::Manifest(
                                "window variates disagree with dataset contract".to_string(),
                            ));
                        }
                        epoch_windows = epoch_windows
                            .checked_add(u32::try_from(windows.len()).map_err(|_| {
                                TrainingError::Budget("window count conversion overflow")
                            })?)
                            .ok_or(TrainingError::Budget("window count overflow"))?;
                        if epoch_windows > input.window_count {
                            return Err(TrainingError::Manifest(
                                "shard contains more windows than declared".to_string(),
                            ));
                        }
                        PendingBatch::Manifest(windows)
                    }
                };
                ensure_execution_allowed(request, started, wall_limit)?;
                if steps >= request_value.budget.max_optimizer_steps {
                    return Err(TrainingError::Budget("optimizer step budget exhausted"));
                }
                let batch = match pending {
                    PendingBatch::Synthetic { offset, size } => {
                        let DatasetSource::Synthetic(synthetic) = &request_value.dataset else {
                            return Err(TrainingError::Manifest(
                                "synthetic source changed during training".to_string(),
                            ));
                        };
                        let batch =
                            synthetic_batch::<B>(synthetic, &config, offset, size, epoch, device)?;
                        synthetic_offset = synthetic_offset
                            .checked_add(size as u32)
                            .ok_or(TrainingError::Budget("synthetic offset overflow"))?;
                        batch
                    }
                    PendingBatch::Manifest(windows) => {
                        jsonl_batch::<B>(&windows, &config, request_value.train.step_ms(), device)?
                    }
                };
                let output = model.forward(batch.input)?;
                let loss = masked_pinball_loss(&output, batch.targets, batch.target_mask, device)?;
                final_loss = loss
                    .clone()
                    .try_into_scalar()
                    .map_err(|error| TrainingError::Tensor(error.to_string()))?
                    .elem::<f32>();
                if !final_loss.is_finite() {
                    return Err(TrainingError::NonFiniteLoss);
                }
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(request_value.optimizer.learning_rate, model, gradients);
                steps += 1;
                ensure_execution_allowed(request, started, wall_limit)?;
            }
            if let DatasetSource::Manifest(input) = &request_value.dataset {
                if epoch_windows != input.window_count {
                    return Err(TrainingError::Manifest(
                        "shard window count disagrees with dataset contract".to_string(),
                    ));
                }
            }
            completed_epochs = epoch + 1;
        }

        if steps == 0 || !final_loss.is_finite() {
            return Err(TrainingError::Manifest(
                "training shard contained no usable windows".to_string(),
            ));
        }

        ensure_execution_allowed(request, started, wall_limit)?;
        let optimizer_elapsed_millis =
            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let (candidate_bytes, manifest_value) = build_candidate(
            request,
            &config,
            record_to_bytes(model.valid())?,
            steps,
            completed_epochs,
        )?;
        let job_id = &request_value.job_id;
        let candidate_sha256 = crate::config::Sha256Digest::of_bytes(&candidate_bytes);
        let manifest_bytes = serde_json::to_vec_pretty(&manifest_value)
            .map_err(|error| TrainingError::Tensor(error.to_string()))?;
        let receipt_value = TrainingReceipt {
            schema_version: 1,
            job_id: job_id.as_str().to_owned(),
            model_profile: request_value.model,
            device: request_value.device,
            generator: match &request_value.dataset {
                DatasetSource::Synthetic(value) => Some(value.clone()),
                DatasetSource::Manifest(_) => None,
            },
            local_manifest: match &request_value.dataset {
                DatasetSource::Manifest(value) => Some(value.clone()),
                DatasetSource::Synthetic(_) => None,
            },
            request_digest: request_digest(request).to_hex(),
            upstream_request_digest: request_value
                .execution_binding()
                .map(CanonicalDigest::to_hex),
            train_spec_digest: request_value.train.canonical_digest().to_hex(),
            split_digest: request_value.train.split_plan().canonical_digest().to_hex(),
            source_digest: match &request_value.dataset {
                DatasetSource::Synthetic(value) => value.canonical_digest().to_hex(),
                DatasetSource::Manifest(_) => request_value.train.dataset_digest().to_hex(),
            },
            build_id: concat!("ruforecast-", env!("CARGO_PKG_VERSION")).to_owned(),
            rust_toolchain: option_env!("RUVIEW_RUSTC_VERSION")
                .unwrap_or("UNVERIFIED")
                .to_owned(),
            target_triple: option_env!("RUVIEW_BUILD_TARGET")
                .unwrap_or("UNVERIFIED")
                .to_owned(),
            cargo_lock_sha256: option_env!("RUVIEW_CARGO_LOCK_SHA256")
                .unwrap_or("UNVERIFIED")
                .to_owned(),
            source_commit: option_env!("RUVIEW_SOURCE_COMMIT")
                .unwrap_or("UNVERIFIED")
                .to_owned(),
            container_digest: option_env!("RUVIEW_CONTAINER_DIGEST")
                .unwrap_or("UNVERIFIED")
                .to_owned(),
            optimizer_steps: steps,
            epochs_completed: completed_epochs,
            final_loss,
            optimizer_elapsed_millis,
            candidate_sha256: candidate_sha256.to_hex(),
            candidate_is_untrusted: true,
            production_signed: false,
            clean_room_exposure: "no_google_code_config_weights_tests_or_outputs".to_owned(),
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt_value)
            .map_err(|error| TrainingError::Tensor(error.to_string()))?;
        ensure_execution_allowed(request, started, wall_limit)?;
        let artifact_bytes = candidate_bytes
            .len()
            .checked_mul(2)
            .and_then(|total| total.checked_add(manifest_bytes.len()))
            .and_then(|total| total.checked_add(receipt_bytes.len()))
            .and_then(|total| u64::try_from(total).ok())
            .ok_or(TrainingError::Budget("artifact byte count overflow"))?;
        if artifact_bytes > request_value.budget.max_artifact_bytes {
            return Err(TrainingError::Budget("artifact byte budget exhausted"));
        }

        let publication = (|| -> Result<_, TrainingError> {
            ensure_publication_allowed(request, cancellation, started, wall_limit)?;
            let candidate =
                self.artifacts
                    .commit_bytes(job_id, ArtifactKind::Model, &candidate_bytes)?;
            self.artifacts.verify(&candidate)?;
            ensure_publication_allowed(request, cancellation, started, wall_limit)?;

            let checkpoint =
                self.artifacts
                    .commit_bytes(job_id, ArtifactKind::Checkpoint, &candidate_bytes)?;
            self.artifacts.verify(&checkpoint)?;
            ensure_publication_allowed(request, cancellation, started, wall_limit)?;

            let manifest =
                self.artifacts
                    .commit_bytes(job_id, ArtifactKind::Manifest, &manifest_bytes)?;
            self.artifacts.verify(&manifest)?;
            ensure_publication_allowed(request, cancellation, started, wall_limit)?;

            let receipt =
                self.artifacts
                    .commit_bytes(job_id, ArtifactKind::Receipt, &receipt_bytes)?;
            self.artifacts.verify(&receipt)?;
            ensure_publication_allowed(request, cancellation, started, wall_limit)?;
            Ok((candidate, checkpoint, manifest, receipt))
        })();
        let (candidate, checkpoint, manifest, receipt) = match publication {
            Ok(descriptors) => descriptors,
            Err(error) => {
                self.artifacts.remove_job_outputs(job_id)?;
                if matches!(error, TrainingError::PublicationCancelled) {
                    let checkpoint = self.commit_checkpoint_bytes(
                        request,
                        &candidate_bytes,
                        started,
                        wall_limit,
                    )?;
                    return Err(TrainingError::Cancelled { checkpoint });
                }
                return Err(error);
            }
        };
        Ok(TrainingOutcome {
            candidate,
            manifest,
            receipt,
            checkpoint,
            optimizer_steps: steps,
            epochs_completed: completed_epochs,
            final_loss,
            optimizer_elapsed_millis,
            production_signed: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn checkpoint<B: burn_core::tensor::backend::Backend>(
        &self,
        request: &ValidatedTrainingRequest,
        config: &ForecastModelConfig,
        model: RuForecastMixer<B>,
        steps: u64,
        epochs: u16,
        started: Instant,
        wall_limit: Duration,
    ) -> Result<ArtifactDescriptor, TrainingError> {
        ensure_execution_allowed(request, started, wall_limit)?;
        let (bytes, _) = build_candidate(request, config, record_to_bytes(model)?, steps, epochs)?;
        self.commit_checkpoint_bytes(request, &bytes, started, wall_limit)
    }

    fn commit_checkpoint_bytes(
        &self,
        request: &ValidatedTrainingRequest,
        bytes: &[u8],
        started: Instant,
        wall_limit: Duration,
    ) -> Result<ArtifactDescriptor, TrainingError> {
        let checkpoint_bytes = u64::try_from(bytes.len())
            .map_err(|_| TrainingError::Budget("checkpoint byte count overflow"))?;
        if checkpoint_bytes > request.get().budget.max_artifact_bytes {
            return Err(TrainingError::Budget("artifact byte budget exhausted"));
        }
        ensure_execution_allowed(request, started, wall_limit)?;
        let job_id = &request.get().job_id;
        let publication = (|| -> Result<_, TrainingError> {
            let descriptor =
                self.artifacts
                    .commit_bytes(job_id, ArtifactKind::Checkpoint, bytes)?;
            self.artifacts.verify(&descriptor)?;
            ensure_execution_allowed(request, started, wall_limit)?;
            Ok(descriptor)
        })();
        match publication {
            Ok(descriptor) => Ok(descriptor),
            Err(error) => {
                self.artifacts.remove_job_outputs(job_id)?;
                Err(error)
            }
        }
    }
}

fn ensure_publication_allowed(
    request: &ValidatedTrainingRequest,
    cancellation: &dyn Cancellation,
    started: Instant,
    wall_limit: Duration,
) -> Result<(), TrainingError> {
    ensure_execution_allowed(request, started, wall_limit)?;
    if cancellation.is_cancelled() {
        return Err(TrainingError::PublicationCancelled);
    }
    Ok(())
}

fn ensure_execution_allowed(
    request: &ValidatedTrainingRequest,
    started: Instant,
    wall_limit: Duration,
) -> Result<(), TrainingError> {
    if started.elapsed() >= wall_limit {
        return Err(TrainingError::Budget("wall-time budget exhausted"));
    }
    let now_ms = unix_time_millis();
    if now_ms == 0 || now_ms >= request.get().train.policy().retention_until_ms() {
        return Err(TrainingError::RetentionExpired);
    }
    Ok(())
}

fn build_candidate(
    request: &ValidatedTrainingRequest,
    config: &ForecastModelConfig,
    weights: Vec<u8>,
    steps: u64,
    epochs: u16,
) -> Result<(Vec<u8>, ArtifactManifest), TrainingError> {
    let request_value = request.get();
    let generator_bytes = match &request_value.dataset {
        DatasetSource::Synthetic(generator) => serde_json::to_vec(generator),
        DatasetSource::Manifest(input) => serde_json::to_vec(input),
    }
    .map_err(|error| TrainingError::Tensor(error.to_string()))?;
    let feature_digest = match &request_value.dataset {
        DatasetSource::Synthetic(generator) => {
            let mut schema = Vec::with_capacity(32);
            schema.extend_from_slice(&1_u16.to_be_bytes());
            schema.extend_from_slice(&generator.variates.to_be_bytes());
            schema.extend_from_slice(&(config.context_len as u64).to_be_bytes());
            schema.extend_from_slice(&(config.horizon as u64).to_be_bytes());
            schema.extend_from_slice(&request_value.train.step_ms().to_be_bytes());
            CanonicalDigest::of_bytes(b"ruview-synthetic-feature-schema-v1", &schema)
        }
        DatasetSource::Manifest(input) => input.feature_schema_digest,
    };
    let mut training_identity = generator_bytes;
    training_identity.extend_from_slice(request_value.train.canonical_digest().as_bytes());
    training_identity.extend_from_slice(request_digest(request).as_bytes());
    training_identity.extend_from_slice(&steps.to_le_bytes());
    training_identity.extend_from_slice(&epochs.to_le_bytes());
    let training_digest = CanonicalDigest::of_bytes(b"training-candidate-v1", &training_identity);
    let manifest = ArtifactManifest {
        schema_version: 1,
        architecture: "ruview-factorized-forecast-mixer-v1".to_string(),
        config: config.clone(),
        parameter_count: config.parameter_count()?,
        feature_schema_digest: *feature_digest.as_bytes(),
        training_manifest_digest: *training_digest.as_bytes(),
        weights_digest: *blake3::hash(&weights).as_bytes(),
        seed: request_value.optimizer.seed,
        release_epoch: 1,
        minimum_runtime_version: ruview_forecast_model::RUNTIME_COMPATIBILITY_VERSION,
        maximum_runtime_version: ruview_forecast_model::RUNTIME_COMPATIBILITY_VERSION,
        expires_at_unix_ms: None,
        build_id: format!("ruforecast-{}", env!("CARGO_PKG_VERSION")),
        teacher_outputs_used: false,
        independently_implemented: true,
    };
    let candidate = ModelArtifact::new(manifest.clone(), weights)?.encode()?;
    Ok((candidate, manifest))
}

fn synthetic_batch<B: AutodiffBackend<FloatElem = f32>>(
    spec: &SyntheticDatasetSpec,
    config: &ForecastModelConfig,
    first_window: u32,
    batch: usize,
    epoch: u16,
    device: &B::Device,
) -> Result<TrainingBatch<B>, TrainingError> {
    let variates = usize::from(spec.variates);
    let mut values = Vec::with_capacity(batch * config.context_len * variates);
    let mut observed = Vec::with_capacity(values.capacity());
    let mut ages = Vec::with_capacity(values.capacity());
    let mut context_time = Vec::with_capacity(batch * config.context_len * config.time_width);
    let mut future_time = Vec::with_capacity(batch * config.horizon * config.time_width);
    let mut descriptors = Vec::with_capacity(batch * variates * config.descriptor_width);
    let mut targets = Vec::with_capacity(batch * variates * config.horizon);
    let mut target_mask = Vec::with_capacity(targets.capacity());

    for sample in 0..batch {
        let window = u64::from(first_window) + sample as u64;
        let mut age = vec![0_u32; variates];
        for context in 0..config.context_len {
            for (variate, age_value) in age.iter_mut().enumerate() {
                let index = window
                    .wrapping_mul(1_000_003)
                    .wrapping_add((context as u64).wrapping_mul(97))
                    .wrapping_add((variate as u64).wrapping_mul(7_919))
                    .wrapping_add(spec.seed)
                    .wrapping_add(u64::from(epoch));
                let is_observed = splitmix64(index) % 1_000 >= u64::from(spec.missing_per_mille);
                let value = synthetic_value(window, context, variate);
                values.push(if is_observed { value } else { 0.0 });
                observed.push(if is_observed { 1.0 } else { 0.0 });
                if is_observed {
                    *age_value = 0;
                } else {
                    *age_value = age_value.saturating_add(1);
                }
                ages.push((*age_value as f32 / config.context_len as f32).min(1.0));
            }
            context_time.extend(time_features(window + context as u64, config.time_width));
        }
        for horizon in 0..config.horizon {
            future_time.extend(time_features(
                window + config.context_len as u64 + horizon as u64,
                config.time_width,
            ));
        }
        for variate in 0..variates {
            let mut descriptor = vec![0.0_f32; config.descriptor_width];
            descriptor[variate % 32] = 1.0;
            descriptor[32 + variate % 8] = 1.0;
            descriptors.extend(descriptor);
        }
        for variate in 0..variates {
            for horizon in 0..config.horizon {
                let time = config.context_len + horizon;
                targets.push(synthetic_value(window, time, variate));
                let present = splitmix64(
                    window
                        .wrapping_mul(65_537)
                        .wrapping_add(horizon as u64)
                        .wrapping_add((variate as u64) << 32)
                        .wrapping_add(spec.seed),
                ) % 1_000
                    >= u64::from(spec.missing_per_mille);
                target_mask.push(if present { 1.0 } else { 0.0 });
            }
        }
    }

    let input = ModelInput::new(
        config,
        Tensor::from_data(
            TensorData::new(values, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(observed, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(ages, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(context_time, [batch, config.context_len, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(future_time, [batch, config.horizon, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(descriptors, [batch, variates, config.descriptor_width]),
            device,
        ),
        Tensor::ones([batch, variates], device),
    )?;
    Ok(TrainingBatch {
        input,
        targets: Tensor::from_data(
            TensorData::new(targets, [batch, variates, config.horizon]),
            device,
        ),
        target_mask: Tensor::from_data(
            TensorData::new(target_mask, [batch, variates, config.horizon]),
            device,
        ),
    })
}

fn jsonl_batch<B: AutodiffBackend<FloatElem = f32>>(
    windows: &[JsonlWindow],
    config: &ForecastModelConfig,
    step_ms: u64,
    device: &B::Device,
) -> Result<TrainingBatch<B>, TrainingError> {
    let batch = windows.len();
    let variates = usize::from(windows[0].variates);
    if windows
        .iter()
        .any(|window| usize::from(window.variates) != variates)
    {
        return Err(TrainingError::Manifest(
            "one batch contains different variate counts".to_string(),
        ));
    }
    let context_cells = batch * config.context_len * variates;
    let target_cells = batch * variates * config.horizon;
    let mut values = Vec::with_capacity(context_cells);
    let mut observed = Vec::with_capacity(context_cells);
    let mut ages = Vec::with_capacity(context_cells);
    let mut context_time = Vec::with_capacity(batch * config.context_len * config.time_width);
    let mut future_time = Vec::with_capacity(batch * config.horizon * config.time_width);
    let mut descriptors = Vec::with_capacity(batch * variates * config.descriptor_width);
    let mut targets = Vec::with_capacity(target_cells);
    let mut target_mask = Vec::with_capacity(target_cells);
    for window in windows {
        let mut age = vec![0_u32; variates];
        for row in 0..config.context_len {
            for (variate, age_value) in age.iter_mut().enumerate() {
                let index = row * variates + variate;
                let present = window.observed_mask[index] == 1;
                values.push(if present { window.values[index] } else { 0.0 });
                observed.push(if present { 1.0 } else { 0.0 });
                if present {
                    *age_value = 0
                } else {
                    *age_value = age_value.saturating_add(1)
                }
                ages.push((*age_value as f32 / config.context_len as f32).min(1.0));
            }
            let timestamp = window
                .context_start_ms
                .checked_add((row as u64).saturating_mul(step_ms))
                .ok_or(TrainingError::Budget("context timestamp overflow"))?;
            context_time.extend(time_features(timestamp / 1_000, config.time_width));
        }
        for horizon in 0..config.horizon {
            let row = config.context_len + horizon;
            let timestamp = window
                .context_start_ms
                .checked_add((row as u64).saturating_mul(step_ms))
                .ok_or(TrainingError::Budget("future timestamp overflow"))?;
            future_time.extend(time_features(timestamp / 1_000, config.time_width));
        }
        for variate in 0..variates {
            let mut descriptor = vec![0.0_f32; config.descriptor_width];
            descriptor[variate % 32] = 1.0;
            descriptor[32 + variate % 8] = 1.0;
            descriptors.extend(descriptor);
        }
        targets.extend(window.targets.iter().copied());
        target_mask.extend(window.target_mask.iter().map(|value| f32::from(*value)));
    }
    let input = ModelInput::new(
        config,
        Tensor::from_data(
            TensorData::new(values, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(observed, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(ages, [batch, config.context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(context_time, [batch, config.context_len, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(future_time, [batch, config.horizon, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(descriptors, [batch, variates, config.descriptor_width]),
            device,
        ),
        Tensor::ones([batch, variates], device),
    )?;
    Ok(TrainingBatch {
        input,
        targets: Tensor::from_data(
            TensorData::new(targets, [batch, variates, config.horizon]),
            device,
        ),
        target_mask: Tensor::from_data(
            TensorData::new(target_mask, [batch, variates, config.horizon]),
            device,
        ),
    })
}

fn request_digest(request: &ValidatedTrainingRequest) -> CanonicalDigest {
    fn push_bytes(output: &mut Vec<u8>, input: &[u8]) {
        output.extend_from_slice(&(input.len() as u64).to_be_bytes());
        output.extend_from_slice(input);
    }

    let value = request.get();
    let mut bytes = Vec::with_capacity(256);
    push_bytes(&mut bytes, value.job_id.as_str().as_bytes());
    bytes.extend_from_slice(value.train.canonical_digest().as_bytes());
    bytes.push(match value.model {
        ModelProfile::TinyCi => 0,
        ModelProfile::LargeLinux => 1,
    });
    match value.device {
        TrainingDevice::Cpu => bytes.extend_from_slice(&[0, 0]),
        TrainingDevice::Cuda { ordinal } => bytes.extend_from_slice(&[1, ordinal]),
    }
    match &value.dataset {
        DatasetSource::Synthetic(generator) => {
            bytes.push(0);
            bytes.extend_from_slice(generator.canonical_digest().as_bytes());
        }
        DatasetSource::Manifest(input) => {
            bytes.push(1);
            push_bytes(&mut bytes, input.path.as_str().as_bytes());
            bytes.extend_from_slice(&input.size_bytes.to_be_bytes());
            bytes.extend_from_slice(input.sha256.as_bytes());
            bytes.extend_from_slice(&input.window_count.to_be_bytes());
            bytes.extend_from_slice(&input.variates.to_be_bytes());
            bytes.extend_from_slice(input.feature_schema_digest.as_bytes());
        }
    }
    bytes.extend_from_slice(&value.optimizer.epochs.to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.batch_size.to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.learning_rate.to_bits().to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.weight_decay.to_bits().to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.gradient_clip_norm.to_bits().to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.checkpoint_every_epochs.to_be_bytes());
    bytes.extend_from_slice(&value.optimizer.seed.to_be_bytes());
    bytes.extend_from_slice(&value.budget.max_optimizer_steps.to_be_bytes());
    bytes.extend_from_slice(&value.budget.max_wall_time_seconds.to_be_bytes());
    bytes.extend_from_slice(&value.budget.max_memory_bytes.to_be_bytes());
    bytes.extend_from_slice(&value.budget.max_artifact_bytes.to_be_bytes());
    bytes.extend_from_slice(&value.budget.max_checkpoints.to_be_bytes());
    match value.execution_binding() {
        Some(binding) => {
            bytes.push(1);
            bytes.extend_from_slice(binding.as_bytes());
        }
        None => bytes.push(0),
    }
    push_bytes(
        &mut bytes,
        option_env!("RUVIEW_CARGO_LOCK_SHA256")
            .unwrap_or("UNVERIFIED")
            .as_bytes(),
    );
    push_bytes(
        &mut bytes,
        option_env!("RUVIEW_SOURCE_COMMIT")
            .unwrap_or("UNVERIFIED")
            .as_bytes(),
    );
    push_bytes(
        &mut bytes,
        option_env!("RUVIEW_CONTAINER_DIGEST")
            .unwrap_or("UNVERIFIED")
            .as_bytes(),
    );
    push_bytes(
        &mut bytes,
        option_env!("RUVIEW_BUILD_TARGET")
            .unwrap_or("UNVERIFIED")
            .as_bytes(),
    );
    push_bytes(&mut bytes, env!("CARGO_PKG_VERSION").as_bytes());
    CanonicalDigest::of_bytes(b"ruview-training-request-v1", &bytes)
}

fn synthetic_value(window: u64, time: usize, variate: usize) -> f32 {
    let phase = (window as f64 * 0.013) + (time as f64 * 0.071) + (variate as f64 * 0.37);
    (phase.sin() + 0.25 * (phase * 0.17).cos()) as f32
}

fn time_features(step: u64, width: usize) -> Vec<f32> {
    let periods = [60.0_f64, 3_600.0, 86_400.0, 604_800.0];
    let mut values = Vec::with_capacity(width);
    for period in periods {
        let angle = std::f64::consts::TAU * (step as f64 % period) / period;
        values.push(angle.sin() as f32);
        values.push(angle.cos() as f32);
    }
    values.resize(width, 0.0);
    values.truncate(width);
    values
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Current Unix time in milliseconds, used only in receipts and policy checks.
pub fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactKind, ArtifactStore},
        cancel::NeverCancel,
        config::{synthetic_train_spec, JobId, OptimizerSpec, TrainingBudget, TrainingRequest},
    };
    use burn_autodiff::Autodiff;
    use burn_ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn synthetic_batch_is_deterministic_and_shaped() {
        type Backend = Autodiff<NdArray<f32>>;
        let config = ForecastModelConfig::tiny_ci();
        let spec = SyntheticDatasetSpec {
            windows: 2,
            variates: 3,
            missing_per_mille: 100,
            seed: 7,
        };
        let device = NdArrayDevice::default();
        let first = synthetic_batch::<Backend>(&spec, &config, 0, 2, 0, &device).unwrap();
        let second = synthetic_batch::<Backend>(&spec, &config, 0, 2, 0, &device).unwrap();
        assert_eq!(first.input.batch_variates(), [2, 3]);
        assert_eq!(first.targets.dims(), [2, 3, 12]);
        assert_eq!(first.targets.into_data(), second.targets.into_data());
    }

    #[test]
    fn retention_shorter_than_wall_budget_fails_before_publication() {
        let generator = SyntheticDatasetSpec {
            windows: 2,
            variates: 3,
            missing_per_mille: 0,
            seed: 7,
        };
        let retention_until_ms = unix_time_millis().saturating_add(30_000);
        let train =
            synthetic_train_spec(ModelProfile::TinyCi, &generator, false, retention_until_ms)
                .expect("synthetic contract");
        let request = TrainingRequest::new_local(
            JobId::new("expired-policy").expect("job"),
            train,
            DatasetSource::Synthetic(generator),
            ModelProfile::TinyCi,
            TrainingDevice::Cpu,
            OptimizerSpec {
                epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 1e-4,
                gradient_clip_norm: 1.0,
                checkpoint_every_epochs: 1,
                seed: 11,
            },
            TrainingBudget {
                max_optimizer_steps: 1,
                max_wall_time_seconds: 60,
                max_memory_bytes: 1024 * 1024 * 1024,
                max_artifact_bytes: 64 * 1024 * 1024,
                max_checkpoints: 1,
            },
        )
        .and_then(TrainingRequest::into_validated)
        .expect("validated request");
        let output = tempfile::tempdir().expect("artifact root");
        let result = BurnTrainer::new(ArtifactStore::new(output.path()).expect("store"))
            .train(&request, &NeverCancel);
        assert!(matches!(result, Err(TrainingError::RetentionExpired)));
        assert!(!output.path().join("expired-policy/model.mpk").exists());
    }

    #[test]
    fn partial_publication_failure_rolls_back_every_fixed_output() {
        let generator = SyntheticDatasetSpec {
            windows: 2,
            variates: 3,
            missing_per_mille: 0,
            seed: 7,
        };
        let train = synthetic_train_spec(ModelProfile::TinyCi, &generator, false, u64::MAX)
            .expect("synthetic contract");
        let request = TrainingRequest::new_local(
            JobId::new("publication-rollback").expect("job"),
            train,
            DatasetSource::Synthetic(generator),
            ModelProfile::TinyCi,
            TrainingDevice::Cpu,
            OptimizerSpec {
                epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 1e-4,
                gradient_clip_norm: 1.0,
                checkpoint_every_epochs: 1,
                seed: 11,
            },
            TrainingBudget {
                max_optimizer_steps: 1,
                max_wall_time_seconds: 60,
                max_memory_bytes: 1024 * 1024 * 1024,
                max_artifact_bytes: 64 * 1024 * 1024,
                max_checkpoints: 1,
            },
        )
        .and_then(TrainingRequest::into_validated)
        .expect("validated request");
        let output = tempfile::tempdir().expect("artifact root");
        let store = ArtifactStore::new(output.path()).expect("store");
        store.fail_after_successful_commits(1);
        let result = BurnTrainer::new(store.clone()).train(&request, &NeverCancel);
        assert!(matches!(result, Err(TrainingError::Artifact(_))));
        for kind in [
            ArtifactKind::Model,
            ArtifactKind::Manifest,
            ArtifactKind::Receipt,
            ArtifactKind::Checkpoint,
        ] {
            assert!(store
                .existing_descriptor(&request.get().job_id, kind)
                .expect("post-rollback lookup")
                .is_none());
        }
    }
}
