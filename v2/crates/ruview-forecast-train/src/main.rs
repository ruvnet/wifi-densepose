use std::path::PathBuf;
#[cfg(all(feature = "server", any(feature = "cpu", feature = "cuda")))]
use std::sync::Arc;

#[cfg(any(feature = "fal-client", feature = "server"))]
use anyhow::Context;
use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ruview_forecast_core::{
    CanonicalDigest, DataPolicy, HoldoutKey, NormalizationPolicy, PrivacyClass, QuantileSet,
    SeriesKey, SplitMember, SplitStrategy, TemporalSplitPlan, TimeRange,
};
#[cfg(any(feature = "fal-client", feature = "cpu", feature = "cuda"))]
use ruview_forecast_train::config::SyntheticDatasetSpec;
use ruview_forecast_train::config::{
    DatasetInput, DatasetSource, JobId, LocalTrainSpecWire, LocalTrainingRequestWire, ModelProfile,
    OptimizerSpec, RelativeDataPath, Sha256Digest, TrainingBudget, TrainingDevice,
};
use ruview_forecast_train::corpus::JsonlWindow;

// Keep all executable paths in Rust and all training choices typed. There is
// intentionally no command/argument passthrough to a shell.
#[cfg(all(feature = "fal-client", not(any(feature = "cpu", feature = "cuda"))))]
use ruview_forecast_train::config::synthetic_train_spec;
#[cfg(any(feature = "cpu", feature = "cuda"))]
use ruview_forecast_train::{
    artifact::ArtifactStore,
    config::{synthetic_train_spec, TrainingRequest},
};
#[cfg(any(feature = "cpu", feature = "cuda"))]
use ruview_forecast_train::{cancel::NeverCancel, config::load_request};

#[derive(Parser)]
#[command(
    name = "ruforecast",
    version,
    about = "RuView Forecast training and fal orchestration"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a tiny real Burn optimizer step on deterministic synthetic data.
    Smoke {
        /// Private artifact root.
        #[arg(long, default_value = "./artifacts")]
        output: PathBuf,
        /// Stable idempotency key.
        #[arg(long, default_value = "smoke")]
        job_id: String,
        /// Generated windows.
        #[arg(long, default_value_t = 4)]
        windows: u32,
    },
    /// Train locally from a validated JSON/TOML request.
    TrainLocal {
        /// Request file.
        #[arg(long)]
        request: PathBuf,
        /// Root beneath which a manifest's relative JSONL shard is opened.
        #[arg(long)]
        dataset_root: PathBuf,
        /// Private artifact root.
        #[arg(long)]
        output: PathBuf,
    },
    /// Create a complete local JSONL and TOML specimen in a new directory.
    PrepareLocalExample {
        /// New directory that will receive `train.jsonl` and `train-local.toml`.
        #[arg(long)]
        directory: PathBuf,
    },
    /// Run fal Direct Server routes on the configured bind address.
    Serve {
        /// Bind address.
        #[arg(long, default_value = "0.0.0.0:8000")]
        bind: String,
        /// Candidate-only artifact root.
        #[arg(long, default_value = "/data/ruview-forecast/artifacts")]
        output: PathBuf,
    },
    /// Synthetic-only fal queue operations.
    Fal {
        #[command(subcommand)]
        command: FalCommand,
    },
    /// Decode and integrity-check an unsigned candidate. It does not activate it.
    VerifyCandidate {
        /// Candidate envelope path.
        #[arg(long)]
        candidate: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HostedModelProfile {
    TinyCi,
    LargeLinux,
}

#[cfg(feature = "fal-client")]
impl From<HostedModelProfile> for ModelProfile {
    fn from(value: HostedModelProfile) -> Self {
        match value {
            HostedModelProfile::TinyCi => Self::TinyCi,
            HostedModelProfile::LargeLinux => Self::LargeLinux,
        }
    }
}

#[derive(Subcommand)]
enum FalCommand {
    /// Submit a redacted synthetic recipe exactly once.
    Submit {
        /// Generated windows.
        #[arg(long, default_value_t = 1024)]
        windows: u32,
        /// Coupled variates.
        #[arg(long, default_value_t = 8)]
        variates: u16,
        /// Generator seed.
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Allowlisted fixed architecture.
        #[arg(long, value_enum, default_value = "tiny-ci")]
        model_profile: HostedModelProfile,
        /// Full passes over the generated windows.
        #[arg(long, default_value_t = 1)]
        epochs: u16,
        /// Windows per optimizer step.
        #[arg(long, default_value_t = 8)]
        batch_size: u16,
        /// AdamW learning rate.
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// AdamW decoupled weight decay.
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f64,
        /// Gradient norm cap.
        #[arg(long, default_value_t = 1.0)]
        gradient_clip_norm: f64,
        /// Worker wall-clock ceiling.
        #[arg(long, default_value_t = 3_300)]
        max_wall_time_seconds: u64,
        /// Operator-approved provider billable-seconds ceiling.
        #[arg(long, default_value_t = 3_300)]
        max_billable_seconds: u64,
        /// Conservative peak worker-memory reservation.
        #[arg(long, default_value_t = 32 * 1024 * 1024 * 1024)]
        max_memory_bytes: u64,
        /// Operator-approved maximum micro-USD.
        #[arg(long)]
        max_micro_usd: u64,
        /// Acknowledge that fal does not enforce this client-side cost ceiling.
        #[arg(long, action = clap::ArgAction::SetTrue, required = true)]
        ack_unenforced_provider_cost: bool,
        /// Reservation lifetime; also transmitted and digest-bound.
        #[arg(long, default_value_t = 6_300)]
        expires_in_seconds: u64,
    },
    /// Read queue status.
    Status {
        /// Provider request id.
        #[arg(long)]
        request_id: String,
        /// Redacted request SHA-256 returned by submit.
        #[arg(long)]
        request_digest: String,
        /// Fresh opaque hosted job digest returned by submit.
        #[arg(long)]
        job_digest: String,
        /// Candidate expiry returned by submit.
        #[arg(long)]
        artifacts_expire_at_ms: u64,
        /// Cumulative artifact-byte cap returned by submit.
        #[arg(long)]
        max_artifact_bytes: u64,
    },
    /// Request queue cancellation.
    Cancel {
        /// Provider request id.
        #[arg(long)]
        request_id: String,
        /// Redacted request SHA-256.
        #[arg(long)]
        request_digest: String,
        /// Fresh opaque hosted job digest.
        #[arg(long)]
        job_digest: String,
        /// Candidate expiry returned by submit.
        #[arg(long)]
        artifacts_expire_at_ms: u64,
        /// Cumulative artifact-byte cap returned by submit.
        #[arg(long)]
        max_artifact_bytes: u64,
    },
    /// Retrieve and strictly validate the four-descriptor hosted result.
    Result {
        /// Provider request id.
        #[arg(long)]
        request_id: String,
        /// Exact redacted request digest.
        #[arg(long)]
        request_digest: String,
        /// Fresh opaque hosted job digest.
        #[arg(long)]
        job_digest: String,
        /// Candidate expiry returned by submit.
        #[arg(long)]
        artifacts_expire_at_ms: u64,
        /// Cumulative artifact-byte cap returned by submit.
        #[arg(long)]
        max_artifact_bytes: u64,
    },
    /// Retrieve the result and export all four artifacts into quarantine.
    Download {
        /// Provider request id.
        #[arg(long)]
        request_id: String,
        /// Exact redacted request digest.
        #[arg(long)]
        request_digest: String,
        /// Fresh opaque hosted job digest.
        #[arg(long)]
        job_digest: String,
        /// Candidate expiry returned by submit.
        #[arg(long)]
        artifacts_expire_at_ms: u64,
        /// Cumulative artifact-byte cap returned by submit.
        #[arg(long)]
        max_artifact_bytes: u64,
        /// Caller-selected root; artifact names remain capability-confined.
        #[arg(long)]
        quarantine: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    match Cli::parse().command {
        Command::Smoke {
            output,
            job_id,
            windows,
        } => smoke(output, job_id, windows),
        Command::TrainLocal {
            request,
            dataset_root,
            output,
        } => train_local(request, dataset_root, output),
        Command::PrepareLocalExample { directory } => prepare_local_example(directory),
        Command::Serve { bind, output } => serve(bind, output).await,
        Command::Fal { command } => fal(command).await,
        Command::VerifyCandidate { candidate } => verify_candidate(candidate),
    }
}

fn prepare_local_example(directory: PathBuf) -> Result<()> {
    let mut directory_builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        directory_builder.mode(0o700);
    }
    directory_builder.create(&directory)?;

    let model = ModelProfile::TinyCi.config();
    let train_key = SeriesKey::new("example-train", "device-a", "session-a")?;
    let test_key = SeriesKey::new("example-test", "device-b", "session-b")?;
    let variates = 3_usize;
    let window = JsonlWindow {
        version: 1,
        series_key: train_key.clone(),
        context_start_ms: 1_000,
        variates: u16::try_from(variates)?,
        values: (0..model.context_len * variates)
            .map(|index| (index as f32 * 0.01).sin())
            .collect(),
        observed_mask: vec![1; model.context_len * variates],
        targets: (0..model.horizon * variates)
            .map(|index| (index as f32 * 0.02).cos())
            .collect(),
        target_mask: vec![1; model.horizon * variates],
    };
    let mut shard = serde_json::to_vec(&window)?;
    shard.push(b'\n');
    write_private_new(&directory.join("train.jsonl"), &shard)?;

    let sha256 = Sha256Digest::of_bytes(&shard);
    let feature_schema_digest = CanonicalDigest::of_bytes(
        b"ruview-local-example-feature-schema-v1",
        b"example-0,example-1,example-2",
    );
    let split_plan = TemporalSplitPlan::new(
        SplitStrategy::EntityHoldout(HoldoutKey::Strict),
        vec![SplitMember::new(train_key, TimeRange::new(1, 100_000)?)],
        vec![],
        vec![SplitMember::new(test_key, TimeRange::new(1, 100_000)?)],
        model.horizon,
        1_000,
        0,
    )?;
    let policy = DataPolicy::new(
        PrivacyClass::P3,
        "synthetic-example",
        "synthetic-example",
        "synthetic-example",
        "local-pipeline-validation-only",
        CanonicalDigest::of_bytes(b"ruview-local-example-policy-v1", b"not-real-approval"),
        None,
        None,
        None,
        unix_ms().saturating_add(7 * 24 * 60 * 60 * 1_000),
        true,
    )?;
    let request = LocalTrainingRequestWire {
        job_id: JobId::new("local-example")?,
        train: LocalTrainSpecWire {
            context_length: model.context_len,
            horizon: model.horizon,
            step_ms: 1_000,
            quantiles: QuantileSet::new(model.quantiles.to_vec())?,
            split_plan,
            normalization: NormalizationPolicy::None,
            dataset_digest: CanonicalDigest::of_bytes(
                b"ruview-jsonl-window-shard-v1",
                sha256.as_bytes(),
            ),
            policy,
        },
        dataset: DatasetSource::Manifest(DatasetInput {
            path: RelativeDataPath::new("train.jsonl")?,
            size_bytes: u64::try_from(shard.len())?,
            sha256,
            window_count: 1,
            variates: u16::try_from(variates)?,
            feature_schema_digest,
        }),
        model: ModelProfile::TinyCi,
        device: TrainingDevice::Cpu,
        optimizer: OptimizerSpec {
            epochs: 1,
            batch_size: 1,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            gradient_clip_norm: 1.0,
            checkpoint_every_epochs: 1,
            seed: 11,
        },
        budget: TrainingBudget {
            max_optimizer_steps: 1,
            max_wall_time_seconds: 300,
            max_memory_bytes: 4 * 1024 * 1024 * 1024,
            max_artifact_bytes: 512 * 1024 * 1024,
            max_checkpoints: 1,
        },
    };
    let request_bytes = toml::to_string_pretty(&request)?.into_bytes();
    write_private_new(&directory.join("train-local.toml"), &request_bytes)?;
    println!(
        "created synthetic local-only example at {}; replace its data, split, schema digest, and policy with governed values before real training",
        directory.display()
    );
    Ok(())
}

fn write_private_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn smoke(output: PathBuf, job_id: String, windows: u32) -> Result<()> {
    #[cfg(not(feature = "cpu"))]
    {
        let _ = (output, job_id, windows);
        bail!("rebuild with --features cpu,cli");
    }
    #[cfg(feature = "cpu")]
    {
        let generator = SyntheticDatasetSpec {
            windows,
            variates: 3,
            missing_per_mille: 50,
            seed: 7,
        };
        let train = synthetic_train_spec(ModelProfile::TinyCi, &generator, false, u64::MAX)?;
        let request = TrainingRequest::new_local(
            JobId::new(job_id)?,
            train,
            ruview_forecast_train::config::DatasetSource::Synthetic(generator),
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
                max_optimizer_steps: u64::from(windows).div_ceil(2),
                max_wall_time_seconds: 300,
                max_memory_bytes: 4 * 1024 * 1024 * 1024,
                max_artifact_bytes: 512 * 1024 * 1024,
                max_checkpoints: 1,
            },
        )?
        .into_validated()?;
        let outcome = ruview_forecast_train::runner::BurnTrainer::new(ArtifactStore::new(output)?)
            .train(&request, &NeverCancel)?;
        println!("{}", serde_json::to_string_pretty(&outcome)?);
        Ok(())
    }
}

fn train_local(request: PathBuf, dataset_root: PathBuf, output: PathBuf) -> Result<()> {
    #[cfg(not(any(feature = "cpu", feature = "cuda")))]
    {
        let _ = (request, dataset_root, output);
        bail!("rebuild with --features cpu,cli or cuda,cli");
    }
    #[cfg(any(feature = "cpu", feature = "cuda"))]
    {
        let request = load_request(&request)?;
        let trainer = ruview_forecast_train::runner::BurnTrainer::with_dataset_root(
            ArtifactStore::new(output)?,
            dataset_root,
        );
        let outcome = trainer.train(&request, &NeverCancel)?;
        println!("{}", serde_json::to_string_pretty(&outcome)?);
        Ok(())
    }
}

async fn serve(bind: String, output: PathBuf) -> Result<()> {
    #[cfg(not(all(feature = "server", any(feature = "cpu", feature = "cuda"))))]
    {
        let _ = (bind, output);
        bail!("rebuild with --features cli,server,cpu or cli,server,cuda");
    }
    #[cfg(all(feature = "server", any(feature = "cpu", feature = "cuda")))]
    {
        let executor = Arc::new(BurnServerExecutor {
            artifacts: ArtifactStore::new(output)?,
        });
        let listener = tokio::net::TcpListener::bind(&bind)
            .await
            .context("bind Direct Server")?;
        axum::serve(listener, ruview_forecast_train::server::router(executor)).await?;
        Ok(())
    }
}

#[cfg(all(feature = "server", any(feature = "cpu", feature = "cuda")))]
struct BurnServerExecutor {
    artifacts: ArtifactStore,
}
#[cfg(all(feature = "server", any(feature = "cpu", feature = "cuda")))]
impl ruview_forecast_train::server::SyntheticJobExecutor for BurnServerExecutor {
    fn execute(
        &self,
        payload: ruview_forecast_train::fal::HostedSyntheticPayload,
        cancel: &ruview_forecast_train::cancel::CancelToken,
    ) -> Result<ruview_forecast_train::fal::HostedTrainingOutcome, String> {
        payload
            .validate_for_worker(unix_ms())
            .map_err(|e| e.to_string())?;
        let request_digest = payload.request_digest;
        let job_digest = payload.job_digest;
        let worker_build_id = payload.worker_build_id.clone();
        let build_manifest_digest = payload.build_manifest_digest;
        let artifacts_expire_at_ms = payload.expires_at_ms;
        let generator = SyntheticDatasetSpec {
            windows: payload.windows,
            variates: payload.variates,
            missing_per_mille: payload.missing_per_mille,
            seed: payload.generator_seed,
        };
        let train = synthetic_train_spec(
            payload.model_profile,
            &generator,
            false,
            payload.expires_at_ms,
        )
        .map_err(|e| e.to_string())?;
        let optimizer = OptimizerSpec {
            epochs: payload.optimizer.epochs,
            batch_size: payload.optimizer.batch_size,
            learning_rate: payload.optimizer.learning_rate,
            weight_decay: payload.optimizer.weight_decay,
            gradient_clip_norm: payload.optimizer.gradient_clip_norm,
            checkpoint_every_epochs: payload.optimizer.epochs,
            seed: payload.optimizer.seed,
        };
        #[cfg(feature = "cuda")]
        let device = TrainingDevice::Cuda { ordinal: 0 };
        #[cfg(all(not(feature = "cuda"), feature = "cpu"))]
        let device = TrainingDevice::Cpu;
        let job_id = JobId::new(payload.job_digest.to_hex()).map_err(|e| e.to_string())?;
        self.schedule_job_cleanup(&job_id, artifacts_expire_at_ms);
        let request = TrainingRequest::new_local(
            job_id,
            train,
            ruview_forecast_train::config::DatasetSource::Synthetic(generator),
            payload.model_profile,
            device,
            optimizer,
            TrainingBudget {
                max_optimizer_steps: payload.budget.max_optimizer_steps,
                max_wall_time_seconds: payload.budget.max_wall_time_seconds,
                max_memory_bytes: payload.budget.max_memory_bytes,
                max_artifact_bytes: payload.budget.max_artifact_bytes,
                max_checkpoints: 1,
            },
        )
        .and_then(|request| {
            request.bind_hosted_synthetic_execution(
                ruview_forecast_core::CanonicalDigest::from_bytes(*request_digest.as_bytes()),
            )
        })
        .and_then(|r| r.into_validated())
        .map_err(|e| e.to_string())?;
        let outcome = ruview_forecast_train::runner::BurnTrainer::new(self.artifacts.clone())
            .train(&request, cancel)
            .map_err(|e| e.to_string())?;
        let hosted = ruview_forecast_train::fal::HostedTrainingOutcome::new(
            request_digest,
            job_digest,
            worker_build_id,
            build_manifest_digest,
            artifacts_expire_at_ms,
            outcome.candidate,
            outcome.manifest,
            outcome.receipt,
            outcome.checkpoint,
        )
        .map_err(|error| error.to_string())?;
        Ok(hosted)
    }
}

#[cfg(all(feature = "server", any(feature = "cpu", feature = "cuda")))]
impl BurnServerExecutor {
    fn schedule_job_cleanup(&self, job_id: &JobId, expires_at_ms: u64) {
        let job_directory = self.artifacts.root().join(job_id.as_str());
        std::thread::spawn(move || {
            loop {
                let now = unix_ms();
                if now >= expires_at_ms {
                    break;
                }
                let remaining = expires_at_ms.saturating_sub(now).min(60_000);
                std::thread::sleep(std::time::Duration::from_millis(remaining.max(1)));
            }
            for kind in [
                ruview_forecast_train::artifact::ArtifactKind::Model,
                ruview_forecast_train::artifact::ArtifactKind::Manifest,
                ruview_forecast_train::artifact::ArtifactKind::Receipt,
                ruview_forecast_train::artifact::ArtifactKind::Checkpoint,
            ] {
                let filename = kind.filename();
                let _ = std::fs::remove_file(job_directory.join(filename));
                let _ = std::fs::remove_file(job_directory.join(format!(".{filename}.partial")));
            }
            for lock in [".job.lock", ".run.lock"] {
                let _ = std::fs::remove_file(job_directory.join(lock));
            }
            let _ = std::fs::remove_dir(job_directory);
        });
    }
}

async fn fal(command: FalCommand) -> Result<()> {
    #[cfg(not(feature = "fal-client"))]
    {
        let _ = command;
        bail!("rebuild with --features cli,fal-client");
    }
    #[cfg(feature = "fal-client")]
    {
        use ruview_forecast_train::fal::*;
        let key = FalKey::new(std::env::var("FAL_KEY").context("FAL_KEY is required")?)?;
        match command {
            FalCommand::Submit {
                windows,
                variates,
                seed,
                model_profile,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
                gradient_clip_norm,
                max_wall_time_seconds,
                max_billable_seconds,
                max_memory_bytes,
                max_micro_usd,
                ack_unenforced_provider_cost,
                expires_in_seconds,
            } => {
                if !ack_unenforced_provider_cost {
                    bail!("hosted submit requires explicit acknowledgement of unenforced cost");
                }
                let now = unix_ms();
                let expires_at_ms = expires_in_seconds
                    .checked_mul(1_000)
                    .and_then(|duration| now.checked_add(duration))
                    .context("hosted expiry overflow")?;
                let generator = SyntheticDatasetSpec {
                    windows,
                    variates,
                    missing_per_mille: 50,
                    seed,
                };
                let model_profile = ModelProfile::from(model_profile);
                let train = synthetic_train_spec(
                    model_profile,
                    &generator,
                    true,
                    now.saturating_add(86_400_000),
                )?;
                let optimizer = OptimizerSpec {
                    epochs,
                    batch_size,
                    learning_rate,
                    weight_decay,
                    gradient_clip_norm,
                    checkpoint_every_epochs: epochs,
                    seed: 11,
                };
                optimizer.validate()?;
                let max_optimizer_steps = u64::from(windows)
                    .div_ceil(u64::from(batch_size))
                    .checked_mul(u64::from(epochs))
                    .context("hosted optimizer-step count overflow")?;
                let budget = HostedBudget {
                    max_optimizer_steps,
                    max_wall_time_seconds,
                    max_billable_seconds,
                    max_micro_usd,
                    max_artifact_bytes: ruview_forecast_model::MAX_ARTIFACT_BYTES as u64,
                    max_memory_bytes,
                    cost_basis: HostedCostBasis::UnmeasuredOperatorCap,
                };
                let (worker_build_id, build_manifest_digest) = configured_worker_identity()?;
                let plan = ReservedSyntheticSubmission::reserve(
                    &train,
                    &generator,
                    model_profile,
                    &optimizer,
                    budget,
                    worker_build_id,
                    build_manifest_digest,
                    expires_at_ms,
                    now,
                )?;
                let client = FalQueueClient::new(key, configured_fal_app()?)?;
                let handle = client.submit(&plan).await?;
                println!("{}", serde_json::to_string_pretty(&handle)?);
            }
            FalCommand::Status {
                request_id,
                request_digest,
                job_digest,
                artifacts_expire_at_ms,
                max_artifact_bytes,
            } => {
                let c = FalQueueClient::new(key, configured_fal_app()?)?;
                let h = fal_handle(
                    request_id,
                    request_digest,
                    job_digest,
                    artifacts_expire_at_ms,
                    max_artifact_bytes,
                )?;
                println!("{}", serde_json::to_string_pretty(&c.status(&h).await?)?);
            }
            FalCommand::Cancel {
                request_id,
                request_digest,
                job_digest,
                artifacts_expire_at_ms,
                max_artifact_bytes,
            } => {
                let c = FalQueueClient::new(key, configured_fal_app()?)?;
                let h = fal_handle(
                    request_id,
                    request_digest,
                    job_digest,
                    artifacts_expire_at_ms,
                    max_artifact_bytes,
                )?;
                c.cancel(&h).await?;
                println!("cancel accepted");
            }
            FalCommand::Result {
                request_id,
                request_digest,
                job_digest,
                artifacts_expire_at_ms,
                max_artifact_bytes,
            } => {
                let c = FalQueueClient::new(key, configured_fal_app()?)?;
                let h = fal_handle(
                    request_id,
                    request_digest,
                    job_digest,
                    artifacts_expire_at_ms,
                    max_artifact_bytes,
                )?;
                println!("{}", serde_json::to_string_pretty(&c.result(&h).await?)?);
            }
            FalCommand::Download {
                request_id,
                request_digest,
                job_digest,
                artifacts_expire_at_ms,
                max_artifact_bytes,
                quarantine,
            } => {
                let c = FalQueueClient::new(key, configured_fal_app()?)?;
                let h = fal_handle(
                    request_id,
                    request_digest,
                    job_digest,
                    artifacts_expire_at_ms,
                    max_artifact_bytes,
                )?;
                let outcome = c.result(&h).await?;
                let store = ruview_forecast_train::artifact::ArtifactStore::new(quarantine)?;
                let downloaded = c.download_outcome(&h, &outcome, &store).await?;
                println!("{}", serde_json::to_string_pretty(&downloaded)?);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "fal-client")]
fn configured_fal_app() -> Result<ruview_forecast_train::fal::FalApp> {
    let value = std::env::var("RUVIEW_FAL_APP").context("RUVIEW_FAL_APP is required")?;
    Ok(ruview_forecast_train::fal::FalApp::new(value)?)
}

#[cfg(feature = "fal-client")]
fn fal_handle(
    request_id: String,
    request_digest: String,
    job_digest: String,
    artifacts_expire_at_ms: u64,
    max_artifact_bytes: u64,
) -> Result<ruview_forecast_train::fal::FalRequestHandle> {
    let (worker_build_id, build_manifest_digest) = configured_worker_identity()?;
    Ok(ruview_forecast_train::fal::FalRequestHandle {
        request_id: ruview_forecast_train::fal::FalRequestId::new(request_id)?,
        request_digest: Sha256Digest::from_hex(&request_digest)?,
        job_digest: Sha256Digest::from_hex(&job_digest)?,
        worker_build_id,
        build_manifest_digest,
        artifacts_expire_at_ms,
        max_artifact_bytes,
    })
}

#[cfg(feature = "fal-client")]
fn configured_worker_identity() -> Result<(String, Sha256Digest)> {
    let worker_build_id =
        std::env::var("RUVIEW_WORKER_BUILD_ID").context("RUVIEW_WORKER_BUILD_ID is required")?;
    let build_manifest_sha256 = std::env::var("RUVIEW_BUILD_MANIFEST_SHA256")
        .context("RUVIEW_BUILD_MANIFEST_SHA256 is required")?;
    Ok((
        worker_build_id,
        Sha256Digest::from_hex(&build_manifest_sha256)?,
    ))
}

fn verify_candidate(path: PathBuf) -> Result<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(ruview_forecast_model::MAX_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > ruview_forecast_model::MAX_ARTIFACT_BYTES {
        bail!("candidate exceeds cap");
    }
    let candidate = ruview_forecast_model::ModelArtifact::decode(&bytes)?;
    println!(
        "candidate valid (UNTRUSTED, unsigned): {}",
        candidate
            .manifest()
            .training_manifest_digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    Ok(())
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_submit_requires_cost_ack_and_accepts_typed_large_profile() {
        assert!(
            Cli::try_parse_from(["ruforecast", "fal", "submit", "--max-micro-usd", "1000",])
                .is_err()
        );
        assert!(Cli::try_parse_from([
            "ruforecast",
            "fal",
            "submit",
            "--max-micro-usd",
            "1000",
            "--ack-unenforced-provider-cost",
            "--model-profile",
            "large-linux",
            "--epochs",
            "2",
            "--batch-size",
            "2",
        ])
        .is_ok());
        assert!(Cli::try_parse_from([
            "ruforecast",
            "fal",
            "status",
            "--app",
            "attacker/app",
            "--request-id",
            "request",
            "--request-digest",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--job-digest",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--artifacts-expire-at-ms",
            "1",
            "--max-artifact-bytes",
            "4",
        ])
        .is_err());
    }
}
