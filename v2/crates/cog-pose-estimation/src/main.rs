//! `cog-pose-estimation` — Cognitum Cog binary entrypoint.
//!
//! Implements the ADR-100 runtime contract:
//!     cog-pose-estimation version
//!     cog-pose-estimation manifest
//!     cog-pose-estimation health
//!     cog-pose-estimation run --config <path>
//!
//! Each subcommand writes structured JSON to stdout. `run` is long-running
//! and emits one `pose.frame` event per inferred CSI window.

use clap::{Parser, Subcommand};
use cog_pose_estimation::{
    config::CogConfig,
    inference::{InferenceEngine, SyntheticInput},
    manifest::ManifestSpec,
    publisher::{emit_event, Event},
};
use std::path::PathBuf;

const COG_ID: &str = "pose-estimation";
const COG_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = COG_ID, version = COG_VERSION)]
#[command(about = "Cognitum Cog: 17-keypoint pose estimation from WiFi CSI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print `<id> <version>` and exit.
    Version,
    /// Print the embedded manifest as JSON.
    Manifest,
    /// One-shot health check. Exit 0 if the cog can come up healthy.
    Health,
    /// Long-running inference loop.
    Run {
        /// Path to runtime config JSON. See `cog/config.schema.json`.
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
        /// Optional per-room LoRA calibration adapter (ADR-150 §3.5): a safetensors with
        /// `fc1.a`/`fc1.b`/`fc2.a`/`fc2.b` low-rank deltas for this model's pose head,
        /// fitted from a short labeled in-room capture. Attaching it recovers accuracy in
        /// an unseen room/person. (Same mechanism as `aether-arena/calibration/`, but that
        /// reference tool targets the MM-Fi transformer model — adapters are model-specific.)
        #[arg(long, value_name = "PATH")]
        adapter: Option<PathBuf>,
    },
}

fn main() -> std::process::ExitCode {
    init_logging();

    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Version => cmd_version(),
        Cmd::Manifest => cmd_manifest(),
        Cmd::Health => cmd_health(),
        Cmd::Run { config, adapter } => cmd_run(config, adapter),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{COG_ID}: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .json()
        .try_init();
}

fn cmd_version() -> Result<(), Box<dyn std::error::Error>> {
    println!("{COG_ID} {COG_VERSION}");
    Ok(())
}

fn cmd_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let spec = ManifestSpec::embedded(COG_ID, COG_VERSION);
    println!("{}", serde_json::to_string_pretty(&spec)?);
    Ok(())
}

fn cmd_health() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new()?;
    let synthetic = SyntheticInput;
    let out = engine.infer(&synthetic.as_window())?;
    if out.is_finite() {
        emit_event(&Event::health_ok(COG_ID, engine.backend(), out.confidence));
        Ok(())
    } else {
        Err("inference produced non-finite output".into())
    }
}

fn cmd_run(
    config_path: PathBuf,
    adapter: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = CogConfig::load(&config_path)?;
    emit_event(&Event::run_started(COG_ID, &cfg));

    // Disclosure: pose_v1 has no confidence head, so every frame carries the
    // same `MODEL_TYPICAL_CONFIDENCE`. A `min_confidence` above that silently
    // suppresses *all* pose.frame events. Warn loudly rather than drop quietly.
    if cfg.min_confidence > cog_pose_estimation::inference::MODEL_TYPICAL_CONFIDENCE {
        tracing::warn!(
            min_confidence = cfg.min_confidence,
            model_typical_confidence = cog_pose_estimation::inference::MODEL_TYPICAL_CONFIDENCE,
            "configured min_confidence exceeds the model's typical confidence; \
             no pose.frame events will be emitted until this is lowered"
        );
    }

    let engine = InferenceEngine::with_adapter(adapter.as_deref())?;
    if engine.is_calibrated() {
        tracing::info!("per-room calibration adapter loaded");
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(cog_pose_estimation::runtime::run_loop(cfg, engine))?;
    Ok(())
}
