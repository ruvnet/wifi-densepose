//! `cog-person-count` — Cognitum Cog binary entrypoint.
//!
//! Implements the ADR-100 runtime contract:
//!     cog-person-count version
//!     cog-person-count manifest
//!     cog-person-count health
//!     cog-person-count run --config <path>

use clap::{Parser, Subcommand};
use cog_person_count::{
    inference::{InferenceEngine, SyntheticInput},
    publisher,
    COG_ID, COG_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cog-person-count", version = COG_VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Version,
    Manifest,
    Health,
    Run {
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct RunConfig {
    #[serde(default = "default_sensing_url")]
    sensing_url: String,
    model_path: Option<PathBuf>,
    #[serde(default = "default_poll_ms")]
    poll_ms: u64,
}

fn default_sensing_url() -> String { "http://127.0.0.1:3000/api/v1/sensing/latest".to_string() }
fn default_poll_ms() -> u64 { 40 }

fn main() -> std::process::ExitCode {
    init_logging();
    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Version => cmd_version(),
        Cmd::Manifest => cmd_manifest(),
        Cmd::Health => cmd_health(),
        Cmd::Run { config } => cmd_run(config),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cog-person-count: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .with_target(false)
        .try_init();
}

fn cmd_version() -> Result<(), Box<dyn std::error::Error>> {
    println!("{COG_ID} {COG_VERSION}");
    Ok(())
}

fn cmd_manifest() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&json!({
        "id": COG_ID,
        "version": COG_VERSION,
        "binary_url": Value::Null,
        "binary_bytes": Value::Null,
        "binary_sha256": Value::Null,
        "binary_signature": Value::Null,
        "installed_at": Value::Null,
        "status": Value::Null,
    }))?);
    Ok(())
}

fn cmd_health() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new()?;
    let pred = engine.infer(&SyntheticInput::default().as_window())?;
    if !pred.is_finite() {
        return Err("inference produced non-finite output".into());
    }
    publisher::health_ok(COG_ID, engine.backend(), &pred);
    Ok(())
}

fn cmd_run(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read config at {}: {}", config_path.display(), e))?;
    let cfg: RunConfig = serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse config at {}: {}", config_path.display(), e))?;

    let engine = InferenceEngine::with_weights(cfg.model_path.as_deref())?;
    publisher::run_started(
        COG_ID,
        &cfg.sensing_url,
        cfg.poll_ms,
        &cfg.model_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(auto-discover)".to_string()),
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(cog_person_count::runtime::run_loop(
        cog_person_count::runtime::RunConfig {
            sensing_url: cfg.sensing_url,
            poll_ms: cfg.poll_ms,
        },
        engine,
    ))
}
