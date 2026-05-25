//! CLI argument types for `homecore-migrate`.
//!
//! Shared between `src/main.rs` and integration tests. The `clap`-derived
//! `Cli` struct is the entry-point; `Command` is the subcommand enum.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// homecore-migrate — migrate from Python Home Assistant to HOMECORE.
#[derive(Debug, Parser)]
#[command(name = "homecore-migrate", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect what is in the HA .storage directory and flag unsupported versions.
    Inspect(InspectArgs),
    /// Import entity registry from HA into a HOMECORE storage directory.
    ImportEntities(ImportEntitiesArgs),
    /// Import device registry (P1: parses and reports; wiring to HOMECORE P2).
    ImportDevices(ImportDevicesArgs),
    /// Inspect config entries (P1: count + domain list; conversion is P2).
    InspectConfigEntries(InspectConfigEntriesArgs),
    /// Parse secrets.yaml and report secret names (values redacted).
    InspectSecrets(InspectSecretsArgs),
    /// Count and list automations from automations.yaml (conversion is P2).
    InspectAutomations(InspectAutomationsArgs),
}

#[derive(Debug, clap::Args)]
pub struct InspectArgs {
    /// Path to the HA `.storage/` directory.
    #[arg(long)]
    pub storage: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ImportEntitiesArgs {
    /// Path to the HA `.storage/` directory.
    #[arg(long)]
    pub storage: PathBuf,
    /// Path to the HOMECORE storage directory (destination).
    #[arg(long)]
    pub to: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ImportDevicesArgs {
    /// Path to the HA `.storage/` directory.
    #[arg(long)]
    pub storage: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct InspectConfigEntriesArgs {
    /// Path to the HA `.storage/` directory.
    #[arg(long)]
    pub storage: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct InspectSecretsArgs {
    /// Path to the HA config directory (contains `secrets.yaml`).
    #[arg(long)]
    pub config_dir: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct InspectAutomationsArgs {
    /// Path to the HA config directory (contains `automations.yaml`).
    #[arg(long)]
    pub config_dir: PathBuf,
}
