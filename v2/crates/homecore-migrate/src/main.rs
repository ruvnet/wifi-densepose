//! `homecore-migrate` binary — CLI entry point.

use clap::Parser;
use homecore_migrate::cli::{Cli, Command};
use serde::Serialize;

#[derive(Serialize)]
struct ImportSummary {
    kind: &'static str,
    imported: usize,
    warning_count: usize,
    warnings: Vec<serde_json::Value>,
    destination: std::path::PathBuf,
}

fn print_summary(summary: &ImportSummary) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string(summary)?);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Inspect(args) => {
            println!(
                "Inspecting HA .storage directory: {}",
                args.storage.display()
            );
            // Probe entity_registry
            let entity_path = args.storage.join("core.entity_registry");
            if entity_path.exists() {
                match homecore_migrate::entity_registry::read_entity_registry(&entity_path) {
                    Ok(entries) => println!("  core.entity_registry: {} entities", entries.len()),
                    Err(e) => println!("  core.entity_registry: ERROR — {e}"),
                }
            }
            // Probe device_registry
            let device_path = args.storage.join("core.device_registry");
            if device_path.exists() {
                match homecore_migrate::device_registry::read_device_registry(&device_path) {
                    Ok(devices) => println!("  core.device_registry: {} devices", devices.len()),
                    Err(e) => println!("  core.device_registry: ERROR — {e}"),
                }
            }
            // Probe config_entries
            let ce_path = args.storage.join("core.config_entries");
            if ce_path.exists() {
                match homecore_migrate::config_entries::inspect_config_entries(&ce_path) {
                    Ok(s) => println!(
                        "  core.config_entries: {} entries, domains: {}",
                        s.count,
                        s.domains.join(", ")
                    ),
                    Err(e) => println!("  core.config_entries: ERROR — {e}"),
                }
            }
        }

        Command::ImportEntities(args) => {
            let entity_path = args.storage.join("core.entity_registry");
            let entries = homecore_migrate::entity_registry::read_entity_registry(&entity_path)?;
            let destination =
                homecore_migrate::entity_registry::write_entity_registry(&args.to, &entries)?;
            print_summary(&ImportSummary {
                kind: "entity_registry",
                imported: entries.len(),
                warning_count: 0,
                warnings: vec![],
                destination,
            })?;
        }

        Command::ImportDevices(args) => {
            let device_path = args.storage.join("core.device_registry");
            let devices = homecore_migrate::device_registry::read_device_registry(&device_path)?;
            let destination =
                homecore_migrate::device_registry::write_device_registry(&args.to, &devices)?;
            print_summary(&ImportSummary {
                kind: "device_registry",
                imported: devices.len(),
                warning_count: 0,
                warnings: vec![],
                destination,
            })?;
        }

        Command::InspectConfigEntries(args) => {
            let ce_path = args.storage.join("core.config_entries");
            let summary = homecore_migrate::config_entries::inspect_config_entries(&ce_path)?;
            println!(
                "config_entries: {} total, domains: {}",
                summary.count,
                summary.domains.join(", ")
            );
        }

        Command::ImportConfigEntries(args) => {
            let source = args.storage.join("core.config_entries");
            let converted = homecore_migrate::config_entries::convert_config_entries(&source)?;
            let imported = converted.data.entries.len();
            let warning_count = converted.data.warnings.len();
            let warnings = converted
                .data
                .warnings
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()?;
            let destination =
                homecore_migrate::config_entries::write_config_entries(&args.to, &converted)?;
            print_summary(&ImportSummary {
                kind: "config_entries",
                imported,
                warning_count,
                warnings,
                destination,
            })?;
        }

        Command::InspectSecrets(args) => {
            let secrets_path = args.config_dir.join("secrets.yaml");
            let secrets = homecore_migrate::secrets::read_secrets(&secrets_path)?;
            println!("{} secrets found:", secrets.len());
            let mut keys: Vec<_> = secrets.keys().collect();
            keys.sort();
            for k in keys {
                println!("  {} = <redacted>", k);
            }
        }

        Command::InspectAutomations(args) => {
            let auto_path = args.config_dir.join("automations.yaml");
            let summary = homecore_migrate::automations::read_automations(&auto_path)?;
            println!("{} automations:", summary.count);
            for a in &summary.automations {
                println!(
                    "  id={} alias={}",
                    a.id,
                    a.alias.as_deref().unwrap_or("<unnamed>")
                );
            }
        }
    }

    Ok(())
}
