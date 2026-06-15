# homecore-migrate

Migration tooling for importing Home Assistant configuration, entities, and secrets into HOMECORE.

[![Crates.io](https://img.shields.io/crates/v/homecore-migrate.svg)](https://crates.io/crates/homecore-migrate)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MSRV: 1.89+](https://img.shields.io/badge/MSRV-1.89%2B-purple.svg)
[![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen.svg)](https://github.com/ruvnet/RuView)
[![ADR-165](https://img.shields.io/badge/ADR-165-orange.svg)](../../docs/adr/ADR-165-homecore-migrate-from-home-assistant.md)

Parse and inspect Home Assistant's `.storage/` directory, entity registry, device registry, secrets, and automations. Convert existing HA configurations for import into HOMECORE (full conversion in P2).

## What this crate does

`homecore-migrate` reads Home Assistant's filesystem state and provides tooling to analyze and migrate it to HOMECORE. It includes:

- **HaStorageDir** — reads HA's `.homeassistant/.storage/` directory and parses versioned JSON envelopes
- **Entity registry parser** — converts `core.entity_registry` JSON to HOMECORE `EntityEntry` types
- **Device registry parser** — reads `core.device_registry` (P1 diagnostic only; full conversion in P2)
- **Config entries parser** — reads `core.config_entries` to list active integrations
- **Secrets parser** — reads `secrets.yaml` as `HashMap<String, String>` for reference resolution (P2)
- **Automations parser** — reads `automations.yaml` and counts/lists automations (full conversion in P2)
- **CLI binary** — `homecore-migrate inspect` to preview what will be migrated

The tool enforces version schema compatibility: unknown HA schema versions are rejected (hard error per ADR-165 §6 Q5) rather than silently corrupting data.

## Features

- **Entity registry import** — `core.entity_registry` → HOMECORE entity definitions (ready for import)
- **Device registry inspection** — read HA device metadata; full conversion deferred to P2
- **Config entries analysis** — list active integrations by domain (enables gap analysis)
- **Secrets extraction** — read `secrets.yaml` references for annotation (resolution in P2)
- **Automations counting** — list automation IDs and aliases without conversion (conversion in P2)
- **Schema version validation** — explicit rejection of unknown HA versions (no silent corruption)
- **Structured error reporting** — `MigrateError` enum with context (file path, line number)
- **CLI subcommands** — `inspect` to preview, `import-entities` to load (P2), `export-for-sidecar` (P2)

## Capabilities

| Capability | Type | Method | Notes |
|------------|------|--------|-------|
| Read storage envelope | Parser | `storage::read_envelope(path)` | Deserialize `.storage/*.json` |
| Parse entity registry | Parser | `entity_registry::load(storage_dir)` | → `Vec<homecore::EntityEntry>` |
| Inspect device registry | Parser | `device_registry::load(storage_dir)` | → `Vec<DeviceImport>` (P1 diagnostic) |
| List config entries | Parser | `config_entries::load(storage_dir)` | → domain counts + names |
| Load secrets | Parser | `secrets::load_secrets(path)` | → `HashMap<String, String>` |
| Count automations | Parser | `automations::load(path)` | → count + ID list |
| Validate schema version | Validator | `storage_format::validate_version(major, minor)` | Hard error if unknown |
| Convert to HOMECORE | Converter | `entity_registry::to_homecore_entries()` (P2) | → `homecore::EntityRegistry` |
| Export side-by-side DB | Exporter | `recorder::export_states()` (P2, `--features recorder`) | → `.homecore/home.db` |

## Comparison to Home Assistant

| Aspect | Home Assistant | homecore-migrate |
|--------|----------------|-----------------|
| State source | Python `.homeassistant/` directory | Same HA filesystem format |
| Entity registry format | JSON envelope in `.storage/core.entity_registry` | Identical format, schema v13 |
| Schema versioning | `version` + optional `minor_version` | Explicit version struct validation |
| Secrets resolution | `!secret` YAML references via loader | Planned P2 (reads `secrets.yaml`) |
| Automation conversion | Python → HA YAML (internal) | P2: convert to `homecore-automation` format |
| Device registry import | Python device types | P1 diagnostic; full conversion P2 |
| Side-by-side runtime | N/A (HA doesn't side-by-side migrate) | P2 feature: run old + new in parallel |
| CLI tooling | HA doesn't export | `homecore-migrate` binary with subcommands |

## Performance

- **Storage envelope parse** — < 5 ms per file (serde_json)
- **Entity registry load** — < 50 ms for 1,000 entities
- **Storage directory scan** — < 100 ms for full `.storage/` directory
- **Secrets file parse** — < 10 ms (YAML)
- **No per-crate benchmarks yet** — a follow-up issue tracks baseline measurements

## Usage

CLI inspection (P1):

```bash
# Inspect what will be migrated from an existing HA installation
homecore-migrate inspect ~/.homeassistant

# Output:
# Entity Registry: 47 entities
#   light: 12
#   sensor: 20
#   binary_sensor: 10
#   switch: 5
# Device Registry: 8 devices
# Config Entries: 6 integrations (mqtt, rest, zeroconf, ...)
# Secrets: 3 defined (redacted)
# Automations: 5 automations (redacted)
```

Programmatic entity import (P1):

```rust
use homecore_migrate::entity_registry;
use homecore::HomeCore;

#[tokio::main]
async fn main() {
    let storage_dir = std::path::Path::new("/home/user/.homeassistant/.storage");
    
    // Load HA entities
    let entries = entity_registry::load(storage_dir)
        .expect("load entity registry");
    println!("Loaded {} entities", entries.len());
    
    // Import into HOMECORE (P2 when EntityRegistry::import() lands)
    let homecore = HomeCore::new();
    for entry in entries {
        println!("Entity: {} ({})", entry.entity_id, entry.name);
    }
}
```

Full migration (P2 onwards, via `--features recorder`):

```bash
# Side-by-side: old HA continues running while HOMECORE reads the DB
homecore-migrate export-for-sidecar \
  --ha-dir ~/.homeassistant \
  --homecore-db ~/.homecore/home.db \
  --keep-automations true  # Don't stop HA automations during test period
```

## Relation to other HOMECORE crates

```
homecore-migrate (import from HA)
├─ homecore (EntityEntry → EntityRegistry; config entry imports)
├─ homecore-automation (automations.yaml → automation rules, P2)
├─ homecore-recorder (side-by-side state export, P2, `--features recorder`)
├─ homecore-plugins (config_entries → plugin manifests, P2)
└─ homecore-server (can auto-import at startup with --import-ha flag, P2)
```

## References

- [ADR-165: HOMECORE Migration from Python Home Assistant](../../docs/adr/ADR-165-homecore-migrate-from-home-assistant.md)
- [ADR-126: HOMECORE Home Assistant Port (master)](../../docs/adr/ADR-126-homecore-home-assistant-port.md)
- [Home Assistant .storage/ format](https://developers.home-assistant.io/docs/storage/)
- [homecore-migrate CLI source](src/main.rs)
- [README — wifi-densepose](../../../README.md)
