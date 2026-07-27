# homecore-migrate

Migration tooling for importing Home Assistant filesystem storage into
HOMECORE. The implementation follows
[ADR-165](../../docs/adr/ADR-165-homecore-migrate-from-home-assistant.md).

## Implemented

- Reads versioned HA `.storage` JSON and rejects unknown schema versions.
- Converts entity registry metadata to `homecore::EntityEntry`.
- Converts the supported HA v13 device-registry fields to
  `homecore::DeviceEntry`, including identifiers, connections, versions,
  serial number, labels, topology, and config-entry links.
- Converts `core.config_entries` to a versioned
  `homecore.config_entries` file. Each original row is retained verbatim.
  Unsupported domains and non-portable fields produce typed warnings.
- Publishes destination JSON through a synced same-directory temporary file
  and an atomic no-clobber link. Existing destination files are never replaced.
- Emits one-line JSON summaries from every import command.
- Parses secrets with redacted errors and inspects automations.

## CLI

The import commands take the HA `.storage` directory and the HOMECORE storage
destination:

```bash
homecore-migrate import-entities \
  --storage ~/.homeassistant/.storage \
  --to ~/.homecore/storage

homecore-migrate import-devices \
  --storage ~/.homeassistant/.storage \
  --to ~/.homecore/storage

homecore-migrate import-config-entries \
  --storage ~/.homeassistant/.storage \
  --to ~/.homecore/storage
```

Successful imports print a machine-readable JSON object:

```json
{"kind":"device_registry","imported":8,"warning_count":0,"warnings":[],"destination":"/home/user/.homecore/storage/core.device_registry"}
```

`inspect`, `inspect-config-entries`, `inspect-secrets`, and
`inspect-automations` are read-only.

## Destination files

| Source | Destination | Format |
|---|---|---|
| `core.entity_registry` | `core.entity_registry` | HA-compatible v1/minor 13 envelope |
| `core.device_registry` | `core.device_registry` | HA-compatible v1/minor 13 envelope |
| `core.config_entries` | `homecore.config_entries` | HOMECORE v1/minor 0 envelope |

Config entries are storage-compatible, not runtime-compatible: importing an
entry does not install or execute its HA integration. A HOMECORE plugin must
explicitly claim the domain and consume the preserved source payload.

## Remaining limitations

- Automation conversion is not implemented; the tool only inspects
  `automations.yaml`.
- `!secret` reference resolution in other YAML files is not implemented.
- Deleted entity/device tombstones are not imported.
- Device fields newer than HA registry minor version 13 require an explicit
  parser update; unknown versions fail closed.
- No side-by-side HA recorder database exporter is provided.

## Validation

```bash
cargo test -p homecore-migrate
cargo clippy -p homecore-migrate --all-targets -- -D warnings
```
