# homecore-server

`homecore-server` is the alpha integration binary for HOMECORE. It runs one
shared state machine, event bus, service registry, REST/WebSocket API, optional
SQLite recorder, automation engine, intent endpoint, BFF gateway, and static
dashboard.

It is not a drop-in replacement for the complete Home Assistant API. The exact
implemented and deferred surfaces are listed in
[`docs/homecore-capabilities.md`](../../docs/homecore-capabilities.md).

## Security defaults

Authentication fails closed. Set one or more comma-separated bearer tokens:

```bash
set HOMECORE_TOKENS=replace-with-a-long-random-token
cargo run -p homecore-server
```

`--insecure-dev-auth` explicitly allows any non-empty bearer token and must only
be used in an isolated development environment. The browser UI does not contain
a default token. Configure allowed browser origins with
`HOMECORE_CORS_ORIGINS`.

## Runtime behavior

- SQLite recording is enabled by default at `sqlite://homecore.db`.
- Entity/device registries are restored from `.homecore/storage` before
  recorder states, listeners, automations, and API startup.
- The latest recorder state for each entity is restored deterministically.
  Restored snapshots retain their timestamps and carry a `homecore.restore`
  context marker; malformed rows are logged and isolated.
- Synthetic entities are disabled by default; opt in with
  `--seed-demo-entities`.
- Automations can be loaded with `--automations <file>` or
  `HOMECORE_AUTOMATIONS`.
- `Ctrl-C` initiates graceful HTTP shutdown and emits `HomeCoreStop`.
- The `ruvector` feature enables the recorder's semantic index.
- Packaged plugins are loaded only from explicitly configured directories.
- The HAP network server remains disabled unless the binary is built with
  `--features hap-server` and its bind, identity, advertisement, and pairing
  configuration is supplied.
- RuView semantic ingest is disabled by default. When explicitly enabled it
  polls only the per-node privacy-bounded vitals and semantic-event projection
  endpoints, fails stale/offline evidence closed, and emits anonymous HomeCore
  binary sensors. See [ADR-343](../../docs/adr/ADR-343-homecore-ruview-semantic-ingest-boundary.md).

## API

All API requests require `Authorization: Bearer <token>`.

```bash
curl -H "Authorization: Bearer $HOMECORE_TOKEN" \
  http://127.0.0.1:8123/api/states

curl -X POST \
  -H "Authorization: Bearer $HOMECORE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"light.kitchen"}' \
  http://127.0.0.1:8123/api/services/light/turn_on

curl -X POST \
  -H "Authorization: Bearer $HOMECORE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"utterance":"turn on light.kitchen","language":"en"}' \
  http://127.0.0.1:8123/api/intent/handle
```

Built-in executable services are `homecore.ping`,
`homecore.snapshot_state`, and `turn_on`/`turn_off`/`toggle` for the
`homeassistant`, `light`, and `switch` domains. Unsupported services are left
unregistered and return an error rather than a false acknowledgement.

## Configuration

| Flag | Environment | Default |
|---|---|---|
| `--bind` | `HOMECORE_BIND` | `0.0.0.0:8123` |
| `--db` | `HOMECORE_DB` | `sqlite://homecore.db` |
| `--storage-dir` | `HOMECORE_STORAGE_DIR` | `.homecore/storage` |
| `--restore-limit` | `HOMECORE_RESTORE_LIMIT` | `100000` |
| `--location-name` | `HOMECORE_LOCATION` | `Home` |
| `--automations` | `HOMECORE_AUTOMATIONS` | unset |
| `--insecure-dev-auth` | `HOMECORE_INSECURE_DEV_AUTH` | `false` |
| `--seed-demo-entities` | — | `false` |
| `--no-recorder` | — | `false` |
| `--ruview-ingest` | `HOMECORE_RUVIEW_INGEST` | `false` |
| `--ruview-url` | `HOMECORE_RUVIEW_URL` | `http://127.0.0.1:3000` |
| `--ruview-node-id` | `HOMECORE_RUVIEW_NODE_ID` | required when enabled |
| `--ruview-token` | `HOMECORE_RUVIEW_TOKEN` | required when enabled; secret |
| `--ruview-poll-ms` | `HOMECORE_RUVIEW_POLL_MS` | `1000` |
| `--ruview-timeout-ms` | `HOMECORE_RUVIEW_TIMEOUT_MS` | `2000` |
| `--ruview-max-staleness-ms` | `HOMECORE_RUVIEW_MAX_STALENESS_MS` | `10000` |

Use a RuView token limited to sensing reads. Prefer the environment variable so
the credential is not exposed in shell history or process arguments:

```bash
export HOMECORE_RUVIEW_INGEST=true
export HOMECORE_RUVIEW_URL=http://127.0.0.1:3000
export HOMECORE_RUVIEW_NODE_ID=7
export HOMECORE_RUVIEW_TOKEN=replace-with-a-sensing-read-token
cargo run -p homecore-server --features hap-server
```

Ingest and HAP are independent opt-ins. The example does not configure HAP and
therefore does not advertise an Apple Home accessory by itself.

## Validation

```bash
cargo test -p homecore-server
cargo clippy -p homecore-server --all-targets --all-features -- -D warnings
```
