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
- Plugin and HAP crates are libraries in this workspace, but this binary does
  not claim to load plugins or advertise a HomeKit server yet.

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

## Validation

```bash
cargo test -p homecore-server
cargo clippy -p homecore-server --all-targets --all-features -- -D warnings
```
