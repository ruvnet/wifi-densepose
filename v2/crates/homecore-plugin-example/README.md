# homecore-plugin-example

Example WASM plugin for the HOMECORE plugin system (ADR-128 P2).

Demonstrates the complete ADR-128 host ABI round-trip:

- `plugin_setup` — subscribes to `sensor.test_temp` state changes
- `plugin_handle_state_changed` — sets `binary_sensor.test_alert` to `on` when temp > 25, `off` when temp < 20

## Build

```sh
# Ensure the wasm32 target is installed (once)
rustup target add wasm32-unknown-unknown

# Build the example plugin (from this directory)
cargo build --target wasm32-unknown-unknown --release -p homecore-plugin-example
```

Output: `target/wasm32-unknown-unknown/release/homecore_plugin_example.wasm`

## Run the integration test

```sh
# From v2/
cargo test -p homecore-plugins --features wasmtime
```

## ABI

See `homecore-plugins/src/host_abi.rs` for the authoritative host ABI spec.
