// SPDX-License-Identifier: MIT

export const GUIDANCE_TOPICS = Object.freeze([
  'overview',
  'core',
  'server',
  'api',
  'plugins',
  'integrations',
  'migration',
  'voice',
  'testing',
]);

export const TOPIC_SUMMARIES = Object.freeze({
  overview: 'A source-cited map of Homecore capabilities and maturity.',
  core: 'State, events, services, registries, restore, and recorder behavior.',
  server: 'The integrated Homecore server, configuration, and deployment boundaries.',
  api: 'Home Assistant-compatible REST and WebSocket core contracts.',
  plugins: 'Compiled-in native plugins and bounded external Wasm packages.',
  integrations: 'HAP, Home Assistant compatibility, dashboard, and provider boundaries.',
  migration: 'Versioned Home Assistant registry and config-entry migration.',
  voice: 'Intent, STT/TTS contracts, audio bounds, and satellite sessions.',
  testing: 'Focused Rust feature gates and metaharness validation.',
});

export const CAPABILITIES = Object.freeze([
  {
    id: 'runtime-restore',
    name: 'Core runtime and startup restore',
    topics: ['core', 'server'],
    status: 'implemented',
    evidence: 'REPOSITORY',
    summary: 'Homecore provides concurrent state, entity/device registries, event buses, and services. Server startup restores registries before recent recorder states and isolates malformed rows within configured limits.',
    sources: [
      'v2/docs/homecore-capabilities.md',
      'v2/crates/homecore/src/lib.rs',
      'v2/crates/homecore-server/src/restore.rs',
      'v2/crates/homecore-recorder/src/db.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore -p homecore-recorder -p homecore-server --no-default-features',
    ],
    limitations: [
      'Restore depends on configured persistent storage and recorder availability.',
      'Malformed rows are reported and isolated rather than silently accepted.',
    ],
  },
  {
    id: 'automation-recorder',
    name: 'Automation and state history',
    topics: ['core', 'server'],
    status: 'implemented',
    evidence: 'REPOSITORY',
    summary: 'The automation crate evaluates state, numeric, event, and time triggers. The recorder persists SQLite history, restores latest states, recovers from event lag, and can add an optional semantic index.',
    sources: [
      'v2/crates/homecore-automation/src/lib.rs',
      'v2/crates/homecore-recorder/src/lib.rs',
      'v2/docs/homecore-capabilities.md',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-automation -p homecore-recorder --no-default-features',
    ],
    limitations: [
      'Optional semantic search requires its feature and backend.',
      'Automation availability depends on the server configuration and loaded definitions.',
    ],
  },
  {
    id: 'ha-core-api',
    name: 'Home Assistant-compatible REST and WebSocket API',
    topics: ['api', 'server', 'integrations'],
    status: 'implemented-core-contract',
    evidence: 'REPOSITORY',
    summary: 'The authenticated API covers documented core state, service, event, template, history, logbook, calendar, camera, intent, registry-list, subscription, and feature-negotiation contracts.',
    sources: [
      'v2/docs/homecore-capabilities.md',
      'v2/crates/homecore-api/README.md',
      'v2/crates/homecore-api/src/lib.rs',
      'v2/crates/homecore-api/src/ws.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-api -p homecore-server --no-default-features',
    ],
    limitations: [
      'This is not parity with every endpoint supplied by the Home Assistant integration ecosystem.',
      'Media, calendar, camera, registry mutation, and Lovelace behavior may require configured providers or backends.',
    ],
  },
  {
    id: 'wasm-plugins',
    name: 'Native registration and Wasmtime plugin loading',
    topics: ['plugins', 'server', 'testing'],
    status: 'feature-gated',
    evidence: 'REPOSITORY',
    summary: 'Native plugins are compiled into an explicit registry. External plugin packages are bounded, path-checked, signature-verified WebAssembly and execute through Wasmtime when enabled.',
    sources: [
      'v2/docs/homecore-capabilities.md',
      'v2/crates/homecore-server/src/plugins.rs',
      'v2/crates/homecore-plugins/src/lib.rs',
      'v2/crates/homecore-plugins/src/verify.rs',
      'v2/crates/homecore-plugins/src/wasmtime_runtime.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-plugins --no-default-features',
      'cargo test --manifest-path v2/Cargo.toml -p homecore-plugins --features wasmtime',
      'cargo test --manifest-path v2/Cargo.toml -p homecore-server --features wasmtime',
    ],
    limitations: [
      'Wasmtime is opt-in.',
      'Arbitrary native dynamic libraries are not loaded.',
      'Unsigned Wasm requires an explicit development-only override.',
    ],
  },
  {
    id: 'hap-network',
    name: 'Network HomeKit Accessory Protocol server',
    topics: ['integrations', 'server', 'testing'],
    status: 'feature-gated',
    evidence: 'REPOSITORY',
    summary: 'The HAP feature wires bounded TCP handling, persisted pairing records, encrypted control sessions, live accessory synchronization, and _hap._tcp mDNS lifecycle into the server.',
    sources: [
      'v2/docs/homecore-capabilities.md',
      'v2/crates/homecore-hap/README.md',
      'v2/crates/homecore-hap/src/lib.rs',
      'v2/crates/homecore-hap/src/server.rs',
      'v2/crates/homecore-hap/src/mdns.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-hap --no-default-features',
      'cargo test --manifest-path v2/Cargo.toml -p homecore-hap --features hap-server',
      'cargo test --manifest-path v2/Cargo.toml -p homecore-server --features hap-server',
    ],
    limitations: [
      'HAP is disabled by default and requires explicit network and durable pairing configuration.',
      'Internal protocol tests are not Apple certification or proof against a current Apple Home controller.',
      'Some writable, timed, and resource behavior remains incomplete.',
    ],
  },
  {
    id: 'ha-migration',
    name: 'Home Assistant registry and config-entry migration',
    topics: ['migration', 'integrations', 'testing'],
    status: 'implemented-bounded',
    evidence: 'REPOSITORY',
    summary: 'Migration tooling version-checks entity/device registries and config entries, preserves unknown compatible fields, reports unsupported data, and writes atomically without overwriting destinations.',
    sources: [
      'docs/adr/ADR-165-homecore-migrate-from-home-assistant.md',
      'v2/crates/homecore-migrate/README.md',
      'v2/crates/homecore-migrate/src/lib.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-migrate',
      'cargo clippy --manifest-path v2/Cargo.toml -p homecore-migrate --all-targets -- -D warnings',
    ],
    limitations: [
      'Imported config entries do not install or execute Home Assistant integrations.',
      'Automation conversion, secret-reference resolution, tombstones, and recorder export are not complete.',
    ],
  },
  {
    id: 'voice-satellite',
    name: 'STT/TTS and satellite voice protocols',
    topics: ['voice', 'integrations', 'testing'],
    status: 'provider-required',
    evidence: 'REPOSITORY',
    summary: 'Homecore defines bounded PCM16 audio, asynchronous STT/TTS provider contracts, an intent pipeline, and an authenticated transport-independent satellite session state machine.',
    sources: [
      'v2/docs/homecore-capabilities.md',
      'v2/crates/homecore-assist/src/lib.rs',
      'v2/crates/homecore-assist/src/voice.rs',
      'v2/crates/homecore-assist/src/satellite.rs',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-assist --no-default-features',
    ],
    limitations: [
      'Deployments must supply real STT and TTS providers.',
      'Disabled providers return typed errors and do not fabricate results.',
      'A concrete transport adapter is still required for deployment.',
    ],
  },
  {
    id: 'integrated-server',
    name: 'Integrated server and dashboard boundary',
    topics: ['server', 'api', 'integrations'],
    status: 'implemented-configurable',
    evidence: 'REPOSITORY',
    summary: 'homecore-server wires the core, API, recorder, plugins, automations, assist, optional HAP, static UI, and typed upstream gateway responses into one process.',
    sources: [
      'v2/crates/homecore-server/Cargo.toml',
      'v2/crates/homecore-server/src/main.rs',
      'v2/crates/homecore-server/src/gateway.rs',
      'docs/adr/ADR-161-homecore-server-layer-security.md',
    ],
    validation: [
      'cargo test --manifest-path v2/Cargo.toml -p homecore-server --no-default-features',
    ],
    limitations: [
      'Production authentication and network bindings require explicit secure configuration.',
      'Unavailable upstreams return typed unavailable responses rather than simulated data.',
    ],
  },
  {
    id: 'developer-metaharness',
    name: 'WASM-first developer metaharness',
    topics: ['testing', 'plugins', 'api'],
    status: 'implemented-in-package',
    evidence: 'POLICY',
    summary: 'The npm package provides source-cited guidance, reviewed shared knowledge, WASM kernel diagnostics, a bounded MCP surface, focused test profiles, and guarded local Claude Code and Codex delegation.',
    sources: [
      'harness/homecore/README.md',
      'harness/homecore/src/kernel.js',
      'harness/homecore/src/policy.js',
      'docs/adr/ADR-285-homecore-wasm-first-metaharness.md',
    ],
    validation: [
      'cd harness/homecore && npm test',
      'cd harness/homecore && npm run test:security',
      'cd harness/homecore && npm run brain:verify -- --repo ../..',
      'cd harness/homecore && npm run manifest:verify',
    ],
    limitations: [
      'The harness is developer tooling, not the Homecore server runtime.',
      'The MCP surface is read-only; Cargo verification and host delegation are CLI-only.',
      'It never self-promotes shared knowledge or learning output.',
      'Host writes require explicit double opt-in.',
    ],
  },
]);
