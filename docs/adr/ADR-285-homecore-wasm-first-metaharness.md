# ADR-285: WASM-first Homecore developer metaharness via `npx homecore`

- **Status**: Accepted — implemented and validated
- **Date**: 2026-07-29
- **Deciders**: ruv
- **Tags**: homecore, metaharness, wasm, mcp, npm, codex, claude-code

## Context

Homecore is now a multi-crate Rust subsystem with a concurrent state machine,
startup restore, recorder, automation engine, authenticated Home
Assistant-compatible REST/WebSocket core, migration tooling, compiled-in and
Wasmtime plugin paths, a network HAP server, and voice/satellite protocol
contracts. The implementation is intentionally bounded: features are gated,
several deployments require providers or backends, and core compatibility is
not the same as parity with the entire Home Assistant integration ecosystem.

The existing `@ruvnet/ruview` contributor harness contains source-cited
Homecore guidance, but it serves the whole RuView repository. Homecore needs a
focused entry point that can:

1. explain current capabilities without overstating maturity;
2. lead contributors to the correct source, ADRs, and focused tests;
3. exercise Wasmtime and HAP feature gates deliberately;
4. expose a small MCP guidance surface;
5. delegate exploration to local Claude Code or Codex CLIs at least authority;
6. remain removable from the Homecore server runtime.

The requested user experience is the exact command:

```bash
npx homecore
```

An npm package named `@ruvnet/homecore` can expose a `homecore` binary after it
is installed, but `npx homecore` resolves an unscoped package named
`homecore`. ADR-265 normally reserves new packages for the `@ruvnet` scope, so
the executable naming decision requires an explicit, narrow exception.

## Decision

Create `harness/homecore/` as an independently testable npm package named
`homecore`, with the `homecore` binary. This unscoped package is the executable
front door only. Future import-oriented libraries remain under `@ruvnet/*`.
When accepted, this ADR amends ADR-265 only for that one executable package;
all other new RuView npm packages remain subject to ADR-265's scoped-name rule.

The package is developer tooling, not a second Homecore runtime. It may inspect
a trusted RuView checkout and run fixed test commands, but it does not start
the server, alter home state, migrate user data, modify pairing records,
install plugins, or publish changes.

### 1. WASM-first metaharness kernel

Pin `@metaharness/kernel` exactly. Unless the operator explicitly chooses a
backend with `METAHARNESS_KERNEL_BACKEND`, the harness requests the packaged
WebAssembly backend first.

The loaded kernel validates the MCP server specification. The actual backend
is always reported:

- `wasm` is the preferred result;
- a native or JavaScript fallback is allowed for portability;
- `homecore wasm status --strict` fails when WASM is unavailable;
- fallback execution is never relabelled as WASM.

The kernel specification and generated host configuration pin the current
package version. Packaged project templates invoke an already-installed
`homecore` binary; no committed MCP configuration executes
`homecore@latest`.

This kernel boundary is separate from application plugins. Homecore's plugin
architecture remains:

- native plugins are compiled in and registered explicitly;
- external packages are bounded, path-checked, signature-verified Wasm;
- Wasmtime execution is opt-in through Cargo features;
- arbitrary native dynamic libraries are not loaded.

The `wasm` verification profile runs the Wasmtime-specific plugin and server
tests from fixed argument arrays with `shell: false`.

### 2. CLI and MCP surface

The CLI provides:

- source-cited `guidance` and `capabilities`;
- reviewed local `brain search`, citation verification, and proposal output;
- `doctor` and strict/non-strict WASM diagnostics;
- fixed `core`, `wasm`, `hap`, and `full` verification profiles;
- skills and tool-schema discovery;
- an MCP stdio server;
- configuration output for Claude Code and Codex;
- guarded local host delegation.

The MCP server exposes only:

- `homecore_guidance`;
- `homecore_wasm_status`;
- `homecore_doctor`;
- `homecore_memory_search`.

All MCP tools are read-only. The fixed verification profiles remain local CLI
commands because Cargo writes build artifacts, executes repository code, and
may consume substantial resources. There are no MCP tools for test execution,
server start, migration writes, pairing, plugin installation, agent
delegation, GitHub mutation, release, or publication.

JSON-RPC request size, queue depth, per-process tool-call budget, output, and
tool/subprocess duration are bounded. Tool schemas reject unknown fields.
Repository roots are realpath-verified against fixed RuView/Homecore markers.
Child processes use argument arrays, `shell: false`, a scrubbed environment,
bounded output, and secret redaction. MCP repository access is anchored once
at server startup from the launch checkout or `HOMECORE_TRUSTED_REPO`; request
arguments cannot self-declare a new trust root.

### 3. Local Claude Code and Codex adapters

Both adapters operate on an exact trusted checkout and consume prompts through
stdin.

Codex uses:

- `codex exec -`;
- `-C <trusted-root>`;
- `--sandbox read-only` by default;
- ephemeral JSONL output;
- strict configuration parsing;
- ignored user config while repository exec-policy rules remain active.

Claude Code uses:

- `claude -p --safe-mode`;
- plan mode with read/search tools by default;
- JSON output;
- no session persistence.

Workspace writes require both `--allow-write` and `--confirm`. Neither adapter
emits a permission or sandbox bypass. Host delegation is CLI-only and is not
reachable through MCP, avoiding recursive agent authority.

### 4. Reviewed guidance and shared brain

Capability records carry:

- an honest maturity label;
- repository source paths;
- fixed validation commands;
- explicit limitations.

Canonical brain records are committed, reviewed, bounded, evidence-labelled,
source-relative, and digest-covered. Search is deterministic. `brain propose`
prints an unreviewed JSONL candidate and never edits canonical knowledge.
Retrieved content is evidence, not instruction or permission. Private vector
indexes, overlays, and raw transcripts remain untracked and unpackaged.

No Darwin/Flywheel candidate can self-promote through this harness. A future
learning loop requires a separate reviewed decision and the same frozen
holdout, provenance, security, and maintainer gates as ADR-283.

### 5. Distribution and release

Extend the ADR-265 npm matrix and provenance-only release workflow to
`harness/homecore`. The gate must run on supported Node versions and verify:

- exact lockfile installation;
- tests and security tests;
- package version single-sourcing;
- an explicit unpacked-size budget and no source maps;
- installation and execution from the real tarball;
- the WASM backend from the installed tarball;
- MCP initialization and exports;
- README claim checking;
- the package provenance manifest.

Publication remains CI-only with npm provenance. The release job runs on a
trusted-publishing-compatible Node/npm runtime, accepts only `main`, uses the
protected `npm-release` environment, and publishes the exact digest-checked
tarball that passed smoke tests. The environment must restrict deployment to
`main`, require review, and prevent self-review. The unscoped npm name being
available during development is not treated as permanent ownership; release
must still confirm registry access and package identity.

## Consequences

### Positive

- Contributors get a focused `npx homecore` entry point without coupling the
  Rust server to an agent framework.
- WASM is used for the portable kernel and explicitly exercised for Homecore
  plugin verification.
- Capability guidance can distinguish implemented code, feature gates,
  provider requirements, ecosystem limitations, and certification boundaries.
- Local agent execution is portable across Claude Code and Codex while
  remaining read-only by default.
- MCP authority is small enough to audit and contains no direct home, network,
  GitHub, or release mutation.

### Negative

- `homecore` is a narrow exception to the `@ruvnet/*` package namespace rule.
- The package adds one exact runtime dependency for the WASM kernel.
- The Wasmtime and HAP verification profiles can be expensive and write Cargo
  build artifacts.
- A packaged guidance catalog can become stale; citation verification and
  reviewed updates are required.

### Neutral

- The harness does not change Homecore's protocol, persistence, migration,
  plugin, HAP, or voice implementation.
- A passing software profile does not establish a production deployment,
  third-party ecosystem parity, Apple certification, or hardware behavior.
- Ruflo remains an optional development coordinator and is not a runtime
  dependency of `homecore`.

## Links

- [ADR-126](ADR-126-ruview-native-ha-port-master.md) - Homecore master decision.
- [ADR-128](ADR-128-homecore-integration-plugin-system.md) - plugin boundary.
- [ADR-130](ADR-130-homecore-rest-websocket-api.md) - REST/WebSocket contract.
- [ADR-133](ADR-133-homecore-assist-ruflo.md) - assist and agent bridge.
- [ADR-161](ADR-161-homecore-server-layer-security.md) - server security.
- [ADR-165](ADR-165-homecore-migrate-from-home-assistant.md) - migration trust boundary.
- [ADR-182](ADR-182-npx-ruview-harness-via-metaharness.md) - RuView metaharness.
- [ADR-263](ADR-263-ruview-npm-harness-deep-review.md) - harness hardening.
- [ADR-265](ADR-265-ruview-npm-distribution-strategy.md) - npm distribution policy.
- [ADR-283](ADR-283-ruview-community-metaharness-flywheel.md) - shared brain and learning gates.
- `harness/homecore/`
- `v2/docs/homecore-capabilities.md`
