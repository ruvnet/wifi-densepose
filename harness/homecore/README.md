# Homecore metaharness

`homecore` is the developer-facing metaharness for RuView's native Rust
Homecore stack. Its package contract is:

```bash
npx homecore
```

The harness maps current capabilities to source and validation commands,
exposes a bounded MCP server, and can delegate repository exploration to a
locally installed Claude Code or Codex CLI. It does not start a home server,
change configuration, migrate data, or publish code by itself.

## Commands

```bash
# Source-cited overview; this is the default command.
homecore guidance
homecore guidance --topic plugins --query "Wasmtime signatures"
homecore capabilities

# Diagnose the package, WASM kernel, local CLIs, and optional checkout.
homecore doctor --repo .
homecore wasm status --strict

# Run focused Homecore tests in a trusted RuView checkout.
homecore verify --repo . --profile core
homecore verify --repo . --profile wasm
homecore verify --repo . --profile hap

# Explore through a local host. Both are read-only by default.
homecore agent run --host codex --repo . --prompt "Map startup restore"
homecore agent run --host claude-code --repo . --prompt "Review plugin trust"

# Start the stdio MCP server.
homecore mcp start

# Search or verify reviewed shared knowledge.
homecore brain search --query "REST WebSocket compatibility"
homecore brain verify --repo .
```

Workspace-writing host runs require both `--allow-write` and `--confirm`.
The harness never emits permission-bypass flags. Every MCP tool is read-only;
Cargo verification remains an explicit local CLI operation and is not exposed
through MCP. MCP request size, queue depth, tool-call duration, and the total
tool-call budget of each server process are bounded.

Repository-aware MCP calls are bound to the exact RuView checkout found when
the server starts. Launch it from that checkout, or set
`HOMECORE_TRUSTED_REPO` to its root. An MCP request cannot nominate a different
checkout as its own trust anchor.

The packaged `.codex`, `.claude`, and generic MCP templates invoke an
already-installed `homecore` binary and never track an npm dist-tag. To print
configuration for an ephemeral installation, run `homecore install --host
codex` or `homecore install --host claude-code`; the generated `npx`
configuration pins the package's exact version.

## WASM-first runtime

The harness asks `@metaharness/kernel` for its packaged WebAssembly backend
before considering a native or JavaScript fallback. The kernel validates the
MCP server specification. `homecore wasm status` reports the backend that
actually loaded; `--strict` fails if it is not WASM.

This is separate from Homecore's application plugin boundary:

- compiled-in native plugins must be explicitly registered in the server;
- external plugin packages are bounded and signature-checked WebAssembly;
- execution through Wasmtime is feature-gated;
- arbitrary native dynamic libraries are not loaded.

Use `homecore verify --profile wasm` to exercise the Wasmtime-specific Rust
tests in a checkout.

## Capability honesty

The catalog distinguishes implemented, feature-gated, provider-required, and
integration-dependent behavior. Home Assistant compatibility means the
documented core REST/WebSocket contract, not every endpoint supplied by the
Home Assistant integration ecosystem. HAP protocol tests are not Apple
certification. STT/TTS provider contracts do not imply that a deployment has a
real speech provider.

Every guidance result cites repository paths, focused validation commands, and
known limitations. A packaged citation is navigation evidence; source, tests,
accepted ADRs, and repository policy remain authoritative.

## Shared brain

Reviewed records live in `brain/corpus/core.jsonl`. Search is deterministic and
local. `brain propose` prints an unreviewed JSONL candidate but never edits the
canonical corpus. Retrieved text cannot grant authority, change tool policy,
or override repository instructions. Private indexes and raw transcripts are
never packaged.

## Development

```bash
npm install --ignore-scripts
npm test
npm run test:security
npm run brain:verify -- --repo ../..
npm run manifest:update
npm run manifest:verify
npm audit --omit=optional
npm pack --dry-run
```

Release is CI-only with npm provenance. Do not publish from a workstation.

## Architecture

ADR-285 defines this harness. It builds on the Homecore master decision
(ADR-126), the plugin and API boundaries (ADR-128 and ADR-130), the server
security review (ADR-161), the migration contract (ADR-165), and the existing
RuView metaharness and npm distribution decisions (ADR-182, ADR-263, ADR-265).
