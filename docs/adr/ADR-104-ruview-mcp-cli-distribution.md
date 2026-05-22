# ADR-104: RuView MCP Server + CLI Distribution

- **Status:** Accepted
- **Date:** 2026-05-21
- **Deciders:** ruv
- **Related:** ADR-100 (Cog packaging), ADR-101 (pose cog), ADR-102 (edge registry), ADR-103 (count cog)
- **Implementation:** `tools/ruview-mcp/`, `tools/ruview-cli/`

---

## Context

The Cognitum cog ecosystem ships binaries to appliances via a signed GCS catalog (ADR-100). The cogs themselves run inside `/var/lib/cognitum/apps/` on a Pi 5 or Pi+Hailo cluster node. This is the right deployment target for production inference — sub-5 ms per frame, Hailo hardware acceleration, offline operation.

However, three user classes need to interact with RuView capabilities **without owning a Cognitum appliance**:

1. **Developer agents** — Claude Code, Cursor, Codex instances that want to call `ruview_pose_infer` during a research session (e.g. the SOTA loop in `docs/research/sota-2026-05-22/PROGRESS.md`).
2. **CI pipelines** — automated tests that want to assert "a synthetic CSI window produces a finite pose output" without a full appliance setup.
3. **Shell scripts and researchers** — `npx ruview pose infer --window ./window.json` from any machine with Node 20, no Rust toolchain, no Cognitum account, no clone of this repo required.

The existing surface does not serve these users:
- The sensing-server REST API (`/api/v1/sensing/latest`, `/api/v1/edge/registry`) is a Rust binary that requires building from source.
- The cog binaries are signed Linux aarch64/x86_64 executables — no macOS/Windows builds, no `npx` entrypoint.
- There is no MCP server — Claude Code cannot call RuView capabilities as tools without one.

This ADR defines two new distribution artifacts:
- `@ruv/ruview-mcp` — an MCP server exposing RuView as tools.
- `@ruv/ruview-cli` — a CLI exposing the same surface as `npx ruview <subcommand>`.

---

## Decision

### MCP server: `@ruv/ruview-mcp`

A Node 20 TypeScript package implementing the Model Context Protocol using `@modelcontextprotocol/sdk`. The server communicates over stdio (the standard MCP transport) and exposes six tools:

| Tool | Description | Backend |
|------|-------------|---------|
| `ruview_csi_latest` | Pull the latest CSI window from the sensing-server | GET /api/v1/sensing/latest (ADR-102) |
| `ruview_pose_infer` | 17-keypoint COCO pose estimation on a CSI window | cog-pose-estimation binary (ADR-101) subprocess |
| `ruview_count_infer` | Person count with calibrated confidence interval | cog-person-count binary (ADR-103) subprocess |
| `ruview_registry_list` | List Cognitum cogs from the edge registry | GET /api/v1/edge/registry (ADR-102) |
| `ruview_train_count` | Kick off a count-cog Candle training run | cargo run -p wifi-densepose-train subprocess |
| `ruview_job_status` | Poll a background training job | reads ~/.ruview/jobs/<id>.log |

**Fail-open principle:** every tool returns `{ok: false, warn: true, error: "...", hint: "..."}` rather than throwing. This matches the pattern used by the Cog binaries (ADR-100 §"Failure modes") and ensures a broken sensing-server does not crash a research agent's session.

### CLI: `@ruv/ruview-cli`

The same surface as a Yargs-based CLI published to npm as `@ruv/ruview-cli` with the binary name `ruview`:

| Subcommand | Equivalent MCP tool |
|------------|-------------------|
| `ruview csi tail` | streaming poll of `ruview_csi_latest` |
| `ruview pose infer [--window <path>]` | `ruview_pose_infer` |
| `ruview count infer [--window <path>]` | `ruview_count_infer` |
| `ruview cogs list [--category] [--search]` | `ruview_registry_list` |
| `ruview train count --paired <jsonl>` | `ruview_train_count` |
| `ruview job status --id <uuid>` | `ruview_job_status` |

All subcommands write JSON to stdout and exit 0 on success. WARN-level outputs (missing cog binary, unreachable sensing-server) go to stderr; exit code stays 0 so pipelines are not broken by transient unavailability.

### Inference backend: subprocess, not in-process

The MCP server and CLI **shell out** to the cog binaries rather than embedding a JS/WASM inference engine. Reasons:

1. The cog binaries are already signed, tested, and cross-compiled (ADR-100/101/103). Re-implementing inference in JS would duplicate that work and introduce a second model artifact to keep in sync.
2. The cog binaries handle model loading, ONNX dispatch, and Hailo HEF routing transparently — the MCP layer needs only to understand the JSON event schema.
3. For training, `cargo run -p wifi-densepose-train` is the proven path (2.1 s on RTX 5080, ADR-103). Replicating the Candle training loop in JS would be a significant engineering investment with no user benefit.

The npm packages therefore act as a **thin orchestration layer** over the existing Rust/cog infrastructure. No ML framework is bundled.

### ruvector library usage

Where a ruvector npm package provides the required capability, it is preferred over reimplementation. The subcarrier-saliency analysis in `examples/research-sota/r5_subcarrier_saliency.py` already depends on `ruvector-mincut` (Rust crate) for Stoer-Wagner min-cut. On the npm side:

- `@ruv/rvcsi` — the typed CSI frame schema and validation. When available at install time, `ruview_csi_latest` will validate incoming frames against the `rvcsi-core` schema. If not installed, falls back to opaque JSON passthrough.
- HNSW, RaBitQ, and contrastive embedding primitives are Rust-native; the npm packages do not replicate them. Instead, `ruview_pose_infer` and `ruview_count_infer` delegate to the cog binary which embeds the Candle inference engine.

### Source layout

```
tools/
├── ruview-mcp/                   # @ruv/ruview-mcp
│   ├── package.json
│   ├── tsconfig.json
│   ├── jest.config.js
│   ├── src/
│   │   ├── index.ts              # MCP server entry + tool registry
│   │   ├── types.ts              # shared domain types
│   │   ├── config.ts             # env-var config loader
│   │   ├── http.ts               # fetch wrapper with timeout + Result<T>
│   │   ├── cog.ts                # subprocess wrapper for cog binaries
│   │   └── tools/
│   │       ├── csi-latest.ts     # ruview_csi_latest
│   │       ├── pose-infer.ts     # ruview_pose_infer
│   │       ├── count-infer.ts    # ruview_count_infer
│   │       ├── registry-list.ts  # ruview_registry_list
│   │       └── train-count.ts    # ruview_train_count + ruview_job_status
│   └── tests/
│       └── tools.test.ts         # stub smoke tests (M1) + integration tests (M6)
└── ruview-cli/                   # @ruv/ruview-cli
    ├── package.json
    ├── tsconfig.json
    ├── src/
    │   ├── index.ts              # yargs CLI entry + command registration
    │   ├── config.ts             # env-var config loader
    │   ├── http.ts               # fetch wrapper
    │   ├── cog.ts                # subprocess wrapper
    │   └── commands/
    │       ├── csi.ts            # ruview csi tail
    │       ├── pose.ts           # ruview pose infer
    │       ├── count.ts          # ruview count infer
    │       ├── cogs.ts           # ruview cogs list
    │       ├── train.ts          # ruview train count
    │       └── job.ts            # ruview job status
    └── tests/                    # (M6)
```

---

## Security

### Authentication

The sensing-server uses a Bearer token (`RUVIEW_API_TOKEN`) for all `/api/v1/*` routes when the token is configured. The MCP server and CLI propagate this token in the `Authorization` header for every sensing-server call. Token is sourced **only from environment variables** — never from CLI flags or tool arguments (which could appear in logs or agent histories).

The cog binaries are called as local subprocesses. No network authentication is involved in cog invocation — the binary is trusted by virtue of being installed on the local machine (and having passed Ed25519 signature verification at install time, per ADR-100).

### Threat table

| # | Threat | Mitigation |
|---|--------|-----------|
| **T1** | **MCP tool spoofing** — a malicious process registers a tool named `ruview_pose_infer` before the legitimate server and intercepts agent calls | MCP servers are registered by the operator in the Claude Code / Cursor config. The operator must explicitly `claude mcp add ruview -- node …`. Impersonation requires compromising the operator's shell config. |
| **T2** | **CLI subcommand injection** — a caller passes a crafted `--paired` path containing shell metacharacters to escape the `cargo` invocation | All subprocess arguments are passed as an array (never through a shell string) via Node's `spawn(binary, args, {})` — no shell expansion. Path metacharacters cannot escape. |
| **T3** | **Token leakage** — `RUVIEW_API_TOKEN` appears in process arguments, agent histories, or log files | Token is only used in the `Authorization` HTTP header, which is set programmatically. It is never printed, never passed as a CLI argument, and never written to `~/.ruview/jobs/<id>.log`. |
| **T4** | **Model substitution** — an attacker replaces the cog binary with a malicious version | The cog binary must pass Ed25519 signature verification (`binary_sha256` + `binary_signature`) at install time per ADR-100. The MCP/CLI layer does not re-verify at invocation time — this is the cog-gateway's job. |
| **T5** | **Output validation bypass** — cog returns malformed JSON and the MCP server forwards it without validation | `ruview_pose_infer` and `ruview_count_infer` parse cog stdout as JSON and validate the schema against `PoseInferResult` / `CountInferResult` types (Zod, M2+). On parse failure, return `{ok:false, error: "unexpected cog output: …"}`. |
| **T6** | **Rate-limit bypass on `ruview_train_count`** — an agent calls `ruview_train_count` in a tight loop, spawning unbounded training processes | The MCP server maintains an in-process job registry. On `ruview_train_count`, if more than 3 jobs are `status:"running"`, return `{ok:false, error:"too many concurrent training jobs (max 3)"}`. Training jobs are CPU/GPU-bound and self-limit on the host. |

### What this ADR does NOT secure

- **MCP transport encryption** — MCP over stdio is process-local; no TLS is involved. If the MCP server is exposed over a TCP socket in future, TLS must be added.
- **Cog binary authentication at invocation** — we trust the OS file permissions and the at-install-time signature check (ADR-100). If a binary is replaced after install, the MCP layer will not detect it.
- **Multi-tenant token isolation** — the server process serves all connected clients under a single token. Multi-user deployments must run one MCP server instance per user.

---

## Packaging

### Version alignment

The npm package versions track the cog crate versions:
- `@ruv/ruview-mcp@0.0.1` ships when `cog-pose-estimation@0.0.1` + `cog-person-count@0.0.2` are on GCS.
- Semver: major bump when the MCP tool schema changes (breaking for calling agents); minor for new tools; patch for bug fixes.

### npm package configuration

Both packages are published to the public npm registry under the `@ruv` scope:

```
@ruv/ruview-mcp   — npm install -g @ruv/ruview-mcp  (then: ruview-mcp)
@ruv/ruview-cli   — npm install -g @ruv/ruview-cli  (then: ruview --version)
```

The `bin` entry in `package.json` points to `dist/index.js` (compiled from TypeScript). Both packages target Node 20 (`"engines": {"node": ">=20.0.0"}`).

`private: true` is set during development; **the user must flip this to `false` before publishing** (or delete the field). The `publishConfig.access: "public"` is already set.

### MCP registration

After installing (global or npx):

```bash
# Via npx (no install required):
claude mcp add ruview -- npx @ruv/ruview-mcp

# Via global install:
npm install -g @ruv/ruview-mcp
claude mcp add ruview -- ruview-mcp

# Verify:
claude mcp list    # should show "ruview"
```

---

## Distribution

`npx ruview …` works from any machine with Node 20 installed. No clone of this repository, no Rust toolchain, no Cognitum appliance is required to run the CLI commands that do not depend on a cog binary (e.g. `ruview cogs list` only needs a sensing-server URL).

For commands that call a cog binary (`ruview pose infer`, `ruview count infer`), the cog binary must be downloaded from GCS and placed in a directory on `PATH` or pointed to via `RUVIEW_POSE_COG_BINARY` / `RUVIEW_COUNT_COG_BINARY`. The download URL follows ADR-100 naming:

```
https://storage.googleapis.com/cognitum-apps/cogs/x86_64/cog-pose-estimation-x86_64
https://storage.googleapis.com/cognitum-apps/cogs/arm/cog-pose-estimation-arm
https://storage.googleapis.com/cognitum-apps/cogs/x86_64/cog-person-count-x86_64
https://storage.googleapis.com/cognitum-apps/cogs/arm/cog-person-count-arm
```

A future `ruview install cogs` subcommand can automate this download + chmod + PATH placement.

---

## Failure modes

| Scenario | Behaviour |
|---|---|
| Sensing-server not running | `ruview_csi_latest` / `ruview_registry_list` return `{ok:false, warn:true, error:"…", hint:"…"}`. Exit code 0 on CLI. MCP tool returns isError:false (it's a warn, not a crash). |
| Cog binary not installed | `ruview_pose_infer` / `ruview_count_infer` return `{ok:false, warn:true, error:"…", hint:"…"}` with install instructions. |
| Cog binary returns non-zero | Propagated as `{ok:false, error:"Cog exited with code N. stderr: …"}`. |
| Training job crashes immediately | Log file records `# exit code: <N>`. `ruview_job_status` returns `{status:"failed", recent_log:[…]}`. |
| MCP server process dies mid-session | In-process job registry is lost. Jobs that were running continue in background (detached); operator reads log files directly. |
| Node < 20 | `fetch` is unavailable. The CLI prints a clear error: "Node 20+ required for built-in fetch". |

---

## Acceptance gates

| Gate | Test |
|------|------|
| `npx ruview --version` works | `ruview --version` prints `0.0.1` and exits 0. |
| `ruview_pose_infer` returns finite output for synthetic CSI | M2 integration test: spawn MCP server, call tool with a synthetic window JSON, assert `result.n_persons >= 0` and all keypoint values in `[0, 1]`. |
| MCP server passes `claude mcp list` check | `claude mcp add ruview -- node dist/index.js && claude mcp list` shows `ruview` with 6 tools. |
| `npm run build` clean in both packages | TypeScript compilation exits 0, no errors. |
| Stub smoke tests pass (M1) | `npm test` in `tools/ruview-mcp/` passes all 6 stub tests. |
| Integration tests pass (M6) | 6 tool calls with mocked sensing-server + real node binary as cog stub all return `{ok: true}`. |

---

## Migration / rollout

1. **This PR** — land scaffold (`tools/ruview-mcp/`, `tools/ruview-cli/`) + ADR-104. Both packages at `private: true`.
2. **M2** — wire real inference: sensing-server CSI window → cog subprocess → parsed output. Remove `stub: true` from responses.
3. **M3** — wire `ruview_csi_latest` + `ruview_registry_list` with live sensing-server round-trip test.
4. **M4** — wire `ruview_train_count` with real cargo invocation; verify job log populates.
5. **M6** — integration tests green. Update acceptance gates.
6. **User publish step** — flip `private` from `true` to `false` in both `package.json` files, then:

```bash
# Publish MCP server:
cd tools/ruview-mcp
npm version patch          # or minor/major per semver
npm publish --access public

# Publish CLI:
cd tools/ruview-cli
npm version patch
npm publish --access public
```

---

## See also

- ADR-100: Cognitum Cog Packaging Specification — the signing + GCS distribution model this ADR sits on top of.
- ADR-101: Pose Estimation Cog — the binary invoked by `ruview_pose_infer`.
- ADR-102: Edge Module Registry — the `/api/v1/edge/registry` endpoint used by `ruview_registry_list`.
- ADR-103: Learned Multi-Person Counter Cog — the binary invoked by `ruview_count_infer`.
- `docs/research/sota-2026-05-22/PROGRESS.md` — the SOTA research loop that motivated the MCP server.
- `v2/crates/cog-pose-estimation/` — Rust source for the pose-estimation cog.
- `v2/crates/cog-person-count/` — Rust source for the person-count cog.
