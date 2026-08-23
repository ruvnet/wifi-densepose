# `npx @ruvnet/ruview` — RuView WiFi-sensing operator harness

An AI agent harness that knows how to operate **RuView** (WiFi-DensePose): onboard a
newcomer, provision an ESP32 CSI node, calibrate a room, train pose models, and —
crucially — **refuse to overstate accuracy**. Minted from the RuView monorepo via
[`metaharness`](https://www.npmjs.com/package/metaharness) and hardened per **ADR-182**.

WiFi sensing infers *coarse* pose/presence/breathing from Channel State Information.
It is **not a camera**. Every accuracy number this harness emits must be MEASURED
against a baseline — that rule is enforced in code (`ruview_claim_check`).

## Quick start

```bash
npx @ruvnet/ruview                       # onboard — pick a setup path
npx @ruvnet/ruview claim-check --file REPORT.md   # the honesty guardrail (non-zero exit on untagged claims)
npx @ruvnet/ruview verify                # run the deterministic proof (VERDICT: PASS)
npx @ruvnet/ruview doctor                # self-check (tools, adapters, local CLIs)
npx @ruvnet/ruview guidance --topic homecore --query "Wasmtime plugins"
npx @ruvnet/ruview nlos plan
npx @ruvnet/ruview nlos verify --repo .
npx @ruvnet/ruview spaces --resource spaces
npx @ruvnet/ruview spaces --resource events --limit 25
npx @ruvnet/ruview --help
```

The operator tools are pure Node and the published package has no runtime
dependencies (ADR-263 O3). MetaHarness, Darwin and Flywheel are exact-pinned
development dependencies used only for scoring, evolution proposals and
replay verification.

## Tools (`ruview_*`)

Exposed both as CLI verbs and as an MCP server (`npx @ruvnet/ruview mcp start`):

| Tool | What it does |
|------|--------------|
| `ruview_onboard` | Pick docker-demo / repo-build / live-esp32; print the next command |
| `ruview_claim_check` | Lint text for untagged / overstated accuracy claims (guardrail) |
| `ruview_verify` | Run `verify.py` deterministic proof → VERDICT |
| `ruview_node_monitor` | Assert CSI is flowing on an ESP32 (read-only) |
| `ruview_calibrate` | ADR-151 room pipeline (baseline→enroll→train-room→room-watch) |
| `ruview_node_flash` | Build+flash firmware (Windows/ESP-IDF; mutating, guarded) |
| `ruview_guidance` | Source-cited code map, capability maturity, validation commands, and limitations |
| `ruview_nlos_plan` | Advisory, phase-gated consumer transient-LiDAR and CSI fusion plan |
| `ruview_nlos_verify` | Static/build inspection plus fail-closed live-hardware evidence arithmetic |
| `ruview_spaces_list` | OAuth-only paging for sites/buildings/floors/spaces/zones/entities/events/alerts (guarded over MCP) |
| `ruview_memory_search` | Search the reviewed, source-cited contributor brain |

Every tool is **fail-closed**: missing repo / python / binary / port → an honest
negative, never a fabricated success.

### Consumer NLOS research gate

`nlos plan` and `nlos verify` are optional contributor aids for ADR-328 through
ADR-331. They do not capture sensors and are not required by the RuView runtime.
The verifier discovers `v2/crates/ruview-nlos`, `ui/ios-nlos`, and the NLOS
surface in `ui/mobile`. An absent optional surface is reported as `ABSENT`; a
partial, escaping, oversized, or marker-incompatible surface fails shallow
discovery. Contract tests remain authoritative. Add `--run-builds` to
run the build/test commands supported by the current host. Xcode and hardware
remain explicit skips when unavailable.

Repository selection and build execution are local-CLI capabilities. The MCP
tool auto-detects its checkout, performs read-only inspection/evidence checks,
and rejects `repo` or `run_builds` arguments. Run CLI builds only in a trusted
checkout: the environment is allowlisted and tails redacted, but Cargo/Swift/
Jest/Expo execution is not a sandbox.

The research gate accepts only a preregistered `LIVE_HARDWARE` record with an
external ground-truth capture manifest. It requires the pinned VL53L8CH
baseline to run at least 27 Hz and CSI fusion to reduce mean position error or
lost-track rate by at least 25 percent over 100 or more paired sequences, with
adjusted confidence,
paired aggregate arithmetic, independent CSI ablation, frozen guardrails, and
witness/privacy/security report digests. `SYNTHETIC` and replay
frames can validate software, but can never pass that gate:

```bash
npx @ruvnet/ruview nlos verify --repo . --run-builds
npx @ruvnet/ruview nlos verify --repo . \
  --evidence-file evidence/nlos/acceptance.json \
  --require-research-pass
```

The boundary is deliberate: the MIT method needs per-zone photon-arrival
histograms. Apple currently documents processed ARKit scene depth and mesh, so
the native adapter uses those for pose/context only; the web adapter consumes
authenticated tracks or visibly labelled deterministic replay. Neither surface
claims direct iPhone NLOS reconstruction.

### Cognitum Spaces OAuth

Activate the additional read scope through the Rust CLI, then use the same
validated client through the metaharness:

```bash
wifi-densepose login --spaces
wifi-densepose whoami
npx @ruvnet/ruview spaces
npx @ruvnet/ruview spaces --resource sites --limit 50
npx @ruvnet/ruview spaces --resource events --cursor '<opaque-next-cursor>'
```

The metaharness never accepts a bearer token or API key and removes
`COGNITUM_SPACES_API` from the child environment, so this surface cannot
silently fall back to the compatibility API-key path. The API origin is fixed
to `https://api.cognitum.one`, and the credentialed adapter requires an
installed `wifi-densepose` binary rather than running Cargo build scripts from
an auto-detected checkout. It returns only the bounded P2/P3 semantic
projection. `--resource` selects one of `sites`, `buildings`, `floors`,
`spaces`, `zones`, `entities`, `events`, or `alerts`; `--limit` is 1–100 and
`--cursor` is the opaque value from the prior page. An empty list is a valid
authenticated result, not sensing-quality evidence. An expired session may
rotate the stored refresh credential before the read completes.

MCP use is denied unless the server operator starts it with
`RUVIEW_MCP_GRANTS=credential-use`. Set `RUVIEW_CREDENTIALS_PATH` in the MCP
server environment when a non-default store is needed; MCP calls cannot choose
an arbitrary credential file or URL. `spaces:read` grants no write, pairing,
command, policy-approval, spending, or actuator authority. See the bundled
`cognitum-spaces` skill for the full playbook.

### Codebase guidance

`ruview_guidance` is the read-only starting point for unfamiliar work. Filter
by `architecture`, `sensing`, `hardware`, `training`, `homecore`,
`integrations`, `deployment`, `community`, or `testing`, and optionally add a
free-text query:

```bash
npx @ruvnet/ruview guidance --topic sensing --query "UDP CSI ingestion"
npx @ruvnet/ruview guidance --topic homecore --query "restore migration voice"
```

Each result separates implementation maturity from evidence, cites current
repository paths, names focused validation commands, and states known
limitations. In a RuView checkout, cited paths are checked before the result
passes. Outside a checkout, the tool labels them as a reviewed packaged
catalog. Related shared-brain records are bounded, reviewed, and treated only
as evidence.

## Skills

Host-neutral playbooks in `skills/` (`onboard`, `provision-node`, `calibrate-room`,
`train-pose`, `verify`, `cognitum-spaces`, `consumer-nlos`). `npx @ruvnet/ruview skill <name>`
prints one.

## Use as a Claude Code MCP server

The bundled `.claude/settings.json` registers the `ruview` MCP server
(`npx -y @ruvnet/ruview mcp start`). Drop this package's `.claude/` into a repo, or run
`npx @ruvnet/ruview install --host claude-code`.

## Hosts

Claude Code and Codex are implemented directly and tested with the local,
non-interactive CLIs:

```bash
npx @ruvnet/ruview agent run --host claude-code --repo . --prompt "Map the sensing-server startup path"
npx @ruvnet/ruview agent run --host codex --repo . --prompt "Find the nearest tests for HomeCore restore state"
```

Prompts travel over stdin, never through a shell. Both adapters are read-only by
default (`claude -p --safe-mode` in plan mode; `codex exec` in its read-only
sandbox with user config and exec rules ignored), use a scrubbed environment,
bound output/time, redact secrets, and require a trusted RuView checkout.
Workspace writes require both `--allow-write` and `--confirm`; dangerous bypass
flags are never emitted.

## Shared contributor brain

The committed `brain/corpus/core.jsonl` is a small, reviewable source of
repository facts. Every record has a source citation, evidence tier, tags, and
review state:

```bash
npx @ruvnet/ruview brain search --query "darwin community memory"
npx @ruvnet/ruview brain verify --repo .
npx @ruvnet/ruview brain propose --id finding-id --title "Finding" \
  --content "Source-bound observation" --sourcePath README.md --sourceLine 1 \
  --tags onboarding,docs --contributor github-user
```

Proposals are unreviewed JSONL for a normal pull request. Local vector indexes,
private overlays, raw agent transcripts, CSI/person data, and credentials are
never part of the shared corpus. Retrieved text is quoted evidence, not an
instruction or authority grant.

## Ruflo + Darwin/Flywheel

Development tooling is exact-pinned in `devDependencies`: `metaharness@0.4.1`,
`@metaharness/darwin@0.8.0`, and `@metaharness/flywheel@0.1.7`. Ruflo remains an
optional contributor coordinator rather than cold-start weight for the
dependency-free published MCP server:

```bash
claude mcp add --scope project ruflo -- npx -y ruflo@3.32.26 mcp start
codex mcp add ruflo -- npx -y ruflo@3.32.26 mcp start
```

`npm run flywheel:plan` is read-only. Darwin execution is human-triggered with
`node flywheel/run.mjs --confirm`; it writes only an untrusted
`.metaharness/` proposal archive. The protected gate requires frozen-anchor
retention, holdout lift, security and legacy-test success, verified provenance,
and human approval. No contributor run can directly replace or publish the
champion.

## License

MIT © ruvnet
