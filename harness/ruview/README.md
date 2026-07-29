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

Every tool is **fail-closed**: missing repo / python / binary / port → an honest
negative, never a fabricated success.

## Skills

Host-neutral playbooks in `skills/` (`onboard`, `provision-node`, `calibrate-room`,
`train-pose`, `verify`). `npx @ruvnet/ruview skill <name>` prints one.

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
