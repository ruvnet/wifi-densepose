# ADR-263: `@ruvnet/ruview` npm Harness — Deep Review + Optimization Strategy

| Field | Value |
|-------|-------|
| **Status** | Accepted — **implemented** (O1–O9, `@ruvnet/ruview@0.2.0`): fail-closed `claim-check`, async MCP dispatch (ping answered mid-`verify`, pinned by e2e test), zero-dependency install, bounded output tails, argv-passed monitor port, package.json-sourced version, prepack skill sync, memoized `which()`, underscore-canonical tools with dotted aliases, word-boundary guardrail matching. 30/30 tests (MEASURED, `node --test test/*.test.mjs`); CI gate in ADR-265's `npm-packages.yml` |
| **Date** | 2026-07-02 |
| **Deciders** | ruv |
| **Codename** | **RUVIEW-NPM-REVIEW-1** |
| **Supersedes / amends** | none (records review of the ADR-182 P1+P2 artifact; feeds ADR-265 distribution strategy) |

## Context

ADR-182 minted and published **`@ruvnet/ruview@0.1.0`** (`harness/ruview/`) — the
`npx ruview` operator harness: a dependency-free ESM CLI + minimal MCP stdio server
exposing six `ruview.*` tools (onboard / claim_check / verify / node_monitor /
calibrate / node_flash), five skill playbooks, and the executable
MEASURED-vs-CLAIMED guardrail (`src/guardrails.js`). The package is live on npm
(0.1.0, 49.5 kB unpacked / 21 files — MEASURED, `npm view @ruvnet/ruview` +
`npm pack --dry-run`) and is the recommended MCP registration path
(`npx -y @ruvnet/ruview mcp start` in the bundled `.claude/settings.json`).

This ADR is the first dedicated deep review of that npm artifact: correctness,
fail-open/fail-closed posture, performance (cold start + request handling),
packaging hygiene, and security of the subprocess surface. All 17 bundled tests
pass on Node 22 (MEASURED, `node --test test/*.test.mjs`, 17/17, ~108 ms).

## Findings

Severity reflects impact on the package's stated contract: *fail-closed operator
tools + an honesty guardrail that must never fail open*.

### F1 (HIGH, fail-open): `claim-check` passes silently on empty input

`bin/cli.js` `claim-check` with **neither `--text` nor `--file`** sends
`text: undefined` → `claimCheck(String(args.text ?? ''))` → `''` → `ok: true`,
**exit 0**. A CI hook wired as `npx ruview claim-check --text "$BODY"` where
`$BODY` expands empty therefore reports PASS. This is the single tool whose whole
purpose is to fail closed; empty input must be an error, not a pass.
Reproducer: `node bin/cli.js claim-check` → `{"ok": true}`, exit 0.

### F2 (HIGH, head-of-line blocking): MCP server is fully synchronous

`src/mcp-server.js` dispatches `tools/call` inside the readline `line` handler,
and every heavyweight handler in `src/tools.js` uses **`spawnSync`**
(`ruview.verify` up to 180 s, `ruview.calibrate` up to 300–600 s,
`ruview.node_monitor` up to `seconds+10`). While one call runs, the event loop is
blocked: `ping`, `tools/list`, and concurrent `tools/call` requests are not even
read from stdin. Hosts that health-check with `ping` during a long `calibrate`
will conclude the server is dead and kill it mid-run.

### F3 (MEDIUM, cold start): optionalDependencies triple the `npx` install for a path that never uses them

`package.json` declares `optionalDependencies` on `@metaharness/kernel` and
`@metaharness/host-claude-code`. npm installs optional deps **by default**, so
every cold `npx -y @ruvnet/ruview mcp start` fetches 3 extra packages (kernel +
host + transitive `@ruvector/emergent-time`). MEASURED (npm 10.9.7, this
container): default install = **4 packages, 620 kB, 71 files**; with
`--omit=optional` = **1 package, 172 kB, 22 files**. The operator-tool and MCP
paths never import these — only `doctor`/`install` do, and both already
dynamic-import inside `try/catch` and degrade gracefully when absent
(`kernel/host: not installed (ok…)`). The optional deps buy nothing on the hot
path and cost 3 registry round-trips + ~450 kB on every cold start.

### F4 (MEDIUM, silent truncation): `spawnSync` default `maxBuffer` (1 MiB)

`run()` in `src/tools.js` never sets `maxBuffer`. `cargo run -p
wifi-densepose-cli` (the `calibrate` fallback path) and a chatty `verify.py` can
exceed 1 MiB of stdout, at which point the child is killed with `ENOBUFS` and the
tool reports a spawn error that looks like a proof/calibration failure. The
handlers only ever consume the last 8 kB/1.5 kB; buffering should be bounded but
generous (e.g. `maxBuffer: 16 MiB`) or streamed with a tail ring.

### F5 (MEDIUM, injection surface): `node_monitor` interpolates the port into Python source

The handler builds a `python -c` script by string interpolation:
`` `ser=serial.Serial(${JSON.stringify(port)},115200,…)` `` and
`` `while time.time()-t<${dur}:` ``. `JSON.stringify` produces a *JavaScript*
string literal; Python string-literal semantics differ at the edges (`\uXXXX` is
shared, but e.g. JS emits raw U+2028/U+2029 unescaped pre-ES2019 rules aside, and
any future non-JSON-safe field added the same way would be executable). `port`
arrives from the MCP caller (an agent), so this is an agent-controlled string
concatenated into an interpreter invocation. `dur` is `Number()`-guarded; `port`
should be passed out-of-band (`sys.argv`/env), never spliced into source.

### F6 (LOW, drift): server version hardcoded

`SERVER_INFO = { name: 'ruview', version: '0.1.0' }` in `src/mcp-server.js`
duplicates `package.json.version` (the CLI's `--version` already reads
package.json at runtime). First release bump will drift the MCP handshake
version.

### F7 (LOW, duplication): every skill ships twice

`skills/*.md` and `.claude/skills/*/SKILL.md` are byte-identical (same sha256 in
`.harness/manifest.json`). ~8 kB of the 49.5 kB unpacked payload is duplicate
content, and — worse than size — two copies must be kept in sync by hand.

### F8 (LOW, perf + portability): `which()` is uncached and shells out

`which()` runs up to twice per tool call (`python` then `python3`), each a
blocking `spawnSync`; the POSIX branch spawns a shell (`shell: true`). Results
are stable for the process lifetime and should be memoized; the lookup can be
done dep-free with a PATH scan instead of a shell.

### F9 (LOW, interop): dot-named tools + minimal protocol surface

Tool names (`ruview.onboard`, `ruview.claim_check`, …) contain dots. MCP itself
does not restrict names, but downstream host APIs commonly enforce
`^[a-zA-Z0-9_-]{1,64}$` for tool names; hosts must then sanitize or reject.
The server also answers `resources/list` / `prompts/list` with `-32601` (it does
not advertise those capabilities, so this is spec-legal, but empty-list stubs are
cheaper than every host's error path). Protocol version is pinned to
`2024-11-05` with no negotiation fallback. None of this breaks Claude Code today;
it narrows portability, which is the harness's whole pitch (9 hosts, ADR-182).

### F10 (LOW, CI gap): the published package has zero CI

No workflow under `.github/workflows/` runs `harness/ruview` tests (checked:
no workflow references `harness/ruview`, `ruview-mcp`, or `ruview-cli`), and
`ci.yml` pins `NODE_VERSION: '18'` while the package declares
`engines.node >= 20`. Note also `node --test test/` (directory form) fails on
Node 22 while the documented glob form passes — CI should pin the working
invocation. Consolidated CI/publish strategy is ADR-265.

### F11 (MEDIUM, guardrail precision): `METRIC_TERMS` substring matching false-positives on ordinary prose

Found by dogfooding this review: `claimCheck` matches metric terms with
`lower.includes(t)`, so the two-character terms `'map'` and `'f1'` fire inside
ordinary words and labels — "source **map**s", "the **map**s can never
resolve", finding IDs like "**F1** (HIGH…)". MEASURED reproducer: running
`npx ruview claim-check --file` over this ADR and ADR-264 yields 4 and 16
medium findings respectively, the majority of which are `map`/`F1`
false positives on lines carrying no accuracy claim. A guardrail that cries
wolf trains people to ignore it — precision is part of its fail-closed
contract. Short/ambiguous terms need word-boundary matching (`\bmap\b`,
`\bf1\b`, likewise `auc`, `iou`), and section-heading label patterns
(`F\d+`, `O\d+`) should not count as metric mentions.

## Decision

Adopt the following optimization strategy, in priority order. Each item is
independently shippable; F-numbers map to findings.

- **O1 (F1):** `claim-check` with no `--text`/`--file` (or empty text after read)
  exits 2 with a usage error. Add a regression test pinning exit ≠ 0.
- **O2 (F2):** make the MCP dispatch async: convert `run()`/`which()` to
  promise-based `spawn`, make `tools/call` handlers `async`, and keep reading
  stdin while calls run (respond to `ping`/`tools/list` concurrently; serialize
  only same-tool hardware operations). Acceptance: `ping` round-trips < 50 ms
  while a synthetic 30 s `calibrate` is in flight.
- **O3 (F3):** drop the two `optionalDependencies`; `doctor`/`install` already
  degrade and should print the exact `npm i @metaharness/kernel
  @metaharness/host-claude-code` hint on the miss path. Acceptance: cold
  `npm i @ruvnet/ruview` installs exactly 1 package (MEASURED baseline above).
- **O4 (F4):** set `maxBuffer: 16 * 1024 * 1024` in `run()` (or stream + tail).
- **O5 (F5):** pass `port` to the monitor script via `sys.argv`
  (`python -c script -- <port>`), never by source interpolation.
- **O6 (F6):** read the MCP `serverInfo.version` from `package.json` once at
  startup (same pattern the CLI already uses).
- **O7 (F7):** make `skills/*.md` the single source and generate
  `.claude/skills/*/SKILL.md` in a `prepack` script (or vice versa); manifest
  hashes then pin one canonical set.
- **O8 (F8, F9):** memoize `which()`; add underscore aliases for the dot-named
  tools (accept both in `tools/call`, advertise the underscore form) and add
  empty `resources/list` / `prompts/list` stubs.
- **O9 (F11):** switch `METRIC_TERMS` matching to word-boundary regexes for
  short terms (`map`, `f1`, `auc`, `iou`) and skip label tokens matching
  `\b[FO]\d+\b`. Acceptance: `claim-check --file` over ADR-263/264/265 reports
  only the genuinely tagged-or-taggable percentage lines, and the existing 17
  guardrail tests still pass plus new false-positive pins ("source maps",
  "F1 (HIGH)" → no finding).

Non-goals: no new runtime dependencies (the zero-dep MCP server is a feature,
not an accident — keep it), no build step, no change to the fail-closed tool
contracts.

## Consequences

- The honesty guardrail becomes fail-closed end-to-end (its current empty-input
  pass is the exact failure mode the guardrail exists to prevent).
- `npx` cold start drops ~450 kB / 3 packages (MEASURED baseline in F3) with no
  feature loss; `doctor` output already communicates the optional-dep story.
- Long-running `verify`/`calibrate` no longer starve the MCP channel — the
  harness survives host health checks during real calibration runs.
- Two-copy skill drift becomes impossible at pack time.
- Costs: async conversion touches every handler signature in `src/tools.js`
  (mechanical, ~6 handlers); alias tools add a small compatibility table.
- Verification for the implementing PR: bundled tests extended for O1/O2/O5
  (target ≥ 20 tests), `npm pack --dry-run` file-count asserted, and the F3
  install measurement re-run and quoted MEASURED in the PR body — which must
  itself pass `npx ruview claim-check`.
