# ADR-264: `@ruvnet/rvagent` MCP Server + `@ruv/ruview-cli` — Deep Review + Optimization Strategy

| Field | Value |
|-------|-------|
| **Status** | Accepted — **implemented** (O1–O9, `@ruvnet/rvagent@0.2.0`): `exports` fixed (types-first, no phantom `.cjs`), map-free tarball (127,704 B unpacked / 46 files / 0 maps — MEASURED, `npm pack --dry-run`, from 188 kB), Streamable HTTP **wired** behind `RVAGENT_HTTP_PORT` with per-session transports + 1 MiB body cap + port-aware origin gate, underscore tool names with dotted router aliases, single Zod validation gate with generated JSON Schemas, fd-leak fixed + persisted job records + bounded log tails, probing `detectCogBinary`, package.json-sourced version, `ruview-cli` bin renamed. 99/99 jest tests (MEASURED); both transports smoke-tested live |
| **Date** | 2026-07-02 |
| **Deciders** | ruv |
| **Codename** | **RUVIEW-NPM-REVIEW-2** |
| **Supersedes / amends** | none (reviews the ADR-104/ADR-124 artifacts; feeds ADR-265 distribution strategy) |

## Context

Two TypeScript npm packages expose RuView sensing to agents and shells:

- **`@ruvnet/rvagent@0.1.0`** (`tools/ruview-mcp/`) — SENSE-BRIDGE, the MCP
  server over the sensing-server HTTP API + cog binaries: 12 tools
  (csi/pose/count/registry/train/job + ADR-124 BFLD/presence/vitals). Published
  (188 kB unpacked — MEASURED, `npm view @ruvnet/rvagent`). Deps:
  `@modelcontextprotocol/sdk` + `zod`.
- **`@ruv/ruview-cli@0.0.1`** (`tools/ruview-cli/`) — `private: true` yargs CLI
  mirroring the same capabilities; intentionally duplicates `http.ts`/`cog.ts`/
  `config.ts` (~150 lines) to stay standalone.

This ADR records a deep review of both: packaging correctness (verified against
the **published** tarball, not just the source tree), protocol/interop, resource
lifecycle, and the honesty of the package's own self-description — the same
MEASURED-vs-CLAIMED bar the project applies to accuracy numbers.

## Findings

### F1 (HIGH, broken export): `require` condition points at a file that does not exist

`package.json` `exports["."].require = "./dist/index.cjs"`, but the build is
plain `tsc` (ESM only) and **the published 0.1.0 tarball contains no
`index.cjs`** (verified by listing the registry tarball). Any CJS consumer doing
`require('@ruvnet/rvagent')` resolves to a nonexistent file →
`ERR_MODULE_NOT_FOUND`. Additionally the `types` condition is listed **after**
`import`/`require`; TypeScript requires `types` first or it may be ignored under
`moduleResolution: bundler/node16`.

### F2 (MEDIUM, tarball bloat): a third of the published package is dead source maps

The 0.1.0 tarball ships **44 `.map` files = 62,698 B** against 78,209 B of
actual `.js` (MEASURED, extracted registry tarball). `src/` is not published, so
every `sourceMappingURL` points at `../src/*.ts` that consumers do not have —
the maps can never resolve. Also `files` lists `CHANGELOG.md`, which does not
exist in `tools/ruview-mcp/` (npm silently skips it), so the advertised file set
is partly fictional.

### F3 (MEDIUM, honesty): the package description claims a transport it does not start

The description reads "**dual-transport MCP server (stdio + Streamable HTTP)**",
but `main()` in `src/index.ts` wires **stdio only**. `http-transport.ts` is a
complete, tested scaffold that nothing imports at runtime — there is no flag,
env var, or subcommand that starts it. By this project's own rule this is a
CLAIMED capability presented as shipped. Either wire it (`--http` /
`RVAGENT_HTTP_PORT` gate) or de-claim the description until it is.

### F4 (MEDIUM, interop + inconsistency): two tool-naming conventions, one of them dot-based

Six tools use `ruview_snake_case`; six (ADR-124 additions) use
`ruview.dotted.names`. Same interop caveat as ADR-263 F9 (host tool-name
regexes commonly `^[a-zA-Z0-9_-]{1,64}$`), plus the split convention makes the
tool surface look like two products. Standardize on underscores and accept the
dotted forms as aliases for one deprecation cycle.

### F5 (MEDIUM, double work + drift): every tool input is validated twice from two hand-maintained schemas

`CallToolRequestSchema` handler runs `TOOL_INPUT_SCHEMAS[name].safeParse(args)`,
then each tool handler runs its own `schema.parse(args)` again — two full Zod
passes per call. Separately, the `inputSchema` JSON advertised via `tools/list`
is **hand-written** and duplicates the Zod schema field-by-field (defaults,
min/max, descriptions) — schema drift between what is advertised and what is
enforced is a matter of time. Parse once at the gate, pass the typed result to
handlers, and generate the advertised JSON Schema from the Zod source
(`zod-to-json-schema` at build time, or Zod 4's native `z.toJSONSchema` when the
SDK's peer range allows).

### F6 (MEDIUM, resource lifecycle): `train_count` leaks 2 fds per job; job registry is process-local

`trainCount` opens `logFdOut`/`logFdErr` with `openSync` and never closes them
in the parent — the spawned cargo child inherits duplicates, but the parent's
descriptors stay open for the MCP server's lifetime: 2 leaked fds per training
job. `jobRegistry` is an in-memory `Map`, so `ruview_job_status` after a server
restart reports "not found" for a training run that is still burning GPU (the
source comments acknowledge this; the fix — persist `~/.ruview/jobs/<id>.json`,
already the documented layout — is small). Also `jobStatus` re-`import`s
`node:fs` on every poll and reads the entire log to return 20 lines.

### F7 (MEDIUM, security/robustness of the HTTP scaffold): unbounded body + one shared session transport

`http-transport.ts` buffers the request body with no size cap (memory DoS the
moment it is wired to a socket), reuses a **single**
`StreamableHTTPServerTransport` with `sessionIdGenerator` for all clients (the
SDK's stateful mode expects one transport per session — a second client's
`initialize` collides), and the Origin allowlist is exact-match
(`http://localhost` will not match a real browser origin `http://localhost:5173`).
Must be fixed **before** F3 wires it in; bearer-token + 127.0.0.1 defaults are
already right.

### F8 (LOW, dead/misleading code): `detectCogBinary` always returns the bare name

It builds a 4-candidate appliance-path array and then returns
`candidates[candidates.length - 1]` — i.e. always `name` — without checking
existence. The candidates are dead weight that reads as if path detection
happens. Either probe with `existsSync` or delete the array.

### F9 (LOW, drift + hygiene): hardcoded versions, unused/mismatched devDeps, bin-name collision

`PACKAGE_VERSION = "0.1.0"` (index.ts) duplicates package.json;
`@types/express` is unused (`http-transport` uses `node:http`); `@types/jest@30`
against `jest@29`; `ruview-cli` hardcodes `.version("0.0.1")`. And
`@ruv/ruview-cli` claims the **`ruview`** bin name, which collides with
`@ruvnet/ruview`'s bin (ADR-182) if both are ever installed globally —
ADR-263/265 give the `ruview` name to the harness; the CLI must rename or fold.

## Decision

- **O1 (F1):** fix `exports`: drop the `require` condition (ESM-only is fine for
  a bin-first package) or add a real CJS build; put `types` first. Add a CI
  smoke test that does `npm pack` + `node -e "import('<tarball install>')"`.
- **O2 (F2):** publish without maps: `declarationMap: false`, `sourceMap: false`
  in a `tsconfig.build.json` used by `prepack` (or add `!dist/**/*.map` to
  `files`). Remove the phantom `CHANGELOG.md` entry or create the file.
  Acceptance: unpacked size ≤ ~125 kB (from 188 kB — MEASURED, `npm pack --dry-run`).
- **O3 (F3, F7):** wire the HTTP transport behind an explicit opt-in
  (`RVAGENT_HTTP_PORT` or `--http`), after F7 fixes: per-session transport map
  keyed by `mcp-session-id`, 1 MiB body cap, origin matching that honors ports
  (compare `URL.origin` prefixes or document exact origins). Until then, change
  the description to "stdio MCP server (Streamable HTTP scaffold, unwired)".
- **O4 (F4):** rename dotted tools to underscore (`ruview_bfld_last_scan`, …),
  keep dotted aliases in the call router for one release, note it in the README.
- **O5 (F5):** single validation gate: the registry maps name → Zod schema →
  typed handler; advertised `inputSchema` generated from Zod at build time.
- **O6 (F6):** close parent fds after spawn (`closeSync` post-`spawn` — the
  child holds its own copies), persist job records to
  `<jobsDir>/<id>.json`, and read log tails with a bounded read.
- **O7 (F8):** make `detectCogBinary` actually probe (`existsSync` over the
  candidates) — it is the entire reason the function exists.
- **O8 (F9):** single-source versions from package.json; drop `@types/express`;
  align `@types/jest` with jest 29 (or move to `node:test` like the harness and
  drop the jest toolchain entirely — it is the heaviest devDep in both
  packages).
- **O9 (F9, scope):** fold `@ruv/ruview-cli` into `rvagent` as a second bin
  (`rvagent-cli`) sharing `http/cog/config`, or keep it private-forever and say
  so in its README. Its `ruview` bin name is surrendered to `@ruvnet/ruview`
  either way.

## Consequences

- CJS consumers stop hitting a guaranteed-broken export path (F1 is the only
  finding that fails for every consumer of that entry point deterministically).
- The published artifact shrinks ~33% (MEASURED, F2 tarball listing: 62,698 B
  of maps in a 188 kB unpacked payload) and stops advertising files/transports
  it does not contain — the package description itself passes the project's
  claim-check bar.
- One schema source ends advertised-vs-enforced drift and halves per-call
  validation cost; naming unification makes the 12-tool surface read as one
  product and survive strict host tool-name validation.
- Long-lived MCP servers stop accumulating fds during training campaigns, and
  job polling survives restarts.
- Costs: the alias cycle (O4) briefly doubles the advertised tool count unless
  aliases are router-only (recommended: router-only, advertise underscore names
  exclusively); folding the CLI (O9) retires a package name already in use in
  scripts, so it needs a deprecation note.
- Verification for the implementing PR: `npm pack --dry-run` asserted file list
  (no `.map`, no phantom entries), pack-size budget in CI (ADR-265), jest/`node
  --test` suite green, and a tarball-install smoke test for both `import` and
  the `rvagent` bin.
