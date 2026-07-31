# ADR-286: `wifi-densepose-sar-harness` — a MetaHarness minted via `vendor/metaharness`

| Field | Value |
|-------|-------|
| **Status** | Accepted — implemented, **published** |
| **Date** | 2026-07-30 |
| **Parent** | ADR-287 (`wifi-densepose-sar`, the crate this harness assists development on) |
| **Relates to** | ADR-182 (`harness/ruview/`, the first MetaHarness-minted harness in this repo), ADR-285 (`harness/homecore/`, the WASM-first pattern this harness's `@metaharness/kernel` dependency follows) |
| **Published** | [`wifi-densepose-sar-harness` v0.1.0](https://www.npmjs.com/package/wifi-densepose-sar-harness) on npm (2026-07-31) |

## 0. PROOF discipline

Every claim below about what's "real" versus "illustrative"/"SYNTHETIC" is checked by a passing test in this harness's own suite (14 tests: 5 router + 5 flywheel + 4 install-smoke). Nothing here is asserted without a corresponding `__tests__/*.test.ts` file exercising it.

## 1. Context

`wifi-densepose-sar` (ADR-287) is a new, narrowly-scoped research crate. Rather than hand-roll a bespoke development-assistance setup for it, `vendor/metaharness` (the `ruvnet/metaharness` generator, vendored as a git submodule alongside this repo's other `vendor/*` submodules) was used to scaffold one directly: `npx metaharness analyze v2/crates/wifi-densepose-sar --scaffold wifi-densepose-sar-harness --host claude-code` recommended and generated `template: vertical:coding` with four agents (architect/implementer/reviewer/test-writer) and `doctor`/`review-diff` commands — the same generator that produced `harness/ruview/` (ADR-182) and `harness/homecore/` (ADR-285).

The user's ask that shaped this ADR's scope was specific: wire in **darwin, router, and flywheel** — three complementary `@metaharness/*` packages the base scaffold doesn't include by default (only Darwin Mode ships built-in).

## 2. Decision

Land the scaffold at `harness/wifi-densepose-sar/`, and add real wiring for the three requested pieces, each as an actual npm dependency (not a stub, not a `try/catch` optional import):

1. **`@metaharness/darwin`** (devDependency) — wired by the scaffold itself. `npm run evolve` (real sandbox) / `evolve:dry` (mock sandbox) mutates the harness's own operating config and keeps only measurably-improving changes.
2. **`@metaharness/router`** — `src/router.ts` wires a real `Router` (k-NN over labelled examples, cost-optimal selection against a quality bar) with two example model tiers (`cheap-tier` $1/MTok, `frontier-tier` $15/MTok). Exposed as a CLI command (`route <e0> <e1> <e2> <e3>`) with a matching `.claude/commands/route.md` guidance file.
3. **`@metaharness/flywheel`** — `src/flywheel.ts` wires the real `runFlywheelGenerations` promotion loop (propose → evaluate → gate → promote, Ed25519-signed, independently replayable via `verifyReplayBundle`) with a SYNTHETIC proposer/evaluator (`dataSource: 'SYNTHETIC'`, no live model call). Exposed as `flywheel [generations]` with a matching `.claude/commands/flywheel.md` guidance file.

Every new CLI subcommand gets a `.claude/commands/<name>.md` file, matching the pattern the base scaffold's `doctor`/`review-diff` already establish — the MCP tool listing (`mcp__wifi-densepose-sar-harness__*`) is derived from these, so a command without one isn't fully wired into the harness's own guidance surface even if the CLI itself works.

## 3. What this explicitly is NOT

- **Not evolving the crate.** Darwin/Flywheel mutate the harness's own operating policy (agent prompts, review checklist depth) — not `wifi-densepose-sar`'s Rust code or its runtime performance. Actually optimizing the crate (the incremental-phasor-rotation work, ADR-287 §7) was done directly, not through this harness's self-improvement loop.
- **Not a live routing/promotion system.** The router's labelled examples are illustrative seed data, not measured eval-log observations. The flywheel's proposer/evaluator are deterministic stand-ins, not a real model call or a real coding-task benchmark suite. Both are honestly labeled as such in their own source files and in this harness's `CLAUDE.md`.
- **Not manifest-verified.** `.harness/manifest.json`/`manifest.sha256` reflect the initial scaffold output and were not regenerated after adding `router.ts`/`flywheel.ts` — this scaffold has no `manifest:update` script (unlike `harness/homecore/`). Documented as a known gap in the harness's own README.

## 4. A real bug the flywheel wiring found

The first version of the SYNTHETIC evaluator returned a constant `noopRate`. `@metaharness/flywheel`'s default promotion gate requires `noopRate` to *strictly improve* generation over generation (one of its five conjunctive clauses) — a constant value, however good, fails that clause forever, so nothing could ever be promoted. Fixed by making `noopRate` actually respond to the (synthetic) policy content; every generation promotes now. Kept as a cautionary note in `src/flywheel.ts`'s comments: a flywheel evaluator with a frozen metric is silently broken, not silently fine.

## 5. Consequences

- 14 tests (5 router + 5 flywheel + 4 install-smoke), 0 failed; `npm run build` clean under strict TypeScript.
- Published to npm as `wifi-densepose-sar-harness` v0.1.0 — `npx wifi-densepose-sar-harness init` works from a cold install.
- No risk to any other harness or crate in this repo — this harness only reads/assists on `wifi-densepose-sar`, and its MCP server, memory namespace, and Claude Code plugin are scoped to its own name.
