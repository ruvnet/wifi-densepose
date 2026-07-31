# wifi-densepose-sar-harness

Harness for [wifi-densepose-sar](https://crates.io/crates/wifi-densepose-sar) (ADR-287) — the coherent wideband RF tomography research crate this harness assists development on. Both are published: the crate on crates.io, this harness itself as [`wifi-densepose-sar-harness`](https://www.npmjs.com/package/wifi-densepose-sar-harness) on npm.

> Advanced Coding harness · domain: `software-engineering`. Generated with [create-agent-harness](https://github.com/ruvnet/agent-harness-generator).

## Behavioral rules

- Use the harness's MCP tools (`mcp__wifi-densepose-sar-harness__*`) for orchestration
- Memory and routing are handled by the kernel — you don't need to learn them
- Defer destructive operations to the user

## Agents

| Agent | Tier | Role |
|---|---|---|
| `architect` | opus | Designs the change before code is written. |
| `implementer` | sonnet | Writes code that matches the surrounding style. |
| `reviewer` | opus | Hunts correctness bugs in the diff. |
| `test-writer` | sonnet | Adds the missing tests for the change. |
## Skills

- `/plan-change` — Turn a feature request into a minimal, file-level implementation plan before any code.
- `/evolve` — Run Darwin Mode (`npm run evolve` / `evolve:dry`) to self-mutate the harness's own operating policy and keep only measurable improvements.

## Commands

Each command below has a matching `.claude/commands/<name>.md` guidance file — the MCP tool listing (`mcp__wifi-densepose-sar-harness__*`) is derived from these, so a new CLI subcommand isn't fully wired up until it has one too.

- `doctor` — Health-check the harness: kernel load, MCP wiring, memory backend, host adapter.
- `review-diff` — Review the current working diff for correctness, security, and reuse.
- `route <e0> <e1> <e2> <e3>` — cost-optimal model routing via `@metaharness/router` (needs `npm run build` first).
- `flywheel [generations]` — run the SYNTHETIC self-improvement demo via `@metaharness/flywheel` (needs `npm run build` first).

## Architecture

This harness uses [@metaharness/kernel](https://www.npmjs.com/package/@metaharness/kernel) — a Rust-compiled WASM module with a NAPI-RS native fallback — so the same code runs identically on every platform.

### Darwin, router, flywheel

Three complementary self-improvement/cost pieces, all real npm dependencies (not aspirational):

- **Darwin Mode** (`@metaharness/darwin`, devDependency) — `npm run evolve` (real sandbox) / `npm run evolve:dry` (mock sandbox) mutates the harness's own config and keeps only changes that measurably improve it. Wired by the scaffold itself.
- **Router** (`@metaharness/router`) — `src/router.ts` wires a real `Router` with a `qualityBar: 0.8` cost-optimal policy over two example model tiers. Its labelled examples are illustrative/seed data (see the file's honesty note), not measured eval-log observations — the routing *mechanism* is real and tested (`__tests__/router.test.ts`), the specific decisions it makes today are not yet backed by real data.
- **Flywheel** (`@metaharness/flywheel`) — `src/flywheel.ts` wires the real `runFlywheelGenerations` promotion loop (propose → evaluate → gate → promote, Ed25519-signed, independently replayable) with a SYNTHETIC proposer/evaluator (`dataSource: 'SYNTHETIC'`, no model call). It proves the wiring end-to-end (`__tests__/flywheel.test.ts` checks a real lift curve and a passing `verifyReplayBundle`); a LIVE run needs a real Proposer (model call) and Evaluator (real coding-task holdout/anchor suites) supplied by the operator — see the file's comments for exactly what those seams are.
