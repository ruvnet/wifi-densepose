# ADR-289: `wifi-densepose-privshield-harness` — a MetaHarness for the VEIL privacy shield

| Field | Value |
|-------|-------|
| **Status** | Proposed — implemented (P1) |
| **Date** | 2026-08-09 |
| **Parent** | ADR-288 (`wifi-densepose-privshield` / VEIL, the crate this harness assists development on) |
| **Relates to** | ADR-286 (`wifi-densepose-sar-harness`, the per-crate harness scaffold this one mirrors), ADR-285 (`harness/homecore/`, the WASM-first `@metaharness/kernel` pattern), ADR-182 (`harness/ruview/`, the first minted harness), ADR-282 (L0–L5 evidence ladder) |
| **Location** | `harness/wifi-densepose-privshield/` |

## 0. PROOF discipline

Every claim below about what is "real" versus "illustrative"/"SYNTHETIC" is
checked by a test in this harness's own suite (router + flywheel + install-smoke
+ guidance). The dependency-free `guidance` surface is covered by
`__tests__/guidance.test.ts`, which runs even before `npm install`. Nothing here
asserts a MEASURED defense result — the harness surfaces the VEIL crate's
SYNTHETIC/L0 numbers with that label intact.

## 1. Context

`wifi-densepose-privshield` (ADR-288) is the VEIL privacy shield — a new,
narrowly-scoped crate. Following the pattern ADR-286 set for
`wifi-densepose-sar`, it gets a dedicated per-crate MetaHarness rather than a
bespoke setup: the `vertical:coding` scaffold (architect/implementer/reviewer/
test-writer, `doctor`) with `@metaharness/router`, `@metaharness/flywheel`, and
Darwin Mode wired in, plus a VEIL-specific, dependency-free `guidance` surface.

## 2. Decision

Land the harness at `harness/wifi-densepose-privshield/`, mirroring
`wifi-densepose-sar-harness`, with two deliberate improvements:

1. **Dynamic dependency imports.** `bin/cli.js` imports the `@metaharness/*`
   packages *inside* the commands that need them, not at module top. So
   `guidance`, `--help`, and the guidance test run with **zero dependencies
   installed** — useful for offline/air-gapped review and for this repo's CI
   before `npm install`. Only `init`/`doctor`/`route`/`flywheel` touch the
   kernel/host/router/flywheel packages.
2. **A VEIL `guidance` command.** A self-contained, source-cited, read-only
   capability map (topics: `overview`, `threat`, `countermeasure`,
   `compliance`, `optimization`, `experiment`), each entry carrying a summary,
   repo-relative source citations, focused validation commands, and explicit
   limitations — the `ruview_guidance` shape, specialized to VEIL. It labels all
   defense evidence `SYNTHETIC/L0` and states plainly that guidance is
   navigation, not authority.

The standard three self-improvement/cost pieces are wired as real npm
dependencies (not stubs):

- **`@metaharness/darwin`** (devDependency) — `npm run evolve` / `evolve:dry`
  mutates the harness's own operating config, keeping only measurable gains.
- **`@metaharness/router`** — `src/router.ts` wires a real cost-optimal `Router`
  (`qualityBar: 0.8`, k=1) over two model tiers, with four VEIL-shaped task axes
  (threatModeling / complianceReview / optimizerTuning / docWriting). Labelled
  examples are illustrative seed data (honesty note in-file).
- **`@metaharness/flywheel`** — `src/flywheel.ts` wires the real
  `runFlywheelGenerations` promotion loop (propose → evaluate → gate → promote,
  Ed25519-signed, independently replayable) with a SYNTHETIC proposer/evaluator
  (`dataSource: 'SYNTHETIC'`, no model call), over VEIL policy levers
  (`complianceReview`, `threatTriage`).

## 3. What this explicitly is NOT

- **Not a VEIL runtime.** The harness does not run a radio, emit RF, or jam. It
  assists *development* on the crate; it cannot execute the shield on hardware.
- **Not evolving the crate.** Darwin/Flywheel mutate the harness's own policy
  (agent prompts, review-checklist depth), not VEIL's Rust code. The crate's
  actual hyper-optimization (ADR-288 §opt) was done directly, in the crate.
- **Not a live routing/promotion system.** The router's examples are seed data;
  the flywheel's proposer/evaluator are deterministic stand-ins — both honestly
  labelled in-source and in `CLAUDE.md`.
- **Not a replacement for the crate's gates.** The authoritative check for a
  VEIL change remains `cargo test -p wifi-densepose-privshield`.
- **Not a re-labeller.** The harness must never present VEIL's SYNTHETIC results
  as MEASURED, and never scaffold interference-based ("jamming") defenses — both
  are hard rules in the harness `CLAUDE.md`.

## 4. Consequences

- The harness ships `guidance`/`doctor`/`init`/`route`/`flywheel`; `guidance`
  and `--help` work offline (validated here via `node bin/cli.js`), the rest
  after `npm install` + `npm run build` (CI).
- `.harness/manifest.json` + `manifest.sha256` are generated with real per-file
  hashes at creation (unlike ADR-286's scaffold, whose manifest was historical).
- Scoped to its own name: its plugin, permissions, and (future) MCP surface only
  read/assist on `wifi-densepose-privshield`. No risk to other harnesses/crates.

## 5. Validation

```bash
cd harness/wifi-densepose-privshield
node bin/cli.js guidance --topic overview   # dependency-free
npm ci && npm run build && npm test          # full suite (CI; needs registry access)
```
