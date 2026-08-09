# wifi-densepose-privshield-harness

Harness for [`wifi-densepose-privshield`](../../v2/crates/wifi-densepose-privshield)
(WiFi Veil — the compliant-waveform WiFi-sensing privacy shield, ADR-288). This
package is defined by ADR-289.

> Advanced Coding harness · domain: `software-engineering`. Modeled on the
> `wifi-densepose-sar-harness` scaffold (ADR-286), generated with
> [create-agent-harness](https://github.com/ruvnet/agent-harness-generator).

## Behavioral rules

- Use the harness's tools for orchestration; memory and routing are handled by
  the kernel.
- Defer destructive operations to the user.
- **Never present WiFi sensing as camera-grade, and never relabel WiFi Veil's
  SYNTHETIC/L0 results as MEASURED** — a hardware witness is required first
  (CLAUDE.md hardware rule; ADR-282 ladder). The harness is a development aid;
  it does not run a radio and cannot emit RF.
- WiFi Veil uses compliant waveform controls only — **never jamming.** Do not add,
  suggest, or scaffold interference-based "defenses."

## Commands

- `init` — boot the kernel + host adapter.
- `doctor` — verify the install end-to-end (kernel, host, guidance map).
- `guidance --topic <t> [--query <q>]` — read-only WiFi Veil capability map
  (dependency-free; topics: `overview`, `threat`, `countermeasure`,
  `compliance`, `optimization`, `experiment`). Source-cited and
  evidence-labelled; navigation only, not authority.
- `route <e0> <e1> <e2> <e3>` — cost-optimal model routing via
  `@metaharness/router` (needs `npm run build`).
- `flywheel [generations]` — SYNTHETIC self-improvement demo via
  `@metaharness/flywheel` (needs `npm run build`).

## Architecture

Uses [@metaharness/kernel](https://www.npmjs.com/package/@metaharness/kernel)
(Rust-compiled WASM with a NAPI-RS native fallback) so the same code runs on
every platform. The `@metaharness/*` packages are imported *dynamically* inside
the commands that need them, so `guidance`/`--help` work with no dependencies
installed.

### Darwin, router, flywheel

- **Darwin Mode** (`@metaharness/darwin`, devDependency) — `npm run evolve` /
  `evolve:dry` mutates the harness's own config and keeps only measurable
  improvements.
- **Router** (`@metaharness/router`) — `src/router.ts` wires a real cost-optimal
  `Router` (`qualityBar: 0.8`) over two model tiers. Its labelled examples are
  illustrative seed data (see the file's honesty note), not measured eval-log
  observations.
- **Flywheel** (`@metaharness/flywheel`) — `src/flywheel.ts` wires the real
  promotion loop (propose → evaluate → gate → promote, Ed25519-signed,
  independently replayable) with a SYNTHETIC proposer/evaluator
  (`dataSource: 'SYNTHETIC'`, no model call). A LIVE run needs a real Proposer
  and Evaluator supplied by the operator — see the file's comments.

## Relationship to the crate

This harness assists development *on* the WiFi Veil crate; it does not replace the
crate's own gates. The authoritative validation for a WiFi Veil change is still:

```bash
cargo test -p wifi-densepose-privshield --no-default-features
cargo clippy -p wifi-densepose-privshield --all-targets -- -D warnings
```
