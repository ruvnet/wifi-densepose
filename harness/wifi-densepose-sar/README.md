# wifi-densepose-sar-harness

Harness for wifi-densepose-sar

> **Advanced Coding** — Architect → implement → review → test, with a code-index MCP and push-guarded git perms.
>
> Generated with [`create-agent-harness`](https://github.com/ruvnet/agent-harness-generator). Multi-host scaffolding with a kernel that resolves native → wasm → js (js backend in the published beta; see `harness doctor`).

## Install

```bash
npm install -g wifi-densepose-sar-harness
wifi-densepose-sar-harness init
wifi-densepose-sar-harness doctor
```

## Agents

| Agent | Role |
|---|---|
| `architect` | Designs the change before code is written. |
| `implementer` | Writes code that matches the surrounding style. |
| `reviewer` | Hunts correctness bugs in the diff. |
| `test-writer` | Adds the missing tests for the change. |

This harness ships with the **claude-code** adapter.

## Darwin, router, flywheel

- `npm run evolve` / `evolve:dry` — Darwin Mode self-mutation of the harness's own config (`@metaharness/darwin`).
- `npm run route -- <e0> <e1> <e2> <e3>` (after `npm run build`) — cost-optimal model routing via `@metaharness/router`.
- `npm run flywheel:dry` — the SYNTHETIC `@metaharness/flywheel` self-improvement demo (propose → evaluate → gate → promote, signed + independently replayable).

See `CLAUDE.md`'s "Darwin, router, flywheel" section and the comments at the top of `src/router.ts` / `src/flywheel.ts` for what's real wiring vs. illustrative/synthetic data.

## Known gaps

- `.harness/manifest.json` / `manifest.sha256` reflect the initial `metaharness analyze --scaffold` output and were not regenerated after adding `src/router.ts`, `src/flywheel.ts`, and their tests — this scaffold has no `manifest:update` script (unlike `harness/homecore/`). Treat the manifest as historical provenance for the scaffold step, not a current file-integrity check.

## License

MIT
