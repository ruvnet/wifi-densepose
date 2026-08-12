# wifi-veil-harness

A metaharness (contributor harness) for
[`wifi-veil`](..) — **WiFi Veil**,
the compliant-waveform WiFi-sensing privacy shield (ADR-288). Defined by ADR-289.

> **Advanced Coding** — architect → implement → review → test, plus a
> dependency-free WiFi Veil guidance surface. Modeled on `wifi-densepose-sar-harness`
> (ADR-286). Multi-host scaffold with a kernel that resolves native → wasm → js.

## Install

```bash
npm install -g wifi-veil-harness
wifi-veil-harness doctor
```

Or run without installing:

```bash
npx wifi-veil-harness guidance --topic overview
```

## Commands

| Command | Deps needed | Purpose |
|---|---|---|
| `init` | kernel + host | Boot the kernel + host adapter |
| `doctor` | kernel + host | Verify the install end-to-end |
| `guidance --topic <t>` | **none** | Read-only WiFi Veil capability map (source-cited, evidence-labelled) |
| `route <e0..e3>` | router + `npm run build` | Cost-optimal model routing |
| `flywheel [gens]` | flywheel + `npm run build` | SYNTHETIC self-improvement demo |

`guidance` topics: `overview`, `threat`, `countermeasure`, `compliance`,
`optimization`, `experiment`. It needs no dependencies or build step, so it
works offline and in CI before `npm install`.

## What WiFi Veil is

WiFi Veil shapes a node's **own** beamforming feedback with keyed Givens rotations so
a third-party passive sniffer cannot re-identify people, while a keyed receiver
sees an essentially unchanged link. **Compliant waveform controls only — never
jamming.** Reference results are **SYNTHETIC / evidence level L0** (reproduced by
`cargo test`), never MEASURED until a hardware witness exists. See the crate's
[ADR-288](../../docs/adr/ADR-288-veil-privacy-shield-compliant-waveform.md) and
[research bundle](../../docs/research/privacy-shield/).

## Darwin, router, flywheel

- `npm run evolve` / `evolve:dry` — Darwin Mode self-mutation of the harness
  config (`@metaharness/darwin`).
- `npm run route -- <e0> <e1> <e2> <e3>` (after `npm run build`) — cost-optimal
  model routing (`@metaharness/router`).
- `npm run flywheel:dry` — the SYNTHETIC `@metaharness/flywheel` demo
  (propose → evaluate → gate → promote, signed + independently replayable).

See `CLAUDE.md` and the honesty notes atop `src/router.ts` / `src/flywheel.ts`
for what is real wiring vs. illustrative/synthetic data.

## Scope

The harness is a **development aid**. It does not run a WiFi Veil radio, does not
emit RF, and cannot jam. It does not replace the crate's own gates — the
authoritative check for a WiFi Veil change is `cargo test`.

## License

MIT
