---
name: evolve
description: "Evolve this harness with Darwin Mode — frozen model, evolving harness (real, sandboxed, safety-gated)."
---

# evolve — Darwin Mode self-improvement

`wifi-densepose-sar-harness` ships with **Darwin Mode** (`@metaharness/darwin`, ADR-070…146): the model
is frozen; the *harness* evolves. Each generation mutates ONE of the 7 surface files
(planner, contextBuilder, reviewer, retry/tool/memory/score policy), sandboxes each
child, scores it, and keeps only variants that *measurably* improve — building an
archive of successful descendants.

## Run it

```bash
npm run evolve        # real substrate: runs your test command per variant (deterministic mutator — no API key, no network)
npm run evolve:dry    # mock substrate: fast, fully offline, no test execution
```

Or directly:

```bash
npx metaharness-darwin evolve . --sandbox real --generations 3 --children 4
```

## Safety (secure by default)

- **Deterministic mutator** is the default — **no network, no API key, air-gapped**.
- Every mutation passes the `validateGeneratedCode` gate: no new imports, network,
  filesystem, shell, env access, or dependencies — pure refactor/tuning only.
- Mutations run in a **sandbox**; only variants that pass your tests are archived.
- Nothing is promoted without measured improvement (guard against Goodharting).

See `@metaharness/darwin` for selection strategies (`--selection`, `--crossover`,
`--curriculum`), statistical gates (`--fdr`, `--bench`), and the real-LLM mutator (library API).

## What the benchmarks taught us (measured, full SWE-bench Lite 300)

Defaults worth carrying into how you evolve and run this harness (full evidence + CIs in
`@metaharness/darwin`'s `LEARNINGS.md` / `bench/results/RESULTS.md`):

1. **Closed-loop repair is the #1 lever (~2×).** Feeding test/compiler failure back and retrying took
   resolve-rate 7.7% → 15.3% on the *same cheap model*. Iterate against ground truth, don't single-shot.
2. **Cheap-first + cost-aware routing.** Track **$/resolve**, not just resolve-rate; a cheap model
   resolved 31× cheaper per fix than a frontier one. Reserve frontier for *measured* capability gaps.
3. **Tier the models (Barbarian & Scholar).** Cheap sweep + frontier on *only the residual* = 33.3%
   at ~6× lower cost than running frontier everywhere.
4. **Put the output-format contract in a system message + example**, and size prompts to the model's
   real context window — this alone took a weak local model from 0% to ~50% valid output.
5. **Only trust batch evaluation of the final artifact** — in-loop counters drift 1.5–5×.
6. **The harness multiplies the model; it can't rescue one below the task's reasoning floor.** Pick
   the smallest model *above* the floor, then let evolution do the rest.
