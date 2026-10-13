# Synthetic active-sensor KGE experiment

This uses the **published `@ruvector/kge` 0.2.0**, pinned by exact version and npm integrity in `package-lock.json`. It uses the package's real native/WASM loader; there is no mock scorer or substitute fallback. The local Linux x64 run used the bundled native engine (internal version `2.4.0`, distinct from npm version `0.2.0`). Package license: MIT. Node 18 or later is required. No hardware requests, network calls during inference, model promotion, or occupancy claims are implemented.

## Reproduce

```sh
cd scripts/active-kge
npm ci --ignore-scripts --no-audit --no-fund
npm test
mkdir -p runs
node cli.mjs --train runs/seed42 --context context_0_8 > runs/ranking.json
node cli.mjs --artifact runs/seed42 --context context_1_8 --k 3 > runs/ranking-next.json
```

`--train` requires a new directory; it will not overwrite any experiment. Model and receipt are written with exclusive creation and made read-only after success. Failed training writes a separate `failed.json`. `--artifact` verifies the model SHA-256 and bounded fixture contract before loading. The loader also checks the exact authored entity/relation vocabulary, indexed training triples, split tags, model configuration and training receipt; rehashing an altered graph cannot satisfy that contract. Source hashes are captured before training and checked again after measurement. These hashes detect accidental changes, not adversarial replacement of both model and receipt; they are not signatures.

Use `node cli.mjs` for clean stdout, or `npm --silent run rank -- ...`. Successful stdout is exactly one JSON object:

```json
{"schemaVersion":1,"evidence":"SYNTHETIC","candidates":[{"sensorId":"wifi","score":2.9},{"sensorId":"door","score":1.3},{"sensorId":"radar","score":1.2}]}
```

Those numbers illustrate the shape, not a measured receipt. The allowlist is exactly `wifi`, `radar`, `door`; top K is an integer from 1 to 3. Scores are uncalibrated link-ranking values, **not probabilities**. The generated file is the input for the Rust probe's `--active-loop --ranking FILE`; it does not itself trigger any sensor.

## Frozen experiment and evidence limits

`fixture()` authors three abstract feature groups with ten contexts each. All thirty `observedFeature` links are training side information. Eight contexts per group have a training `relevantSensor` link (24 total); the other two target links per group are held out (six total). Held-out target links never enter `addTriples`, training, optimizer selection, or the permutation control. All 54 admitted facts carry explicit `train` tags. This is **transductive link holdout**, not unseen-context or unseen-room generalization: the held-out context entities and their feature links are already known.

A single frozen HolE configuration uses dimensions 32, seed 42, 100 epochs and learning rate 0.05. No hyperparameter search or automatic optimization/promotion is called. Exact `predict` scores all 36 entities, then filters to the three sensor IDs. HolE relation composition is not used or claimed.

The receipt reports MRR and Hits@1 against (1) a training-frequency baseline with fixed wifi/radar/door tie breaking and (2) another trained KGE whose 24 relevance labels are shuffled deterministically while preserving their class counts. It also records the package/backend, model and control hashes, native/WASM artifact hash, fixture hash, source/lock hashes and environment. The backend seed is passed at model creation and training; tests confirm identical model hashes on repeated runs in the tested environment, not across all platforms.

The allocation-reducing cached-context candidate remains experimental: its first fixed paired comparison failed the round-one p95 gate, so the original validation path remains the default and no speed improvement is claimed. Later receipt timings do not automatically promote it.

Latency measures the complete `rank` call (validation, statistics, exact prediction, filtering and sorting), with 100 warmups followed by 1,000 calls per runtime in each of three paired rounds, rotating the six held-out contexts and alternating runtime order on an identical model. Both median and p95 must be lower in every round to pass the predeclared comparison gate. Loading and training are excluded. Throughput is derived from mean query latency; it is not sensor throughput. Timings are machine-specific.

The synthetic groups encode the task by construction. Strong results demonstrate that this package can recover those authored links, not that WiFi, radar or door sensors have measured relevance in a real room. The six-query evaluation is tiny, and its uncertainty must not be generalized to occupancy sensing. No real signals, participant data, empty-room captures or independent operational labels were used.

## Primary package sources

- Published metadata/version: https://registry.npmjs.org/@ruvector%2fkge/0.2.0 (verified 2026-10-13; published 2026-09-28).
- Package repository pin reported by npm: https://github.com/ruvnet/ruvector/tree/e7b0bc37e4042382a7d981ce03e03aa1e5db35a6/npm/packages/kge
- The installed release's `dist/client.d.ts`, `dist/types.d.ts`, `dist/binding.js` and `index.js` were inspected before implementation. Used API: `createKge`, `addTriples`, `train`, `predict`, `stats`, `save`, `loadKge`.
