---
description: "Run the SYNTHETIC @metaharness/flywheel self-improvement demo and report the lift curve."
---

Run the propose → evaluate → gate → promote loop and report what it found.

1. Run `npm run build` first if `dist/flywheel.js` doesn't exist yet.
2. Run `wifi-densepose-sar-harness flywheel [generations]` (default 3 if omitted).
3. Report each generation's `primary` score and `delta`, the total promotions, and whether `verifyReplayBundle` passed.
4. State plainly: this run's `dataSource` is `SYNTHETIC` — the proposer and evaluator are deterministic stand-ins (see `src/flywheel.ts`'s honesty note), not a real model call or a real coding-task benchmark. Do not report its numbers as if they reflect real harness improvement.
