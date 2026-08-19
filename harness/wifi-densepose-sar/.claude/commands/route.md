---
description: "Route a 4-axis task embedding to the cost-optimal model tier via @metaharness/router."
---

Route one query to the cheapest model tier predicted to clear the quality bar.

1. Run `npm run build` first if `dist/router.js` doesn't exist yet.
2. Run `wifi-densepose-sar-harness route <physicsExplanation> <codeReview> <numericalDebugging> <docWriting>` — four 0..1 numbers scoring how much the query looks like each of those four shapes (see `src/router.ts` for the axis definitions).
3. Report the picked tier (`cheap-tier` or `frontier-tier`), its predicted quality, and whether it cleared the 0.8 quality bar.
4. Remind the user: the labelled examples behind this decision are illustrative seed data (see `src/router.ts`'s honesty note), not measured eval logs — the routing mechanism is real, the specific pick isn't backed by production data yet.
