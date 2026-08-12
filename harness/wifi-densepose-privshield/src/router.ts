// SPDX-License-Identifier: MIT
//
// Cost-optimal task routing for the wifi-densepose-privshield (VEIL) harness,
// via @metaharness/router: route each agent query to the cheapest model
// predicted to clear a quality bar, instead of defaulting every query to the
// frontier tier.
//
// HONESTY NOTE: the candidate `examples` below are SEED/ILLUSTRATIVE data —
// four hand-picked (embedding, quality) points per candidate, not measured
// eval-log observations. They exist so `veilTaskRouter` is a real, runnable
// k-NN router out of the box (see __tests__/router.test.ts), not so its routing
// decisions should be trusted for production cost savings. Replace
// `VEIL_ROUTER_CANDIDATES[*].examples` with real (query embedding → quality
// achieved) rows from your own eval logs before relying on this.

import { Router, type RouterCandidate } from '@metaharness/router';

/**
 * A 4-axis feature embedding for a harness query (each axis 0..1):
 *   [0] threatModeling    — "is this attack in scope / what does VEIL defend"-shaped
 *   [1] complianceReview  — "does this stay compliant / not jamming"-shaped
 *   [2] optimizerTuning   — "tune passes/bits / re-run the optimizer"-shaped
 *   [3] docWriting        — "write/update the research bundle or ADR"-shaped
 * A caller with a real embedding model should project onto that model's
 * dimensionality instead — the router only needs consistent vectors.
 */
export type VeilTaskEmbedding = readonly [number, number, number, number];

export const VEIL_ROUTER_CANDIDATES: RouterCandidate[] = [
  {
    id: 'cheap-tier',
    costPerMTok: 1,
    examples: [
      { embedding: [1, 0, 0, 0], quality: 0.88 }, // threat-model Q&A: cheap tier is fine
      { embedding: [0, 0, 0, 1], quality: 0.85 }, // doc writing: cheap tier is fine
      { embedding: [0, 1, 0, 0], quality: 0.55 }, // compliance review: cheap tier is weak
      { embedding: [0, 0, 1, 0], quality: 0.5 }, // optimizer tuning: cheap tier is weak
    ],
  },
  {
    id: 'frontier-tier',
    costPerMTok: 15,
    examples: [
      { embedding: [1, 0, 0, 0], quality: 0.95 },
      { embedding: [0, 0, 0, 1], quality: 0.93 },
      { embedding: [0, 1, 0, 0], quality: 0.93 }, // compliance review: frontier tier needed
      { embedding: [0, 0, 1, 0], quality: 0.92 }, // optimizer tuning: frontier tier needed
    ],
  },
];

/**
 * Cost-optimal router for the harness's four query shapes above. `qualityBar`
 * of 0.8: return the cheapest candidate predicted to clear 80% quality, or the
 * best-predicted candidate if none do. k=1 because each candidate has only 4
 * orthogonal one-hot examples (see the SAR harness note on why the default k=5
 * would collapse every query to the same prediction here).
 */
export const veilTaskRouter = new Router({
  qualityBar: 0.8,
  candidates: VEIL_ROUTER_CANDIDATES,
  k: 1,
});

/** Route one query embedding to the cost-optimal model tier. */
export function routeVeilQuery(queryEmbedding: VeilTaskEmbedding) {
  return veilTaskRouter.route([...queryEmbedding]);
}
