// SPDX-License-Identifier: MIT
//
// Cost-optimal task routing for the wifi-densepose-sar harness, via
// @metaharness/router (ADR-040's DRACO Phase-2 finding, productized): route
// each agent query to the cheapest model predicted to clear a quality bar,
// instead of sending every query to the frontier tier by default.
//
// HONESTY NOTE: the candidate `examples` below are SEED/ILLUSTRATIVE data —
// four hand-picked (embedding, quality) points per candidate, not measured
// eval-log observations. They exist so `sarTaskRouter` is a real, runnable
// k-NN router out of the box (see __tests__/router.test.ts), not so its
// routing decisions should be trusted for production cost savings. Replace
// `SAR_ROUTER_CANDIDATES[*].examples` with real (query embedding → quality
// achieved) rows from your own eval logs before relying on this for
// production routing — see @metaharness/router's README ("the more
// examples, the closer it gets to the per-query oracle").

import { Router, type RouterCandidate } from '@metaharness/router';

/**
 * A 4-axis feature embedding for a harness query, used only to pick a
 * nearby labelled example — NOT a real text embedding. Axes (each 0..1):
 *   [0] physicsExplanation — "explain the range-resolution formula"-shaped
 *   [1] codeReview         — "review this diff for correctness"-shaped
 *   [2] numericalDebugging — "why did this reconstruction test fail"-shaped
 *   [3] docWriting         — "write/update the tutorial"-shaped
 * A caller with a real embedding model should project onto whatever
 * dimensionality that model produces instead — the router only needs
 * consistent vectors, not these specific four axes.
 */
export type SarTaskEmbedding = readonly [number, number, number, number];

export const SAR_ROUTER_CANDIDATES: RouterCandidate[] = [
  {
    id: 'cheap-tier',
    costPerMTok: 1,
    examples: [
      { embedding: [1, 0, 0, 0], quality: 0.9 }, // physics explanations: cheap tier does fine
      { embedding: [0, 0, 0, 1], quality: 0.85 }, // doc writing: cheap tier does fine
      { embedding: [0, 1, 0, 0], quality: 0.55 }, // code review: cheap tier is weak
      { embedding: [0, 0, 1, 0], quality: 0.5 }, // numerical debugging: cheap tier is weak
    ],
  },
  {
    id: 'frontier-tier',
    costPerMTok: 15,
    examples: [
      { embedding: [1, 0, 0, 0], quality: 0.95 },
      { embedding: [0, 0, 0, 1], quality: 0.93 },
      { embedding: [0, 1, 0, 0], quality: 0.92 }, // code review: frontier tier needed
      { embedding: [0, 0, 1, 0], quality: 0.9 }, // numerical debugging: frontier tier needed
    ],
  },
];

/**
 * Cost-optimal router for the harness's four query shapes above. `qualityBar`
 * of 0.8 matches the router README's worked example: return the cheapest
 * candidate predicted to clear 80% quality, or the best-predicted candidate
 * if none do.
 */
export const sarTaskRouter = new Router({
  qualityBar: 0.8,
  candidates: SAR_ROUTER_CANDIDATES,
  // k=1: each candidate has only 4 (orthogonal, one-hot) examples covering
  // the 4 task axes. The router's default k=5 would average ALL of a
  // candidate's examples regardless of query similarity once a candidate
  // has <=5 examples, collapsing every query to the same prediction. k=1
  // makes it pick the single nearest labelled task type, which is what
  // this small illustrative dataset is shaped for.
  k: 1,
});

/** Route one query embedding to the cost-optimal model tier. */
export function routeSarQuery(queryEmbedding: SarTaskEmbedding) {
  return sarTaskRouter.route([...queryEmbedding]);
}
