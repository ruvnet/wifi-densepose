// SPDX-License-Identifier: MIT
//
// The wifi-densepose-sar harness's self-improvement loop, via
// @metaharness/flywheel: run -> measure -> mutate -> verify -> promote,
// with a frozen, conjunctive promotion gate and a signed, replayable lineage
// (see @metaharness/darwin's `npm run evolve` for the harness-wide mutation
// entry point this formalizes the promotion loop for).
//
// HONESTY NOTE (load-bearing): `runSarFlywheelDemo()` below wires the real
// @metaharness/flywheel API end-to-end, but its Proposer and Evaluator are
// SYNTHETIC stand-ins — a deterministic string mutation and a deterministic
// scoring function over that string, with NO model call and NO real
// benchmark suite. It exists to prove the wiring works (see
// __tests__/flywheel.test.ts: a non-empty lift curve, a verifiable replay
// bundle) and to give a `dataSource: 'SYNTHETIC'`-stamped demo, exactly as
// the flywheel's own design requires callers to be honest about data
// provenance. A LIVE run needs the operator to supply:
//   - a real Proposer: an actual model call that improves one policy lever
//     (e.g. the review-diff checklist, the architect's planning prompt);
//   - a real Evaluator: scores that policy against real coding-task
//     holdout/anchor suites (e.g. "did review-diff catch the seeded bug").
// Neither exists in this repo — wiring them is a live-API-key decision for
// whoever operates this harness, not something to fake here.

import {
  runFlywheelGenerations,
  meetsPromotionRule,
  makeSigner,
  verifyReplayBundle,
  type Policy,
  type PolicyGenome,
  type Proposer,
  type Evaluator,
  type Suite,
  type FlywheelResult,
} from '@metaharness/flywheel';

/** The gen-0 operating policy for the SAR harness's review agents. Opaque
 * string levers — the flywheel never interprets their meaning, only the
 * Evaluator does. */
export const SAR_ROOT_POLICY: Policy = {
  reviewDepth: 'standard-checklist',
  architectPlanning: 'single-pass',
};

/**
 * SYNTHETIC proposer: deterministically lengthens/varies the target lever's
 * value rather than calling a model. Stands in for a real model call that
 * would draft an improved lever value.
 */
const syntheticProposer: Proposer = async (base: PolicyGenome, target: string) => {
  const current = base.policy[target] ?? '';
  return `${current}+g${base.generation + 1}`;
};

/**
 * SYNTHETIC evaluator: scores a policy purely as a function of its own
 * string content (longer, more "refined"-looking levers score marginally
 * higher on quality and lower on no-op rate, both with a floor/ceiling) —
 * a deterministic stand-in for actually running the harness's agents
 * against a real coding-task suite. `noopRate` must move (not stay
 * constant) for anything to ever promote: the default gate's clause 2
 * requires it to strictly improve generation over generation (ADR-226's
 * "the executor policy is the part that mattered" finding, encoded as a
 * hard requirement) — a constant noopRate, even a "good" one, gates every
 * candidate out forever.
 */
const syntheticEvaluator: Evaluator = async (policy: Policy, _suite: Suite) => {
  const totalLength = Object.values(policy).reduce((s, v) => s + v.length, 0);
  const primary = Math.min(0.5 + totalLength / 200, 0.98);
  const noopRate = Math.max(0.3 - totalLength / 300, 0.02);
  return {
    primary,
    noopRate,
    costPerWin: 1 / primary,
    regressed: false,
  };
};

const SAR_HOLDOUT: Suite = {
  id: 'sar-harness-holdout-synthetic',
  items: ['seeded-review-task-1', 'seeded-review-task-2', 'seeded-review-task-3'],
};

const SAR_ANCHOR: Suite = {
  id: 'sar-harness-anchor-synthetic',
  items: ['frozen-regression-task-1'],
};

/**
 * Run a small, fully SYNTHETIC flywheel demo end-to-end: propose, evaluate,
 * gate, and (when it clears the gate) promote a few generations of mutated
 * policy, returning the real @metaharness/flywheel result — a genuine lift
 * curve and a signed, independently replayable bundle, just built from
 * synthetic (not live) evidence.
 */
export async function runSarFlywheelDemo(maxGenerations = 3): Promise<FlywheelResult> {
  return runFlywheelGenerations({
    rootPolicy: SAR_ROOT_POLICY,
    proposer: syntheticProposer,
    evaluator: syntheticEvaluator,
    promotionRule: meetsPromotionRule,
    holdout: SAR_HOLDOUT,
    anchor: SAR_ANCHOR,
    maxGenerations,
    signer: makeSigner(),
    dataSource: 'SYNTHETIC',
  });
}

/** Independently verify a flywheel demo's replay bundle (no trust in the producer). */
export function verifySarFlywheelDemo(result: FlywheelResult) {
  return verifyReplayBundle(result.replayBundle);
}
