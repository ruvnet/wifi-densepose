// SPDX-License-Identifier: MIT
//
// The wifi-densepose-privshield (VEIL) harness's self-improvement loop, via
// @metaharness/flywheel: run -> measure -> mutate -> verify -> promote, with a
// frozen, conjunctive promotion gate and a signed, replayable lineage.
//
// HONESTY NOTE (load-bearing): `runVeilFlywheelDemo()` wires the real
// @metaharness/flywheel API end-to-end, but its Proposer and Evaluator are
// SYNTHETIC stand-ins — a deterministic string mutation and a deterministic
// scoring function over that string, with NO model call and NO real benchmark.
// It proves the wiring works (see __tests__/flywheel.test.ts: a non-empty lift
// curve, a verifiable replay bundle) and gives a `dataSource: 'SYNTHETIC'`-
// stamped demo. A LIVE run needs the operator to supply:
//   - a real Proposer: a model call that improves one policy lever (e.g. the
//     compliance-review checklist, the threat-model triage prompt);
//   - a real Evaluator: scores that policy against real tasks (e.g. "did the
//     compliance reviewer catch a non-energy-preserving perturbation").
// Neither exists in this repo — wiring them is a live-API-key decision for the
// harness operator, not something to fake here.

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

/** The gen-0 operating policy for the VEIL harness's review agents. Opaque
 * string levers — the flywheel never interprets their meaning, only the
 * Evaluator does. */
export const VEIL_ROOT_POLICY: Policy = {
  complianceReview: 'energy-ratio-checklist',
  threatTriage: 'single-pass',
};

/** SYNTHETIC proposer: deterministically varies the target lever's value
 * rather than calling a model. */
const syntheticProposer: Proposer = async (base: PolicyGenome, target: string) => {
  const current = base.policy[target] ?? '';
  return `${current}+g${base.generation + 1}`;
};

/** SYNTHETIC evaluator: scores a policy purely as a function of its own string
 * content — a deterministic stand-in for running the harness's agents against a
 * real task suite. `noopRate` must move for anything to promote (the default
 * gate requires it to strictly improve generation over generation). */
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

const VEIL_HOLDOUT: Suite = {
  id: 'veil-harness-holdout-synthetic',
  items: ['seeded-compliance-task-1', 'seeded-threat-task-2', 'seeded-optimizer-task-3'],
};

const VEIL_ANCHOR: Suite = {
  id: 'veil-harness-anchor-synthetic',
  items: ['frozen-not-jamming-regression-1'],
};

/**
 * Run a small, fully SYNTHETIC flywheel demo end-to-end and return the real
 * @metaharness/flywheel result — a genuine lift curve and a signed,
 * independently replayable bundle, built from synthetic (not live) evidence.
 */
export async function runVeilFlywheelDemo(maxGenerations = 3): Promise<FlywheelResult> {
  return runFlywheelGenerations({
    rootPolicy: VEIL_ROOT_POLICY,
    proposer: syntheticProposer,
    evaluator: syntheticEvaluator,
    promotionRule: meetsPromotionRule,
    holdout: VEIL_HOLDOUT,
    anchor: VEIL_ANCHOR,
    maxGenerations,
    signer: makeSigner(),
    dataSource: 'SYNTHETIC',
  });
}

/** Independently verify a flywheel demo's replay bundle (no trust in the producer). */
export function verifyVeilFlywheelDemo(result: FlywheelResult) {
  return verifyReplayBundle(result.replayBundle);
}
