import { makeSigner, runFlywheelGenerations } from '@metaharness/flywheel';
import { evaluateGenome, loadEvaluation, ruviewPromotionRule } from './gate.mjs';

export async function createHonestNullReplay(genome) {
  const suites = loadEvaluation();
  return runFlywheelGenerations({
    rootPolicy: genome.surfaces,
    proposer: async (base, target) => base.policy[target],
    evaluator: async (policy, suite) => evaluateGenome({ surfaces: policy }, suite.items),
    promotionRule: ruviewPromotionRule,
    holdout: { id: 'ruview-holdout-v1', items: suites.holdout },
    anchor: { id: 'ruview-anchor-v1', items: suites.anchor },
    mutationTargets: ['planner'],
    maxGenerations: 1,
    signer: makeSigner(),
    now: (generation) => `fixture-generation-${generation}`,
    dataSource: 'SYNTHETIC',
    rootId: 'ruview-gen0',
  });
}
