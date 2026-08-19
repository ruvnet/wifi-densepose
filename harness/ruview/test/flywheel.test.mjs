import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { evaluateGenome, gateFingerprint, loadEvaluation, ruviewPromotionRule } from '../flywheel/gate.mjs';
import { verifyReplayBundle } from '@metaharness/flywheel';
import { createHonestNullReplay } from '../flywheel/fixture.mjs';

const genome = JSON.parse(readFileSync(new URL('../flywheel/genome.json', import.meta.url), 'utf8'));

test('frozen anchor and holdout describe the committed genome', () => {
  const suites = loadEvaluation();
  assert.equal(evaluateGenome(genome, suites.anchor).regressed, false);
  assert.match(gateFingerprint(), /^[a-f0-9]{64}$/);
});

test('promotion rule requires strict lift and frozen-anchor retention', () => {
  const score = { primary: 0.8, noopRate: 0.2, costPerWin: 1, regressed: false };
  const verified = {
    securityPassed: true, legacyTestsPassed: true, provenanceVerified: true, humanApproved: true,
    blockedActions: 0, secretExposures: 0,
  };
  assert.equal(ruviewPromotionRule({ ...verified, baseline: score, candidate: { ...score, primary: 0.9 }, anchor: { baseline: 1, candidate: 1 } }).promote, true);
  assert.equal(ruviewPromotionRule({ ...verified, baseline: score, candidate: { ...score, primary: 0.9 }, anchor: { baseline: 1, candidate: 0.9 } }).promote, false);
  assert.equal(ruviewPromotionRule({ ...verified, humanApproved: false, baseline: score, candidate: { ...score, primary: 0.9 }, anchor: { baseline: 1, candidate: 1 } }).promote, false);
  assert.equal(ruviewPromotionRule({ ...verified, secretExposures: 1, baseline: score, candidate: { ...score, primary: 0.9 }, anchor: { baseline: 1, candidate: 1 } }).promote, false);
});

test('honest-null replay verifies and tampering fails', async () => {
  const result = await createHonestNullReplay(genome);
  const options = { pinnedGateFingerprint: gateFingerprint(), promotionRule: ruviewPromotionRule };
  assert.equal(verifyReplayBundle(result.replayBundle, options).pass, true);
  assert.equal(result.replayBundle.verified_improvements, 0);
  const tampered = structuredClone(result.replayBundle);
  tampered.chain[0].receipt.signature = `${tampered.chain[0].receipt.signature.slice(0, -2)}aa`;
  assert.equal(verifyReplayBundle(tampered, options).pass, false);
});
