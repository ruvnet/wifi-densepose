#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { verifyReplayBundle } from '@metaharness/flywheel';
import { gateFingerprint, ruviewPromotionRule } from './gate.mjs';
import { createHonestNullReplay } from './fixture.mjs';

const args = process.argv.slice(2);
if (args.includes('--self-test')) {
  const genome = JSON.parse(readFileSync(new URL('./genome.json', import.meta.url), 'utf8'));
  const result = await createHonestNullReplay(genome);
  const verdict = verifyReplayBundle(result.replayBundle, {
    pinnedGateFingerprint: gateFingerprint(),
    promotionRule: ruviewPromotionRule,
  });
  const ok = verdict.pass && result.replayBundle.verified_improvements === 0;
  console.log(JSON.stringify({ ok, honestNull: true, gateFingerprint: gateFingerprint(), verdict }, null, 2));
  process.exit(ok ? 0 : 1);
}
const index = args.indexOf('--bundle');
if (index < 0 || !args[index + 1]) {
  console.error('Usage: node flywheel/replay.mjs --bundle <replay.json> [--pinned-gate <sha256>]');
  process.exit(2);
}
const bundle = JSON.parse(readFileSync(args[index + 1], 'utf8'));
const pinIndex = args.indexOf('--pinned-gate');
const verdict = verifyReplayBundle(bundle, {
  pinnedGateFingerprint: pinIndex >= 0 ? args[pinIndex + 1] : gateFingerprint(),
  promotionRule: ruviewPromotionRule,
});
console.log(JSON.stringify(verdict, null, 2));
process.exit(verdict.pass ? 0 : 1);
