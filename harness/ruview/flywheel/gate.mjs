// SPDX-License-Identifier: MIT
import { readFileSync } from 'node:fs';
import { gateFingerprint as fingerprintRule } from '@metaharness/flywheel';

export function evaluateGenome(genome, suite) {
  const failures = [];
  for (const item of suite) {
    const text = String(genome.surfaces?.[item.surface] || '').toLowerCase();
    for (const required of item.requires || []) {
      if (!text.includes(required.toLowerCase())) failures.push(`${item.id}:missing:${required}`);
    }
    for (const forbidden of item.forbids || []) {
      if (text.includes(forbidden.toLowerCase())) failures.push(`${item.id}:forbidden:${forbidden}`);
    }
  }
  return {
    primary: suite.length ? (suite.length - new Set(failures.map((f) => f.split(':')[0])).size) / suite.length : 0,
    noopRate: suite.length ? new Set(failures.map((f) => f.split(':')[0])).size / suite.length : 1,
    costPerWin: suite.length ? 1 / Math.max(0.01, suite.length - failures.length) : 100,
    regressed: failures.length > 0,
    failures,
  };
}

export function ruviewPromotionRule(evidence) {
  const reasons = [];
  if (!(evidence.candidate.primary > evidence.baseline.primary)) reasons.push('holdout did not strictly improve');
  if (evidence.candidate.regressed) reasons.push('candidate regressed');
  if (!(evidence.candidate.noopRate <= evidence.baseline.noopRate)) reasons.push('noop rate regressed');
  if (!(evidence.candidate.costPerWin <= evidence.baseline.costPerWin)) reasons.push('cost per win regressed');
  if (evidence.anchor && evidence.anchor.candidate < evidence.anchor.baseline) reasons.push('frozen anchor regressed');
  if (evidence.securityPassed !== true) reasons.push('security gate not verified');
  if (evidence.legacyTestsPassed !== true) reasons.push('legacy tests not verified');
  if (evidence.provenanceVerified !== true) reasons.push('provenance not verified');
  if (evidence.humanApproved !== true) reasons.push('maintainer approval missing');
  if ((evidence.blockedActions ?? 0) !== 0) reasons.push('blocked actions recorded');
  if ((evidence.secretExposures ?? 0) !== 0) reasons.push('secret exposure recorded');
  return { promote: reasons.length === 0, reasons };
}

export function gateFingerprint() {
  return fingerprintRule(ruviewPromotionRule);
}

export function loadEvaluation(path = new URL('./evaluations.json', import.meta.url)) {
  return JSON.parse(readFileSync(path, 'utf8'));
}
