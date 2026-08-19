// SPDX-License-Identifier: MIT
import { describe, it, expect } from 'vitest';
import { sarTaskRouter, routeSarQuery, SAR_ROUTER_CANDIDATES } from '../src/router.js';

describe('sarTaskRouter — @metaharness/router wiring', () => {
  it('has both a cheap and a frontier candidate', () => {
    const ids = SAR_ROUTER_CANDIDATES.map((c) => c.id);
    expect(ids).toContain('cheap-tier');
    expect(ids).toContain('frontier-tier');
  });

  it('routes a physics-explanation-shaped query to the cheap tier', () => {
    const pick = routeSarQuery([1, 0, 0, 0]);
    expect(pick.id).toBe('cheap-tier');
    expect(pick.metBar).toBe(true);
  });

  it('routes a code-review-shaped query to the frontier tier (cheap tier misses the quality bar)', () => {
    const pick = routeSarQuery([0, 1, 0, 0]);
    expect(pick.id).toBe('frontier-tier');
  });

  it('always returns a candidate with a nonnegative predicted quality and a positive cost', () => {
    for (const q of [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] as const) {
      const pick = routeSarQuery(q);
      expect(pick.predictedQuality).toBeGreaterThanOrEqual(0);
      expect(pick.costPerMTok).toBeGreaterThan(0);
    }
  });

  it('sarTaskRouter.route and routeSarQuery agree (same underlying router)', () => {
    const a = sarTaskRouter.route([1, 0, 0, 0]);
    const b = routeSarQuery([1, 0, 0, 0]);
    expect(a).toEqual(b);
  });
});
