// SPDX-License-Identifier: MIT
// Verifies the cost-optimal router mechanism (not its illustrative data): cheap
// query shapes route to the cheap tier; hard shapes escalate to the frontier.

import { describe, it, expect } from 'vitest';
import { routeVeilQuery } from '../src/router.js';

describe('wifi-densepose-privshield-harness — router', () => {
  it('routes a threat-model query (cheap-tier-capable) to the cheap tier', () => {
    const pick = routeVeilQuery([1, 0, 0, 0]);
    expect(pick.id).toBe('cheap-tier');
    expect(pick.metBar).toBe(true);
  });

  it('escalates a compliance-review query to the frontier tier', () => {
    const pick = routeVeilQuery([0, 1, 0, 0]);
    expect(pick.id).toBe('frontier-tier');
  });

  it('escalates an optimizer-tuning query to the frontier tier', () => {
    const pick = routeVeilQuery([0, 0, 1, 0]);
    expect(pick.id).toBe('frontier-tier');
  });
});
