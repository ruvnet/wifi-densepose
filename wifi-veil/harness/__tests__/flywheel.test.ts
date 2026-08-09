// SPDX-License-Identifier: MIT
// Verifies the SYNTHETIC flywheel demo wires end-to-end: a non-empty lift curve
// and a replay bundle that verifies independently. Does NOT assert any real
// self-improvement — the proposer/evaluator are deterministic stand-ins.

import { describe, it, expect } from 'vitest';
import { runVeilFlywheelDemo, verifyVeilFlywheelDemo } from '../src/flywheel.js';

describe('wifi-veil-harness — flywheel (SYNTHETIC)', () => {
  it('produces a non-empty lift curve', async () => {
    const result = await runVeilFlywheelDemo(3);
    expect(result.liftCurve.length).toBeGreaterThan(0);
    expect(result.generationsRun).toBeGreaterThan(0);
  });

  it('produces an independently verifiable replay bundle', async () => {
    const result = await runVeilFlywheelDemo(3);
    const verdict = verifyVeilFlywheelDemo(result);
    expect(verdict.pass).toBe(true);
  });

  it('stamps the run as SYNTHETIC provenance', async () => {
    const result = await runVeilFlywheelDemo(2);
    expect(result.dataSource).toBe('SYNTHETIC');
  });
});
