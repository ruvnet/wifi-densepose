// SPDX-License-Identifier: MIT
import { describe, it, expect } from 'vitest';
import { runSarFlywheelDemo, verifySarFlywheelDemo, SAR_ROOT_POLICY } from '../src/flywheel.js';

describe('runSarFlywheelDemo — @metaharness/flywheel wiring (SYNTHETIC data)', () => {
  it('runs the configured number of generations and stamps SYNTHETIC provenance', async () => {
    const result = await runSarFlywheelDemo(3);
    expect(result.generationsRun).toBeGreaterThan(0);
    expect(result.generationsRun).toBeLessThanOrEqual(3);
    expect(result.replayBundle.data_source).toBe('SYNTHETIC');
  });

  it('produces a non-empty lift curve rooted at the initial policy', async () => {
    const result = await runSarFlywheelDemo(3);
    expect(result.liftCurve.length).toBeGreaterThan(0);
    expect(result.liftCurve[0].generation).toBe(0);
  });

  it('the final policy still carries every root lever', async () => {
    const result = await runSarFlywheelDemo(2);
    for (const key of Object.keys(SAR_ROOT_POLICY)) {
      expect(result.finalPolicy).toHaveProperty(key);
    }
  });

  it('the replay bundle independently verifies (no trust in the producer)', async () => {
    const result = await runSarFlywheelDemo(3);
    const verdict = verifySarFlywheelDemo(result);
    expect(verdict.pass).toBe(true);
  });

  it('is deterministic in structure across repeated runs (same generations requested)', async () => {
    const a = await runSarFlywheelDemo(2);
    const b = await runSarFlywheelDemo(2);
    expect(a.generationsRun).toBe(b.generationsRun);
  });
});
