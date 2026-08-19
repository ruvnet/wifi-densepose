// SPDX-License-Identifier: MIT
// The VEIL guidance map is dependency-free (no @metaharness/* import), so this
// test runs even before `npm install` resolves the kernel. It guards the
// read-only capability map the MCP/CLI `guidance` surface exposes.

import { describe, it, expect } from 'vitest';
import { run, guidanceReport } from '../bin/cli.js';

describe('wifi-veil-harness — guidance', () => {
  it('returns a source-cited report for a known topic', () => {
    const r = guidanceReport('optimization');
    expect(r.ok).toBe(true);
    expect(r.summary.length).toBeGreaterThan(0);
    expect(r.sources.some((s: string) => s.includes('optimize.rs'))).toBe(true);
    expect(r.authority).toContain('read-only');
  });

  it('labels evidence as SYNTHETIC/L0', () => {
    const r = guidanceReport('experiment');
    expect(r.evidence).toContain('SYNTHETIC');
  });

  it('rejects an unknown topic and lists the valid ones', () => {
    const r = guidanceReport('not-a-topic');
    expect(r.ok).toBe(false);
    expect(r.topics).toContain('overview');
    expect(r.topics).toContain('compliance');
  });

  it('CLI `guidance --topic overview` exits 0; unknown topic exits non-zero', async () => {
    expect(await run(['guidance', '--topic', 'overview'])).toBe(0);
    expect(await run(['guidance', '--topic', 'nope'])).not.toBe(0);
  });
});
