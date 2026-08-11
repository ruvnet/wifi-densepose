// SPDX-License-Identifier: MIT
// A real smoke test for wifi-densepose-privshield-harness: it boots the actual
// kernel + host adapter the harness depends on, so `npm test` fails loudly if
// @metaharness/kernel or @metaharness/host-claude-code is missing, broken, or
// version-skewed. Fastest signal that `npm install` produced a runnable harness.

import { describe, it, expect } from 'vitest';
import { loadKernel } from '@metaharness/kernel';
import adapter from '@metaharness/host-claude-code';
import { run } from '../bin/cli.js';

describe('wifi-densepose-privshield-harness — install smoke test', () => {
  it('loads the kernel and reports a version + a known backend', async () => {
    const kernel = await loadKernel();
    const info = kernel.kernelInfo();
    expect(typeof info.version).toBe('string');
    expect(info.version.length).toBeGreaterThan(0);
    expect(['native', 'wasm', 'js']).toContain(kernel.backend);
  });

  it('resolves the host adapter with a name', () => {
    expect(typeof adapter.name).toBe('string');
    expect(adapter.name.length).toBeGreaterThan(0);
  });

  it('the CLI doctor command succeeds (exit 0)', async () => {
    const code = await run(['doctor']);
    expect(code).toBe(0);
  });

  it('an unknown CLI command exits non-zero', async () => {
    const code = await run(['definitely-not-a-command']);
    expect(code).not.toBe(0);
  });
});
