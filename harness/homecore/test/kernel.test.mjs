import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getKernelStatus } from '../src/kernel.js';

const PACKAGE = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('loads and uses the packaged WASM kernel', async () => {
  const status = await getKernelStatus({ strict: true });
  assert.equal(status.ok, true);
  assert.equal(status.resolvedBackend, 'wasm');
  assert.equal(status.mcpValidation, null);
  assert.equal(status.info.target, 'wasm32-unknown-unknown');
  assert.match(status.mcpSpec.command[2], /^homecore@\d+\.\d+\.\d+$/);
  assert.ok(!status.mcpSpec.command.includes('homecore@latest'));
});

test('concurrent kernel loads share initialization and restore the backend environment', async () => {
  const previous = process.env.METAHARNESS_KERNEL_BACKEND;
  delete process.env.METAHARNESS_KERNEL_BACKEND;
  try {
    const module = await import(`../src/kernel.js?concurrency=${Date.now()}`);
    const statuses = await Promise.all(
      Array.from({ length: 8 }, () => module.getKernelStatus({ strict: true })),
    );
    assert.ok(statuses.every(({ ok, resolvedBackend }) => ok && resolvedBackend === 'wasm'));
    assert.equal(process.env.METAHARNESS_KERNEL_BACKEND, undefined);
  } finally {
    if (previous === undefined) delete process.env.METAHARNESS_KERNEL_BACKEND;
    else process.env.METAHARNESS_KERNEL_BACKEND = previous;
  }
});

test('packaged MCP templates invoke the installed binary without a floating tag', () => {
  const pkg = JSON.parse(readFileSync(resolve(PACKAGE, 'package.json'), 'utf8'));
  assert.ok(pkg.files.includes('.claude/settings.json'));
  assert.ok(pkg.files.includes('.claude/skills/'));
  assert.ok(!pkg.files.includes('.claude/'));
  for (const path of ['.codex/config.toml', '.claude/settings.json', '.mcp/servers.json']) {
    const config = readFileSync(resolve(PACKAGE, path), 'utf8');
    assert.ok(!config.includes('@latest'), path);
    assert.match(config, /homecore/, path);
  }
  const claude = JSON.parse(readFileSync(resolve(PACKAGE, '.claude/settings.json'), 'utf8'));
  assert.ok(claude.permissions.deny.includes('Read(./.env)'));
  assert.ok(claude.permissions.deny.includes('Read(./.env.*)'));
});

test('MCP policy declares bounded least-authority defaults', () => {
  const policy = JSON.parse(readFileSync(resolve(PACKAGE, '.harness/mcp-policy.json'), 'utf8'));
  assert.equal(policy.defaultDeny, true);
  assert.equal(policy.requireApprovalForDangerous, true);
  assert.ok(policy.toolTimeoutMs > 0);
  assert.ok(policy.maxToolCallsPerTurn > 0);
  assert.deepEqual(policy.cliOnlyTools, ['homecore_verify']);
});
