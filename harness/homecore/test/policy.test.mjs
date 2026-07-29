import test from 'node:test';
import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { authorizeTool, validateArguments } from '../src/policy.js';
import { runTool } from '../src/tools.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

test('schemas reject additional and wrong-typed arguments', () => {
  const schema = {
    type: 'object',
    additionalProperties: false,
    required: ['query'],
    properties: {
      query: { type: 'string', minLength: 2 },
      limit: { type: 'integer', minimum: 1, maximum: 5 },
    },
  };
  assert.deepEqual(validateArguments(schema, { query: 'ok', limit: 2 }), []);
  assert.ok(validateArguments(schema, { query: 'x', extra: true }).length >= 2);
});

test('MCP cannot invoke the CLI-only verification tool', () => {
  assert.equal(
    authorizeTool('homecore_verify', {}, { source: 'mcp' }).reason,
    'mcp_not_exposed',
  );
});

test('read tools need no mutation grant', async () => {
  const output = await runTool(
    'homecore_guidance',
    { topic: 'migration', query: 'schema version', repo: REPO },
    { source: 'mcp', grants: [], trustedRoot: REPO },
  );
  assert.equal(output.ok, true);
  assert.equal(output.sourceCheck.verified, true);
});

test('verification uses fixed cargo arguments and a trusted root', async () => {
  const calls = [];
  const runner = async (command, args, options) => {
    calls.push({ command, args, cwd: options.cwd });
    return { code: 0, stdout: 'ok', stderr: '', truncated: false };
  };
  const output = await runTool(
    'homecore_verify',
    { profile: 'wasm', repo: REPO },
    { source: 'cli', runner },
  );
  assert.equal(output.ok, true);
  assert.equal(calls.length, 2);
  assert.ok(calls.every(({ command }) => command === 'cargo'));
  assert.ok(calls.every(({ cwd }) => cwd === REPO));
  assert.ok(calls.every(({ args }) => args.includes('wasmtime')));
});

test('MCP repository access is anchored at server startup', async () => {
  const unanchored = await runTool(
    'homecore_guidance',
    { topic: 'core', repo: REPO },
    { source: 'mcp', grants: [] },
  );
  assert.equal(unanchored.ok, false);
  assert.match(unanchored.message, /trusted root configured at server startup/);

  const differentRoot = await runTool(
    'homecore_guidance',
    { topic: 'core', repo: resolve(REPO, 'v2') },
    { source: 'mcp', grants: [], trustedRoot: REPO },
  );
  assert.equal(differentRoot.ok, false);
  assert.match(differentRoot.message, /does not match the configured trusted root/);
});
