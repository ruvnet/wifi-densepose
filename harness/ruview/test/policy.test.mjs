import test from 'node:test';
import assert from 'node:assert/strict';
import { authorizeTool, validateArguments } from '../src/policy.js';
import { runTool } from '../src/tools.js';

test('schema validation rejects unknown and mistyped arguments', async () => {
  const result = await runTool('ruview_onboard', { path: 7, injected: true });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'invalid_arguments');
  assert.ok(result.errors.some((error) => error.includes('injected')));
});

test('MCP workspace writes require confirmation and an explicit grant', () => {
  assert.equal(authorizeTool('ruview_calibrate', {}, { source: 'mcp', grants: [] }).reason, 'not_confirmed');
  assert.equal(authorizeTool('ruview_calibrate', { confirm: true }, { source: 'mcp', grants: [] }).reason, 'authority_denied');
  assert.equal(authorizeTool('ruview_calibrate', { confirm: true }, { source: 'mcp', grants: ['workspace-write'] }).ok, true);
});

test('read-only tools remain available with no mutation grants', () => {
  assert.equal(authorizeTool('ruview_claim_check', { text: 'safe' }, { source: 'mcp', grants: [] }).ok, true);
  assert.deepEqual(validateArguments({ type: 'object', properties: {} }, {}), []);
});
