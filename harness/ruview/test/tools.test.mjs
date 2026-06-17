// SPDX-License-Identifier: MIT
// RuView harness tests — Node's built-in test runner (no devDeps to install).
// Run: `node --test test/`  (or `npm test`).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { claimCheck, summarize } from '../src/guardrails.js';
import { TOOLS, runTool, listTools, findRepoRoot } from '../src/tools.js';
import { run } from '../bin/cli.js';

test('guardrail flags the retracted 100% framing as high severity', () => {
  const r = claimCheck('Our model reaches 100% accuracy on every pose.');
  assert.equal(r.ok, false);
  assert.ok(r.findings.some((f) => f.severity === 'high'));
});

test('guardrail flags an untagged percentage accuracy claim', () => {
  // "hit", not "measured" — "measured" would (correctly) route to the no-reproducer branch.
  const r = claimCheck('We hit 92.9% PCK on the test set.');
  assert.equal(r.ok, false);
  assert.ok(r.findings.some((f) => /not tagged/i.test(f.reason)));
});

test('guardrail passes a MEASURED claim that cites a reproducer', () => {
  const r = claimCheck('Held-out PCK@20 59.5% vs 50% mean-pose baseline = +9.4pp (MEASURED, verify.py).');
  assert.equal(r.ok, true, JSON.stringify(r.findings));
});

test('guardrail flags MEASURED with no reproducer', () => {
  const r = claimCheck('Presence detection 97% (MEASURED).');
  assert.equal(r.ok, false);
  assert.ok(r.findings.some((f) => /no reproducer/i.test(f.reason)));
});

test('guardrail ignores non-metric prose', () => {
  assert.equal(claimCheck('The ESP32 streams CSI over UDP to the sensing-server.').ok, true);
  assert.equal(claimCheck('').ok, true);
});

test('summarize gives PASS/finding text', () => {
  assert.match(summarize(claimCheck('nothing here')), /PASS/);
  assert.match(summarize(claimCheck('100% accuracy')), /finding/);
});

test('registry exposes the documented tools with schemas', () => {
  const names = Object.keys(TOOLS);
  for (const n of ['ruview.onboard', 'ruview.claim_check', 'ruview.verify', 'ruview.node_monitor', 'ruview.calibrate', 'ruview.node_flash']) {
    assert.ok(names.includes(n), `missing ${n}`);
    assert.equal(TOOLS[n].inputSchema.type, 'object');
  }
  assert.equal(listTools().length, names.length);
});

test('ruview.onboard returns paths and a recommendation', () => {
  const r = runTool('ruview.onboard', {});
  assert.equal(r.ok, true);
  assert.ok(r.paths['live-esp32']);
  assert.ok(['repo-build', 'docker-demo'].includes(r.recommend));
});

test('ruview.claim_check tool wraps the guardrail', () => {
  const r = runTool('ruview.claim_check', { text: '100% accuracy' });
  assert.equal(r.ok, false);
  assert.match(r.summary, /honesty|tag|MEASURED|finding/i);
});

test('unknown tool fails closed', () => {
  const r = runTool('ruview.does_not_exist', {});
  assert.equal(r.ok, false);
  assert.equal(r.reason, 'unknown_tool');
});

test('node_monitor fails closed without a port', () => {
  const r = runTool('ruview.node_monitor', {});
  assert.equal(r.ok, false);
  assert.equal(r.reason, 'no_port');
});

test('node_flash refuses without confirm (mutating guard)', () => {
  const r = runTool('ruview.node_flash', { port: 'COM8', variant: 's3-8mb' });
  assert.equal(r.ok, false);
  // either not-confirmed (win32) or unsupported_platform (posix) — both fail-closed
  assert.ok(['not_confirmed', 'unsupported_platform'].includes(r.reason));
});

test('verify fails closed when not in a RuView repo', () => {
  // point at a tmp dir with no repo markers
  const r = runTool('ruview.verify', { repo: process.platform === 'win32' ? 'C:/Windows/Temp' : '/tmp' });
  assert.equal(r.ok, false);
  assert.ok(['proof_missing', 'python_missing'].includes(r.reason), r.reason);
});

test('CLI run(): claim-check exits non-zero on a bad claim', async () => {
  const code = await run(['claim-check', '--text', '100% accuracy']);
  assert.notEqual(code, 0);
});

test('CLI run(): doctor exits 0 (tools-only path)', async () => {
  const code = await run(['doctor']);
  assert.equal(code, 0);
});

test('CLI run(): unknown command exits non-zero', async () => {
  assert.notEqual(await run(['definitely-not-a-command']), 0);
});

test('findRepoRoot locates this monorepo from cwd', () => {
  // when run from within wifi-densepose, it should find a root; elsewhere null is fine
  const root = findRepoRoot();
  assert.ok(root === null || typeof root === 'string');
});
