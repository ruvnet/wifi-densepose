import test from 'node:test';
import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCodexArgs } from '../src/hosts/codex.js';
import { buildClaudeCodeArgs } from '../src/hosts/claude-code.js';
import { assertTrustedHomecoreRepo } from '../src/repo-trust.js';
import { runProcess, scrubEnvironment } from '../src/process-runner.js';
import { redact, REDACTED } from '../src/redact.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

test('Codex adapter is read-only by default and never emits bypasses', () => {
  const read = buildCodexArgs(REPO);
  const write = buildCodexArgs(REPO, { write: true });
  assert.equal(read[0], 'exec');
  assert.equal(read.at(-1), '-');
  assert.equal(read[read.indexOf('--sandbox') + 1], 'read-only');
  assert.equal(write[write.indexOf('--sandbox') + 1], 'workspace-write');
  assert.ok(read.includes('--ephemeral'));
  assert.ok(read.includes('--ignore-user-config'));
  assert.ok(!read.includes('--ignore-rules'));
  assert.ok(!write.includes('--ignore-rules'));
  assert.ok(!read.some((item) => item.includes('bypass')));
  assert.ok(!write.some((item) => item.includes('bypass')));
});

test('Claude Code adapter uses safe non-persistent plan mode', () => {
  const read = buildClaudeCodeArgs();
  const write = buildClaudeCodeArgs({ write: true });
  assert.ok(read.includes('-p'));
  assert.ok(read.includes('--safe-mode'));
  assert.ok(read.includes('--no-session-persistence'));
  assert.equal(read[read.indexOf('--permission-mode') + 1], 'plan');
  assert.equal(write[write.indexOf('--permission-mode') + 1], 'acceptEdits');
  assert.ok(!read.some((item) => item.includes('bypass')));
  assert.ok(!write.some((item) => item.includes('bypass')));
});

test('repository trust accepts only the exact marked root', () => {
  assert.equal(assertTrustedHomecoreRepo(REPO), REPO);
  assert.throws(
    () => assertTrustedHomecoreRepo(resolve(REPO, 'v2'), { trustedRoot: REPO }),
    /does not match/,
  );
});

test('child environments drop credentials and output redaction catches tokens', () => {
  const clean = scrubEnvironment({
    PATH: 'safe',
    HOMECORE_TOKENS: 'private-token',
    ANTHROPIC_API_KEY: 'sk-ant-1234567890123456',
  });
  assert.deepEqual(clean, { PATH: 'safe' });
  const authorization = redact('Authorization: Bearer abcdefghijklmnop');
  assert.ok(authorization.includes(REDACTED));
  assert.ok(!authorization.includes('abcdefghijklmnop'));
  assert.ok(!redact('token=abcdefghijklmnop').includes('abcdefghijklmnop'));
  assert.ok(!redact('{"password":"alpha bravo charlie"}').includes('alpha bravo charlie'));
  assert.ok(!redact('Cookie: session=abc; preference=dark').includes('session=abc'));
  assert.ok(!redact('Authorization: Digest username="admin", response="private"').includes('admin'));
  const privateKey = '-----BEGIN PRIVATE KEY-----\nYWxwaGEgYnJhdm8=\n-----END PRIVATE KEY-----';
  assert.equal(redact(privateKey), REDACTED);
  assert.equal(
    redact('test pairing::tests::valid_pairing_flow ... ok'),
    'test pairing::tests::valid_pairing_flow ... ok',
  );
});

test('process execution rejects disabled or unbounded timeouts', () => {
  assert.throws(
    () => runProcess(process.execPath, ['--version'], { timeoutMs: 0 }),
    /between 1000 and 1800000/,
  );
  assert.throws(
    () => runProcess(process.execPath, ['--version'], { timeoutMs: 1_800_001 }),
    /between 1000 and 1800000/,
  );
});

test('process timeout terminates the spawned process tree', async () => {
  const source = [
    "const { spawn } = require('node:child_process');",
    "const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });",
    'console.log(child.pid);',
    'setInterval(() => {}, 1000);',
  ].join('');
  let failure;
  await assert.rejects(
    runProcess(process.execPath, ['-e', source], { timeoutMs: 1_000 }),
    (error) => {
      failure = error;
      return error.timedOut === true;
    },
  );
  const descendantPid = Number(String(failure.stdout).trim());
  assert.ok(Number.isSafeInteger(descendantPid) && descendantPid > 0);
  await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  assert.throws(
    () => process.kill(descendantPid, 0),
    (error) => error?.code === 'ESRCH',
    `descendant process ${descendantPid} survived timeout`,
  );
});
