import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withToolBounds } from '../src/mcp-server.js';

const PACKAGE = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('tool calls fail closed on cancellation and timeout', async () => {
  const controller = new AbortController();
  const cancelled = withToolBounds(new Promise(() => {}), {
    signal: controller.signal,
    timeoutMs: 1_000,
  });
  controller.abort();
  await assert.rejects(cancelled, (error) => error.rpcCode === -32800);

  await assert.rejects(
    withToolBounds(new Promise(() => {}), { timeoutMs: 10 }),
    (error) => error.rpcCode === -32001,
  );
});

test('MCP server initializes and lists the bounded tool surface', async () => {
  const child = spawn(process.execPath, ['bin/cli.js', 'mcp', 'start'], {
    cwd: PACKAGE,
    env: {
      ...process.env,
      METAHARNESS_KERNEL_BACKEND: 'wasm',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (chunk) => { stdout += chunk; });
  child.stderr.on('data', (chunk) => { stderr += chunk; });

  child.stdin.write(`${'x'.repeat((256 * 1024) + 1)}\n`);
  child.stdin.write('null\n');
  child.stdin.write(`${JSON.stringify({ jsonrpc: '1.0', id: 0, method: 'ping' })}\n`);
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: {}, method: 'ping' })}\n`);
  child.stdin.write(`${JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'test', version: '0' },
    },
  })}\n`);
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} })}\n`);
  child.stdin.end();

  const code = await new Promise((resolveCode, reject) => {
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`MCP timeout\n${stderr}`));
    }, 20_000);
    child.once('error', reject);
    child.once('close', (value) => {
      clearTimeout(timer);
      resolveCode(value);
    });
  });
  assert.equal(code, 0, stderr);
  const messages = stdout.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  assert.equal(messages.filter(({ error }) => error?.code === -32600).length, 3);
  assert.equal(messages.find(({ id }) => id === 1).result.serverInfo.name, 'homecore');
  const tools = messages.find(({ id }) => id === 2).result.tools;
  assert.deepEqual(
    tools.map(({ name }) => name),
    [
      'homecore_guidance',
      'homecore_wasm_status',
      'homecore_doctor',
      'homecore_memory_search',
    ],
  );
  assert.equal(tools.some(({ name }) => name === 'homecore_verify'), false);
  assert.match(stderr, /kernel wasm/);
  assert.match(stderr, /oversized JSON-RPC line dropped/);
});
