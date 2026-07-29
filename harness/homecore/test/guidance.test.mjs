import test from 'node:test';
import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getGuidance, listGuidanceTopics } from '../src/guidance.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

test('lists stable Homecore guidance topics', () => {
  const topics = listGuidanceTopics().map(({ topic }) => topic);
  assert.deepEqual(topics, [
    'overview',
    'core',
    'server',
    'api',
    'plugins',
    'integrations',
    'migration',
    'voice',
    'testing',
  ]);
});

test('finds Wasmtime plugin guidance with verified local citations', () => {
  const output = getGuidance(
    { topic: 'plugins', query: 'Wasmtime signatures', limit: 3 },
    { repoRoot: REPO },
  );
  assert.equal(output.ok, true);
  assert.equal(output.sourceCheck.verified, true);
  assert.equal(output.capabilities[0].id, 'wasm-plugins');
  assert.match(output.capabilities[0].summary, /WebAssembly/);
  assert.ok(output.recommendedCommands.some((command) => command.includes('--features wasmtime')));
});

test('keeps Home Assistant parity limitations explicit', () => {
  const output = getGuidance({ topic: 'api', query: 'parity ecosystem' });
  const api = output.capabilities.find(({ id }) => id === 'ha-core-api');
  assert.ok(api);
  assert.ok(api.limitations.some((item) => item.includes('not parity')));
});

test('rejects unbounded or unknown guidance input', () => {
  assert.throws(() => getGuidance({ topic: 'unknown' }), /unsupported/);
  assert.throws(() => getGuidance({ query: 'x' }), /2\.\.500/);
  assert.throws(() => getGuidance({ limit: 21 }), /between 1 and 20/);
});
