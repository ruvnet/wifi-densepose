import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getGuidance, GUIDANCE_TOPICS, listGuidanceTopics } from '../src/guidance.js';
import { runTool } from '../src/tools.js';

const REPO_ROOT = fileURLToPath(new URL('../../..', import.meta.url));

test('guidance topics are stable, unique, and described', () => {
  assert.equal(new Set(GUIDANCE_TOPICS).size, GUIDANCE_TOPICS.length);
  assert.ok(GUIDANCE_TOPICS.includes('homecore'));
  assert.deepEqual(
    listGuidanceTopics().map(({ topic }) => topic),
    GUIDANCE_TOPICS,
  );
  for (const item of listGuidanceTopics()) assert.ok(item.summary.length > 20);
});

test('overview returns a source-cited capability map and verifies local paths', () => {
  const result = getGuidance({}, { repoRoot: REPO_ROOT });
  assert.equal(result.ok, true, JSON.stringify(result.sourceCheck));
  assert.equal(result.topic, 'overview');
  assert.ok(result.capabilities.length >= 10);
  assert.equal(result.sourceCheck.mode, 'local-checkout');
  assert.equal(result.sourceCheck.verified, true);
  assert.deepEqual(result.sourceCheck.missing, []);
  assert.ok(result.entryPoints.includes('v2/Cargo.toml'));
  assert.ok(result.recommendedCommands.length > 0);
  for (const capability of result.capabilities) {
    assert.match(capability.id, /^[a-z0-9][a-z0-9-]+$/);
    assert.ok(capability.summary);
    assert.ok(capability.status);
    assert.ok(capability.evidence);
    assert.ok(capability.sources.length > 0);
    assert.ok(capability.validation.length > 0);
    assert.ok(capability.limitations.length > 0);
    for (const source of capability.sources) {
      assert.ok(!source.startsWith('/'));
      assert.ok(!source.includes('..'));
    }
  }
  result.capabilities[0].sources[0] = 'mutated';
  assert.notEqual(getGuidance({}, { repoRoot: REPO_ROOT }).capabilities[0].sources[0], 'mutated');
});

test('homecore guidance exposes requested capabilities and honest boundaries', () => {
  const result = getGuidance({ topic: 'homecore' }, { repoRoot: REPO_ROOT });
  const ids = new Set(result.capabilities.map(({ id }) => id));
  for (const id of [
    'homecore-runtime-restore',
    'homecore-plugins',
    'homecore-ha-api',
    'homecore-hap',
    'homecore-migration',
    'homecore-voice',
  ]) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
  assert.equal(result.capabilities.find(({ id }) => id === 'homecore-plugins').status, 'feature-gated');
  assert.equal(result.capabilities.find(({ id }) => id === 'homecore-voice').status, 'provider-required');
  assert.match(
    result.capabilities.find(({ id }) => id === 'homecore-ha-api').limitations.join(' '),
    /not parity/i,
  );
});

test('query ranks the matching capability and searches reviewed knowledge', () => {
  const result = getGuidance(
    { topic: 'homecore', query: 'Wasmtime plugin', limit: 3 },
    { repoRoot: REPO_ROOT },
  );
  assert.equal(result.ok, true);
  assert.equal(result.capabilities[0].id, 'homecore-plugins');
  assert.ok(result.capabilities.length <= 3);
  assert.ok(Array.isArray(result.relatedKnowledge));
  for (const record of result.relatedKnowledge) {
    assert.match(record.citation, /:\d+$/);
    assert.equal(record.reviewed, true);
  }
  const shared = getGuidance({ query: 'guidance' }, { repoRoot: REPO_ROOT });
  assert.ok(shared.relatedKnowledge.some(({ id }) => id === 'guidance-entrypoint'));
});

test('packaged guidance is explicit when no checkout is available', () => {
  const result = getGuidance({ topic: 'architecture', limit: 1 });
  assert.equal(result.ok, true);
  assert.equal(result.sourceCheck.mode, 'packaged-catalog');
  assert.equal(result.sourceCheck.verified, false);
  assert.match(result.sourceCheck.note, /not checked/i);
});

test('local source drift fails closed', () => {
  const empty = mkdtempSync(join(tmpdir(), 'ruview-guidance-'));
  try {
    const result = getGuidance({ topic: 'homecore', limit: 1 }, { repoRoot: empty });
    assert.equal(result.ok, false);
    assert.equal(result.sourceCheck.verified, false);
    assert.ok(result.sourceCheck.missing.length > 0);
  } finally {
    rmSync(empty, { recursive: true, force: true });
  }
});

test('guidance direct API and MCP schema reject malformed input', async () => {
  assert.throws(() => getGuidance([]), /input must be an object/);
  assert.throws(() => getGuidance({ query: {} }), /query must be a string/);
  assert.throws(() => getGuidance({}, { repoRoot: 7 }), /repoRoot/);
  assert.throws(() => getGuidance({ topic: 'unknown' }), /unsupported guidance topic/);
  assert.throws(() => getGuidance({ query: 'x' }), /2\.\.500/);
  assert.throws(() => getGuidance({ limit: 21 }), /between 1 and 20/);

  const bad = await runTool('ruview_guidance', { topic: 'homecore', injected: true });
  assert.equal(bad.ok, false);
  assert.equal(bad.reason, 'invalid_arguments');
  const short = await runTool('ruview_guidance', { query: 'x' });
  assert.equal(short.ok, false);
  assert.equal(short.reason, 'invalid_arguments');
});
