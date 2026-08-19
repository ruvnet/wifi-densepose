// SPDX-License-Identifier: MIT
import test from 'node:test';
import assert from 'node:assert/strict';
import { listCognitumSpaces, parseSpacesOutput } from '../src/spaces.js';
import { runTool } from '../src/tools.js';

function validResponse(kind = 'spaces') {
  return {
    object: 'list',
    kind,
    schemaVersion: '1.0',
    data: [{
      id: 'room-1', tenantId: 'tenant-1', workspaceId: '11111111-1111-7111-8111-111111111111', siteId: 'site-1', name: 'Room',
      buildingId: 'building-1', floorId: 'floor-1', kind, schemaVersion: '1.0',
      messageId: 'message-1', eventSequence: 1, version: 1, privacy: 'P2',
      confidence: 0.9, provenance: {}, attributes: {}, observedAt: '2026-08-19T00:00:00Z', expiresAt: null,
    }],
    nextCursor: null,
    boundary: {
      authoritativeState: 'HomeCore Edge',
      cloudRole: 'tenant-scoped semantic synchronization',
      excluded: ['raw_csi', 'cir', 'rf_tensors', 'recordings', 'pose_frames', 'vital_waveforms', 'identity_observations'],
    },
  };
}

test('Spaces adapter invokes OAuth-only CLI args in a scrubbed environment', async () => {
  const credentialPath = 'C:/private/ruview-credentials.json';
  const secretApiKey = ['cog', 'DO', 'NOT', 'FORWARD'].join('_');
  let observed;
  const result = await listCognitumSpaces(
    { credentials_path: credentialPath },
    {
      source: 'cli',
      binary: 'wifi-densepose-test-double',
      env: { PATH: 'test-path', COGNITUM_SPACES_API: secretApiKey, RUVIEW_CREDENTIALS_PATH: credentialPath },
      execute: async (command, args, options) => {
        observed = { command, args, options };
        return { stdout: JSON.stringify(validResponse()), stderr: '', code: 0 };
      },
    },
  );

  assert.equal(result.ok, true);
  assert.equal(result.authentication, 'oauth');
  assert.equal(result.count, 1);
  assert.equal(observed.command, 'wifi-densepose-test-double');
  assert.deepEqual(observed.args, [
    'spaces', '--json', '--base-url', 'https://api.cognitum.one', '--resource', 'spaces', '--limit', '50',
    '--credentials-path', credentialPath,
  ]);
  assert.ok(observed.options.envAllowlist.includes('RUVIEW_CREDENTIALS_PATH'));
  assert.ok(!observed.options.envAllowlist.includes('COGNITUM_SPACES_API'));
  assert.ok(!observed.args.join(' ').includes(secretApiKey));
});

test('MCP cannot select an arbitrary credential path even with a credential-use grant', async () => {
  const result = await runTool(
    'ruview_spaces_list',
    { credentials_path: 'C:/private/credentials.json' },
    { source: 'mcp', grants: ['credential-use'] },
  );
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'credentials_path_not_allowed');
});

test('MCP denies a Spaces read before touching local credentials or the network', async () => {
  const result = await runTool('ruview_spaces_list', {}, { source: 'mcp', grants: [] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'authority_denied');
  assert.equal(result.requiredGrant, 'credential-use');
});

test('metaharness rejects forbidden raw fields from a child process', () => {
  const response = validResponse();
  response.data[0].attributes.raw_csi = [1, 2, 3];
  assert.throws(() => parseSpacesOutput(JSON.stringify(response)), /forbidden raw field/i);
});

test('metaharness rejects incomplete privacy boundaries and invalid confidence', () => {
  const incomplete = validResponse();
  incomplete.boundary.excluded = ['raw_csi'];
  assert.throws(() => parseSpacesOutput(JSON.stringify(incomplete)), /incomplete edge privacy boundary/i);

  const invalid = validResponse();
  invalid.data[0].confidence = 2;
  assert.throws(() => parseSpacesOutput(JSON.stringify(invalid)), /invalid confidence/i);
});

test('versioned hierarchy, events, alerts, and cursor args stay OAuth-only', async () => {
  let observed;
  const response = validResponse('events');
  response.data[0].spaceId = 'room-1';
  response.data[0].eventType = 'occupancy.changed';
  response.data[0].buildingId = null;
  response.data[0].floorId = null;
  const result = await listCognitumSpaces(
    { resource: 'events', limit: 25, cursor: 'opaque-cursor' },
    {
      source: 'mcp',
      binary: 'wifi-densepose-test-double',
      env: { PATH: 'test-path', COGNITUM_SPACES_API: 'cog_never_forward' },
      execute: async (command, args, options) => {
        observed = { command, args, options };
        return { stdout: JSON.stringify(response), stderr: '', code: 0 };
      },
    },
  );
  assert.equal(result.ok, true);
  assert.equal(result.resource, 'events');
  assert.deepEqual(observed.args, [
    'spaces', '--json', '--base-url', 'https://api.cognitum.one', '--resource', 'events', '--limit', '25',
    '--cursor', 'opaque-cursor',
  ]);
  assert.ok(!observed.options.envAllowlist.includes('COGNITUM_SPACES_API'));
});

test('metaharness rejects raw aliases and malformed kind-specific records', () => {
  const raw = validResponse();
  raw.data[0].attributes.packet_capture = 'forbidden';
  assert.throws(() => parseSpacesOutput(JSON.stringify(raw), 'spaces'), /forbidden raw field/i);

  const entity = validResponse('entities');
  entity.data[0].spaceId = 'room-1';
  entity.data[0].entityType = 'person';
  entity.data[0].identityMode = 'named';
  assert.throws(() => parseSpacesOutput(JSON.stringify(entity), 'entities'), /entity privacy contract/i);

  const invalidWorkspace = validResponse();
  invalidWorkspace.data[0].workspaceId = 'workspace-1';
  assert.throws(() => parseSpacesOutput(JSON.stringify(invalidWorkspace), 'spaces'), /versioned spatial identity/i);

  const invalidTimestamp = validResponse();
  invalidTimestamp.data[0].observedAt = 'not-a-timestamp';
  assert.throws(() => parseSpacesOutput(JSON.stringify(invalidTimestamp), 'spaces'), /versioned spatial identity/i);
});

test('command failures redact API keys and JWT-shaped tokens', async () => {
  const secret = `cog_${'test-value-'.repeat(4)}`;
  const jwt = 'eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.signature-material';
  const result = await listCognitumSpaces({}, {
    source: 'cli',
    binary: 'wifi-densepose-test-double',
    env: { PATH: 'test-path', COGNITUM_SPACES_API: secret },
    execute: async () => { throw new Error(`failed token=${jwt} api_key=${secret}`); },
  });
  assert.equal(result.ok, false);
  assert.ok(!result.detail.includes(secret));
  assert.ok(!result.detail.includes(jwt));
  assert.match(result.detail, /REDACTED/);
});

test('credentialed calls never fall back to Cargo build scripts', async () => {
  let executed = false;
  const result = await listCognitumSpaces({}, {
    source: 'cli',
    cargo: 'cargo',
    repoRoot: 'C:/trusted/ruview',
    execute: async () => { executed = true; },
  });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'cli_missing');
  assert.equal(executed, false);
});
