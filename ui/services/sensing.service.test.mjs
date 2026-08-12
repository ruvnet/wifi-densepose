// Regression coverage for issue #1526.
//
// This executes the status-probe boundary with a stubbed fetch/localStorage.
// It verifies request authentication and UI source classification, but it is
// not browser or ESP32 hardware validation.

import { beforeEach, test } from 'node:test';
import assert from 'node:assert/strict';

const STORAGE_KEY = 'ruview-api-token';
const ORIGIN = 'http://ruview.test:3000';

let stored = { [STORAGE_KEY]: 'status-probe-token' };
let fetchCalls = [];
let fetchImpl;

globalThis.window = {
  location: {
    origin: ORIGIN,
    protocol: 'http:',
    host: 'ruview.test:3000',
    hostname: 'ruview.test',
    port: '3000',
  },
};
globalThis.localStorage = {
  getItem: (key) => stored[key] ?? null,
  setItem: (key, value) => { stored[key] = String(value); },
  removeItem: (key) => { delete stored[key]; },
};
globalThis.fetch = async (...args) => {
  fetchCalls.push(args);
  return fetchImpl(...args);
};

const { apiService } = await import('./api.service.js');
const { sensingService } = await import('./sensing.service.js');

beforeEach(() => {
  stored = { [STORAGE_KEY]: 'status-probe-token' };
  fetchCalls = [];
  apiService.setAuthToken('status-probe-token');
  sensingService._state = 'connected';
  sensingService._dataSource = 'reconnecting';
  sensingService._serverSource = null;
  sensingService._sourceProbeGeneration = 0;
  sensingService._sourceObservationRevision = 0;
});

test('status probe uses the configured bearer and applies the server source', async () => {
  fetchImpl = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ source: 'simulated', source_state: 'synthetic' }),
  });

  await sensingService._detectServerSource();

  assert.equal(fetchCalls.length, 1);
  const [url, init] = fetchCalls[0];
  assert.equal(url, `${ORIGIN}/api/v1/status`);
  assert.equal(init.method, 'GET');
  assert.equal(init.headers.Authorization, 'Bearer status-probe-token');
  assert.equal(sensingService.dataSource, 'server-simulated');
  assert.equal(sensingService.serverSource, 'simulated');
});

test('canonical source state prevents a stale hardware label from claiming live', async () => {
  fetchImpl = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ source: 'esp32:offline', source_state: 'disconnected' }),
  });

  sensingService._dataSource = 'live';
  await sensingService._detectServerSource();

  assert.equal(sensingService.dataSource, 'server-simulated');
  assert.equal(sensingService.serverSource, 'esp32:offline');
});

test('an unauthorised status probe never claims live hardware', async () => {
  fetchImpl = async () => ({
    ok: false,
    status: 401,
    statusText: 'Unauthorized',
    json: async () => ({ message: 'authentication required' }),
  });

  sensingService._dataSource = 'live';
  await sensingService._detectServerSource();

  assert.equal(sensingService.dataSource, 'server-simulated');
  assert.equal(sensingService.serverSource, null);
});

test('a network failure never claims live hardware', async () => {
  fetchImpl = async () => { throw new Error('offline'); };

  sensingService._dataSource = 'live';
  await sensingService._detectServerSource();

  assert.equal(sensingService.dataSource, 'server-simulated');
});

test('a source frame wins over a slower failed status probe', async () => {
  let rejectProbe;
  fetchImpl = async () => new Promise((_, reject) => {
    rejectProbe = reject;
  });

  const probe = sensingService._detectServerSource();
  while (!rejectProbe) await Promise.resolve();

  sensingService._handleData({ source: 'esp32' });
  assert.equal(sensingService.dataSource, 'live');
  assert.equal(sensingService.serverSource, 'esp32');

  rejectProbe(new Error('status unavailable'));
  await probe;

  assert.equal(sensingService.dataSource, 'live');
  assert.equal(sensingService.serverSource, 'esp32');

  // The same source remains authoritative on subsequent frames as well.
  sensingService._handleData({ source: 'esp32' });
  assert.equal(sensingService.dataSource, 'live');
});
