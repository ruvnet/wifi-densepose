// Executed regression test for issue #1461: the pose/event WebSocket path
// (unlike sensing.service.js) opened a bare `new WebSocket(url)` with no
// ADR-272 ticket exchange, and never stripped the long-lived bearer that
// pose.service.js puts on the URL as `?token=`. Both meant the pose stream
// 401'd whenever RUVIEW_API_TOKEN was set.
//
// Run: node --test ui/sw.test.mjs ui/services/ws-ticket.test.mjs ui/services/websocket.service.test.mjs
//
// This EXECUTES createWebSocketWithTimeout in Node with a stub `WebSocket`
// class and stubbed `fetch`/`localStorage`, so it verifies the actual ticket
// exchange + token stripping wiring, not just that the file parses.

import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

const STORAGE_KEY = 'ruview-api-token';

let stored = {};
let fetchCalls = [];
let fetchImpl = async () => ({ ok: true, status: 200, json: async () => ({ ticket: 'T' }) });
let lastConstructedUrl = null;

globalThis.localStorage = {
  getItem: (k) => (k in stored ? stored[k] : null),
  setItem: (k, v) => { stored[k] = String(v); },
  removeItem: (k) => { delete stored[k]; },
};
globalThis.fetch = async (...args) => { fetchCalls.push(args); return fetchImpl(...args); };

// Minimal WebSocket stub: records the URL it was opened with and fires
// `onopen` on the next microtask — by then createWebSocketWithTimeout's
// Promise executor has already assigned `ws.onopen`, so this resolves the
// real promise the way a successful upgrade would, instead of leaving the
// 10s connection-timeout timer dangling for the whole test run.
class FakeWebSocket {
  constructor(url) {
    lastConstructedUrl = url;
    this.url = url;
    this.readyState = 0; // CONNECTING
    queueMicrotask(() => { if (this.onopen) this.onopen(); });
  }
  close() {}
}
globalThis.WebSocket = FakeWebSocket;

const { WebSocketService } = await import('./websocket.service.js');

beforeEach(() => {
  stored = {};
  fetchCalls = [];
  fetchImpl = async () => ({ ok: true, status: 200, json: async () => ({ ticket: 'T' }) });
  lastConstructedUrl = null;
});

test('with no stored bearer, the socket opens with the URL unchanged', async () => {
  const svc = new WebSocketService();
  await svc.createWebSocketWithTimeout('ws://host/api/v1/stream/pose?min_confidence=0.3');
  assert.equal(lastConstructedUrl, 'ws://host/api/v1/stream/pose?min_confidence=0.3');
  assert.equal(fetchCalls.length, 0, 'no ticket should be minted when auth is off');
});

test('a stored bearer is exchanged for a single-use ticket before the socket opens', async () => {
  stored[STORAGE_KEY] = 'secret-bearer';
  const svc = new WebSocketService();
  await svc.createWebSocketWithTimeout('ws://host/api/v1/stream/pose?min_confidence=0.3');

  const url = new URL(lastConstructedUrl);
  assert.equal(url.searchParams.get('ticket'), 'T');
  assert.equal(url.searchParams.get('min_confidence'), '0.3', 'other params must survive');
  assert.ok(!lastConstructedUrl.includes('secret-bearer'), `bearer leaked into URL: ${lastConstructedUrl}`);

  const [path, init] = fetchCalls[0];
  assert.equal(path, '/api/v1/ws-ticket');
  assert.equal(init.headers.Authorization, 'Bearer secret-bearer');
});

test('a stray ?token= from a caller (pose.service.js) is stripped, not forwarded', async () => {
  // Regression for #1461: pose.service.js puts the long-lived bearer on the
  // URL as `?token=`. That must never reach the actual WebSocket upgrade —
  // the ticket exchange above is the only credential that belongs in the URL.
  stored[STORAGE_KEY] = 'secret-bearer';
  const svc = new WebSocketService();
  await svc.createWebSocketWithTimeout('ws://host/api/v1/stream/pose?token=secret-bearer&max_fps=30');

  const url = new URL(lastConstructedUrl);
  assert.equal(url.searchParams.get('token'), null, 'token param must be stripped');
  assert.equal(url.searchParams.get('ticket'), 'T', 'a real ticket must replace it');
  assert.equal(url.searchParams.get('max_fps'), '30', 'unrelated params must survive');
  assert.ok(!lastConstructedUrl.includes('secret-bearer'));
});

test('with no ADR-272 endpoint on the server (404), the socket still opens without a ticket', async () => {
  stored[STORAGE_KEY] = 'secret-bearer';
  fetchImpl = async () => ({ ok: false, status: 404, json: async () => ({}) });
  const svc = new WebSocketService();
  await svc.createWebSocketWithTimeout('ws://host/api/v1/stream/pose');
  assert.equal(lastConstructedUrl, 'ws://host/api/v1/stream/pose');
});
