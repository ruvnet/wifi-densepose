// Executed tests for the service worker's request routing (ADR-271/272).
//
// WHY THIS EXISTS.
// Browser sign-in appeared to fail: after a successful OAuth round-trip the
// settings panel still offered "Sign in with Cognitum", and only a hard reload
// showed the truth. The server was correct throughout — `/oauth/status` fell
// through to the service worker's cache-first catch-all, so the first
// (signed-out) response was stored in the Cache API and replayed forever.
//
// The Cache API is not the HTTP cache. It ignores `Cache-Control` entirely, so
// the `no-store` the server already sent could not prevent this. Nothing in the
// Rust suite, the UI unit tests, or a curl probe could observe it: curl has no
// service worker, and a hard reload bypasses one. It was only visible by
// driving a real browser.
//
// These tests load the REAL `sw.js` with stubbed worker globals and assert on
// which strategy each path is routed to.
//
// Run: node --test ui/sw.test.mjs ui/services/ws-ticket.test.mjs
// (a directory argument does not work — Node resolves it as a module)

import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const SW_SOURCE = readFileSync(fileURLToPath(new URL('./sw.js', import.meta.url)), 'utf8');
const ORIGIN = 'http://127.0.0.1:8099';

/** Load sw.js in a fresh context and return its registered `fetch` listener. */
function loadServiceWorker() {
  const listeners = {};
  const cachePuts = [];

  // A real in-memory cache, so purge and put behaviour can be observed rather
  // than assumed.
  const entries = new Map();
  const cacheStub = {
    addAll: async () => {},
    put: async (req, res) => {
      const url = String(req.url ?? req);
      cachePuts.push(url);
      entries.set(url, res);
    },
    keys: async () => Array.from(entries.keys()).map((url) => ({ url })),
    delete: async (req) => entries.delete(String(req.url ?? req)),
    match: async () => undefined,
  };

  const sandbox = {
    self: {
      addEventListener: (name, fn) => { listeners[name] = fn; },
      location: { origin: ORIGIN },
      skipWaiting: () => {},
      clients: { claim: () => {} },
    },
    caches: {
      open: async () => cacheStub,
      keys: async () => [],
      delete: async () => true,
      match: async () => undefined,
    },
    // Never actually reached: every assertion below inspects the routing
    // decision, not the network.
    fetch: async () => ({ ok: true, clone: () => ({}) }),
    URL,
    Response: class { constructor(body, init) { this.body = body; Object.assign(this, init); } },
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(SW_SOURCE, sandbox);

  return { listeners, cachePuts, sandbox, entries };
}

/**
 * Route one request and report the decision.
 * `handled: false` means the SW called neither respondWith nor cache — the
 * request goes to the network untouched, which is the only safe outcome for a
 * credentialed endpoint.
 */
function route(path, opts = {}) {
  return dispatch(path, opts).handled;
}

/** Route one request and expose everything the worker did with it. */
function dispatch(path, { method = 'GET', mode = 'cors', headers = {}, sw = null } = {}) {
  const worker = sw ?? loadServiceWorker();
  let handled = false;
  let responded = null;
  const waited = [];
  const request = {
    url: `${ORIGIN}${path}`,
    method,
    mode,
    headers: { get: (k) => headers[k] ?? headers[k.toLowerCase()] ?? null },
  };
  worker.listeners.fetch({
    request,
    respondWith: (p) => { handled = true; responded = p; },
    waitUntil: (p) => { waited.push(p); },
  });
  return { handled, responded, waited, worker };
}

let sw;
beforeEach(() => { sw = loadServiceWorker(); });

// --- the defect this file exists for ----------------------------------------

test('/oauth/status is never handled by the service worker', () => {
  // The exact request whose cached signed-out copy made sign-in look broken.
  assert.equal(route('/oauth/status'), false);
});

test('every /oauth/ path bypasses the worker, not just status', () => {
  // start/callback/logout all carry or clear credentials. A cached redirect or
  // Set-Cookie replayed later is worse than a cached status.
  for (const p of ['/oauth/start', '/oauth/callback?code=x&state=y', '/oauth/logout']) {
    assert.equal(route(p), false, `${p} must go straight to the network`);
  }
});

// --- the underlying defect: cache-first was the catch-all -------------------

test('an unrecognised path is left to the network rather than cached', () => {
  // This is what made the OAuth bug possible in the first place: any endpoint
  // added outside /api/ was silently frozen on its first response. An
  // allowlist means the next such endpoint is safe by default.
  assert.equal(route('/some/future/endpoint'), false);
});

test('static assets are still served cache-first', () => {
  // The offline shell is the point of the worker; the fix must not disable it.
  for (const p of ['/app.js', '/style.css', '/components/TabManager.js', '/icons/logo.svg']) {
    assert.equal(route(p), true, `${p} should be cache-first`);
  }
});

test('a navigation request is still served cache-first', () => {
  assert.equal(route('/ui/', { mode: 'navigate' }), true);
});

test('API paths are still routed through the worker', () => {
  // Handled, but network-only — see the "not written to the cache" test below.
  assert.equal(route('/api/v1/models'), true);
  assert.equal(route('/health/live'), true);
});

// --- pre-existing guards, pinned so the rewrite did not drop them -----------

test('non-GET requests are ignored', () => {
  assert.equal(route('/api/v1/models', { method: 'POST' }), false);
});

test('websocket upgrades are ignored', () => {
  assert.equal(route('/ws/sensing', { headers: { Upgrade: 'websocket' } }), false);
});

test('cross-origin requests are ignored', () => {
  const { listeners } = loadServiceWorker();
  let handled = false;
  listeners.fetch({
    request: {
      url: 'https://auth.cognitum.one/oauth/authorize',
      method: 'GET',
      mode: 'cors',
      headers: { get: () => null },
    },
    respondWith: () => { handled = true; },
  });
  assert.equal(handled, false);
});

// --- authenticated API responses must not be retained ------------------------
// Filed by the cross-vendor prosecutor in the qe-court round after the
// /oauth/status fix: closing the /oauth/ leg left the /api/ leg open.

test('a successful API response is NOT written to the cache', async () => {
  // The leak: cache keys are URLs, nothing partitions them by session, and
  // nothing purged them at sign-out. Sign in as A, fetch sensing data, sign
  // out, sign in as B, lose the network -> B is served A's data.
  const { responded, worker } = dispatch('/api/v1/sensing/latest');
  await responded;
  assert.deepEqual(worker.cachePuts, [], 'API responses must not be cached');
});

test('an API request with no network returns 503 rather than stale data', async () => {
  // Also a correctness property, not only an authorization one: replaying a
  // stale pose reading as current can show a room occupied after the person
  // has left.
  const worker = loadServiceWorker();
  worker.sandbox.fetch = async () => { throw new Error('offline'); };
  worker.entries.set(`${ORIGIN}/api/v1/sensing/latest`, { stale: true });

  const { responded } = dispatch('/api/v1/sensing/latest', { sw: worker });
  const res = await responded;
  assert.equal(res.status, 503);
  assert.match(String(res.body), /offline/);
});

test('signing out purges cached API data but keeps the offline shell', async () => {
  const worker = loadServiceWorker();
  worker.entries.set(`${ORIGIN}/api/v1/sensing/latest`, {});
  worker.entries.set(`${ORIGIN}/health/live`, {});
  worker.entries.set(`${ORIGIN}/app.js`, {});

  const { handled, waited } = dispatch('/oauth/logout', { sw: worker });
  // Observed, not intercepted — the logout request itself must still reach the
  // server, or signing out would not actually sign anyone out.
  assert.equal(handled, false, '/oauth/logout must still go to the network');
  assert.equal(waited.length, 1, 'the purge must be kept alive via waitUntil');
  await Promise.all(waited);

  const left = Array.from(worker.entries.keys());
  assert.deepEqual(left, [`${ORIGIN}/app.js`], 'only the static shell should survive');
});

// --- cache hygiene -----------------------------------------------------------

test('the cache name is bumped so clients holding the poisoned v1 evict it', () => {
  // `activate` deletes every cache whose name !== CACHE_NAME. Browsers that
  // already ran the old worker hold a signed-out /oauth/status in `ruview-v1`;
  // only a name change removes it for them.
  assert.ok(!/['"]ruview-v1['"]/.test(SW_SOURCE), 'CACHE_NAME must not still be ruview-v1');
  assert.ok(/CACHE_NAME\s*=\s*['"]ruview-v[2-9]/.test(SW_SOURCE));
});
