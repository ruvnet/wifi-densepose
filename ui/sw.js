// RuView Service Worker
// Strategy: NETWORK-FIRST for everything (cache only as an offline fallback).
// This avoids the stale-asset problem the previous cache-first SW caused:
// updated UI files (e.g. observatory/js/main.js) were served from cache
// forever, so fixes never reached the browser. On activation this SW also
// deletes ALL old caches, so a single reload flushes any stale content left
// over from the old cache-first worker.

const CACHE_NAME = 'ruview-v5';

// Install — take over as soon as possible.
self.addEventListener('install', () => {
  self.skipWaiting();
});

// Activate — nuke every cache (flush stale assets) then claim clients.
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.clients.claim();
    // Force any open pages to reload once so they immediately pick up fresh
    // assets after this worker takes over — turns the old "needs 2 reloads /
    // clear cache" trap into a single reload. Fires once per worker version.
    const clients = await self.clients.matchAll({ type: 'window' });
    for (const c of clients) { try { c.navigate(c.url); } catch {} }
  })());
});

// Fetch — network-first; cache is only consulted when the network fails.
self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.method !== 'GET') return;
  if (request.headers.get('Upgrade') === 'websocket') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    try {
      const response = await fetch(request);
      // Cache successful same-origin responses for offline fallback only.
      if (response && response.ok) {
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, response.clone());
      }
      return response;
    } catch {
      const cached = await caches.match(request);
      if (cached) return cached;
      if (request.headers.get('Accept')?.includes('text/html')) {
        const fallback = await caches.match('/index.html');
        if (fallback) return fallback;
      }
      return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
    }
  })());
});
