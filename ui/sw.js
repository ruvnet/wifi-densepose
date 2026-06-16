// RuView Service Worker - Offline caching for the dashboard shell
// Strategy: Network-first for API calls, Cache-first for static assets

const CACHE_NAME = 'ruview-v17';
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/style.css',
  '/app.js',
  '/config/api.config.js',
  '/components/TabManager.js',
  '/components/DashboardTab.js',
  '/components/HardwareTab.js',
  '/components/LiveViewTab.js',
  '/components/SensingTab.js',
  '/components/PoseDetectionCanvas.js',
  '/services/api.service.js',
  '/services/websocket.service.js',
  '/services/health.service.js',
  '/services/sensing.service.js',
  '/services/pose.service.js',
  '/services/stream.service.js',
  '/utils/backend-detector.js',
  '/utils/keyboard-shortcuts.js',
  '/utils/perf-monitor.js',
  '/utils/toast.js',
  '/utils/theme-toggle.js',
  '/utils/command-palette.js',
  '/utils/activity-log.js',
  '/utils/data-export.js',
  '/utils/fullscreen.js',
  '/utils/connection-status.js',
  '/utils/mobile-nav.js'
];

// Install - cache shell assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(SHELL_ASSETS).catch((err) => {
        // Don't fail install if some assets are missing (dev mode)
        console.warn('[SW] Some assets failed to cache:', err);
      });
    })
  );
  self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

// Fetch - network-first for API, cache-first for static
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // Skip WebSocket upgrade requests
  if (request.headers.get('Upgrade') === 'websocket') return;

  // Skip cross-origin requests
  if (url.origin !== self.location.origin) return;

  // API calls and launch-critical JS/config: network-first with cache fallback.
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/health/') ||
    url.pathname === '/pose-fusion.html' ||
    url.pathname.startsWith('/pose-fusion/') ||
    url.pathname === '/index.html' ||
    url.pathname === '/observatory.html' ||
    url.pathname === '/observatory/css/observatory.css' ||
    url.pathname === '/style.css' ||
    url.pathname === '/app.js' ||
    url.pathname === '/sw.js' ||
    url.pathname === '/config/api.config.js' ||
    url.pathname === '/components/DashboardTab.js' ||
    url.pathname === '/components/LiveViewTab.js' ||
    url.pathname === '/components/SensingTab.js' ||
    url.pathname === '/services/sensing.service.js' ||
    url.pathname === '/observatory/js/main.js' ||
    url.pathname === '/observatory/js/hud-controller.js' ||
    url.pathname === '/utils/connection-status.js' ||
    url.pathname === '/utils/backend-detector.js'
  ) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Static assets: cache-first with network fallback
  event.respondWith(cacheFirst(request));
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    // Return offline fallback for HTML navigation
    if (request.headers.get('Accept')?.includes('text/html')) {
      const fallback = await caches.match('/index.html');
      if (fallback) return fallback;
    }
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(JSON.stringify({ error: 'offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
