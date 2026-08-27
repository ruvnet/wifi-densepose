// RuView operator console — entry point.
// Mobile-first shell: desktop sidebar + mobile bottom tab bar, hash router.
// Reuses the proven ui/services/* data layer; presentation is fresh Tailwind.

import { icons } from './icons.js';
import { html, $ } from './lib.js';
import { sensingService } from '../services/sensing.service.js';

import dashboard from './views/dashboard.js';
import sensing from './views/sensing.js';
import nodes from './views/nodes.js';
import training from './views/training.js';
import about from './views/about.js';

// Function-first ordering: the tools come before the docs.
const ROUTES = [dashboard, sensing, nodes, training, about];

// Standalone Three.js pages — kept as-is, linked (not re-implemented).
const EXTERNAL = [
  { id: 'pose-fusion.html', label: 'Pose Fusion', icon: icons.fusion },
  { id: 'observatory.html', label: 'Observatory', icon: icons.observatory },
];

const byId = Object.fromEntries(ROUTES.map((r) => [r.id, r]));
let current = null; // { route, cleanup }

// ── Layout ────────────────────────────────────────────────────────────
function renderSidebar() {
  const aside = $('#sidebar');
  aside.innerHTML = '';
  aside.appendChild(html`
    <div class="flex items-center gap-2.5 h-14 px-4 border-b border-ink-3">
      <span class="text-brand-400 w-7 h-7 block">${icons.logo}</span>
      <div class="leading-tight">
        <div class="font-semibold text-sm">RuView</div>
        <div class="text-[10px] text-ink-muted uppercase tracking-wider">Sensing Console</div>
      </div>
    </div>`);
  const nav = html`<nav class="flex-1 overflow-y-auto p-3 space-y-1" aria-label="Sections"></nav>`;
  ROUTES.forEach((r) => nav.appendChild(html`
    <a href="#${r.id}" class="nav-link" data-route="${r.id}">
      <span>${r.icon}</span><span>${r.label}</span>
    </a>`));
  nav.appendChild(html`<div class="pt-3 mt-3 border-t border-ink-3 text-[10px] uppercase tracking-wider text-ink-muted px-3 pb-1">Visualizers</div>`);
  EXTERNAL.forEach((e) => nav.appendChild(html`
    <a href="${e.id}" class="nav-link"><span>${e.icon}</span><span class="flex-1">${e.label}</span><span class="w-4 h-4 text-ink-muted">${icons.ext}</span></a>`));
  aside.appendChild(nav);
}

function renderTabbar() {
  const bar = $('#tabbar');
  bar.innerHTML = '';
  // Show the 5 most-used tools on the mobile bar (About lives behind the header menu).
  ROUTES.filter((r) => r.id !== 'about').forEach((r) => bar.appendChild(html`
    <a href="#${r.id}" class="tabbar-link" data-route="${r.id}">
      <span>${r.icon}</span><span>${r.label}</span>
    </a>`));
}

function renderTopbar() {
  const bar = $('#topbar');
  bar.innerHTML = '';
  bar.appendChild(html`<span class="md:hidden text-brand-400 w-6 h-6 block">${icons.logo}</span>`);
  bar.appendChild(html`<h1 id="view-title" class="text-base font-semibold flex-1 truncate">RuView</h1>`);
  bar.appendChild(html`
    <a href="#about" class="md:hidden nav-link !px-2 !py-1.5" aria-label="About" title="About">
      <span class="w-5 h-5">${icons.about}</span>
    </a>`);
  bar.appendChild(html`<div id="conn-badge"></div>`);
}

// ── Connection / data-source badge ──────────────────────────────────────
const SOURCE_LABELS = {
  live: ['badge-ok', 'LIVE'],
  'server-simulated': ['badge-warn', 'SIM (server)'],
  simulated: ['badge-warn', 'SIM (offline)'],
  reconnecting: ['badge-mut', 'Connecting…'],
};
function renderConnBadge() {
  const slot = $('#conn-badge');
  if (!slot) return;
  const src = sensingService.dataSource || 'reconnecting';
  const [cls, label] = SOURCE_LABELS[src] || SOURCE_LABELS.reconnecting;
  const live = src === 'live';
  slot.innerHTML = '';
  slot.appendChild(html`
    <span class="${cls}"><span class="dot ${live ? 'bg-ok pulse-live' : 'bg-current opacity-70'}"></span>${label}</span>`);
}

// ── Router ──────────────────────────────────────────────────────────────
function setActive(id) {
  document.querySelectorAll('[data-route]').forEach((a) => {
    if (a.dataset.route === id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

async function navigate() {
  const id = (location.hash.replace(/^#/, '') || 'dashboard').split('?')[0];
  const route = byId[id] || byId.dashboard;

  if (current?.cleanup) { try { current.cleanup(); } catch (e) { console.warn(e); } }

  const view = $('#view');
  view.innerHTML = '';
  view.scrollTo?.(0, 0);
  window.scrollTo(0, 0);

  const titleEl = $('#view-title');
  if (titleEl) titleEl.textContent = route.label;
  setActive(route.id);
  document.title = `RuView — ${route.label}`;

  let cleanup = null;
  try {
    cleanup = (await route.mount(view)) || null;
  } catch (err) {
    console.error(`[router] view "${route.id}" failed:`, err);
    view.appendChild(html`<div class="card card-pad text-bad">This view failed to load: ${err.message}</div>`);
  }
  current = { route, cleanup };
}

// ── Boot ─────────────────────────────────────────────────────────────────
function boot() {
  renderSidebar();
  renderTabbar();
  renderTopbar();
  renderConnBadge();

  sensingService.onStateChange(() => renderConnBadge());
  sensingService.start();

  window.addEventListener('hashchange', navigate);
  navigate();

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
}

document.addEventListener('DOMContentLoaded', boot);
