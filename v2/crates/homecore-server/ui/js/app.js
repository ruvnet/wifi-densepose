// HOMECORE-UI bootstrap + shell + router — ADR-131 §5.
//
// Builds the Cognitum-shell top nav (Framework | Guide | Cog Store |
// HOMECORE | Status) with HOMECORE active, a left sub-nav for the nine
// HOMECORE sections, and a hash router. One shared WebSocket feeds a bus
// that every panel subscribes to (no per-panel sockets, no polling).

import { h, clear, lagIndicator } from './ui.js';
import { api } from './api.js';
import { connect } from './ws.js';

import dashboard from './panels/dashboard.js';
import fleet from './panels/fleet.js';
import seedDetail from './panels/seed-detail.js';
import entities from './panels/entities.js';
import rooms from './panels/rooms.js';
import cogs from './panels/cogs.js';
import calibration from './panels/calibration.js';
import events from './panels/events.js';
import audit from './panels/audit.js';
import settings from './panels/settings.js';

// Section registry. order drives the left sub-nav (§5).
const SECTIONS = [
  { id: 'dashboard', label: 'Dashboard', icon: '◳', mod: dashboard },
  { id: 'fleet', label: 'SEED Fleet', icon: '⬡', mod: fleet },
  { id: 'entities', label: 'Entities', icon: '◈', mod: entities },
  { id: 'rooms', label: 'Rooms', icon: '⌂', mod: rooms },
  { id: 'cogs', label: 'COGs', icon: '⚙', mod: cogs },
  { id: 'calibration', label: 'Calibration', icon: '⊹', mod: calibration },
  { id: 'events', label: 'Events', icon: '⚡', mod: events },
  { id: 'audit', label: 'Audit', icon: '⛨', mod: audit },
  { id: 'settings', label: 'Settings', icon: '⚒', mod: settings },
];
// Detail routes not shown in the sub-nav.
const ROUTES = { 'seed': seedDetail };

// Shared event bus fed by the single WS connection.
const bus = new EventTarget();
let wsState = { state: 'connecting', lagged: false };

const ctx = {
  api,
  bus,
  wsStatus: () => wsState,
  navigate: (hash) => { location.hash = hash; },
  onEvent(handler) {
    const fn = (e) => handler(e.detail);
    bus.addEventListener('hc-event', fn);
    return () => bus.removeEventListener('hc-event', fn);
  },
  onWs(handler) {
    const fn = (e) => handler(e.detail);
    bus.addEventListener('hc-ws', fn);
    handler(wsState);
    return () => bus.removeEventListener('hc-ws', fn);
  },
};

let cleanup = null;

function buildShell() {
  const topnav = h('.topnav',
    h('.brand',
      h('span.logo', 'C'),
      h('span.brand-name', 'Cognitum'),
      h('span.brand-sep', '/'),
      h('span.brand-tag', 'HOMECORE')),
    h('span.nav-spacer'),
    lagIndicatorHost());
  const sidenav = h('.sidenav', ...SECTIONS.map((s) => sideLink(s)));
  const content = h('.content#hc-content');
  const shell = h('.shell', sidenav, content);
  const root = document.getElementById('app');
  clear(root);
  root.appendChild(topnav);
  root.appendChild(shell);
  return content;
}

function sideLink(section) {
  return h('a', { href: '#/' + section.id, 'data-section': section.id },
    h('span.ico', section.icon || '•'), h('span.lbl', section.label));
}

function lagIndicatorHost() {
  const host = h('span');
  const paint = () => { clear(host); host.appendChild(lagIndicator(wsState.state, wsState.lagged)); };
  bus.addEventListener('hc-ws', paint);
  paint();
  return host;
}

function highlightNav(id) {
  document.querySelectorAll('.sidenav a').forEach((a) => {
    a.classList.toggle('active', a.getAttribute('data-section') === id);
  });
}

async function route() {
  const hash = location.hash.replace(/^#\/?/, '') || 'dashboard';
  const [head, ...rest] = hash.split('/');
  const content = document.getElementById('hc-content') || buildShell();

  if (typeof cleanup === 'function') { try { cleanup(); } catch {} cleanup = null; }
  clear(content);

  let mod, params = {};
  const section = SECTIONS.find((s) => s.id === head);
  if (section) { mod = section.mod; highlightNav(head); }
  else if (ROUTES[head]) { mod = ROUTES[head]; params.id = rest[0]; highlightNav('fleet'); }
  else { mod = SECTIONS[0].mod; highlightNav('dashboard'); }

  try {
    const result = await mod.render(content, { ...ctx, params });
    if (typeof result === 'function') cleanup = result;
  } catch (e) {
    content.appendChild(h('.banner.red', 'Panel error: ' + (e && e.message ? e.message : e)));
    console.error(e);
  }
}

function start() {
  buildShell();
  // Attach routing + render the first panel BEFORE opening the socket.
  // connect() invokes its status callback synchronously, so the WS wiring
  // must not be on the critical render path (a thrown callback here would
  // otherwise blank the whole dashboard).
  window.addEventListener('hashchange', route);
  route();
  const ctrl = connect(
    (evt) => bus.dispatchEvent(new CustomEvent('hc-event', { detail: evt })),
    (st) => { wsState = { state: st.state, lagged: !!st.lagged }; bus.dispatchEvent(new CustomEvent('hc-ws', { detail: wsState })); },
  );
  ctx.ws = ctrl;
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();

export { SECTIONS, ctx };
