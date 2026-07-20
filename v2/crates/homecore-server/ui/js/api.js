// HOMECORE-UI API client — ADR-131 §2 / §11.
//
// Production path: every method issues a SAME-ORIGIN request to the
// homecore-server BFF gateway (§2.1). There is NO mock fallback in
// production — a failed upstream rejects, and the panel renders a typed
// error/empty state (§2.2, §11.11). The in-browser mock layer is a
// DEV-ONLY fixture, reachable only when demo mode is on:
//   ?demo=1  in the URL,  globalThis.HOMECORE_UI_DEMO,  or
//   localStorage 'homecore_demo' = '1'.
//
// Gateway route map: ADR-131 §11.2.

// DEV-ONLY fixtures. Loaded via DYNAMIC import so a production bundle that
// never enters demo mode never pulls mock.js into the graph (§2.2). Cached
// after first use so repeated demo calls don't re-import.
let _mock = null;
async function loadMock() {
  if (!_mock) _mock = await import('./mock.js');
  return _mock;
}

const demoFlags = {};

/** Demo mode = explicit dev opt-in only; never the production default. */
export function demoMode() {
  try { if (typeof location !== 'undefined' && /[?&]demo=1(\b|&|$)/.test(location.search || '')) return true; } catch {}
  try { if (typeof globalThis !== 'undefined' && globalThis.HOMECORE_UI_DEMO) return true; } catch {}
  try { if (typeof localStorage !== 'undefined' && localStorage.getItem('homecore_demo') === '1') return true; } catch {}
  return false;
}

export const api = {
  base: '',
  token: () => { try { return localStorage.getItem('homecore_token') || 'dev-token'; } catch { return 'dev-token'; } },
  isDemo: (key) => !!demoFlags[key],
  anyDemo: () => demoMode() && Object.keys(demoFlags).length > 0,
  demoMode,

  async _get(path) {
    const r = await fetch(this.base + path, { headers: { Authorization: 'Bearer ' + this.token() } });
    if (!r.ok) throw httpError(path, r.status);
    return r.json();
  },
  async _post(path, body) {
    const r = await fetch(this.base + path, {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + this.token(), 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw httpError(path, r.status);
    return r.json();
  },
  async _delete(path) {
    const r = await fetch(this.base + path, { method: 'DELETE', headers: { Authorization: 'Bearer ' + this.token() } });
    if (!r.ok) throw httpError(path, r.status);
    return r.status === 204 ? {} : r.json();
  },

  // demo-gated data accessor: real gateway GET in prod, mock fixture in demo.
  // The mock module is dynamically imported ONLY on the demo branch, so prod
  // never loads it. `mockFn` receives the loaded module.
  async _data(key, path, mockFn) {
    if (demoMode()) { demoFlags[key] = true; return mockFn(await loadMock()); }
    delete demoFlags[key];
    return this._get(path);
  },

  // ── homecore-api (real, already served) ───────────────────────────
  async config() { return this._get('/api/config'); },
  async states() {
    if (demoMode()) { demoFlags.states = true; return demoEntities(); }
    delete demoFlags.states;
    return this._get('/api/states');
  },
  async services() { return this._data('services', '/api/services', () => []); },
  async callService(domain, service, data) { return this._post(`/api/services/${domain}/${service}`, data); },
  async setState(entityId, state, attributes) { return this._post(`/api/states/${entityId}`, { state, attributes: attributes || {} }); },

  // ── gateway /api/homecore/* + /api/events (§11.2) ─────────────────
  async appliance() { return this._data('appliance', '/api/homecore/appliance', (m) => m.applianceHealth()); },
  async seeds() { return this._data('fleet', '/api/homecore/seeds', (m) => m.seeds()); },
  async seed(id) { return this._data('fleet', '/api/homecore/seeds/' + encodeURIComponent(id), (m) => m.seed(id)); },
  async esp32Warnings() {
    if (demoMode()) { demoFlags.fleet = true; return (await loadMock()).esp32Warnings(); }
    const seeds = await this._get('/api/homecore/seeds');
    return seeds.flatMap((s) => (s.warnings || []).map((issue) => ({ node_id: s.device_id, seed: s.device_id, issue })));
  },
  async cogs() { return this._data('cogs', '/api/homecore/cogs', (m) => m.cogs()); },
  async cogUpdates() { return this._data('cogs', '/api/homecore/cogs/updates', (m) => m.cogUpdates()); },
  async hailo() { return this._data('cogs', '/api/homecore/hailo', (m) => ({ worker: 'connected', cogs: m.cogs().filter((c) => c.arch === 'hailo10') })); },
  async roomStates() { return this._data('rooms', '/api/homecore/rooms', (m) => m.roomStates()); },
  async federation() { return this._data('fleet', '/api/homecore/federation', (m) => m.federation()); },
  async witnessLog(page = 0, size = 12) { return this._data('audit', `/api/homecore/witness?page=${page}&size=${size}`, (m) => m.witnessLog(page, size)); },
  async privacyModes() { return this._data('audit', '/api/homecore/privacy', (m) => m.privacyModes()); },
  async setPrivacy(seed, modeValue) { if (demoMode()) return { seed, mode: modeValue }; return this._post('/api/homecore/privacy', { seed, mode: modeValue }); },
  async eventHistory(n = 40) { return this._data('events', `/api/events?limit=${n}`, (m) => m.recentEvents(n)); },
  recentEvents(n) { return this.eventHistory(n); }, // back-compat alias (async)
  async settings() { return this._data('settings', '/api/homecore/settings', (m) => m.settings()); },
  async automations() { return this._data('automations', '/api/homecore/automations', () => []); },
  async saveAutomation(a) { if (demoMode()) return a; return this._post('/api/homecore/automations', a); },
  async tokens() { return this._data('settings', '/api/homecore/tokens', (m) => m.settings().tokens); },

  // calibration (ADR-151) — real proxy in prod, simulated in demo.
  calibration: makeCalibration(),
};

function httpError(path, status) {
  const e = new Error(`${path} → HTTP ${status}`);
  e.status = status;
  e.upstreamUnavailable = status === 503 || status === 504;
  return e;
}

// Demo-only entity fixture (prod path uses real GET /api/states).
function demoEntities() {
  return [
    { entity_id: 'sensor.living_room_presence', state: 'true', attributes: { friendly_name: 'Living Room Presence', source: 'esp32-lr-01', seed: 'seed-livingroom-a1' }, last_changed: new Date().toISOString(), last_updated: new Date().toISOString(), context: { id: 'ctx-1', user_id: null, parent_id: null } },
    { entity_id: 'sensor.bedroom_1_breathing_rate', state: '14.5', attributes: { friendly_name: 'Bedroom 1 Breathing Rate', unit_of_measurement: 'BPM', source: 'esp32-br1-01', seed: 'seed-bedroom-1' }, last_changed: new Date().toISOString(), last_updated: new Date().toISOString(), context: { id: 'ctx-2', user_id: null, parent_id: 'ctx-1' } },
  ];
}

/**
 * Resolve an entity's tier provenance (§4.4 / §11.9). Prefers the
 * explicit `attributes.seed`/`attributes.cog` lineage that integrations
 * are expected to stamp; falls back to parsing the ESP32 node id. In demo
 * mode it may consult the mock node registry. Missing lineage → 'unknown'
 * (never fabricated).
 */
export function entityProvenance(entity) {
  const attrs = (entity && entity.attributes) || {};
  const src = String(attrs.source || '');
  const nodeMatch = src.match(/esp32[-\w]*/i);
  const node = attrs.node || (nodeMatch ? nodeMatch[0] : null);
  let seed = attrs.seed || null;
  // Demo-only enrichment: consult the mock node registry IF it has already
  // been dynamically loaded by a prior demo data call (this fn is sync, so it
  // cannot await the import). Prod never has `_mock` set → seed stays null
  // (never fabricated).
  if (!seed && demoMode() && node && _mock) {
    const cfg = _mock.settings().esp32.find((n) => n.node_id === node);
    seed = cfg ? cfg.seed : null;
  }
  const hailo = /hailo|pose/i.test(src) || /hailo/i.test(String(attrs.cog || ''));
  const cog = attrs.cog || (/matter|bfld|mmwave|mr60/i.test(src) ? 'cog-ha-matter' : (hailo ? 'cog-pose-estimation' : null));
  return { esp32: node, seed: seed || (node ? 'unknown' : null), cog: cog || 'unknown', hailo };
}

// Calibration: per-call branch on demo mode. Prod proxies the real
// calibrate-serve API via the gateway (/api/cal/v1/*). All methods are
// async (the §4.7 wizard awaits them).
function makeCalibration() {
  const ANCHORS = ['empty', 'stand_still', 'sit', 'lie_down', 'breathe_slow', 'breathe_normal', 'small_move', 'sleep_posture'];
  // demo session state
  let frames = 0; const target = 1200; const accepted = new Set();
  const get = (p) => api._get('/api/cal/v1' + p);
  const post = (p, b) => api._post('/api/cal/v1' + p, b);
  return {
    ANCHORS,
    get demo() { return demoMode(); },
    async start() {
      if (demoMode()) { frames = 0; return { baseline_id: 'bl-demo-' + ANCHORS.length }; }
      return post('/calibration/start', {});
    },
    async stop() { if (demoMode()) return { stopped: true }; return post('/calibration/stop', {}); },
    async status() {
      if (demoMode()) { frames = Math.min(target, frames + 180); return { frames, target, eta_s: Math.max(0, Math.round((target - frames) / 180)), z_median: 0.41, motion_flagged: frames < 360 }; }
      return get('/calibration/status');
    },
    async anchor(label) {
      if (demoMode()) {
        const ok = label !== 'sleep_posture' || accepted.size >= 6;
        if (ok) accepted.add(label);
        return { label, accepted: ok, reason: ok ? null : 'insufficient stillness — retry', features: { mean: 0.12, variance: 0.04, breathing_score: 0.7, heart_score: 0.55 } };
      }
      return post('/enroll/anchor', { label });
    },
    async enrollStatus() {
      if (demoMode()) return { accepted: [...accepted], total: ANCHORS.length };
      return get('/enroll/status');
    },
    async train(room_id) {
      if (demoMode()) {
        const trained = accepted.size >= 6;
        return {
          presence: trained ? { threshold: 0.31, occupied_var: 0.08 } : null,
          posture: trained ? { prototypes: 4 } : null,
          breathing: accepted.has('breathe_normal') ? { min_score: 0.6 } : null,
          heartbeat: accepted.has('breathe_normal') ? { min_score: 0.5 } : null,
          restlessness: trained ? { calm: 0.05, active: 0.6 } : null,
          anomaly: trained ? { prototypes: 8, scale: 1.4 } : null,
        };
      }
      return post('/room/train', { room_id });
    },
    reset() { accepted.clear(); frames = 0; },
  };
}
