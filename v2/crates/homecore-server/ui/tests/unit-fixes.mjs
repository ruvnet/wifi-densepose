// Regression tests pinning the ADR-131 PR-1082 review fixes:
//   * dashboard renders a not-available state ('—') for null appliance
//     metrics — never "null%"/"null°C" (§6 honesty / fabricated-data fix).
//   * cogs panel does NOT throw when the gateway forwards a `hef` that is a
//     string (or other non-array) instead of an array (crash/robustness fix).
//   * cogs Hailo worker pill reflects the real probe, not a hardcoded
//     "connected" (§6 honesty fix).
// Run: node tests/unit-fixes.mjs
import { install } from './dom-shim.mjs';
install();
globalThis.HOMECORE_UI_DEMO = false; // production path — no fixtures

const fails = [], passes = [];
async function t(name, fn) {
  try { await fn(); passes.push(name); }
  catch (e) { fails.push(`${name}: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || 'assertion failed'); };

const { api } = await import('../js/api.js');

// Shared ctx; per-test we override the api accessors we need.
function ctxWith(overrides) {
  return {
    api: Object.assign(Object.create(api), overrides),
    navigate() {},
    params: {},
    onEvent() { return () => {}; },
    onWs(fn) { fn({ state: 'closed', lagged: false }); return () => {}; },
  };
}

// ── dashboard: null metrics → '—', never "null%"/"null°C" ─────────────
await t('dashboard renders not-available for null hailo metrics (no "null%")', async () => {
  const mod = await import('../js/panels/dashboard.js');
  const root = document.createElement('div');
  const ctx = ctxWith({
    appliance: async () => ({
      cpu_pct: 12.5, ram_pct: 40.1,
      hailo_load_pct: null, hailo_temp_c: null, // the fabricated-data trap
      uptime_s: null,
      services: [{ name: 'ruview-mcp-brain', port: 9876, status: 'unreachable' }],
      event_rate: [], channel_capacity: 4096, channel_lag: 0,
    }),
    seeds: async () => [],
    esp32Warnings: async () => [],
    cogs: async () => [],
    anyDemo: () => false,
  });
  const cleanup = await mod.default.render(root, ctx);
  const text = root.textContent;
  assert(!/null\s*%/.test(text), `dashboard showed "null%": ${text.slice(0, 200)}`);
  assert(!/null\s*°C/.test(text), `dashboard showed "null°C": ${text.slice(0, 200)}`);
  assert(text.includes('—'), 'dashboard should render the "—" not-available marker for null metrics');
  // real values must still concatenate their unit
  assert(text.includes('12.5%'), 'real CPU value must still render with its unit');
  if (typeof cleanup === 'function') cleanup();
});

// ── cogs: string `hef` must not throw ─────────────────────────────────
await t('cogs does not throw when hef is a string (non-array)', async () => {
  const mod = await import('../js/panels/cogs.js');
  const root = document.createElement('div');
  const ctx = ctxWith({
    cogs: async () => [
      { id: 'cog-pose', version: '1.0', arch: 'hailo10', status: 'running', pid: 42,
        sha256_verified: true, signature_verified: true, throughput_fps: 30,
        hef: 'pose_estimation.hef' }, // STRING, not array — the crash trap
    ],
    cogUpdates: async () => [],
    appliance: async () => ({ services: [{ name: 'ruvector-hailo-worker', port: 50051, status: 'running' }] }),
    isDemo: () => false,
  });
  // If asArray() weren't applied, .forEach/.join/.length on a string would throw.
  const cleanup = await mod.default.render(root, ctx);
  assert(root.children.length > 0, 'cogs rendered nothing');
  // The string hef should surface as a single loaded HEF row.
  assert(root.textContent.includes('pose_estimation.hef'), 'string hef should render as one HEF entry');
  if (typeof cleanup === 'function') cleanup();
});

// ── cogs: Hailo worker pill reflects the real probe, not hardcoded ────
await t('cogs Hailo worker pill is unknown when appliance probe is unavailable', async () => {
  const mod = await import('../js/panels/cogs.js');
  const root = document.createElement('div');
  const ctx = ctxWith({
    cogs: async () => [],
    cogUpdates: async () => [],
    appliance: async () => { throw new Error('appliance upstream down'); }, // probe fails
    isDemo: () => false,
  });
  const cleanup = await mod.default.render(root, ctx);
  // statusPill('unknown') → grey pill containing the literal label "unknown".
  assert(root.textContent.includes('unknown'), 'worker status should be honestly "unknown" when probe fails');
  assert(!/connected/.test(root.textContent), 'worker pill must not fabricate "connected"');
  if (typeof cleanup === 'function') cleanup();
});

console.log(`\n${passes.length} passed, ${fails.length} failed`);
if (fails.length) { console.error('\nFAILURES:'); fails.forEach((f) => console.error('  ✗ ' + f)); process.exit(1); }
console.log('OK — dashboard not-available, cogs string-hef + honest worker pill pinned');
