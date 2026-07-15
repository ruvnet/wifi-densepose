// Production-mode test (ADR-131 §2.2 / §11.11): with demo mode OFF and
// the gateway unreachable, every panel must render a typed empty/error
// state WITHOUT throwing and WITHOUT showing fabricated data.
// Run: node tests/prod-errors.mjs
import { install } from './dom-shim.mjs';
install();
globalThis.HOMECORE_UI_DEMO = false; // PRODUCTION path — no fixtures
// fetch already rejects in the shim → simulates an unreachable gateway.

const fails = [], passes = [];
async function t(name, fn) {
  try { await fn(); passes.push(name); }
  catch (e) { fails.push(`${name}: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || 'assertion failed'); };

const { api, demoMode } = await import('../js/api.js');

await t('demoMode() is false in production', () => assert(demoMode() === false));
await t('api.anyDemo() is false in production', () => assert(api.anyDemo() === false));

const PANELS = ['dashboard', 'fleet', 'seed-detail', 'entities', 'rooms', 'cogs', 'calibration', 'events', 'audit', 'settings'];
const ctx = {
  api, navigate() {}, params: { id: 'seed-livingroom-a1' },
  onEvent() { return () => {}; },
  onWs(fn) { fn({ state: 'closed', lagged: false }); return () => {}; },
};

for (const name of PANELS) {
  await t(`prod render (gateway down): ${name} shows a state, never throws`, async () => {
    const mod = await import(`../js/panels/${name}.js`);
    const root = document.createElement('div');
    const cleanup = await mod.default.render(root, ctx);
    // must render SOMETHING (header + error/empty state), not crash, not blank
    assert(root.children.length > 0, 'panel rendered nothing in prod error mode');
    if (typeof cleanup === 'function') cleanup();
  });
}

// No data accessor may have flipped a demo flag in production.
await t('no demo flags set after production renders', () => assert(api.anyDemo() === false, 'a panel served mock data in production'));

console.log(`\n${passes.length} passed, ${fails.length} failed`);
if (fails.length) { console.error('\nFAILURES:'); fails.forEach((f) => console.error('  ✗ ' + f)); process.exit(1); }
console.log('OK — every panel renders a typed empty/error state in production with no mock fallback');
