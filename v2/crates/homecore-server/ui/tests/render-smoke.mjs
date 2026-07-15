// Render-smoke test — actually executes every HOMECORE-UI panel against
// the DOM shim and asserts each builds a non-empty DOM subtree without
// throwing. Also exercises the ui.js helpers and the mock contract.
// Run: node tests/render-smoke.mjs   (from the ui/ dir)
import { install } from './dom-shim.mjs';
install();
globalThis.HOMECORE_UI_DEMO = true; // render panels against fixtures

const fails = [];
const passes = [];
function check(name, fn) {
  try { fn(); passes.push(name); }
  catch (e) { fails.push(`${name}: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); }
}
async function checkAsync(name, fn) {
  try { await fn(); passes.push(name); }
  catch (e) { fails.push(`${name}: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); }
}

const ui = await import('../js/ui.js');
const { api, entityProvenance } = await import('../js/api.js');
const mock = await import('../js/mock.js');

// ── ui.js helper unit checks ────────────────────────────────────────
check('ui.h builds element with class/id', () => {
  const n = ui.h('div.card#x', { 'data-k': 'v' }, 'hi');
  if (n.tagName !== 'DIV') throw new Error('tag');
  if (!n.classList.contains('card')) throw new Error('class');
  if (n.id !== 'x') throw new Error('id');
});
check('ui.statusPill maps running→green', () => {
  const p = ui.statusPill('running');
  if (!p.classList.contains('green')) throw new Error('expected green pill');
});
check('ui.statusPill maps offline→red', () => {
  if (!ui.statusPill('offline').classList.contains('red')) throw new Error('expected red');
});
check('ui.bar applies threshold colour', () => {
  const b = ui.bar(0.9, 1, [{ lt: 0.3, color: 'green' }, { lt: 0.6, color: 'amber' }, { lt: 1.01, color: 'red' }]);
  if (!b.firstChild.classList.contains('red')) throw new Error('expected red fill at 0.9');
});
check('ui.confidenceBar amber under 0.4', () => {
  if (!ui.confidenceBar(0.2).firstChild.classList.contains('amber')) throw new Error('low conf should be amber');
});
check('ui.provenanceBadge marks hailo', () => {
  const p = ui.provenanceBadge({ esp32: 'e', seed: 's', cog: 'c', hailo: true });
  if (!p.querySelector('.hailo')) throw new Error('hailo class missing');
});
check('ui.sparkline yields svg polyline', () => {
  const s = ui.sparkline([1, 2, 3, 4]);
  if (!s.querySelector('polyline')) throw new Error('no polyline');
});

// ── mock contract checks ────────────────────────────────────────────
check('mock RoomState distinguishes null vs withheld', () => {
  const rs = mock.roomStates();
  const office = rs.find((r) => r.room_id === 'office');
  if (office.posture !== null) throw new Error('office posture should be null (not trained)');
  const kitchen = rs.find((r) => r.room_id === 'kitchen');
  if (!kitchen.vetoed) throw new Error('kitchen should be vetoed');
  if (kitchen.posture.value !== null) throw new Error('vetoed posture value should be null/withheld, not zero');
});
check('analysis covers at least 3 bedrooms', () => {
  const beds = mock.roomStates().filter((r) => /^bedroom/.test(r.room_id));
  if (beds.length < 3) throw new Error(`expected ≥3 bedrooms in RoomState analysis, got ${beds.length}`);
  const bedSeeds = mock.seeds().filter((s) => /bedroom/i.test(s.zone));
  if (bedSeeds.length < 3) throw new Error(`expected ≥3 bedroom SEED nodes, got ${bedSeeds.length}`);
});
check('mock fleet has an offline seed with red tint semantics', () => {
  if (!mock.seeds().some((s) => !s.online)) throw new Error('need an offline seed for §4.1 tint');
});
check('mock federation states the raw-CSI invariant', () => {
  if (!/never raw CSI/i.test(mock.federation().invariant)) throw new Error('invariant text missing');
});
check('entityProvenance derives node→seed chain', () => {
  const prov = entityProvenance({ attributes: { source: 'esp32-lr-01 BFLD' } });
  if (prov.esp32 !== 'esp32-lr-01') throw new Error('node parse failed');
  if (!prov.seed) throw new Error('seed mapping failed');
});

// ── render every panel ──────────────────────────────────────────────
const PANELS = ['dashboard', 'fleet', 'seed-detail', 'entities', 'rooms', 'cogs', 'calibration', 'events', 'audit', 'settings'];
const ctx = {
  api,
  navigate() {},
  params: { id: 'seed-livingroom-a1' },
  onEvent() { return () => {}; },
  onWs(fn) { fn({ state: 'open', lagged: false }); return () => {}; },
  wsStatus: () => ({ state: 'open', lagged: false }),
  bus: new globalThis.EventTarget(),
};

for (const name of PANELS) {
  await checkAsync(`render panel: ${name}`, async () => {
    const mod = await import(`../js/panels/${name}.js`);
    const panel = mod.default;
    if (!panel || typeof panel.render !== 'function') throw new Error('no default.render export');
    if (!panel.meta || !panel.meta.title) throw new Error('missing meta.title');
    const root = document.createElement('div');
    const cleanup = await panel.render(root, ctx);
    if (root.children.length === 0) throw new Error('rendered nothing into root');
    if (cleanup && typeof cleanup === 'function') cleanup(); // must not throw
  });
}

// ── report ──────────────────────────────────────────────────────────
console.log(`\n${passes.length} passed, ${fails.length} failed`);
if (fails.length) { console.error('\nFAILURES:'); fails.forEach((f) => console.error('  ✗ ' + f)); process.exit(1); }
console.log('OK — all ui helpers, mock contracts, and 10 panels render without throwing');
