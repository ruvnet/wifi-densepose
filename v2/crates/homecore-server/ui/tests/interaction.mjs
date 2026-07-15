// Interaction tests — the dynamic behaviours that syntax/render checks
// cannot reach: the live WebSocket entity patch (§4.4 "never poll"), the
// ws.js handshake + event parse (ADR-130), and the calibration backend
// driving the §4.7 wizard. Run: node tests/interaction.mjs
import { install } from './dom-shim.mjs';
install();
globalThis.HOMECORE_UI_DEMO = true; // exercise the demo/calibration fixture path

const fails = [], passes = [];
async function t(name, fn) {
  try { await fn(); passes.push(name); }
  catch (e) { fails.push(`${name}: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || 'assertion failed'); };

// ── 1. entities panel patches state live over the bus (no polling) ──
await t('entities: live state_changed patches the row in place', async () => {
  const entities = (await import('../js/panels/entities.js')).default;
  const { api } = await import('../js/api.js');
  let handler = null;
  const ctx = {
    api, navigate() {}, params: {},
    onEvent(fn) { handler = fn; return () => {}; },
    onWs(fn) { fn({ state: 'open', lagged: false }); return () => {}; },
  };
  const root = document.createElement('div');
  await entities.render(root, ctx);
  assert(typeof handler === 'function', 'panel must register an onEvent handler (it must not poll)');

  const before = root.querySelectorAll('.t1').map((n) => n.textContent);
  assert(before.some((x) => x === 'true'), 'living_room_presence should start "true" from the mock fallback');

  // Fire a live event; ws.js delivers new_state as a StateView object.
  handler({ event_type: 'state_changed', entity_id: 'sensor.living_room_presence', old_state: { state: 'true' }, new_state: { state: 'false' } });

  const after = root.querySelectorAll('.t1').map((n) => n.textContent);
  assert(after.some((x) => x === 'false'), 'row should now show patched state "false"');
});

// ── 2. ws.js performs the HA-compat handshake and parses events ─────
await t('ws.js: handshake → subscribe_events → parsed event', async () => {
  const sent = [];
  let inst = null;
  globalThis.WebSocket = class { constructor(url) { this.url = url; inst = this; } send(m) { sent.push(JSON.parse(m)); } close() { this.onclose && this.onclose(); } };
  const { connect } = await import('../js/ws.js?ws-test');
  const got = [], status = [];
  const ctrl = connect((e) => got.push(e), (s) => status.push(s));
  assert(inst, 'WebSocket should be constructed');

  inst.onmessage({ data: JSON.stringify({ type: 'auth_required', ha_version: 'x' }) });
  assert(sent[0] && sent[0].type === 'auth' && 'access_token' in sent[0], 'must reply to auth_required with an auth token');

  inst.onmessage({ data: JSON.stringify({ type: 'auth_ok', ha_version: 'x' }) });
  assert(sent.some((m) => m.type === 'subscribe_events' && m.event_type === 'state_changed'), 'must subscribe_events after auth_ok');

  inst.onmessage({ data: JSON.stringify({ type: 'event', event: { event_type: 'state_changed', data: { entity_id: 'light.x', old_state: { state: 'off' }, new_state: { state: 'on' } } } }) });
  assert(got.length === 1, 'one event expected');
  assert(got[0].entity_id === 'light.x' && got[0].new_state.state === 'on', 'event fields must parse through');

  inst.onmessage({ data: JSON.stringify({ type: 'lagged' }) });
  assert(ctrl.isLagged(), 'lag signal should set isLagged');
  ctrl.close();
});

// ── 3. calibration backend drives the 5-step wizard contract ───────
await t('calibration: start→status→anchor→train contract', async () => {
  const { api } = await import('../js/api.js');
  const cal = api.calibration;
  cal.reset();
  const bl = await cal.start();
  assert(bl.baseline_id, 'start() returns a baseline_id (the STALE anchor)');
  let st;
  for (let i = 0; i < 10; i++) { st = await cal.status(); if (st.frames >= st.target) break; }
  assert(st.frames >= st.target, 'status() converges to target frames');

  for (const label of cal.ANCHORS) await cal.anchor(label);
  assert((await cal.enrollStatus()).accepted.length >= 6, 'most anchors accepted after enrollment');

  const trained = await cal.train();
  assert(trained.presence && trained.anomaly, 'train() returns non-null specialists when enrolled');
  cal.reset();
});

console.log(`\n${passes.length} passed, ${fails.length} failed`);
if (fails.length) { console.error('\nFAILURES:'); fails.forEach((f) => console.error('  ✗ ' + f)); process.exit(1); }
console.log('OK — live WS patch, ws.js handshake/parse, and calibration contract verified');
