// §4.5 RoomState / Sensing Panel — mixture-of-specialists output.
// Per-room cards from GET /api/v1/room/state?bank=<room_id>.
//
// UX invariants (§4.5/§6): STALE and VETOED are never subtle; veto-
// suppressed values render as withheld, NOT zero; null specialists are
// "Not trained" (calibrate to enable), visually distinct from errors.

import { h, card, pill, statusPill, sectionHeader, bar, confidenceBar, banner, button } from '../ui.js';

export default {
  meta: { title: 'Rooms' },
  async render(root, ctx) {
    const { api } = ctx;
    root.appendChild(sectionHeader('RoomState / Sensing', 'Highest-level per-room sensing from the calibration mixture-of-specialists'));
    let rooms;
    try {
      rooms = await api.roomStates();
    } catch (e) {
      root.appendChild(banner(`RoomState unavailable — ${e && e.message ? e.message : e}. ${e && e.upstreamUnavailable ? 'Calibration service (ADR-151) not reachable through the gateway.' : ''}`, 'red'));
      return () => {};
    }
    if (api.isDemo('rooms')) root.appendChild(banner('DEMO mode (?demo=1) — fixture RoomState, not live calibration output (ADR-131 §2.2).', 'amber'));
    if (!rooms.length) { root.appendChild(h('.muted-empty', 'No calibrated rooms yet — run the Calibration wizard to enable sensing.')); return () => {}; }
    const grid = h('.grid.cols-2');
    rooms.forEach((r) => grid.appendChild(roomCard(r, ctx)));
    root.appendChild(grid);
    return () => {};
  },
};

function roomCard(r, ctx) {
  const tint = r.stale ? 'amber' : (r.vetoed ? 'red' : null);
  const children = [
    h('.flex.spread',
      h('strong.mono', r.room_id),
      h('.flex.gap-sm',
        r.seeds.length > 1 ? pill(r.seeds.length + ' seeds fused', 'purple') : null,
        r.vetoed ? pill('veto active', 'red') : null,
        r.stale ? pill('stale', 'amber') : null)),
  ];

  // STALE banner — must never be subtle (§4.5)
  if (r.stale) {
    children.push(banner('Bank stale — baseline has changed', 'amber',
      button('Recalibrate room', { variant: 'ghost', onClick: () => ctx.navigate('#/calibration') })));
  }
  if (r.vetoed) {
    children.push(banner('Anomaly veto active — implausible window; vitals/posture withheld', 'red'));
  }

  children.push(specRow('Presence', presenceChip(r.presence), r.presence));
  children.push(specRow('Posture', postureView(r), r.posture));
  children.push(vitalRow('Breathing', r.breathing_bpm, 'BPM', [6, 30], r));
  children.push(vitalRow('Heart rate', r.heart_bpm, 'BPM', [40, 120], r));
  children.push(specRow('Restlessness', barOr(r.restlessness, 1), r.restlessness));
  children.push(anomalyRow(r.anomaly));

  return card({ tint, children });
}

function specRow(label, valueNode, spec) {
  const right = h('.flex.gap-sm');
  right.appendChild(valueNode);
  if (spec && spec.confidence != null) right.appendChild(confidenceBar(spec.confidence));
  return h('.row', h('span.k', label), right);
}

function presenceChip(p) {
  if (!p) return notTrainedNode(); // null = not trained
  return statusPill(p.value); // occupied → green, absent → grey
}

function postureView(r) {
  if (r.posture === null) return notTrainedNode();            // not trained
  if (r.vetoed && (!r.posture || r.posture.value == null)) return withheld(); // suppressed, not zero
  if (!r.posture || r.posture.value == null) return withheld();
  return statusPill(r.posture.value);
}

function vitalRow(label, spec, unit, range, r) {
  let valueNode;
  if (spec === null) valueNode = notTrainedNode();
  else if (r.vetoed && (spec.value == null)) valueNode = withheld();
  else if (spec.value == null) valueNode = withheld();
  else valueNode = h('span.cyan', `${spec.value} ${unit} `, h('span.t3', `(${range[0]}–${range[1]})`));
  return specRow(label, valueNode, spec);
}

function anomalyRow(a) {
  if (!a) return specRow('Anomaly', notTrainedNode(), null);
  // §6 honesty: a null threshold is WITHHELD (the upstream RoomState carried
  // none) — show the value but flag the threshold as unavailable rather than
  // judging anomalous/normal against a fabricated 0.8 default.
  if (a.threshold == null) {
    const wrap = h('div', { style: { width: '160px' } },
      bar(a.value, 1),
      h('small.ts', { title: 'no anomaly threshold from upstream — withheld' }, `${a.value} · threshold —`));
    return specRow('Anomaly', wrap, a);
  }
  const over = a.value > a.threshold;
  const b = bar(a.value, 1, [{ lt: a.threshold, color: 'green' }, { lt: 1.01, color: 'red' }]);
  const wrap = h('div', { style: { width: '160px' } }, b,
    h('small.ts', over ? 'anomalous' : 'normal', ` · ${a.value}`));
  return specRow('Anomaly', wrap, a);
}

function barOr(spec, max) {
  if (spec === null) return notTrainedNode();
  if (!spec || spec.value == null) return withheld();
  const wrap = h('div', { style: { width: '140px' } }, bar(spec.value, max), h('small.ts', String(spec.value)));
  return wrap;
}

function notTrainedNode() {
  return h('span.t3', { title: 'null specialist — calibrate to enable' }, 'Not trained');
}
function withheld() {
  return h('span.red', { title: 'suppressed by veto — value withheld, not zero' }, '— withheld');
}
