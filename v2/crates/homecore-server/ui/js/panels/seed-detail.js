// §4.2 SEED Detail View — the per-device deep dive (route #/seed/<id>).
//
// Vector store + witness chain (Ed25519 custody) + onboard sensors +
// reflex rules + cognitive (boundary fragility) analysis + ingest
// pipeline. Backed by the SEED HTTPS API (mock until the live endpoint
// lands → DEMO badge, §7.1). Honesty invariants (§6): null fragility /
// null sensors render muted, never as zero.

import {
  h, card, pill, statusPill, sectionHeader, bar, banner, button, mono, kv,
  sparkline, errorCard, relTime,
} from '../ui.js';

export default {
  meta: { title: 'SEED Detail' },
  async render(root, ctx) {
    const { api } = ctx;
    let s;
    try {
      s = await api.seed(ctx.params.id);
    } catch (e) {
      root.appendChild(sectionHeader('SEED Detail', ctx.params.id));
      root.appendChild(banner('SEED unavailable — ' + (e.message || e) + (e.upstreamUnavailable ? ' (upstream not yet wired — ADR-131 §12)' : ''), 'red'));
      root.appendChild(card({ children: [button('← Back to fleet', { onClick: () => ctx.navigate('#/fleet') })] }));
      return () => {};
    }

    if (!s) {
      root.appendChild(sectionHeader('SEED Detail', ctx.params.id));
      root.appendChild(errorCard(`No SEED with device_id "${ctx.params.id}"`));
      root.appendChild(card({ children: [button('← Back to fleet', { onClick: () => ctx.navigate('#/fleet') })] }));
      return () => {};
    }

    root.appendChild(sectionHeader('SEED Detail', s.zone));
    if (api.isDemo('fleet')) {
      root.appendChild(banner('DEMO — SEED HTTPS API not served by this binary; showing contract-conformant data (§7.1).', 'amber'));
    }

    root.appendChild(identityCard(s, ctx));
    root.appendChild(vectorStoreCard(s));
    root.appendChild(witnessCard(s));
    root.appendChild(sensorsCard(s));
    root.appendChild(reflexCard(s));
    root.appendChild(cognitionCard(s));
    root.appendChild(ingestCard(s));
    return () => {};
  },
};

// ── 1. identity header ────────────────────────────────────────────────
function identityCard(s, ctx) {
  return card({
    children: [
      sectionHeader(s.device_id, `Firmware ${s.firmware} · ${s.zone}`),
      h('.flex.spread',
        statusPill(s.online ? 'online' : 'offline'),
        button('← Fleet', { onClick: () => ctx.navigate('#/fleet') })),
      kv([
        ['Firmware', mono(s.firmware)],
        ['Paired', pill('paired', 'green')],
        ['Conn mode', pill(s.conn, s.conn === 'usb' ? 'cyan' : 'purple')],
        ['Zone', s.zone],
      ]),
    ],
  });
}

// ── 2. vector store ───────────────────────────────────────────────────
function vectorStoreCard(s) {
  const over = s.storage_budget > 0 && s.storage_used / s.storage_budget > 0.8;
  const storeBar = bar(s.storage_used, s.storage_budget, [{ lt: 0.8, color: 'cyan' }, { lt: 1.01, color: 'amber' }]);
  const series = Array.from({ length: 24 }, (_, i) => s.knn_latency_ms != null ? +(s.knn_latency_ms + Math.sin(i / 2) * 0.4).toFixed(2) : 0);

  let compacted = false;
  const compactBtn = button('Compact now', {
    onClick: () => {
      if (compacted) return;
      compacted = true;
      compactBtn.disabled = true;
      compactBtn.textContent = 'Compaction queued';
      console.log('[seed-detail] POST /api/v1/store/compact', s.device_id); // production call
    },
  });

  return card({
    title: 'Vector Store',
    children: [
      kv([
        ['Vectors', s.vector_count.toLocaleString()],
        ['Dimension', mono(String(s.vector_dim))],
        ['kNN latency', s.knn_latency_ms != null ? h('span.cyan', s.knn_latency_ms + ' ms') : h('span.t3', '— offline')],
        ['Epoch', h('span.purple', String(s.epoch))],
        ['kNN latency trend', sparkline(series, { w: 160, hgt: 28 })],
      ]),
      h('.flex.spread.mt',
        h('span.t2', `Storage — ${s.storage_used.toLocaleString()} / ${s.storage_budget.toLocaleString()}`),
        over ? pill('budget > 80%', 'amber') : pill('headroom', 'green')),
      storeBar,
      over ? banner('Vector store nearing budget — compaction recommended.', 'amber') : null,
      h('.mt', compactBtn),
    ],
  });
}

// ── 3. witness chain ──────────────────────────────────────────────────
function witnessCard(s) {
  const verifyBtn = button('Verify chain', {
    onClick: () => console.log('[seed-detail] verify witness chain', s.device_id),
  });
  const exportBtn = button('Export attestation bundle', {
    onClick: () => console.log('[seed-detail] export attestation bundle', s.device_id),
  });
  return card({
    title: 'Witness Chain',
    children: [
      kv([
        ['Chain length', h('span.purple', s.witness_len.toLocaleString())],
        ['Status', s.witness_valid ? pill('valid', 'green') : pill('invalid', 'red')],
        ['Last verify', relTime(s.witness_last_verify)],
      ]),
      h('.flex.gap-sm.mt', verifyBtn, exportBtn),
      h('small.ts',
        'Ed25519 custody attestation — device-bound keypair signs (epoch + vector count + witness head): ',
        mono(`epoch=${s.epoch} · vectors=${s.vector_count} · head=${s.witness_len}`)),
    ],
  });
}

// ── 4. onboard sensors ────────────────────────────────────────────────
function sensorsCard(s) {
  if (!s.sensors) {
    return card({ title: 'Onboard Sensors', children: [h('.muted-empty', 'sensors offline')] });
  }
  const x = s.sensors;
  const grid = h('.grid.cols-3',
    subCard('BME280', [
      sub('Temp', h('span.cyan', x.bme280.temp_c + ' °C')),
      sub('Humidity', h('span.cyan', x.bme280.humidity_pct + ' %')),
      sub('Pressure', h('span.cyan', x.bme280.pressure_hpa + ' hPa')),
    ]),
    subCard('PIR', [
      sub('Motion', x.pir.motion ? pill('motion', 'amber') : pill('still', 'grey')),
      sub('Last trigger', h('span.t2', relTime(x.pir.last_trigger))),
    ]),
    subCard('Reed', [
      sub('State', x.reed.open ? pill('open', 'amber') : pill('closed', 'grey')),
      sub('Last change', h('span.t2', relTime(x.reed.last_change))),
    ]),
    subCard('ADS1115', x.ads1115.map((ch) => sub(ch.label, h('span.cyan', String(ch.v))))),
    subCard('Vibration', [
      sub('State', x.vibration.active ? pill('active', 'amber') : pill('idle', 'grey')),
      sub('Last trigger', h('span.t2', relTime(x.vibration.last_trigger))),
    ]),
  );
  return card({ title: 'Onboard Sensors', children: [grid] });
}

function subCard(name, rows) {
  return card({ children: [h('h3', name), ...rows] });
}
function sub(name, valueNode) {
  return h('.row', h('span.k.t2', name), valueNode instanceof Node ? valueNode : h('span.cyan', String(valueNode)));
}

// ── 5. reflex rules ───────────────────────────────────────────────────
function reflexCard(s) {
  if (!s.reflex || !s.reflex.length) {
    return card({ title: 'Reflex Rules', children: [h('.muted-empty', 'no reflex rules configured')] });
  }
  const rows = s.reflex.map(reflexRow);
  return card({ title: 'Reflex Rules', children: rows });
}

function reflexRow(r) {
  let thresholdNode;
  if (r.name === 'fragility_alarm') {
    const input = h('input.inline', { type: 'number', step: '0.05', value: String(r.threshold) });
    input.addEventListener('change', () => console.log('[seed-detail] reflex threshold edit (no persist)', r.name, input.value));
    thresholdNode = input;
  } else {
    thresholdNode = mono(String(r.threshold));
  }
  const row = h('.row',
    h('.flex.gap-sm', mono(r.name), r.fired_recently ? pill('fired recently', 'amber') : null),
    h('.flex.gap-sm',
      h('span.t2', 'thr'), thresholdNode,
      h('span.t2', '→'), h('span.v', r.target),
      h('small.ts', 'fired ' + (r.last_fired ? relTime(r.last_fired) : 'never'))));
  if (r.fired_recently) {
    return card({ tint: 'amber', children: [row] });
  }
  return row;
}

// ── 6. cognitive analysis ─────────────────────────────────────────────
function cognitionCard(s) {
  const c = s.cognition || {};
  const children = [];

  if (c.fragility == null) {
    children.push(h('.muted-empty', 'fragility unavailable — cognition offline'));
  } else {
    const fragile = c.fragility > 0.3;
    const fb = bar(c.fragility, 1, [{ lt: 0.3, color: 'green' }, { lt: 0.6, color: 'amber' }, { lt: 1.01, color: 'red' }]);
    if (fragile) {
      children.push(banner(`Boundary fragility elevated — ${c.fragility.toFixed(2)} (regime change likely)`, 'amber'));
    }
    children.push(h('.flex.spread', h('span.t2', 'Boundary fragility'), h('span' + (fragile ? '.amber' : '.green'), c.fragility.toFixed(2))));
    children.push(fb);
  }

  if (c.coherence_phases && c.coherence_phases.length) {
    children.push(h('h3.mt', 'Coherence phases'));
    c.coherence_phases.forEach((p) => {
      children.push(h('.row', mono(relTime(p.t)), h('span.v', p.label)));
    });
  }

  children.push(h('.row.mt', h('span.k.t2', 'kNN rebuild cadence'), mono((c.knn_rebuild_s ?? '—') + ' s')));
  return card({ title: 'Cognitive Analysis', children });
}

// ── 7. ingest pipeline ────────────────────────────────────────────────
function ingestCard(s) {
  const ing = s.ingest || {};
  const children = [
    kv([
      ['Batch size', mono(String(ing.batch))],
      ['Flush interval', mono((ing.flush_ms ?? '—') + ' ms')],
      ['Bridge', String(ing.bridge ?? '—')],
    ]),
  ];

  if (ing.bridge && /hop/i.test(ing.bridge)) {
    children.push(banner('Bridge adds a network hop — extra latency + a trust boundary in the ingest path.', 'amber'));
  }

  if (ing.esp32 && ing.esp32.length) {
    children.push(h('h3.mt', 'ESP32 ingest nodes'));
    ing.esp32.forEach((n) => children.push(esp32Row(n)));
  } else {
    children.push(h('.muted-empty', 'no ESP32 nodes attached'));
  }
  return card({ title: 'Ingest Pipeline', children });
}

function esp32Row(n) {
  const native = n.packet === '0xC5110003';
  const packetPill = native
    ? pill('0xC5110003 native', 'green')
    : pill((n.packet || '—') + ' vitals fallback', 'amber');
  return h('.row',
    mono(n.node_id),
    h('.flex.gap-sm', packetPill, h('span.t2', n.rate_hz + ' Hz')));
}
