// §4.1 System Dashboard — the "home screen".
// v0 Appliance health strip (always top) + SEED fleet overview +
// ESP32 summary + COG runtime status row + event-bus sparkline.

import { h, clear, card, metric, pill, statusPill, sectionHeader, sparkline, provenanceBadge } from '../ui.js';

export default {
  meta: { title: 'System Dashboard' },
  async render(root, ctx) {
    const { api } = ctx;
    root.appendChild(sectionHeader('System Dashboard', 'Cognitum v0 Appliance — the machine you are looking at'));
    if (api.anyDemo()) root.appendChild(h('.banner.amber', 'DEMO mode (?demo=1) — panels show contract-conformant fixture data, not live (ADR-131 §2.2).'));

    // Each section loads independently so one offline upstream can't blank
    // the dashboard (§11.1). A failed section renders a typed error card.
    let cleanupEvent = () => {};

    // ── v0 Appliance health strip (always at top) ──────────────────
    await section(root, 'v0 Appliance health', async () => {
      const a = await api.appliance();
      const strip = h('.metric-grid',
        metric({ icon: '🖥', value: pctOrNA(a.cpu_pct), label: 'CPU' }),
        metric({ icon: '🧠', value: pctOrNA(a.ram_pct), label: 'RAM' }),
        metric({ icon: '⚡', value: pctOrNA(a.hailo_load_pct), label: 'Hailo-10H load' }),
        metric({ icon: '🌡', value: unitOrNA(a.hailo_temp_c, '°C'), label: 'Hailo temp' }),
        metric({ icon: '⏱', value: fmtUptime(a.uptime_s), label: 'Uptime', color: 'green' }));
      const healthCard = card({ title: 'v0 Appliance health', children: [strip, servicesRow(a.services)] });
      return h('div', healthCard, eventBus(a, ctx, (fn) => { cleanupEvent = fn; }));
    });

    // ── SEED fleet overview + ESP32 summary ────────────────────────
    await section(root, 'SEED Fleet', async () => {
      const wrap = h('div');
      const seeds = await api.seeds();
      const warnings = await api.esp32Warnings().catch(() => []);
      const grid = h('.grid.cols-3');
      seeds.forEach((s) => grid.appendChild(seedCard(s, ctx)));
      wrap.appendChild(h('h2', 'SEED Fleet'));
      wrap.appendChild(grid);
      wrap.appendChild(esp32Summary(seeds, warnings));
      return wrap;
    });

    // ── COG runtime status row ─────────────────────────────────────
    await section(root, 'COG Runtime', async () => cogRow(await api.cogs(), ctx));

    return () => cleanupEvent();
  },
};

// Run one dashboard section; on failure append a typed error card instead
// of throwing (so the rest of the dashboard still renders).
async function section(root, label, build) {
  try { root.appendChild(await build()); }
  catch (e) {
    root.appendChild(card({ children: [
      h('.banner.red', `${label} unavailable — ${e && e.message ? e.message : e}`),
      h('small.ts', e && e.upstreamUnavailable ? 'upstream not yet wired (ADR-131 §12)' : 'check the gateway / homecore-server'),
    ] }));
  }
}

function servicesRow(services) {
  const wrap = h('.flex.wrap.mt');
  services.forEach((s) => wrap.appendChild(h('span.flex.gap-sm', statusPill(s.status), h('span.mono.t2', `${s.name}:${s.port}`))));
  return wrap;
}

function seedCard(s, ctx) {
  const offline = !s.online;
  const c = card({
    tint: offline ? 'red' : null, clickable: true,
    onClick: () => ctx.navigate('#/seed/' + s.device_id),
    children: [
      h('.flex.spread', h('strong.mono', s.device_id), statusPill(s.online ? 'online' : 'offline')),
      h('.kv.mt',
        h('span.k', 'Firmware'), h('span.v.mono', s.firmware),
        h('span.k', 'Epoch'), h('span.v.purple', String(s.epoch)),
        h('span.k', 'Vectors'), h('span.v', s.vector_count.toLocaleString()),
        h('span.k', 'Last ingest'), h('span.v', relAgo(s.last_ingest)),
        h('span.k', 'Witness'), s.witness_valid ? pill('valid', 'green') : pill('invalid', 'red')),
      sensorSummary(s.sensors),
    ],
  });
  return c;
}

function sensorSummary(sensors) {
  if (!sensors) return h('.muted-empty', 'sensors offline');
  return h('.flex.wrap.gap-sm.mt',
    pill('PIR ' + (sensors.pir.motion ? 'motion' : 'still'), sensors.pir.motion ? 'amber' : 'grey'),
    pill('door ' + (sensors.reed.open ? 'open' : 'closed'), sensors.reed.open ? 'amber' : 'grey'),
    pill(sensors.bme280.temp_c + '°C', 'cyan'));
}

function esp32Summary(seeds, warnings) {
  const total = seeds.reduce((n, s) => n + s.esp32_nodes, 0);
  const body = h('div',
    h('.flex.wrap',
      ...seeds.filter((s) => s.esp32_nodes > 0).map((s) =>
        h('span.flex.gap-sm', h('span.mono.t2', s.device_id), pill(s.esp32_nodes + ' nodes', 'cyan'), h('span.t2', s.frame_rate_hz + ' Hz')))));
  if (warnings.length) {
    body.appendChild(h('.mt', h('h3', 'Warnings (target 100 Hz CSI + 1 Hz vectors)')));
    warnings.forEach((w) => body.appendChild(h('.row', h('span.mono', w.node_id), h('span.amber', w.issue))));
  }
  return card({ title: `ESP32 Nodes — ${total} active`, children: [body] });
}

function cogRow(cogs, ctx) {
  const row = h('.flex.wrap.gap-sm');
  cogs.forEach((c) => {
    const p = statusPill(c.status);
    const wrap = h('span.flex.gap-sm.clickable', { style: { cursor: 'pointer' }, onClick: () => ctx.navigate('#/cogs') },
      p, h('span.mono.t2', c.id), c.arch === 'hailo10' ? pill('hailo', 'purple') : null);
    row.appendChild(wrap);
  });
  return card({ title: 'COG Runtime', children: [row] });
}

function eventBus(a, ctx, setCleanup) {
  const rates = a.event_rate || [];
  const spark = sparkline(rates, { w: 240, hgt: 36 });
  const rate = rates.length ? rates[rates.length - 1] : 0;
  const lag = a.channel_lag || 0;
  const cap = a.channel_capacity || 4096;
  const body = h('div',
    h('.flex.spread', h('span.val.cyan', { style: { fontSize: '20px' } }, rate + ' ev/s'),
      h('span.t2', `capacity ${cap.toLocaleString()}`)),
    spark);
  if (lag > 0) body.appendChild(h('.banner.amber.mt', `Subscriber falling behind — ${lag} events lagged against the ${cap.toLocaleString()} capacity`));
  const host = h('span.t2');
  const un = ctx.onWs((st) => { clear(host); host.appendChild(document.createTextNode(st.state === 'open' ? (st.lagged ? ' · WS lagging' : ' · WS live') : ' · WS offline')); });
  body.appendChild(host);
  if (setCleanup) setCleanup(un);
  return card({ title: 'Event Bus activity', children: [body] });
}

// §6 honesty: a null/undefined metric must render a distinct not-available
// state ('—'), never a fabricated value like "null%"/"null°C".
function pctOrNA(v) { return v == null ? '—' : v + '%'; }
function unitOrNA(v, unit) { return v == null ? '—' : v + unit; }

function fmtUptime(s) {
  if (s == null) return '—';
  const d = Math.floor(s / 86400), hh = Math.floor((s % 86400) / 3600);
  return d > 0 ? `${d}d ${hh}h` : `${hh}h`;
}
function relAgo(iso) {
  const s = Math.round((Date.now() - Date.parse(iso)) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  return Math.round(s / 3600) + 'h ago';
}
