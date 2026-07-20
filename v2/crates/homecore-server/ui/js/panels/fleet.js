// §4.2 SEED Fleet overview + §4.3 SEED Fleet Map (node topology +
// ESP-NOW mesh + cross-SEED event dedup) + ADR-105 federation config.
//
// One panel covering: the fleet card grid, the v0→SEED→ESP32 node
// hierarchy, the mesh-link table, the cross-SEED fusion badges, and the
// federation round config — with the §3.3 "model deltas only — never raw
// CSI" invariant surfaced prominently (ADR-105 privacy guarantee).

import { h, card, pill, statusPill, sectionHeader, relTime, banner } from '../ui.js';

export default {
  meta: { title: 'SEED Fleet' },
  async render(root, ctx) {
    const { api } = ctx;

    root.appendChild(sectionHeader('SEED Fleet', 'Cross-SEED topology, ESP-NOW mesh & ADR-105 federation'));

    // ── Load seeds + federation independently so one failing upstream
    //    doesn't blank the whole panel (ADR-131 §2.2 / §11.11). ───────
    let seeds = null, fed = null;
    try { seeds = await api.seeds(); } catch (e) {
      root.appendChild(banner('SEED fleet unavailable — ' + (e.message || e)
        + (e.upstreamUnavailable ? ' (upstream not yet wired — ADR-131 §12)' : ''), 'red'));
    }
    try { fed = await api.federation(); } catch (e) {
      root.appendChild(banner('SEED fleet unavailable — ' + (e.message || e)
        + (e.upstreamUnavailable ? ' (upstream not yet wired — ADR-131 §12)' : ''), 'red'));
    }

    if (api.isDemo('fleet')) {
      root.appendChild(h('.banner.amber',
        'DEMO — the SEED HTTPS API and the ADR-105 federation service are not served by this homecore-server binary. '
        + 'These panels render against their defined contract with contract-conformant mock data (ADR-131 §7.1).'));
    }

    // ── §4.2 SEED fleet overview ──────────────────────────────────────
    if (seeds) {
      root.appendChild(h('h2', 'Fleet overview'));
      const grid = h('.grid.cols-3');
      seeds.forEach((s) => grid.appendChild(seedCard(s, ctx)));
      root.appendChild(grid);

      // ── §4.3 Node hierarchy (v0 → SEED → ESP32) ─────────────────────
      root.appendChild(card({ title: 'Node hierarchy', children: [hierarchy(seeds)] }));
    }

    if (fed) {
      // ── §4.3 ESP-NOW mesh links ─────────────────────────────────────
      root.appendChild(card({ title: 'ESP-NOW mesh links', children: [meshLinks(fed.mesh_links)] }));

      // ── Cross-SEED event dedup / fusion ─────────────────────────────
      root.appendChild(card({ title: 'Cross-SEED event dedup', children: [fusionBadges(fed.fused_events)] }));

      // ── ADR-105 federation config ───────────────────────────────────
      root.appendChild(federationConfig(fed));
    }

    return () => {};
  },
};

// ── §4.2 SEED card ──────────────────────────────────────────────────
function seedCard(s, ctx) {
  const offline = !s.online;
  return card({
    tint: offline ? 'red' : null, clickable: true,
    onClick: () => ctx.navigate('#/seed/' + s.device_id),
    children: [
      h('.flex.spread',
        h('strong.mono', s.device_id),
        statusPill(s.online ? 'online' : 'offline')),
      h('.kv.mt',
        h('span.k', 'Zone'), h('span.v', s.zone),
        h('span.k', 'Firmware'), h('span.v.mono', s.firmware),
        h('span.k', 'Epoch'), h('span.v.purple', String(s.epoch)),
        h('span.k', 'Vectors'), h('span.v', (s.vector_count || 0).toLocaleString()),
        h('span.k', 'Last ingest'), h('span.v', relTime(s.last_ingest))),
      h('.flex.wrap.gap-sm.mt',
        s.witness_valid ? pill('witness valid', 'green') : pill('witness invalid', 'red')),
      sensorSummary(s.sensors),
    ],
  });
}

function sensorSummary(sensors) {
  if (!sensors) return h('.muted-empty', 'sensors offline');
  return h('.flex.wrap.gap-sm.mt',
    pill('PIR ' + (sensors.pir.motion ? 'motion' : 'still'), sensors.pir.motion ? 'amber' : 'grey'),
    pill('door ' + (sensors.reed.open ? 'open' : 'closed'), sensors.reed.open ? 'amber' : 'grey'),
    pill(sensors.bme280.temp_c + '°C', 'cyan'));
}

// ── §4.3 Node hierarchy diagram (nested indented rows) ──────────────
// v0 Appliance (ROOT) → SEEDs grouped by zone → ESP32 nodes (leaves).
function hierarchy(seeds) {
  const wrap = h('.mono', { style: { fontSize: '12.5px', lineHeight: '1.9' } });

  // ROOT — the v0 appliance.
  wrap.appendChild(treeRow(0, '●', 'cog-v0-appliance', pill('ROOT', 'purple'), null));

  // Second tier — SEEDs grouped by .zone.
  const byZone = groupBy(seeds, (s) => s.zone || 'unzoned');
  const zones = Object.keys(byZone);
  zones.forEach((zone, zi) => {
    const lastZone = zi === zones.length - 1;
    wrap.appendChild(treeRow(1, lastZone ? '└─' : '├─', zone, pill('zone', 'cyan'), null, true));

    const zoneSeeds = byZone[zone];
    zoneSeeds.forEach((s, si) => {
      const lastSeed = si === zoneSeeds.length - 1;
      wrap.appendChild(treeRow(2, lastSeed ? '└─' : '├─', s.device_id,
        statusPill(s.online ? 'online' : 'offline'), null));

      // Leaves — the ESP32 nodes attached to this SEED.
      const nodes = (s.ingest && s.ingest.esp32) || [];
      if (!nodes.length) {
        wrap.appendChild(treeRow(3, '·', '(no ESP32 nodes)', null, null, true));
      }
      nodes.forEach((n, ni) => {
        const lastNode = ni === nodes.length - 1;
        wrap.appendChild(treeRow(3, lastNode ? '└─' : '├─', n.node_id,
          pill(n.rate_hz + ' Hz', 'grey'), n.packet));
      });
    });
  });
  return wrap;
}

function treeRow(depth, connector, label, badge, suffix, muted) {
  const row = h('.flex.gap-sm', { style: { paddingLeft: (depth * 18) + 'px' } });
  row.appendChild(h('span.t3', connector));
  row.appendChild(h(muted ? 'span.t3' : 'span', label));
  if (badge) row.appendChild(badge);
  if (suffix) row.appendChild(h('span.t3', suffix));
  return row;
}

// ── §4.3 ESP-NOW mesh links (dashed rows coloured by .health) ───────
function meshLinks(links) {
  if (!links || !links.length) return h('.muted-empty', 'no mesh links reported');
  const wrap = h('div');
  const colour = { green: 'green', amber: 'amber', red: 'red' };
  links.forEach((l) => {
    const k = colour[l.health] || 'grey';
    wrap.appendChild(h('.flex.gap-sm', { style: { padding: '6px 0' } },
      h('span.mono', l.a),
      h(`span.${k}`, { style: { letterSpacing: '1px' } }, '╌╌╌'),
      h('span.mono', l.b),
      pill(l.health, k)));
  });
  return wrap;
}

// ── Cross-SEED event dedup — fusion badges (kind + n contributing) ──
function fusionBadges(events) {
  if (!events || !events.length) return h('.muted-empty', 'no fused cross-SEED events');
  const wrap = h('.flex.wrap.gap-sm');
  events.forEach((e) => {
    const seeds = (e.seeds || []).join(', ');
    wrap.appendChild(h('span.flex.gap-sm', { style: { alignItems: 'center' } },
      pill(e.kind, 'cyan'),
      pill(e.n + ' SEEDs', 'purple'),
      h('span.t2.mono', { style: { fontSize: '11px' } }, seeds)));
  });
  return wrap;
}

// ── ADR-105 federation config ───────────────────────────────────────
function federationConfig(fed) {
  const body = h('div');

  // CRITICAL invariant — the "model deltas only, never raw CSI" guarantee.
  body.appendChild(h('.banner.purple',
    { style: { background: 'var(--purple-d)', color: 'var(--purple)', border: '0.67px solid var(--purple)' } },
    h('strong', 'Federation invariant: '),
    h('span.mono', fed.invariant)));

  body.appendChild(h('.kv.mt',
    h('span.k', 'Coordinator SEED'), h('span.v.mono', fed.coordinator),
    h('span.k', 'Round'), h('span.v.purple', String(fed.round)),
    h('span.k', 'k_healthy'), h('span.v', String(fed.k_healthy)),
    h('span.k', 'Delta status'), statusPill(fed.delta_status === 'exchanging' ? 'updating' : fed.delta_status),
    h('span.k', 'Krum (f)'), h('span.v', String(fed.krum && fed.krum.f)),
    h('span.k', 'Krum mode'), h('span.v', fed.krum && fed.krum.multi ? 'multi-Krum' : 'Krum'),
    h('span.k', 'Cadence'), h('span.v', (fed.cadence_min != null ? fed.cadence_min + ' min' : '—'))));

  return card({ title: 'Federation config (ADR-105)', accent: true, children: [body] });
}

// ── helpers ─────────────────────────────────────────────────────────
function groupBy(arr, keyFn) {
  const out = {};
  for (const item of arr) {
    const k = keyFn(item);
    (out[k] || (out[k] = [])).push(item);
  }
  return out;
}
