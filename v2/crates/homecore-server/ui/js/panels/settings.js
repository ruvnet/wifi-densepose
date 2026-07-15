// §4.10 Settings & Integration Config — ADR-131.
// One card per sub-section: SEED fleet management, ESP32 provisioning,
// MQTT / cog-ha-matter config, long-lived access tokens, federation
// config. Security invariants are surfaced as first-class banners
// (USB-only pairing window; "model deltas only, never raw CSI").
//
// Mutations are local-state-only here (no live mutate endpoint yet); the
// node→room assignment edits persist into an in-memory map and the panel
// is flagged DEMO whenever the mock layer is serving it (§7.1 honesty).

import {
  h, clear, card, pill, statusPill, sectionHeader, mono, button, banner, kv, relTime,
} from '../ui.js';

export default {
  meta: { title: 'Settings' },
  async render(root, ctx) {
    const { api } = ctx;

    // Load each card's data independently so one failure doesn't blank the page.
    let s = null, sErr = null;
    let seeds = null, seedsErr = null;
    let fed = null, fedErr = null;
    try { s = await api.settings(); } catch (e) { sErr = e; }
    try { seeds = await api.seeds(); } catch (e) { seedsErr = e; }
    try { fed = await api.federation(); } catch (e) { fedErr = e; }

    root.appendChild(sectionHeader('Settings & Integration Config', 'SEED fleet, ESP32 provisioning, MQTT / cog-ha-matter, access tokens & federation (ADR-131 §4.10)'));

    if (api.isDemo('settings') || api.isDemo('fleet')) {
      root.appendChild(banner('DEMO — settings & fleet are served by the contract-conformant mock layer until their live endpoints land (ADR-131 §7.1). Edits are local-state only.', 'amber'));
    }

    // ── §4.10.1 SEED fleet ──
    if (seedsErr) root.appendChild(cardBanner('SEED Fleet Management', 'SEED fleet unavailable — ' + errText(seedsErr)));
    else root.appendChild(seedFleetCard(seeds));

    // ── §4.10.2/.3/.4 ESP32 + MQTT + tokens (all from settings) ──
    if (sErr) {
      root.appendChild(cardBanner('ESP32 Node Provisioning', 'ESP32 provisioning unavailable — ' + errText(sErr)));
      root.appendChild(cardBanner('MQTT / cog-ha-matter', 'MQTT / cog-ha-matter config unavailable — ' + errText(sErr)));
      root.appendChild(cardBanner('Long-Lived Access Tokens', 'Access tokens unavailable — ' + errText(sErr)));
    } else {
      root.appendChild(esp32Card(s.esp32));
      root.appendChild(mqttCard(s.mqtt, s.ha_disco_entities, s.esp32));
      root.appendChild(tokensCard(s.tokens));
    }

    // ── §4.10.5 Federation (needs federation + seeds) ──
    if (fedErr || seedsErr) root.appendChild(cardBanner('Federation Config', 'Federation config unavailable — ' + errText(fedErr || seedsErr)));
    else root.appendChild(federationCard(fed, seeds));

    return () => {};
  },
};

// ── §4.10.1 SEED fleet management ───────────────────────────────────
function seedFleetCard(seeds) {
  const body = h('div');

  // PROMINENT USB-only pairing invariant (security invariant).
  body.appendChild(banner('Pairing window only opens via 169.254.42.1 (USB), never WiFi — security invariant.', 'red'));

  const list = h('div.mt');
  seeds.forEach((sd) => list.appendChild(seedRow(sd)));
  body.appendChild(list);

  body.appendChild(h('.flex.wrap.gap-sm.mt',
    button('Add SEED', { variant: 'ghost', onClick: () => toggleNote(addNote) }),
    button('Reprovision', { variant: 'ghost', onClick: () => toggleNote(addNote) })));

  const addNote = inlineNote('Provisioning flow', [
    '1. Connect the SEED over USB — it presents a link-local pairing endpoint at 169.254.42.1.',
    '2. Pairing NEVER opens over WiFi; the device refuses pairing on any non-USB interface.',
    '3. Issue a bearer token over the USB link, then attach the SEED to the appliance.',
    '4. Verify the witness chain before accepting the SEED into the fleet.',
  ]);
  body.appendChild(addNote);

  return card({ title: 'SEED Fleet Management', children: [body] });
}

function seedRow(sd) {
  const offline = !sd.online;
  const tokenKind = offline ? 'grey' : 'green';
  const tokenLabel = offline ? 'token idle' : 'token valid';
  const note = inlineNote('Secure token rotation — ' + sd.device_id, [
    '1. Operator confirms physical presence; pairing must be re-opened over USB (169.254.42.1) — never WiFi.',
    '2. Appliance mints a new bearer token and stages it on the SEED over the USB link.',
    '3. SEED acknowledges; the appliance flips the active token and revokes the old one.',
    '4. Witness chain records the rotation (ed25519); old token rejected on next ingest.',
  ]);
  const head = h('.row',
    h('strong.mono', sd.device_id),
    h('.flex.gap-sm',
      h('span.t2', sd.firmware),
      pill(tokenLabel, tokenKind),
      statusPill(sd.online ? 'online' : 'offline'),
      button('Rotate token', { variant: 'ghost', onClick: () => toggleNote(note) }),
      button('Remove', { variant: 'ghost', onClick: () => toggleNote(note) })));
  return h('div', head, note);
}

// ── §4.10.2 ESP32 node provisioning ─────────────────────────────────
function esp32Card(nodes) {
  // local-state room assignment map (node_id → room) — no live endpoint.
  const roomMap = {};
  nodes.forEach((n) => { roomMap[n.node_id] = n.room; });

  const body = h('div');
  nodes.forEach((n) => {
    const sel = h('input.inline', {
      value: roomMap[n.node_id],
      title: 'Editable node→room assignment (local state)',
      onChange: (e) => { roomMap[n.node_id] = e.target.value.trim(); },
    });
    body.appendChild(h('.row',
      h('.flex.gap-sm',
        h('strong.mono', n.node_id),
        mono(n.ip + ':' + n.port),
        h('span.t2', 'fw ' + n.firmware),
        pill(n.seed, 'cyan')),
      h('.flex.gap-sm', h('span.k', 'room'), sel)));
  });

  body.appendChild(h('.t3.mt', 'Provision a new node with the firmware tool: ',
    mono('firmware/esp32-csi-node/provision.py'),
    ' (set --target-ip to this appliance).'));

  body.appendChild(h('.flex.wrap.gap-sm.mt',
    button('Add ESP32 node', { variant: 'ghost', onClick: () => alert('Run provision.py over USB — see hint above.') }),
    button('Apply room map', { variant: 'ghost', onClick: () => alert('Room map persisted locally: ' + JSON.stringify(roomMap)) })));

  return card({ title: 'ESP32 Node Provisioning', children: [body] });
}

// ── §4.10.3 MQTT / cog-ha-matter config ─────────────────────────────
function mqttCard(mqtt, haEntities, esp32) {
  const dotCls = mqtt.connected ? '' : '.err';
  const liveDot = h('span.lag',
    h('span.dot' + dotCls),
    h('span.t2', mqtt.connected ? 'connected' : 'disconnected'));

  const conf = kv([
    ['Broker', mono(mqtt.broker)],
    ['User', mqtt.user],
    ['Credentials', mono('••••••')],
    ['mDNS advertisement', mono(mqtt.mdns)],
    ['Connection', liveDot],
  ]);

  // HA-DISCO entities per node with via_device assignments.
  const disco = h('div.mt',
    h('h3', `HA-DISCO entities — ${haEntities} per node`),
    h('.t3', 'Each ESP32 node publishes its discovery entities with a via_device pointing at its SEED:'));
  esp32.forEach((n) => disco.appendChild(h('.row',
    h('span.mono', n.node_id),
    h('.flex.gap-sm', pill(haEntities + ' entities', 'cyan'), h('span.t2', 'via_device'), mono(n.seed)))));

  return card({ title: 'MQTT / cog-ha-matter', children: [conf, disco] });
}

// ── §4.10.4 Long-lived access tokens ────────────────────────────────
function tokensCard(tokens) {
  const body = h('div');
  tokens.forEach((t) => {
    body.appendChild(h('.row',
      h('.flex.gap-sm', h('strong', t.name), pill('long-lived', 'purple')),
      h('.flex.gap-sm',
        h('span.t2', 'last used ' + relTime(t.last_used)),
        h('span.t3', 'created ' + relTime(t.created)),
        button('Revoke', { variant: 'ghost', onClick: () => alert('Revoking "' + t.name + '" — token rejected on next request (local demo).') }))));
  });

  body.appendChild(h('.flex.wrap.gap-sm.mt',
    button('Create token', { variant: 'primary', onClick: () => alert('A new long-lived token would be minted and shown once (demo).') })));

  // HA companion-app pairing QR placeholder box.
  const qr = h('.muted-empty.mt', { style: { border: '0.67px dashed var(--border)', borderRadius: '8px', padding: '24px', textAlign: 'center' } },
    'HA companion-app pairing QR surfaces here — scan from the Home Assistant mobile app to pair this appliance (placeholder).');
  body.appendChild(qr);

  return card({ title: 'Long-Lived Access Tokens', children: [body] });
}

// ── §4.10.5 Federation config (ADR-105) ─────────────────────────────
function federationCard(fed, seeds) {
  const body = h('div');

  // CRITICAL invariant — model deltas only, never raw CSI (purple).
  body.appendChild(purpleBanner('Federation invariant — ' + fed.invariant + '.'));

  body.appendChild(kv([
    ['Coordinator SEED', mono(fed.coordinator)],
    ['Round', h('span.purple', String(fed.round))],
    ['Healthy SEEDs (k)', String(fed.k_healthy)],
    ['Delta exchange', statusPill(fed.delta_status === 'exchanging' ? 'updating' : fed.delta_status)],
    ['Round cadence', fed.cadence_min + ' min'],
    ['Krum aggregation', h('.flex.gap-sm', pill('f = ' + fed.krum.f, 'cyan'), pill(fed.krum.multi ? 'multi-Krum' : 'single-Krum', 'purple'), h('span.t3', 'ADR-105'))],
  ]));

  // ESP-NOW mesh sync status — rows coloured by health.
  const mesh = h('div.mt', h('h3', 'ESP-NOW mesh sync — cross-SEED epoch alignment'));
  fed.mesh_links.forEach((l) => {
    const epochA = epochOf(seeds, l.a);
    const epochB = epochOf(seeds, l.b);
    const aligned = epochA != null && epochA === epochB;
    mesh.appendChild(h('.row',
      h('.flex.gap-sm', h('span.mono', l.a), h('span.t3', '↔'), h('span.mono', l.b)),
      h('.flex.gap-sm',
        h('span.t2', `epoch ${fmtEpoch(epochA)} / ${fmtEpoch(epochB)}`),
        pill(aligned ? 'aligned' : 'epoch skew', aligned ? 'green' : 'amber'),
        pill(l.health, healthKind(l.health)))));
  });
  body.appendChild(mesh);

  return card({ title: 'Federation Config', children: [body] });
}

// ── helpers ─────────────────────────────────────────────────────────
/** Format a load error, surfacing the §12 upstream-not-wired hint. */
function errText(e) {
  return (e && e.message ? e.message : String(e)) + (e && e.upstreamUnavailable ? ' (upstream not yet wired — ADR-131 §12)' : '');
}
/** Render a card whose body is a red unavailability banner (one card's data failed). */
function cardBanner(title, msg) {
  return card({ title, children: [banner(msg, 'red')] });
}
function epochOf(seeds, id) {
  const s = seeds.find((x) => x.device_id === id);
  return s ? s.epoch : null;
}
function fmtEpoch(e) { return e == null ? '—' : String(e); }
function healthKind(h0) {
  const m = { green: 'green', red: 'red', amber: 'amber' };
  return m[String(h0).toLowerCase()] || 'grey';
}

/** Purple banner for federation invariants (no .banner.purple in CSS). */
function purpleBanner(text) {
  return h('.banner', {
    style: { background: 'var(--purple-d)', color: 'var(--purple)', border: '0.67px solid var(--purple)' },
  }, text);
}

/** A hidden, toggleable multi-step note describing a secure flow. */
function inlineNote(title, steps) {
  const node = h('.banner', {
    style: { background: 'var(--bg2)', border: '0.67px solid var(--border)', color: 'var(--t1)', display: 'none' },
  }, h('strong', title));
  steps.forEach((line) => node.appendChild(h('.t2', { style: { marginTop: '4px' } }, line)));
  return node;
}
function toggleNote(node) {
  node.style.display = node.style.display === 'none' ? 'block' : 'none';
}
