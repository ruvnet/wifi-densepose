// §4.6 v0 Appliance COG Management — ADR-131.
// Installed COGs (start/stop/restart/logs/config + sha256+sig shield),
// COG Store / App Registry (mirrors seed.cognitum.one/store), OTA
// Updates diff panels, and Hailo HEF status. Mirrors the Cog Store
// visual conventions (card layout, category pills, install/details pair).

import { h, clear, card, pill, statusPill, sectionHeader, mono, button, collapsible, banner } from '../ui.js';

export default {
  meta: { title: 'COGs' },
  async render(root, ctx) {
    const { api } = ctx;
    root.appendChild(sectionHeader('COGs', 'v0 Appliance COG runtime & OTA updates'));
    if (api.isDemo('cogs')) {
      root.appendChild(h('.banner.amber', 'COG management shows contract-conformant DEMO data until the live cog-supervisor endpoint lands (ADR-131 §7.1).'));
    }

    let cogs, updates;
    try {
      cogs = await api.cogs();
      updates = await api.cogUpdates();
    } catch (e) {
      root.appendChild(banner('COG runtime unavailable — ' + (e.message || e) + (e.upstreamUnavailable ? ' (upstream not yet wired — ADR-131 §12)' : ''), 'red'));
      return () => {};
    }

    // ── Installed COGs ─────────────────────────────────────────────
    root.appendChild(h('.flex.gap-sm', h('h2', 'Installed'), pill(String(cogs.length), 'cyan')));
    const installed = h('.grid.cols-2');
    cogs.forEach((c) => installed.appendChild(installedCogCard(c)));
    root.appendChild(installed);

    // ── OTA Updates ────────────────────────────────────────────────
    root.appendChild(h('.flex.gap-sm.mt', h('h2', 'Updates'), pill(String(updates.length), updates.length ? 'amber' : 'grey')));
    if (!updates.length) {
      root.appendChild(card({ children: [h('.muted-empty', 'All COGs up to date.')] }));
    } else {
      updates.forEach((u) => root.appendChild(updateCard(u)));
    }

    // ── Hailo HEF status ───────────────────────────────────────────
    // §6 honesty: the worker pill must reflect the REAL probe, not a
    // hardcoded "connected". Probe the appliance services for the
    // ruvector-hailo-worker; if that upstream is unavailable, show the
    // status as unknown rather than fabricating "connected".
    let workerStatus = 'unknown';
    try {
      const appliance = await api.appliance();
      const svc = (appliance.services || []).find((s) => s.name === 'ruvector-hailo-worker');
      if (svc && svc.status) workerStatus = svc.status;
    } catch { /* leave 'unknown' — honest not-available, never fabricated */ }

    root.appendChild(h('h2.mt', 'Hailo-10H accelerator'));
    root.appendChild(hailoStatus(cogs, workerStatus));

    return () => {};
  },
};

// ── Installed COG card ───────────────────────────────────────────────
function installedCogCard(c) {
  const verified = c.sha256_verified && c.signature_verified;
  const shield = h(`span.shield.${verified ? 'ok' : 'bad'}`, (verified ? '✓ ' : '✗ ') + 'verified');
  const archPill = c.arch === 'hailo10' ? pill('hailo10', 'purple') : pill('arm', 'cyan');

  const body = h('div',
    h('.flex.spread',
      h('strong.mono', `${c.id} ${c.version}`),
      statusPill(c.status)),
    h('.flex.wrap.gap-sm.mt', archPill, shield,
      h('span.t2', 'PID '), mono(c.pid == null ? '—' : c.pid)));

  if (c.status === 'failed' && c.error) {
    body.appendChild(h('.red.mt', { style: { fontFamily: 'var(--mono)', fontSize: '12px' } }, c.error));
  }

  // action ghost buttons
  const actions = h('.flex.wrap.gap-sm.mt',
    button('Start', { onClick: () => {} }),
    button('Stop', { onClick: () => {} }),
    button('Restart', { onClick: () => {} }));
  body.appendChild(actions);

  // View logs drawer
  const logDrawer = h('pre.log.mt.hidden', logText(c));
  let logsOpen = false;
  const logsBtn = button('View logs', {
    onClick: () => { logsOpen = !logsOpen; logDrawer.classList.toggle('hidden', !logsOpen); logsBtn.textContent = logsOpen ? 'Hide logs' : 'View logs'; },
  });
  actions.appendChild(logsBtn);

  // Edit config.json drawer (textarea, no persistence)
  const cfgArea = h('textarea.json.mt.hidden', { rows: 8, spellcheck: 'false' });
  cfgArea.value = configJson(c);
  let cfgOpen = false;
  const cfgBtn = button('Edit config.json', {
    onClick: () => { cfgOpen = !cfgOpen; cfgArea.classList.toggle('hidden', !cfgOpen); cfgBtn.textContent = cfgOpen ? 'Close config' : 'Edit config.json'; },
  });
  actions.appendChild(cfgBtn);

  body.appendChild(logDrawer);
  body.appendChild(cfgArea);

  return card({ tint: c.status === 'failed' ? 'red' : null, children: [body] });
}

function logText(c) {
  if (c.status === 'failed' && c.error) {
    return [
      `[error] ${c.id} v${c.version} exited`,
      `[error] ${c.error}`,
      `[info]  supervisor: marking ${c.id} failed; PID was ${c.pid == null ? 'none' : c.pid}`,
    ].join('\n');
  }
  if (c.status === 'stopped') {
    return `[info]  ${c.id} v${c.version} stopped by operator\n[info]  supervisor: PID released`;
  }
  return [
    `[info]  ${c.id} v${c.version} running (pid ${c.pid})`,
    `[info]  arch=${c.arch} sha256_verified=${c.sha256_verified} signature_verified=${c.signature_verified}`,
    c.arch === 'hailo10' ? `[info]  hailo: ${asArray(c.hef).join(', ') || 'no HEF loaded'} @ ${c.throughput_fps || '—'} fps` : '[info]  cpu-only worker, no Hailo offload',
    '[info]  heartbeat ok',
  ].join('\n');
}

function configJson(c) {
  const cfg = {
    id: c.id,
    version: c.version,
    arch: c.arch,
    autostart: c.status !== 'stopped',
  };
  if (c.arch === 'hailo10') {
    cfg.hef = asArray(c.hef);
    cfg.target_fps = c.throughput_fps || null;
  }
  return JSON.stringify(cfg, null, 2);
}

// Coerce a forwarded manifest `hef` (array | string | object | null) into an
// array so a non-array value degrades gracefully instead of throwing on
// .forEach/.join/.length (the gateway forwards it verbatim — §11).
function asArray(v) {
  if (Array.isArray(v)) return v;
  if (v == null || v === '') return [];
  return [v];
}

// ── OTA update diff card ─────────────────────────────────────────────
function updateCard(u) {
  const diff = h('div',
    h('.flex.gap-sm',
      h('strong.mono', u.id),
      mono(u.from), h('span.t3', '→'), h('span.mono.green', u.to)),
    diffList('New entities', u.new_entities, 'green'),
    diffList('Config changes', u.config_changes, 'amber'),
    h('.flex.gap-sm.mt',
      button('Update', { variant: 'primary', onClick: () => {} }),
      button('Skip', { onClick: () => {} })));
  return card({ children: [diff] });
}

function diffList(title, items, color) {
  if (!items || !items.length) return null;
  const list = h('div.mt', h('h3', title));
  items.forEach((e) => list.appendChild(h('.row', h(`span.mono.${color}`, e))));
  return list;
}

// ── Hailo HEF status ─────────────────────────────────────────────────
function hailoStatus(cogs, workerStatus = 'unknown') {
  const hailoCogs = cogs.filter((c) => c.arch === 'hailo10');
  // statusPill maps 'running'/'connected'→green, 'unreachable'/'error'→red,
  // 'unknown'→grey; the real probe drives the colour, never a hardcode.
  const worker = h('.flex.gap-sm', statusPill(workerStatus), h('span.mono.t2', 'ruvector-hailo-worker:50051'));
  const body = h('div', worker);

  if (!hailoCogs.length) {
    body.appendChild(h('.muted-empty', 'No Hailo-sourced COGs loaded.'));
  } else {
    hailoCogs.forEach((c) => {
      const hef = asArray(c.hef); // gateway forwards manifest `hef` verbatim — may be a string
      const hefRows = h('div',
        h('.flex.spread', h('strong.mono', `${c.id} ${c.version}`), pill((c.throughput_fps || 0) + ' fps', 'purple')));
      hef.forEach((f) => hefRows.appendChild(h('.row', h('span.mono.purple', f), h('span.t2', 'loaded'))));
      if (!hef.length) hefRows.appendChild(h('.muted-empty', 'no .hef files loaded'));
      body.appendChild(h('.mt', hefRows));
    });
  }

  body.appendChild(h('.t3.mt', { style: { fontSize: '12px' } },
    'RF Foundation Encoder (ADR-150) will appear here once available.'));
  return card({ children: [body] });
}
