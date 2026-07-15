// §4.8 Event Bus & Automation Feed — ADR-131 / ADR-129.
//
// Live event stream (seeded from /api/events, then prepended live from
// the shared WS bus — never polled, §2/§4.4), a context-causality
// breadcrumb on row expand (Context.id → parent_id → grandparent_id),
// and a trigger→condition→action automation builder (ADR-129 scope:
// UI-only, no backend persistence — rules live in a local array).

import {
  h, clear, card, pill, statusPill, sectionHeader, mono, relTime,
  collapsible, lagIndicator, button, banner,
} from '../ui.js';

const MAX_ROWS = 200; // virtualization-lite: cap DOM rows, drop oldest.

// event-type → pill colour variant (§4.8).
const VARIANT = {
  StateChanged: 'cyan',
  EntityRegistered: 'green',
  ConfigReloaded: 'purple',
};
function typePill(type) {
  return pill(type, VARIANT[type] || 'grey');
}

// A live WS event carries event_type:'state_changed'; normalise it into
// the same record shape as api.recentEvents() so the row renderer is one
// code path.
function normalizeLive(evt) {
  return {
    type: 'StateChanged',
    entity_id: evt.entity_id,
    old_state: evt.old_state,
    new_state: evt.new_state,
    ts: new Date().toISOString(),
    user_id: null,
    context: { id: null, parent_id: null, grandparent_id: null },
    source: 'live',
    _live: true,
  };
}

const domainOf = (id) => String(id || '').split('.')[0] || '';

export default {
  meta: { title: 'Events' },
  async render(root, ctx) {
    const { api } = ctx;
    const unsubs = [];

    root.appendChild(sectionHeader('Event Bus & Automation', 'Live entity events + causality + automation builder (ADR-131 §4.8, ADR-129)'));
    if (api.isDemo('events')) {
      root.appendChild(banner('DEMO — event history is contract-conformant mock data until the live /api/events feed lands (§7.1). New rows still arrive over the WS bus.', 'amber'));
    }

    // ── live lag indicator (top, fed by the shared WS bus) ──────────
    const lagHost = h('span');
    const paintLag = (st) => { clear(lagHost); lagHost.appendChild(lagIndicator(st.state, st.lagged)); };
    unsubs.push(ctx.onWs(paintLag)); // fires immediately

    // ── filter bar (mirrors the Cog Store .search field) ────────────
    let filter = '';
    const search = h('input.search', {
      type: 'text',
      placeholder: 'Filter by entity domain · event type · source (e.g. "sensor", "ConfigReloaded", "seed-")',
    });
    search.addEventListener('input', () => { filter = search.value.trim().toLowerCase(); applyFilter(); });

    const list = h('.event-stream', { style: { maxHeight: '460px', overflowY: 'auto' } });
    let rows = []; // { record, node } newest-first, capped to MAX_ROWS.

    function matches(rec) {
      if (!filter) return true;
      const hay = [rec.type, rec.entity_id, domainOf(rec.entity_id), rec.source, rec.user_id]
        .filter(Boolean).join(' ').toLowerCase();
      return hay.includes(filter);
    }
    function applyFilter() {
      for (const r of rows) r.node.classList.toggle('hidden', !matches(r.record));
    }

    function prepend(rec) {
      const node = eventRow(rec);
      rows.unshift({ record: rec, node });
      list.insertBefore(node, list.firstChild);
      node.classList.toggle('hidden', !matches(rec));
      while (rows.length > MAX_ROWS) {
        const old = rows.pop();
        if (old.node.parentNode) old.node.parentNode.removeChild(old.node);
      }
    }

    // seed from history (oldest first → prepend so newest ends on top).
    // Wrap ONLY the history load: a missing/unwired recorder must NOT fail
    // the panel — render an inline note and continue with an empty history.
    // The live ctx.onEvent feed (below) attaches regardless (§12 W3).
    let history = [];
    let historyNote = null;
    try {
      history = await api.recentEvents(40);
    } catch (e) {
      history = [];
      historyNote = banner('Event history unavailable — ' + (e.message || e) + (e.upstreamUnavailable ? ' (recorder not yet wired — ADR-131 §12 W3)' : ''), 'amber');
    }
    for (let i = history.length - 1; i >= 0; i--) prepend(history[i]);
    if (!rows.length) list.appendChild(h('.muted-empty', 'No events yet — live events will appear here as they arrive.'));

    // live events prepend as they arrive (never poll).
    unsubs.push(ctx.onEvent((evt) => {
      // strip the placeholder empty-state once real rows arrive.
      const empty = list.querySelector('.muted-empty');
      if (empty) empty.remove();
      prepend(normalizeLive(evt));
    }));

    root.appendChild(card({
      title: 'Live event stream',
      children: [historyNote, h('.flex.spread.mb', h('span.t2', 'Newest first · capped to ' + MAX_ROWS + ' rows'), lagHost), search, list],
    }));

    // ── automation builder (ADR-129) ────────────────────────────────
    root.appendChild(automationBuilder(api));

    return () => { unsubs.forEach((u) => { try { u(); } catch {} }); };
  },
};

// ── event row + causality breadcrumb ──────────────────────────────────
function eventRow(rec) {
  const head = h('.flex.gap-sm.wrap',
    typePill(rec.type),
    h('strong.mono', rec.entity_id),
    rec.type === 'StateChanged'
      ? h('span.t2', mono(rec.old_state == null ? '∅' : rec.old_state), h('span.arr.t3', { style: { margin: '0 6px' } }, '→'), mono(rec.new_state == null ? '∅' : rec.new_state))
      : null,
    h('span', { style: { marginLeft: 'auto' } }, h('small.ts', relTime(rec.ts))),
    rec.user_id ? pill('@' + rec.user_id, 'amber') : h('small.ts', 'system'),
    rec.source ? h('span.mono.t3', rec.source) : null);

  return h('.event-row', { style: { padding: '6px 0', borderBottom: '0.67px solid var(--border)' } },
    collapsible(head, () => causalityBreadcrumb(rec.context), false));
}

function causalityBreadcrumb(c) {
  const wrap = h('.causality', { style: { padding: '8px 0 4px' } });
  wrap.appendChild(h('span.t2', { style: { marginRight: '8px' } }, 'Context chain'));
  const chain = [
    ['id', c && c.id],
    ['parent', c && c.parent_id],
    ['grandparent', c && c.grandparent_id],
  ].filter(([, v]) => v != null);
  if (!chain.length) {
    wrap.appendChild(h('span.t3', 'no context recorded for this event'));
    return wrap;
  }
  chain.forEach(([label, val], i) => {
    if (i > 0) wrap.appendChild(h('span.arr.t3', { style: { margin: '0 8px' } }, '→'));
    wrap.appendChild(h('span.flex.gap-sm', { style: { display: 'inline-flex' } },
      h('small.ts', label), mono(val)));
  });
  return wrap;
}

// ── automation builder (trigger → condition → action) ─────────────────
const TRIGGERS = [
  { id: 'state_changed', label: 'state_changed on RoomState entity' },
  { id: 'seed_reflex', label: 'SEED reflex rule fired' },
  { id: 'custom_event', label: 'custom domain_event topic' },
];
const REFLEX_RULES = ['fragility_alarm', 'hd_anomaly_indicator'];
const ACTION_KINDS = [
  { id: 'call_service', label: 'Call service' },
  { id: 'fire_event', label: 'Fire domain event' },
];

function automationBuilder(api) {
  const rules = [];
  const listHost = h('div');

  // Default callable-service options; enriched asynchronously from the
  // live service registry when reachable (failures are swallowed — the
  // builder stays usable with defaults, and we never leave a dangling
  // rejected promise in production).
  const serviceOpts = ['light.turn_on', 'light.turn_off', 'notify.mobile', 'homecore.recalibrate_room'];
  Promise.resolve()
    .then(() => api.services())
    .then((services) => {
      (services || []).forEach((s) => {
        const name = (s.domain && s.service) ? `${s.domain}.${s.service}` : String(s.name || s.id || s);
        if (name && !serviceOpts.includes(name)) { serviceOpts.push(name); serviceSel.appendChild(h('option', { value: name }, name)); }
      });
    })
    .catch(() => {});

  // ── trigger editor ──
  const triggerSel = sel(TRIGGERS.map((t) => [t.id, t.label]));
  const thresholdInput = h('input.search.mono', { type: 'text', placeholder: 'threshold expression — e.g. anomaly.value > 0.8' });
  const reflexSel = sel(REFLEX_RULES.map((r) => [r, r]));
  const customInput = h('input.search.mono', { type: 'text', placeholder: 'domain_event topic — e.g. presence.regime_change' });
  const triggerExtra = h('div', { style: { marginTop: '8px' } });
  function paintTriggerExtra() {
    clear(triggerExtra);
    if (triggerSel.value === 'state_changed') triggerExtra.appendChild(thresholdInput);
    else if (triggerSel.value === 'seed_reflex') triggerExtra.appendChild(field('Reflex rule', reflexSel));
    else triggerExtra.appendChild(customInput);
  }
  triggerSel.addEventListener('change', paintTriggerExtra);
  paintTriggerExtra();

  // ── condition editor ──
  const conditionInput = h('input.search.mono', { type: 'text', placeholder: 'condition expression — e.g. room.living_room.presence == "occupied"' });

  // ── action editor ──
  const actionSel = sel(ACTION_KINDS.map((a) => [a.id, a.label]));
  const serviceSel = sel(serviceOpts.map((s) => [s, s]));
  const eventInput = h('input.search.mono', { type: 'text', placeholder: 'domain event to fire — e.g. automation.lr_night_dim' });
  const actionExtra = h('div', { style: { marginTop: '8px' } });
  function paintActionExtra() {
    clear(actionExtra);
    if (actionSel.value === 'call_service') actionExtra.appendChild(field('Service', serviceSel));
    else actionExtra.appendChild(eventInput);
  }
  actionSel.addEventListener('change', paintActionExtra);
  paintActionExtra();

  function buildTrigger() {
    if (triggerSel.value === 'state_changed') return { kind: 'state_changed', entity: 'RoomState', threshold: thresholdInput.value.trim() };
    if (triggerSel.value === 'seed_reflex') return { kind: 'seed_reflex', rule: reflexSel.value };
    return { kind: 'custom_event', topic: customInput.value.trim() };
  }
  function buildAction() {
    if (actionSel.value === 'call_service') return { kind: 'call_service', service: serviceSel.value };
    return { kind: 'fire_event', event: eventInput.value.trim() };
  }

  const addBtn = button('Add automation', {
    variant: 'primary',
    onClick: () => {
      rules.push({ trigger: buildTrigger(), condition: conditionInput.value.trim(), action: buildAction() });
      thresholdInput.value = ''; customInput.value = ''; conditionInput.value = ''; eventInput.value = '';
      renderRules();
    },
  });

  function renderRules() {
    clear(listHost);
    if (!rules.length) { listHost.appendChild(h('.muted-empty', 'No automations defined yet (UI-only — not persisted).')); return; }
    rules.forEach((r, i) => listHost.appendChild(ruleCard(r, i, () => { rules.splice(i, 1); renderRules(); })));
  }
  renderRules();

  const builder = card({
    title: 'Automation builder',
    children: [
      h('.t3.mb', 'Trigger → condition → action (ADR-129). UI scope only — assembled rules are held locally, not persisted to the appliance.'),
      h('.grid.cols-3',
        card({ title: 'Trigger', tint: null, children: [field('When', triggerSel), triggerExtra] }),
        card({ title: 'Condition', children: [field('And', conditionInput)] }),
        card({ title: 'Action', children: [field('Then', actionSel), actionExtra] })),
      h('.flex.mt', addBtn),
    ],
  });

  return h('div', builder, card({ title: 'Defined automations', children: [listHost] }));
}

function ruleCard(r, i, onDelete) {
  return card({
    children: [
      h('.flex.spread',
        h('strong', 'Automation #' + (i + 1)),
        button('Remove', { variant: 'ghost', onClick: onDelete })),
      h('.flex.gap-sm.wrap.mt',
        pill('TRIGGER', 'cyan'), triggerSummary(r.trigger)),
      r.condition
        ? h('.flex.gap-sm.wrap.mt', pill('IF', 'amber'), mono(r.condition))
        : h('.flex.gap-sm.wrap.mt', pill('IF', 'grey'), h('span.t3', 'always')),
      h('.flex.gap-sm.wrap.mt',
        pill('ACTION', 'purple'), actionSummary(r.action)),
    ],
  });
}

function triggerSummary(t) {
  if (t.kind === 'state_changed') return h('span', mono('RoomState'), ' ', t.threshold ? mono(t.threshold) : h('span.t3', '(any change)'));
  if (t.kind === 'seed_reflex') return h('span', h('span.t2', 'reflex '), mono(t.rule || '—'));
  return h('span', h('span.t2', 'event '), mono(t.topic || '—'));
}
function actionSummary(a) {
  if (a.kind === 'call_service') return h('span', h('span.t2', 'call '), mono(a.service || '—'));
  return h('span', h('span.t2', 'fire '), mono(a.event || '—'));
}

// ── small form helpers ────────────────────────────────────────────────
function sel(pairs) {
  const s = h('select.inline', { style: { width: '100%' } });
  for (const [val, label] of pairs) {
    const o = document.createElement('option');
    o.value = val; o.textContent = label;
    s.appendChild(o);
  }
  return s;
}
function field(label, control) {
  return h('label', { style: { display: 'block', marginTop: '8px' } },
    h('span.k.t2', { style: { display: 'block', marginBottom: '4px', fontSize: '12.5px' } }, label),
    control);
}
