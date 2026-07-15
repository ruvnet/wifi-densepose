// §4.4 Entity & State Browser — live /api/states (real homecore REST).
//
// Entities grouped by domain (prefix before '.') in collapsible sections.
// Each row carries entity_id (mono), current state, last-changed (relTime),
// an INLINE provenanceBadge (§6 invariant 1 — SEED chain never collapsed),
// and a collapsible attributes JSON view. A keyword filter (entity_id +
// attribute keys/values) runs live; semantic search (ADR-132) is a future
// hint. State changes arrive over WebSocket (ctx.onEvent) — rows patch in
// place and flash; NEVER poll. The broadcast-channel lag indicator
// (ctx.onWs) warns when the subscriber falls behind the 4,096 capacity.

import {
  h, clear, card, pill, sectionHeader, mono, provenanceBadge,
  slideover, collapsible, lagIndicator, relTime, banner,
} from '../ui.js';
import { api, entityProvenance } from '../api.js';

export default {
  meta: { title: 'Entities' },
  async render(root, ctx) {
    root.appendChild(sectionHeader('Entity & State Browser', 'Live /api/states — every entity, grouped by domain, with SEED provenance'));

    // ── lag indicator (broadcast channel vs 4,096 capacity) ─────────
    const lagHost = h('.flex.spread.mb');
    const lagSlot = h('span', lagIndicator('connecting', false));
    lagHost.appendChild(lagSlot);
    root.appendChild(lagHost);

    // ── search / filter controls ────────────────────────────────────
    const search = h('input.search', {
      type: 'text',
      placeholder: 'Filter entities — id, attribute keys & values (case-insensitive)…',
    });
    const semantic = h('input.search', { type: 'text', placeholder: 'Semantic search (ADR-132)' });
    semantic.disabled = true;
    semantic.style.opacity = '0.5';
    root.appendChild(h('.flex.wrap.mb', { style: { gap: '8px' } },
      h('div', { style: { flex: '2', minWidth: '220px' } }, search),
      h('div', { style: { flex: '1', minWidth: '180px' } }, semantic)));

    // ── load live state view ────────────────────────────────────────
    const listHost = h('div');
    root.appendChild(listHost);

    // Production /api/states now THROWS on failure — there is NO mock
    // fallback. A failed load is an error state, not a DEMO substitution.
    let states;
    try {
      states = await api.states();
    } catch (e) {
      listHost.appendChild(banner('/api/states unavailable — ' + (e && e.message ? e.message : e), 'red'));
      return () => {};
    }
    if (!Array.isArray(states)) states = [];

    // Demo mode legitimately serves fixtures (demoFlags.states is set by a
    // successful api.states() in demo mode) — label that, not a fallback.
    if (api.isDemo('states')) {
      root.insertBefore(banner('Demo mode — showing contract-conformant fixture entities (§7.1).', 'amber'), listHost);
    }

    // index by entity_id so WS patches are O(1)
    const byId = new Map();
    states.forEach((s) => byId.set(s.entity_id, s));
    // per-entity row controllers (set state text + flash)
    const rows = new Map();

    function render() {
      clear(listHost);
      const q = search.value.trim().toLowerCase();
      const groups = groupByDomain([...byId.values()], q);
      if (!groups.size) {
        listHost.appendChild(h('.muted-empty', q ? 'No entities match the filter.' : 'No entities reported.'));
        return;
      }
      // stable alphabetical domain order
      [...groups.keys()].sort().forEach((domain) => {
        const ents = groups.get(domain).sort((a, b) => a.entity_id.localeCompare(b.entity_id));
        const header = h('.flex.gap-sm', h('strong.mono', domain), pill(ents.length, 'cyan'));
        const section = collapsible(header, () => {
          const body = h('div');
          ents.forEach((e) => body.appendChild(entityRow(e)));
          return body;
        }, true);
        listHost.appendChild(card({ children: [section] }));
      });
    }

    function entityRow(e) {
      const stateText = h('span.t1.mono', String(e.state));
      const changed = h('span.t3', relTime(e.last_changed));
      const top = h('.flex.spread', { style: { cursor: 'pointer', gap: '12px' }, onClick: () => openDetail(e) },
        h('.flex.wrap.gap-sm', { style: { flex: '1', minWidth: '0' } },
          mono(e.entity_id),
          stateText,
          changed),
        // SEED provenance badge — INLINE, never collapsed (§6 invariant 1)
        provenanceBadge(entityProvenance(e)));
      const attrs = collapsible(h('span.t2', 'attributes'),
        () => h('pre.json', JSON.stringify(e.attributes || {}, null, 2)), false);
      const wrap = h('.entity-row', { style: { padding: '8px 0', borderBottom: '0.67px solid var(--border)' } }, top, attrs);
      rows.set(e.entity_id, { stateText, changed, wrap });
      return wrap;
    }

    function openDetail(e) {
      const chain = contextChain(e.context, byId);
      const content = h('div',
        h('.kv',
          h('span.k', 'entity_id'), h('span.v.mono', e.entity_id),
          h('span.k', 'state'), h('span.v.mono', String(e.state)),
          h('span.k', 'last changed'), h('span.v', relTime(e.last_changed)),
          h('span.k', 'last updated'), h('span.v', relTime(e.last_updated))),
        h('.mt', h('h3', 'Provenance'), provenanceBadge(entityProvenance(e))),
        h('.mt', h('h3', 'Context causality'), chain),
        h('.mt', h('h3', 'Attributes'), h('pre.json', JSON.stringify(e.attributes || {}, null, 2))));
      slideover(e.entity_id, content);
    }

    render();
    search.addEventListener('input', render);

    // ── live WebSocket: patch state in place + flash (never poll) ────
    const unEvent = ctx.onEvent((ev) => {
      if (!ev || ev.event_type !== 'state_changed' || !ev.entity_id) return;
      const cur = byId.get(ev.entity_id);
      const ns = ev.new_state || {};
      if (cur) {
        // merge live fields onto the existing record
        cur.state = ns.state != null ? ns.state : cur.state;
        if (ns.attributes) cur.attributes = ns.attributes;
        if (ns.last_changed) cur.last_changed = ns.last_changed;
        if (ns.last_updated) cur.last_updated = ns.last_updated;
        if (ns.context) cur.context = ns.context;
        patchRow(ev.entity_id);
      } else {
        // a newly-appeared entity — fold it in and re-render the group
        byId.set(ev.entity_id, {
          entity_id: ev.entity_id,
          state: ns.state != null ? ns.state : 'unknown',
          attributes: ns.attributes || {},
          last_changed: ns.last_changed || new Date().toISOString(),
          last_updated: ns.last_updated || new Date().toISOString(),
          context: ns.context || { id: null, user_id: null, parent_id: null },
        });
        render();
        patchRow(ev.entity_id);
      }
    });

    function patchRow(id) {
      const e = byId.get(id);
      const r = rows.get(id);
      if (!e || !r) return;
      r.stateText.textContent = String(e.state);
      r.changed.textContent = relTime(e.last_changed);
      // flash cyan then revert after 800ms (§4.4 live feedback)
      r.stateText.style.color = 'var(--cyan)';
      r.stateText.style.transition = 'none';
      setTimeout(() => {
        r.stateText.style.transition = 'color .6s ease';
        r.stateText.style.color = '';
      }, 800);
    }

    // ── broadcast-channel lag indicator ─────────────────────────────
    const unWs = ctx.onWs((st) => {
      clear(lagSlot);
      lagSlot.appendChild(lagIndicator(st.state, st.lagged));
      if (st.lagged) {
        lagSlot.title = 'Subscriber behind the 4,096-event capacity — some state_changed events were dropped';
      }
    });

    return () => { unEvent(); unWs(); };
  },
};

/**
 * Group entities by domain (prefix before the first '.'), applying the
 * keyword filter across entity_id AND attribute keys/values.
 */
function groupByDomain(entities, q) {
  const groups = new Map();
  for (const e of entities) {
    if (q && !matches(e, q)) continue;
    const dot = e.entity_id.indexOf('.');
    const domain = dot > 0 ? e.entity_id.slice(0, dot) : '(no domain)';
    if (!groups.has(domain)) groups.set(domain, []);
    groups.get(domain).push(e);
  }
  return groups;
}

/** Case-insensitive match across entity_id, state and attribute keys/values. */
function matches(e, q) {
  if (e.entity_id.toLowerCase().includes(q)) return true;
  if (String(e.state).toLowerCase().includes(q)) return true;
  const attrs = e.attributes || {};
  for (const [k, v] of Object.entries(attrs)) {
    if (k.toLowerCase().includes(q)) return true;
    try {
      if (String(typeof v === 'object' ? JSON.stringify(v) : v).toLowerCase().includes(q)) return true;
    } catch (_) { /* circular/unstringifiable — skip */ }
  }
  return false;
}

/**
 * Render the Context causality chain (context.id → parent_id) as a mono
 * breadcrumb trail. Walks parent_id up through known contexts when the
 * parent entity is present, otherwise shows the raw id.
 */
function contextChain(ctxObj, byId) {
  if (!ctxObj || !ctxObj.id) return h('span.t3', 'no context');
  const seen = new Set();
  const ids = [];
  let cur = ctxObj;
  while (cur && cur.id && !seen.has(cur.id)) {
    seen.add(cur.id);
    ids.unshift(cur.id);
    if (!cur.parent_id) break;
    ids.unshift(cur.parent_id);
    seen.add(cur.parent_id);
    cur = findContext(cur.parent_id, byId);
  }
  const trail = h('.flex.wrap.gap-sm');
  ids.forEach((id, i) => {
    if (i > 0) trail.appendChild(h('span.arr.t3', '→'));
    trail.appendChild(mono(id));
  });
  return trail;
}

function findContext(id, byId) {
  for (const e of byId.values()) {
    if (e.context && e.context.id === id) return e.context;
  }
  return null;
}
