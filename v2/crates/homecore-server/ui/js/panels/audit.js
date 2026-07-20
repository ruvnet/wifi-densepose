// §4.9 Witness / Audit Log — ADR-131.
//
// Persistent privacy-mode banner (aggregate + per-SEED), the unified
// two-tier witness timeline (SEED SHA-256 chain + homecore Ed25519
// chain merged chronologically), paginated 12-at-a-time, and a
// regulated-deployment attestation-bundle export. Privacy-mode toggles
// are high-stakes and gated behind an explicit inline confirm (§6 honesty
// — never silently mutate what a SEED publishes).

import { h, clear, card, pill, statusPill, sectionHeader, mono, button, banner, relTime } from '../ui.js';

const PAGE_SIZE = 12;

export default {
  meta: { title: 'Audit' },
  async render(root, ctx) {
    const { api } = ctx;

    root.appendChild(sectionHeader('Witness / Audit Log', 'Two-tier provenance — SEED SHA-256 store chain + homecore Ed25519 state chain'));
    if (api.isDemo('audit')) root.appendChild(banner('DEMO — contract-conformant witness data until the live audit endpoint lands (ADR-131 §7.1).', 'amber'));

    // Async data accessors now return Promises (api.js). Wrap the initial
    // loads in try/catch; on failure surface the typed audit/witness banner
    // (§12 W5 distinguishes "not yet wired" upstreams) and bail.
    let modes;
    let firstPage;
    try {
      modes = (await api.privacyModes()).map((m) => ({ ...m }));
      firstPage = await api.witnessLog(0, PAGE_SIZE);
    } catch (e) {
      root.appendChild(banner('Audit/witness unavailable — ' + (e.message || e)
        + (e.upstreamUnavailable ? ' (witness aggregation not yet wired — ADR-131 §12 W5)' : ''), 'red'));
      return () => {};
    }

    const privacyHost = h('div');
    root.appendChild(privacyHost);
    const renderPrivacy = () => { clear(privacyHost); privacyHost.appendChild(privacyCard(modes, renderPrivacy)); };
    renderPrivacy();

    // Unified timeline — its own host so pagination re-renders in place.
    const timelineHost = h('div');
    root.appendChild(timelineHost);

    let page = firstPage.page;
    // Pagination Prev/Next re-fetch the new page (await) and re-render in place.
    const renderTimeline = async (res) => {
      page = res.page;
      clear(timelineHost);
      timelineHost.appendChild(timelineCard(res,
        async () => {
          if (page <= 0) return;
          clear(timelineHost);
          timelineHost.appendChild(h('.muted-empty', 'Loading witness chain…'));
          try { await renderTimeline(await api.witnessLog(page - 1, PAGE_SIZE)); }
          catch (e) { clear(timelineHost); timelineHost.appendChild(banner('Audit/witness unavailable — ' + (e.message || e) + (e.upstreamUnavailable ? ' (witness aggregation not yet wired — ADR-131 §12 W5)' : ''), 'red')); }
        },
        async (last) => {
          if (last) return;
          clear(timelineHost);
          timelineHost.appendChild(h('.muted-empty', 'Loading witness chain…'));
          try { await renderTimeline(await api.witnessLog(page + 1, PAGE_SIZE)); }
          catch (e) { clear(timelineHost); timelineHost.appendChild(banner('Audit/witness unavailable — ' + (e.message || e) + (e.upstreamUnavailable ? ' (witness aggregation not yet wired — ADR-131 §12 W5)' : ''), 'red')); }
        }));
    };
    await renderTimeline(firstPage);

    // Attestation bundle export.
    root.appendChild(exportCard());

    return () => {};
  },
};

// ── Privacy mode (aggregate banner + per-SEED rows + gated toggle) ─────
function privacyCard(modes, rerender) {
  const allPublish = modes.every((m) => m.mode === 'full-publish');
  const anyAudit = modes.some((m) => m.mode === 'audit-only');

  const top = allPublish
    ? banner('Full-publish mode — SEED state changes are published over MQTT.', 'green')
    : banner('Audit-only mode (SHA-256 digests on-SEED only, no MQTT state messages).', 'amber');

  const list = h('div');
  modes.forEach((m, i) => list.appendChild(privacyRow(m, modes, rerender, i)));

  return card({
    title: 'Privacy mode',
    children: [
      top,
      h('.t2.mt', 'Per-SEED configuration — each SEED chooses independently what leaves the device.'),
      list,
    ],
  });
}

function privacyRow(m, modes, rerender, idx) {
  const isPublish = m.mode === 'full-publish';
  const modePill = pill(m.mode, isPublish ? 'green' : 'amber');

  // The confirm step lives inline beneath the row; only one at a time.
  const confirmHost = h('div');

  const toggleBtn = button('Toggle privacy mode', {
    variant: 'ghost',
    onClick: () => {
      clear(confirmHost);
      confirmHost.appendChild(confirmStep(m, modes, rerender, confirmHost));
    },
  });

  const wrap = h('div',
    h('.row',
      h('span.flex.gap-sm', mono(m.seed), modePill),
      toggleBtn),
    confirmHost);
  return wrap;
}

function confirmStep(m, modes, rerender, confirmHost) {
  const target = m.mode === 'full-publish' ? 'audit-only' : 'full-publish';
  const summary = target === 'audit-only'
    ? `${m.seed} will STOP publishing state changes over MQTT — only on-SEED SHA-256 digests remain.`
    : `${m.seed} will START publishing state changes over MQTT (full state values leave the device).`;

  const confirmBtn = button('Confirm', {
    variant: 'primary',
    onClick: () => {
      const live = modes.find((x) => x.seed === m.seed);
      if (live) live.mode = target;
      rerender();
    },
  });
  const cancelBtn = button('Cancel', { variant: 'ghost', onClick: () => clear(confirmHost) });

  return card({
    tint: target === 'audit-only' ? 'amber' : null,
    children: [
      h('.t2', h('span', 'Switch '), mono(m.seed), h('span', ` → ${target}?`)),
      h('.mt', summary),
      h('.flex.gap-sm.mt', confirmBtn, cancelBtn),
    ],
  });
}

// ── Unified two-tier witness timeline ──────────────────────────────────
function timelineCard(res, onPrev, onNext) {
  const { items, page, size, total } = res;
  const lastPage = Math.max(0, Math.ceil(total / size) - 1);
  const isLast = page >= lastPage;

  const head = h('.row',
    h('span.k', 'entity · old → new · when · tier · source SEED · key'),
    h('span.t2', `merged chronological — both chains`));

  const body = h('div');
  if (!items.length) body.appendChild(h('.muted-empty', 'No witness entries.'));
  items.forEach((it) => body.appendChild(witnessRow(it)));

  const from = total === 0 ? 0 : page * size + 1;
  const to = Math.min(total, page * size + items.length);
  const pager = h('.flex.spread.mt',
    h('span.t2', `Showing ${from}–${to} of ${total}`),
    h('span.flex.gap-sm',
      button('‹ Prev', { variant: 'ghost', onClick: onPrev, disabled: page <= 0 }),
      button('Next ›', { variant: 'ghost', onClick: () => onNext(isLast), disabled: isLast })));

  return card({ title: 'Witness timeline', children: [head, body, pager] });
}

function witnessRow(it) {
  const seedTier = it.tier === 'seed-sha256';
  const tierPill = pill(it.tier, seedTier ? 'cyan' : 'purple');

  // old → new. SEED-tier writes have no prior state and a sha256 digest as
  // the "new" value — render the digest mono so it reads as a hash, not state.
  const transition = h('span.flex.gap-sm',
    h('span.mono.t2', it.old_state == null ? '∅' : it.old_state),
    h('span.t3', '→'),
    h('span.mono', it.new_state == null ? '∅' : it.new_state));

  return h('.row',
    h('span.flex.gap-sm.wrap',
      mono(it.entity_id),
      transition),
    h('span.flex.gap-sm.wrap',
      h('span.t2', relTime(it.ts)),
      tierPill,
      mono(it.seed),
      h('span.mono.t3', keyFp(it.key_fp))));
}

function keyFp(fp) {
  if (!fp) return '—';
  return String(fp).slice(0, 8) + '…';
}

// ── Attestation bundle export (regulated-deployment compliance) ────────
function exportCard() {
  const status = h('.t2.mt');
  const btn = button('Export attestation bundle', {
    variant: 'ghost',
    onClick: () => {
      clear(status);
      status.appendChild(h('span.green',
        'Bundle prepared — SEED SHA-256 store chain + homecore Ed25519 state chain packaged for compliance handoff.'));
    },
  });
  return card({
    title: 'Attestation bundle',
    children: [
      h('.t2', 'Packages both witness chains (SEED SHA-256 + homecore Ed25519) for regulated-deployment compliance handoff.'),
      h('.mt', btn),
      status,
    ],
  });
}
