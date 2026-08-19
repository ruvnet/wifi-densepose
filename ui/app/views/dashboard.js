// Dashboard — live operator overview. No marketing; leads with real data.
import { icons } from '../icons.js';
import { html, $, fetchJSON, fmt, setMeter, sparkPath } from '../lib.js';
import { sensingService } from '../../services/sensing.service.js';

function bigStat(label, id, unit = '') {
  return `<div class="stat">
    <span class="stat-label">${label}</span>
    <span class="flex items-baseline gap-1"><span class="stat-value" id="${id}">—</span>
    ${unit ? `<span class="text-sm text-ink-muted">${unit}</span>` : ''}</span>
  </div>`;
}

export default {
  id: 'dashboard',
  label: 'Dashboard',
  icon: icons.dashboard,

  async mount(root) {
    root.appendChild(html`
      <section class="space-y-5">
        <!-- Presence banner -->
        <div id="presence-card" class="card card-pad flex items-center gap-4">
          <span id="presence-dot" class="dot w-3.5 h-3.5 bg-ink-4"></span>
          <div class="flex-1">
            <div id="presence-text" class="text-lg font-semibold">Waiting for data…</div>
            <div id="presence-sub" class="text-sm text-ink-muted">Connecting to sensing stream</div>
          </div>
          <span id="motion-badge" class="badge-mut">—</span>
        </div>

        <!-- Key live stats -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
          ${bigStat('People', 'stat-people')}
          ${bigStat('Confidence', 'stat-conf')}
          ${bigStat('Breathing', 'stat-br', 'bpm')}
          ${bigStat('Heart rate', 'stat-hr', 'bpm')}
        </div>

        <div class="grid gap-4 lg:grid-cols-2">
          <!-- Signal -->
          <div class="card card-pad space-y-4">
            <div class="flex items-center justify-between">
              <h2 class="card-title">Signal</h2>
              <span id="rssi-now" class="text-sm font-mono text-ink-soft">—</span>
            </div>
            <svg id="rssi-spark" viewBox="0 0 300 60" preserveAspectRatio="none" class="w-full h-16">
              <path id="rssi-path" fill="none" stroke="currentColor" class="text-brand-400" stroke-width="1.5"/>
            </svg>
            <dl class="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <div class="flex justify-between"><dt class="text-ink-muted">Variance</dt><dd id="f-var" class="font-mono">—</dd></div>
              <div class="flex justify-between"><dt class="text-ink-muted">Motion band</dt><dd id="f-motion" class="font-mono">—</dd></div>
              <div class="flex justify-between"><dt class="text-ink-muted">Breath band</dt><dd id="f-breath" class="font-mono">—</dd></div>
              <div class="flex justify-between"><dt class="text-ink-muted">Dom. freq</dt><dd id="f-freq" class="font-mono">—</dd></div>
            </dl>
          </div>

          <!-- System health -->
          <div class="card card-pad space-y-4">
            <h2 class="card-title">System</h2>
            <div class="space-y-3">
              ${['CPU', 'Memory', 'Disk'].map((m) => `
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-ink-muted">${m}</span><span id="sys-${m.toLowerCase()}-val" class="font-mono">—</span>
                  </div>
                  <div class="meter"><span id="sys-${m.toLowerCase()}" style="width:0%"></span></div>
                </div>`).join('')}
            </div>
            <div class="flex items-center justify-between pt-1 border-t border-ink-3">
              <span class="text-sm text-ink-muted">Active nodes</span>
              <a href="#nodes" class="text-sm font-semibold text-brand-300">
                <span id="nodes-active">—</span> online →
              </a>
            </div>
          </div>
        </div>
      </section>`);

    // ── Live sensing frames ──
    const off = sensingService.onData((d) => this.applyFrame(d));

    const refreshSystem = async () => {
      const m = await fetchJSON('/health/metrics');
      const sm = m?.system_metrics;
      if (!sm) return;
      const set = (k, pct) => {
        const v = (pct ?? 0) / 100;
        const valEl = $(`#sys-${k}-val`); const barEl = $(`#sys-${k}`);
        if (valEl) valEl.textContent = `${(pct ?? 0).toFixed(1)}%`;
        if (barEl) setMeter(barEl, v);
      };
      set('cpu', sm.cpu?.percent); set('memory', sm.memory?.percent); set('disk', sm.disk?.percent);
    };
    const refreshNodes = async () => {
      const n = await fetchJSON('/api/v1/nodes');
      const active = (n?.nodes || []).filter((x) => x.status === 'active').length;
      const el = $('#nodes-active'); if (el) el.textContent = String(active);
    };
    const refreshVitals = async () => {
      // Vitals may not ride every WS frame — poll the dedicated endpoint.
      const v = await fetchJSON('/api/v1/vital-signs');
      const vs = v?.vital_signs; if (!vs) return;
      if (vs.breathing_rate_bpm) $('#stat-br').textContent = fmt.num(vs.breathing_rate_bpm, 1);
      if (vs.heart_rate_bpm) $('#stat-hr').textContent = fmt.num(vs.heart_rate_bpm, 0);
    };

    refreshSystem(); refreshNodes(); refreshVitals();
    const t1 = setInterval(refreshSystem, 5000);
    const t2 = setInterval(refreshNodes, 5000);
    const t3 = setInterval(refreshVitals, 4000);

    return () => { off(); clearInterval(t1); clearInterval(t2); clearInterval(t3); };
  },

  applyFrame(d) {
    if (!d || !d.classification) return;
    const c = d.classification;
    const present = !!c.presence;
    const people = d.estimated_persons ?? (d.persons?.length) ?? (present ? 1 : 0);

    const dot = $('#presence-dot'), txt = $('#presence-text'), sub = $('#presence-sub'), mb = $('#motion-badge');
    if (dot) dot.className = `dot w-3.5 h-3.5 ${present ? 'bg-ok pulse-live' : 'bg-ink-4'}`;
    if (txt) txt.textContent = present ? (people > 1 ? `${people} people present` : 'Person present') : 'Room empty';
    if (sub) sub.textContent = d.source === 'simulated' ? 'Simulated data (no live hardware)' : `${(c.motion_level || 'unknown').replace(/_/g, ' ')} · source: ${d.source}`;
    const ml = c.motion_level || 'absent';
    if (mb) {
      const kind = ml === 'active' ? 'badge-ok' : ml.includes('still') || ml.includes('present') ? 'badge-warn' : 'badge-mut';
      mb.className = kind; mb.textContent = ml.replace(/_/g, ' ');
    }

    const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
    set('#stat-people', String(people));
    set('#stat-conf', fmt.pct(c.confidence, 0));
    if (d.vital_signs?.breathing_rate_bpm) set('#stat-br', fmt.num(d.vital_signs.breathing_rate_bpm, 1));
    if (d.vital_signs?.heart_rate_bpm) set('#stat-hr', fmt.num(d.vital_signs.heart_rate_bpm, 0));

    const f = d.features || {};
    set('#f-var', fmt.num(f.variance, 3));
    set('#f-motion', fmt.num(f.motion_band_power, 3));
    set('#f-breath', fmt.num(f.breathing_band_power, 3));
    set('#f-freq', `${fmt.num(f.dominant_freq_hz, 2)} Hz`);
    set('#rssi-now', fmt.dbm(f.mean_rssi));

    const hist = sensingService.getRssiHistory();
    const path = $('#rssi-path');
    if (path && hist.length > 1) path.setAttribute('d', sparkPath(hist, 300, 60, -90, -30));
  },
};
