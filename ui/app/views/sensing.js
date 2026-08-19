// Sensing — live CSI features, classification, vital signs + signal-field heatmap.
import { icons } from '../icons.js';
import { html, $, fmt } from '../lib.js';
import { sensingService } from '../../services/sensing.service.js';

// Teal→amber→red ramp for the field heatmap.
function heat(v) {
  v = Math.max(0, Math.min(1, v));
  const stops = [[15, 30, 38], [33, 128, 141], [245, 158, 11], [239, 68, 68]];
  const seg = v * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const t = seg - i;
  const c = (a, b) => Math.round(a + (b - a) * t);
  const [r, g, b] = [0, 1, 2].map((k) => c(stops[i][k], stops[i + 1][k]));
  return `rgb(${r},${g},${b})`;
}

export default {
  id: 'sensing',
  label: 'Sensing',
  icon: icons.sensing,

  async mount(root) {
    root.appendChild(html`
      <section class="space-y-5">
        <div class="card card-pad space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="card-title">Signal field</h2>
            <span id="sf-status" class="badge-mut">no data</span>
          </div>
          <canvas id="sf-canvas" class="w-full rounded-lg bg-ink-0 aspect-[2/1]" aria-label="Signal field heatmap"></canvas>
          <p class="text-xs text-ink-muted">CSI energy map across the sensing grid. Brighter = stronger perturbation.</p>
        </div>

        <div class="grid gap-4 sm:grid-cols-2">
          <div class="card card-pad space-y-3">
            <h2 class="card-title">Classification</h2>
            <dl class="space-y-2 text-sm" id="cls-list">
              ${[['Presence', 'cls-presence'], ['Motion level', 'cls-motion'], ['Confidence', 'cls-conf'], ['Posture', 'cls-posture'], ['Signal quality', 'cls-quality']]
                .map(([l, id]) => `<div class="flex justify-between border-b border-ink-3 pb-2 last:border-0"><dt class="text-ink-muted">${l}</dt><dd id="${id}" class="font-mono">—</dd></div>`).join('')}
            </dl>
          </div>

          <div class="card card-pad space-y-3">
            <h2 class="card-title">Vital signs</h2>
            <div class="grid grid-cols-2 gap-3">
              <div class="stat"><span class="stat-label">Breathing</span><span class="stat-value" id="vs-br">—</span><span class="text-xs text-ink-muted" id="vs-br-c">confidence —</span></div>
              <div class="stat"><span class="stat-label">Heart rate</span><span class="stat-value" id="vs-hr">—</span><span class="text-xs text-ink-muted" id="vs-hr-c">confidence —</span></div>
            </div>
            <div class="text-xs text-ink-muted" id="vs-buf">buffer —</div>
          </div>
        </div>

        <div class="card card-pad space-y-3">
          <h2 class="card-title">CSI features</h2>
          <dl class="grid grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-3 text-sm">
            ${[['Mean RSSI', 'ft-rssi'], ['Variance', 'ft-var'], ['Motion band', 'ft-motion'], ['Breath band', 'ft-breath'], ['Dominant freq', 'ft-freq'], ['Change points', 'ft-cp'], ['Spectral power', 'ft-spec'], ['Tick', 'ft-tick']]
              .map(([l, id]) => `<div><dt class="text-ink-muted text-xs">${l}</dt><dd id="${id}" class="font-mono">—</dd></div>`).join('')}
          </dl>
        </div>
      </section>`);

    const canvas = $('#sf-canvas');
    const ctx = canvas.getContext('2d');
    const resize = () => { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; };
    resize();
    window.addEventListener('resize', resize);

    const draw = (field) => {
      if (!field?.values?.length || !field.grid_size) return;
      const [gx, , gz] = field.grid_size;
      const cols = gx || Math.sqrt(field.values.length) | 0;
      const rows = gz || (field.values.length / cols) | 0;
      const cw = canvas.width / cols, ch = canvas.height / rows;
      for (let z = 0; z < rows; z++) {
        for (let x = 0; x < cols; x++) {
          ctx.fillStyle = heat(field.values[z * cols + x] || 0);
          ctx.fillRect(x * cw, z * ch, cw + 1, ch + 1);
        }
      }
    };

    const off = sensingService.onData((d) => {
      const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
      const c = d.classification || {}, f = d.features || {};
      const present = !!c.presence;
      const pEl = $('#cls-presence');
      if (pEl) { pEl.textContent = present ? 'PRESENT' : 'empty'; pEl.className = `font-mono ${present ? 'text-ok' : 'text-ink-muted'}`; }
      set('#cls-motion', (c.motion_level || '—').replace(/_/g, ' '));
      set('#cls-conf', fmt.pct(c.confidence, 0));
      set('#cls-posture', d.posture || '—');
      set('#cls-quality', d.quality_verdict || (d.signal_quality_score != null ? fmt.num(d.signal_quality_score, 2) : '—'));

      set('#ft-rssi', fmt.dbm(f.mean_rssi)); set('#ft-var', fmt.num(f.variance, 4));
      set('#ft-motion', fmt.num(f.motion_band_power, 4)); set('#ft-breath', fmt.num(f.breathing_band_power, 4));
      set('#ft-freq', `${fmt.num(f.dominant_freq_hz, 2)} Hz`); set('#ft-cp', fmt.int(f.change_points));
      set('#ft-spec', fmt.num(f.spectral_power, 4)); set('#ft-tick', fmt.int(d.tick));

      if (d.vital_signs) {
        const vs = d.vital_signs;
        set('#vs-br', vs.breathing_rate_bpm ? fmt.num(vs.breathing_rate_bpm, 1) : '—');
        set('#vs-hr', vs.heart_rate_bpm ? fmt.num(vs.heart_rate_bpm, 0) : '—');
        set('#vs-br-c', `confidence ${fmt.pct(vs.breathing_confidence, 0)}`);
        set('#vs-hr-c', `confidence ${fmt.pct(vs.heartbeat_confidence, 0)}`);
      }

      const st = $('#sf-status');
      if (st) { st.textContent = d.source === 'simulated' ? 'simulated' : 'live'; st.className = d.source === 'simulated' ? 'badge-warn' : 'badge-ok'; }
      draw(d.signal_field);
    });

    return () => { off(); window.removeEventListener('resize', resize); };
  },
};
