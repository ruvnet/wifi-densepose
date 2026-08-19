// Nodes — per-node CSI sensor health, mesh status, hardware spec.
import { icons } from '../icons.js';
import { html, $, fetchJSON, fmt } from '../lib.js';

function nodeRow(n) {
  const active = n.status === 'active';
  return `<tr class="border-b border-ink-3 last:border-0">
    <td class="py-2.5 pr-3 font-mono">#${n.node_id}</td>
    <td class="py-2.5 pr-3"><span class="${active ? 'badge-ok' : 'badge-bad'}"><span class="dot ${active ? 'bg-ok' : 'bg-bad'}"></span>${n.status}</span></td>
    <td class="py-2.5 pr-3 font-mono text-right">${fmt.dbm(n.rssi_dbm)}</td>
    <td class="py-2.5 pr-3 text-ink-soft">${(n.motion_level || '—').replace(/_/g, ' ')}</td>
    <td class="py-2.5 pr-3 text-right font-mono">${n.person_count ?? 0}</td>
    <td class="py-2.5 text-right text-ink-muted">${fmt.ago((n.last_seen_ms ?? 0) / 1000)}</td>
  </tr>`;
}

export default {
  id: 'nodes',
  label: 'Nodes',
  icon: icons.nodes,

  async mount(root) {
    root.appendChild(html`
      <section class="space-y-5">
        <div class="grid grid-cols-3 gap-3">
          <div class="stat"><span class="stat-label">Total nodes</span><span class="stat-value" id="n-total">—</span></div>
          <div class="stat"><span class="stat-label">Online</span><span class="stat-value text-ok" id="n-online">—</span></div>
          <div class="stat"><span class="stat-label">Stale</span><span class="stat-value text-warn" id="n-stale">—</span></div>
        </div>

        <div class="card card-pad space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="card-title">Sensor nodes</h2>
            <button id="n-refresh" class="btn-ghost !py-1.5 !px-3 text-xs">Refresh</button>
          </div>
          <div class="overflow-x-auto -mx-1">
            <table class="w-full text-sm min-w-[480px]">
              <thead><tr class="text-left text-xs uppercase tracking-wide text-ink-muted border-b border-ink-3">
                <th class="py-2 pr-3 font-medium">Node</th><th class="py-2 pr-3 font-medium">Status</th>
                <th class="py-2 pr-3 font-medium text-right">RSSI</th><th class="py-2 pr-3 font-medium">Motion</th>
                <th class="py-2 pr-3 font-medium text-right">People</th><th class="py-2 font-medium text-right">Last seen</th>
              </tr></thead>
              <tbody id="n-body"><tr><td colspan="6" class="py-6 text-center text-ink-muted">Loading…</td></tr></tbody>
            </table>
          </div>
        </div>

        <div class="grid gap-4 sm:grid-cols-2">
          <div class="card card-pad space-y-2">
            <h2 class="card-title">Mesh</h2>
            <pre id="mesh-box" class="text-xs font-mono text-ink-soft whitespace-pre-wrap break-words">—</pre>
          </div>
          <div class="card card-pad space-y-2">
            <h2 class="card-title">Hardware reference</h2>
            <dl class="text-sm space-y-2">
              ${[['Node chip', 'ESP32-S3 / C6'], ['Band', '2.4 GHz WiFi CSI'], ['Subcarriers', 'up to 114'], ['Sample rate', '~100 Hz'], ['mmWave option', 'Seeed MR60BHA2 (60 GHz)']]
                .map(([k, v]) => `<div class="flex justify-between border-b border-ink-3 pb-2 last:border-0"><dt class="text-ink-muted">${k}</dt><dd class="font-mono text-right">${v}</dd></div>`).join('')}
            </dl>
          </div>
        </div>
      </section>`);

    const refresh = async () => {
      const data = await fetchJSON('/api/v1/nodes');
      const body = $('#n-body');
      const list = data?.nodes || [];
      const online = list.filter((n) => n.status === 'active').length;
      $('#n-total').textContent = data?.total ?? list.length;
      $('#n-online').textContent = online;
      $('#n-stale').textContent = list.length - online;
      body.innerHTML = list.length
        ? list.map(nodeRow).join('')
        : '<tr><td colspan="6" class="py-6 text-center text-ink-muted">No nodes reporting. Power on an ESP32 CSI node and provision it to this server.</td></tr>';
    };
    const refreshMesh = async () => {
      const m = await fetchJSON('/api/v1/mesh');
      const el = $('#mesh-box');
      if (el) el.textContent = m ? JSON.stringify(m, null, 2) : 'Mesh data unavailable';
    };

    refresh(); refreshMesh();
    $('#n-refresh').addEventListener('click', refresh);
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  },
};
