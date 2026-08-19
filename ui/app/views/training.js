// Training — models, recordings and training-run control. Wired to /api/v1/*.
import { icons } from '../icons.js';
import { html, $, fetchJSON, fmt, toast } from '../lib.js';

async function post(url, body) {
  return fetchJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
}
async function del(url) { return fetchJSON(url, { method: 'DELETE' }); }

function modelCard(m, activeId) {
  const id = m.id ?? m.name ?? '?';
  const isActive = activeId && id === activeId;
  const meta = [m.kind, m.size_mb ? `${fmt.num(m.size_mb, 1)} MB` : null, m.created].filter(Boolean).join(' · ');
  return `<div class="flex items-center gap-3 rounded-lg bg-ink-1 border border-ink-3 p-3">
    <div class="flex-1 min-w-0">
      <div class="font-medium truncate">${id} ${isActive ? '<span class="badge-ok ml-1">active</span>' : ''}</div>
      <div class="text-xs text-ink-muted truncate">${meta || '—'}</div>
    </div>
    ${isActive
      ? `<button data-act="unload" class="btn-ghost !py-1.5 !px-3 text-xs">Unload</button>`
      : `<button data-act="load" data-id="${id}" class="btn-primary !py-1.5 !px-3 text-xs">Load</button>`}
    <button data-act="del-model" data-id="${id}" class="btn-danger !py-1.5 !px-3 text-xs" aria-label="Delete model">✕</button>
  </div>`;
}

export default {
  id: 'training',
  label: 'Training',
  icon: icons.training,

  async mount(root) {
    root.appendChild(html`
      <section class="space-y-5">
        <!-- Training run -->
        <div class="card card-pad space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="card-title">Training run</h2>
            <span id="tr-status" class="badge-mut">—</span>
          </div>
          <div class="flex gap-2">
            <button id="tr-start" class="btn-primary flex-1">Start training</button>
            <button id="tr-stop" class="btn-ghost flex-1">Stop</button>
          </div>
          <pre id="tr-config" class="text-xs font-mono text-ink-muted whitespace-pre-wrap break-words max-h-32 overflow-auto"></pre>
        </div>

        <!-- Recording -->
        <div class="card card-pad space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="card-title">CSI recording</h2>
            <span id="rec-state" class="badge-mut">idle</span>
          </div>
          <div class="flex gap-2">
            <input id="rec-id" placeholder="recording id (optional)" class="flex-1 rounded-lg bg-ink-1 border border-ink-3 px-3 py-2.5 text-sm focus-visible:ring-2 focus-visible:ring-brand-400" />
            <button id="rec-start" class="btn-primary">Record</button>
            <button id="rec-stop" class="btn-ghost">Stop</button>
          </div>
          <div id="rec-list" class="space-y-2"></div>
        </div>

        <!-- Models -->
        <div class="card card-pad space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="card-title">Models</h2>
            <button id="m-refresh" class="btn-ghost !py-1.5 !px-3 text-xs">Refresh</button>
          </div>
          <div id="m-list" class="space-y-2"><div class="text-sm text-ink-muted">Loading…</div></div>
        </div>
      </section>`);

    // ── Models ──
    const loadModels = async () => {
      const [list, active] = await Promise.all([fetchJSON('/api/v1/models'), fetchJSON('/api/v1/models/active')]);
      const models = list?.models || [];
      const activeId = active?.active?.id ?? null;
      const box = $('#m-list');
      box.innerHTML = models.length ? models.map((m) => modelCard(m, activeId)).join('')
        : '<div class="text-sm text-ink-muted">No models found. Record CSI data and train a model, or drop an .rvf file in the models directory.</div>';
    };
    $('#m-list').addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-act]'); if (!btn) return;
      const { act, id } = btn.dataset;
      if (act === 'load') { await post('/api/v1/models/load', { id }); toast(`Loading ${id}`, 'ok'); }
      else if (act === 'unload') { await post('/api/v1/models/unload'); toast('Model unloaded', 'ok'); }
      else if (act === 'del-model') { if (!confirm(`Delete model ${id}?`)) return; await del(`/api/v1/models/${encodeURIComponent(id)}`); toast(`Deleted ${id}`, 'warn'); }
      loadModels();
    });
    $('#m-refresh').addEventListener('click', loadModels);

    // ── Recording ──
    const loadRecordings = async () => {
      const r = await fetchJSON('/api/v1/recording/list');
      const recs = r?.recordings || [];
      const box = $('#rec-list');
      box.innerHTML = recs.length ? recs.map((rec) => {
        const id = rec.id ?? rec.name ?? rec;
        return `<div class="flex items-center gap-3 rounded-lg bg-ink-1 border border-ink-3 p-2.5 text-sm">
          <span class="flex-1 truncate font-mono">${id}</span>
          ${rec.size_mb ? `<span class="text-xs text-ink-muted">${fmt.num(rec.size_mb, 1)} MB</span>` : ''}
          <button data-rid="${id}" class="btn-danger !py-1 !px-2.5 text-xs">✕</button>
        </div>`;
      }).join('') : '<div class="text-sm text-ink-muted">No recordings yet.</div>';
    };
    $('#rec-list').addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-rid]'); if (!btn) return;
      if (!confirm(`Delete recording ${btn.dataset.rid}?`)) return;
      await del(`/api/v1/recording/${encodeURIComponent(btn.dataset.rid)}`); toast('Recording deleted', 'warn'); loadRecordings();
    });
    $('#rec-start').addEventListener('click', async () => {
      const id = $('#rec-id').value.trim();
      const r = await post('/api/v1/recording/start', id ? { id } : {});
      if (r?.success === false) toast(r.error || 'Recording already running', 'warn');
      else { toast('Recording started', 'ok'); $('#rec-state').textContent = 'recording'; $('#rec-state').className = 'badge-bad'; }
      loadRecordings();
    });
    $('#rec-stop').addEventListener('click', async () => {
      await post('/api/v1/recording/stop'); toast('Recording stopped', 'ok');
      $('#rec-state').textContent = 'idle'; $('#rec-state').className = 'badge-mut'; loadRecordings();
    });

    // ── Training ──
    const loadTrain = async () => {
      const t = await fetchJSON('/api/v1/train/status');
      const st = $('#tr-status');
      const status = t?.status || 'idle';
      st.textContent = status;
      st.className = /run|train|active/i.test(String(status)) ? 'badge-ok' : 'badge-mut';
      $('#tr-config').textContent = t?.config ? JSON.stringify(t.config, null, 2) : '';
    };
    $('#tr-start').addEventListener('click', async () => { await post('/api/v1/train/start'); toast('Training started', 'ok'); loadTrain(); });
    $('#tr-stop').addEventListener('click', async () => { await post('/api/v1/train/stop'); toast('Training stopped', 'warn'); loadTrain(); });

    loadModels(); loadRecordings(); loadTrain();
    const t = setInterval(loadTrain, 4000);
    return () => clearInterval(t);
  },
};
