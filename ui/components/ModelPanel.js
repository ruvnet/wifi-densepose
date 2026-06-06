// ModelPanel Component for WiFi-DensePose UI
// Dark-mode panel for model management: listing, loading, LoRA profiles.

import { modelService } from '../services/model.service.js';

const MP_STYLES = `
.mp-panel{background:var(--color-surface);border:1px solid var(--color-card-border);border-radius:8px;color:var(--color-text);overflow:hidden}
.mp-header{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;background:var(--color-background);border-bottom:1px solid var(--color-card-border-inner)}
.mp-title{font-size:14px;font-weight:650;color:var(--color-text)}
.mp-badge{background:var(--color-secondary);color:var(--color-text-secondary);font-size:11px;font-weight:650;padding:3px 9px;border-radius:999px;border:1px solid var(--color-border)}
.mp-error{background:rgba(var(--color-error-rgb),.1);color:var(--color-error);border:1px solid rgba(var(--color-error-rgb),.28);border-radius:6px;padding:9px 12px;margin:12px 14px 0;font-size:12px}
.mp-active-card{margin:14px 16px;padding:12px;background:var(--color-background);border:1px solid var(--color-card-border-inner);border-left:4px solid var(--color-success);border-radius:6px}
.mp-active-name{font-size:14px;font-weight:650;color:var(--color-text);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mp-active-meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.mp-active-stats{font-size:12px;color:var(--color-text-secondary);margin-bottom:10px}
.mp-stat-label{color:var(--color-text-secondary)}.mp-stat-value{color:var(--color-text);font-weight:650}.mp-stat-sep{color:var(--color-border);margin:0 6px}
.mp-lora-row{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.mp-lora-label{font-size:12px;color:var(--color-text-secondary)}
.mp-lora-select{flex:1;padding:8px 10px;background:var(--color-background);border:1px solid var(--color-border);border-radius:6px;color:var(--color-text);font-size:12px;min-width:0}
.mp-list-section{padding:0 16px 14px}
.mp-section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0;color:var(--color-text-secondary);padding:12px 0 8px}
.mp-model-card{padding:10px;margin-bottom:8px;background:var(--color-background);border:1px solid var(--color-card-border-inner);border-radius:6px;transition:border-color .2s,background .2s}
.mp-model-card:hover{border-color:var(--color-primary)}
.mp-card-name{font-size:13px;font-weight:600;color:var(--color-text);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mp-card-meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.mp-meta-tag{background:var(--color-secondary);color:var(--color-text-secondary);font-size:10px;padding:3px 7px;border-radius:999px;border:1px solid var(--color-border)}
.mp-card-actions{display:flex;gap:8px;flex-wrap:wrap}
.mp-empty{color:var(--color-text-secondary);font-size:12px;padding:18px 0;text-align:center;line-height:1.45}
.mp-readme{margin:12px 16px;padding:10px 12px;background:var(--color-background);border:1px solid var(--color-card-border-inner);border-radius:6px;color:var(--color-text-secondary);font-size:12px;line-height:1.45}
.mp-readme strong{color:var(--color-text)}
.mp-footer{padding:12px 16px;border-top:1px solid var(--color-card-border-inner);display:flex;justify-content:flex-end}
.mp-btn{padding:7px 12px;border-radius:6px;font-size:12px;font-weight:650;cursor:pointer;border:1px solid transparent;transition:background .15s,border-color .15s,color .15s}
.mp-btn:disabled{opacity:.5;cursor:not-allowed}
.mp-btn-success{background:var(--color-primary);color:var(--color-btn-primary-text);border-color:var(--color-primary)}
.mp-btn-success:hover:not(:disabled){background:var(--color-primary-hover)}
.mp-btn-danger{background:rgba(var(--color-error-rgb),.1);color:var(--color-error);border-color:rgba(var(--color-error-rgb),.28)}
.mp-btn-danger:hover:not(:disabled){background:rgba(var(--color-error-rgb),.18)}
.mp-btn-secondary{background:var(--color-secondary);color:var(--color-text);border-color:var(--color-border)}
.mp-btn-secondary:hover:not(:disabled){background:var(--color-secondary-hover)}
.mp-btn-muted{background:transparent;color:var(--color-text-secondary);border-color:var(--color-border);font-size:11px;padding:5px 9px}
.mp-btn-muted:hover:not(:disabled){color:var(--color-error);border-color:rgba(var(--color-error-rgb),.35)}
`;

export default class ModelPanel {
  constructor(container) {
    this.container = typeof container === 'string'
      ? document.getElementById(container) : container;
    if (!this.container) throw new Error('ModelPanel: container element not found');

    this.state = { models: [], activeModel: null, loraProfiles: [], loading: false, error: null };
    this.unsubs = [];
    this._injectStyles();
    this.render();
    this.refresh();
    this.unsubs.push(
      modelService.on('model-loaded', () => this.refresh()),
      modelService.on('model-unloaded', () => this.refresh()),
      modelService.on('lora-activated', () => this.refresh())
    );
  }

  // --- Data ---

  async refresh() {
    this._set({ loading: true, error: null });
    try {
      const [listRes, active] = await Promise.all([
        modelService.listModels().catch(() => ({ models: [] })),
        modelService.getActiveModel().catch(() => null)
      ]);
      let lora = [];
      if (active) lora = await modelService.getLoraProfiles().catch(() => []);
      this._set({ models: listRes?.models ?? [], activeModel: active, loraProfiles: lora, loading: false });
    } catch (e) { this._set({ loading: false, error: e.message }); }
  }

  // --- Actions ---

  async _load(id) {
    this._set({ loading: true, error: null });
    try { await modelService.loadModel(id); await this.refresh(); }
    catch (e) { this._set({ loading: false, error: `Load failed: ${e.message}` }); }
  }

  async _unload() {
    this._set({ loading: true, error: null });
    try { await modelService.unloadModel(); await this.refresh(); }
    catch (e) { this._set({ loading: false, error: `Unload failed: ${e.message}` }); }
  }

  async _delete(id) {
    this._set({ loading: true, error: null });
    try { await modelService.deleteModel(id); await this.refresh(); }
    catch (e) { this._set({ loading: false, error: `Delete failed: ${e.message}` }); }
  }

  async _loraChange(modelId, profile) {
    if (!profile) return;
    this._set({ loading: true, error: null });
    try { await modelService.activateLoraProfile(modelId, profile); await this.refresh(); }
    catch (e) { this._set({ loading: false, error: `LoRA failed: ${e.message}` }); }
  }

  _set(p) { Object.assign(this.state, p); this.render(); }

  // --- Render ---

  render() {
    const el = this.container;
    el.innerHTML = '';
    const panel = this._el('div', 'mp-panel');

    // Header
    const hdr = this._el('div', 'mp-header');
    hdr.appendChild(this._el('span', 'mp-title', 'Model Library'));
    hdr.appendChild(this._el('span', 'mp-badge', String(this.state.models.length)));
    panel.appendChild(hdr);
    const readme = this._el('div', 'mp-readme');
    readme.innerHTML = '<strong>How data is used:</strong> .rvf files are loaded as inference models, LoRA profiles adjust the active model, and live features are evaluated only after a model is loaded.';
    panel.appendChild(readme);

    if (this.state.error) panel.appendChild(this._el('div', 'mp-error', this.state.error));

    // Active model
    if (this.state.activeModel) panel.appendChild(this._renderActive());

    // List
    const ls = this._el('div', 'mp-list-section');
    ls.appendChild(this._el('div', 'mp-section-title', 'Available Models'));
    const models = this.state.models.filter(
      m => !(this.state.activeModel && this.state.activeModel.model_id === m.id)
    );
    if (models.length === 0 && !this.state.loading) {
      ls.appendChild(this._el('div', 'mp-empty', 'No .rvf models found. Train a model or place .rvf files in data/models/'));
    } else {
      models.forEach(m => ls.appendChild(this._renderCard(m)));
    }
    panel.appendChild(ls);

    // Footer
    const ft = this._el('div', 'mp-footer');
    const rb = this._btn('Refresh', 'mp-btn mp-btn-secondary', () => this.refresh());
    rb.disabled = this.state.loading;
    ft.appendChild(rb);
    panel.appendChild(ft);

    el.appendChild(panel);
  }

  _renderActive() {
    const am = this.state.activeModel;
    const card = this._el('div', 'mp-active-card');
    card.appendChild(this._el('div', 'mp-active-name', am.model_id || 'Active Model'));

    const full = this.state.models.find(m => m.id === am.model_id);
    if (full) {
      const meta = this._el('div', 'mp-active-meta');
      if (full.version) meta.appendChild(this._tag('v' + full.version));
      if (full.pck_score != null) meta.appendChild(this._tag('PCK ' + (full.pck_score * 100).toFixed(1) + '%'));
      card.appendChild(meta);
    }

    if (am.avg_inference_ms != null) {
      const st = this._el('div', 'mp-active-stats');
      st.innerHTML = `<span class="mp-stat-label">Inference:</span> <span class="mp-stat-value">${am.avg_inference_ms.toFixed(1)} ms</span><span class="mp-stat-sep">|</span><span class="mp-stat-label">Frames:</span> <span class="mp-stat-value">${am.frames_processed ?? 0}</span>`;
      card.appendChild(st);
    }

    if (this.state.loraProfiles.length > 0) {
      const row = this._el('div', 'mp-lora-row');
      row.appendChild(this._el('span', 'mp-lora-label', 'LoRA Profile:'));
      const sel = document.createElement('select');
      sel.className = 'mp-lora-select';
      const def = document.createElement('option');
      def.value = ''; def.textContent = '-- none --'; sel.appendChild(def);
      this.state.loraProfiles.forEach(p => {
        const o = document.createElement('option');
        o.value = p; o.textContent = p; sel.appendChild(o);
      });
      sel.addEventListener('change', () => this._loraChange(am.model_id, sel.value));
      row.appendChild(sel);
      card.appendChild(row);
    }

    const ub = this._btn('Unload', 'mp-btn mp-btn-danger', () => this._unload());
    ub.disabled = this.state.loading;
    card.appendChild(ub);
    return card;
  }

  _renderCard(model) {
    const card = this._el('div', 'mp-model-card');
    card.appendChild(this._el('div', 'mp-card-name', model.filename || model.id));
    const meta = this._el('div', 'mp-card-meta');
    if (model.version) meta.appendChild(this._tag('v' + model.version));
    if (model.size_bytes != null) meta.appendChild(this._tag(this._fmtB(model.size_bytes)));
    if (model.pck_score != null) meta.appendChild(this._tag('PCK ' + (model.pck_score * 100).toFixed(1) + '%'));
    if (model.lora_profiles && model.lora_profiles.length > 0) meta.appendChild(this._tag(model.lora_profiles.length + ' LoRA'));
    card.appendChild(meta);

    const acts = this._el('div', 'mp-card-actions');
    const lb = this._btn('Load', 'mp-btn mp-btn-success', () => this._load(model.id));
    lb.disabled = this.state.loading;
    const db = this._btn('Delete', 'mp-btn mp-btn-muted', () => this._delete(model.id));
    db.disabled = this.state.loading;
    acts.appendChild(lb); acts.appendChild(db);
    card.appendChild(acts);
    return card;
  }

  // --- Helpers ---

  _el(tag, cls, txt) { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }
  _btn(txt, cls, fn) { const b = document.createElement('button'); b.className = cls; b.textContent = txt; b.addEventListener('click', fn); return b; }
  _tag(txt) { return this._el('span', 'mp-meta-tag', txt); }
  _fmtB(b) { return b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(1) + ' KB' : (b / 1048576).toFixed(1) + ' MB'; }

  _injectStyles() {
    if (document.getElementById('model-panel-styles')) return;
    const s = document.createElement('style');
    s.id = 'model-panel-styles';
    s.textContent = MP_STYLES;
    document.head.appendChild(s);
  }

  destroy() {
    this.unsubs.forEach(fn => fn());
    this.unsubs = [];
    if (this.container) this.container.innerHTML = '';
  }

  dispose() {
    this.destroy();
  }
}
