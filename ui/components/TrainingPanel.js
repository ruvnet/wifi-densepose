// TrainingPanel Component for WiFi-DensePose UI
// Dark-mode panel for training management, CSI recordings, and progress charts.

import { trainingService } from '../services/training.service.js';

const TP_STYLES = `
.tp-panel{background:var(--color-surface);border:1px solid var(--color-card-border);border-radius:8px;color:var(--color-text);overflow:hidden}
.tp-header{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--color-card-border-inner);background:var(--color-background)}
.tp-title{font-size:14px;font-weight:650;color:var(--color-text)}
.tp-badge{font-size:11px;font-weight:650;padding:3px 9px;border-radius:999px;border:1px solid var(--color-border);text-transform:uppercase}
.tp-badge-idle{background:var(--color-secondary);color:var(--color-text-secondary)}
.tp-badge-active{background:rgba(var(--color-success-rgb),.14);color:var(--color-success);border-color:rgba(var(--color-success-rgb),.3);animation:tp-pulse 1.5s ease-in-out infinite}
.tp-badge-done{background:rgba(var(--color-primary-rgb,33,128,141),.14);color:var(--color-primary);border-color:rgba(var(--color-primary-rgb,33,128,141),.3)}
@keyframes tp-pulse{0%,100%{opacity:1}50%{opacity:.65}}
.tp-error{background:rgba(var(--color-error-rgb),.1);color:var(--color-error);border:1px solid rgba(var(--color-error-rgb),.28);border-radius:6px;padding:9px 12px;margin:12px 14px 0;font-size:12px}
.tp-section{padding:14px 16px;border-bottom:1px solid var(--color-card-border-inner)}
.tp-section:last-child{border-bottom:none}
.tp-section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0;color:var(--color-text-secondary);margin-bottom:10px}
.tp-readme{display:grid;gap:10px;padding:14px 16px;border-bottom:1px solid var(--color-card-border-inner);background:var(--color-background)}
.tp-readme-title{font-size:13px;font-weight:700;color:var(--color-text)}
.tp-readme-copy{font-size:12px;line-height:1.45;color:var(--color-text-secondary);margin:0}
.tp-readme-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:0;padding:0;list-style:none}
.tp-readme-list li{font-size:12px;color:var(--color-text);background:var(--color-surface);border:1px solid var(--color-card-border-inner);border-radius:6px;padding:8px 9px;min-width:0}
.tp-empty{color:var(--color-text-secondary);font-size:12px;padding:14px 0;text-align:center}
.tp-rec-row{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:10px;padding:9px 10px;margin-bottom:8px;background:var(--color-background);border:1px solid var(--color-card-border-inner);border-radius:6px}
.tp-rec-info{display:flex;flex-direction:column;gap:2px;min-width:0}
.tp-rec-name{font-size:12px;color:var(--color-text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tp-rec-meta{font-size:11px;color:var(--color-text-secondary)}
.tp-rec-actions{margin-top:10px}
.tp-config-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.tp-config-form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 12px}
.tp-label{font-size:12px;color:var(--color-text-secondary);display:block;margin-bottom:4px}
.tp-input-row{display:flex;flex-direction:column;gap:4px}
.tp-input{width:100%;padding:8px 10px;background:var(--color-background);border:1px solid var(--color-border);border-radius:6px;color:var(--color-text);font-size:13px}
.tp-input:focus{outline:none;border-color:var(--color-primary);box-shadow:var(--focus-ring)}
.tp-ds-container{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:2px;max-height:132px;overflow-y:auto}
.tp-ds-item{display:flex;align-items:center;gap:8px;padding:7px 9px;border:1px solid var(--color-card-border-inner);border-radius:6px;background:var(--color-background);font-size:12px;color:var(--color-text);cursor:pointer;min-width:0}
.tp-ds-item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tp-ds-item input{width:14px;height:14px;flex:0 0 auto}
.tp-train-actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.tp-progress-bar{height:8px;background:var(--color-secondary);border-radius:999px;overflow:hidden;margin-bottom:6px}
.tp-progress-fill{height:100%;background:var(--color-primary);border-radius:999px;transition:width .3s}
.tp-progress-label{font-size:12px;color:var(--color-text-secondary);text-align:right;margin-bottom:12px}
.tp-chart-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:12px}
.tp-chart-row canvas{border:1px solid var(--color-card-border-inner);border-radius:6px;width:100%;min-width:0;background:var(--color-background)}
.tp-metrics-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.tp-metric-cell{background:var(--color-background);border:1px solid var(--color-card-border-inner);border-radius:6px;padding:8px 10px;min-width:0}
.tp-metric-label{font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:0}
.tp-metric-value{font-size:14px;color:var(--color-text);font-weight:650;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tp-btn{padding:7px 12px;border-radius:6px;font-size:12px;font-weight:650;cursor:pointer;border:1px solid transparent;transition:background .15s,border-color .15s,color .15s}
.tp-btn:disabled{opacity:.5;cursor:not-allowed}
.tp-btn-success{background:var(--color-primary);color:var(--color-btn-primary-text);border-color:var(--color-primary)}
.tp-btn-success:hover:not(:disabled){background:var(--color-primary-hover)}
.tp-btn-danger{background:rgba(var(--color-error-rgb),.1);color:var(--color-error);border-color:rgba(var(--color-error-rgb),.28)}
.tp-btn-danger:hover:not(:disabled){background:rgba(var(--color-error-rgb),.18)}
.tp-btn-secondary,.tp-btn-rec{background:var(--color-secondary);color:var(--color-text);border-color:var(--color-border)}
.tp-btn-secondary:hover:not(:disabled),.tp-btn-rec:hover:not(:disabled){background:var(--color-secondary-hover)}
.tp-btn-muted{background:transparent;color:var(--color-text-secondary);border-color:var(--color-border);font-size:11px;padding:4px 8px}
.tp-btn-muted:hover:not(:disabled){color:var(--color-text);border-color:var(--color-primary)}
@media(max-width:760px){.tp-config-form,.tp-chart-row,.tp-metrics-grid,.tp-readme-list{grid-template-columns:1fr}.tp-progress-label{text-align:left}}
`;

export default class TrainingPanel {
  constructor(container) {
    this.container = typeof container === 'string'
      ? document.getElementById(container) : container;
    if (!this.container) throw new Error('TrainingPanel: container element not found');

    this.state = {
      recordings: [], trainingStatus: null, isRecording: false,
      configOpen: true, loading: false, error: null
    };
    this.config = {
      epochs: 100, batch_size: 32, learning_rate: 3e-4, patience: 15,
      selectedRecordings: [], base_model: '', lora_profile_name: ''
    };
    this.progressData = { losses: [], pcks: [] };
    this.unsubscribers = [];
    this._injectStyles();
    this.render();
    this.refresh();
    this._bindEvents();
  }

  _bindEvents() {
    this.unsubscribers.push(
      trainingService.on('progress', (d) => this._onProgress(d)),
      trainingService.on('training-started', () => this.refresh()),
      trainingService.on('training-stopped', () => {
        trainingService.disconnectProgressStream();
        this.refresh();
      })
    );
  }

  _onProgress(data) {
    if (data.train_loss != null) this.progressData.losses.push(data.train_loss);
    if (data.val_pck != null) this.progressData.pcks.push(data.val_pck);
    this._set({ trainingStatus: { ...this.state.trainingStatus, ...data } });
  }

  // --- Data ---

  async refresh() {
    this._set({ loading: true, error: null });
    try {
      const [recordings, status] = await Promise.all([
        trainingService.listRecordings().catch(() => []),
        trainingService.getTrainingStatus().catch(() => null)
      ]);
      if (status && !status.active) this.progressData = { losses: [], pcks: [] };
      this._set({ recordings, trainingStatus: status, loading: false });
    } catch (e) { this._set({ loading: false, error: e.message }); }
  }

  // --- Actions ---

  async _startRec() {
    this._set({ loading: true, error: null });
    try {
      await trainingService.startRecording({ session_name: `rec_${Date.now()}`, label: 'pose' });
      this._set({ isRecording: true, loading: false });
      await this.refresh();
    } catch (e) { this._set({ loading: false, error: `Recording failed: ${e.message}` }); }
  }

  async _stopRec() {
    this._set({ loading: true, error: null });
    try {
      await trainingService.stopRecording();
      this._set({ isRecording: false, loading: false });
      await this.refresh();
    } catch (e) { this._set({ loading: false, error: `Stop recording failed: ${e.message}` }); }
  }

  async _delRec(id) {
    this._set({ loading: true, error: null });
    try {
      await trainingService.deleteRecording(id);
      this.config.selectedRecordings = this.config.selectedRecordings.filter(r => r !== id);
      await this.refresh();
    } catch (e) { this._set({ loading: false, error: `Delete failed: ${e.message}` }); }
  }

  async _launchTraining(method, extraCfg = {}) {
    this._set({ loading: true, error: null });
    this.progressData = { losses: [], pcks: [] };
    try {
      const payload = {
        dataset_ids: this.config.selectedRecordings,
        config: {
          epochs: this.config.epochs,
          batch_size: this.config.batch_size,
          learning_rate: this.config.learning_rate,
          ...extraCfg
        }
      };
      const data = await trainingService[method](payload);
      if (data?.active !== false && data?.status !== 'completed') {
        trainingService.connectProgressStream();
      }
      await this.refresh();
    } catch (e) { this._set({ loading: false, error: `Training failed: ${e.message}` }); }
  }

  async _stopTraining() {
    this._set({ loading: true, error: null });
    try { await trainingService.stopTraining(); await this.refresh(); }
    catch (e) { this._set({ loading: false, error: `Stop failed: ${e.message}` }); }
  }

  _set(p) { Object.assign(this.state, p); this.render(); }

  // --- Render ---

  render() {
    const el = this.container;
    el.innerHTML = '';
    const panel = this._el('div', 'tp-panel');
    panel.appendChild(this._renderHeader());
    panel.appendChild(this._renderReadme());
    if (this.state.error) panel.appendChild(this._el('div', 'tp-error', this.state.error));
    panel.appendChild(this._renderRecordings());
    const ts = this.state.trainingStatus;
    const active = ts && ts.active;
    if (active) panel.appendChild(this._renderProgress());
    else if (ts && !ts.active && this.progressData.losses.length > 0) panel.appendChild(this._renderComplete());
    else panel.appendChild(this._renderConfig());
    el.appendChild(panel);
    if (active) requestAnimationFrame(() => this._drawCharts());
  }

  _renderHeader() {
    const h = this._el('div', 'tp-header');
    h.appendChild(this._el('span', 'tp-title', 'Training'));
    const ts = this.state.trainingStatus;
    let cls = 'tp-badge tp-badge-idle', txt = 'Idle';
    if (ts && ts.active) { cls = 'tp-badge tp-badge-active'; txt = 'Training'; }
    else if (ts && !ts.active && this.progressData.losses.length > 0) { cls = 'tp-badge tp-badge-done'; txt = 'Completed'; }
    h.appendChild(this._el('span', cls, txt));
    return h;
  }

  _renderReadme() {
    const s = this._el('div', 'tp-readme');
    s.appendChild(this._el('div', 'tp-readme-title', 'Training README'));
    s.appendChild(this._el('p', 'tp-readme-copy',
      'This page turns live RuView CSI packets into reusable datasets, then uses those datasets to request pose-model training and manage exported .rvf model files.'
    ));
    const list = this._el('ul', 'tp-readme-list');
    [
      'Record: capture live CSI frames into data/recordings as .csi.jsonl sessions.',
      'Select: choose recorded sessions as the dataset for supervised, pretraining, or LoRA runs.',
      'Train: send normalized training requests and show loss, PCK, OKS, learning rate, and phase.',
      'Models: list, load, unload, delete, and inspect .rvf model files from data/models.',
      'How data is used: recordings become datasets; training output becomes .rvf models; loaded models can consume live features for inference.'
    ].forEach(item => list.appendChild(this._el('li', null, item)));
    s.appendChild(list);
    return s;
  }

  _renderRecordings() {
    const s = this._el('div', 'tp-section');
    s.appendChild(this._el('div', 'tp-section-title', 'CSI Recordings'));
    if (this.state.recordings.length === 0 && !this.state.loading) {
      s.appendChild(this._el('div', 'tp-empty', 'Start recording CSI data to train a model'));
    } else {
      this.state.recordings.forEach(rec => {
        const row = this._el('div', 'tp-rec-row');
        const info = this._el('div', 'tp-rec-info');
        info.appendChild(this._el('span', 'tp-rec-name', rec.name || rec.id));
        const parts = [];
        if (rec.frame_count != null) parts.push(rec.frame_count + ' frames');
        if (rec.file_size_bytes != null) parts.push(this._fmtB(rec.file_size_bytes));
        if (rec.started_at && rec.ended_at) parts.push(Math.round((new Date(rec.ended_at) - new Date(rec.started_at)) / 1000) + 's');
        info.appendChild(this._el('span', 'tp-rec-meta', parts.join(' / ')));
        row.appendChild(info);
        const del = this._btn('Delete', 'tp-btn tp-btn-muted', () => this._delRec(rec.id));
        del.disabled = this.state.loading;
        row.appendChild(del);
        s.appendChild(row);
      });
    }
    const acts = this._el('div', 'tp-rec-actions');
    if (this.state.isRecording) {
      const b = this._btn('Stop Recording', 'tp-btn tp-btn-danger', () => this._stopRec());
      b.disabled = this.state.loading; acts.appendChild(b);
    } else {
      const b = this._btn('Start Recording', 'tp-btn tp-btn-rec', () => this._startRec());
      b.disabled = this.state.loading; acts.appendChild(b);
    }
    s.appendChild(acts);
    return s;
  }

  _renderConfig() {
    const s = this._el('div', 'tp-section');
    const hdr = this._el('div', 'tp-config-header');
    hdr.appendChild(this._el('span', 'tp-section-title', 'Training Configuration'));
    hdr.appendChild(this._btn(this.state.configOpen ? 'Collapse' : 'Expand', 'tp-btn tp-btn-muted',
      () => { this.state.configOpen = !this.state.configOpen; this.render(); }));
    s.appendChild(hdr);
    if (!this.state.configOpen) return s;

    const form = this._el('div', 'tp-config-form');
    if (this.state.recordings.length > 0) {
      form.appendChild(this._el('label', 'tp-label', 'Datasets'));
      const dc = this._el('div', 'tp-ds-container');
      this.state.recordings.forEach(rec => {
        const lb = this._el('label', 'tp-ds-item');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = this.config.selectedRecordings.includes(rec.id);
        cb.addEventListener('change', () => {
          if (cb.checked) { if (!this.config.selectedRecordings.includes(rec.id)) this.config.selectedRecordings.push(rec.id); }
          else { this.config.selectedRecordings = this.config.selectedRecordings.filter(r => r !== rec.id); }
        });
        lb.appendChild(cb);
        lb.appendChild(this._el('span', null, rec.name || rec.id));
        dc.appendChild(lb);
      });
      form.appendChild(dc);
    }
    const ir = (l, t, v, fn) => {
      const r = this._el('div', 'tp-input-row');
      r.appendChild(this._el('label', 'tp-label', l));
      const inp = document.createElement('input');
      inp.type = t; inp.className = 'tp-input'; inp.value = v;
      inp.addEventListener('change', () => fn(inp.value));
      r.appendChild(inp); return r;
    };
    form.appendChild(ir('Epochs', 'number', this.config.epochs, v => { this.config.epochs = parseInt(v) || 100; }));
    form.appendChild(ir('Batch Size', 'number', this.config.batch_size, v => { this.config.batch_size = parseInt(v) || 32; }));
    form.appendChild(ir('Learning Rate', 'text', this.config.learning_rate, v => { this.config.learning_rate = parseFloat(v) || 3e-4; }));
    form.appendChild(ir('Early Stop Patience', 'number', this.config.patience, v => { this.config.patience = parseInt(v) || 15; }));
    form.appendChild(ir('Base Model (opt.)', 'text', this.config.base_model, v => { this.config.base_model = v; }));
    form.appendChild(ir('LoRA Profile (opt.)', 'text', this.config.lora_profile_name, v => { this.config.lora_profile_name = v; }));
    s.appendChild(form);

    const acts = this._el('div', 'tp-train-actions');
    const btns = [
      this._btn('Start Training', 'tp-btn tp-btn-success', () => this._launchTraining('startTraining', { patience: this.config.patience, base_model: this.config.base_model || undefined })),
      this._btn('Pretrain', 'tp-btn tp-btn-secondary', () => this._launchTraining('startPretraining')),
      this._btn('LoRA', 'tp-btn tp-btn-secondary', () => this._launchTraining('startLoraTraining', { base_model: this.config.base_model || undefined, profile_name: this.config.lora_profile_name || 'default' }))
    ];
    btns.forEach(b => { b.disabled = this.state.loading; acts.appendChild(b); });
    s.appendChild(acts);
    return s;
  }

  _renderProgress() {
    const ts = this.state.trainingStatus || {};
    const s = this._el('div', 'tp-section');
    s.appendChild(this._el('div', 'tp-section-title', 'Training Progress'));

    const pct = ts.total_epochs ? Math.round((ts.epoch / ts.total_epochs) * 100) : 0;
    const bar = this._el('div', 'tp-progress-bar');
    const fill = this._el('div', 'tp-progress-fill');
    fill.style.width = pct + '%';
    bar.appendChild(fill); s.appendChild(bar);
    s.appendChild(this._el('div', 'tp-progress-label', `Epoch ${ts.epoch ?? 0} / ${ts.total_epochs ?? '?'}  (${pct}%)`));

    const cr = this._el('div', 'tp-chart-row');
    const lc = document.createElement('canvas'); lc.id = 'tp-loss-chart'; lc.width = 260; lc.height = 140;
    const pc = document.createElement('canvas'); pc.id = 'tp-pck-chart'; pc.width = 260; pc.height = 140;
    cr.appendChild(lc); cr.appendChild(pc); s.appendChild(cr);

    const g = this._el('div', 'tp-metrics-grid');
    const mc = (l, v) => { const c = this._el('div', 'tp-metric-cell'); c.appendChild(this._el('div', 'tp-metric-label', l)); c.appendChild(this._el('div', 'tp-metric-value', v)); return c; };
    g.appendChild(mc('Loss', ts.train_loss != null ? ts.train_loss.toFixed(4) : '--'));
    g.appendChild(mc('PCK', ts.val_pck != null ? (ts.val_pck * 100).toFixed(1) + '%' : '--'));
    g.appendChild(mc('OKS', ts.val_oks != null ? ts.val_oks.toFixed(3) : '--'));
    g.appendChild(mc('LR', ts.lr != null ? ts.lr.toExponential(1) : '--'));
    g.appendChild(mc('Best PCK', ts.best_pck != null ? (ts.best_pck * 100).toFixed(1) + '% (e' + (ts.best_epoch ?? '?') + ')' : '--'));
    g.appendChild(mc('Patience', ts.patience_remaining != null ? String(ts.patience_remaining) : '--'));
    g.appendChild(mc('ETA', ts.eta_secs != null ? this._fmtEta(ts.eta_secs) : '--'));
    g.appendChild(mc('Phase', ts.phase || '--'));
    s.appendChild(g);

    const stop = this._btn('Stop Training', 'tp-btn tp-btn-danger', () => this._stopTraining());
    stop.disabled = this.state.loading; stop.style.marginTop = '10px'; s.appendChild(stop);
    return s;
  }

  _renderComplete() {
    const ts = this.state.trainingStatus || {};
    const s = this._el('div', 'tp-section');
    s.appendChild(this._el('div', 'tp-section-title', 'Training Complete'));
    const g = this._el('div', 'tp-metrics-grid');
    const mc = (l, v) => { const c = this._el('div', 'tp-metric-cell'); c.appendChild(this._el('div', 'tp-metric-label', l)); c.appendChild(this._el('div', 'tp-metric-value', v)); return c; };
    const losses = this.progressData.losses;
    g.appendChild(mc('Final Loss', losses.length > 0 ? losses[losses.length - 1].toFixed(4) : '--'));
    g.appendChild(mc('Best PCK', ts.best_pck != null ? (ts.best_pck * 100).toFixed(1) + '%' : '--'));
    g.appendChild(mc('Best Epoch', ts.best_epoch != null ? String(ts.best_epoch) : '--'));
    g.appendChild(mc('Total Epochs', String(losses.length)));
    s.appendChild(g);
    const acts = this._el('div', 'tp-train-actions');
    acts.appendChild(this._btn('New Training', 'tp-btn tp-btn-secondary', () => {
      this.progressData = { losses: [], pcks: [] }; this._set({ trainingStatus: null });
    }));
    s.appendChild(acts);
    return s;
  }

  // --- Chart drawing ---

  _drawCharts() {
    this._drawChart('tp-loss-chart', this.progressData.losses, { color: '#c0152f', label: 'Loss', yMin: 0, yMax: null });
    this._drawChart('tp-pck-chart', this.progressData.pcks, { color: '#21808d', label: 'PCK', yMin: 0, yMax: 1 });
  }

  _drawChart(id, data, opts) {
    const cv = document.getElementById(id);
    if (!cv) return;
    const ctx = cv.getContext('2d'), w = cv.width, h = cv.height;
    const p = { t: 20, r: 10, b: 24, l: 44 };
    ctx.fillStyle = '#f7f7f4'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#626c71'; ctx.font = '11px -apple-system,sans-serif'; ctx.fillText(opts.label, p.l, 14);
    if (!data.length) { ctx.fillStyle = '#626c71'; ctx.fillText('No data', w / 2 - 20, h / 2); return; }
    const pw = w - p.l - p.r, ph = h - p.t - p.b;
    let yMin = opts.yMin ?? Math.min(...data), yMax = opts.yMax ?? Math.max(...data);
    if (yMax === yMin) yMax = yMin + 1;
    ctx.strokeStyle = 'rgba(94,82,64,.18)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = p.t + (ph / 4) * i;
      ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(w - p.r, y); ctx.stroke();
      const v = yMax - ((yMax - yMin) / 4) * i;
      ctx.fillStyle = '#626c71'; ctx.font = '9px sans-serif'; ctx.fillText(v.toFixed(v >= 1 ? 2 : 3), 2, y + 3);
    }
    const xl = Math.min(data.length, 5);
    for (let i = 0; i < xl; i++) {
      const idx = Math.round((data.length - 1) * (i / (xl - 1 || 1)));
      ctx.fillStyle = '#626c71'; ctx.fillText(String(idx + 1), p.l + (pw * idx) / (data.length - 1 || 1) - 4, h - 4);
    }
    ctx.strokeStyle = opts.color; ctx.lineWidth = 1.5; ctx.beginPath();
    data.forEach((v, i) => {
      const x = p.l + (pw * i) / (data.length - 1 || 1);
      const y = p.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (data.length > 0) {
      const ly = p.t + ph - ((data[data.length - 1] - yMin) / (yMax - yMin)) * ph;
      ctx.fillStyle = opts.color; ctx.beginPath(); ctx.arc(p.l + pw, ly, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  // --- Helpers ---

  _el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  _btn(txt, cls, fn) {
    const b = document.createElement('button');
    b.className = cls; b.textContent = txt;
    b.addEventListener('click', fn); return b;
  }

  _fmtB(b) { return b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(1) + ' KB' : (b / 1048576).toFixed(1) + ' MB'; }
  _fmtEta(s) { return s < 60 ? Math.round(s) + 's' : s < 3600 ? Math.round(s / 60) + 'm' : (s / 3600).toFixed(1) + 'h'; }

  _injectStyles() {
    if (document.getElementById('training-panel-styles')) return;
    const s = document.createElement('style');
    s.id = 'training-panel-styles';
    s.textContent = TP_STYLES;
    document.head.appendChild(s);
  }

  destroy() {
    this.unsubscribers.forEach(fn => fn());
    this.unsubscribers = [];
    trainingService.disconnectProgressStream();
    if (this.container) this.container.innerHTML = '';
  }

  dispose() {
    this.destroy();
  }
}
