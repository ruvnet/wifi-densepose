/**
 * SensingTab — Live WiFi Sensing Visualization
 *
 * Connects to the sensing WebSocket service and renders:
 *   1. A 3D Gaussian-splat signal field (via gaussian-splats.js)
 *   2. An overlay HUD with real-time metrics (RSSI, variance, bands, classification)
 */

import { sensingService } from '../services/sensing.service.js';
import { GaussianSplatRenderer } from './gaussian-splats.js';

export class SensingTab {
  /** @param {HTMLElement} container - the #sensing section element */
  constructor(container) {
    this.container = container;
    this.splatRenderer = null;
    this._unsubData = null;
    this._unsubState = null;
    this._resizeObserver = null;
    this._threeLoaded = false;
  }

  async init() {
    this._buildDOM();
    await this._loadThree();
    this._initSplatRenderer();
    this._connectService();
    this._setupResize();
  }

  // ---- DOM construction --------------------------------------------------

  _buildDOM() {
    this.container.innerHTML = `
      <h2>Live WiFi Sensing</h2>

      <!-- Data-source status banner — updated by _onStateChange -->
      <div id="sensingSourceBanner" class="sensing-source-banner sensing-source-reconnecting"
           role="status" aria-live="polite">
        RECONNECTING...
      </div>

      <div class="sensing-layout">
        <!-- 3D viewport -->
        <div class="sensing-viewport" id="sensingViewport">
          <div class="sensing-loading">Loading 3D engine...</div>
        </div>

        <!-- Side panel -->
        <div class="sensing-panel">
          <!-- Connection -->
          <div class="sensing-card">
            <div class="sensing-card-title">Connection</div>
            <div class="sensing-connection">
              <span class="sensing-dot" id="sensingDot"></span>
              <span id="sensingState">Connecting...</span>
              <span class="sensing-source" id="sensingSource"></span>
            </div>
          </div>

          <!-- RSSI -->
          <div class="sensing-card">
            <div class="sensing-card-title">RSSI</div>
            <div class="sensing-big-value" id="sensingRssi">-- dBm</div>
            <canvas id="sensingSparkline" width="200" height="40"></canvas>
          </div>

          <!-- Signal Features -->
          <div class="sensing-card">
            <div class="sensing-card-title">Signal Features</div>
            <div class="sensing-meters">
              <div class="sensing-meter">
                <label>Variance</label>
                <div class="sensing-bar"><div class="sensing-bar-fill" id="barVariance"></div></div>
                <span class="sensing-meter-val" id="valVariance">0</span>
              </div>
              <div class="sensing-meter">
                <label>Motion Band</label>
                <div class="sensing-bar"><div class="sensing-bar-fill motion" id="barMotion"></div></div>
                <span class="sensing-meter-val" id="valMotion">0</span>
              </div>
              <div class="sensing-meter">
                <label>Breathing Band</label>
                <div class="sensing-bar"><div class="sensing-bar-fill breath" id="barBreath"></div></div>
                <span class="sensing-meter-val" id="valBreath">0</span>
              </div>
              <div class="sensing-meter">
                <label>Spectral Power</label>
                <div class="sensing-bar"><div class="sensing-bar-fill spectral" id="barSpectral"></div></div>
                <span class="sensing-meter-val" id="valSpectral">0</span>
              </div>
            </div>
          </div>

          <!-- Classification -->
          <div class="sensing-card">
            <div class="sensing-card-title">Classification</div>
            <div class="sensing-classification" id="sensingClassification">
              <div class="sensing-class-label" id="classLabel">ABSENT</div>
              <div class="sensing-confidence">
                <label>Confidence</label>
                <div class="sensing-bar"><div class="sensing-bar-fill confidence" id="barConfidence"></div></div>
                <span class="sensing-meter-val" id="valConfidence">0%</span>
              </div>
            </div>
          </div>

          <!-- Setup info -->
          <div class="sensing-card">
            <div class="sensing-card-title">About This Data</div>
            <p class="sensing-about-text">
              Metrics are computed from WiFi Channel State Information (CSI).
              With <strong><span id="sensingNodeCount">0</span> ESP32 node(s)</strong> you get presence detection, breathing
              estimation, and gross motion. Add <strong>3-4+ ESP32 nodes</strong>
              around the room for spatial resolution and limb-level tracking.
            </p>
          </div>

          <!-- Node Status -->
          <div class="sensing-card" id="sensingNodeCards">
            <div class="sensing-card-title">NODE STATUS</div>
            <div id="nodeStatusContainer"></div>
          </div>

          <!-- Extra info -->
          <div class="sensing-card">
            <div class="sensing-card-title">Details</div>
            <div class="sensing-details">
              <div class="sensing-detail-row">
                <span>Dominant Freq</span><span id="valDomFreq">0 Hz</span>
              </div>
              <div class="sensing-detail-row">
                <span>Change Points</span><span id="valChangePoints">0</span>
              </div>
              <div class="sensing-detail-row">
                <span>Sample Rate</span><span id="valSampleRate">--</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ---- Three.js loading --------------------------------------------------

  async _loadThree() {
    if (window.THREE) {
      this._threeLoaded = true;
      return;
    }

    return new Promise((resolve) => {
      let settled = false;
      const finish = (loaded) => {
        if (settled) return;
        settled = true;
        this._threeLoaded = loaded;
        resolve();
      };
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
      script.onload = () => finish(true);
      script.onerror = () => {
        console.warn('[SensingTab] Three.js CDN unavailable; using canvas renderer');
        finish(false);
      };
      document.head.appendChild(script);
      setTimeout(() => {
        console.warn('[SensingTab] Three.js load timed out; using canvas renderer');
        finish(false);
      }, 2500);
    });
  }

  // ---- Splat renderer ----------------------------------------------------

  _initSplatRenderer() {
    const viewport = this.container.querySelector('#sensingViewport');
    if (!viewport) return;

    // Remove loading message
    viewport.innerHTML = '';

    try {
      const Renderer = window.THREE ? GaussianSplatRenderer : CanvasSensingRenderer;
      this.splatRenderer = new Renderer(viewport, {
        width: viewport.clientWidth,
        height: viewport.clientHeight || 500,
      });
    } catch (e) {
      console.error('[SensingTab] Failed to init splat renderer:', e);
      this.splatRenderer = new CanvasSensingRenderer(viewport, {
        width: viewport.clientWidth,
        height: viewport.clientHeight || 500,
      });
    }
  }

  // ---- Service connection ------------------------------------------------

  _connectService() {
    sensingService.start();

    this._unsubData = sensingService.onData((data) => this._onSensingData(data));
    this._unsubState = sensingService.onStateChange((state) => this._onStateChange(state));
  }

  _onSensingData(data) {
    // Update 3D view
    if (this.splatRenderer) {
      this.splatRenderer.update(data);
    }

    // Update HUD
    this._updateHUD(data);

    // Update per-node panels
    this._updateNodePanels(data);
  }

  _onStateChange(state) {
    const dot    = this.container.querySelector('#sensingDot');
    const text   = this.container.querySelector('#sensingState');
    const banner = this.container.querySelector('#sensingSourceBanner');

    if (dot && text) {
      const dataSource = sensingService.dataSource;
      const stateLabels = {
        disconnected: 'Disconnected',
        connecting:   'Connecting...',
        connected:    'Connected',
        reconnecting: 'Reconnecting...',
        simulated:    'Disconnected',
      };
      const sourceLabels = {
        live: 'Live',
        stale: 'Stale',
      };
      const displayState = dataSource === 'live' ? 'connected'
        : dataSource === 'stale' ? 'reconnecting'
        : state;
      dot.className = 'sensing-dot ' + displayState;
      text.textContent = sourceLabels[dataSource] || stateLabels[state] || state;
    }

    if (banner) {
      // Map the service's dataSource to banner text and CSS modifier class.
      const dataSource = sensingService.dataSource;
      const bannerConfig = {
        'live':              { text: 'LIVE \u2014 ESP32 HARDWARE',           cls: 'sensing-source-live' },
        'stale':             { text: 'STALE \u2014 FEATURE STATE',           cls: 'sensing-source-stale' },
        'server-simulated':  { text: 'OFFLINE \u2014 NO LIVE HARDWARE',      cls: 'sensing-source-simulated' },
        'reconnecting':      { text: 'RECONNECTING...',                    cls: 'sensing-source-reconnecting' },
        'simulated':         { text: 'OFFLINE \u2014 NO LIVE DATA',          cls: 'sensing-source-simulated' },
      };
      const cfg = bannerConfig[dataSource] || bannerConfig.reconnecting;
      banner.textContent = cfg.text;
      banner.className = 'sensing-source-banner ' + cfg.cls;
    }
  }

  // ---- HUD update --------------------------------------------------------

  _updateHUD(data) {
    const f = data.features || {};
    const c = data.classification || {};

    // Node count
    const nodeCount = (data.nodes || []).length;
    const countEl = this.container.querySelector('#sensingNodeCount');
    if (countEl) countEl.textContent = String(nodeCount);

    // RSSI
    this._setText('sensingRssi', `${(f.mean_rssi || -80).toFixed(1)} dBm`);
    this._setText('sensingSource', data.source || '');

    // Bars (scale to 0-100%)
    this._setBar('barVariance', f.variance, 10, 'valVariance', f.variance);
    this._setBar('barMotion', f.motion_band_power, 0.5, 'valMotion', f.motion_band_power);
    this._setBar('barBreath', f.breathing_band_power, 0.3, 'valBreath', f.breathing_band_power);
    this._setBar('barSpectral', f.spectral_power, 2.0, 'valSpectral', f.spectral_power);

    // Classification
    const label = this.container.querySelector('#classLabel');
    if (label) {
      const level = (c.motion_level || 'absent').toUpperCase();
      label.textContent = level;
      label.className = 'sensing-class-label ' + (c.motion_level || 'absent');
    }

    const confPct = ((c.confidence || 0) * 100).toFixed(0);
    this._setBar('barConfidence', c.confidence, 1.0, 'valConfidence', confPct + '%');

    // Details
    this._setText('valDomFreq', (f.dominant_freq_hz || 0).toFixed(3) + ' Hz');
    this._setText('valChangePoints', String(f.change_points || 0));
    const srcLabel = (data.source === 'simulated' || data.source === 'simulate') ? 'offline' : data.source || 'live';
    this._setText('valSampleRate', srcLabel);

    // Sparkline
    this._drawSparkline();
  }

  _setText(id, text) {
    const el = this.container.querySelector('#' + id);
    if (el) el.textContent = text;
  }

  _setBar(barId, value, maxVal, valId, displayVal) {
    const bar = this.container.querySelector('#' + barId);
    if (bar) {
      const pct = Math.min(100, Math.max(0, ((value || 0) / maxVal) * 100));
      bar.style.width = pct + '%';
    }
    if (valId && displayVal != null) {
      const el = this.container.querySelector('#' + valId);
      if (el) el.textContent = typeof displayVal === 'number' ? displayVal.toFixed(3) : displayVal;
    }
  }

  _drawSparkline() {
    const canvas = this.container.querySelector('#sensingSparkline');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const history = sensingService.getRssiHistory();
    if (history.length < 2) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const min = Math.min(...history) - 2;
    const max = Math.max(...history) + 2;
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = '#32b8c6';
    ctx.lineWidth = 1.5;

    for (let i = 0; i < history.length; i++) {
      const x = (i / (history.length - 1)) * w;
      const y = h - ((history[i] - min) / range) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ---- Per-node panels ---------------------------------------------------

  _updateNodePanels(data) {
    const container = this.container.querySelector('#nodeStatusContainer');
    if (!container) return;
    const nodeFeatures = data.node_features || [];
    if (nodeFeatures.length === 0) {
      container.textContent = '';
      const msg = document.createElement('div');
      msg.style.cssText = 'color:#888;font-size:12px;padding:8px;';
      msg.textContent = 'No nodes detected';
      container.appendChild(msg);
      return;
    }
    const NODE_COLORS = ['#00ccff', '#ff6600', '#00ff88', '#ff00cc', '#ffcc00', '#8800ff', '#00ffcc', '#ff0044'];
    container.textContent = '';
    for (const nf of nodeFeatures) {
      const color = NODE_COLORS[nf.node_id % NODE_COLORS.length];
      const statusColor = nf.stale ? '#888' : '#0f0';

      const row = document.createElement('div');
      row.style.cssText = `display:flex;align-items:center;gap:8px;padding:6px 8px;margin-bottom:4px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:3px solid ${color};`;

      const idCol = document.createElement('div');
      idCol.style.minWidth = '50px';
      const nameEl = document.createElement('div');
      nameEl.style.cssText = `font-size:11px;font-weight:600;color:${color};`;
      nameEl.textContent = 'Node ' + nf.node_id;
      const statusEl = document.createElement('div');
      statusEl.style.cssText = `font-size:9px;color:${statusColor};`;
      statusEl.textContent = nf.stale ? 'STALE' : 'ACTIVE';
      idCol.appendChild(nameEl);
      idCol.appendChild(statusEl);

      const metricsCol = document.createElement('div');
      metricsCol.style.cssText = 'flex:1;font-size:10px;color:#aaa;';
      metricsCol.textContent = (nf.rssi_dbm || -80).toFixed(0) + ' dBm · var ' + (nf.features?.variance || 0).toFixed(1);

      const classCol = document.createElement('div');
      classCol.style.cssText = 'font-size:10px;font-weight:600;color:#ccc;';
      const motion = (nf.classification?.motion_level || 'absent').toUpperCase();
      const conf = ((nf.classification?.confidence || 0) * 100).toFixed(0);
      classCol.textContent = motion + ' ' + conf + '%';

      row.appendChild(idCol);
      row.appendChild(metricsCol);
      row.appendChild(classCol);
      container.appendChild(row);
    }
  }

  // ---- Resize ------------------------------------------------------------

  _setupResize() {
    const viewport = this.container.querySelector('#sensingViewport');
    if (!viewport || !window.ResizeObserver) return;

    this._resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (this.splatRenderer) {
          this.splatRenderer.resize(entry.contentRect.width, entry.contentRect.height);
        }
      }
    });
    this._resizeObserver.observe(viewport);
  }

  // ---- Cleanup -----------------------------------------------------------

  dispose() {
    if (this._unsubData) this._unsubData();
    if (this._unsubState) this._unsubState();
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this.splatRenderer) this.splatRenderer.dispose();
    sensingService.stop();
  }
}

class CanvasSensingRenderer {
  constructor(container, opts = {}) {
    this.container = container;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.width = opts.width || container.clientWidth || 800;
    this.height = opts.height || 500;
    this._lastData = null;
    this._animFrame = null;
    this.resize(this.width, this.height);
    container.appendChild(this.canvas);
    this._animate();
  }

  update(data) {
    this._lastData = data;
  }

  resize(width, height) {
    this.width = Math.max(240, Math.floor(width || this.container.clientWidth || 800));
    this.height = Math.max(240, Math.floor(height || this.container.clientHeight || 500));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(this.width * dpr);
    this.canvas.height = Math.floor(this.height * dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  dispose() {
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
    if (this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
  }

  _animate() {
    this._animFrame = requestAnimationFrame(() => this._animate());
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const data = this._lastData;
    ctx.clearRect(0, 0, this.width, this.height);
    this._drawBackground(ctx);
    this._drawSignalField(ctx, data);
    this._drawNodes(ctx, data);
    this._drawPresence(ctx, data);
    this._drawOverlay(ctx, data);
  }

  _drawBackground(ctx) {
    const gradient = ctx.createLinearGradient(0, 0, this.width, this.height);
    gradient.addColorStop(0, '#071015');
    gradient.addColorStop(1, '#101421');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, this.width, this.height);

    ctx.strokeStyle = 'rgba(50, 184, 198, 0.14)';
    ctx.lineWidth = 1;
    const grid = 10;
    for (let i = 0; i <= grid; i++) {
      const x = (i / grid) * this.width;
      const y = (i / grid) * this.height;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
      ctx.stroke();
    }
  }

  _drawSignalField(ctx, data) {
    const values = data?.signal_field?.values;
    const size = Array.isArray(data?.signal_field?.grid_size)
      ? data.signal_field.grid_size[0]
      : 20;
    if (!Array.isArray(values) || values.length === 0) return;

    const cellW = this.width / size;
    const cellH = this.height / size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const value = Math.max(0, Math.min(1, Number(values[y * size + x]) || 0));
        const hue = 205 - value * 190;
        const alpha = 0.18 + value * 0.62;
        ctx.fillStyle = `hsla(${hue}, 90%, ${32 + value * 34}%, ${alpha})`;
        ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  _drawNodes(ctx, data) {
    const nodes = Array.isArray(data?.nodes) ? data.nodes : [];
    const colors = ['#32b8c6', '#ff8a3d', '#35d88f', '#d96cff'];
    nodes.forEach((node, index) => {
      const point = this._roomToCanvas(node.position || [0, 0, 0]);
      ctx.beginPath();
      ctx.arc(point.x, point.y, 9, 0, Math.PI * 2);
      ctx.fillStyle = colors[index % colors.length];
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.stroke();
      ctx.fillStyle = '#f5f7fb';
      ctx.font = '11px sans-serif';
      ctx.fillText(`Node ${node.node_id}`, point.x + 12, point.y + 4);
    });
  }

  _drawPresence(ctx, data) {
    if (!data?.classification?.presence) return;
    const confidence = Math.max(0, Math.min(1, Number(data.classification.confidence) || 0));
    const motion = Math.max(0, Math.min(1, Number(data.features?.motion_band_power) || 0));
    const pulse = 0.7 + Math.sin(Date.now() * 0.006) * 0.18;
    const radius = (52 + motion * 80) * pulse;
    const x = this.width * (0.5 + Math.sin(Date.now() * 0.00045) * 0.11);
    const y = this.height * (0.52 + Math.cos(Date.now() * 0.00033) * 0.08);
    const gradient = ctx.createRadialGradient(x, y, 4, x, y, radius);
    gradient.addColorStop(0, `rgba(255, 80, 62, ${0.35 + confidence * 0.35})`);
    gradient.addColorStop(0.5, `rgba(53, 216, 143, ${0.18 + confidence * 0.22})`);
    gradient.addColorStop(1, 'rgba(50, 184, 198, 0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  _drawOverlay(ctx, data) {
    const label = data
      ? `${(data.classification?.motion_level || 'live').toUpperCase()}  ${(data.features?.mean_rssi ?? 0).toFixed(0)} dBm`
      : 'WAITING FOR LIVE CSI';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.42)';
    ctx.fillRect(14, 14, 220, 34);
    ctx.fillStyle = '#e8f9fb';
    ctx.font = '13px sans-serif';
    ctx.fillText(label, 24, 36);
  }

  _roomToCanvas(position) {
    const x = Array.isArray(position) ? Number(position[0]) || 0 : 0;
    const z = Array.isArray(position) ? Number(position[2]) || 0 : 0;
    return {
      x: this.width * (0.5 + x / 20),
      y: this.height * (0.5 + z / 20),
    };
  }
}
