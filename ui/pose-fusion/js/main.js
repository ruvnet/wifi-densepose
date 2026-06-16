/**
 * RuView — Dual-Modal Pose Estimation
 *
 * Main orchestration: video capture → CNN embedding → CSI processing → fusion → rendering
 */

import { VideoCapture } from './video-capture.js?v=15';
import { CnnEmbedder } from './cnn-embedder.js?v=15';
import { FusionEngine } from './fusion-engine.js?v=15';
import { PoseDecoder } from './pose-decoder.js?v=15';
import { CanvasRenderer } from './canvas-renderer.js?v=15';

// === State ===
let mode = 'dual';  // 'dual' | 'video' | 'csi'
let isRunning = false;
let isPaused = false;
let startTime = 0;
let frameCount = 0;
let fps = 0;
let lastFpsTime = 0;
let confidenceThreshold = 0.3;
let cameraAvailable = null;

// Latency tracking
const latency = { video: 0, csi: 0, fusion: 0, total: 0 };

// === Components ===
const videoCapture = new VideoCapture(document.getElementById('webcam'));
const visualCnn = new CnnEmbedder({ inputSize: 56, embeddingDim: 128, seed: 42 });
const csiCnn = new CnnEmbedder({ inputSize: 56, embeddingDim: 128, seed: 137 });
const fusionEngine = new FusionEngine(128);
const poseDecoder = new PoseDecoder(128);
const renderer = new CanvasRenderer();

class LiveCsiSource {
  constructor() {
    this.ws = null;
    this.pollTimer = null;
    this.latest = { active: false, heatmap: null, rssi: null, snr: 0, presence: 0 };
    this.rows = [];
    this.maxRows = 20;
    this.width = 56;
  }

  async connectLive(url = '') {
    this.disconnect();
    if (url.startsWith('ws://') || url.startsWith('wss://')) {
      const ok = await this._connectWebSocket(url);
      if (ok) return true;
    }
    this._startApiPolling();
    await this._pollOnce();
    return this.latest.active;
  }

  disconnect() {
    if (this.ws) {
      try { this.ws.close(); } catch (_) {}
      this.ws = null;
    }
    if (this.pollTimer) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  getSnapshot() {
    return this.latest;
  }

  async _connectWebSocket(url) {
    return new Promise((resolve) => {
      let settled = false;
      const finish = (ok) => {
        if (settled) return;
        settled = true;
        resolve(ok);
      };
      try {
        const ws = new WebSocket(url);
        const timer = window.setTimeout(() => {
          try { ws.close(); } catch (_) {}
          finish(false);
        }, 900);
        ws.addEventListener('open', () => {
          window.clearTimeout(timer);
          this.ws = ws;
          finish(true);
        });
        ws.addEventListener('message', (event) => {
          try { this._ingest(JSON.parse(event.data)); } catch (_) {}
        });
        ws.addEventListener('error', () => {
          window.clearTimeout(timer);
          finish(false);
        });
        ws.addEventListener('close', () => {
          if (this.ws === ws) this._startApiPolling();
        });
      } catch (_) {
        finish(false);
      }
    });
  }

  _startApiPolling() {
    if (this.pollTimer) return;
    this._pollOnce();
    this.pollTimer = window.setInterval(() => this._pollOnce(), 500);
  }

  async _pollOnce() {
    try {
      const res = await fetch('/api/v1/cardputer/status', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this._ingest(await res.json());
    } catch (_) {
      this.latest = { ...this.latest, active: false };
    }
  }

  _ingest(payload) {
    const nodes = Array.isArray(payload?.nodes) ? payload.nodes.filter(n => n.live) : [];
    const source = nodes.find(n => n.edge_feature_live && n.edge_feature?.features)
      || nodes.find(n => n.feature_state_live && n.feature_state)
      || nodes.find(n => n.edge_vitals_live && n.edge_vitals)
      || nodes[0]
      || payload;
    const values = this._featureValues(source);
    if (!values.length) {
      this.latest = {
        ...this.latest,
        active: Boolean(payload?.live || nodes.length),
        rssi: this._rssi(source),
        snr: this._snr(source),
      };
      return;
    }
    const row = this._normalizeRow(values);
    this.rows.push(row);
    if (this.rows.length > this.maxRows) this.rows.shift();
    const heatmap = this._heatmap();
    const rssi = this._rssi(source);
    this.latest = {
      active: true,
      heatmap,
      rssi,
      snr: this._snr(source),
      presence: this._presence(source),
      nodeCount: Number(payload?.live_node_count || nodes.length || 0),
    };
  }

  _featureValues(source) {
    if (Array.isArray(source?.edge_feature?.features)) return source.edge_feature.features;
    const fs = source?.feature_state || source?.edge_vitals || source || {};
    const keys = [
      'motion_score', 'presence_score', 'respiration_bpm', 'respiration_conf',
      'heartbeat_bpm', 'heartbeat_conf', 'anomaly_score', 'env_shift_score',
      'node_coherence', 'breathing_bpm', 'rssi_dbm', 'n_persons', 'motion_energy',
    ];
    return keys.map(k => Number(fs[k])).filter(Number.isFinite);
  }

  _normalizeRow(values) {
    const row = new Float32Array(this.width);
    const clean = values.map(v => Number.isFinite(v) ? v : 0);
    let min = Math.min(...clean);
    let max = Math.max(...clean);
    if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) {
      min = -1; max = 1;
    }
    for (let i = 0; i < this.width; i++) {
      const idx = Math.floor((i / this.width) * clean.length);
      const v = clean[Math.min(clean.length - 1, idx)] ?? 0;
      row[i] = Math.max(0, Math.min(1, (v - min) / (max - min)));
    }
    return row;
  }

  _heatmap() {
    const data = new Float32Array(this.width * this.maxRows);
    const offset = this.maxRows - this.rows.length;
    this.rows.forEach((row, i) => data.set(row, (offset + i) * this.width));
    return { data, width: this.width, height: this.maxRows };
  }

  _rssi(source) {
    const value = source?.rssi_dbm ?? source?.edge_vitals?.rssi_dbm ?? source?.feature_state?.rssi_dbm;
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }

  _snr(source) {
    const rssi = this._rssi(source);
    const noise = Number(source?.noise_floor_dbm ?? -92);
    if (rssi == null) return 0;
    return Math.max(0, Math.min(35, rssi - noise));
  }

  _presence(source) {
    const value = source?.feature_state?.presence_score
      ?? source?.edge_vitals?.presence_score
      ?? source?.presence_score
      ?? source?.motion_score;
    return Math.max(0, Math.min(1, Number(value) || 0));
  }
}

const csiSimulator = new LiveCsiSource();

// === Canvas Elements ===
const skeletonCanvas = document.getElementById('skeleton-canvas');
const skeletonCtx = skeletonCanvas.getContext('2d');
const csiCanvas = document.getElementById('csi-canvas');
const csiCtx = csiCanvas.getContext('2d');
const embeddingCanvas = document.getElementById('embedding-canvas');
const embeddingCtx = embeddingCanvas.getContext('2d');

// === UI Elements ===
const modeSelect = document.getElementById('mode-select');
const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const fpsDisplay = document.getElementById('fps-display');
const cameraPrompt = document.getElementById('camera-prompt');
const startCameraBtn = document.getElementById('start-camera-btn');
const pauseBtn = document.getElementById('pause-btn');
const confSlider = document.getElementById('confidence-slider');
const confValue = document.getElementById('confidence-value');
const wsUrlInput = document.getElementById('ws-url');
const connectWsBtn = document.getElementById('connect-ws-btn');

// Fusion bar elements
const videoBar = document.getElementById('video-bar');
const csiBar = document.getElementById('csi-bar');
const fusedBar = document.getElementById('fused-bar');
const videoBarVal = document.getElementById('video-bar-val');
const csiBarVal = document.getElementById('csi-bar-val');
const fusedBarVal = document.getElementById('fused-bar-val');

// Latency elements
const latVideoEl = document.getElementById('lat-video');
const latCsiEl = document.getElementById('lat-csi');
const latFusionEl = document.getElementById('lat-fusion');
const latTotalEl = document.getElementById('lat-total');

// Cross-modal similarity
const crossModalEl = document.getElementById('cross-modal-sim');

// RSSI elements
const rssiBarEl = document.getElementById('rssi-bar');
const rssiValueEl = document.getElementById('rssi-value');
const rssiQualityEl = document.getElementById('rssi-quality');
const rssiSparkCanvas = document.getElementById('rssi-sparkline');
const rssiSparkCtx = rssiSparkCanvas ? rssiSparkCanvas.getContext('2d') : null;
const rssiHistory = [];
const RSSI_HISTORY_MAX = 80;

// === Initialize ===
function init() {
  console.log('[PoseFusion] init() v4 — live-only sensing, starting...');
  resizeCanvases();
  console.log(`[PoseFusion] canvases: skeleton=${skeletonCanvas.width}x${skeletonCanvas.height}, csi=${csiCanvas.width}x${csiCanvas.height}, emb=${embeddingCanvas.width}x${embeddingCanvas.height}`);
  window.addEventListener('resize', resizeCanvases);

  // Mode change
  modeSelect.addEventListener('change', (e) => {
    mode = e.target.value;
    updateModeUI();
  });

  // Camera start
  startCameraBtn.addEventListener('click', startCamera);

  // Pause
  pauseBtn.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? '▶ Resume' : '⏸ Pause';
    pauseBtn.classList.toggle('active', isPaused);
  });

  // Confidence slider
  confSlider.addEventListener('input', (e) => {
    confidenceThreshold = parseFloat(e.target.value);
    confValue.textContent = confidenceThreshold.toFixed(2);
  });

  // WebSocket connect
  connectWsBtn.addEventListener('click', async () => {
    const url = wsUrlInput.value.trim();
    if (!url || url === 'desktop-api') {
      connectWsBtn.textContent = 'Connecting...';
      const ok = await csiSimulator.connectLive('');
      connectWsBtn.textContent = ok ? '✓ Live ESP32' : 'Connect';
      connectWsBtn.classList.toggle('active', ok);
      if (ok) {
        statusLabel.textContent = 'LIVE CSI';
        statusDot.classList.remove('offline');
      }
      return;
    }
    connectWsBtn.textContent = 'Connecting...';
    const ok = await csiSimulator.connectLive(url);
    connectWsBtn.textContent = ok ? '✓ Connected' : 'Connect';
    if (ok) {
      connectWsBtn.classList.add('active');
    }
  });

  // Try to load RuVector Attention WASM embedders (non-blocking)
  const wasmBase = new URL('../pkg/ruvector-attention', import.meta.url).href;
  visualCnn.tryLoadWasm(wasmBase).then((ok) => {
    // Share the WASM module with FusionEngine for cosine_similarity, normalize, etc.
    if (visualCnn.rvModule) fusionEngine.setWasmModule(visualCnn.rvModule);
    // Update footer backend label
    const backendEl = document.getElementById('cnn-backend');
    if (backendEl) {
      backendEl.textContent = ok && visualCnn.useRuVector
        ? `RuVector WASM v${visualCnn.rvModule.version()} — 6 attention mechanisms`
        : 'ruvector-cnn (JS fallback)';
    }
  });
  csiCnn.tryLoadWasm(wasmBase);

  // Auto-connect to local sensing server WebSocket if available
  if (wsUrlInput) wsUrlInput.value = 'desktop-api';
  csiSimulator.connectLive('').then(ok => {
    if (ok && connectWsBtn) {
      connectWsBtn.textContent = '✓ Live ESP32';
      connectWsBtn.classList.add('active');
      statusLabel.textContent = 'LIVE CSI';
      statusDot.classList.remove('offline');
      updateModeUI();
    }
  });

  detectCameraAvailability().then(hasCamera => {
    cameraAvailable = hasCamera;
    const promptText = cameraPrompt?.querySelector('p');
    if (!hasCamera && promptText) {
      promptText.textContent = 'No local webcam detected. Dual mode is running from live CSI until a camera is attached.';
    }
    updateModeUI();
  });

  startTime = performance.now() / 1000;
  isRunning = true;
  requestAnimationFrame(mainLoop);
}

async function detectCameraAvailability() {
  if (!navigator.mediaDevices?.enumerateDevices) return false;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.some(device => device.kind === 'videoinput');
  } catch (_) {
    return false;
  }
}

async function startCamera() {
  cameraPrompt.style.display = 'none';
  const ok = await videoCapture.start();
  if (ok) {
    statusDot.classList.remove('offline');
    statusLabel.textContent = 'LIVE';
    resizeCanvases();
  } else {
    cameraAvailable = false;
    updateModeUI();
    statusDot.classList.remove('offline');
    statusLabel.textContent = csiSimulator.getSnapshot()?.active
      ? (mode === 'dual' ? 'DUAL CSI' : 'LIVE CSI')
      : 'CSI ONLY';
  }
}

function updateModeUI() {
  const needsVideo = mode !== 'csi';
  const csiLive = Boolean(csiSimulator.getSnapshot()?.active);
  const canFallbackToCsi = mode === 'dual' && csiLive;

  // Show/hide camera prompt
  if (needsVideo && !videoCapture.isActive && !canFallbackToCsi) {
    cameraPrompt.style.display = 'flex';
  } else {
    cameraPrompt.style.display = 'none';
  }

  // Update mode label in both the overlay and the camera prompt
  const labelMap = { dual: 'DUAL FUSION', video: 'VIDEO ONLY', csi: 'CSI ONLY' };
  const modeLabel = document.getElementById('mode-label');
  const promptLabel = document.getElementById('prompt-mode-label');
  if (modeLabel) modeLabel.textContent = labelMap[mode] || mode;
  if (promptLabel) promptLabel.textContent = labelMap[mode] || mode;
  if (mode === 'dual' && !videoCapture.isActive && csiLive) {
    statusDot.classList.remove('offline');
    statusLabel.textContent = cameraAvailable === false ? 'DUAL CSI' : 'LIVE CSI';
  }
}

function resizeCanvases() {
  const videoPanel = document.querySelector('.video-panel');
  if (videoPanel) {
    const rect = videoPanel.getBoundingClientRect();
    skeletonCanvas.width = rect.width;
    skeletonCanvas.height = rect.height;
  }

  // CSI canvas (min 200px width)
  csiCanvas.width = Math.max(200, csiCanvas.parentElement.clientWidth);
  csiCanvas.height = 120;

  // Embedding canvas (min 200px width)
  embeddingCanvas.width = Math.max(200, embeddingCanvas.parentElement.clientWidth);
  embeddingCanvas.height = 140;
}

function heatmapToRgb(heatmap, outW, outH) {
  const rgb = new Uint8Array(outW * outH * 3);
  if (!heatmap?.data?.length) return rgb;
  const { data, width, height } = heatmap;
  for (let y = 0; y < outH; y++) {
    const sy = Math.min(height - 1, Math.floor((y / outH) * height));
    for (let x = 0; x < outW; x++) {
      const sx = Math.min(width - 1, Math.floor((x / outW) * width));
      const v = Math.max(0, Math.min(1, data[sy * width + sx] || 0));
      const i = (y * outW + x) * 3;
      rgb[i] = Math.round(v * 255);
      rgb[i + 1] = Math.round(Math.sqrt(v) * 220);
      rgb[i + 2] = Math.round((1 - v) * 120);
    }
  }
  return rgb;
}

// === Main Loop ===
let _loopErrorShown = false;
let _diagDone = false;
function mainLoop(timestamp) {
  if (!isRunning) return;
  requestAnimationFrame(mainLoop);

  if (isPaused) return;

  try {
  const elapsed = performance.now() / 1000 - startTime;
  const totalStart = performance.now();

  // --- Video Pipeline ---
  let videoEmb = null;
  let motionRegion = null;
  let videoBrightness = 0;
  let videoMotion = 0;
  if (mode !== 'csi' && videoCapture.isActive) {
    const t0 = performance.now();
    const frame = videoCapture.captureFrame(56, 56);
    if (frame) {
      videoEmb = visualCnn.extract(frame.rgb, frame.width, frame.height);
      motionRegion = videoCapture.detectMotionRegion(56, 56);
      videoBrightness = frame.brightness;
      videoMotion = frame.motion;
    }
    latency.video = performance.now() - t0;
  }

  // --- CSI Pipeline ---
  let csiEmb = null;
  const csiSnapshot = csiSimulator.getSnapshot();
  if (mode !== 'video') {
    const t0 = performance.now();
    if (csiSnapshot?.heatmap) {
      renderer.drawCsiHeatmap(csiCtx, csiSnapshot.heatmap, csiCanvas.width, csiCanvas.height);
      const csiRgb = heatmapToRgb(csiSnapshot.heatmap, 56, 56);
      csiEmb = csiCnn.extract(csiRgb, 56, 56);
    } else {
      renderer.drawCsiHeatmap(csiCtx, null, csiCanvas.width, csiCanvas.height);
    }

    latency.csi = performance.now() - t0;
  }
  fusionEngine.updateConfidence(
    videoBrightness,
    videoMotion,
    csiSnapshot?.snr || 0,
    Boolean(csiSnapshot?.active && csiEmb)
  );

  // --- Fusion ---
  const t0f = performance.now();
  const fusedEmb = fusionEngine.fuse(videoEmb, csiEmb, mode);
  latency.fusion = performance.now() - t0f;

  // --- Pose Decode ---
  const csiState = {
    csiPresence: csiSnapshot?.presence || 0,
    isLive: Boolean(csiSnapshot?.active)
  };

  const keypoints = poseDecoder.decode(fusedEmb, motionRegion, elapsed, csiState);

  // --- Render Skeleton ---
  const labelMap = { dual: 'DUAL FUSION', video: 'VIDEO ONLY', csi: 'CSI ONLY' };
  renderer.drawSkeleton(skeletonCtx, keypoints, skeletonCanvas.width, skeletonCanvas.height, {
    minConfidence: confidenceThreshold,
    color: mode === 'csi' ? 'amber' : 'green',
    label: labelMap[mode]
  });

  // --- Render Embedding Space ---
  const embPoints = fusionEngine.getEmbeddingPoints();
  renderer.drawEmbeddingSpace(embeddingCtx, embPoints, embeddingCanvas.width, embeddingCanvas.height);

  // --- Update UI ---
  latency.total = performance.now() - totalStart;

  // FPS
  frameCount++;
  if (timestamp - lastFpsTime > 500) {
    fps = Math.round(frameCount * 1000 / (timestamp - lastFpsTime));
    lastFpsTime = timestamp;
    frameCount = 0;
    fpsDisplay.textContent = `${fps} FPS`;
  }

  // Fusion bars
  const vc = fusionEngine.videoConfidence;
  const cc = fusionEngine.csiConfidence;
  const fc = fusionEngine.fusedConfidence;
  videoBar.style.width = `${vc * 100}%`;
  csiBar.style.width = `${cc * 100}%`;
  fusedBar.style.width = `${fc * 100}%`;
  videoBarVal.textContent = `${Math.round(vc * 100)}%`;
  csiBarVal.textContent = `${Math.round(cc * 100)}%`;
  fusedBarVal.textContent = `${Math.round(fc * 100)}%`;

  // Latency
  latVideoEl.textContent = `${latency.video.toFixed(1)}ms`;
  latCsiEl.textContent = `${latency.csi.toFixed(1)}ms`;
  latFusionEl.textContent = `${latency.fusion.toFixed(1)}ms`;
  latTotalEl.textContent = `${latency.total.toFixed(1)}ms`;

  // Cross-modal similarity
  const sim = fusionEngine.getCrossModalSimilarity();
  crossModalEl.textContent = sim.toFixed(3);

  // RuVector attention pipeline stats
  const rvStats = poseDecoder.attentionStats;
  const rvEnergyEl = document.getElementById('rv-energy');
  const rvRefineEl = document.getElementById('rv-refine');
  const rvImpactEl = document.getElementById('rv-impact');
  if (rvEnergyEl) rvEnergyEl.textContent = (rvStats.energy || 0).toFixed(2);
  if (rvRefineEl) rvRefineEl.textContent = ((rvStats.refinementMag || 0) * 1000).toFixed(1) + 'px';
  if (rvImpactEl) {
    const impact = Math.min(100, (rvStats.refinementMag || 0) * 5000);
    rvImpactEl.textContent = impact.toFixed(0) + '%';
  }
  // Pulse the pipeline stages when active
  if (visualCnn.useRuVector && rvStats.energy > 0.1) {
    document.querySelectorAll('.rv-stage').forEach(el => el.classList.add('active'));
  }

  // RSSI update
  updateRssi(csiSnapshot?.rssi ?? null);

  // One-time diagnostic
  if (!_diagDone) {
    _diagDone = true;
    console.log(`[PoseFusion] frame 1 OK — mode=${mode}, liveCsi=${Boolean(csiSnapshot?.active)}, embPts=${embPoints?.fused?.length ?? 0}`);
  }

  } catch (err) {
    if (!_loopErrorShown) {
      _loopErrorShown = true;
      console.error('[MainLoop]', err);
      // Show error visually on page
      const errDiv = document.createElement('div');
      errDiv.style.cssText = 'position:fixed;bottom:60px;left:24px;right:24px;background:rgba(255,48,64,0.95);color:#fff;padding:12px 16px;border-radius:8px;font:12px/1.4 "JetBrains Mono",monospace;z-index:9999;max-height:120px;overflow:auto';
      errDiv.textContent = `[MainLoop Error] ${err.message}\n${err.stack?.split('\n').slice(0,3).join('\n')}`;
      document.body.appendChild(errDiv);
    }
  }
}

// === RSSI Visualization ===
function updateRssi(dbm) {
  if (!rssiBarEl) return;
  if (typeof dbm !== 'number' || !Number.isFinite(dbm)) {
    rssiBarEl.style.width = '0%';
    rssiValueEl.textContent = '-- dBm';
    rssiQualityEl.textContent = 'No live RSSI';
    if (rssiSparkCtx) {
      rssiSparkCtx.clearRect(0, 0, rssiSparkCanvas.width, rssiSparkCanvas.height);
    }
    return;
  }

  // Clamp to typical WiFi range: -100 (worst) to -30 (best)
  const clamped = Math.max(-100, Math.min(-30, dbm));
  const pct = ((clamped + 100) / 70) * 100; // 0-100%

  rssiBarEl.style.width = `${pct}%`;
  rssiValueEl.textContent = `${Math.round(clamped)} dBm`;

  // Quality label
  let quality;
  if (clamped > -50) quality = 'Excellent';
  else if (clamped > -60) quality = 'Good';
  else if (clamped > -70) quality = 'Fair';
  else if (clamped > -80) quality = 'Weak';
  else quality = 'Poor';
  rssiQualityEl.textContent = quality;

  // Color the dBm value based on quality
  if (clamped > -60) rssiValueEl.style.color = 'var(--green-glow)';
  else if (clamped > -75) rssiValueEl.style.color = 'var(--amber)';
  else rssiValueEl.style.color = 'var(--red-alert)';

  // Sparkline history
  rssiHistory.push(clamped);
  if (rssiHistory.length > RSSI_HISTORY_MAX) rssiHistory.shift();
  drawRssiSparkline();
}

function drawRssiSparkline() {
  if (!rssiSparkCtx || rssiHistory.length < 2) return;
  const w = rssiSparkCanvas.width;
  const h = rssiSparkCanvas.height;
  const ctx = rssiSparkCtx;

  ctx.clearRect(0, 0, w, h);

  // Draw signal strength line
  const len = rssiHistory.length;
  const step = w / (RSSI_HISTORY_MAX - 1);

  // Gradient fill under line
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0,210,120,0.3)');
  grad.addColorStop(1, 'rgba(0,210,120,0)');

  ctx.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (RSSI_HISTORY_MAX - len + i) * step;
    const y = h - ((rssiHistory[i] + 100) / 70) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  // Fill area
  const lastX = (RSSI_HISTORY_MAX - 1) * step;
  const firstX = (RSSI_HISTORY_MAX - len) * step;
  ctx.lineTo(lastX, h);
  ctx.lineTo(firstX, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw line on top
  ctx.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (RSSI_HISTORY_MAX - len + i) * step;
    const y = h - ((rssiHistory[i] + 100) / 70) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#00d878';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Pulsing dot at latest value
  const latestX = lastX;
  const latestY = h - ((rssiHistory[len - 1] + 100) / 70) * h;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 300);
  ctx.beginPath();
  ctx.arc(latestX, latestY, 2 + pulse, 0, Math.PI * 2);
  ctx.fillStyle = '#00d878';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(latestX, latestY, 4 + pulse * 2, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(0,216,120,${0.3 + pulse * 0.3})`;
  ctx.lineWidth = 1;
  ctx.stroke();
}

// Boot
document.addEventListener('DOMContentLoaded', init);
