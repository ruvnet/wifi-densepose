/**
 * Sensing WebSocket Service
 *
 * Manages the connection to the Python sensing WebSocket server
 * (ws://localhost:8765) and provides a callback-based API for the UI.
 *
 * While reconnecting the WebSocket stays in "reconnecting" state and does NOT
 * emit generated frames. The desktop HTTP API can still emit live hardware
 * frames derived from `/api/v1/status`.
 */

const SENSING_WS_PORT_BY_HTTP_PORT = {
  // Docker image: HTTP UI/API on 3000, sensing stream on 3001.
  '3000': '3001',
  // Python sensing stack: UI on 8080, sensing stream on 8765.
  '8080': '8765',
};

export function buildSensingWsUrl(locationLike = (typeof window !== 'undefined' ? window.location : null)) {
  const protocol = locationLike && locationLike.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = locationLike && locationLike.host ? locationLike.host : 'localhost:3001';
  const hostname = locationLike && locationLike.hostname ? locationLike.hostname : host.split(':')[0];
  const port = locationLike && locationLike.port ? locationLike.port : '';
  const wsPort = SENSING_WS_PORT_BY_HTTP_PORT[port];
  const wsHost = wsPort ? `${hostname}:${wsPort}` : host;

  return `${protocol}//${wsHost}/ws/sensing`;
}

const SENSING_WS_URL = buildSensingWsUrl();
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000];
const MAX_RECONNECT_ATTEMPTS = 20;
const STATUS_POLL_INTERVAL = 1000; // ms

class SensingService {
  constructor() {
    /** @type {WebSocket|null} */
    this._ws = null;
    this._listeners = new Set();
    this._stateListeners = new Set();
    this._reconnectAttempt = 0;
    this._reconnectTimer = null;
    this._statusPollTimer = null;
    // Connection state: disconnected | connecting | connected | reconnecting
    this._state = 'disconnected';
    // Data-source label exposed to the UI:
    //   "live"              — real ESP32 hardware connected
    //   "stale"             — hardware UDP is live, but pose feature state is stale
    //   "reconnecting"      — WebSocket/status disconnected, retrying
    this._dataSource = 'reconnecting';
    // The raw source string from the server (e.g. "esp32", "none")
    this._serverSource = null;
    // True when the HTTP API is reachable. The desktop launcher exposes the
    // live Cardputer feed over HTTP even when the optional WebSocket is absent.
    this._httpBackendReachable = false;
    this._lastMessage = null;
    this._httpFrameSeq = 0;

    // Ring buffer of recent RSSI values for sparkline
    this._rssiHistory = [];
    this._maxHistory = 60;
  }

  // ---- Public API --------------------------------------------------------

  /** Start the service. */
  start() {
    this._startStatusPolling();
    this._connect();
  }

  /** Stop the service entirely. */
  stop() {
    this._clearTimers();
    if (this._ws) {
      this._ws.close(1000, 'client stop');
      this._ws = null;
    }
    this._setState('disconnected');
  }

  /** Register a callback for sensing data updates. Returns unsubscribe fn. */
  onData(callback) {
    this._listeners.add(callback);
    // Immediately push last known data if available
    if (this._lastMessage) callback(this._lastMessage);
    return () => this._listeners.delete(callback);
  }

  /** Register a callback for connection state changes. Returns unsubscribe fn. */
  onStateChange(callback) {
    this._stateListeners.add(callback);
    callback(this._state);
    return () => this._stateListeners.delete(callback);
  }

  /** Get the RSSI sparkline history (array of floats). */
  getRssiHistory() {
    return [...this._rssiHistory];
  }

  /** Get per-node RSSI history (object keyed by node_id). */
  getPerNodeRssiHistory() {
    return { ...(this._perNodeRssiHistory || {}) };
  }

  /** Current connection state. */
  get state() {
    return this._state;
  }

  /**
   * Current data source label.
   * "live"         — frames are arriving from the real ESP32 over WebSocket
   * "reconnecting" — WebSocket disconnected; actively retrying, no frames emitted
   */
  get dataSource() {
    return this._dataSource;
  }

  // ---- Connection --------------------------------------------------------

  _connect() {
    if (this._ws && this._ws.readyState <= WebSocket.OPEN) return;

    this._setState('connecting');

    try {
      this._ws = new WebSocket(SENSING_WS_URL);
    } catch (err) {
      console.warn('[Sensing] WebSocket constructor failed:', err.message);
      this._setState('reconnecting');
      this._setDataSource('reconnecting');
      return;
    }

    this._ws.onopen = () => {
      console.info('[Sensing] Connected to', SENSING_WS_URL);
      this._reconnectAttempt = 0;
      this._setState('connected');
      // Don't assume "live" yet — wait for first frame's source field.
      // Fetch server status to determine actual data source immediately.
      this._detectServerSource();
    };

    this._ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this._handleData(data);
      } catch (e) {
        console.warn('[Sensing] Invalid message:', e.message);
      }
    };

    this._ws.onerror = () => {
      // onerror is always followed by onclose, so we handle reconnect there
    };

    this._ws.onclose = (evt) => {
      console.info('[Sensing] Connection closed (code=%d)', evt.code);
      this._ws = null;
      if (evt.code !== 1000) {
        this._scheduleReconnect();
      } else {
        this._setState('disconnected');
        this._setDataSource('reconnecting');
      }
    };
  }

  _scheduleReconnect() {
    if (this._reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('[Sensing] Max reconnect attempts (%d) reached; no live data source', MAX_RECONNECT_ATTEMPTS);
      this._setState('disconnected');
      this._setDataSource('reconnecting');
      return;
    }

    const delay = RECONNECT_DELAYS[Math.min(this._reconnectAttempt, RECONNECT_DELAYS.length - 1)];
    this._reconnectAttempt++;
    console.info('[Sensing] Reconnecting in %dms (attempt %d/%d)', delay, this._reconnectAttempt, MAX_RECONNECT_ATTEMPTS);

    this._setState('reconnecting');
    if (!this._httpBackendReachable) {
      this._setDataSource('reconnecting');
    }

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connect();
    }, delay);

  }

  // ---- Server source detection -------------------------------------------

  /**
   * Fetch `/api/v1/status` to find out if the server is using real
   * hardware. Called once on WebSocket open.
   */
  async _detectServerSource() {
    try {
      const resp = await fetch('/api/v1/status');
      if (resp.ok) {
        this._httpBackendReachable = true;
        const json = await resp.json();
        const hardware = json.hardware || {};
        const hardwareLive = Boolean(hardware.live);
        if (hardwareLive && hardware.feature_state_live === false && hardware.stale_feature_state) {
          this._applyServerSource('stale');
        } else if (hardwareLive) {
          this._applyServerSource('esp32');
        } else {
          this._applyServerSource(json.source);
        }
        const frame = this._statusToSensingData(json);
        if (frame) {
          this._publishData(frame);
        }
      } else {
        this._httpBackendReachable = false;
        this._setDataSource('reconnecting');
      }
    } catch {
      this._httpBackendReachable = false;
      this._setDataSource('reconnecting');
    }
  }

  _startStatusPolling() {
    if (this._statusPollTimer) return;
    this._detectServerSource();
    this._statusPollTimer = setInterval(() => {
      this._detectServerSource();
    }, STATUS_POLL_INTERVAL);
  }

  /**
   * Map a raw server source string to the UI data-source label.
   */
  _applyServerSource(rawSource) {
    this._serverSource = rawSource;
    if (rawSource === 'esp32' || rawSource === 'wifi' || rawSource === 'live') {
      this._setDataSource('live');
    } else if (rawSource === 'stale') {
      this._setDataSource('stale');
    } else if (rawSource === 'none') {
      this._setDataSource('reconnecting');
    } else {
      this._setDataSource('reconnecting');
    }
  }

  /** @return {string|null} Raw server source (e.g. "esp32", "none") */
  get serverSource() {
    return this._serverSource;
  }

  // ---- Data handling -----------------------------------------------------

  _handleData(data) {
    // Track the server's source field from each frame so the UI
    // can react if the server switches source at runtime.
    if (data.source && this._state === 'connected') {
      const raw = data.source;
      if (raw !== this._serverSource) {
        this._applyServerSource(raw);
      }
    }

    this._publishData(data);
  }

  _publishData(data) {
    this._lastMessage = data;

    // Update RSSI history for sparkline
    if (data.features && data.features.mean_rssi != null) {
      this._rssiHistory.push(data.features.mean_rssi);
      if (this._rssiHistory.length > this._maxHistory) {
        this._rssiHistory.shift();
      }
    }

    // Per-node RSSI tracking
    if (!this._perNodeRssiHistory) this._perNodeRssiHistory = {};
    if (data.node_features) {
      for (const nf of data.node_features) {
        if (!this._perNodeRssiHistory[nf.node_id]) {
          this._perNodeRssiHistory[nf.node_id] = [];
        }
        this._perNodeRssiHistory[nf.node_id].push(nf.rssi_dbm);
        if (this._perNodeRssiHistory[nf.node_id].length > this._maxHistory) {
          this._perNodeRssiHistory[nf.node_id].shift();
        }
      }
    }

    // Notify all listeners
    for (const cb of this._listeners) {
      try {
        cb(data);
      } catch (e) {
        console.error('[Sensing] Listener error:', e);
      }
    }
  }

  _statusToSensingData(status) {
    const hardware = status?.hardware || status || {};
    const nodes = Array.isArray(hardware.nodes) ? hardware.nodes : [];
    const hardwareLive = Boolean(hardware.live || nodes.some(node => node.live));
    if (!hardwareLive && nodes.length === 0) return null;

    const liveNodes = nodes.filter(node => node.live);
    const primary = liveNodes.find(node => node.feature_state_live || node.edge_feature_live || node.edge_vitals_live)
      || liveNodes[0]
      || nodes[0]
      || hardware;
    const primaryVisual = this._nodeToVisualFeature(primary, 0, Math.max(nodes.length, 1));
    const featureState = primary.feature_state || hardware.feature_state || {};
    const edgeVitals = primary.edge_vitals || hardware.edge_vitals || {};
    const edgeFeature = primary.edge_feature || hardware.edge_feature || {};
    const confidence = primaryVisual.classification.confidence;
    const presence = Boolean(
      hardwareLive
      && (
        featureState.presence
        || edgeVitals.presence
        || primaryVisual.features.presence_norm >= 0.35
      )
    );
    const motion = primaryVisual.features.motion_band_power;
    const breathing = primaryVisual.features.breathing_band_power;
    const heartbeat = this._clamp01(this._number(edgeFeature.heartbeat_norm, this._number(featureState.heartbeat_conf, 0.8)));
    const variance = primaryVisual.features.variance;
    const rssi = primaryVisual.rssi_dbm;
    const visualNodes = (nodes.length > 0 ? nodes : [hardware])
      .map((node, index, allNodes) => this._nodeToVisualFeature(node, index, allNodes.length));
    const motionLevel = !presence ? 'absent' : motion > 0.65 ? 'active' : 'present_still';

    const features = {
      mean_rssi: rssi,
      variance,
      std: Math.sqrt(Math.max(variance, 0)),
      motion_band_power: motion,
      breathing_band_power: breathing,
      dominant_freq_hz: this._number(featureState.respiration_bpm, this._number(edgeVitals.breathing_bpm, 0)) / 60,
      change_points: this._number(featureState.anomaly_score, 0) > 0.5 ? 1 : 0,
      spectral_power: this._clamp01(Math.max(motion, breathing, heartbeat, variance)),
      range: Math.abs(this._number(primary.noise_floor_dbm, -96) - rssi),
      iqr: variance / 2,
      skewness: 0,
      kurtosis: 0,
    };
    const classification = {
      motion_level: motionLevel,
      presence,
      confidence,
      fall_detected: Boolean(edgeVitals.fall),
    };

    return {
      type: 'sensing_update',
      timestamp: Date.now() / 1000,
      sequence: ++this._httpFrameSeq,
      source: status?.source || 'esp32',
      nodes: visualNodes.map(node => ({
        node_id: node.node_id,
        rssi_dbm: node.rssi_dbm,
        position: node.position,
        source_addr: node.source_addr,
        packet_count: node.packet_count,
      })),
      node_features: visualNodes,
      features,
      classification,
      signal_field: this._generateSignalField(features, classification, visualNodes),
      vital_signs: {
        heart_rate_bpm: this._number(featureState.heartbeat_bpm, this._number(edgeVitals.heartbeat_bpm, 0)),
        breathing_rate_bpm: this._number(featureState.respiration_bpm, this._number(edgeVitals.breathing_bpm, 0)),
      },
    };
  }

  _nodeToVisualFeature(node, index, total) {
    const featureState = node?.feature_state || {};
    const edgeVitals = node?.edge_vitals || {};
    const edgeFeature = node?.edge_feature || {};
    const features = Array.isArray(edgeFeature.features) ? edgeFeature.features : [];
    const nodeId = this._number(node?.node_id, index + 1);
    const angle = total > 1
      ? (index / total) * Math.PI * 2
      : ((nodeId % 8) / 8) * Math.PI * 2;
    const radius = total > 1 ? 6 : 4.5;
    const presenceScore = this._number(featureState.presence_score, this._number(edgeVitals.presence_score, 0));
    const motionScore = this._number(featureState.motion_score, this._number(edgeVitals.motion_energy, 0));
    const presenceNorm = this._clamp01(this._number(edgeFeature.presence_norm, this._number(features[0], presenceScore / 10)));
    const motionNorm = this._clamp01(this._number(edgeFeature.motion_norm, this._number(features[1], motionScore)));
    const breathingNorm = this._clamp01(this._number(edgeFeature.breathing_norm, this._number(features[2], this._number(featureState.respiration_conf, 0) / 10)));
    const varianceNorm = this._clamp01(this._number(edgeFeature.phase_variance_norm, this._number(features[4], this._number(featureState.node_coherence, 0.2))));
    const rssi = this._number(node?.rssi_dbm, this._number(edgeVitals.rssi_dbm, -80));
    const presence = Boolean(node?.live && (featureState.presence || edgeVitals.presence || presenceNorm >= 0.35));
    const confidence = this._clamp01(presenceNorm || motionNorm);
    const motionLevel = !presence ? 'absent' : motionNorm > 0.65 ? 'active' : 'present_still';

    return {
      node_id: nodeId,
      live: Boolean(node?.live),
      stale: !node?.live,
      rssi_dbm: rssi,
      position: [
        Math.cos(angle) * radius,
        0,
        Math.sin(angle) * radius,
      ],
      source_addr: node?.last_source || '',
      packet_count: this._number(node?.packet_count, 0),
      features: {
        variance: varianceNorm,
        presence_norm: presenceNorm,
        motion_band_power: motionNorm,
        breathing_band_power: breathingNorm,
      },
      classification: {
        presence,
        motion_level: motionLevel,
        confidence,
      },
    };
  }

  _generateSignalField(features, classification, nodes) {
    const gridSize = 20;
    const values = [];
    const t = Date.now() / 1000;
    const bodyX = Math.sin(t * 0.45) * 2.2;
    const bodyZ = Math.cos(t * 0.33) * 1.8;
    const motion = this._clamp01(features.motion_band_power);
    const breathing = this._clamp01(features.breathing_band_power);
    const confidence = this._clamp01(classification.confidence);

    for (let z = 0; z < gridSize; z++) {
      for (let x = 0; x < gridSize; x++) {
        const wx = (x - gridSize / 2) + 0.5;
        const wz = (z - gridSize / 2) + 0.5;
        const distCenter = Math.hypot(wx, wz);
        let value = 0.03 + Math.max(0, 1 - distCenter / 14) * 0.18;

        for (const node of nodes) {
          const [nx, , nz] = node.position || [0, 0, 0];
          const distNode = Math.hypot(wx - nx, wz - nz);
          const nodeEnergy = this._clamp01(
            node.features.presence_norm * 0.4
            + node.features.motion_band_power * 0.4
            + node.features.variance * 0.2
          );
          value += Math.exp(-(distNode * distNode) / 18) * (0.08 + nodeEnergy * 0.28);
        }

        if (classification.presence) {
          const distBody = Math.hypot(wx - bodyX, wz - bodyZ);
          value += Math.exp(-(distBody * distBody) / 10) * (0.22 + motion * 0.5 + confidence * 0.25);
          const breathRadius = 2.5 + Math.sin(t * 2.0) * 0.6;
          const ring = Math.exp(-Math.pow(distBody - breathRadius, 2) / 1.4);
          value += ring * breathing * 0.45;
        }

        values.push(this._clamp01(value));
      }
    }

    return {
      grid_size: [gridSize, 1, gridSize],
      values,
    };
  }

  _number(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  _clamp01(value) {
    return Math.max(0, Math.min(1, this._number(value, 0)));
  }

  // ---- State management --------------------------------------------------

  _setState(newState) {
    if (newState === this._state) return;
    this._state = newState;
    for (const cb of this._stateListeners) {
      try { cb(newState); } catch (e) { /* ignore */ }
    }
  }

  /**
   * Update the dataSource label and notify state listeners so the UI can
   * react without needing a separate subscription.
   * @param {'live'|'stale'|'reconnecting'} source
   */
  _setDataSource(source) {
    if (source === this._dataSource) return;
    this._dataSource = source;
    // Re-use the same state-listener channel — listeners receive the
    // connection state but can read dataSource via service.dataSource.
    for (const cb of this._stateListeners) {
      try { cb(this._state); } catch (e) { /* ignore */ }
    }
  }

  _clearTimers() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._statusPollTimer) {
      clearInterval(this._statusPollTimer);
      this._statusPollTimer = null;
    }
  }
}

// Singleton
export const sensingService = new SensingService();
