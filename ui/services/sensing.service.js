import { withWsTicket } from './ws-ticket.js';
/**
 * Sensing WebSocket Service
 *
 * Manages the connection to the Python sensing WebSocket server
 * (ws://localhost:8765) and provides a callback-based API for the UI.
 *
 * Falls back to simulated data only after MAX_RECONNECT_ATTEMPTS exhausted.
 * While reconnecting the service stays in "reconnecting" state and does NOT
 * emit simulated frames so the UI can clearly distinguish live vs. fallback data.
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

class SensingService {
  constructor() {
    /** @type {WebSocket|null} */
    this._ws = null;
    this._listeners = new Set();
    this._stateListeners = new Set();
    this._reconnectAttempt = 0;
    this._reconnectTimer = null;
    // Connection state: disconnected | connecting | connected | reconnecting
    this._state = 'disconnected';
    // Data-source label exposed to the UI:
    //   "live"                  — real ESP32 hardware connected
    //   "waiting_for_hardware"  — server is running but no real hardware detected
    //   "reconnecting"          — WebSocket disconnected, retrying
    this._dataSource = 'reconnecting';
    // The raw source string from the server (e.g. "esp32", "simulated", "simulate")
    this._serverSource = null;
    this._lastMessage = null;

    // Ring buffer of recent RSSI values for sparkline
    this._rssiHistory = [];
    this._maxHistory = 60;
  }

  // ---- Public API --------------------------------------------------------

  /** Start the service (connect or simulate). */
  start() {
    void this._connect();
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
   * "live"                  — frames are arriving from the real ESP32 over WebSocket
   * "reconnecting"          — WebSocket disconnected; actively retrying, no frames emitted
   * "waiting_for_hardware"  — server is running but no real hardware has been detected
   */
  get dataSource() {
    return this._dataSource;
  }

  // ---- Connection --------------------------------------------------------

  // async because the server gates `/ws/sensing` (ADR-272) and a browser
  // cannot set an Authorization header on an upgrade — so we mint a
  // single-use ticket first. Minted per connect attempt, never cached: a
  // ticket is valid once and expires in seconds, so reusing one across
  // reconnects would fail on the second attempt.
  async _connect() {
    if (this._ws && this._ws.readyState <= WebSocket.OPEN) return;

    this._setState('connecting');

    let url = SENSING_WS_URL;
    try {
      url = await withWsTicket(SENSING_WS_URL);
    } catch {
      // Ticket minting is best-effort: against a server with auth off, or one
      // predating ADR-272, connecting without a ticket is correct.
    }

    try {
      this._ws = new WebSocket(url);
    } catch (err) {
      console.warn('[Sensing] WebSocket constructor failed:', err.message);
      this._setState('reconnecting');
      this._setDataSource('waiting_for_hardware');
      this._scheduleReconnect();
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
      console.warn('[Sensing] Max reconnect attempts (%d) reached, will continue polling', MAX_RECONNECT_ATTEMPTS);
      // Stop retrying to avoid hammering the server, but keep the reconnecting state
      this._setState('reconnecting');
      this._setDataSource('waiting_for_hardware');
      return;
    }

    const delay = RECONNECT_DELAYS[Math.min(this._reconnectAttempt, RECONNECT_DELAYS.length - 1)];
    this._reconnectAttempt++;
    console.info('[Sensing] Reconnecting in %dms (attempt %d/%d)', delay, this._reconnectAttempt, MAX_RECONNECT_ATTEMPTS);

    this._setState('reconnecting');
    this._setDataSource('reconnecting');

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      void this._connect();
    }, delay);
  }

  // ---- Simulation fallback removed in issue #1125 -----------------------------------------------
  // When no hardware is present, the service now idles in "waiting_for_hardware" state
  // instead of fabricating synthetic data. This prevents user confusion about whether
  // they are looking at real or fake data.

  _stopSimulation() {
    // No-op placeholder — simulation fallback has been removed.
  }

  // ---- Server source detection -------------------------------------------

  /**
   * Fetch `/api/v1/status` to find out if the server is using real
   * hardware or simulation. Called once on WebSocket open.
   */
  async _detectServerSource() {
    try {
      const resp = await fetch('/api/v1/status');
      if (resp.ok) {
        const json = await resp.json();
        this._applyServerSource(json.source);
      } else {
        // Can't reach status endpoint — assume live until first frame tells us
        this._setDataSource('live');
      }
    } catch {
      this._setDataSource('live');
    }
  }

  /**
   * Map a raw server source string to the UI data-source label.
   */
  _applyServerSource(rawSource) {
    this._serverSource = rawSource;
    if (rawSource === 'esp32' || rawSource === 'wifi' || rawSource === 'live') {
      this._setDataSource('live');
    } else if (rawSource === 'waiting_for_hardware') {
      this._setDataSource('waiting_for_hardware');
    } else {
      // Unknown source or old enum values — report as waiting
      this._setDataSource('waiting_for_hardware');
    }
  }

  /** @return {string|null} Raw server source (e.g. "esp32", "simulated") */
  get serverSource() {
    return this._serverSource;
  }

  // ---- Data handling -----------------------------------------------------

  _handleData(data) {
    this._lastMessage = data;

    // Track the server's source field from each frame so the UI
    // can react if the server switches between esp32 ↔ simulated at runtime.
    if (data.source && this._state === 'connected') {
      const raw = data.source;
      if (raw !== this._serverSource) {
        this._applyServerSource(raw);
      }
    }

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
   * @param {'live'|'waiting_for_hardware'|'reconnecting'} source
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
  }
}

// Singleton
export const sensingService = new SensingService();
