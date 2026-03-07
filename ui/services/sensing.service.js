/**
 * Sensing WebSocket Service (real-only mode)
 *
 * Connects to `/ws/sensing` and only forwards real frames.
 * No client-side mock or simulation fallback is allowed.
 */

const _wsProto = (typeof window !== 'undefined' && window.location.protocol === 'https:') ? 'wss:' : 'ws:';
const _wsHost = (typeof window !== 'undefined' && window.location.host) ? window.location.host : 'localhost:3000';
const SENSING_WS_URL = `${_wsProto}//${_wsHost}/ws/sensing`;
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000];
const MAX_RECONNECT_ATTEMPTS = 20;
const NON_REAL_SOURCES = new Set(['simulated', 'simulate', 'mock', 'demo', 'animated_demo']);
const REAL_SOURCES = new Set(['esp32', 'wifi', 'live', 'csi', 'hardware']);

class SensingService {
  constructor() {
    /** @type {WebSocket|null} */
    this._ws = null;
    this._listeners = new Set();
    this._stateListeners = new Set();
    this._reconnectAttempt = 0;
    this._reconnectTimer = null;
    this._state = 'disconnected';
    this._dataSource = 'offline';
    this._serverSource = null;
    this._lastMessage = null;
    this._warnedNonRealData = false;
    this._rssiHistory = [];
    this._maxHistory = 60;
  }

  start() {
    this._connect();
  }

  stop() {
    this._clearTimers();
    if (this._ws) {
      this._ws.close(1000, 'client stop');
      this._ws = null;
    }
    this._setState('disconnected');
    this._setDataSource('offline');
  }

  onData(callback) {
    this._listeners.add(callback);
    if (this._lastMessage) callback(this._lastMessage);
    return () => this._listeners.delete(callback);
  }

  onStateChange(callback) {
    this._stateListeners.add(callback);
    callback(this._state);
    return () => this._stateListeners.delete(callback);
  }

  getRssiHistory() {
    return [...this._rssiHistory];
  }

  get state() {
    return this._state;
  }

  /**
   * Current data source label:
   * - `live`: only verified real stream data is being consumed
   * - `reconnecting`: transport reconnect in progress
   * - `offline`: no live stream available
   */
  get dataSource() {
    return this._dataSource;
  }

  get serverSource() {
    return this._serverSource;
  }

  _connect() {
    if (this._ws && this._ws.readyState <= WebSocket.OPEN) return;

    this._setState('connecting');
    this._setDataSource('reconnecting');

    try {
      this._ws = new WebSocket(SENSING_WS_URL);
    } catch (err) {
      console.warn('[Sensing] WebSocket constructor failed:', err.message);
      this._setState('disconnected');
      this._setDataSource('offline');
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      console.info('[Sensing] Connected to', SENSING_WS_URL);
      this._reconnectAttempt = 0;
      this._setState('connected');
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
      // handled on close
    };

    this._ws.onclose = (evt) => {
      console.info('[Sensing] Connection closed (code=%d)', evt.code);
      this._ws = null;
      if (evt.code !== 1000) {
        this._scheduleReconnect();
      } else {
        this._setState('disconnected');
        this._setDataSource('offline');
      }
    };
  }

  _scheduleReconnect() {
    if (this._reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('[Sensing] Max reconnect attempts (%d) reached, no simulation fallback allowed', MAX_RECONNECT_ATTEMPTS);
      this._setState('disconnected');
      this._setDataSource('offline');
      return;
    }

    const delay = RECONNECT_DELAYS[Math.min(this._reconnectAttempt, RECONNECT_DELAYS.length - 1)];
    this._reconnectAttempt++;
    this._setState('reconnecting');
    this._setDataSource('reconnecting');
    console.info('[Sensing] Reconnecting in %dms (attempt %d/%d)', delay, this._reconnectAttempt, MAX_RECONNECT_ATTEMPTS);

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connect();
    }, delay);
  }

  async _detectServerSource() {
    try {
      const resp = await fetch('/api/v1/status');
      if (resp.ok) {
        const json = await resp.json();
        this._applyServerSource(json.source);
      } else {
        this._setDataSource('reconnecting');
      }
    } catch {
      this._setDataSource('reconnecting');
    }
  }

  _applyServerSource(rawSource) {
    const normalized = String(rawSource || '').toLowerCase();
    this._serverSource = normalized || null;

    if (!normalized) return;
    if (REAL_SOURCES.has(normalized)) {
      this._setDataSource('live');
      return;
    }
    if (NON_REAL_SOURCES.has(normalized)) {
      this._setDataSource('offline');
      if (!this._warnedNonRealData) {
        this._warnedNonRealData = true;
        console.warn('[Sensing] Non-real server source detected and rejected:', normalized);
      }
      return;
    }

    this._setDataSource('reconnecting');
  }

  _isNonRealFrame(data) {
    if (!data || typeof data !== 'object') return true;

    const source = String(data.source || data?.metadata?.source || '').toLowerCase();
    if (source && NON_REAL_SOURCES.has(source)) return true;
    if (data._simulated === true) return true;
    if (data?.metadata?.mock_data === true) return true;
    return false;
  }

  _handleData(data) {
    if (data?.source && this._state === 'connected') {
      this._applyServerSource(data.source);
    }

    if (this._isNonRealFrame(data)) {
      if (!this._warnedNonRealData) {
        this._warnedNonRealData = true;
        console.warn('[Sensing] Received non-real frame. Frame ignored.');
      }
      return;
    }

    this._lastMessage = data;
    this._setDataSource('live');

    if (data.features && data.features.mean_rssi != null) {
      this._rssiHistory.push(data.features.mean_rssi);
      if (this._rssiHistory.length > this._maxHistory) {
        this._rssiHistory.shift();
      }
    }

    for (const cb of this._listeners) {
      try {
        cb(data);
      } catch (e) {
        console.error('[Sensing] Listener error:', e);
      }
    }
  }

  _setState(newState) {
    if (newState === this._state) return;
    this._state = newState;
    for (const cb of this._stateListeners) {
      try {
        cb(newState);
      } catch {
        // ignore listener errors
      }
    }
  }

  _setDataSource(source) {
    if (source === this._dataSource) return;
    this._dataSource = source;
    for (const cb of this._stateListeners) {
      try {
        cb(this._state);
      } catch {
        // ignore listener errors
      }
    }
  }

  _clearTimers() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }
}

export const sensingService = new SensingService();