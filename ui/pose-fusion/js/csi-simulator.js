/**
 * CSI Simulator — Generates realistic WiFi Channel State Information data.
 *
 * In live mode, connects to the sensing server via WebSocket.
 * In demo mode, generates synthetic CSI that correlates with detected motion.
 *
 * Outputs: 3-channel pseudo-image (amplitude, phase, temporal diff)
 * matching the ADR-018 frame format expectations.
 */

export class CsiSimulator {
  static VERSION = 'v4-drift';  // Cache-bust verification

  constructor(opts = {}) {
    this.subcarriers = opts.subcarriers || 52; // 802.11n HT20
    this.timeWindow = opts.timeWindow || 56;   // frames in sliding window
    // Provenance of the frames currently rendered. A socket alone is not
    // enough to claim live sensing; `live` is entered only after a validated,
    // explicitly hardware-sourced frame.
    this.mode = 'demo'; // demo | connecting | server-simulated | offline | unverified | live
    this._modeListeners = new Set();
    this._connectionGeneration = 0;
    this._pendingConnect = null;
    this.connectTimeoutMs = opts.connectTimeoutMs ?? 3000;
    this.ws = null;

    // Circular buffer for CSI frames
    this.amplitudeBuffer = [];
    this.phaseBuffer = [];
    this.frameCount = 0;

    // Noise parameters
    this._rng = this._mulberry32(opts.seed || 7);
    this._noiseState = new Float32Array(this.subcarriers);
    this._baseAmplitude = new Float32Array(this.subcarriers);
    this._basePhase = new Float32Array(this.subcarriers);

    // Initialize base CSI profile (empty room)
    for (let i = 0; i < this.subcarriers; i++) {
      this._baseAmplitude[i] = 0.5 + 0.3 * Math.sin(i * 0.12);
      this._basePhase[i] = (i / this.subcarriers) * Math.PI * 2;
    }

    // RSSI tracking
    this.rssiDbm = -70; // default mid-range
    this._rssiTarget = -70;
    this._hasServerRssi = false;

    // Person influence (updated from video motion)
    this.personPresence = 0;
    this.personX = 0.5;
    this.personY = 0.5;
    this.personMotion = 0;
  }

  /**
   * Connect to live sensing server WebSocket
   * @param {string} url - WebSocket URL (e.g. ws://localhost:3030/ws/csi)
   */
  async connectLive(url) {
    if (this._pendingConnect) this._pendingConnect.cancel();

    const generation = ++this._connectionGeneration;
    const previousSocket = this.ws;
    if (previousSocket) {
      previousSocket.onopen = null;
      previousSocket.onmessage = null;
      previousSocket.onerror = null;
      previousSocket.onclose = null;
      try { previousSocket.close(); } catch { /* already closed */ }
    }
    this.ws = null;
    this._liveAmplitude = null;
    this._livePhase = null;
    this._hasServerRssi = false;
    this.rssiDbm = -70;
    this._rssiTarget = -70;
    this._setMode('connecting');

    return new Promise((resolve) => {
      let settled = false;
      let timeoutId = null;
      const pending = { cancel: null };
      const finish = (source) => {
        if (settled) return;
        settled = true;
        if (timeoutId !== null) clearTimeout(timeoutId);
        if (this._pendingConnect === pending) this._pendingConnect = null;
        resolve(source);
      };
      pending.cancel = () => finish(null);
      this._pendingConnect = pending;

      try {
        const socket = new WebSocket(url);
        this.ws = socket;
        socket.binaryType = 'arraybuffer';
        socket.onmessage = async (evt) => {
          if (generation !== this._connectionGeneration || this.ws !== socket) return;

          let payload = evt.data;
          if (typeof Blob !== 'undefined' && payload instanceof Blob) {
            try {
              payload = await payload.arrayBuffer();
            } catch {
              return;
            }
            // The connection may have been replaced while the Blob decoded.
            if (generation !== this._connectionGeneration || this.ws !== socket) return;
          }

          // Parsing is synchronous, so no other connection can supersede this
          // socket between the guard and the state mutation.
          const source = this._handleLiveFrame(payload);
          if (
            !source
            || generation !== this._connectionGeneration
            || this.ws !== socket
          ) return;
          this._setMode(source);
          finish(source);
        };
        // Opening the transport does not prove that usable or live CSI exists.
        socket.onopen = () => {};
        socket.onerror = () => {
          if (generation !== this._connectionGeneration || this.ws !== socket) return;
          this._setMode('demo');
          finish(null);
        };
        socket.onclose = () => {
          if (generation !== this._connectionGeneration || this.ws !== socket) return;
          this.ws = null;
          this._liveAmplitude = null;
          this._livePhase = null;
          this._setMode('demo');
          finish(null);
        };
        timeoutId = setTimeout(() => {
          if (generation !== this._connectionGeneration || this.ws !== socket) return;
          this._setMode('demo');
          finish(null);
          try { socket.close(); } catch { /* already closed */ }
        }, this.connectTimeoutMs);
      } catch {
        this._setMode('demo');
        finish(null);
      }
    });
  }

  disconnect() {
    this._connectionGeneration++;
    if (this._pendingConnect) this._pendingConnect.cancel();
    if (this.ws) {
      const socket = this.ws;
      this.ws = null;
      try { socket.close(); } catch { /* already closed */ }
    }
    this._liveAmplitude = null;
    this._livePhase = null;
    this._setMode('demo');
  }

  /** True only once a real frame has been decoded — not merely on socket open. */
  get isLive() { return this.mode === 'live'; }

  /** Subscribe to live/demo source changes. Returns an unsubscribe function. */
  onModeChange(callback) {
    this._modeListeners.add(callback);
    callback(this.mode);
    return () => this._modeListeners.delete(callback);
  }

  _setMode(mode) {
    if (mode === this.mode) return;
    const previousWasSynthetic = this.mode === 'demo' || this.mode === 'connecting';
    const nextIsSynthetic = mode === 'demo' || mode === 'connecting';
    // Buffered rows feed both the heatmap and CNN pseudo-image. Never carry
    // rows across provenance domains: otherwise a newly-live badge could sit
    // above a window that is still mostly demo or stale CSI.
    this.amplitudeBuffer.length = 0;
    this.phaseBuffer.length = 0;
    if (!nextIsSynthetic) {
      // Never carry RSSI across provenance domains. This also covers runtime
      // promotion from server-simulated/offline/unverified to live hardware,
      // not only the initial demo-to-server transition.
      this.rssiDbm = this._hasServerRssi ? this._rssiTarget : null;
    } else if (!previousWasSynthetic && nextIsSynthetic) {
      this._hasServerRssi = false;
      this.rssiDbm = -70;
      this._rssiTarget = -70;
    }
    this.mode = mode;
    for (const callback of this._modeListeners) {
      try { callback(mode); } catch { /* source labels must not break sensing */ }
    }
  }

  /**
   * Update person state from video detection (for correlated demo data).
   * When person exits frame, CSI maintains presence with slow decay
   * (simulating through-wall sensing capability).
   */
  updatePersonState(presence, x, y, motion) {
    // Don't override real CSI sensing with synthetic video-derived state
    if (this.mode !== 'demo' && this.mode !== 'connecting') return;

    if (presence > 0.1) {
      // Person detected in video — update CSI state directly
      this.personPresence = presence;
      this.personX = x;
      this.personY = y;
      this.personMotion = motion;
      this._lastSeenTime = performance.now();
      this._lastSeenX = x;
      this._lastSeenY = y;
    } else if (this._lastSeenTime) {
      // Person NOT in video — CSI "through-wall" persistence
      const elapsed = (performance.now() - this._lastSeenTime) / 1000;
      // CSI can sense through walls for ~10 seconds with decaying confidence
      const decayRate = 0.15; // Lose ~15% per second
      this.personPresence = Math.max(0, 1.0 - elapsed * decayRate);
      // Position slowly drifts (person walking behind wall)
      this.personX = this._lastSeenX;
      this.personY = this._lastSeenY;
      this.personMotion = Math.max(0, motion * 0.5 + this.personPresence * 0.2);

      if (this.personPresence < 0.05) {
        this._lastSeenTime = null;
      }
    } else {
      this.personPresence = 0;
      this.personMotion = 0;
    }
  }

  /**
   * Generate next CSI frame (demo mode) or return latest live frame
   * @param {number} elapsed - Time in seconds
   * @returns {{ amplitude: Float32Array, phase: Float32Array, snr: number }}
   */
  nextFrame(elapsed) {
    const amp = new Float32Array(this.subcarriers);
    const phase = new Float32Array(this.subcarriers);

    if (this.mode === 'demo' || this.mode === 'connecting') {
      this._generateDemoFrame(amp, phase, elapsed);
    } else if (this._liveAmplitude) {
      amp.set(this._liveAmplitude);
      phase.set(this._livePhase);
    }

    // Push to circular buffer
    this.amplitudeBuffer.push(new Float32Array(amp));
    this.phaseBuffer.push(new Float32Array(phase));
    if (this.amplitudeBuffer.length > this.timeWindow) {
      this.amplitudeBuffer.shift();
      this.phaseBuffer.shift();
    }

    // RSSI: smooth toward target (demo mode generates synthetic RSSI)
    if (this.mode === 'demo' || this.mode === 'connecting') {
      // Simulate RSSI based on person presence and slow drift
      this._rssiTarget = -55 - 25 * (1 - this.personPresence) + Math.sin(elapsed * 0.3) * 3;
    }
    if (Number.isFinite(this.rssiDbm) && Number.isFinite(this._rssiTarget)) {
      this.rssiDbm += (this._rssiTarget - this.rssiDbm) * 0.1;
    }

    // SNR estimate
    let signalPower = 0, noisePower = 0;
    for (let i = 0; i < this.subcarriers; i++) {
      signalPower += amp[i] * amp[i];
      noisePower += this._noiseState[i] * this._noiseState[i];
    }
    const snr = noisePower > 0 ? 10 * Math.log10(signalPower / noisePower) : 30;

    this.frameCount++;
    return { amplitude: amp, phase, snr: Math.max(0, Math.min(40, snr)) };
  }

  /**
   * Build 3-channel pseudo-image for CNN input
   * @param {number} targetSize - Output image dimension (square)
   * @returns {Uint8Array} RGB data (targetSize * targetSize * 3)
   */
  buildPseudoImage(targetSize = 56) {
    const buf = this.amplitudeBuffer;
    const pBuf = this.phaseBuffer;
    const frames = buf.length;
    if (frames < 2) {
      return new Uint8Array(targetSize * targetSize * 3);
    }

    const rgb = new Uint8Array(targetSize * targetSize * 3);

    for (let y = 0; y < targetSize; y++) {
      const fi = Math.min(Math.floor(y / targetSize * frames), frames - 1);
      for (let x = 0; x < targetSize; x++) {
        const si = Math.min(Math.floor(x / targetSize * this.subcarriers), this.subcarriers - 1);
        const idx = (y * targetSize + x) * 3;

        // R: Amplitude (normalized to 0-255)
        const ampVal = buf[fi][si];
        rgb[idx] = Math.min(255, Math.max(0, Math.floor(ampVal * 255)));

        // G: Phase (wrapped to 0-255)
        const phaseVal = (pBuf[fi][si] % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
        rgb[idx + 1] = Math.floor(phaseVal / (2 * Math.PI) * 255);

        // B: Temporal difference
        if (fi > 0) {
          const diff = Math.abs(buf[fi][si] - buf[fi - 1][si]);
          rgb[idx + 2] = Math.min(255, Math.floor(diff * 500));
        }
      }
    }

    return rgb;
  }

  /**
   * Get heatmap data for visualization
   * @returns {{ data: Float32Array, width: number, height: number }}
   */
  getHeatmapData() {
    const frames = this.amplitudeBuffer.length;
    const w = this.subcarriers;
    const h = Math.min(frames, this.timeWindow);
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const fi = frames - h + y;
      if (fi >= 0 && fi < frames) {
        for (let x = 0; x < w; x++) {
          data[y * w + x] = this.amplitudeBuffer[fi][x];
        }
      }
    }
    return { data, width: w, height: h };
  }

  // === Private ===

  _generateDemoFrame(amp, phase, elapsed) {
    const rng = this._rng;
    const presence = this.personPresence;
    const motion = this.personMotion;
    const px = this.personX;

    for (let i = 0; i < this.subcarriers; i++) {
      // Base CSI profile (frequency-selective channel)
      let a = this._baseAmplitude[i];
      let p = this._basePhase[i] + elapsed * 0.05;

      // Environmental noise (correlated across subcarriers)
      this._noiseState[i] = 0.95 * this._noiseState[i] + 0.05 * (rng() * 2 - 1) * 0.03;
      a += this._noiseState[i];

      // Ambient temporal drift (multipath fading even in empty room)
      a += 0.06 * Math.sin(elapsed * 0.7 + i * 0.25)
         + 0.04 * Math.sin(elapsed * 1.3 - i * 0.18)
         + 0.03 * Math.cos(elapsed * 2.1 + i * 0.4);

      // Person-induced CSI perturbation
      if (presence > 0.1) {
        // Subcarrier-dependent body reflection (Fresnel zone model)
        const freqOffset = (i - this.subcarriers * px) / (this.subcarriers * 0.3);
        const bodyReflection = presence * 0.25 * Math.exp(-freqOffset * freqOffset);

        // Motion causes amplitude fluctuation
        const motionEffect = motion * 0.15 * Math.sin(elapsed * 3.5 + i * 0.3);

        // Breathing modulation (0.2-0.3 Hz)
        const breathing = presence * 0.02 * Math.sin(elapsed * 1.5 + i * 0.05);

        a += bodyReflection + motionEffect + breathing;
        p += presence * 0.4 * Math.sin(elapsed * 2.1 + i * 0.15);
      }

      amp[i] = Math.max(0, Math.min(1, a));
      phase[i] = p;
    }
  }

  _handleLiveFrame(data) {
    // Handle JSON text frames from the sensing server
    if (typeof data === 'string') {
      try {
        const msg = JSON.parse(data);
        return this._handleJsonFrame(msg);
      } catch (_) { /* ignore malformed JSON */ }
      return null;
    }

    // Handle binary ArrayBuffer frames (ADR-018 format)
    if (!(data instanceof ArrayBuffer)) return null;
    const view = new DataView(data);
    // Check ADR-018 magic: 0xC5110001
    if (data.byteLength < 20) return null;
    const magic = view.getUint32(0, true);
    if (magic !== 0xC5110001) return null;

    const numSub = Math.min(view.getUint16(8, true), this.subcarriers);
    if (numSub === 0) return null;
    this._liveAmplitude = new Float32Array(this.subcarriers);
    this._livePhase = new Float32Array(this.subcarriers);

    const headerSize = 20;
    let parsed = 0;
    for (let i = 0; i < numSub && (headerSize + i * 4 + 3) < data.byteLength; i++) {
      const real = view.getInt16(headerSize + i * 4, true);
      const imag = view.getInt16(headerSize + i * 4 + 2, true);
      this._liveAmplitude[i] = Math.sqrt(real * real + imag * imag) / 2048;
      this._livePhase[i] = Math.atan2(imag, real);
      parsed++;
    }
    if (parsed === 0) return null;
    // Binary ADR-018 frames do not carry RSSI metadata. Treat it as unknown
    // for this frame instead of retaining a value from another provenance.
    this._hasServerRssi = false;
    this._rssiTarget = null;
    this.rssiDbm = null;
    // ADR-018 carries no trustworthy source string, so the transport is usable
    // but its hardware provenance remains explicitly unverified.
    return 'unverified';
  }

  _handleJsonFrame(msg) {
    // Extract amplitude from sensing_update node data
    const node = (msg.nodes && msg.nodes[0]) || msg;
    const rawSource = String(node.source || msg.source || '').toLowerCase();
    const explicitlySimulated = msg._simulated === true
      || node._simulated === true
      || rawSource === 'simulate'
      || rawSource === 'simulated'
      || rawSource.endsWith(':simulated');
    let frameSource = 'unverified';
    let confirmedHardware = false;
    if (explicitlySimulated) {
      frameSource = 'server-simulated';
    } else if (rawSource.endsWith(':offline')) {
      frameSource = 'offline';
    } else {
      const hardwarePrefixes = ['esp32', 'wifi', 'realtek', 'mediatek', 'qualcomm'];
      confirmedHardware = hardwarePrefixes.some((prefix) => (
        rawSource === prefix || rawSource.startsWith(`${prefix}:`)
      ));
      confirmedHardware = confirmedHardware || rawSource === 'live';
    }

    const ampArr = node.amplitude || msg.amplitude;
    const iq = node.iq || msg.iq;
    const hasAmplitude = Array.isArray(ampArr)
      && ampArr.some((value) => Number.isFinite(Number(value)));
    const hasIq = Array.isArray(iq)
      && iq.length >= 2
      && Number.isFinite(Number(iq[0]))
      && Number.isFinite(Number(iq[1]));
    // ADR-295: a hardware label alone is not enough to claim LIVE CSI. Only a
    // decoded CSI payload from an explicitly hardware-sourced frame promotes
    // the UI; payloadless hardware metadata remains visibly unverified.
    if (confirmedHardware && (hasAmplitude || hasIq)) frameSource = 'live';

    // Metadata remains authoritative on privacy-gated and vitals-only frames
    // that intentionally omit raw CSI samples.
    this._hasServerRssi = false;
    this._rssiTarget = null;
    if (typeof node.rssi_dbm === 'number') {
      this._rssiTarget = node.rssi_dbm;
      this._hasServerRssi = true;
    } else if (msg.features && typeof msg.features.mean_rssi === 'number') {
      this._rssiTarget = msg.features.mean_rssi;
      this._hasServerRssi = true;
    } else {
      // RSSI belongs to the current frame. Never reuse a simulated, offline,
      // or otherwise differently sourced value beneath a newer source label.
      this.rssiDbm = null;
    }
    if (this._hasServerRssi && !Number.isFinite(this.rssiDbm)) {
      // Recover immediately when RSSI metadata resumes within the same source
      // mode; the smoothing path cannot advance from an explicit null value.
      this.rssiDbm = this._rssiTarget;
    }
    const cls = msg.classification;
    if (cls && typeof cls.confidence === 'number') {
      this.personPresence = cls.presence ? cls.confidence : 0;
    }

    if (!hasAmplitude && !hasIq) {
      // Source transitions are authoritative even when privacy gating, a
      // vitals-only path, or an offline re-broadcast omits raw CSI. Clear the
      // previous frame so stale/live data is never rendered under the new
      // provenance label.
      this._liveAmplitude = null;
      this._livePhase = null;
      return frameSource;
    }

    // Sensing server sends: { type: "sensing_update", nodes: [{ amplitude: [...], subcarrier_count }], classification, features }
    this._liveAmplitude = new Float32Array(this.subcarriers);
    this._livePhase = new Float32Array(this.subcarriers);

    if (hasAmplitude) {
      const n = Math.min(ampArr.length, this.subcarriers);
      // Server sends raw amplitude (already magnitude), normalize to 0-1
      let maxAmp = 0;
      for (let i = 0; i < n; i++) maxAmp = Math.max(maxAmp, Math.abs(Number(ampArr[i]) || 0));
      const scale = maxAmp > 0 ? 1.0 / maxAmp : 1.0;
      for (let i = 0; i < n; i++) {
        this._liveAmplitude[i] = Math.abs(Number(ampArr[i]) || 0) * scale;
      }
    }

    // Phase from node (if available)
    const phaseArr = node.phase || msg.phase;
    if (phaseArr && Array.isArray(phaseArr)) {
      const n = Math.min(phaseArr.length, this.subcarriers);
      for (let i = 0; i < n; i++) this._livePhase[i] = phaseArr[i];
    } else if (ampArr) {
      // Synthesize phase from amplitude variation (Hilbert-like estimate)
      for (let i = 1; i < this.subcarriers; i++) {
        this._livePhase[i] = this._livePhase[i - 1] + (this._liveAmplitude[i] - this._liveAmplitude[i - 1]) * Math.PI;
      }
    }

    // Handle raw I/Q pairs
    if (hasIq) {
      const n = Math.min(iq.length / 2, this.subcarriers);
      for (let i = 0; i < n; i++) {
        const real = Number(iq[i * 2]) || 0;
        const imag = Number(iq[i * 2 + 1]) || 0;
        this._liveAmplitude[i] = Math.sqrt(real * real + imag * imag) / 2048;
        this._livePhase[i] = Math.atan2(imag, real);
      }
    }

    return frameSource;
  }

  _mulberry32(seed) {
    return function() {
      let t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
}
