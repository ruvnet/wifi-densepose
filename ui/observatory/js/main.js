/**
 * RuView Observatory — Main Scene Orchestrator
 *
 * Room-based WiFi sensing visualization with:
 * - Pool of 4 human wireframe figures (multi-person scenarios)
 * - 7 pose types (standing, walking, lying, sitting, fallen, exercising, gesturing, crouching)
 * - Scenario-specific room props (chair, exercise mat, door, rubble wall, screen, desk)
 * - Dot-matrix mist body mass, particle trails, WiFi waves, signal field
 * - Reflective floor, settings dialog, and practical data HUD
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { NebulaBackground } from './nebula-background.js';
import { PostProcessing } from './post-processing.js';
import { FigurePool, SKELETON_PAIRS } from './figure-pool.js';
import { PoseSystem } from './pose-system.js';
import { ScenarioProps } from './scenario-props.js';
import { HudController, DEFAULTS, SETTINGS_VERSION, PRESETS, SCENARIO_NAMES } from './hud-controller.js';

// ---- Palette ----
const C = {
  greenGlow:  0x00d878,
  greenBright:0x3eff8a,
  greenDim:   0x0a6b3a,
  amber:      0xffb020,
  blueSignal: 0x2090ff,
  redAlert:   0xff3040,
  redHeart:   0xff4060,
  bgDeep:     0x080c14,
};

// SCENARIO_NAMES, DEFAULTS, SETTINGS_VERSION, PRESETS imported from hud-controller.js

// ---- Main Class ----

class Observatory {
  constructor() {
    this._canvas = document.getElementById('observatory-canvas');
    this.settings = { ...DEFAULTS };

    // Load saved settings
    try {
      const ver = localStorage.getItem('ruview-settings-version');
      if (ver === SETTINGS_VERSION) {
        const saved = localStorage.getItem('ruview-observatory-settings');
        if (saved) Object.assign(this.settings, JSON.parse(saved));
      } else {
        localStorage.removeItem('ruview-observatory-settings');
        localStorage.setItem('ruview-settings-version', SETTINGS_VERSION);
      }
    } catch {}

    // Renderer
    this._renderer = new THREE.WebGLRenderer({
      canvas: this._canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.setSize(window.innerWidth, window.innerHeight);
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = this.settings.exposure;
    this._renderer.shadowMap.enabled = true;
    this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Scene
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(C.bgDeep);
    this._scene.fog = new THREE.FogExp2(C.bgDeep, 0.005);

    // Camera
    this._camera = new THREE.PerspectiveCamera(
      this.settings.fov, window.innerWidth / window.innerHeight, 0.1, 300
    );
    this._camera.position.set(6, 5, 8);
    this._camera.lookAt(0, 1.2, 0);

    // Controls
    this._controls = new OrbitControls(this._camera, this._canvas);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.08;
    this._controls.minDistance = 2;
    this._controls.maxDistance = 25;
    this._controls.maxPolarAngle = Math.PI * 0.88;
    this._controls.target.set(0, 1.2, 0);
    this._controls.update();

    this._clock = new THREE.Clock();

    // Data: live-only. Do not fabricate poses, vitals, RSSI, or presence.
    this._currentData = null;
    this._currentScenario = null;
    this._focusedNodeId = null;
    this._livePaused = false;

    // Build scene
    this._setupLighting();
    this._nebula = new NebulaBackground(this._scene);
    this._buildRoom();
    this._buildRouter();
    this._poseSystem = new PoseSystem();
    this._figurePool = new FigurePool(this._scene, this.settings, this._poseSystem);
    this._scenarioProps = new ScenarioProps(this._scene);
    this._buildDotMatrixMist();
    this._buildParticleTrail();
    this._buildWifiWaves();
    this._buildSignalField();

    // Post-processing
    this._postProcessing = new PostProcessing(this._renderer, this._scene, this._camera);
    this._applyPostSettings();

    // HUD controller (settings dialog, sparkline, vital displays)
    this._hud = new HudController(this);

    // State
    this._autopilot = false;
    this._autoAngle = 0;
    this._fpsFrames = 0;
    this._fpsTime = 0;
    this._fpsValue = 60;
    this._showFps = false;
    this._qualityLevel = 2;

    // WebSocket for live data — always try auto-detect on startup
    this._ws = null;
    this._liveData = null;
    this._autoDetectLive();

    // Input
    this._initKeyboard();
    this._hud.initSettings();
    this._hud.initQuickSelect();
    window.addEventListener('resize', () => this._onResize());

    // Start
    this._animate();
  }

  // ---- Lighting ----

  _setupLighting() {
    this._ambient = new THREE.AmbientLight(0xccccdd, this.settings.ambient * 5.0);
    this._scene.add(this._ambient);

    const hemi = new THREE.HemisphereLight(0x6688bb, 0x203040, 1.2);
    this._scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffeedd, 1.2);
    key.position.set(4, 8, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 20;
    key.shadow.camera.left = -8;
    key.shadow.camera.right = 8;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -8;
    this._scene.add(key);

    // Fill light from opposite side
    const fill = new THREE.DirectionalLight(0x8899bb, 0.7);
    fill.position.set(-4, 5, -2);
    this._scene.add(fill);

    // Rim light from above/behind for edge definition
    const rim = new THREE.DirectionalLight(0x6699cc, 0.5);
    rim.position.set(0, 6, -5);
    this._scene.add(rim);

    // Overhead room light — general illumination
    const overhead = new THREE.PointLight(0x8899aa, 1.0, 20, 1.0);
    overhead.position.set(0, 3.8, 0);
    this._scene.add(overhead);
  }

  // ---- Room ----

  _buildRoom() {
    this._grid = new THREE.GridHelper(12, 24, 0x1a4830, 0x0c2818);
    this._grid.material.opacity = 0.5;
    this._grid.material.transparent = true;
    this._scene.add(this._grid);

    const boxGeo = new THREE.BoxGeometry(12, 4, 10);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this._roomWire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
      color: C.greenDim, opacity: 0.3, transparent: true,
    }));
    this._roomWire.position.y = 2;
    this._scene.add(this._roomWire);

    // Reflective floor
    const floorGeo = new THREE.PlaneGeometry(12, 10);
    this._floorMat = new THREE.MeshStandardMaterial({
      color: 0x101810,
      roughness: 1.0 - this.settings.reflect * 0.7,
      metalness: this.settings.reflect * 0.5,
      emissive: 0x020404,
      emissiveIntensity: 0.08,
    });
    const floor = new THREE.Mesh(floorGeo, this._floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this._scene.add(floor);

    // Table under router
    const tableGeo = new THREE.BoxGeometry(0.8, 0.6, 0.5);
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x6b5840, roughness: 0.55, emissive: 0x1a1408, emissiveIntensity: 0.25 });
    const table = new THREE.Mesh(tableGeo, tableMat);
    table.position.set(-4, 0.3, -3);
    table.castShadow = true;
    this._scene.add(table);
  }

  // ---- Router ----

  _buildRouter() {
    this._routerGroup = new THREE.Group();
    this._routerGroup.position.set(-4, 0.92, -3);

    const bodyGeo = new THREE.BoxGeometry(0.6, 0.12, 0.35);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x505060, roughness: 0.2, metalness: 0.7, emissive: 0x101018, emissiveIntensity: 0.2 });
    this._routerGroup.add(new THREE.Mesh(bodyGeo, bodyMat));

    for (let i = -1; i <= 1; i++) {
      const antGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.35);
      const antMat = new THREE.MeshStandardMaterial({ color: 0x606068, roughness: 0.3, metalness: 0.6, emissive: 0x101018, emissiveIntensity: 0.15 });
      const ant = new THREE.Mesh(antGeo, antMat);
      ant.position.set(i * 0.2, 0.24, 0);
      ant.rotation.z = i * 0.15;
      this._routerGroup.add(ant);
    }

    const ledGeo = new THREE.SphereGeometry(0.025);
    this._routerLed = new THREE.Mesh(ledGeo, new THREE.MeshBasicMaterial({ color: C.greenGlow }));
    this._routerLed.position.set(0.22, 0.07, 0.18);
    this._routerGroup.add(this._routerLed);

    this._routerLight = new THREE.PointLight(C.blueSignal, 1.2, 8);
    this._routerLight.position.set(0, 0.3, 0);
    this._routerGroup.add(this._routerLight);

    this._scene.add(this._routerGroup);
  }

  // ---- WiFi Waves ----

  _buildWifiWaves() {
    this._wifiWaves = [];
    for (let i = 0; i < 5; i++) {
      const radius = 0.8 + i * 1.0;
      const geo = new THREE.SphereGeometry(radius, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.6);
      const mat = new THREE.MeshBasicMaterial({
        color: C.blueSignal,
        transparent: true, opacity: 0,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false, wireframe: true,
      });
      const shell = new THREE.Mesh(geo, mat);
      shell.position.copy(this._routerGroup.position);
      shell.position.y += 0.5;
      this._scene.add(shell);
      this._wifiWaves.push({ mesh: shell, mat, phase: i * 0.7 });
    }
  }

  // ========================================
  // DOT MATRIX MIST
  // ========================================

  _buildDotMatrixMist() {
    const COUNT = 800;
    const positions = new Float32Array(COUNT * 3);
    const alphas = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 0.5;
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = Math.random() * 1.8;
      positions[i * 3 + 2] = Math.sin(angle) * r;
      alphas[i] = 0;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        void main() {
          vAlpha = alpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = 3.0 * (200.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vAlpha;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float edge = smoothstep(0.5, 0.2, d);
          gl_FragColor = vec4(uColor, edge * vAlpha);
        }
      `,
      uniforms: { uColor: { value: new THREE.Color(this.settings.wireColor) } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this._mistPoints = new THREE.Points(geo, mat);
    this._scene.add(this._mistPoints);
    this._mistCount = COUNT;
  }

  // ---- Particle Trail ----

  _buildParticleTrail() {
    const COUNT = 200;
    const positions = new Float32Array(COUNT * 3);
    const ages = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) ages[i] = 1;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('age', new THREE.BufferAttribute(ages, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float age;
        varying float vAge;
        void main() {
          vAge = age;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = max(1.0, (1.0 - age) * 5.0 * (150.0 / -mv.z));
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vAge;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float alpha = (1.0 - vAge) * 0.6 * smoothstep(0.5, 0.1, d);
          gl_FragColor = vec4(uColor, alpha);
        }
      `,
      uniforms: { uColor: { value: new THREE.Color(C.greenGlow) } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this._trail = new THREE.Points(geo, mat);
    this._scene.add(this._trail);
    this._trailHead = 0;
    this._trailCount = COUNT;
    this._trailTimer = 0;
  }

  // ---- Signal Field ----

  _buildSignalField() {
    const gridSize = 20;
    const count = gridSize * gridSize;
    const positions = new Float32Array(count * 3);
    this._fieldColors = new Float32Array(count * 3);
    this._fieldSizes = new Float32Array(count);
    for (let iz = 0; iz < gridSize; iz++) {
      for (let ix = 0; ix < gridSize; ix++) {
        const idx = iz * gridSize + ix;
        positions[idx * 3] = (ix - gridSize / 2) * 0.6;
        positions[idx * 3 + 1] = 0.02;
        positions[idx * 3 + 2] = (iz - gridSize / 2) * 0.5;
        this._fieldSizes[idx] = 8;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this._fieldColors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(this._fieldSizes, 1));
    this._fieldMat = new THREE.PointsMaterial({
      size: 0.35, vertexColors: true, transparent: true,
      opacity: this.settings.field, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true,
    });
    this._fieldPoints = new THREE.Points(geo, this._fieldMat);
    this._scene.add(this._fieldPoints);
  }

  // ---- Keyboard ----

  _initKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (this._hud.settingsOpen) return;
      switch (e.key.toLowerCase()) {
        case 'a':
          this._autopilot = !this._autopilot;
          this._controls.enabled = !this._autopilot;
          break;
        case 'd':
          this._cycleLiveNode();
          break;
        case 'f':
          this._showFps = !this._showFps;
          document.getElementById('fps-counter').style.display = this._showFps ? 'block' : 'none';
          break;
        case 'r':
          this._autoDetectLive();
          break;
        case 's': this._hud.toggleSettings(); break;
        case ' ':
          e.preventDefault();
          this._livePaused = !this._livePaused;
          this._hud.updatePauseBadge(this._livePaused);
          break;
      }
    });
  }

  // ---- Settings / HUD methods delegated to HudController ----

  _applyPostSettings() {
    const pp = this._postProcessing;
    pp._bloomPass.strength = this.settings.bloom;
    pp._bloomPass.radius = this.settings.bloomRadius;
    pp._bloomPass.threshold = this.settings.bloomThresh;
    pp._vignettePass.uniforms.uVignetteStrength.value = this.settings.vignette;
    pp._vignettePass.uniforms.uGrainStrength.value = this.settings.grain;
    pp._vignettePass.uniforms.uChromaticStrength.value = this.settings.chromatic;
  }

  _applyColors() {
    const wc = new THREE.Color(this.settings.wireColor);
    const jc = new THREE.Color(this.settings.jointColor);
    this._figurePool.applyColors(wc, jc);
    this._mistPoints.material.uniforms.uColor.value.copy(wc);
  }

  // ---- WebSocket live data ----

  _autoDetectLive() {
    // Probe sensing server health on same origin, then common ports
    const host = window.location.hostname || 'localhost';
    const candidates = [
      window.location.origin,                   // same origin (e.g. :3000)
      `http://${host}:8765`,                     // default WS port
      `http://${host}:3000`,                     // default HTTP port
    ];
    // Deduplicate
    const unique = [...new Set(candidates)];

    const tryNext = (i) => {
      if (i >= unique.length) {
        console.log('[Observatory] No sensing server detected; live-only mode is waiting');
        this.settings.dataSource = 'live';
        this._currentData = null;
        this._hud.updateSourceBadge('offline', null);
        return;
      }
      const base = unique[i];
      fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(data => {
          if (data && data.status === 'ok') {
            const wsProto = base.startsWith('https') ? 'wss:' : 'ws:';
            const urlObj = new URL(base);
            const wsUrl = `${wsProto}//${urlObj.host}/ws/sensing`;
            console.log('[Observatory] Sensing server detected at', base, '→', wsUrl);
            this.settings.dataSource = 'ws';
            this.settings.wsUrl = wsUrl;
            this._connectWS(wsUrl);
          } else if (data && (data.ok === true || data.status === 'alive' || data.status === 'healthy')) {
            console.log('[Observatory] HTTP API detected at', base, 'polling live status');
            this.settings.dataSource = 'http';
            this.settings.apiBase = base;
            this._startHttpPolling(base);
          } else {
            tryNext(i + 1);
          }
        })
        .catch(() => tryNext(i + 1));
    };
    tryNext(0);
  }

  _connectWS(url) {
    this._disconnectWS();
    try {
      this._ws = new WebSocket(url);
      this._ws.onopen = () => {
        console.log('[Observatory] WebSocket connected');
        this._hud.updateSourceBadge('ws', this._ws);
      };
      this._ws.onmessage = (evt) => { try { this._liveData = JSON.parse(evt.data); } catch {} };
      this._ws.onclose = () => {
        console.log('[Observatory] WebSocket closed; live-only mode is waiting');
        this._ws = null;
        this._currentData = null;
        this._hud.updateSourceBadge('offline', null);
      };
      this._ws.onerror = () => {};
    } catch {}
  }

  _disconnectWS() {
    if (this._ws) { this._ws.close(); this._ws = null; }
    this._liveData = null;
  }

  _startHttpPolling(base) {
    if (this._httpPollTimer) clearInterval(this._httpPollTimer);
    const poll = () => {
      fetch(`${base}/api/v1/cardputer/status`, { cache: 'no-store' })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(status => {
          if (!this._livePaused) {
            this._liveData = this._statusToFrame(status);
          }
          this._hud.updateSourceBadge(status.live ? 'http' : 'offline', null);
        })
        .catch(() => {
          this._liveData = null;
          this._currentData = null;
          this._hud.updateSourceBadge('offline', null);
        });
    };
    poll();
    this._httpPollTimer = setInterval(poll, 1000);
  }

  _statusToFrame(status) {
    const nodes = Array.isArray(status?.nodes) ? status.nodes : [];
    const liveNodes = nodes.filter(n => n.live);
    const selected = this._focusedNodeId
      ? nodes.find(n => Number(n.node_id) === this._focusedNodeId)
      : null;
    const primary = selected || liveNodes[0] || nodes[0] || {};
    this._focusedNodeId = Number.isFinite(Number(primary.node_id)) ? Number(primary.node_id) : this._focusedNodeId;
    const featureState = primary.feature_state || status?.feature_state || {};
    const edgeVitals = primary.edge_vitals || status?.edge_vitals || {};
    const edgeFeature = primary.edge_feature || status?.edge_feature || {};
    const battery = primary.battery || status?.battery || null;
    const nodeProfiles = nodes.map((node, index) => this._nodeToProfile(node, index, nodes.length));
    const primaryProfile = this._nodeToProfile(primary, 0, Math.max(nodes.length, 1));
    const rssi = Number.isFinite(primary.rssi_dbm)
      ? primary.rssi_dbm
      : (Number.isFinite(edgeVitals.rssi_dbm) ? edgeVitals.rssi_dbm : null);
    const presenceScore = Number(featureState.presence_score ?? edgeVitals.presence_score ?? 0);
    const motionScore = Number(featureState.motion_score ?? edgeVitals.motion_energy ?? 0);
    const presenceNorm = this._clamp01(Number(edgeFeature.presence_norm ?? primaryProfile.presenceNorm ?? presenceScore / 10));
    const motionNorm = this._clamp01(Number(edgeFeature.motion_norm ?? primaryProfile.motionNorm ?? motionScore));
    const breathingNorm = this._clamp01(Number(edgeFeature.breathing_norm ?? primaryProfile.breathingNorm ?? 0));
    const heartbeatNorm = this._clamp01(Number(edgeFeature.heartbeat_norm ?? featureState.heartbeat_conf ?? 0.8));
    const varianceNorm = this._clamp01(Number(edgeFeature.phase_variance_norm ?? featureState.node_coherence ?? 0.2));
    const presence = Boolean(status?.live && (featureState.presence || edgeVitals.presence || presenceNorm >= 0.25));
    const estimatedPersons = this._estimatePersonCount(status, presence, nodes, edgeVitals);
    const motionLevel = !presence ? 'absent' : motionNorm > 0.55 ? 'active' : 'present_still';
    const features = {
      mean_rssi: rssi,
      variance: varianceNorm * 5,
      motion_band_power: motionNorm,
      breathing_band_power: breathingNorm,
      spectral_power: Math.max(0.08, Math.max(motionNorm, breathingNorm, heartbeatNorm, varianceNorm) * 0.5),
    };
    const classification = {
      presence,
      confidence: Math.max(presenceNorm, motionNorm * 0.7),
      motion_level: motionLevel,
      fall_detected: Boolean(edgeVitals.fall),
    };
    const persons = this._estimatePersons(estimatedPersons, primaryProfile, nodeProfiles, classification);
    return {
      source: 'live-http',
      scenario: primary.node_id != null ? `live_node_${primary.node_id}` : 'live',
      focused_node_id: primary.node_id ?? null,
      nodes,
      persons,
      estimated_persons: estimatedPersons,
      classification,
      features,
      signal_field: this._generateSignalField(features, classification, nodeProfiles),
      vital_signs: {
        heart_rate_bpm: Number(featureState.heartbeat_bpm || edgeVitals.heartbeat_bpm || 0),
        breathing_rate_bpm: Number(featureState.respiration_bpm || edgeVitals.breathing_bpm || 0),
      },
      battery,
    };
  }

  _nodeToProfile(node, index, total) {
    const featureState = node?.feature_state || {};
    const edgeVitals = node?.edge_vitals || {};
    const edgeFeature = node?.edge_feature || {};
    const features = Array.isArray(edgeFeature.features) ? edgeFeature.features : [];
    const nodeId = Number.isFinite(Number(node?.node_id)) ? Number(node.node_id) : index + 1;
    const angle = total > 1 ? (index / total) * Math.PI * 2 : ((nodeId % 8) / 8) * Math.PI * 2;
    const radius = total > 1 ? 2.7 : 1.4;
    const presenceScore = Number(featureState.presence_score ?? edgeVitals.presence_score ?? 0);
    const motionScore = Number(featureState.motion_score ?? edgeVitals.motion_energy ?? 0);
    return {
      nodeId,
      position: [Math.cos(angle) * radius, 0, Math.sin(angle) * radius],
      presenceNorm: this._clamp01(Number(edgeFeature.presence_norm ?? features[0] ?? presenceScore / 10)),
      motionNorm: this._clamp01(Number(edgeFeature.motion_norm ?? features[1] ?? motionScore)),
      breathingNorm: this._clamp01(Number(edgeFeature.breathing_norm ?? features[2] ?? 0)),
      varianceNorm: this._clamp01(Number(edgeFeature.phase_variance_norm ?? features[4] ?? featureState.node_coherence ?? 0.2)),
      rssi: Number.isFinite(Number(node?.rssi_dbm)) ? Number(node.rssi_dbm) : Number(edgeVitals.rssi_dbm ?? -80),
      live: Boolean(node?.live),
    };
  }

  _estimatePersonCount(status, presence, nodes, edgeVitals) {
    if (!presence) return 0;
    const counts = nodes
      .map(node => node.edge_vitals)
      .filter(vitals => vitals?.presence)
      .map(vitals => Number(vitals.n_persons))
      .filter(Number.isFinite);
    const edgeCount = Number(edgeVitals?.n_persons);
    if (edgeVitals?.presence && Number.isFinite(edgeCount) && edgeCount > 0) counts.push(edgeCount);
    if (counts.length === 0) return 1;
    return Math.max(1, Math.min(4, Math.max(...counts)));
  }

  _estimatePersons(count, primaryProfile, nodeProfiles, classification) {
    if (count <= 0) return [];
    const baseMotion = this._clamp01(classification.motion_level === 'active' ? primaryProfile.motionNorm : primaryProfile.motionNorm * 0.6);
    return Array.from({ length: count }, (_, index) => {
      const profile = nodeProfiles[index % Math.max(nodeProfiles.length, 1)] || primaryProfile;
      const angle = (index / Math.max(count, 1)) * Math.PI * 2 + Date.now() * 0.00012;
      const ring = count > 1 ? 1.0 + index * 0.35 : 0;
      return {
        id: `live-${profile.nodeId}-${index}`,
        source: 'hardware_inference',
        position: [
          profile.position[0] * 0.35 + Math.cos(angle) * ring,
          0,
          profile.position[2] * 0.35 + Math.sin(angle) * ring,
        ],
        pose: classification.motion_level === 'active' ? 'walking' : 'standing',
        facing: angle + Math.PI,
        motion_score: 18 + baseMotion * 82,
      };
    });
  }

  _generateSignalField(features, classification, nodeProfiles) {
    const gridSize = 20;
    const values = [];
    const t = Date.now() / 1000;
    const motion = this._clamp01(features.motion_band_power);
    const breathing = this._clamp01(features.breathing_band_power);
    const confidence = this._clamp01(classification.confidence);
    const bodyX = Math.sin(t * 0.35) * 1.5;
    const bodyZ = Math.cos(t * 0.27) * 1.3;

    for (let z = 0; z < gridSize; z++) {
      for (let x = 0; x < gridSize; x++) {
        const wx = (x - gridSize / 2) / 2;
        const wz = (z - gridSize / 2) / 2;
        let value = 0.04 + Math.max(0, 1 - Math.hypot(wx, wz) / 7) * 0.15;
        for (const node of nodeProfiles) {
          const dist = Math.hypot(wx - node.position[0], wz - node.position[2]);
          const nodeEnergy = this._clamp01(node.presenceNorm * 0.4 + node.motionNorm * 0.4 + node.varianceNorm * 0.2);
          value += Math.exp(-(dist * dist) / 5) * (0.08 + nodeEnergy * 0.28);
        }
        if (classification.presence) {
          const distBody = Math.hypot(wx - bodyX, wz - bodyZ);
          value += Math.exp(-(distBody * distBody) / 3.6) * (0.22 + motion * 0.48 + confidence * 0.18);
          const breathRadius = 1.4 + Math.sin(t * 2.0) * 0.3;
          value += Math.exp(-Math.pow(distBody - breathRadius, 2) / 0.55) * breathing * 0.35;
        }
        values.push(this._clamp01(value));
      }
    }
    return {
      grid_size: [gridSize, 1, gridSize],
      values,
    };
  }

  _clamp01(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.max(0, Math.min(1, number));
  }

  _cycleLiveNode() {
    const nodes = Array.isArray(this._liveData?.nodes) ? this._liveData.nodes : [];
    if (nodes.length === 0) return;
    const ids = nodes
      .map(n => Number(n.node_id))
      .filter(id => Number.isFinite(id))
      .sort((a, b) => a - b);
    if (ids.length === 0) return;
    const currentIdx = ids.indexOf(this._focusedNodeId);
    this._focusedNodeId = ids[(currentIdx + 1) % ids.length];
    this._liveData = this._statusToFrame({ nodes, live: nodes.some(n => n.live) });
  }

  // ========================================
  // ANIMATION LOOP
  // ========================================

  _animate() {
    requestAnimationFrame(() => this._animate());
    const dt = Math.min(this._clock.getDelta(), 0.1);
    const elapsed = this._clock.getElapsedTime();

    // Data source
    if ((this.settings.dataSource === 'ws' || this.settings.dataSource === 'http') && this._liveData) {
      this._currentData = this._liveData;
    } else {
      this._currentData = null;
    }
    const data = this._currentData;

    // Updates
    this._nebula.update(dt, elapsed);
    this._figurePool.update(data, elapsed);
    this._scenarioProps.update(data, data?.scenario || 'live');
    this._updateDotMatrixMist(data, elapsed);
    this._updateParticleTrail(data, dt, elapsed);
    this._updateWifiWaves(elapsed);
    this._updateSignalField(data);
    this._hud.updateHUD(data);
    this._hud.updateSparkline(data);

    // Router LED
    this._routerLed.material.opacity = 0.5 + 0.5 * Math.sin(elapsed * 8);
    this._routerLight.intensity = 0.3 + 0.2 * Math.sin(elapsed * 3);

    // Autopilot orbit
    if (this._autopilot) {
      this._autoAngle += dt * this.settings.orbitSpeed;
      const r = 10;
      this._camera.position.set(
        Math.sin(this._autoAngle) * r,
        4.5 + Math.sin(this._autoAngle * 0.5),
        Math.cos(this._autoAngle) * r
      );
      this._controls.target.set(0, 1.2, 0);
      this._controls.update();
    }
    this._controls.update();
    this._postProcessing.update(elapsed);
    this._postProcessing.render();
    this._updateFPS(dt);
  }


  // ========================================
  // MIST & TRAIL
  // ========================================

  _updateDotMatrixMist(data, elapsed) {
    const persons = data?.persons || [];
    const isPresent = data?.classification?.presence || false;
    const pos = this._mistPoints.geometry.attributes.position;
    const alpha = this._mistPoints.geometry.attributes.alpha;

    if (!isPresent || persons.length === 0) {
      for (let i = 0; i < this._mistCount; i++) {
        alpha.array[i] = Math.max(0, alpha.array[i] - 0.02);
      }
      alpha.needsUpdate = true;
      return;
    }

    // Follow primary person
    const pp = persons[0].position || [0, 0, 0];
    const px = pp[0] || 0, pz = pp[2] || 0;
    const ms = persons[0].motion_score || 0;
    const pose = persons[0].pose || 'standing';
    const isLying = pose === 'lying' || pose === 'fallen';
    const bodyH = isLying ? 0.4 : 1.7;
    const bodyBaseY = isLying ? (pp[1] || 0) + 0.05 : 0.05;
    const spread = ms > 50 ? 0.6 : 0.4;

    for (let i = 0; i < this._mistCount; i++) {
      const drift = Math.sin(elapsed * 0.5 + i * 0.1) * 0.003;
      const angle = (i / this._mistCount) * Math.PI * 2 + elapsed * 0.1;
      const layerT = (i % 20) / 20;
      const layerY = bodyBaseY + layerT * bodyH;

      let bodyWidth;
      if (isLying) {
        bodyWidth = 0.25;
      } else {
        bodyWidth = layerT > 0.75 ? 0.15 : (layerT > 0.45 ? 0.25 : 0.18);
      }
      const r = bodyWidth * (0.5 + 0.5 * Math.sin(i * 1.7 + elapsed * 0.3)) * spread;

      const tx = px + Math.cos(angle + i * 0.3) * r + drift;
      const tz = pz + Math.sin(angle + i * 0.5) * r * 0.6;

      pos.array[i * 3] += (tx - pos.array[i * 3]) * 0.05;
      pos.array[i * 3 + 1] += (layerY - pos.array[i * 3 + 1]) * 0.05;
      pos.array[i * 3 + 2] += (tz - pos.array[i * 3 + 2]) * 0.05;

      const targetAlpha = 0.15 + Math.sin(elapsed * 2 + i * 0.5) * 0.08;
      alpha.array[i] += (targetAlpha - alpha.array[i]) * 0.08;
    }
    pos.needsUpdate = true;
    alpha.needsUpdate = true;
  }

  _updateParticleTrail(data, dt, elapsed) {
    if (this.settings.trail <= 0) return;
    const persons = data?.persons || [];
    const isPresent = data?.classification?.presence || false;
    const pos = this._trail.geometry.attributes.position;
    const ages = this._trail.geometry.attributes.age;

    for (let i = 0; i < this._trailCount; i++) {
      ages.array[i] = Math.min(1, ages.array[i] + dt * 0.8);
    }

    // Emit from all active persons
    if (isPresent && persons.length > 0) {
      this._trailTimer += dt;
      const ms = persons[0].motion_score || 0;
      const emitRate = ms > 50 ? 0.02 : 0.08;

      if (this._trailTimer >= emitRate) {
        this._trailTimer = 0;
        for (const p of persons) {
          const pp = p.position || [0, 0, 0];
          const idx = this._trailHead;
          pos.array[idx * 3] = (pp[0] || 0) + (Math.random() - 0.5) * 0.15;
          pos.array[idx * 3 + 1] = Math.random() * 1.5 + 0.1;
          pos.array[idx * 3 + 2] = (pp[2] || 0) + (Math.random() - 0.5) * 0.15;
          ages.array[idx] = 0;
          this._trailHead = (this._trailHead + 1) % this._trailCount;
        }
      }
    }
    pos.needsUpdate = true;
    ages.needsUpdate = true;
  }

  // ---- WiFi Waves ----

  _updateWifiWaves(elapsed) {
    for (const w of this._wifiWaves) {
      const t = (elapsed * 0.8 + w.phase) % 4.5;
      const life = t / 4.5;
      w.mat.opacity = Math.max(0, this.settings.waves * 0.25 * (1 - life));
      const scale = 1 + life * 0.6;
      w.mesh.scale.set(scale, scale, scale);
      w.mesh.rotation.y = elapsed * 0.05;
    }
  }

  // ---- Signal Field ----

  _updateSignalField(data) {
    const field = data?.signal_field?.values;
    if (!field) return;
    const count = Math.min(field.length, 400);
    for (let i = 0; i < count; i++) {
      const v = field[i] || 0;
      let r, g, b;
      if (v < 0.3) { r = 0; g = v * 1.5; b = v * 0.3; }
      else if (v < 0.6) {
        const t = (v - 0.3) / 0.3;
        r = t * 0.3; g = 0.45 + t * 0.4; b = 0.09 - t * 0.05;
      } else {
        const t = (v - 0.6) / 0.4;
        r = 0.3 + t * 0.7; g = 0.85 - t * 0.2; b = 0.04;
      }
      this._fieldColors[i * 3] = r;
      this._fieldColors[i * 3 + 1] = g;
      this._fieldColors[i * 3 + 2] = b;
      this._fieldSizes[i] = 5 + v * 15;
    }
    this._fieldPoints.geometry.attributes.color.needsUpdate = true;
    this._fieldPoints.geometry.attributes.size.needsUpdate = true;
  }

  // ---- FPS ----

  _updateFPS(dt) {
    this._fpsFrames++;
    this._fpsTime += dt;
    if (this._fpsTime >= 1) {
      this._fpsValue = Math.round(this._fpsFrames / this._fpsTime);
      this._fpsFrames = 0;
      this._fpsTime = 0;
      if (this._showFps) {
        document.getElementById('fps-counter').textContent = `${this._fpsValue} FPS`;
      }
      this._adaptQuality();
    }
  }

  _adaptQuality() {
    let nl = this._qualityLevel;
    if (this._fpsValue < 25 && nl > 0) nl--;
    else if (this._fpsValue > 55 && nl < 2) nl++;
    if (nl !== this._qualityLevel) {
      this._qualityLevel = nl;
      this._nebula.setQuality(nl);
      this._postProcessing.setQuality(nl);
    }
  }

  _onResize() {
    const w = window.innerWidth, h = window.innerHeight;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
    this._postProcessing.resize(w, h);
  }
}

new Observatory();
