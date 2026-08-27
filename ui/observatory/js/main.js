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

import { DemoDataGenerator } from './demo-data.js';
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

// Scene floor footprint (units). Configured room dimensions (metres) are
// mapped to fit this footprint, centered on the scene origin.
const FLOOR_W = 12; // scene units along X
const FLOOR_D = 10; // scene units along Z
const ROOM_H = 4;   // scene units (wall height for room wireframe)

// ---- Main Class ----

class Observatory {
  constructor() {
    this._canvas = document.getElementById('observatory-canvas');
    this.settings = {
      ...DEFAULTS,
      // Clone the per-node override maps so we never alias DEFAULTS objects.
      nodeLabels: { ...(DEFAULTS.nodeLabels || {}) },
      nodePositions: { ...(DEFAULTS.nodePositions || {}) },
    };

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

    // Data
    this._demoData = new DemoDataGenerator();
    this._demoData.setCycleDuration(this.settings.cycle || 30);
    if (this.settings.scenario && this.settings.scenario !== 'auto') {
      this._demoData.setScenario(this.settings.scenario);
    }
    this._currentData = null;
    this._currentScenario = null;

    // Build scene
    this._setupLighting();
    this._nebula = new NebulaBackground(this._scene);
    this._buildRoom();
    try { this._buildNodes(); } catch (e) { console.error('[Observatory] _buildNodes failed (continuing):', e); }
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

    // Wire input + settings UI and start the loop FIRST, so the page is always
    // interactive even if data/auto-detect below throws (an init exception here
    // previously left the Settings button unwired and the page frozen).
    this._initKeyboard();
    this._hud.initSettings();
    this._hud.initQuickSelect();
    window.addEventListener('resize', () => this._onResize());

    this._ws = null;
    this._liveData = null;
    this._animate();

    // Auto-detect live data last, guarded so a failure can't break init.
    try { this._autoDetectLive(); } catch (e) { console.error('[Observatory] _autoDetectLive failed (continuing):', e); }
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
    // Grid sized to floor footprint, divisions ≈ 1 per scene unit
    this._grid = new THREE.GridHelper(FLOOR_W, 24, 0x1a4830, 0x0c2818);
    this._grid.material.opacity = 0.5;
    this._grid.material.transparent = true;
    this._scene.add(this._grid);

    const boxGeo = new THREE.BoxGeometry(FLOOR_W, ROOM_H, FLOOR_D);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this._roomWire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
      color: C.greenDim, opacity: 0.3, transparent: true,
    }));
    this._roomWire.position.y = ROOM_H / 2;
    this._scene.add(this._roomWire);

    // Reflective floor
    const floorGeo = new THREE.PlaneGeometry(FLOOR_W, FLOOR_D);
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

    // Dimension label sprites along the floor edges (rebuilt on dim change)
    this._roomDimGroup = new THREE.Group();
    this._scene.add(this._roomDimGroup);
    this._refreshRoomDimLabels();
  }

  // ---- Coordinate mapping (real metres → scene units) ----
  //
  // Room is centered on the scene origin. Node X (0..roomX) spans the floor
  // width, node Y (0..roomY) spans the floor depth, node Z is height in metres
  // (used directly as scene Y, since 1 unit ≈ 1 m vertically).
  _roomDims() {
    let rx = parseFloat(this.settings.roomX);
    let ry = parseFloat(this.settings.roomY);
    if (!Number.isFinite(rx) || rx <= 0) rx = DEFAULTS.roomX;
    if (!Number.isFinite(ry) || ry <= 0) ry = DEFAULTS.roomY;
    return { rx, ry };
  }

  _roomToScene(mx, my, mz) {
    const { rx, ry } = this._roomDims();
    const x = Number.isFinite(mx) ? mx : 0;
    const y = Number.isFinite(my) ? my : 0;
    const z = Number.isFinite(mz) ? mz : 1.0;
    return new THREE.Vector3(
      (x / rx - 0.5) * FLOOR_W,
      z,
      (y / ry - 0.5) * FLOOR_D
    );
  }

  _refreshRoomDimLabels() {
    if (!this._roomDimGroup) return;
    while (this._roomDimGroup.children.length) {
      const c = this._roomDimGroup.children.pop();
      if (c.material) { c.material.map?.dispose(); c.material.dispose(); }
    }
    const { rx, ry } = this._roomDims();
    const wLabel = this._makeLabelSprite(`${rx.toFixed(1)} m`, '#3eff8a');
    wLabel.position.set(0, 0.05, FLOOR_D / 2 + 0.5);
    wLabel.scale.set(1.4, 0.35, 1);
    this._roomDimGroup.add(wLabel);
    const dLabel = this._makeLabelSprite(`${ry.toFixed(1)} m`, '#3eff8a');
    dLabel.position.set(FLOOR_W / 2 + 0.5, 0.05, 0);
    dLabel.scale.set(1.4, 0.35, 1);
    this._roomDimGroup.add(dLabel);
  }

  // ---- Sensor Nodes ----

  _buildNodes() {
    // Nodes are fully dynamic — markers are created/removed to match the set of
    // node_ids currently reporting in the live feed (see _syncNodes, driven from
    // the animate loop). Start empty; nothing is hardcoded.
    this._nodeGroup = new THREE.Group();
    this._scene.add(this._nodeGroup);
    this._nodeMarkers = [];   // [{ group, led, light, scenePos, nodeId }]
    this._nodeIdKey = '';     // signature of the rendered id set (change detect)
  }

  // Room-space position (metres) for a node: user-tagged override if present,
  // else auto-layout — spread the N reporting nodes evenly around the room so
  // they don't overlap. Hardware-agnostic; only node id matters.
  _nodeRoomPos(id, idx, total) {
    const tag = this.settings.nodePositions && this.settings.nodePositions[id];
    if (tag && Number.isFinite(+tag.x) && Number.isFinite(+tag.y)) {
      return { x: +tag.x, y: +tag.y, z: Number.isFinite(+tag.z) ? +tag.z : 1 };
    }
    return this._autoLayoutPos(idx, total);
  }

  // Pure auto-layout (no per-node override): spread `total` nodes evenly around
  // an inset ellipse on the room floor. Also used by the settings node-list so
  // the displayed default positions match the scene.
  _autoLayoutPos(idx, total) {
    const rx = +this.settings.roomX || 4;
    const ry = +this.settings.roomY || 5;
    const ang = (total > 0 ? idx / total : 0) * Math.PI * 2 - Math.PI / 2;
    return { x: rx / 2 + Math.cos(ang) * rx * 0.38, y: ry / 2 + Math.sin(ang) * ry * 0.38, z: 1 };
  }

  _nodeLabel(id) {
    const t = this.settings.nodeLabels && this.settings.nodeLabels[id];
    return (t && String(t).trim()) ? String(t) : `Node ${id}`;
  }

  _makeNodeMarker(id, idx, total) {
    const g = new THREE.Group();
    const rp = this._nodeRoomPos(id, idx, total);
    const p = this._roomToScene(rp.x, rp.y, rp.z);
    g.position.copy(p);

    const boxMat = new THREE.MeshStandardMaterial({
      color: 0x303848, roughness: 0.3, metalness: 0.6,
      emissive: C.blueSignal, emissiveIntensity: 0.6,
    });
    g.add(new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.12, 0.2), boxMat));

    const antMat = new THREE.MeshStandardMaterial({
      color: 0x90b0ff, roughness: 0.3, metalness: 0.6,
      emissive: C.blueSignal, emissiveIntensity: 0.4,
    });
    const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.3), antMat);
    ant.position.set(0, 0.21, 0);
    g.add(ant);

    const led = new THREE.Mesh(new THREE.SphereGeometry(0.03), new THREE.MeshBasicMaterial({ color: C.greenGlow }));
    led.position.set(0, 0.37, 0);
    g.add(led);

    const light = new THREE.PointLight(C.blueSignal, 0.8, 4);
    light.position.set(0, 0.3, 0);
    g.add(light);

    const label = this._makeLabelSprite(this._nodeLabel(id), '#90c0ff');
    label.position.set(0, 0.7, 0);
    label.scale.set(1.6, 0.4, 1);
    g.add(label);

    this._nodeGroup.add(g);
    return { group: g, led, light, scenePos: p.clone(), nodeId: id };
  }

  // Reconcile rendered markers to `ids` (the node_ids currently reporting).
  // Cheap change-detect via a joined key; full rebuild only when the set changes
  // (or force=true after a settings edit) so per-frame cost is ~nil.
  _syncNodes(ids, force = false) {
    if (!this._nodeGroup) return;
    const list = (Array.isArray(ids) ? ids : []).filter(v => v != null);
    const key = list.join(',');
    if (!force && key === this._nodeIdKey) return;
    this._nodeIdKey = key;

    // Dispose current markers, then rebuild for the new set (keeps auto-layout
    // consistent as the count changes).
    this._nodeGroup.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) { o.material.map?.dispose?.(); o.material.dispose?.(); }
    });
    this._nodeGroup.clear();
    this._nodeMarkers = list.map((id, idx) => this._makeNodeMarker(id, idx, list.length));
    this._rebuildWifiWaves();
    // Refresh the settings "Room & Nodes" list to match (it has its own
    // change-detection; cheap to call). Guarded so it can't break the scene.
    try { this._hud && this._hud.refreshNodeList(); } catch (e) {
      if (!this._nodeListErrLogged) { console.error('[Observatory] refreshNodeList failed:', e); this._nodeListErrLogged = true; }
    }
  }

  // Build a CanvasTexture text sprite for a node/dimension label
  _makeLabelSprite(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 64);
    ctx.font = 'bold 30px "JetBrains Mono", monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.fillText(text, 128, 34);
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, depthTest: false,
    });
    return new THREE.Sprite(mat);
  }

  // Live re-place room + node markers when settings change (no reload)
  _rebuildRoomAndNodes() {
    this._refreshRoomDimLabels();
    // Re-place/relabel the currently-rendered nodes with the new settings.
    const ids = (this._nodeMarkers || []).map(m => m.nodeId);
    this._syncNodes(ids, true);
  }

  // ---- WiFi Waves (one ripple set per node) ----

  _buildWifiWaves() {
    this._wifiWaves = [];
    this._rebuildWifiWaves();
  }

  _rebuildWifiWaves() {
    // Dispose any existing wave shells
    if (this._wifiWaves) {
      for (const w of this._wifiWaves) {
        this._scene.remove(w.mesh);
        w.mesh.geometry.dispose();
        w.mat.dispose();
      }
    }
    this._wifiWaves = [];
    const markers = this._nodeMarkers || [];
    markers.forEach((m, ni) => {
      for (let i = 0; i < 3; i++) {
        const radius = 0.3 + i * 0.35;
        const geo = new THREE.SphereGeometry(radius, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.6);
        const mat = new THREE.MeshBasicMaterial({
          color: C.blueSignal,
          transparent: true, opacity: 0,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
          depthWrite: false, wireframe: true,
        });
        const shell = new THREE.Mesh(geo, mat);
        shell.position.copy(m.scenePos);
        shell.position.y += 0.3;
        this._scene.add(shell);
        this._wifiWaves.push({ mesh: shell, mat, phase: ni * 0.5 + i * 0.7, nodeId: m.nodeId });
      }
    });
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
        case 'd': this._demoData.cycleScenario(); break;
        case 'f':
          this._showFps = !this._showFps;
          document.getElementById('fps-counter').style.display = this._showFps ? 'block' : 'none';
          break;
        case 's': this._hud.toggleSettings(); break;
        case ' ':
          e.preventDefault();
          this._demoData.paused = !this._demoData.paused;
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
    // If the user defaults to (or has selected) live WS, connect to the
    // page's own host /ws/sensing immediately. Falls back to demo on close.
    if (this.settings.dataSource === 'ws') {
      const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = this.settings.wsUrl ||
        `${wsProto}//${window.location.host}/ws/sensing`;
      this.settings.wsUrl = wsUrl;
      this._connectWS(wsUrl);
    }
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
        console.log('[Observatory] No sensing server detected, using demo mode');
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
    this._wsUrl = url;
    this._wsShouldRun = true;   // keep reconnecting until explicitly disconnected
    try {
      const ws = new WebSocket(url);
      this._ws = ws;
      ws.onopen = () => {
        this._wsRetry = 0;
        console.log('[Observatory] WebSocket connected');
        this._hud.updateSourceBadge('ws', ws);
      };
      ws.onmessage = (evt) => { try { this._liveData = JSON.parse(evt.data); } catch {} };
      ws.onclose = () => {
        this._ws = null;
        this._liveData = null;              // drop stale frame; stay in 'ws' mode
        this._hud.updateSourceBadge('ws', null);
        // Auto-reconnect with backoff so transient drops / server restarts
        // recover on their own — new nodes keep appearing without a reload.
        if (this._wsShouldRun) {
          const n = this._wsRetry || 0;
          const delay = Math.min(1000 * Math.pow(2, n), 15000);
          this._wsRetry = n + 1;
          console.log('[Observatory] WebSocket closed — reconnecting in', delay, 'ms');
          clearTimeout(this._wsReconnectTimer);
          this._wsReconnectTimer = setTimeout(() => {
            if (this._wsShouldRun) this._connectWS(this._wsUrl);
          }, delay);
        }
      };
      ws.onerror = () => {};
    } catch {
      if (this._wsShouldRun) {
        clearTimeout(this._wsReconnectTimer);
        this._wsReconnectTimer = setTimeout(() => this._connectWS(url), 3000);
      }
    }
  }

  _disconnectWS() {
    this._wsShouldRun = false;
    clearTimeout(this._wsReconnectTimer);
    if (this._ws) { try { this._ws.close(); } catch {} this._ws = null; }
    this._liveData = null;
  }

  // ========================================
  // ANIMATION LOOP
  // ========================================

  _animate() {
    requestAnimationFrame(() => this._animate());
    const dt = Math.min(this._clock.getDelta(), 0.1);
    const elapsed = this._clock.getElapsedTime();

    // Data source
    if (this.settings.dataSource === 'ws' && this._liveData) {
      this._currentData = this._liveData;
    } else {
      this._currentData = this._demoData.update(dt);
    }
    const data = this._currentData;

    // Connected node ids from the live feed (null on demo / no nodes array).
    // Stored on `this` so both the wave updater and the marker loop gate on it.
    this._connIds = (data && Array.isArray(data.nodes))
      ? new Set(data.nodes.map(n => n.node_id))
      : null;

    // Updates (guarded: a single updater throwing on an unexpected live-data
    // shape must not abort the frame — otherwise controls/render below are
    // skipped and the page freezes).
    try {
      this._nebula.update(dt, elapsed);
      this._figurePool.update(data, elapsed);
      this._scenarioProps.update(data, this._demoData.currentScenario);
      this._updateDotMatrixMist(data, elapsed);
      this._updateParticleTrail(data, dt, elapsed);
      this._updateWifiWaves(elapsed);
      this._updateSignalField(data);
      this._hud.updateHUD(data, this._demoData);
      this._hud.updateSparkline(data);
    } catch (e) {
      if (!this._loopErrLogged) {
        console.error('[Observatory] per-frame update error (loop kept alive):', e);
        this._loopErrLogged = true;
      }
    }

    // Reconcile markers to the node_ids currently reporting (one marker per
    // reporting node, any count), then pulse their LEDs. No live nodes -> none.
    const reportIds = this._connIds ? [...this._connIds] : [];
    try { this._syncNodes(reportIds); } catch (e) {
      if (!this._syncErrLogged) { console.error('[Observatory] _syncNodes failed:', e); this._syncErrLogged = true; }
    }
    if (this._nodeMarkers) {
      for (let i = 0; i < this._nodeMarkers.length; i++) {
        const m = this._nodeMarkers[i];
        const ph = i * 1.3;
        m.led.material.opacity = 0.5 + 0.5 * Math.sin(elapsed * 8 + ph);
        m.light.intensity = 0.4 + 0.3 * Math.sin(elapsed * 3 + ph);
      }
    }

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
    const conn = this._connIds;
    for (const w of this._wifiWaves) {
      // Hide ripples for nodes not in the live feed (matches marker gating).
      if (conn && !conn.has(w.nodeId)) { w.mat.opacity = 0; w.mesh.visible = false; continue; }
      w.mesh.visible = true;
      const t = (elapsed * 0.9 + w.phase) % 3.5;
      const life = t / 3.5;
      // Subtle ripple — smaller opacity since markers are densely placed
      w.mat.opacity = Math.max(0, this.settings.waves * 0.18 * (1 - life));
      const scale = 1 + life * 1.2;
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
