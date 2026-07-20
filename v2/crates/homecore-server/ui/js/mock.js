// HOMECORE-UI contract-conformant mock layer — ADR-131 §7.1.
//
// "Where a service is not yet stable, the panel is still built against
//  its defined contract (with a contract-conformant mock standing in for
//  the live endpoint only until that endpoint lands)."
//
// Shapes mirror the schemas described in ADR-131 §4 + the calibration
// RoomState contract (docs/integration/calibration-appliance-integration.md)
// + the SEED HTTPS API. Live endpoints replace these the moment they
// exist; nothing here is presented to the operator as real (the UI shows
// a DEMO badge whenever the mock layer is serving a panel — see api.js).

const now = () => new Date().toISOString();
const ago = (s) => new Date(Date.now() - s * 1000).toISOString();
function jitter(base, amp) { return +(base + (Math.sin(Date.now() / 3000 + base) * amp)).toFixed(2); }
function spark(base, amp, n = 24) {
  return Array.from({ length: n }, (_, i) => +(base + Math.sin(i / 2) * amp + (i % 3) * amp * 0.2).toFixed(2));
}

// Factory for a bedroom SEED node — keeps the three bedrooms consistent
// while varying the values that matter for the analysis views.
function bedroomSeed(o) {
  return {
    device_id: o.device_id, firmware: '0.7.3', online: true, conn: o.conn || 'wifi', epoch: o.epoch,
    vector_count: o.vector_count, vector_dim: 8, knn_latency_ms: o.knn_latency_ms,
    last_ingest: ago(2), witness_valid: true, witness_len: o.witness_len,
    witness_last_verify: ago(1800), zone: o.zone,
    storage_used: o.vector_count, storage_budget: 100000,
    sensors: {
      bme280: { temp_c: o.temp_c, humidity_pct: o.humidity_pct, pressure_hpa: 1013.0 },
      pir: { motion: o.motion, last_trigger: ago(o.motion ? 5 : 640) },
      reed: { open: false, last_change: ago(30000) },
      ads1115: [{ label: 'ch0', v: 0.11 }, { label: 'ch1', v: 0.0 }, { label: 'ch2', v: 0.0 }, { label: 'ch3', v: 0.0 }],
      vibration: { active: false, last_trigger: null },
    },
    reflex: [
      { name: 'fragility_alarm', threshold: 0.3, target: 'relay actuator', last_fired: o.fired ? ago(420) : null, fired_recently: !!o.fired },
      { name: 'drift_cutoff', threshold: 1.0, target: 'ingest gate', last_fired: null, fired_recently: false },
      { name: 'hd_anomaly_indicator', threshold: 200, target: 'PWM brightness', last_fired: null, fired_recently: false },
    ],
    cognition: { fragility: o.fragility, coherence_phases: o.phases, knn_rebuild_s: 10 },
    ingest: { batch: 64, flush_ms: 1000, bridge: 'direct', esp32: [{ node_id: o.node, packet: '0xC5110003', rate_hz: 1.0 }] },
    esp32_nodes: 1, frame_rate_hz: 100,
  };
}

// ── v0 Appliance health (§4.1) ──────────────────────────────────────
export function applianceHealth() {
  return {
    cpu_pct: jitter(34, 6),
    ram_pct: jitter(58, 4),
    hailo_load_pct: jitter(41, 12),
    hailo_temp_c: jitter(52, 3),
    uptime_s: 824510,
    services: [
      { name: 'ruview-mcp-brain', port: 9876, status: 'running' },
      { name: 'cognitum-rvf-agent', port: 9004, status: 'running' },
      { name: 'ruvector-hailo-worker', port: 50051, status: 'running' },
    ],
    event_rate: spark(120, 40),
    channel_capacity: 4096,
    channel_lag: 0,
  };
}

// ── SEED fleet (§4.1 / §4.2) ────────────────────────────────────────
const SEEDS = [
  {
    device_id: 'seed-livingroom-a1',
    firmware: '0.7.3', online: true, conn: 'wifi', epoch: 184,
    vector_count: 71280, vector_dim: 8, knn_latency_ms: 2.1,
    last_ingest: ago(3), witness_valid: true, witness_len: 184210,
    witness_last_verify: ago(900), zone: 'Living Room',
    storage_used: 71280, storage_budget: 100000,
    sensors: {
      bme280: { temp_c: 21.6, humidity_pct: 44, pressure_hpa: 1013.2 },
      pir: { motion: true, last_trigger: ago(8) },
      reed: { open: false, last_change: ago(7200) },
      ads1115: [{ label: 'soil', v: 0.42 }, { label: 'light', v: 0.71 }, { label: 'aux2', v: 0.03 }, { label: 'aux3', v: 0.0 }],
      vibration: { active: false, last_trigger: ago(40000) },
    },
    reflex: [
      { name: 'fragility_alarm', threshold: 0.3, target: 'relay actuator', last_fired: ago(300), fired_recently: true },
      { name: 'drift_cutoff', threshold: 1.0, target: 'ingest gate', last_fired: null, fired_recently: false },
      { name: 'hd_anomaly_indicator', threshold: 200, target: 'PWM brightness', last_fired: ago(12000), fired_recently: false },
    ],
    cognition: { fragility: 0.42, coherence_phases: [{ t: ago(3600), label: 'empty' }, { t: ago(1800), label: 'occupied' }, { t: ago(300), label: 'regime-change' }], knn_rebuild_s: 10 },
    ingest: { batch: 64, flush_ms: 1000, bridge: 'host-laptop hop', esp32: [{ node_id: 'esp32-lr-01', packet: '0xC5110003', rate_hz: 1.0 }, { node_id: 'esp32-lr-02', packet: '0xC5110002', rate_hz: 0.9 }] },
    esp32_nodes: 2, frame_rate_hz: 98,
  },
  bedroomSeed({
    device_id: 'seed-bedroom-1', zone: 'Bedroom 1 (primary)', epoch: 183,
    vector_count: 38110, knn_latency_ms: 1.7, witness_len: 91022,
    temp_c: 20.1, humidity_pct: 47, motion: false, fragility: 0.12,
    phases: [{ t: ago(7200), label: 'empty' }, { t: ago(3600), label: 'sleep' }],
    node: 'esp32-br1-01', conn: 'usb',
  }),
  bedroomSeed({
    device_id: 'seed-bedroom-2', zone: 'Bedroom 2 (guest)', epoch: 181,
    vector_count: 29440, knn_latency_ms: 1.9, witness_len: 70210,
    temp_c: 19.4, humidity_pct: 50, motion: true, fragility: 0.21,
    phases: [{ t: ago(5400), label: 'empty' }, { t: ago(900), label: 'occupied' }],
    node: 'esp32-br2-01', conn: 'wifi',
  }),
  bedroomSeed({
    device_id: 'seed-bedroom-3', zone: 'Bedroom 3 (kids)', epoch: 179,
    vector_count: 24105, knn_latency_ms: 2.0, witness_len: 60880,
    temp_c: 21.0, humidity_pct: 45, motion: false, fragility: 0.34,
    phases: [{ t: ago(9000), label: 'empty' }, { t: ago(4200), label: 'sleep' }, { t: ago(600), label: 'restless' }],
    node: 'esp32-br3-01', conn: 'wifi', fired: true,
  }),
  {
    device_id: 'seed-hallway-c3',
    firmware: '0.6.9', online: false, conn: 'wifi', epoch: 170,
    vector_count: 12044, vector_dim: 8, knn_latency_ms: null,
    last_ingest: ago(5400), witness_valid: true, witness_len: 40110,
    witness_last_verify: ago(86400), zone: 'Hallway',
    storage_used: 12044, storage_budget: 100000,
    sensors: null,
    reflex: [],
    cognition: { fragility: null, coherence_phases: [], knn_rebuild_s: 10 },
    ingest: { batch: 64, flush_ms: 1000, bridge: 'direct', esp32: [] },
    esp32_nodes: 0, frame_rate_hz: 0,
    warnings: ['stale firmware version (0.6.9 < 0.7.3)', 'offline > 1h'],
  },
];
export function seeds() { return SEEDS.map((s) => ({ ...s })); }
export function seed(id) { return SEEDS.find((s) => s.device_id === id) || null; }

// ── ESP32 node warnings (§4.1) ──────────────────────────────────────
export function esp32Warnings() {
  return [
    { node_id: 'esp32-lr-02', seed: 'seed-livingroom-a1', issue: 'presence_score normalisation anomaly' },
    { node_id: 'esp32-hw-01', seed: 'seed-hallway-c3', issue: 'stale firmware version' },
  ];
}

// ── COG runtime (§4.6) ──────────────────────────────────────────────
const COGS = [
  { id: 'cog-ha-matter', version: '1.4.2', arch: 'arm', status: 'running', pid: 4120, sha256_verified: true, signature_verified: true },
  { id: 'cog-pose-estimation', version: '2.1.0', arch: 'hailo10', status: 'running', pid: 4188, sha256_verified: true, signature_verified: true, hef: ['rf_foundation_encoder.hef', 'pose_head.hef'], throughput_fps: 41 },
  { id: 'cog-person-count', version: '0.9.4', arch: 'arm', status: 'running', pid: 4205, sha256_verified: true, signature_verified: true },
  { id: 'cog-calibration', version: '1.0.1', arch: 'arm', status: 'running', pid: 4250, sha256_verified: true, signature_verified: true },
  { id: 'cog-anomaly-watch', version: '0.3.0', arch: 'arm', status: 'failed', pid: null, sha256_verified: true, signature_verified: true, error: 'panic: bank not found' },
  { id: 'cog-legacy-bridge', version: '0.1.2', arch: 'arm', status: 'stopped', pid: null, sha256_verified: false, signature_verified: false },
];
export function cogs() { return COGS.map((c) => ({ ...c })); }
export function cogUpdates() { return [{ id: 'cog-pose-estimation', from: '2.1.0', to: '2.2.0', new_entities: ['sensor.lr_pose_confidence'], config_changes: ['add: max_persons'] }]; }
export function appRegistry() {
  return [
    { id: 'cog-fall-detect', title: 'Fall Detection', desc: 'Multistatic fall detection specialist', category: 'safety', arch: 'arm', featured: true, new_entities: ['binary_sensor.{room}_fall'] },
    { id: 'cog-sleep-stage', title: 'Sleep Staging', desc: 'REM/deep/light from breathing + restlessness', category: 'health', arch: 'hailo10', new_entities: ['sensor.{room}_sleep_stage'] },
    { id: 'cog-gesture', title: 'Gesture Control', desc: 'DTW gesture classifier → service calls', category: 'control', arch: 'arm', new_entities: ['event.{room}_gesture'] },
  ];
}

// ── RoomState / sensing (§4.5) — calibration contract ───────────────
export function roomStates() {
  return [
    {
      room_id: 'living_room', stale: false, vetoed: false, seeds: ['seed-livingroom-a1'],
      presence: { value: 'occupied', confidence: 0.93 },
      posture: { value: 'sitting', confidence: 0.81 },
      breathing_bpm: { value: jitter(15, 1.5), confidence: 0.77 },
      heart_bpm: { value: jitter(72, 3), confidence: 0.64 },
      restlessness: { value: 0.22, confidence: 0.7 },
      anomaly: { value: 0.18, confidence: 0.8, threshold: 0.8 },
    },
    {
      // Bedroom 1 — primary; healthy sleeping vitals.
      room_id: 'bedroom_1', stale: false, vetoed: false, seeds: ['seed-bedroom-1'],
      presence: { value: 'occupied', confidence: 0.91 },
      posture: { value: 'lying', confidence: 0.9 },
      breathing_bpm: { value: jitter(12, 1), confidence: 0.85 },
      heart_bpm: { value: jitter(58, 2), confidence: 0.72 },
      restlessness: { value: 0.08, confidence: 0.8 },
      anomaly: { value: 0.12, confidence: 0.84, threshold: 0.8 },
    },
    {
      // Bedroom 2 — guest; STALE bank (recalibrate demo).
      room_id: 'bedroom_2', stale: true, vetoed: false, seeds: ['seed-bedroom-2'],
      presence: { value: 'occupied', confidence: 0.86 },
      posture: { value: 'sitting', confidence: 0.7 },
      breathing_bpm: { value: jitter(16, 1.5), confidence: 0.66 },
      heart_bpm: { value: jitter(74, 3), confidence: 0.58 },
      restlessness: { value: 0.31, confidence: 0.62 },
      anomaly: { value: 0.4, confidence: 0.6, threshold: 0.8 },
    },
    {
      // Bedroom 3 — kids; heartbeat specialist not yet trained.
      room_id: 'bedroom_3', stale: false, vetoed: false, seeds: ['seed-bedroom-3'],
      presence: { value: 'occupied', confidence: 0.79 },
      posture: { value: 'lying', confidence: 0.74 },
      breathing_bpm: { value: jitter(18, 2), confidence: 0.69 },
      heart_bpm: null,                                     // null = not trained (§6 invariant 3)
      restlessness: { value: 0.46, confidence: 0.6 },
      anomaly: { value: 0.22, confidence: 0.7, threshold: 0.8 },
    },
    {
      room_id: 'kitchen', stale: false, vetoed: true, seeds: ['seed-livingroom-a1', 'seed-hallway-c3'],
      presence: { value: 'occupied', confidence: 0.6 },
      posture: { value: null, confidence: null },        // suppressed by veto — withheld, NOT zero (§4.5)
      breathing_bpm: { value: null, confidence: null },
      heart_bpm: { value: null, confidence: null },
      restlessness: { value: 0.4, confidence: 0.5 },
      anomaly: { value: 0.91, confidence: 0.88, threshold: 0.8 },
    },
    {
      room_id: 'office', stale: false, vetoed: false, seeds: ['seed-bedroom-1'],
      presence: { value: 'absent', confidence: 0.95 },
      posture: null,                                       // null = not trained (§6 invariant 3)
      breathing_bpm: null,
      heart_bpm: null,
      restlessness: { value: 0.0, confidence: 0.9 },
      anomaly: { value: 0.05, confidence: 0.9, threshold: 0.8 },
    },
  ];
}

// ── Fleet map / federation (§4.3) ───────────────────────────────────
export function federation() {
  return {
    coordinator: 'seed-livingroom-a1', round: 47, k_healthy: 4, delta_status: 'exchanging',
    invariant: 'model deltas only — never raw CSI',
    krum: { f: 1, multi: true }, cadence_min: 30,
    mesh_links: [
      { a: 'seed-livingroom-a1', b: 'seed-bedroom-1', health: 'green' },
      { a: 'seed-bedroom-1', b: 'seed-bedroom-2', health: 'green' },
      { a: 'seed-bedroom-2', b: 'seed-bedroom-3', health: 'amber' },
      { a: 'seed-bedroom-1', b: 'seed-hallway-c3', health: 'red' },
    ],
    fused_events: [{ kind: 'fall', seeds: ['seed-livingroom-a1', 'seed-hallway-c3'], n: 2 }, { kind: 'occupant-track', seeds: ['seed-bedroom-1', 'seed-bedroom-2', 'seed-livingroom-a1'], n: 3 }],
  };
}

// ── Witness / audit (§4.9) ──────────────────────────────────────────
export function witnessLog(page = 0, size = 12) {
  const total = 240;
  const items = Array.from({ length: size }, (_, i) => {
    const n = page * size + i;
    const seedTier = n % 2 === 0;
    return {
      entity_id: seedTier ? `rvf.store.write.${184210 - n}` : ['sensor.living_room_presence', 'binary_sensor.front_door', 'sensor.bedroom_breathing_rate'][n % 3],
      old_state: seedTier ? null : ['false', 'off', '14.5'][n % 3],
      new_state: seedTier ? `sha256:${(0x9a3f + n).toString(16)}…` : ['true', 'on', '15.1'][n % 3],
      ts: ago(n * 37),
      tier: seedTier ? 'seed-sha256' : 'homecore-ed25519',
      seed: ['seed-livingroom-a1', 'seed-bedroom-1', 'seed-bedroom-2', 'seed-bedroom-3'][n % 4],
      key_fp: ['a1b2c3d4', 'e5f6a7b8', 'c9d0e1f2', 'b3a4c5d6'][n % 4],
    };
  });
  return { items, page, size, total };
}
export function privacyModes() {
  return [
    { seed: 'seed-livingroom-a1', mode: 'full-publish' },
    { seed: 'seed-bedroom-1', mode: 'audit-only' },
    { seed: 'seed-bedroom-2', mode: 'audit-only' },
    { seed: 'seed-bedroom-3', mode: 'audit-only' },
    { seed: 'seed-hallway-c3', mode: 'audit-only' },
  ];
}

// ── Events / automations (§4.8) ─────────────────────────────────────
export function recentEvents(n = 40) {
  const variants = ['StateChanged', 'EntityRegistered', 'ConfigReloaded'];
  const ents = ['sensor.living_room_presence', 'binary_sensor.front_door', 'light.kitchen_ceiling', 'sensor.bedroom_breathing_rate'];
  return Array.from({ length: n }, (_, i) => ({
    type: variants[i % 3],
    entity_id: ents[i % ents.length],
    old_state: ['off', 'false', '14.5'][i % 3],
    new_state: ['on', 'true', '15.1'][i % 3],
    ts: ago(i * 11),
    user_id: i % 4 === 0 ? 'operator' : null,
    context: { id: 'ctx-' + (1000 + i), parent_id: i % 3 === 0 ? 'ctx-' + (999 + i) : null, grandparent_id: i % 6 === 0 ? 'ctx-' + (998 + i) : null },
    source: ['seed-livingroom-a1', 'cog-ha-matter'][i % 2],
  }));
}

// ── Settings (§4.10) ────────────────────────────────────────────────
export function settings() {
  return {
    mqtt: { broker: 'mqtt://cognitum-v0:1883', user: 'homecore', mdns: '_ruview-ha._tcp', connected: true },
    tokens: [
      { name: 'ios-companion', last_used: ago(120), created: ago(8000000) },
      { name: 'node-red', last_used: ago(60000), created: ago(20000000) },
    ],
    ha_disco_entities: 21,
    esp32: [
      { node_id: 'esp32-lr-01', ip: '192.168.1.31', port: 5566, firmware: '1.2.0', room: 'living_room', seed: 'seed-livingroom-a1' },
      { node_id: 'esp32-br1-01', ip: '192.168.1.32', port: 5566, firmware: '1.2.0', room: 'bedroom_1', seed: 'seed-bedroom-1' },
      { node_id: 'esp32-br2-01', ip: '192.168.1.33', port: 5566, firmware: '1.2.0', room: 'bedroom_2', seed: 'seed-bedroom-2' },
      { node_id: 'esp32-br3-01', ip: '192.168.1.34', port: 5566, firmware: '1.2.0', room: 'bedroom_3', seed: 'seed-bedroom-3' },
    ],
  };
}
