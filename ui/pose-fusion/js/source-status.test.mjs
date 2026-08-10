// Regression coverage for issue #1557.
//
// These tests execute URL derivation and the DOM status boundary. They do not
// claim a browser session, network connection, simulator fidelity, or hardware.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildPoseFusionWsUrl, renderPoseFusionSourceState } from './source-status.js';
import { CsiSimulator } from './csi-simulator.js';

function fakeElement() {
  const classes = new Set();
  return {
    textContent: '',
    classList: {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
    },
  };
}

class FakeWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.closed = false;
    FakeWebSocket.instances.push(this);
  }

  open() {
    if (this.onopen) this.onopen();
  }

  async message(data) {
    if (this.onmessage) await this.onmessage({ data });
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    if (this.onclose) this.onclose({ code: 1000 });
  }
}

globalThis.WebSocket = FakeWebSocket;

test('a non-standard remote HTTP port maps to the adjacent sensing port', () => {
  assert.equal(
    buildPoseFusionWsUrl({
      protocol: 'http:',
      hostname: 'sensor.example',
      host: 'sensor.example:3010',
      port: '3010',
    }),
    'ws://sensor.example:3011/ws/sensing',
  );
});

test('standalone development keeps its explicit 8080 to 8765 mapping', () => {
  assert.equal(
    buildPoseFusionWsUrl({
      protocol: 'http:',
      hostname: '127.0.0.1',
      host: '127.0.0.1:8080',
      port: '8080',
    }),
    'ws://127.0.0.1:8765/ws/sensing',
  );
});

test('an origin without an explicit port keeps the same secure host', () => {
  assert.equal(
    buildPoseFusionWsUrl({
      protocol: 'https:',
      hostname: 'sensor.example',
      host: 'sensor.example',
      port: '',
    }),
    'wss://sensor.example/ws/sensing',
  );
});

test('the DOM always distinguishes simulated CSI from live CSI', () => {
  const elements = {
    statusDot: fakeElement(),
    statusLabel: fakeElement(),
    connectButton: fakeElement(),
  };

  renderPoseFusionSourceState('simulated', elements);
  assert.equal(elements.statusLabel.textContent, 'SIMULATED CSI');
  assert.equal(elements.statusDot.classList.contains('warning'), true);
  assert.equal(elements.connectButton.classList.contains('active'), false);

  renderPoseFusionSourceState('live', elements);
  assert.equal(elements.statusLabel.textContent, 'LIVE CSI');
  assert.equal(elements.statusDot.classList.contains('warning'), false);
  assert.equal(elements.connectButton.textContent, '✓ Connected');
  assert.equal(elements.connectButton.classList.contains('active'), true);
});

test('simulator mode changes keep the DOM provenance synchronized', () => {
  const elements = {
    statusDot: fakeElement(),
    statusLabel: fakeElement(),
    connectButton: fakeElement(),
  };
  const simulator = new CsiSimulator();
  simulator.onModeChange((source) => renderPoseFusionSourceState(source, elements));

  assert.equal(elements.statusLabel.textContent, 'SIMULATED CSI');
  simulator._setMode('live');
  assert.equal(elements.statusLabel.textContent, 'LIVE CSI');
  simulator._setMode('demo');
  assert.equal(elements.statusLabel.textContent, 'SIMULATED CSI');
});

test('an open but silent socket never claims live CSI', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 5 });
  const modes = [];
  simulator.onModeChange((mode) => modes.push(mode));

  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();

  assert.equal(simulator.mode, 'connecting');
  assert.equal(modes.includes('live'), false);

  const result = await connection;
  assert.equal(result, null);
  assert.equal(simulator.mode, 'demo');
  assert.equal(modes.includes('live'), false);
});

test('validated frames expose their actual provenance', async () => {
  const cases = [
    { message: { source: 'simulated' }, expected: 'server-simulated' },
    { message: { source: 'esp32', _simulated: true }, expected: 'server-simulated' },
    { message: { source: 'esp32:offline' }, expected: 'offline' },
    { message: {}, expected: 'unverified' },
    { message: { source: 'esp32' }, expected: 'live' },
  ];

  for (const { message, expected } of cases) {
    const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
    const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
    const socket = FakeWebSocket.instances.at(-1);
    socket.open();
    await socket.message(JSON.stringify({
      ...message,
      type: 'sensing_update',
      nodes: [{ amplitude: [1, 2, 1] }],
    }));

    assert.equal(await connection, expected);
    assert.equal(simulator.mode, expected);
    assert.equal(simulator.isLive, expected === 'live');
    simulator.disconnect();
  }
});

test('a superseded socket cannot downgrade the newer connection', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  const firstConnection = simulator.connectLive('ws://old.example/ws/sensing');
  const firstSocket = FakeWebSocket.instances.at(-1);
  const staleClose = firstSocket.onclose;
  const staleMessage = firstSocket.onmessage;
  firstSocket.open();

  const secondConnection = simulator.connectLive('ws://new.example/ws/sensing');
  const secondSocket = FakeWebSocket.instances.at(-1);
  secondSocket.open();
  await secondSocket.message(JSON.stringify({
    source: 'wifi:office',
    nodes: [{ amplitude: [2, 4, 2] }],
  }));

  assert.equal(await firstConnection, null);
  assert.equal(await secondConnection, 'live');
  const expectedAmplitude = Array.from(simulator._liveAmplitude);
  const expectedRssi = simulator._rssiTarget;

  await staleMessage({
    data: JSON.stringify({
      source: 'simulated',
      features: { mean_rssi: -77 },
      nodes: [{ amplitude: [9, 9, 9] }],
    }),
  });
  staleClose({ code: 1006 });

  assert.equal(simulator.mode, 'live');
  assert.deepEqual(Array.from(simulator._liveAmplitude), expectedAmplitude);
  assert.equal(simulator._rssiTarget, expectedRssi);
});

test('an empty offline frame clears live CSI and updates provenance', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1] }],
  }));

  assert.equal(await connection, 'live');
  assert.ok(simulator._liveAmplitude);

  await socket.message(JSON.stringify({
    source: 'esp32:offline',
    nodes: [{ amplitude: [] }],
  }));

  assert.equal(simulator.mode, 'offline');
  assert.equal(simulator._liveAmplitude, null);
  assert.deepEqual(Array.from(simulator.nextFrame(1).amplitude), Array(52).fill(0));
});

test('entering live mode discards every buffered demo row', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  for (let index = 0; index < simulator.timeWindow; index++) {
    simulator.nextFrame(index / 10);
  }
  assert.equal(simulator.amplitudeBuffer.length, simulator.timeWindow);

  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1] }],
  }));

  assert.equal(await connection, 'live');
  assert.equal(simulator.amplitudeBuffer.length, 0);

  const firstLiveFrame = simulator.nextFrame(1);
  assert.equal(simulator.amplitudeBuffer.length, 1);
  assert.deepEqual(
    Array.from(simulator.amplitudeBuffer[0]),
    Array.from(firstLiveFrame.amplitude),
  );
});

test('payloadless live frames use server metadata without demo CSI', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  simulator.nextFrame(1);

  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [], rssi_dbm: -35 }],
    classification: { presence: true, confidence: 0.9 },
  }));

  assert.equal(await connection, 'live');
  assert.equal(simulator._rssiTarget, -35);
  assert.equal(simulator.rssiDbm, -35);
  assert.equal(simulator.personPresence, 0.9);
  assert.deepEqual(Array.from(simulator.nextFrame(2).amplitude), Array(52).fill(0));
});

test('runtime promotion replaces server-simulated RSSI immediately', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'simulated',
    nodes: [{ amplitude: [1, 2, 1], rssi_dbm: -77 }],
  }));

  assert.equal(await connection, 'server-simulated');
  assert.equal(simulator.rssiDbm, -77);

  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1], rssi_dbm: -35 }],
  }));

  assert.equal(simulator.mode, 'live');
  assert.equal(simulator.rssiDbm, -35);
});

test('runtime promotion never carries simulated RSSI into an RSSI-less live frame', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'simulated',
    nodes: [{ amplitude: [1, 2, 1], rssi_dbm: -77 }],
  }));

  assert.equal(await connection, 'server-simulated');
  assert.equal(simulator.rssiDbm, -77);

  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1] }],
  }));

  assert.equal(simulator.mode, 'live');
  assert.equal(simulator._hasServerRssi, false);
  assert.equal(simulator.rssiDbm, null);
});

test('live RSSI recovers when metadata resumes after an RSSI-less frame', async () => {
  const simulator = new CsiSimulator({ connectTimeoutMs: 100 });
  const connection = simulator.connectLive('ws://sensor.example/ws/sensing');
  const socket = FakeWebSocket.instances.at(-1);
  socket.open();
  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1] }],
  }));

  assert.equal(await connection, 'live');
  assert.equal(simulator.rssiDbm, null);

  await socket.message(JSON.stringify({
    source: 'esp32',
    nodes: [{ amplitude: [1, 2, 1], rssi_dbm: -35 }],
  }));

  assert.equal(simulator.mode, 'live');
  assert.equal(simulator.rssiDbm, -35);
});
