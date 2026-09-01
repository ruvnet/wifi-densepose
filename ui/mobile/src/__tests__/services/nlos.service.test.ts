import {
  NLOS_AUTHENTICATED_SCHEMA,
  NLOS_TICKET_SCHEMA,
  NlosService,
  configureNlosBearerToken,
  createSyntheticNlosFrame,
  hasConfiguredNlosBearerToken,
  type NlosServiceDependencies,
} from '@/services/nlos.service';
import { createLiveNlosFrameFixture } from '@/testUtils/nlosFixtures';
import { NLOS_MAX_MESSAGE_BYTES } from '@/types/nlos';

class MockSocket {
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  close = jest.fn();
}

const NOW = 1_700_000_000_100;
const BEARER_TOKEN = 'e'.repeat(32);
const activeServices = new Set<NlosService>();

const createHarness = () => {
  const socket = new MockSocket();
  const fetchMock = jest.fn(async () => ({
    ok: true,
    status: 200,
    text: async () => JSON.stringify({
      schema: NLOS_TICKET_SCHEMA,
      webSocketUrl: `wss://ruview.example/api/v1/nlos/ws?ticket=${'a'.repeat(64)}`,
      expiresAtUnixMs: NOW + 30_000,
    }),
  }));
  const dependencies: NlosServiceDependencies = {
    fetch: fetchMock,
    createWebSocket: jest.fn(() => socket),
    now: jest.fn(() => NOW),
    setInterval: globalThis.setInterval.bind(globalThis),
    clearInterval: globalThis.clearInterval.bind(globalThis),
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  };
  const service = new NlosService(dependencies);
  activeServices.add(service);
  return { service, socket, fetchMock, dependencies };
};

const authenticate = (socket: MockSocket) => {
  socket.onmessage?.({
    data: JSON.stringify({
      schema: NLOS_AUTHENTICATED_SCHEMA,
      sessionId: 'live-session-1',
      expiresAtUnixMs: NOW + 25_000,
    }),
  });
};

describe('NlosService', () => {
  afterEach(() => {
    for (const service of activeServices) service.disconnect();
    activeServices.clear();
    configureNlosBearerToken(null);
  });

  it('exchanges a transport Bearer token for a one time socket before accepting live frames', async () => {
    const { service, socket, fetchMock } = createHarness();
    const listener = jest.fn();
    service.subscribe(listener);

    await expect(service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN })).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://ruview.example/api/v1/nlos/ws-ticket',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ Authorization: `Bearer ${BEARER_TOKEN}` }),
      }),
    );

    authenticate(socket);
    expect(service.getStatus()).toBe('live');
    const frame = createLiveNlosFrameFixture();
    socket.onmessage?.({ data: JSON.stringify(frame) });
    expect(listener).toHaveBeenCalledWith({
      frame,
      channel: 'authenticated_stream',
      receivedAtUnixMs: NOW,
    });
  });

  it('rejects data before the authenticated session acknowledgement', async () => {
    const { service, socket } = createHarness();
    const rejected = jest.fn();
    service.subscribeRejected(rejected);
    await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });

    socket.onmessage?.({ data: JSON.stringify(createLiveNlosFrameFixture()) });
    expect(rejected).toHaveBeenCalledWith('unauthenticated');
    expect(socket.close).toHaveBeenCalledWith(1008, 'authentication required');
  });

  it('accepts authenticated synthetic server frames without promoting their evidence', async () => {
    const { service, socket } = createHarness();
    const listener = jest.fn();
    service.subscribe(listener);
    await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });
    authenticate(socket);
    const frame = {
      ...createSyntheticNlosFrame(3, NOW),
      sessionId: 'live-session-1',
      provenance: {
        ...createSyntheticNlosFrame(3, NOW).provenance,
        histogramPreserved: true,
      },
    };
    socket.onmessage?.({ data: JSON.stringify(frame) });
    expect(listener).toHaveBeenCalledWith({
      frame,
      channel: 'authenticated_stream',
      receivedAtUnixMs: NOW,
    });
    expect(frame.evidenceLevel).toBe('l0_synthetic');
  });

  it('bounds the authenticated socket handshake to five seconds', async () => {
    jest.useFakeTimers();
    try {
      const { service, socket } = createHarness();
      const rejected = jest.fn();
      service.subscribeRejected(rejected);
      await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });
      jest.advanceTimersByTime(5_000);
      expect(rejected).toHaveBeenCalledWith('unauthenticated');
      expect(socket.close).toHaveBeenCalledWith(1008, 'authentication timeout');
      expect(service.getStatus()).toBe('error');
      service.disconnect();
    } finally {
      jest.useRealTimers();
    }
  });

  it('expires an authenticated session even when the socket is idle', async () => {
    jest.useFakeTimers();
    try {
      const { service, socket } = createHarness();
      const rejected = jest.fn();
      service.subscribeRejected(rejected);
      await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });
      authenticate(socket);
      expect(service.getStatus()).toBe('live');
      jest.advanceTimersByTime(25_000);
      expect(rejected).toHaveBeenCalledWith('unauthenticated');
      expect(socket.close).toHaveBeenCalledWith(1008, 'session expired');
      expect(service.getStatus()).toBe('error');
      service.disconnect();
    } finally {
      jest.useRealTimers();
    }
  });

  it('rejects duplicate and out of order sequences', async () => {
    const { service, socket } = createHarness();
    const listener = jest.fn();
    const rejected = jest.fn();
    service.subscribe(listener);
    service.subscribeRejected(rejected);
    await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });
    authenticate(socket);

    socket.onmessage?.({ data: JSON.stringify(createLiveNlosFrameFixture({ sequence: 5 })) });
    socket.onmessage?.({ data: JSON.stringify(createLiveNlosFrameFixture({ sequence: 5 })) });
    socket.onmessage?.({ data: JSON.stringify(createLiveNlosFrameFixture({ sequence: 4 })) });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(rejected).toHaveBeenCalledTimes(2);
    expect(rejected).toHaveBeenLastCalledWith('out_of_order');
  });

  it('bounds messages and rejects binary payloads', async () => {
    const { service, socket } = createHarness();
    const rejected = jest.fn();
    service.subscribeRejected(rejected);
    await service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: BEARER_TOKEN });
    authenticate(socket);

    socket.onmessage?.({ data: `{"padding":"${'x'.repeat(NLOS_MAX_MESSAGE_BYTES)}"}` });
    socket.onmessage?.({ data: new Uint8Array([1, 2, 3]) });
    expect(rejected).toHaveBeenCalledWith('message_too_large');
    expect(rejected).toHaveBeenCalledWith('unsupported_binary');
  });

  it('rejects remote cleartext server URLs', async () => {
    const { service, fetchMock } = createHarness();
    await expect(service.connectLive({ serverUrl: 'http://ruview.example', bearerToken: BEARER_TOKEN })).resolves.toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(service.getStatus()).toBe('error');
  });

  it('rejects a ticket that redirects the socket to another authority', async () => {
    const { service, fetchMock, dependencies } = createHarness();
    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        schema: NLOS_TICKET_SCHEMA,
        webSocketUrl: `wss://attacker.example/api/v1/nlos/ws?ticket=${'a'.repeat(64)}`,
        expiresAtUnixMs: NOW + 10_000,
      }),
    });
    await expect(service.connectLive({
      serverUrl: 'https://ruview.example',
      bearerToken: BEARER_TOKEN,
    })).resolves.toBe(false);
    expect(dependencies.createWebSocket).not.toHaveBeenCalled();
  });

  it('enforces the 32 to 512 character ephemeral credential bound', async () => {
    const short = createHarness();
    await expect(short.service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: 'x'.repeat(31) })).resolves.toBe(false);
    expect(short.fetchMock).not.toHaveBeenCalled();

    const long = createHarness();
    await expect(long.service.connectLive({ serverUrl: 'https://ruview.example', bearerToken: 'x'.repeat(513) })).resolves.toBe(false);
    expect(long.fetchMock).not.toHaveBeenCalled();
  });

  it('keeps configured credentials memory-only and applies the same length bound', () => {
    expect(configureNlosBearerToken('x'.repeat(31))).toBe(false);
    expect(hasConfiguredNlosBearerToken()).toBe(false);
    expect(configureNlosBearerToken(`${'x'.repeat(31)} `)).toBe(false);
    expect(configureNlosBearerToken('x'.repeat(513))).toBe(false);
    expect(hasConfiguredNlosBearerToken()).toBe(false);

    expect(configureNlosBearerToken('x'.repeat(32))).toBe(true);
    expect(hasConfiguredNlosBearerToken()).toBe(true);
    expect(configureNlosBearerToken(null)).toBe(true);
    expect(hasConfiguredNlosBearerToken()).toBe(false);
  });

  it('emits bounded, visibly synthetic deterministic replay frames', () => {
    jest.useFakeTimers();
    try {
      const { service } = createHarness();
      const listener = jest.fn();
      service.subscribe(listener);
      service.startDeterministicReplay(1_000);
      expect(service.getStatus()).toBe('synthetic_replay');
      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener.mock.calls[0][0]).toEqual({
        frame: createSyntheticNlosFrame(0, NOW, 30),
        channel: 'deterministic_replay',
        receivedAtUnixMs: NOW,
      });
      jest.advanceTimersByTime(34);
      expect(listener).toHaveBeenCalledTimes(2);
      service.disconnect();
    } finally {
      jest.useRealTimers();
    }
  });

  it('rejects unsafe synthetic sequences and bounds non-finite replay rates', () => {
    expect(() => createSyntheticNlosFrame(Number.MAX_SAFE_INTEGER + 1, NOW)).toThrow(RangeError);
    expect(() => createSyntheticNlosFrame(0, Number.MAX_SAFE_INTEGER)).toThrow(RangeError);

    jest.useFakeTimers();
    try {
      const { service } = createHarness();
      const listener = jest.fn();
      service.subscribe(listener);
      service.startDeterministicReplay(Number.NaN);
      jest.advanceTimersByTime(66);
      expect(listener).toHaveBeenCalledTimes(1);
      jest.advanceTimersByTime(1);
      expect(listener).toHaveBeenCalledTimes(2);
      service.disconnect();
    } finally {
      jest.useRealTimers();
    }
  });

  it('reports velocity in metres per second at the selected replay rate', () => {
    const slow = createSyntheticNlosFrame(0, NOW, 10).tracks[0].velocityMps;
    const fast = createSyntheticNlosFrame(0, NOW, 20).tracks[0].velocityMps;
    expect(fast.x).toBeCloseTo(slow.x * 2, 6);
    expect(fast.y).toBeCloseTo(slow.y * 2, 6);
    expect(fast.z).toBeCloseTo(slow.z * 2, 6);
  });
});
