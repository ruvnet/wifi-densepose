// We test the WsService class by importing a fresh instance.
// We need to mock the poseStore to prevent side effects.
jest.mock('@/stores/poseStore', () => ({
  usePoseStore: {
    getState: jest.fn(() => ({
      setConnectionStatus: jest.fn(),
    })),
  },
}));

jest.mock('@/services/simulation.service', () => ({
  generateSimulatedData: jest.fn(() => ({
    type: 'sensing_update',
    timestamp: Date.now(),
    source: 'simulated',
    nodes: [],
    features: { mean_rssi: -45, variance: 1 },
    classification: { motion_level: 'absent', presence: false, confidence: 0.5 },
    signal_field: { grid_size: [20, 1, 20], values: [] },
  })),
}));

// Create a fresh WsService for each test to avoid shared state
function createWsService() {
  // Use jest.isolateModules to get a fresh module instance
  let service: any;
  jest.isolateModules(() => {
    service = require('@/services/ws.service').wsService;
  });
  return service;
}

describe('WsService', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('buildWsUrl', () => {
    it('uses the same port as the HTTP URL, not a hardcoded port', () => {
      // This is the critical bug-fix verification.
      // buildWsUrl is private, so we test it indirectly via connect().
      // We mock WebSocket to capture the URL it is called with.
      const capturedUrls: string[] = [];
      const OrigWebSocket = globalThis.WebSocket;

      class MockWebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        onopen: (() => void) | null = null;
        onclose: (() => void) | null = null;
        onerror: (() => void) | null = null;
        onmessage: (() => void) | null = null;
        close() {}
        constructor(url: string) {
          capturedUrls.push(url);
        }
      }

      globalThis.WebSocket = MockWebSocket as any;

      try {
        const ws = createWsService();

        // Test with port 3000
        ws.connect('http://192.168.1.10:3000');
        expect(capturedUrls[capturedUrls.length - 1]).toBe('ws://192.168.1.10:3000/ws/sensing');

        // Clean up, create another service
        ws.disconnect();
        const ws2 = createWsService();

        // Test with port 8080
        ws2.connect('http://myserver.local:8080');
        expect(capturedUrls[capturedUrls.length - 1]).toBe('ws://myserver.local:8080/ws/sensing');
        ws2.disconnect();

        // Test HTTPS -> WSS upgrade (port 443 is default for HTTPS so host drops it)
        const ws3 = createWsService();
        ws3.connect('https://secure.example.com:443');
        expect(capturedUrls[capturedUrls.length - 1]).toBe('wss://secure.example.com/ws/sensing');
        ws3.disconnect();

        // Test WSS input
        const ws4 = createWsService();
        ws4.connect('wss://secure.example.com');
        expect(capturedUrls[capturedUrls.length - 1]).toBe('wss://secure.example.com/ws/sensing');
        ws4.disconnect();

        // Verify port 3001 is NOT hardcoded anywhere
        for (const url of capturedUrls) {
          expect(url).not.toContain(':3001');
        }
      } finally {
        globalThis.WebSocket = OrigWebSocket;
      }
    });
  });

  describe('timestamp normalization', () => {
    it('converts Rust Unix seconds to React Native milliseconds', () => {
      const { normalizeFrameTimestamp } = require('@/services/ws.service');
      const frame = { timestamp: 1_787_587_655, nodes: [], features: {}, classification: {}, signal_field: {} };
      expect(normalizeFrameTimestamp(frame).timestamp).toBe(1_787_587_655_000);
    });

    it('leaves millisecond timestamps unchanged', () => {
      const { normalizeFrameTimestamp } = require('@/services/ws.service');
      const frame = { timestamp: 1_787_587_655_000, nodes: [], features: {}, classification: {}, signal_field: {} };
      expect(normalizeFrameTimestamp(frame)).toBe(frame);
    });
  });

  describe('connect with empty URL', () => {
    it('falls back to simulation mode when URL is empty', () => {
      const ws = createWsService();
      ws.connect('');
      expect(ws.getStatus()).toBe('simulated');
      ws.disconnect();
    });
  });

  describe('connection fallback', () => {
    it('uses simulation when an active server connection closes normally', () => {
      const OrigWebSocket = globalThis.WebSocket;
      const sockets: any[] = [];

      class MockWebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        onopen: (() => void) | null = null;
        onclose: ((event: { code: number }) => void) | null = null;
        onerror: (() => void) | null = null;
        onmessage: (() => void) | null = null;
        close() {}
        constructor() {
          sockets.push(this);
        }
      }

      globalThis.WebSocket = MockWebSocket as any;
      try {
        const ws = createWsService();
        ws.connect('http://localhost:3000');
        sockets[0].onclose?.({ code: 1000 });

        expect(ws.getStatus()).toBe('simulated');
        ws.disconnect();
      } finally {
        globalThis.WebSocket = OrigWebSocket;
      }
    });

    it('preserves retry progress instead of restarting the first attempt', () => {
      const OrigWebSocket = globalThis.WebSocket;
      const sockets: any[] = [];

      class MockWebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        onopen: (() => void) | null = null;
        onclose: ((event: { code: number }) => void) | null = null;
        onerror: (() => void) | null = null;
        onmessage: (() => void) | null = null;
        close() {}
        constructor() {
          sockets.push(this);
        }
      }

      globalThis.WebSocket = MockWebSocket as any;
      try {
        const ws = createWsService();
        ws.connect('http://localhost:3000');
        sockets[0].onclose?.({ code: 1006 });
        jest.runOnlyPendingTimers();
        sockets[1].onclose?.({ code: 1006 });

        expect(ws.getStatus()).toBe('simulated');
        expect(sockets).toHaveLength(2);
        ws.disconnect();
      } finally {
        globalThis.WebSocket = OrigWebSocket;
      }
    });
  });

  describe('hardware presentation rate', () => {
    it('drops burst frames above 4 fps so native screens stay current', () => {
      const OrigWebSocket = globalThis.WebSocket;
      const sockets: any[] = [];

      class MockWebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        onopen: (() => void) | null = null;
        onclose: (() => void) | null = null;
        onerror: (() => void) | null = null;
        onmessage: ((event: { data: string }) => void) | null = null;
        close() {}
        constructor() { sockets.push(this); }
      }

      globalThis.WebSocket = MockWebSocket as any;
      try {
        const ws = createWsService();
        const listener = jest.fn();
        ws.subscribe(listener);
        ws.connect('http://localhost:3000');
        sockets[0].onopen?.();
        const frame = JSON.stringify({
          timestamp: Date.now(), nodes: [], features: {}, classification: {}, signal_field: {},
        });

        sockets[0].onmessage?.({ data: frame });
        sockets[0].onmessage?.({ data: frame });
        expect(listener).toHaveBeenCalledTimes(1);

        jest.advanceTimersByTime(250);
        sockets[0].onmessage?.({ data: frame });
        expect(listener).toHaveBeenCalledTimes(2);
        ws.disconnect();
      } finally {
        globalThis.WebSocket = OrigWebSocket;
      }
    });

    it('does not replace the screen with partial edge-vitals events', () => {
      const OrigWebSocket = globalThis.WebSocket;
      const sockets: any[] = [];

      class MockWebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        onopen: (() => void) | null = null;
        onclose: (() => void) | null = null;
        onerror: (() => void) | null = null;
        onmessage: ((event: { data: string }) => void) | null = null;
        close() {}
        constructor() { sockets.push(this); }
      }

      globalThis.WebSocket = MockWebSocket as any;
      try {
        const ws = createWsService();
        const listener = jest.fn();
        ws.subscribe(listener);
        ws.connect('http://localhost:3000');
        sockets[0].onopen?.();

        sockets[0].onmessage?.({ data: JSON.stringify({
          type: 'edge_vitals', node_id: 7, breathing_rate_bpm: 18, heartrate_bpm: 72,
        }) });

        expect(listener).not.toHaveBeenCalled();
        ws.disconnect();
      } finally {
        globalThis.WebSocket = OrigWebSocket;
      }
    });
  });

  describe('subscribe and unsubscribe', () => {
    it('adds a listener and returns an unsubscribe function', () => {
      const ws = createWsService();
      const listener = jest.fn();
      const unsub = ws.subscribe(listener);
      expect(typeof unsub).toBe('function');
      unsub();
      ws.disconnect();
    });

    it('listener receives simulated frames', () => {
      const ws = createWsService();
      const listener = jest.fn();
      ws.subscribe(listener);
      ws.connect('');

      // Advance timer to trigger simulation
      jest.advanceTimersByTime(600);

      expect(listener).toHaveBeenCalled();
      const frame = listener.mock.calls[0][0];
      expect(frame).toHaveProperty('type', 'sensing_update');
      ws.disconnect();
    });

    it('unsubscribed listener does not receive frames', () => {
      const ws = createWsService();
      const listener = jest.fn();
      const unsub = ws.subscribe(listener);
      unsub();
      ws.connect('');

      jest.advanceTimersByTime(600);

      expect(listener).not.toHaveBeenCalled();
      ws.disconnect();
    });
  });

  describe('disconnect', () => {
    it('clears state and sets status to disconnected', () => {
      const ws = createWsService();
      ws.connect('');
      expect(ws.getStatus()).toBe('simulated');
      ws.disconnect();
      expect(ws.getStatus()).toBe('disconnected');
    });
  });

  describe('getStatus', () => {
    it('returns disconnected initially', () => {
      const ws = createWsService();
      expect(ws.getStatus()).toBe('disconnected');
    });
  });
});
