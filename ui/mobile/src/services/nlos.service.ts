import {
  NLOS_MAX_MESSAGE_BYTES,
  NLOS_TRACK_SCHEMA,
  type NlosFrameEvent,
  type NlosRejectReason,
  type NlosStreamStatus,
  type NlosTrackFrame,
} from '@/types/nlos';
import { parseNlosTrackFrame, utf8ByteLength } from './nlos.validation';
import { normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

export const NLOS_WS_TICKET_PATH = '/api/v1/nlos/ws-ticket';
export const NLOS_TICKET_SCHEMA = 'ruview.nlos.ws-ticket.v1' as const;
export const NLOS_AUTHENTICATED_SCHEMA = 'ruview.nlos.authenticated.v1' as const;

const MAX_TICKET_RESPONSE_BYTES = 8 * 1024;
const MAX_TICKET_TTL_MS = 30_000;
const MAX_AUTHENTICATED_SESSION_TTL_MS = 60 * 60 * 1_000;
const MIN_BEARER_TOKEN_LENGTH = 32;
const MAX_BEARER_TOKEN_LENGTH = 512;
const MAX_CLOCK_SKEW_MS = 1_000;
const TRANSPORT_HANDSHAKE_TIMEOUT_MS = 5_000;
const MAX_REPLAY_FPS = 30;
const SYNTHETIC_SESSION_ID = 'synthetic-replay-v1';
const ZERO_CALIBRATION_HASH = '0'.repeat(64);

type FrameListener = (event: NlosFrameEvent) => void;
type StatusListener = (status: NlosStreamStatus) => void;
type RejectListener = (reason: NlosRejectReason) => void;

interface TicketResponse {
  schema: typeof NLOS_TICKET_SCHEMA;
  webSocketUrl: string;
  expiresAtUnixMs: number;
}

interface AuthenticatedMessage {
  schema: typeof NLOS_AUTHENTICATED_SCHEMA;
  sessionId: string;
  expiresAtUnixMs: number;
}

interface FetchResponseLike {
  ok: boolean;
  status: number;
  text: () => Promise<string>;
}

type FetchLike = (input: string, init: RequestInit) => Promise<FetchResponseLike>;

interface WebSocketLike {
  readyState: number;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: (() => void) | null;
  onclose: ((event: { code: number }) => void) | null;
  close: (code?: number, reason?: string) => void;
}

export interface NlosServiceDependencies {
  fetch: FetchLike;
  createWebSocket: (url: string) => WebSocketLike;
  now: () => number;
  setInterval: typeof globalThis.setInterval;
  clearInterval: typeof globalThis.clearInterval;
  setTimeout: typeof globalThis.setTimeout;
  clearTimeout: typeof globalThis.clearTimeout;
}

export interface NlosLiveConfig {
  serverUrl: string;
  bearerToken: string;
}

let ephemeralBearerToken: string | null = null;

const isValidBearerToken = (value: string): boolean =>
  value.length >= MIN_BEARER_TOKEN_LENGTH &&
  value.length <= MAX_BEARER_TOKEN_LENGTH &&
  /^[!-~]+$/.test(value);

/** Stores an NLOS credential in module memory only. It is never persisted. */
export const configureNlosBearerToken = (token: string | null): boolean => {
  if (token === null) {
    ephemeralBearerToken = null;
    return true;
  }
  if (!isValidBearerToken(token)) return false;
  ephemeralBearerToken = token;
  return true;
};

export const hasConfiguredNlosBearerToken = (): boolean => ephemeralBearerToken !== null;

const consumeConfiguredNlosBearerToken = (): string | null => ephemeralBearerToken;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasExactKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean => {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
};

const isSafeId = (value: unknown): value is string =>
  typeof value === 'string' &&
  value.length >= 1 &&
  value.length <= 64 &&
  /^[A-Za-z0-9._:-]+$/.test(value);

const isSafeUnixMs = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

const parseServerUrl = (raw: string): URL | null => {
  const validation = normalizeNlosServerUrl(raw);
  return validation.valid && validation.normalized ? new URL(validation.normalized) : null;
};

const effectivePort = (url: URL): string => {
  if (url.port) return url.port;
  return url.protocol === 'https:' || url.protocol === 'wss:' ? '443' : '80';
};

const parseWebSocketUrl = (raw: string, serverUrl: URL): string | null => {
  try {
    const url = new URL(raw);
    const secure = url.protocol === 'wss:';
    const loopback = url.protocol === 'ws:' &&
      (url.hostname === 'localhost' ||
        url.hostname === '127.0.0.1' ||
        url.hostname === '[::1]' ||
        url.hostname === '::1');
    const expectedProtocol = serverUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const ticketKeys = Array.from(url.searchParams.keys());
    const ticket = url.searchParams.get('ticket');
    if (
      (!secure && !loopback) ||
      url.protocol !== expectedProtocol ||
      url.hostname !== serverUrl.hostname ||
      effectivePort(url) !== effectivePort(serverUrl) ||
      url.pathname !== '/api/v1/nlos/ws' ||
      ticketKeys.length !== 1 ||
      ticketKeys[0] !== 'ticket' ||
      !ticket ||
      !/^[0-9a-f]{64}$/.test(ticket) ||
      url.username ||
      url.password ||
      url.hash
    ) return null;
    return url.toString();
  } catch {
    return null;
  }
};

const parseTicket = (raw: string, now: number, serverUrl: URL): TicketResponse | null => {
  if (utf8ByteLength(raw) > MAX_TICKET_RESPONSE_BYTES) return null;
  try {
    const value = JSON.parse(raw) as unknown;
    if (
      !isRecord(value) ||
      !hasExactKeys(value, ['schema', 'webSocketUrl', 'expiresAtUnixMs']) ||
      value.schema !== NLOS_TICKET_SCHEMA ||
      typeof value.webSocketUrl !== 'string' ||
      parseWebSocketUrl(value.webSocketUrl, serverUrl) === null ||
      !isSafeUnixMs(value.expiresAtUnixMs) ||
      value.expiresAtUnixMs <= now ||
      value.expiresAtUnixMs - now > MAX_TICKET_TTL_MS
    ) {
      return null;
    }
    return value as unknown as TicketResponse;
  } catch {
    return null;
  }
};

const parseAuthenticatedMessage = (raw: string, now: number): AuthenticatedMessage | null => {
  if (utf8ByteLength(raw) > NLOS_MAX_MESSAGE_BYTES) return null;
  try {
    const value = JSON.parse(raw) as unknown;
    if (
      !isRecord(value) ||
      !hasExactKeys(value, ['schema', 'sessionId', 'expiresAtUnixMs']) ||
      value.schema !== NLOS_AUTHENTICATED_SCHEMA ||
      !isSafeId(value.sessionId) ||
      !isSafeUnixMs(value.expiresAtUnixMs) ||
      value.expiresAtUnixMs <= now ||
      value.expiresAtUnixMs - now > MAX_AUTHENTICATED_SESSION_TTL_MS + MAX_CLOCK_SKEW_MS
    ) {
      return null;
    }
    return value as unknown as AuthenticatedMessage;
  } catch {
    return null;
  }
};

export const createSyntheticNlosFrame = (
  sequence: number,
  now: number,
  fps = 15,
): NlosTrackFrame => {
  if (
    !Number.isSafeInteger(sequence) ||
    sequence < 0 ||
    !Number.isSafeInteger(now) ||
    now < 0 ||
    now > Number.MAX_SAFE_INTEGER - 1_000 ||
    !Number.isFinite(fps) ||
    fps < 1 ||
    fps > MAX_REPLAY_FPS
  ) {
    throw new RangeError('Synthetic sequence and timestamp must be safe unsigned integers');
  }
  const phase = sequence / 12;
  const phaseRate = fps / 12;
  const x = 2.4 + Math.sin(phase) * 1.3;
  const y = 1.05 + Math.sin(phase * 0.4) * 0.08;
  const zPhase = phase * 0.7;
  const z = 3.3 + Math.cos(zPhase) * 0.9;
  const vx = Math.cos(phase) * 1.3 * phaseRate;
  const vy = Math.cos(phase * 0.4) * 0.08 * 0.4 * phaseRate;
  const vz = -Math.sin(zPhase) * 0.9 * 0.7 * phaseRate;

  return {
    schema: NLOS_TRACK_SCHEMA,
    sessionId: SYNTHETIC_SESSION_ID,
    sequence,
    capturedAtUnixMs: now,
    expiresAtUnixMs: now + 1_000,
    source: 'synthetic',
    evidenceLevel: 'l0_synthetic',
    algorithmVersion: 'synthetic-replay-v1',
    calibrationHash: ZERO_CALIBRATION_HASH,
    provenance: {
      sensorId: 'synthetic-sensor',
      sensorModel: 'deterministic-fixture',
      firmwareVersion: 'fixture-v1',
      transientKind: 'replay',
      histogramPreserved: false,
      transport: 'replay',
    },
    tracks: [
      {
        trackId: 'synthetic-target-1',
        state: 'tracking',
        positionM: { x, y, z },
        velocityMps: { x: vx, y: vy, z: vz },
        covarianceDiagonalM2: { x: 0.12, y: 0.18, z: 0.14 },
        confidence: 0.72,
        posteriorEntropy: 0.68,
        signalQuality: 0.64,
        modalityContributions: { lidar: 0.55, csi: 0.45 },
      },
    ],
  };
};

const defaultDependencies = (): NlosServiceDependencies => ({
  fetch: (input, init) => fetch(input, init) as Promise<FetchResponseLike>,
  createWebSocket: (url) => new WebSocket(url) as unknown as WebSocketLike,
  now: Date.now,
  setInterval: globalThis.setInterval.bind(globalThis),
  clearInterval: globalThis.clearInterval.bind(globalThis),
  setTimeout: globalThis.setTimeout.bind(globalThis),
  clearTimeout: globalThis.clearTimeout.bind(globalThis),
});

export class NlosService {
  private readonly dependencies: NlosServiceDependencies;
  private frameListeners = new Set<FrameListener>();
  private statusListeners = new Set<StatusListener>();
  private rejectListeners = new Set<RejectListener>();
  private socket: WebSocketLike | null = null;
  private replayTimer: ReturnType<typeof setInterval> | null = null;
  private authenticationTimer: ReturnType<typeof setTimeout> | null = null;
  private sessionExpiryTimer: ReturnType<typeof setTimeout> | null = null;
  private ticketAbortController: AbortController | null = null;
  private status: NlosStreamStatus = 'idle';
  private generation = 0;
  private authenticatedSession: AuthenticatedMessage | null = null;
  private lastSequence = -1;
  private replaySequence = 0;

  constructor(dependencies: NlosServiceDependencies = defaultDependencies()) {
    this.dependencies = dependencies;
  }

  subscribe(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    return () => this.frameListeners.delete(listener);
  }

  subscribeStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  subscribeRejected(listener: RejectListener): () => void {
    this.rejectListeners.add(listener);
    return () => this.rejectListeners.delete(listener);
  }

  getStatus(): NlosStreamStatus {
    return this.status;
  }

  async connectLive(config: NlosLiveConfig): Promise<boolean> {
    this.stopTransport();
    const generation = this.generation;
    const serverUrl = parseServerUrl(config.serverUrl);
    if (!serverUrl || !isValidBearerToken(config.bearerToken)) {
      this.setStatus('error');
      this.emitRejected('unauthenticated');
      return false;
    }

    this.setStatus('authenticating');
    const ticketUrl = new URL(NLOS_WS_TICKET_PATH, serverUrl).toString();
    const abortController = new AbortController();
    this.ticketAbortController = abortController;
    const ticketTimer = this.dependencies.setTimeout(
      () => abortController.abort(),
      TRANSPORT_HANDSHAKE_TIMEOUT_MS,
    );

    try {
      const response = await this.dependencies.fetch(ticketUrl, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          Authorization: `Bearer ${config.bearerToken}`,
        },
        body: '',
        credentials: 'omit',
        redirect: 'error',
        signal: abortController.signal,
      });
      if (generation !== this.generation) return false;
      if (!response.ok) {
        this.setStatus('error');
        this.emitRejected('unauthenticated');
        return false;
      }

      const ticket = parseTicket(await response.text(), this.dependencies.now(), serverUrl);
      if (generation !== this.generation) return false;
      if (!ticket) {
        this.setStatus('error');
        this.emitRejected('invalid_shape');
        return false;
      }

      this.openSocket(ticket.webSocketUrl, generation);
      return true;
    } catch {
      if (generation === this.generation) {
        this.setStatus('error');
        this.emitRejected('unauthenticated');
      }
      return false;
    } finally {
      this.dependencies.clearTimeout(ticketTimer);
      if (this.ticketAbortController === abortController) this.ticketAbortController = null;
    }
  }

  connectConfiguredLive(serverUrl: string): Promise<boolean> {
    const token = consumeConfiguredNlosBearerToken();
    if (!token) {
      this.setStatus('error');
      this.emitRejected('unauthenticated');
      return Promise.resolve(false);
    }
    return this.connectLive({ serverUrl, bearerToken: token });
  }

  startDeterministicReplay(fps = 15): void {
    this.stopTransport();
    const requestedFps = Number.isFinite(fps) ? Math.floor(fps) : 15;
    const boundedFps = Math.max(1, Math.min(MAX_REPLAY_FPS, requestedFps));
    const emitFrame = () => {
      const receivedAtUnixMs = this.dependencies.now();
      const frame = createSyntheticNlosFrame(this.replaySequence, receivedAtUnixMs, boundedFps);
      this.replaySequence += 1;
      this.emitFrame({ frame, channel: 'deterministic_replay', receivedAtUnixMs });
    };

    this.setStatus('synthetic_replay');
    emitFrame();
    this.replayTimer = this.dependencies.setInterval(emitFrame, Math.ceil(1_000 / boundedFps));
  }

  disconnect(): void {
    this.stopTransport();
    this.setStatus('idle');
  }

  private openSocket(url: string, generation: number): void {
    this.setStatus('connecting');
    const socket = this.dependencies.createWebSocket(url);
    this.socket = socket;
    this.authenticationTimer = this.dependencies.setTimeout(() => {
      if (generation !== this.generation || this.authenticatedSession) return;
      this.authenticationTimer = null;
      this.emitRejected('unauthenticated');
      this.setStatus('error');
      socket.close(1008, 'authentication timeout');
    }, TRANSPORT_HANDSHAKE_TIMEOUT_MS);

    socket.onopen = () => {
      if (generation === this.generation) this.setStatus('connecting');
    };
    socket.onmessage = (event) => {
      if (generation !== this.generation || socket !== this.socket) return;
      this.handleSocketMessage(event.data);
    };
    socket.onerror = () => {
      if (generation === this.generation) this.setStatus('error');
    };
    socket.onclose = (event) => {
      if (generation !== this.generation) return;
      this.socket = null;
      this.authenticatedSession = null;
      this.clearAuthenticationTimer();
      this.clearSessionExpiryTimer();
      if (event.code !== 1000) this.setStatus('error');
      else this.setStatus('idle');
    };
  }

  private handleSocketMessage(data: unknown): void {
    if (typeof data !== 'string') {
      this.emitRejected('unsupported_binary');
      return;
    }

    const now = this.dependencies.now();
    if (!this.authenticatedSession) {
      const authenticated = parseAuthenticatedMessage(data, now);
      if (!authenticated) {
        this.emitRejected('unauthenticated');
        this.clearAuthenticationTimer();
        this.setStatus('error');
        this.socket?.close(1008, 'authentication required');
        return;
      }
      this.authenticatedSession = authenticated;
      this.lastSequence = -1;
      this.clearAuthenticationTimer();
      this.scheduleSessionExpiry(authenticated, now);
      this.setStatus('live');
      return;
    }

    if (this.authenticatedSession.expiresAtUnixMs <= now) {
      this.emitRejected('unauthenticated');
      this.socket?.close(1008, 'session expired');
      return;
    }

    const parsed = parseNlosTrackFrame(data);
    if (!parsed.ok) {
      this.emitRejected(parsed.reason);
      return;
    }
    const frame = parsed.value;
    const authenticatedLive =
      frame.source === 'live' &&
      frame.provenance.transport === 'ruview_server' &&
      frame.provenance.histogramPreserved;
    const authenticatedSynthetic =
      frame.source === 'synthetic' && frame.provenance.transport === 'replay';
    if (!authenticatedLive && !authenticatedSynthetic) {
      this.emitRejected('invalid_provenance');
      return;
    }
    if (frame.sessionId !== this.authenticatedSession.sessionId) {
      this.emitRejected('session_mismatch');
      return;
    }
    if (frame.sequence <= this.lastSequence) {
      this.emitRejected('out_of_order');
      return;
    }
    if (frame.capturedAtUnixMs > now + MAX_CLOCK_SKEW_MS) {
      this.emitRejected('future_frame');
      return;
    }
    if (frame.expiresAtUnixMs <= now) {
      this.emitRejected('expired');
      return;
    }

    this.lastSequence = frame.sequence;
    this.emitFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: now });
  }

  private stopTransport(): void {
    this.generation += 1;
    this.authenticatedSession = null;
    this.lastSequence = -1;
    this.replaySequence = 0;
    this.ticketAbortController?.abort();
    this.ticketAbortController = null;
    this.clearAuthenticationTimer();
    this.clearSessionExpiryTimer();
    if (this.replayTimer !== null) {
      this.dependencies.clearInterval(this.replayTimer);
      this.replayTimer = null;
    }
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.close(1000, 'client disconnect');
    }
  }

  private setStatus(status: NlosStreamStatus): void {
    if (status === this.status) return;
    this.status = status;
    this.statusListeners.forEach((listener) => listener(status));
  }

  private clearAuthenticationTimer(): void {
    if (this.authenticationTimer !== null) {
      this.dependencies.clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
  }

  private scheduleSessionExpiry(session: AuthenticatedMessage, now: number): void {
    this.clearSessionExpiryTimer();
    const delay = Math.max(0, session.expiresAtUnixMs - now);
    this.sessionExpiryTimer = this.dependencies.setTimeout(() => {
      if (this.authenticatedSession !== session) return;
      this.sessionExpiryTimer = null;
      this.authenticatedSession = null;
      this.emitRejected('unauthenticated');
      this.setStatus('error');
      this.socket?.close(1008, 'session expired');
    }, delay);
  }

  private clearSessionExpiryTimer(): void {
    if (this.sessionExpiryTimer !== null) {
      this.dependencies.clearTimeout(this.sessionExpiryTimer);
      this.sessionExpiryTimer = null;
    }
  }

  private emitFrame(event: NlosFrameEvent): void {
    this.frameListeners.forEach((listener) => listener(event));
  }

  private emitRejected(reason: NlosRejectReason): void {
    this.rejectListeners.forEach((listener) => listener(reason));
  }
}

export const nlosService = new NlosService();
