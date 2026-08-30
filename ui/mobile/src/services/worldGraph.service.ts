import type { TwinEnvelope, TwinMessage, WorldEdgeRecord, WorldGraphSnapshot, WorldNode } from '@/types/worldGraph';

const MAX_MESSAGE_BYTES = 512 * 1024;
const SUPPORTED_PROTOCOL = 1;
const SUPPORTED_SCHEMAS = new Set([1, 2]);

export interface WorldGraphUpdate { graph: WorldGraphSnapshot; epoch: string; seq: number }
export interface WorldGraphCallbacks {
  onStatus: (status: 'connecting' | 'live' | 'error' | 'idle', error?: string) => void;
  onGraph: (update: WorldGraphUpdate) => void;
}

export const buildWorldGraphStreamUrl = (origin: string): string => {
  const parsed = new URL(origin);
  parsed.protocol = parsed.protocol === 'https:' || parsed.protocol === 'wss:' ? 'wss:' : 'ws:';
  parsed.pathname = '/v1/twin/ws';
  parsed.search = '';
  parsed.hash = '';
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname);
  if (parsed.protocol === 'ws:' && !loopback) throw new Error('WorldGraph requires TLS (wss://) outside loopback');
  return parsed.toString();
};

export const buildWorldGraphSnapshotUrl = (origin: string): string => {
  const parsed = new URL(origin);
  if (parsed.protocol === 'ws:') parsed.protocol = 'http:';
  if (parsed.protocol === 'wss:') parsed.protocol = 'https:';
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') throw new Error('WorldGraph server URL must use HTTP or HTTPS');
  parsed.pathname = '/api/v1/worldgraph/snapshot';
  parsed.search = '';
  parsed.hash = '';
  return parsed.toString();
};

const isObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object';
const isWorldNode = (value: unknown): value is WorldNode => isObject(value) && typeof value.kind === 'string' && Number.isSafeInteger(value.id);
const normalizeGraph = (value: unknown): WorldGraphSnapshot => {
  if (!isObject(value) || !SUPPORTED_SCHEMAS.has(Number(value.schema_version)) || !Array.isArray(value.nodes) || !value.nodes.every(isWorldNode) || !Array.isArray(value.edges)) {
    throw new Error('WorldGraph snapshot failed schema validation');
  }
  return value as unknown as WorldGraphSnapshot;
};
const isEnvelope = (value: unknown): value is TwinEnvelope => {
  if (!isObject(value) || value.protocol_version !== SUPPORTED_PROTOCOL || typeof value.stream_epoch !== 'string' || !Number.isSafeInteger(value.seq) || !isObject(value.message)) return false;
  return typeof value.message.op === 'string';
};
const edgeId = (edge: WorldGraphSnapshot['edges'][number]): number | null => Array.isArray(edge) ? null : edge.id;

export class WorldGraphStreamClient {
  private socket: WebSocket | null = null;
  private graph: WorldGraphSnapshot | null = null;
  private epoch: string | null = null;
  private seq: number | null = null;
  private awaitingSnapshot = true;

  async fetchSnapshot(origin: string, token = ''): Promise<WorldGraphUpdate> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (token.trim()) headers.Authorization = `Bearer ${token.trim()}`;
    const response = await fetch(buildWorldGraphSnapshotUrl(origin), { method: 'GET', headers });
    let value: unknown;
    try { value = await response.json(); } catch { throw new Error(`WorldGraph returned HTTP ${response.status} with invalid JSON`); }
    if (!response.ok) {
      const message = isObject(value) && typeof value.message === 'string' ? value.message : `WorldGraph returned HTTP ${response.status}`;
      throw new Error(message);
    }
    if (!isObject(value) || value.protocol_version !== SUPPORTED_PROTOCOL || typeof value.stream_epoch !== 'string' || !Number.isSafeInteger(value.seq)) {
      throw new Error('WorldGraph snapshot response failed schema validation');
    }
    return { graph: normalizeGraph(value.graph), epoch: value.stream_epoch, seq: Number(value.seq) };
  }

  connect(origin: string, token: string, callbacks: WorldGraphCallbacks): () => void {
    this.disconnect();
    if (!token.trim()) throw new Error('A short-lived WorldGraph read token is required');
    const socket = new WebSocket(buildWorldGraphStreamUrl(origin));
    this.socket = socket;
    callbacks.onStatus('connecting');
    socket.onopen = () => socket.send(JSON.stringify({
      type: 'client_hello', supported_protocol_versions: [SUPPORTED_PROTOCOL],
      capabilities: ['snapshot', 'delta'], access_token: token.trim(),
    }));
    socket.onmessage = (event) => {
      if (typeof event.data !== 'string' || event.data.length > MAX_MESSAGE_BYTES) {
        callbacks.onStatus('error', 'WorldGraph message was invalid or too large'); return;
      }
      try {
        const value: unknown = JSON.parse(event.data);
        if (isObject(value) && value.type === 'server_hello') {
          if (value.protocol_version !== SUPPORTED_PROTOCOL || typeof value.stream_epoch !== 'string') throw new Error('Unsupported WorldGraph negotiation');
          this.epoch = value.stream_epoch; callbacks.onStatus('live'); return;
        }
        if (isEnvelope(value)) this.applyEnvelope(value, callbacks);
      } catch (error) { callbacks.onStatus('error', error instanceof Error ? error.message : 'WorldGraph stream error'); }
    };
    socket.onerror = () => callbacks.onStatus('error', 'WorldGraph connection failed');
    socket.onclose = () => { if (this.socket === socket) callbacks.onStatus('idle'); };
    return () => this.disconnect();
  }

  disconnect(): void { this.socket?.close(); this.socket = null; this.graph = null; this.epoch = null; this.seq = null; this.awaitingSnapshot = true; }

  private requestSnapshot(reason: string): void {
    this.socket?.send(JSON.stringify({ type: 'request_snapshot', reason, stream_epoch: this.epoch }));
  }

  private applyEnvelope(envelope: TwinEnvelope, callbacks: WorldGraphCallbacks): void {
    const isSnapshot = envelope.message.op === 'snapshot';
    if (this.epoch !== null && envelope.stream_epoch !== this.epoch) {
      this.epoch = envelope.stream_epoch; this.seq = null; this.awaitingSnapshot = true;
      if (!isSnapshot) { this.requestSnapshot('epoch_changed'); return; }
    }
    if (!isSnapshot && (this.awaitingSnapshot || !this.graph)) { this.requestSnapshot('snapshot_required'); return; }
    if (!isSnapshot && this.seq !== null && envelope.seq !== this.seq + 1) {
      if (envelope.seq <= this.seq) return;
      this.awaitingSnapshot = true; this.requestSnapshot('sequence_gap'); return;
    }

    this.graph = this.applyMessage(envelope.message, this.graph);
    this.epoch = envelope.stream_epoch; this.seq = envelope.seq; this.awaitingSnapshot = false;
    callbacks.onGraph({ graph: this.graph, epoch: envelope.stream_epoch, seq: envelope.seq });
  }

  private applyMessage(message: TwinMessage, current: WorldGraphSnapshot | null): WorldGraphSnapshot {
    if (message.op === 'snapshot') {
      if (!SUPPORTED_SCHEMAS.has(message.graph_schema_version)) throw new Error(`Unsupported WorldGraph schema ${message.graph_schema_version}`);
      return normalizeGraph(JSON.parse(message.rvf_json));
    }
    if (!current) throw new Error('WorldGraph delta received before snapshot');
    if (message.op === 'upsert_node') return { ...current, nodes: current.nodes.some((node) => node.id === message.node.id) ? current.nodes.map((node) => node.id === message.node.id ? message.node : node) : [...current.nodes, message.node] };
    if (message.op === 'remove_node') return { ...current, nodes: current.nodes.filter((node) => node.id !== message.id), edges: current.edges.filter((edge) => Array.isArray(edge) ? edge[0] !== message.id && edge[1] !== message.id : edge.from !== message.id && edge.to !== message.id) };
    if (message.op === 'upsert_edge') {
      const next: WorldEdgeRecord = { id: message.id, from: message.from, to: message.to, edge: message.edge };
      return { ...current, edges: current.edges.some((edge) => edgeId(edge) === message.id) ? current.edges.map((edge) => edgeId(edge) === message.id ? next : edge) : [...current.edges, next] };
    }
    if (message.op === 'remove_edge') return { ...current, edges: current.edges.filter((edge) => edgeId(edge) !== message.id) };
    return current;
  }
}

export const worldGraphService = new WorldGraphStreamClient();
