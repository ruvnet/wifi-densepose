import { buildWorldGraphSnapshotUrl, buildWorldGraphStreamUrl, WorldGraphStreamClient } from '@/services/worldGraph.service';

class FakeSocket {
  static instances: FakeSocket[] = [];
  readyState = 1; sent: string[] = [];
  onopen: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null; onclose: (() => void) | null = null;
  constructor(public url: string) { FakeSocket.instances.push(this); }
  send(value: string) { this.sent.push(value); }
  close() { this.onclose?.(); }
  emit(value: unknown) { this.onmessage?.({ data: JSON.stringify(value) }); }
}

describe('WorldGraphStreamClient', () => {
  beforeEach(() => { FakeSocket.instances = []; (global as any).WebSocket = FakeSocket; });

  it('builds the governed twin endpoint and rejects plaintext remote origins', () => {
    expect(buildWorldGraphStreamUrl('https://graph.example/base')).toBe('wss://graph.example/v1/twin/ws');
    expect(buildWorldGraphStreamUrl('http://localhost:8080')).toBe('ws://localhost:8080/v1/twin/ws');
    expect(() => buildWorldGraphStreamUrl('http://graph.example')).toThrow('requires TLS');
  });

  it('reads the real sensing-server WorldGraph snapshot with an optional bearer', async () => {
    const graph = { schema_version: 1, nodes: [], edges: [] };
    const fetchMock = jest.fn(async () => ({ ok: true, status: 200, json: async () => ({ protocol_version: 1, stream_epoch: 'server-7', seq: 12, graph }) }));
    (global as any).fetch = fetchMock;
    const client = new WorldGraphStreamClient();
    await expect(client.fetchSnapshot('http://192.168.1.20:8080/base', 'read-token')).resolves.toEqual({ graph, epoch: 'server-7', seq: 12 });
    expect(buildWorldGraphSnapshotUrl('ws://localhost:8080/base')).toBe('http://localhost:8080/api/v1/worldgraph/snapshot');
    expect(fetchMock).toHaveBeenCalledWith('http://192.168.1.20:8080/api/v1/worldgraph/snapshot', expect.objectContaining({ headers: expect.objectContaining({ Authorization: 'Bearer read-token' }) }));
  });

  it('negotiates without putting the token in the URL and applies a snapshot', () => {
    const client = new WorldGraphStreamClient(); const graphs: any[] = [];
    client.connect('https://graph.example', 'secret-token', { onStatus: jest.fn(), onGraph: (value) => graphs.push(value) });
    const socket = FakeSocket.instances[0]; expect(socket.url).not.toContain('secret-token'); socket.onopen?.();
    expect(JSON.parse(socket.sent[0])).toMatchObject({ type: 'client_hello', access_token: 'secret-token' });
    socket.emit({ type: 'server_hello', protocol_version: 1, capabilities: ['snapshot', 'delta'], stream_epoch: 'epoch-a' });
    socket.emit({ protocol_version: 1, stream_epoch: 'epoch-a', seq: 1, message: { op: 'snapshot', graph_schema_version: 2, rvf_json: JSON.stringify({ schema_version: 2, nodes: [], edges: [] }) } });
    expect(graphs[0]).toMatchObject({ epoch: 'epoch-a', seq: 1, graph: { nodes: [] } });
  });

  it('requests a new snapshot when a delta sequence has a gap', () => {
    const client = new WorldGraphStreamClient();
    client.connect('https://graph.example', 'token', { onStatus: jest.fn(), onGraph: jest.fn() });
    const socket = FakeSocket.instances[0]; socket.onopen?.();
    socket.emit({ type: 'server_hello', protocol_version: 1, capabilities: [], stream_epoch: 'e' });
    socket.emit({ protocol_version: 1, stream_epoch: 'e', seq: 1, message: { op: 'snapshot', graph_schema_version: 2, rvf_json: JSON.stringify({ schema_version: 2, nodes: [], edges: [] }) } });
    socket.emit({ protocol_version: 1, stream_epoch: 'e', seq: 3, message: { op: 'upsert_node', node: { kind: 'sensor', id: 1, device_id: 'x', position: { east_m: 0, north_m: 0, up_m: 0 }, modality: 'wifi_csi' } } });
    expect(socket.sent.map((entry) => JSON.parse(entry)).at(-1)).toMatchObject({ type: 'request_snapshot', reason: 'sequence_gap' });
  });
});
