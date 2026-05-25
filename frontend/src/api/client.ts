/**
 * HOMECORE API client.
 *
 * REST:  fetch-based, bearer token auth. Base URL defaults to window.location.origin
 *        so the Vite dev-server proxy handles the `/api` → `:8123` rewrite.
 * WS:    native WebSocket, mirrors HA's ws handshake protocol (auth_required → auth → auth_ok).
 */

import type {
  ApiConfig,
  ServiceDomainView,
  StateView,
  WsAuthOk,
  WsAuthRequired,
  WsServerMessage,
} from './types.js';

export interface ClientOptions {
  baseUrl?: string;
  token: string;
}

export class HomecoreClient {
  private readonly base: string;
  private readonly token: string;

  constructor(options: ClientOptions) {
    this.base = options.baseUrl ?? '';
    this.token = options.token;
  }

  // ── REST helpers ────────────────────────────────────────────────────────────

  private headers(): HeadersInit {
    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };
  }

  private async get<T>(path: string): Promise<T> {
    const resp = await fetch(`${this.base}${path}`, {
      method: 'GET',
      headers: this.headers(),
    });
    if (!resp.ok) {
      throw new Error(`GET ${path} → ${resp.status} ${resp.statusText}`);
    }
    return resp.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const resp = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(`POST ${path} → ${resp.status} ${resp.statusText}`);
    }
    return resp.json() as Promise<T>;
  }

  // ── REST endpoints (mirrors rest.rs) ─────────────────────────────────────

  getConfig(): Promise<ApiConfig> {
    return this.get<ApiConfig>('/api/config');
  }

  getStates(): Promise<StateView[]> {
    return this.get<StateView[]>('/api/states');
  }

  getState(entityId: string): Promise<StateView> {
    return this.get<StateView>(`/api/states/${encodeURIComponent(entityId)}`);
  }

  setState(entityId: string, state: string, attributes?: Record<string, unknown>): Promise<StateView> {
    return this.post<StateView>(`/api/states/${encodeURIComponent(entityId)}`, {
      state,
      attributes: attributes ?? {},
    });
  }

  getServices(): Promise<ServiceDomainView[]> {
    return this.get<ServiceDomainView[]>('/api/services');
  }

  callService(domain: string, service: string, data?: unknown): Promise<unknown> {
    return this.post<unknown>(`/api/services/${domain}/${service}`, data ?? {});
  }

  // ── WebSocket ────────────────────────────────────────────────────────────

  /**
   * Open an authenticated WebSocket connection.
   * Resolves once `auth_ok` is received; rejects on auth failure or network error.
   * Returns the live socket; caller is responsible for `.close()`.
   */
  openWebSocket(wsBase?: string): Promise<WebSocket> {
    const resolved = wsBase ?? this.base.replace(/^http/, 'ws');
    const origin = resolved || window.location.origin.replace(/^http/, 'ws');
    const url = `${origin}/api/websocket`;

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);

      ws.onmessage = (evt: MessageEvent<string>) => {
        const msg = JSON.parse(evt.data) as WsServerMessage;

        if ((msg as WsAuthRequired).type === 'auth_required') {
          ws.send(JSON.stringify({ type: 'auth', access_token: this.token }));
          return;
        }

        if ((msg as WsAuthOk).type === 'auth_ok') {
          ws.onmessage = null;
          resolve(ws);
          return;
        }

        if (msg.type === 'auth_invalid') {
          ws.close();
          reject(new Error(`WS auth_invalid`));
        }
      };

      ws.onerror = () => reject(new Error('WebSocket connection error'));
      ws.onclose = () => reject(new Error('WebSocket closed before auth_ok'));
    });
  }
}
