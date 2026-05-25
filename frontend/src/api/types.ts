/**
 * TypeScript types mirroring the JSON shapes from homecore-api/src/rest.rs and ws.rs.
 * Keep in sync with Rust `StateView`, `ApiConfig`, `ServiceDomainView`.
 */

/** Context for a state change — mirrors Rust `ContextView`. */
export interface ContextView {
  id: string;
  user_id: string | null;
  parent_id: string | null;
}

/** Snapshot of a single entity state — mirrors Rust `StateView`. */
export interface StateView {
  entity_id: string;
  state: string;
  /** Arbitrary JSON attributes attached to the entity. */
  attributes: Record<string, unknown>;
  /** RFC 3339 timestamp of last state value change. */
  last_changed: string;
  /** RFC 3339 timestamp of last update (attributes may have changed). */
  last_updated: string;
  context: ContextView;
}

/** HOMECORE configuration — mirrors Rust `ApiConfig`. */
export interface ApiConfig {
  location_name: string;
  version: string;
  state: 'RUNNING' | 'STARTING' | 'STOPPING';
  components: string[];
}

/** Services grouped by domain — mirrors Rust `ServiceDomainView`. */
export interface ServiceDomainView {
  domain: string;
  /** Keyed by service name; value is the service schema (may be empty `{}`). */
  services: Record<string, unknown>;
}

// ── WebSocket protocol types ──────────────────────────────────────────────────

/** Sent by server immediately upon WS upgrade. */
export interface WsAuthRequired {
  type: 'auth_required';
  ha_version: string;
}

/** Sent by client to authenticate. */
export interface WsAuth {
  type: 'auth';
  access_token: string;
}

/** Sent by server on successful auth. */
export interface WsAuthOk {
  type: 'auth_ok';
  ha_version: string;
}

/** Sent by server on failed auth. */
export interface WsAuthInvalid {
  type: 'auth_invalid';
  message: string;
}

/** Generic result message from server. */
export interface WsResult<T = unknown> {
  id: number;
  type: 'result';
  success: boolean;
  result?: T;
  error?: { code: string; message: string };
}

/** State-changed event pushed by server via `subscribe_events`. */
export interface WsStateChangedEvent {
  id: number;
  type: 'event';
  event: {
    event_type: 'state_changed';
    data: {
      entity_id: string;
      old_state: StateView | null;
      new_state: StateView | null;
    };
    origin: 'LOCAL' | 'REMOTE';
    time_fired: string;
  };
}

/** Union of all inbound WS server messages. */
export type WsServerMessage =
  | WsAuthRequired
  | WsAuthOk
  | WsAuthInvalid
  | WsResult
  | WsStateChangedEvent;
