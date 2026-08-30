/** ruvnet/worldgraph serde and twin-stream wire contract. */
export interface EnuPoint { east_m: number; north_m: number; up_m: number }
export type WorldBounds =
  | { shape: 'rectangle'; min_e: number; min_n: number; max_e: number; max_n: number }
  | { shape: 'circle'; center_e: number; center_n: number; radius_m: number }
  | { shape: 'polygon'; vertices: Array<[number, number]> };
export type WorldNode =
  | { kind: 'room'; id: number; area_id: string | null; name: string; bounds_enu: WorldBounds; floor: number }
  | { kind: 'zone'; id: number; parent_room: number; name: string; bounds_enu: WorldBounds }
  | { kind: 'wall'; id: number; a: EnuPoint; b: EnuPoint; rf_attenuation_db: number }
  | { kind: 'doorway'; id: number; center: EnuPoint; width_m: number }
  | { kind: 'sensor'; id: number; device_id: string; position: EnuPoint; modality: string }
  | { kind: 'rf_link'; id: number; tx: number; rx: number; link_group_id: string | null; center_freq_mhz: number }
  | { kind: 'person_track'; id: number; track_id: number; last_position: EnuPoint; reid_embedding_ref: string | null }
  | { kind: 'object_anchor'; id: number; position: EnuPoint; anchor_kind: string; confidence: number }
  | { kind: 'event'; id: number; event_type: string; at_unix_ms: number; located_in: number | null }
  | { kind: 'semantic_state'; id: number; statement: string; confidence: number; provenance: { evidence: string[]; model_version: string; calibration_version: string; privacy_decision: string }; valid_from_unix_ms: number };
export interface WorldEdgeRecord { id: number; from: number; to: number; edge: Record<string, unknown> }
export type LegacyWorldEdge = [number, number, Record<string, unknown>];
export interface WorldGraphSnapshot {
  schema_version: number; registration?: { origin: { lat: number; lon: number; alt: number }; heading_deg: number; scale: number };
  next_id?: number; next_edge_id?: number; nodes: WorldNode[]; edges: Array<WorldEdgeRecord | LegacyWorldEdge>;
}
export type WorldGraphStreamStatus = 'idle' | 'connecting' | 'live' | 'error';
export type TwinMessage =
  | { op: 'snapshot'; graph_schema_version: number; rvf_json: string }
  | { op: 'upsert_node'; node: WorldNode }
  | { op: 'remove_node'; id: number }
  | { op: 'upsert_edge'; id: number; from: number; to: number; edge: Record<string, unknown> }
  | { op: 'remove_edge'; id: number }
  | { op: 'presence'; updates: unknown[] };
export interface TwinEnvelope { protocol_version: number; stream_epoch: string; seq: number; message: TwinMessage }
