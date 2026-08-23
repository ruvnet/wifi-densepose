export const NLOS_TRACK_SCHEMA = 'ruview.nlos.track.v1' as const;
export const NLOS_MAX_MESSAGE_BYTES = 256 * 1024;
export const NLOS_MAX_TRACKS = 16;
export const NLOS_MAX_EXPIRY_WINDOW_MS = 5_000;
export const NLOS_STALE_AFTER_MS = 1_500;

export type NlosSource = 'live' | 'replay' | 'synthetic';
export type NlosEvidenceLevel =
  | 'l0_synthetic'
  | 'l1_measured'
  | 'l2_calibrated'
  | 'l3_corroborated';
export type NlosTransientKind =
  | 'raw_histogram'
  | 'compact_normalized_histogram'
  | 'depth_only'
  | 'replay';
export type NlosTransport = 'usb_serial' | 'ruview_server' | 'replay';
export type NlosTrackState = 'tracking' | 'degraded' | 'unknown';

export interface NlosVector3 {
  x: number;
  y: number;
  z: number;
}

export interface NlosModalityContributions {
  lidar: number;
  csi: number;
}

export interface NlosTrack {
  trackId: string;
  state: NlosTrackState;
  positionM: NlosVector3;
  velocityMps: NlosVector3;
  covarianceDiagonalM2: NlosVector3;
  confidence: number;
  posteriorEntropy: number;
  signalQuality: number;
  modalityContributions: NlosModalityContributions;
}

export interface NlosProvenance {
  sensorId: string;
  sensorModel: string;
  firmwareVersion: string;
  transientKind: NlosTransientKind;
  histogramPreserved: boolean;
  transport: NlosTransport;
}

export interface NlosTrackFrame {
  schema: typeof NLOS_TRACK_SCHEMA;
  sessionId: string;
  sequence: number;
  capturedAtUnixMs: number;
  expiresAtUnixMs: number;
  source: NlosSource;
  evidenceLevel: NlosEvidenceLevel;
  algorithmVersion: string;
  calibrationHash: string;
  provenance: NlosProvenance;
  tracks: NlosTrack[];
}

export type NlosFreshness = 'unknown' | 'fresh' | 'stale';
export type NlosStreamStatus =
  | 'idle'
  | 'authenticating'
  | 'connecting'
  | 'live'
  | 'synthetic_replay'
  | 'error';

export type NlosFrameChannel = 'authenticated_stream' | 'deterministic_replay';

export interface NlosFrameEvent {
  frame: NlosTrackFrame;
  channel: NlosFrameChannel;
  receivedAtUnixMs: number;
}

export type NlosRejectReason =
  | 'message_too_large'
  | 'malformed_json'
  | 'invalid_schema'
  | 'invalid_shape'
  | 'invalid_bounds'
  | 'invalid_provenance'
  | 'expired'
  | 'future_frame'
  | 'session_mismatch'
  | 'out_of_order'
  | 'unauthenticated'
  | 'unsupported_binary';

export type NlosValidationResult =
  | { ok: true; value: NlosTrackFrame }
  | { ok: false; reason: NlosRejectReason };
