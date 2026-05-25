/**
 * Shared domain types for the RuView MCP server.
 *
 * These mirror the JSON schemas emitted by cog-pose-estimation (ADR-101) and
 * cog-person-count (ADR-103), and the REST payloads from wifi-densepose-sensing-server
 * (ADR-102).
 */

// ── CSI ────────────────────────────────────────────────────────────────────

/**
 * A single CSI window as stored in paired JSONL files.
 * 56 subcarriers × 20 frames per window (the standard ESP32-S3 shape).
 */
export interface CsiWindow {
  /** Timestamp of the last frame in the window (seconds since epoch). */
  ts: number;
  /** Subcarrier amplitudes [56][20]. */
  amplitudes: number[][];
  /** Subcarrier phases [56][20], unwrapped (radians). */
  phases: number[][];
  /** Number of TX/RX antenna paths captured (1×1 SISO = 1). */
  n_paths: number;
  /** Source node MAC address, if known. */
  node_mac?: string | undefined;
}

/**
 * Sensing-server `/api/v1/sensing/latest` response shape.
 */
export interface SensingLatestResponse {
  window: CsiWindow;
  /** Sensing server schema version (pinned to 2 per ADR-101 frame_subscriber.rs). */
  schema_version: number;
  /** ISO-8601 wall timestamp when the server last received a frame. */
  captured_at: string;
}

// ── Pose ──────────────────────────────────────────────────────────────────

/**
 * A single detected person's 17 COCO keypoints.
 * Each keypoint is [x, y] in [0, 1] image-normalized coords.
 */
export interface PersonPose {
  /** 17 keypoints in COCO order (nose, left_eye, right_eye, …, right_ankle). */
  keypoints: [number, number][];
  /** Model confidence in this person's pose estimate [0, 1]. */
  confidence: number;
}

/** Output of ruview_pose_infer. */
export interface PoseInferResult {
  ts: number;
  n_persons: number;
  persons: PersonPose[];
  /** Backend used ("candle-cuda" | "candle-cpu" | "onnx" | "stub"). */
  backend: string;
  /** Inference latency (ms). */
  latency_ms: number;
}

// ── Person Count ──────────────────────────────────────────────────────────

/** Output of ruview_count_infer (ADR-103 person-count cog). */
export interface CountInferResult {
  ts: number;
  count: number;
  confidence: number;
  count_p95_low: number;
  count_p95_high: number;
  /** Per-node breakdown when multi-node fusion was applied. */
  per_node_breakdown?: Array<{ node_mac: string; count: number; confidence: number }> | undefined;
  backend: string;
  latency_ms: number;
}

// ── Registry ──────────────────────────────────────────────────────────────

/** A single cog entry from the Cognitum app-registry.json. */
export interface CogEntry {
  id: string;
  name: string;
  category: string;
  version: string;
  description: string;
  size_kb: number;
  difficulty: string;
  sha256?: string | undefined;
  binary_size?: number | undefined;
}

/** Output of ruview_registry_list. */
export interface RegistryListResult {
  fetched_at: number;
  ttl_seconds: number;
  stale: boolean;
  upstream_url: string;
  upstream_sha256: string;
  cogs: CogEntry[];
}

// ── Training ──────────────────────────────────────────────────────────────

/** Output of ruview_train_count — a job handle. */
export interface TrainJobResult {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  /** Absolute path to the job log file (~/.ruview/jobs/<id>.log). */
  log_path: string;
  /** Timestamp when the job was enqueued (seconds since epoch). */
  queued_at: number;
}

/** Output of ruview_job_status. */
export interface JobStatusResult {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  progress_pct?: number | undefined;
  /** Most recent log lines (last 20). */
  recent_log: string[];
  log_path: string;
  /** Epoch count completed, if training. */
  epochs_done?: number | undefined;
  /** Total epochs scheduled. */
  epochs_total?: number | undefined;
}

// ── Vitals (ADR-124 §6 Python surface parity: ws.py:74-88) ───────────────

/**
 * Mirrors python/wifi_densepose/client/ws.py EdgeVitalsMessage (ws.py:74-88).
 * Returned by sensing-server GET /api/v1/vitals/<node_id>/latest.
 */
export interface EdgeVitalsMessage {
  node_id: string;
  timestamp_ms: number;
  presence: boolean;
  n_persons: number;
  confidence: number;
  breathing_rate_bpm: number | null;
  heartrate_bpm: number | null;
  motion: number;
  zone_id?: string | undefined;
}

// ── Config ────────────────────────────────────────────────────────────────

/** Runtime configuration, typically sourced from env vars. */
export interface RuviewConfig {
  /** Base URL of the local sensing-server (default: http://localhost:3000). */
  sensingServerUrl: string;
  /** Bearer token for /api/v1/* endpoints. Set RUVIEW_API_TOKEN to enable. */
  apiToken: string | undefined;
  /** Absolute path to the cog-pose-estimation binary. */
  poseCogBinary: string;
  /** Absolute path to the cog-person-count binary. */
  countCogBinary: string;
  /** Directory for job logs (default: ~/.ruview/jobs/). */
  jobsDir: string;
}
