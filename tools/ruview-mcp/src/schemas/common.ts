/**
 * Shared Zod sub-schemas reused across the ADR-124 §4.1 tool catalog.
 *
 * All constraints are sourced from the ADR-124 decision record; comments cite
 * the specific table row or section that defines the constraint.
 */

import { z } from "zod";

// ── Shared primitives ──────────────────────────────────────────────────────

/**
 * Optional node_id — present on almost every tool. Defaults to the single
 * active node when only one is registered; required for multi-node fleets.
 */
export const NodeIdSchema = z
  .string()
  .min(1)
  .optional()
  .describe("Target node id. Omit to use the single active node.");

/**
 * Subscription duration in seconds. ADR-124 policy layer caps this at the
 * value returned by ruview.policy.can_subscribe.max_duration_s; the schema
 * enforces a hard ceiling of 3600 s (1 h) as a first-line guard.
 */
export const DurationSSchema = z
  .number()
  .positive()
  .max(3600)
  .describe("Subscription duration in seconds (max 3600).");

/**
 * Optional window in seconds for vitals averaging. Positive, max 300 s.
 * ADR-124 §4.1 rows vitals.get_breathing / vitals.get_heart_rate.
 */
export const WindowSSchema = z
  .number()
  .positive()
  .max(300)
  .optional()
  .describe("Averaging window in seconds (max 300).");

/**
 * The 10 semantic primitive kinds defined in ADR-115 and mirrored in
 * python/wifi_densepose/client/primitives.py:36-45.
 */
export const SemanticPrimitiveKindSchema = z.enum([
  "presence",
  "n_persons",
  "fall_detected",
  "breathing_rate",
  "heart_rate",
  "gesture",
  "zone_entry",
  "zone_exit",
  "movement_intensity",
  "sleep_quality",
]);

export type SemanticPrimitiveKind = z.infer<typeof SemanticPrimitiveKindSchema>;

/**
 * A single 17-keypoint COCO pose result as stored and returned by the
 * ruvector HNSW index (ADR-016). Used by ruview.vector.store_pose input.
 */
export const PosePersonResultSchema = z.object({
  keypoints: z
    .array(z.tuple([z.number(), z.number()]))
    .length(17)
    .describe("17 COCO keypoints as [x,y] pairs in image-normalised coords."),
  confidence: z.number().min(0).max(1).describe("Pose confidence score [0,1]."),
  person_id: z
    .string()
    .optional()
    .describe("AETHER re-ID token, if available."),
});

export type PosePersonResult = z.infer<typeof PosePersonResultSchema>;
