/**
 * MCP tool: ruview_pose_infer
 *
 * Run a single-shot pose estimation inference against a CSI window.
 *
 * M1 (this file): stubs the inference after verifying the cog binary is healthy.
 * M2 wires the real forward pass via the sensing-server CSI window + cog `run`.
 *
 * The 17 COCO keypoints in the output follow the standard COCO body ordering:
 *   0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
 *   5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
 *   9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
 *   13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
 */

import { z } from "zod";
import type { RuviewConfig, PoseInferResult } from "../types.js";
import { cogInferStub } from "../cog.js";

export const poseInferSchema = z.object({
  /**
   * Path to a CSI window JSON file (as produced by ruview_csi_latest or
   * examples/research-sota/r5_subcarrier_saliency.py).
   * Optional — when absent, uses the latest window from the sensing-server.
   */
  window_path: z
    .string()
    .optional()
    .describe("Path to a CSI window JSON file. Omit to use the live sensing-server."),
  /** Override the cog binary path for this call. */
  cog_binary: z
    .string()
    .optional()
    .describe("Path to cog-pose-estimation binary. Default: RUVIEW_POSE_COG_BINARY env var."),
});

export type PoseInferInput = z.infer<typeof poseInferSchema>;

export async function poseInfer(
  input: PoseInferInput,
  config: RuviewConfig
): Promise<object> {
  const binary = input.cog_binary ?? config.poseCogBinary;

  // M1: health-check the cog, return stub keypoints.
  // M2: replace stub with real CSI window + cog run session.
  const stubResult = await cogInferStub(binary, "pose");

  if (!stubResult.ok) {
    return {
      ok: false,
      warn: true,
      error: stubResult.error,
      hint:
        "Set RUVIEW_POSE_COG_BINARY to the path of the cog-pose-estimation binary. " +
        "Install it from gs://cognitum-apps/cogs/<arch>/cog-pose-estimation-<arch>. " +
        "See ADR-101 for installation instructions.",
    };
  }

  const ts = Date.now() / 1000;
  const result: PoseInferResult = {
    ts,
    n_persons: 0,
    persons: [],
    backend: stubResult.data.backend,
    latency_ms: stubResult.data.latency_ms,
  };

  return {
    ok: true,
    stub: stubResult.data.stub,
    note:
      "M1 stub — real inference wired in M2. " +
      "Cog health check passed; binary is reachable.",
    result,
  };
}
