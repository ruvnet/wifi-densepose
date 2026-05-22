/**
 * MCP tool: ruview_count_infer
 *
 * Run a single-shot person-count inference against a CSI window.
 *
 * Uses the cog-person-count binary (ADR-103).  The output includes a
 * calibrated confidence score and a 95% prediction interval, matching the
 * Stoer-Wagner + confidence-weighted log-sum fusion design in ADR-103.
 *
 * M1 (this file): stubs the inference after verifying the cog binary is healthy.
 * M2 wires the real forward pass.
 */

import { z } from "zod";
import type { RuviewConfig, CountInferResult } from "../types.js";
import { cogInferStub } from "../cog.js";

export const countInferSchema = z.object({
  /**
   * Path to a CSI window JSON file.
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
    .describe("Path to cog-person-count binary. Default: RUVIEW_COUNT_COG_BINARY env var."),
  /**
   * Maximum number of persons to consider in the output distribution.
   * Capped at 7 per the count head's softmax over {0..7}.
   */
  max_persons: z
    .number()
    .int()
    .min(1)
    .max(7)
    .optional()
    .default(7)
    .describe("Upper bound on person count (1–7). Default: 7."),
});

export type CountInferInput = z.infer<typeof countInferSchema>;

export async function countInfer(
  input: CountInferInput,
  config: RuviewConfig
): Promise<object> {
  const binary = input.cog_binary ?? config.countCogBinary;

  const stubResult = await cogInferStub(binary, "count");

  if (!stubResult.ok) {
    return {
      ok: false,
      warn: true,
      error: stubResult.error,
      hint:
        "Set RUVIEW_COUNT_COG_BINARY to the path of the cog-person-count binary. " +
        "Install it from gs://cognitum-apps/cogs/<arch>/cog-person-count-<arch>. " +
        "See ADR-103 for installation instructions.",
    };
  }

  const ts = Date.now() / 1000;
  const result: CountInferResult = {
    ts,
    count: 0,
    confidence: 0,
    count_p95_low: 0,
    count_p95_high: 0,
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
