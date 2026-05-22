/**
 * MCP tool: ruview_train_count + ruview_job_status
 *
 * Kick off a cog-person-count training run and poll its status.
 *
 * The training pipeline used here is the Candle GPU trainer from
 * `v2/crates/wifi-densepose-train` — the same one that produced
 * `count_v1.safetensors` in 2.1 s on the RTX 5080 (ADR-103).
 *
 * The MCP server shells out to `cargo run -p wifi-densepose-train --` with the
 * paired JSONL path as input, redirecting stdout/stderr to a log file.  The
 * returned job_id can be used with ruview_job_status to poll progress.
 *
 * M1: job is enqueued (background process spawned, log file created).
 * M4: full training arguments + real output artifact path returned.
 */

import { z } from "zod";
import { randomUUID } from "node:crypto";
import { mkdirSync, appendFileSync, openSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import type { RuviewConfig, TrainJobResult, JobStatusResult } from "../types.js";

export const trainCountSchema = z.object({
  /**
   * Path to the paired JSONL file for training.
   * Produced by scripts/align-ground-truth.js.
   * E.g. data/paired/wiflow-p7-2026-05-19.paired.jsonl
   */
  paired_jsonl: z
    .string()
    .describe("Absolute or relative path to the paired JSONL training file."),
  /** Number of training epochs (default: 400, matching ADR-103 recipe). */
  epochs: z
    .number()
    .int()
    .min(1)
    .max(10_000)
    .optional()
    .default(400)
    .describe("Training epochs (default: 400)."),
  /**
   * Learning rate.  The ADR-103 recipe uses 1e-3 with frozen encoder for the
   * first 50 epochs, then 1e-4 for joint fine-tuning.
   */
  learning_rate: z
    .number()
    .optional()
    .default(1e-3)
    .describe("Initial learning rate (default: 0.001)."),
  /** Directory where the trained model artifacts are written. */
  output_dir: z
    .string()
    .optional()
    .describe(
      "Directory for model artifacts (default: v2/crates/cog-person-count/cog/artifacts/)."
    ),
});

export type TrainCountInput = z.infer<typeof trainCountSchema>;

export const jobStatusSchema = z.object({
  job_id: z.string().uuid().describe("Job ID returned by ruview_train_count."),
});

export type JobStatusInput = z.infer<typeof jobStatusSchema>;

// In-process job registry (survives for the lifetime of the MCP server process).
// For a production implementation, persist to ~/.ruview/jobs/<id>.json.
const jobRegistry = new Map<
  string,
  {
    status: "queued" | "running" | "done" | "failed";
    log_path: string;
    queued_at: number;
    epochs_total: number;
  }
>();

export async function trainCount(
  input: TrainCountInput,
  config: RuviewConfig
): Promise<object> {
  const jobId = randomUUID();
  const logDir = config.jobsDir;
  mkdirSync(logDir, { recursive: true });
  const logPath = path.join(logDir, `${jobId}.log`);
  const queuedAt = Date.now() / 1000;

  // Default output directory matches ADR-103 repo layout.
  const outputDir =
    input.output_dir ?? "v2/crates/cog-person-count/cog/artifacts";

  // Record the job immediately so ruview_job_status can find it.
  jobRegistry.set(jobId, {
    status: "queued",
    log_path: logPath,
    queued_at: queuedAt,
    epochs_total: input.epochs,
  });

  // Write the header synchronously so the log file exists before spawn.
  const header = [
    `# RuView training job ${jobId}`,
    `# started: ${new Date().toISOString()}`,
    `# paired_jsonl: ${input.paired_jsonl}`,
    `# epochs: ${input.epochs}`,
    `# learning_rate: ${input.learning_rate}`,
    `# output_dir: ${outputDir}`,
    "",
  ].join("\n");
  appendFileSync(logPath, header);

  // Open log file descriptors synchronously (avoids WriteStream-before-open bug on Windows).
  const logFdOut = openSync(logPath, "a");
  const logFdErr = openSync(logPath, "a");

  const args = [
    "run",
    "--release",
    "-p",
    "wifi-densepose-train",
    "--",
    "--task",
    "count",
    "--paired",
    input.paired_jsonl,
    "--epochs",
    String(input.epochs),
    "--lr",
    String(input.learning_rate),
    "--output-dir",
    outputDir,
  ];

  // M1: cargo may not be on PATH on non-Rust machines — spawn fails gracefully.
  const child = spawn("cargo", args, {
    detached: true,
    stdio: ["ignore", logFdOut, logFdErr],
  });

  child.unref(); // Allow the MCP server process to exit without waiting for training.

  const entry = jobRegistry.get(jobId);
  if (entry) {
    entry.status = "running";
  }

  child.on("error", (e) => {
    appendFileSync(logPath, `\n# ERROR: ${e.message}\n`);
    const rec = jobRegistry.get(jobId);
    if (rec) rec.status = "failed";
  });

  child.on("close", (code) => {
    appendFileSync(logPath, `\n# exit code: ${code}\n`);
    const rec = jobRegistry.get(jobId);
    if (rec) rec.status = code === 0 ? "done" : "failed";
  });

  const result: TrainJobResult = {
    job_id: jobId,
    status: "running",
    log_path: logPath,
    queued_at: queuedAt,
  };

  return {
    ok: true,
    result,
    note:
      "Training job spawned in the background. " +
      `Poll progress with ruview_job_status({ job_id: "${jobId}" }). ` +
      `Live log: ${logPath}`,
  };
}

export async function jobStatus(
  input: JobStatusInput,
  _config: RuviewConfig
): Promise<object> {
  const job = jobRegistry.get(input.job_id);
  if (!job) {
    return {
      ok: false,
      error: `Job ${input.job_id} not found. ` +
        "The MCP server may have restarted — check the log directory directly.",
    };
  }

  // Read the last 20 lines of the log file.
  let recentLog: string[] = [];
  try {
    const { readFileSync } = await import("node:fs");
    const content = readFileSync(job.log_path, "utf8");
    const lines = content.split("\n");
    recentLog = lines.slice(Math.max(0, lines.length - 20));
  } catch {
    recentLog = ["(log not readable yet)"];
  }

  const result: JobStatusResult = {
    job_id: input.job_id,
    status: job.status,
    log_path: job.log_path,
    recent_log: recentLog,
    epochs_total: job.epochs_total,
  };

  return { ok: true, result };
}
