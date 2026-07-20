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
import {
  mkdirSync,
  appendFileSync,
  openSync,
  closeSync,
  readFileSync,
  writeFileSync,
  statSync,
  readSync,
} from "node:fs";
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

interface JobRecord {
  status: "queued" | "running" | "done" | "failed" | "unknown";
  log_path: string;
  queued_at: number;
  epochs_total: number;
  /**
   * OS pid of the training child. Persisted so a later process (e.g. after an
   * MCP server restart) can tell whether a job still marked 'running' actually
   * outlived the process that spawned it (ADR-264 O6).
   */
  pid?: number | undefined;
  /** Human-readable explanation attached during reconciliation (unknown state). */
  reason?: string | undefined;
}

// In-process job registry, mirrored to <jobsDir>/<id>.json on every state
// change so ruview_job_status survives an MCP server restart (ADR-264 O6).
const jobRegistry = new Map<string, JobRecord>();

function jobRecordPath(jobsDir: string, jobId: string): string {
  return path.join(jobsDir, `${jobId}.json`);
}

function persistJob(jobsDir: string, jobId: string, record: JobRecord): void {
  try {
    writeFileSync(
      jobRecordPath(jobsDir, jobId),
      JSON.stringify({ job_id: jobId, ...record }, null, 2)
    );
  } catch {
    // Persistence is best-effort; the in-memory record still serves this process.
  }
}

function loadPersistedJob(jobsDir: string, jobId: string): JobRecord | undefined {
  try {
    const raw = JSON.parse(readFileSync(jobRecordPath(jobsDir, jobId), "utf8")) as
      Partial<JobRecord>;
    if (typeof raw.log_path !== "string" || typeof raw.status !== "string") {
      return undefined;
    }
    return {
      status: raw.status,
      log_path: raw.log_path,
      queued_at: typeof raw.queued_at === "number" ? raw.queued_at : 0,
      epochs_total: typeof raw.epochs_total === "number" ? raw.epochs_total : 0,
      pid: typeof raw.pid === "number" ? raw.pid : undefined,
      reason: typeof raw.reason === "string" ? raw.reason : undefined,
    };
  } catch {
    return undefined;
  }
}

/**
 * Is `pid` still a live process? `process.kill(pid, 0)` sends no signal but
 * probes existence: ESRCH ⇒ gone; EPERM ⇒ alive but owned by another user
 * (treated as alive so we never falsely reconcile a still-running job).
 */
function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
}

/**
 * Scan log lines (tail) for the "# exit code: N" marker the child.on('close')
 * handler appends. `found:false` means the process died without the marker —
 * i.e. this server never saw the close (it restarted mid-run).
 */
function findExitMarker(lines: string[]): { found: boolean; code: number | null } {
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = /^# exit code: (-?\d+|null)$/.exec((lines[i] ?? "").trim());
    if (m) return { found: true, code: m[1] === "null" ? null : Number(m[1]) };
  }
  return { found: false, code: null };
}

/** Read the last `maxLines` lines of a file without loading the whole log. */
function tailLines(filePath: string, maxLines: number, maxBytes = 64 * 1024): string[] {
  const size = statSync(filePath).size;
  const start = Math.max(0, size - maxBytes);
  const buf = Buffer.alloc(size - start);
  const fd = openSync(filePath, "r");
  try {
    readSync(fd, buf, 0, buf.length, start);
  } finally {
    closeSync(fd);
  }
  const lines = buf.toString("utf8").split("\n");
  return lines.slice(Math.max(0, lines.length - maxLines));
}

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

  // Record the job immediately so ruview_job_status can find it — in memory
  // and on disk (survives server restarts, ADR-264 O6).
  const record: JobRecord = {
    status: "queued",
    log_path: logPath,
    queued_at: queuedAt,
    epochs_total: input.epochs,
  };
  jobRegistry.set(jobId, record);
  persistJob(logDir, jobId, record);

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

  // The child holds its own duplicates of the log fds; close the parent's
  // copies immediately or every job leaks 2 fds for the server's lifetime
  // (ADR-264 F6/O6).
  closeSync(logFdOut);
  closeSync(logFdErr);

  // Record the child pid so a later process can reconcile a stale 'running'
  // record after a server restart (child.pid is undefined only if spawn failed
  // synchronously, in which case the 'error' handler flips status to 'failed').
  record.pid = child.pid;
  record.status = "running";
  persistJob(logDir, jobId, record);

  child.on("error", (e) => {
    appendFileSync(logPath, `\n# ERROR: ${e.message}\n`);
    record.status = "failed";
    persistJob(logDir, jobId, record);
  });

  child.on("close", (code) => {
    appendFileSync(logPath, `\n# exit code: ${code}\n`);
    record.status = code === 0 ? "done" : "failed";
    persistJob(logDir, jobId, record);
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
  config: RuviewConfig
): Promise<object> {
  // Memory first, then the persisted record (survives server restarts).
  let job = jobRegistry.get(input.job_id) ?? loadPersistedJob(config.jobsDir, input.job_id);
  if (!job) {
    return {
      ok: false,
      error: `Job ${input.job_id} not found in this server or in ${config.jobsDir}.`,
    };
  }

  // Reconcile a 'running' record whose owning process is gone. The status flip
  // to done/failed lives only in the spawning process's child.on('close'/'error')
  // handlers; if this server restarted mid-run, the record froze at 'running'
  // (ADR-264 O6). When the pid is dead, recover the true outcome from the log's
  // "# exit code: N" marker, else surface an honest 'unknown'.
  if (job.status === "running" && typeof job.pid === "number" && !isProcessAlive(job.pid)) {
    let tail: string[] = [];
    try {
      tail = tailLines(job.log_path, 40);
    } catch {
      /* log unreadable — treated as no marker below */
    }
    const marker = findExitMarker(tail);
    const reconciled: JobRecord = { ...job };
    if (marker.found) {
      reconciled.status = marker.code === 0 ? "done" : "failed";
      reconciled.reason = undefined;
    } else {
      reconciled.status = "unknown";
      reconciled.reason =
        "process gone, no exit marker — server likely restarted mid-run";
    }
    jobRegistry.set(input.job_id, reconciled);
    persistJob(config.jobsDir, input.job_id, reconciled);
    job = reconciled;
  }

  // Bounded tail read — never load a multi-GB training log wholesale.
  let recentLog: string[] = [];
  try {
    recentLog = tailLines(job.log_path, 20);
  } catch {
    recentLog = ["(log not readable yet)"];
  }

  const result: JobStatusResult = {
    job_id: input.job_id,
    status: job.status,
    log_path: job.log_path,
    recent_log: recentLog,
    epochs_total: job.epochs_total,
    ...(job.reason !== undefined ? { reason: job.reason } : {}),
  };

  return { ok: true, result };
}
