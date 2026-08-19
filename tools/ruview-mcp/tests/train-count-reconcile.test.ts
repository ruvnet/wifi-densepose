/**
 * ADR-264 O6 — post-restart job reconciliation.
 *
 * When the MCP server restarts mid-run, the persisted job record stays frozen
 * at 'running' (the child.on('close') that flips it lived in the dead process).
 * ruview_job_status must reconcile such a record against the recorded pid and
 * the log's "# exit code: N" marker.
 *
 * We fabricate a persisted record pointing at a KNOWN-DEAD pid (a synchronous
 * child that has already exited) and assert the reconciled status.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { jobStatus } from "../src/tools/train-count.js";
import type { RuviewConfig } from "../src/types.js";

/** A pid that has certainly exited: spawnSync waits for the child to finish. */
function deadPid(): number {
  const r = spawnSync(process.execPath, ["-e", ""]);
  if (typeof r.pid !== "number") throw new Error("could not spawn probe child");
  return r.pid;
}

function makeConfig(jobsDir: string): RuviewConfig {
  return {
    sensingServerUrl: "http://127.0.0.1:19999",
    apiToken: undefined,
    poseCogBinary: "nonexistent",
    countCogBinary: "nonexistent",
    jobsDir,
  };
}

/** Write a fake persisted 'running' record + its log, return {jobId, config}. */
function seedRunningJob(logBody: string): { jobId: string; config: RuviewConfig } {
  const jobsDir = mkdtempSync(path.join(os.tmpdir(), "rvagent-jobs-"));
  const jobId = randomUUID();
  const logPath = path.join(jobsDir, `${jobId}.log`);
  writeFileSync(logPath, logBody);
  const record = {
    job_id: jobId,
    status: "running",
    log_path: logPath,
    queued_at: Date.now() / 1000,
    epochs_total: 5,
    pid: deadPid(),
  };
  writeFileSync(
    path.join(jobsDir, `${jobId}.json`),
    JSON.stringify(record, null, 2)
  );
  return { jobId, config: makeConfig(jobsDir) };
}

describe("ruview_job_status reconciliation (ADR-264 O6)", () => {
  it("reconciles a dead 'running' job with exit 0 to 'done'", async () => {
    const { jobId, config } = seedRunningJob(
      "# training...\nepoch 5/5\n# exit code: 0\n"
    );
    const out = (await jobStatus({ job_id: jobId }, config)) as Record<string, unknown>;
    expect(out["ok"]).toBe(true);
    const res = out["result"] as Record<string, unknown>;
    expect(res["status"]).toBe("done");
  });

  it("reconciles a dead 'running' job with non-zero exit to 'failed'", async () => {
    const { jobId, config } = seedRunningJob(
      "# training...\npanic: cuda oom\n# exit code: 101\n"
    );
    const out = (await jobStatus({ job_id: jobId }, config)) as Record<string, unknown>;
    const res = out["result"] as Record<string, unknown>;
    expect(res["status"]).toBe("failed");
  });

  it("marks a dead 'running' job with no exit marker as 'unknown' with a reason", async () => {
    const { jobId, config } = seedRunningJob("# training...\nepoch 2/5\n");
    const out = (await jobStatus({ job_id: jobId }, config)) as Record<string, unknown>;
    const res = out["result"] as Record<string, unknown>;
    expect(res["status"]).toBe("unknown");
    expect(typeof res["reason"]).toBe("string");
    expect(res["reason"]).toMatch(/restarted/i);
  });

  it("treats a signal-killed marker (null) as 'failed'", async () => {
    const { jobId, config } = seedRunningJob(
      "# training...\n# exit code: null\n"
    );
    const out = (await jobStatus({ job_id: jobId }, config)) as Record<string, unknown>;
    const res = out["result"] as Record<string, unknown>;
    expect(res["status"]).toBe("failed");
  });
});
