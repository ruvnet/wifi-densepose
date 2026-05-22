/**
 * ruview job — Job management commands.
 *
 * job status --id <job_id>  — poll a background training job.
 */

import type { Argv } from "yargs";
import { readFileSync, existsSync } from "node:fs";
import { loadConfig } from "../config.js";

export function jobCommand(cli: Argv): void {
  cli.command(
    "job <action>",
    "Job management commands",
    (y) =>
      y
        .positional("action", {
          choices: ["status"] as const,
          description: "Action to perform",
        })
        .option("id", {
          type: "string",
          demandOption: true,
          description: "Job ID returned by ruview train count",
        }),
    async (args) => {
      const config = loadConfig();

      if (args.action === "status") {
        const jobId = args.id as string;
        const { default: path } = await import("node:path");
        const logPath = path.join(config.jobsDir, `${jobId}.log`);

        if (!existsSync(logPath)) {
          process.stdout.write(
            JSON.stringify({
              ok: false,
              error: `Job ${jobId} not found at ${logPath}. ` +
                "The CLI process that started the job may have been restarted.",
            }) + "\n"
          );
          process.exit(0);
        }

        const content = readFileSync(logPath, "utf8");
        const lines = content.split("\n");
        const recentLog = lines.slice(Math.max(0, lines.length - 20));

        // Derive status from the log content.
        let status: string = "running";
        if (content.includes("# exit code: 0")) {
          status = "done";
        } else if (content.includes("# exit code:") || content.includes("# ERROR:")) {
          status = "failed";
        }

        process.stdout.write(
          JSON.stringify(
            {
              ok: true,
              job_id: jobId,
              status,
              log_path: logPath,
              recent_log: recentLog,
            },
            null,
            2
          ) + "\n"
        );
      }
    }
  );
}
