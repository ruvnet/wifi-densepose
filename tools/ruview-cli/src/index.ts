#!/usr/bin/env node
/**
 * @ruv/ruview-cli — RuView CLI
 *
 * Shell access to RuView sensing, inference, and training capabilities.
 *
 * Subcommands:
 *   ruview csi tail [--url <url>]                    stream live CSI frames
 *   ruview pose infer [--window <path>]              17-keypoint pose estimation
 *   ruview count infer [--window <path>]             person-count inference
 *   ruview cogs list [--category <cat>] [--search q] list edge module registry
 *   ruview train count --paired <jsonl>              kick off count-cog training
 *   ruview job status --id <job_id>                  poll a training job
 *
 * All subcommands write JSON to stdout and exit 0 on success.
 * WARN-level outputs write to stderr; the exit code is still 0 so pipelines
 * are not broken by a temporarily unreachable sensing-server.
 *
 * Usage:
 *   npx ruview --version
 *   npx ruview csi tail
 *   npx ruview pose infer --window ./window.json
 *   RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 npx ruview cogs list
 *
 * See ADR-104 for the full design rationale and security model.
 */

import { createRequire } from "node:module";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { csiCommand } from "./commands/csi.js";
import { poseCommand } from "./commands/pose.js";
import { countCommand } from "./commands/count.js";
import { cogsCommand } from "./commands/cogs.js";
import { trainCommand } from "./commands/train.js";
import { jobCommand } from "./commands/job.js";

// Single-source the version from package.json (ADR-265 D3).
const require = createRequire(import.meta.url);
const VERSION: string = (require("../package.json") as { version: string }).version;

// Bin name is `ruview-cli`: the bare `ruview` bin belongs to @ruvnet/ruview
// (ADR-264 O9 / ADR-265 D4).
const cli = yargs(hideBin(process.argv))
  .scriptName("ruview-cli")
  .version(VERSION)
  .usage("$0 <command> [options]")
  .strict()
  .help()
  .wrap(100);

// Register all top-level commands.
csiCommand(cli);
poseCommand(cli);
countCommand(cli);
cogsCommand(cli);
trainCommand(cli);
jobCommand(cli);

cli.demandCommand(1, "Specify a subcommand. Use --help for a list.").parse();
