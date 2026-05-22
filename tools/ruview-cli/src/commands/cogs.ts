/**
 * ruview cogs — Cognitum edge module registry commands.
 *
 * cogs list  — list cogs from the registry (via sensing-server ADR-102 proxy).
 */

import type { Argv } from "yargs";
import { sensingGet } from "../http.js";
import { loadConfig } from "../config.js";

export function cogsCommand(cli: Argv): void {
  cli.command(
    "cogs <action>",
    "Edge module registry commands",
    (y) =>
      y
        .positional("action", {
          choices: ["list"] as const,
          description: "Action to perform",
        })
        .option("category", {
          type: "string",
          description:
            "Filter by category: health, security, building, retail, industrial, " +
            "research, ai, swarm, signal, network, developer",
        })
        .option("search", {
          type: "string",
          description: "Search substring matched against cog id and name (case-insensitive)",
        })
        .option("refresh", {
          type: "boolean",
          default: false,
          description: "Bypass the 1-hour registry cache",
        })
        .option("url", {
          type: "string",
          description: "Override the sensing-server URL",
        }),
    async (args) => {
      const config = loadConfig();
      const baseUrl = (args["url"] as string | undefined) ?? config.sensingServerUrl;

      if (args.action === "list") {
        const qs = args.refresh ? "?refresh=1" : "";
        const result = await sensingGet<{
          registry?: { cogs?: object[]; apps?: object[] };
        }>(baseUrl, `/api/v1/edge/registry${qs}`, config.apiToken);

        if (!result.ok) {
          process.stderr.write(`[WARN] ${result.error}\n`);
          process.stdout.write(
            JSON.stringify({ ok: false, warn: true, error: result.error }) + "\n"
          );
          process.exit(0);
        }

        const payload = result.data;
        let cogs: object[] =
          payload.registry?.cogs ?? payload.registry?.apps ?? [];

        if (args.category) {
          const cat = (args.category as string).toLowerCase();
          cogs = cogs.filter(
            (c) =>
              (c as Record<string, unknown>)["category"]
                ?.toString()
                .toLowerCase() === cat
          );
        }
        if (args.search) {
          const q = (args.search as string).toLowerCase();
          cogs = cogs.filter((c) => {
            const rec = c as Record<string, unknown>;
            return (
              rec["id"]?.toString().toLowerCase().includes(q) ||
              rec["name"]?.toString().toLowerCase().includes(q)
            );
          });
        }

        process.stdout.write(
          JSON.stringify({ ok: true, total: cogs.length, cogs }, null, 2) + "\n"
        );
      }
    }
  );
}
