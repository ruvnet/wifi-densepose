/**
 * ruview csi — CSI frame commands.
 *
 * csi tail  — stream live CSI frames from the sensing-server.
 */

import type { Argv } from "yargs";
import { sensingGet } from "../http.js";
import { loadConfig } from "../config.js";

export function csiCommand(cli: Argv): void {
  cli.command(
    "csi <action>",
    "CSI frame commands",
    (y) =>
      y
        .positional("action", {
          choices: ["tail"] as const,
          description: "Action to perform",
        })
        .option("url", {
          type: "string",
          description:
            "Sensing-server URL (default: RUVIEW_SENSING_SERVER_URL or http://localhost:3000)",
        })
        .option("interval", {
          type: "number",
          default: 500,
          description: "Polling interval in milliseconds (default: 500)",
        }),
    async (args) => {
      const config = loadConfig();
      const baseUrl = (args["url"] as string | undefined) ?? config.sensingServerUrl;

      if (args.action === "tail") {
        process.stderr.write(
          `[ruview csi tail] Streaming from ${baseUrl} every ${args.interval}ms. Ctrl-C to stop.\n`
        );

        // Streaming poll loop.
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const result = await sensingGet<object>(
            baseUrl,
            "/api/v1/sensing/latest",
            config.apiToken
          );

          if (!result.ok) {
            process.stderr.write(
              `[WARN] ${result.error} — retrying in ${args.interval}ms\n`
            );
          } else {
            process.stdout.write(JSON.stringify(result.data) + "\n");
          }

          await new Promise<void>((resolve) =>
            setTimeout(resolve, args.interval as number)
          );
        }
      }
    }
  );
}
