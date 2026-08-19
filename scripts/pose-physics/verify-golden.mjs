import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const input = process.argv[2];
if (!input) throw new Error("usage: replay-pose-physics-golden.sh <golden-results.jsonl>");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const result = spawnSync(
  "cargo",
  ["run", "--quiet", "-p", "wifi-densepose-physics", "--features", "deterministic", "--example", "verify_golden", "--", path.resolve(input)],
  { cwd: path.join(root, "v2"), encoding: "utf8" },
);
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
