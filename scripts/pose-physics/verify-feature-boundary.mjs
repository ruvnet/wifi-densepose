import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const result = spawnSync(
  "cargo",
  ["tree", "-p", "wifi-densepose-physics", "-e", "normal", "--prefix", "none"],
  { cwd: path.join(root, "v2"), encoding: "utf8" },
);

if (result.error) throw result.error;
if (result.status !== 0) {
  if (result.stdout) process.stderr.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  process.exit(result.status ?? 1);
}

const forbidden = /^(?:burn(?:-|\s|$)|rapier3d(?:\s|$)|tch(?:\s|$)|ort(?:\s|$))/m;
if (forbidden.test(result.stdout)) {
  process.stderr.write(result.stdout);
  throw new Error("default physics dependency graph contains an optional heavy backend");
}

process.stdout.write(JSON.stringify({ verdict: "PASS", profile: "default-kinematic" }) + "\n");
