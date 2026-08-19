import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const directory = path.dirname(fileURLToPath(import.meta.url));
const run = (script, fixture) =>
  spawnSync(process.execPath, [path.join(directory, script), path.join(directory, "testdata", fixture)], {
    cwd: path.resolve(directory, "../.."),
    encoding: "utf8",
  });

const strict = run("verify-splits.mjs", "strict-split-manifest.json");
if (strict.status !== 0 || !strict.stdout.includes('"verdict":"PASS"')) {
  throw new Error(`strict split fixture failed: ${strict.stderr || strict.stdout}`);
}

const leaky = run("verify-splits.mjs", "leaky-split-manifest.json");
if (leaky.status === 0 || !leaky.stderr.includes("leakage:")) {
  throw new Error("leaky split fixture was not rejected with a typed leakage error");
}

const golden = run("verify-golden.mjs", "golden-results.jsonl");
if (golden.status !== 0 || !golden.stdout.includes('"verdict":"PASS"')) {
  throw new Error(`golden replay fixture failed: ${golden.stderr || golden.stdout}`);
}

process.stdout.write(JSON.stringify({ verdict: "PASS", checks: 3 }) + "\n");
