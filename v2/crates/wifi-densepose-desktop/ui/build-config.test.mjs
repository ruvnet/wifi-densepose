import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const uiDir = dirname(fileURLToPath(import.meta.url));
const packageJson = JSON.parse(readFileSync(join(uiDir, "package.json"), "utf8"));
const tauriConfig = JSON.parse(
  readFileSync(join(uiDir, "..", "tauri.conf.json"), "utf8"),
);

function major(versionRange) {
  const match = versionRange.match(/\d+/);
  assert.ok(match, `Expected a semver range, got ${versionRange}`);
  return Number(match[0]);
}

test("React runtime and type packages use the same major version", () => {
  assert.equal(
    major(packageJson.dependencies.react),
    major(packageJson.dependencies["react-dom"]),
  );
  assert.equal(
    major(packageJson.devDependencies["@types/react"]),
    major(packageJson.devDependencies["@types/react-dom"]),
  );
  assert.equal(
    major(packageJson.dependencies.react),
    major(packageJson.devDependencies["@types/react"]),
  );
});

test("Tauri frontend hooks run from the UI directory", () => {
  assert.deepEqual(tauriConfig.build.beforeDevCommand, {
    cwd: "ui",
    script: "npm run dev",
  });
  assert.deepEqual(tauriConfig.build.beforeBuildCommand, {
    cwd: "ui",
    script: "npm run build",
  });
});

test("Vite client types cover CSS side-effect imports", () => {
  const declaration = readFileSync(join(uiDir, "src", "vite-env.d.ts"), "utf8");
  assert.match(declaration, /reference types=["']vite\/client["']/);
});
