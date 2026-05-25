/**
 * ADR-124 §2 manifest validation test.
 *
 * Guards that package.json satisfies every structural decision from ADR-124 §2:
 *   1. Package name is @ruvnet/rvagent
 *   2. Version is >= 0.1.0
 *   3. engines.node is >= 20
 *   4. bin includes the "rvagent" key (npx @ruvnet/rvagent invocation)
 *   5. exports["." ] includes both "import" and "types" keys (ESM + types in tarball)
 *   6. publishConfig.access === "public" (scoped package must be explicit)
 *   7. @modelcontextprotocol/sdk is a runtime dependency (dual-transport server)
 *   8. zod is a runtime dependency (input schema validation)
 *   9. type === "module" (ESM-first, Node.js 20+ native)
 *  10. license === "Apache-2.0"
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgPath = resolve(__dirname, "../package.json");

// Parse once; keep raw for snapshot assertions.
const raw = readFileSync(pkgPath, "utf-8");
const pkg = JSON.parse(raw) as Record<string, unknown>;

// Helper to assert string field value.
function assertField(field: string, expected: string): void {
  expect(pkg[field]).toBe(expected);
}

// Helper to get a nested value.
function nested<T>(obj: Record<string, unknown>, ...keys: string[]): T {
  let cur: unknown = obj;
  for (const k of keys) {
    if (typeof cur !== "object" || cur === null) {
      throw new Error(`Expected object at key "${k}"`);
    }
    cur = (cur as Record<string, unknown>)[k];
  }
  return cur as T;
}

describe("@ruvnet/rvagent package.json (ADR-124 §2)", () => {
  it("§2.1 — name is @ruvnet/rvagent", () => {
    assertField("name", "@ruvnet/rvagent");
  });

  it("§2.2 — version is semver >= 0.1.0", () => {
    const version = pkg["version"] as string;
    expect(typeof version).toBe("string");
    const [major, minor] = version.split(".").map(Number);
    const isAtLeast010 = (major ?? 0) > 0 || (minor ?? 0) >= 1;
    expect(isAtLeast010).toBe(true);
  });

  it("§2.3 — engines.node requires Node.js >= 20", () => {
    const nodeRange = nested<string>(pkg, "engines", "node");
    expect(typeof nodeRange).toBe("string");
    // Accept >=20 or >=20.0.0 patterns.
    expect(nodeRange).toMatch(/>=\s*20/);
  });

  it("§2.4 — bin.rvagent is defined (npx @ruvnet/rvagent invocation)", () => {
    const bin = nested<Record<string, string>>(pkg, "bin");
    expect(typeof bin["rvagent"]).toBe("string");
    expect(bin["rvagent"]).toMatch(/dist\/index\.js/);
  });

  it("§2.5 — exports['.'] has import + types keys (ESM + TypeScript declarations)", () => {
    const exports = nested<Record<string, Record<string, string>>>(pkg, "exports");
    const dotExport = exports["."];
    expect(dotExport).toBeDefined();
    expect(typeof dotExport?.["import"]).toBe("string");
    expect(typeof dotExport?.["types"]).toBe("string");
  });

  it("§2.6 — publishConfig.access is 'public' (scoped package requirement)", () => {
    const access = nested<string>(pkg, "publishConfig", "access");
    expect(access).toBe("public");
  });

  it("§2.7 — @modelcontextprotocol/sdk is a runtime dependency", () => {
    const deps = nested<Record<string, string>>(pkg, "dependencies");
    expect(typeof deps["@modelcontextprotocol/sdk"]).toBe("string");
  });

  it("§2.8 — zod is a runtime dependency", () => {
    const deps = nested<Record<string, string>>(pkg, "dependencies");
    expect(typeof deps["zod"]).toBe("string");
  });

  it("§2.9 — type is 'module' (ESM-first, Node.js 20+ native)", () => {
    assertField("type", "module");
  });

  it("§2.10 — license is Apache-2.0", () => {
    assertField("license", "Apache-2.0");
  });
});
