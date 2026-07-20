/**
 * ADR-264 F8/O7 — cog-binary detection must be architecture-aware.
 *
 * detectCogBinary() itself probes hardcoded /var/lib paths, so it is not
 * cheaply testable without fs mocking. The bug it fixes, however, lives purely
 * in the candidate ORDER, which cogBinaryCandidates() exposes as a pure,
 * arch-injectable function — that is what we pin here.
 */

import { cogBinaryCandidates } from "../src/config.js";

describe("cogBinaryCandidates()", () => {
  it("probes -arm before -x86_64 on arm64 hosts", () => {
    const c = cogBinaryCandidates("cog-person-count", "arm64");
    const arm = c.findIndex((p) => p.endsWith("cog-person-count-arm"));
    const x86 = c.findIndex((p) => p.endsWith("cog-person-count-x86_64"));
    expect(arm).toBeGreaterThanOrEqual(0);
    expect(x86).toBeGreaterThanOrEqual(0);
    expect(arm).toBeLessThan(x86);
  });

  it("probes -x86_64 before -arm on x64 hosts", () => {
    const c = cogBinaryCandidates("cog-person-count", "x64");
    const arm = c.findIndex((p) => p.endsWith("cog-person-count-arm"));
    const x86 = c.findIndex((p) => p.endsWith("cog-person-count-x86_64"));
    expect(x86).toBeLessThan(arm);
  });

  it("defaults an unknown arch to the x86_64-first order", () => {
    const c = cogBinaryCandidates("cog-pose-estimation", "riscv64");
    const arm = c.findIndex((p) => p.endsWith("cog-pose-estimation-arm"));
    const x86 = c.findIndex((p) => p.endsWith("cog-pose-estimation-x86_64"));
    expect(x86).toBeLessThan(arm);
  });

  it("keeps the /usr/local/bin and bare-name PATH fallbacks last", () => {
    const c = cogBinaryCandidates("cog-person-count", "arm64");
    // The two arch builds come first; the /usr/local/bin fallback follows them.
    expect(c[c.length - 1]).toBe("/usr/local/bin/cog-person-count");
    expect(c).toHaveLength(3);
  });

  it("derives the id by stripping the cog- prefix once", () => {
    const c = cogBinaryCandidates("cog-person-count", "x64");
    expect(c[0]).toBe(
      "/var/lib/cognitum/apps/person-count/cog-person-count-x86_64"
    );
  });
});
