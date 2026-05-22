/**
 * Tests for runtime schema validators (validate.ts).
 *
 * Pinned to sensing-server schema_version 2 (ADR-101).
 * These tests document the exact shapes we accept and reject so that
 * any schema drift from the sensing-server is caught immediately.
 */

import { validateCsiWindow, validateSensingLatestResponse } from "../src/validate.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAmplitudes(rows = 56, cols = 20): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
}

function makeValidWindow(): unknown {
  return {
    ts: 1716300000.0,
    n_paths: 3,
    amplitudes: makeAmplitudes(),
  };
}

function makeValidResponse(): unknown {
  return {
    schema_version: 2,
    captured_at: "2026-05-21T20:00:00.000Z",
    window: makeValidWindow(),
  };
}

// ---------------------------------------------------------------------------
// validateCsiWindow
// ---------------------------------------------------------------------------

describe("validateCsiWindow", () => {
  it("accepts a valid 56×20 window", () => {
    const result = validateCsiWindow(makeValidWindow());
    expect(result.valid).toBe(true);
  });

  it("rejects null", () => {
    const result = validateCsiWindow(null);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors).toContain("window is not an object");
    }
  });

  it("rejects wrong subcarrier count (e.g. 57)", () => {
    const w = makeValidWindow() as Record<string, unknown>;
    w["amplitudes"] = makeAmplitudes(57, 20);
    const result = validateCsiWindow(w);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("56 rows"))).toBe(true);
    }
  });

  it("rejects wrong frame count (e.g. 10 instead of 20)", () => {
    const w = makeValidWindow() as Record<string, unknown>;
    w["amplitudes"] = makeAmplitudes(56, 10);
    const result = validateCsiWindow(w);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("20 frames"))).toBe(true);
    }
  });

  it("rejects missing ts field", () => {
    const w = makeValidWindow() as Record<string, unknown>;
    delete w["ts"];
    const result = validateCsiWindow(w);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("ts"))).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// validateSensingLatestResponse
// ---------------------------------------------------------------------------

describe("validateSensingLatestResponse", () => {
  it("accepts a valid schema_version 2 response", () => {
    const result = validateSensingLatestResponse(makeValidResponse());
    expect(result.valid).toBe(true);
  });

  it("rejects schema_version 3 (not yet supported)", () => {
    const d = makeValidResponse() as Record<string, unknown>;
    d["schema_version"] = 3;
    const result = validateSensingLatestResponse(d);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("schema_version 3 is not supported"))).toBe(true);
    }
  });

  it("rejects missing captured_at", () => {
    const d = makeValidResponse() as Record<string, unknown>;
    delete d["captured_at"];
    const result = validateSensingLatestResponse(d);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("captured_at"))).toBe(true);
    }
  });

  it("rejects null response", () => {
    const result = validateSensingLatestResponse(null);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.includes("not an object"))).toBe(true);
    }
  });

  it("propagates window validation errors with 'window:' prefix", () => {
    const d = makeValidResponse() as Record<string, unknown>;
    const w = (d["window"] as Record<string, unknown>);
    w["amplitudes"] = makeAmplitudes(57, 20);
    const result = validateSensingLatestResponse(d);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors.some((e) => e.startsWith("window:"))).toBe(true);
    }
  });
});
