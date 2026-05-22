/**
 * Runtime schema validation for sensing-server responses.
 *
 * These validators catch schema drift (when the sensing-server's API
 * changes without updating the MCP layer) and provide actionable errors
 * to the calling agent rather than silently returning malformed data.
 *
 * The schema is pinned to sensing-server schema version 2 per ADR-101
 * frame_subscriber.rs. When the server bumps schema_version, a validation
 * error here is the correct signal to update the MCP types.
 */

export type ValidationResult =
  | { valid: true }
  | { valid: false; errors: string[] };

/**
 * Validate a CsiWindow conforms to the expected 56×20 shape.
 */
export function validateCsiWindow(window: unknown): ValidationResult {
  const errors: string[] = [];

  if (typeof window !== "object" || window === null) {
    return { valid: false, errors: ["window is not an object"] };
  }

  const w = window as Record<string, unknown>;

  if (typeof w["ts"] !== "number") {
    errors.push("window.ts must be a number");
  }

  if (typeof w["n_paths"] !== "number") {
    errors.push("window.n_paths must be a number");
  }

  const amplitudes = w["amplitudes"];
  if (!Array.isArray(amplitudes)) {
    errors.push("window.amplitudes must be an array");
  } else {
    if (amplitudes.length !== 56) {
      errors.push(
        `window.amplitudes must have 56 rows (subcarriers), got ${amplitudes.length}`
      );
    }
    for (let i = 0; i < Math.min(amplitudes.length, 3); i++) {
      if (!Array.isArray(amplitudes[i])) {
        errors.push(`window.amplitudes[${i}] must be an array`);
      } else if ((amplitudes[i] as unknown[]).length !== 20) {
        errors.push(
          `window.amplitudes[${i}] must have 20 frames, got ${(amplitudes[i] as unknown[]).length}`
        );
      }
    }
  }

  return errors.length === 0 ? { valid: true } : { valid: false, errors };
}

/**
 * Validate a full SensingLatestResponse (schema_version 2, ADR-101).
 */
export function validateSensingLatestResponse(data: unknown): ValidationResult {
  const errors: string[] = [];

  if (typeof data !== "object" || data === null) {
    return { valid: false, errors: ["response is not an object"] };
  }

  const d = data as Record<string, unknown>;

  const schemaVersion = d["schema_version"];
  if (typeof schemaVersion !== "number") {
    errors.push("schema_version must be a number");
  } else if (schemaVersion !== 2) {
    errors.push(
      `schema_version ${schemaVersion} is not supported. ` +
        "This MCP server is pinned to schema_version 2 (ADR-101). " +
        "Update tools/ruview-mcp/src/types.ts to support the new schema."
    );
  }

  if (typeof d["captured_at"] !== "string") {
    errors.push("captured_at must be a string (ISO-8601)");
  }

  const windowResult = validateCsiWindow(d["window"]);
  if (!windowResult.valid) {
    errors.push(...windowResult.errors.map((e) => `window: ${e}`));
  }

  return errors.length === 0 ? { valid: true } : { valid: false, errors };
}
