// SPDX-License-Identifier: MIT
// Default-deny MCP authority policy.

export const TOOL_POLICY = Object.freeze({
  homecore_guidance: { class: 'read', readOnly: true },
  homecore_wasm_status: { class: 'read', readOnly: true },
  homecore_doctor: { class: 'read', readOnly: true },
  homecore_memory_search: { class: 'read', readOnly: true },
  homecore_verify: {
    class: 'execute',
    readOnly: false,
    writesBuildArtifacts: true,
    mcpExposed: false,
  },
});

function typeMatches(value, type) {
  if (type === 'array') return Array.isArray(value);
  if (type === 'object') return value !== null && typeof value === 'object' && !Array.isArray(value);
  if (type === 'number') return typeof value === 'number' && Number.isFinite(value);
  if (type === 'integer') return Number.isSafeInteger(value);
  return typeof value === type;
}

export function validateArguments(schema, value, path = '$') {
  const errors = [];
  const type = schema.type || 'object';
  if (!typeMatches(value, type)) return [`${path} must be ${type}`];

  if (type === 'object') {
    const properties = schema.properties || {};
    for (const key of schema.required || []) {
      if (!(key in value)) errors.push(`${path}.${key} is required`);
    }
    for (const [key, item] of Object.entries(value)) {
      if (!Object.hasOwn(properties, key)) {
        if (schema.additionalProperties !== true) errors.push(`${path}.${key} is not allowed`);
        continue;
      }
      errors.push(...validateArguments(properties[key], item, `${path}.${key}`));
    }
  }

  if (type === 'array') {
    if (schema.maxItems !== undefined && value.length > schema.maxItems) {
      errors.push(`${path} exceeds maxItems`);
    }
    if (schema.items) {
      value.forEach((item, index) => {
        errors.push(...validateArguments(schema.items, item, `${path}[${index}]`));
      });
    }
  }

  if (schema.enum && !schema.enum.includes(value)) {
    errors.push(`${path} must be one of ${schema.enum.join(', ')}`);
  }
  if (type === 'string') {
    if (schema.minLength !== undefined && value.length < schema.minLength) {
      errors.push(`${path} is too short`);
    }
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      errors.push(`${path} is too long`);
    }
  }
  if (type === 'number' || type === 'integer') {
    if (schema.minimum !== undefined && value < schema.minimum) {
      errors.push(`${path} is below minimum`);
    }
    if (schema.maximum !== undefined && value > schema.maximum) {
      errors.push(`${path} exceeds maximum`);
    }
  }
  return errors;
}

export function authorizeTool(name, args, context = {}) {
  const policy = TOOL_POLICY[name] || { class: 'unknown', denied: true };
  if (policy.denied) return { ok: false, reason: 'policy_missing', policy };
  if (context.source === 'mcp' && policy.mcpExposed === false) {
    return { ok: false, reason: 'mcp_not_exposed', policy };
  }
  if (context.source !== 'mcp' || policy.readOnly) return { ok: true, policy };
  if (policy.confirmField && args?.[policy.confirmField] !== true) {
    return { ok: false, reason: 'not_confirmed', policy };
  }
  const grants = new Set(context.grants || []);
  if (!grants.has(policy.class)) {
    return {
      ok: false,
      reason: 'authority_denied',
      requiredGrant: policy.class,
      policy,
    };
  }
  return { ok: true, policy };
}

export function mcpAnnotations(name) {
  const policy = TOOL_POLICY[name] || {};
  return {
    readOnlyHint: policy.readOnly === true,
    destructiveHint: false,
    idempotentHint: policy.readOnly === true,
    openWorldHint: false,
  };
}
