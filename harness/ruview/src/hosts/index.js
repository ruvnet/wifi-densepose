// SPDX-License-Identifier: MIT
import claudeCode from './claude-code.js';
import codex from './codex.js';
export { claudeCode, codex };
export const HOSTS = Object.freeze({ 'claude-code': claudeCode, codex });
export function getHost(name) {
  const host = HOSTS[name];
  if (!host) throw new Error(`Unsupported host: ${name}`);
  return host;
}
