// SPDX-License-Identifier: MIT
const SECRET_KEY_RE = /(?:api[_-]?key|token|secret|password|passwd|authorization|cookie|private[_-]?key)/i;
const INLINE_VALUE_RE = /\b([A-Za-z][A-Za-z0-9_.-]*)(\s*[:=]\s*)(["']?)([^\s"',;}\]]+)\3/g;
const AUTH_RE = /\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi;
const TOKEN_RES = [
  /\b(?:sk|sk-ant|sk-proj)-[A-Za-z0-9_-]{16,}\b/g,
  /\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b/g,
  /\bAKIA[0-9A-Z]{16}\b/g,
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
];
export const REDACTED = '[REDACTED]';
function knownSecrets(env) {
  return Object.entries(env ?? {}).filter(([key, value]) => SECRET_KEY_RE.test(key) && typeof value === 'string' && value.length >= 6)
    .map(([, value]) => value).sort((a, b) => b.length - a.length);
}
export function redact(value, { env = process.env } = {}) {
  let text = String(value ?? '');
  for (const secret of knownSecrets(env)) text = text.split(secret).join(REDACTED);
  text = text.replace(AUTH_RE, `$1 ${REDACTED}`);
  text = text.replace(INLINE_VALUE_RE, (match, key, separator, quote) => (
    SECRET_KEY_RE.test(key) ? `${key}${separator}${quote}${REDACTED}${quote}` : match
  ));
  for (const pattern of TOKEN_RES) text = text.replace(pattern, REDACTED);
  return text;
}
