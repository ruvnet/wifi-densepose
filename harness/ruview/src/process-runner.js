// SPDX-License-Identifier: MIT
import { spawn } from 'node:child_process';
import { redact } from './redact.js';
export const DEFAULT_ENV_ALLOWLIST = Object.freeze([
  'PATH', 'Path', 'PATHEXT', 'SYSTEMROOT', 'SystemRoot', 'WINDIR', 'COMSPEC',
  'TEMP', 'TMP', 'TMPDIR', 'HOME', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA',
  'LANG', 'LC_ALL', 'TERM', 'NO_COLOR', 'FORCE_COLOR', 'CI',
]);
export function scrubEnvironment(source = process.env, allowlist = DEFAULT_ENV_ALLOWLIST) {
  const allowed = new Set(allowlist);
  return Object.fromEntries(Object.entries(source).filter(([key, value]) => allowed.has(key) && typeof value === 'string'));
}
export function runProcess(command, args = [], {
  cwd, input = '', timeoutMs = 120_000, signal, maxOutputBytes = 1_048_576,
  env = process.env, envAllowlist = DEFAULT_ENV_ALLOWLIST,
} = {}) {
  if (!command || typeof command !== 'string') throw new TypeError('command must be a non-empty string');
  if (!Array.isArray(args) || !args.every((arg) => typeof arg === 'string')) throw new TypeError('args must be an array of strings');
  if (!Number.isSafeInteger(maxOutputBytes) || maxOutputBytes < 1) throw new RangeError('maxOutputBytes must be a positive safe integer');
  const childEnv = scrubEnvironment(env, envAllowlist);
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env: childEnv, shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    const stdout = []; const stderr = [];
    let outputBytes = 0; let overflow = false; let timedOut = false; let settled = false;
    const append = (chunks, chunk) => {
      const remaining = maxOutputBytes - outputBytes;
      if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
      outputBytes += Math.min(chunk.length, Math.max(remaining, 0));
      if (chunk.length > remaining) { overflow = true; child.kill(); }
    };
    child.stdout.on('data', (chunk) => append(stdout, chunk));
    child.stderr.on('data', (chunk) => append(stderr, chunk));
    const abort = () => child.kill();
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
    const timer = timeoutMs > 0 ? setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs) : undefined;
    timer?.unref();
    child.once('error', (error) => {
      if (settled) return; settled = true;
      if (timer) clearTimeout(timer); signal?.removeEventListener('abort', abort);
      reject(Object.assign(new Error(redact(error.message, { env })), { code: error.code }));
    });
    child.once('close', (code, closeSignal) => {
      if (settled) return; settled = true;
      if (timer) clearTimeout(timer); signal?.removeEventListener('abort', abort);
      const result = {
        code, signal: closeSignal,
        stdout: redact(Buffer.concat(stdout).toString('utf8'), { env }),
        stderr: redact(Buffer.concat(stderr).toString('utf8'), { env }),
        timedOut, aborted: Boolean(signal?.aborted), truncated: overflow,
      };
      if (timedOut || result.aborted || overflow || code !== 0) {
        const reason = timedOut ? 'timed out' : result.aborted ? 'aborted' : overflow ? 'exceeded output limit' : `exited with code ${code}`;
        reject(Object.assign(new Error(`CLI ${reason}${result.stderr ? `: ${result.stderr.trim()}` : ''}`), result));
      } else resolve(result);
    });
    child.stdin.on('error', () => {});
    child.stdin.end(String(input));
  });
}
