#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const quiet = process.argv.includes('--quiet');
const sha = (value) => createHash('sha256').update(value).digest('hex');
const canonicalFile = (path) => readFileSync(path, 'utf8').replace(/\r\n/g, '\n');
const path = join(ROOT, '.harness', 'manifest.json');
const raw = readFileSync(path);
const manifest = JSON.parse(raw);
const findings = [];
for (const [name, expected] of Object.entries(manifest.files || {})) {
  const target = join(ROOT, name);
  if (!existsSync(target)) findings.push(`${name}:missing`);
  else if (sha(canonicalFile(target)) !== expected) findings.push(`${name}:hash-mismatch`);
}
const expectedOuter = readFileSync(join(ROOT, '.harness', 'manifest.sha256'), 'utf8').trim().split(/\s+/)[0];
if (sha(raw) !== expectedOuter) findings.push('manifest.sha256:mismatch');
if (!quiet) console.log(JSON.stringify({ ok: findings.length === 0, files: Object.keys(manifest.files || {}).length, findings }, null, 2));
process.exit(findings.length ? 1 : 0);
