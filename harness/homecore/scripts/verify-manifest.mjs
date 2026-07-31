#!/usr/bin/env node
// SPDX-License-Identifier: MIT

import { createHash } from 'node:crypto';
import {
  existsSync,
  readdirSync,
  readFileSync,
  statSync,
} from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const quiet = process.argv.includes('--quiet');
const INCLUDE = [
  'package.json',
  'bin',
  'src',
  'skills',
  '.claude/settings.json',
  '.claude/skills',
  '.codex',
  '.mcp',
  '.harness/claims.json',
  '.harness/mcp-policy.json',
  'brain/corpus/core.jsonl',
  'scripts',
  'AGENTS.md',
  'CLAUDE.md',
  'README.md',
  'LICENSE',
];

const sha = (value) => createHash('sha256').update(value).digest('hex');
const canonicalFile = (path) => readFileSync(path, 'utf8').replace(/\r\n/g, '\n');
const path = join(ROOT, '.harness', 'manifest.json');
const raw = readFileSync(path);
const manifest = JSON.parse(raw);
const findings = [];
const actualFiles = [];

function walk(target) {
  const stat = statSync(target);
  if (stat.isDirectory()) {
    for (const name of readdirSync(target).sort()) walk(join(target, name));
  } else {
    actualFiles.push(relative(ROOT, target).replaceAll('\\', '/'));
  }
}

for (const entry of INCLUDE) walk(join(ROOT, entry));
const actualSet = new Set(actualFiles);
const expectedSet = new Set(Object.keys(manifest.files || {}));

for (const [name, expected] of Object.entries(manifest.files || {})) {
  const target = join(ROOT, name);
  if (!existsSync(target)) findings.push(`${name}:missing`);
  else if (sha(canonicalFile(target)) !== expected) findings.push(`${name}:hash-mismatch`);
}
for (const name of actualSet) {
  if (!expectedSet.has(name)) findings.push(`${name}:untracked-by-manifest`);
}
for (const name of expectedSet) {
  if (!actualSet.has(name)) findings.push(`${name}:not-in-package-surface`);
}

const expectedOuter = readFileSync(join(ROOT, '.harness', 'manifest.sha256'), 'utf8')
  .trim()
  .split(/\s+/)[0];
if (sha(raw) !== expectedOuter) findings.push('manifest.sha256:mismatch');

if (!quiet) {
  console.log(JSON.stringify({
    ok: findings.length === 0,
    files: expectedSet.size,
    findings,
  }, null, 2));
}
process.exit(findings.length ? 1 : 0);
