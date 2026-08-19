#!/usr/bin/env node
// SPDX-License-Identifier: MIT

import { createHash } from 'node:crypto';
import {
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBrain } from '../src/brain.js';

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
const files = [];

function walk(path) {
  const stat = statSync(path);
  if (stat.isDirectory()) {
    for (const name of readdirSync(path).sort()) walk(join(path, name));
  } else {
    files.push(path);
  }
}

for (const entry of INCLUDE) walk(join(ROOT, entry));
const hashes = Object.fromEntries(files.sort().map((path) => [
  relative(ROOT, path).replaceAll('\\', '/'),
  sha(canonicalFile(path)),
]));
const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
const policy = canonicalFile(join(ROOT, '.harness', 'mcp-policy.json'));
const manifest = {
  schema: 2,
  generator: 'Homecore metaharness provenance v1',
  template: 'vertical:repo-maintainer+homecore',
  name: pkg.name,
  version: pkg.version,
  hosts: ['claude-code', 'codex'],
  kernel: {
    package: '@metaharness/kernel',
    version: pkg.dependencies['@metaharness/kernel'],
    preference: 'wasm-first',
    fallback: 'native-or-js-reported',
  },
  toolPolicy: 'default-deny-execution',
  files: hashes,
  filesDigest: sha(JSON.stringify(hashes)),
  brainDigest: loadBrain().digest,
  policyDigest: sha(policy),
  dependencies: pkg.dependencies,
  meta: {
    surface: 'cli+mcp+brain+wasm+local-hosts',
    adr: 'ADR-285',
  },
};
const json = `${JSON.stringify(manifest, null, 2)}\n`;
writeFileSync(join(ROOT, '.harness', 'manifest.json'), json);
writeFileSync(join(ROOT, '.harness', 'manifest.sha256'), `${sha(json)}  manifest.json\n`);
if (!quiet) {
  console.log(JSON.stringify({ ok: true, files: files.length, digest: sha(json) }));
}
