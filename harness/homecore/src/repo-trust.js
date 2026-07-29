// SPDX-License-Identifier: MIT

import {
  closeSync,
  existsSync,
  openSync,
  readSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { dirname, isAbsolute, join, parse, relative, resolve } from 'node:path';

const REQUIRED_MARKERS = Object.freeze([
  '.git',
  'README.md',
  'v2/Cargo.toml',
  'v2/crates/homecore/Cargo.toml',
  'v2/crates/homecore-server/Cargo.toml',
  'docs/adr/ADR-126-ruview-native-ha-port-master.md',
]);

function isWithin(parent, child) {
  const rel = relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

function readContainedPrefix(root, path, maxBytes) {
  const real = realpathSync(path);
  if (!isWithin(root, real)) {
    throw new Error('Refusing CLI access: repository marker escapes the trusted root');
  }
  const stat = statSync(real);
  if (!stat.isFile()) {
    throw new Error('Refusing CLI access: README marker is not a regular file');
  }
  const buffer = Buffer.alloc(Math.min(stat.size, maxBytes));
  const descriptor = openSync(real, 'r');
  try {
    const bytes = readSync(descriptor, buffer, 0, buffer.length, 0);
    return buffer.subarray(0, bytes).toString('utf8');
  } finally {
    closeSync(descriptor);
  }
}

export function looksLikeHomecoreRepo(path) {
  if (!path || !existsSync(path)) return false;
  return REQUIRED_MARKERS.every((marker) => existsSync(join(path, marker)));
}

export function findHomecoreRepo(start = process.cwd()) {
  let current = resolve(start);
  const root = parse(current).root;
  while (true) {
    if (looksLikeHomecoreRepo(current)) return realpathSync(current);
    if (current === root) return null;
    const parent = dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

export function assertTrustedHomecoreRepo(repoRoot, { trustedRoot = repoRoot } = {}) {
  if (!repoRoot || !trustedRoot) throw new TypeError('repoRoot and trustedRoot are required');
  const root = realpathSync(repoRoot);
  const trustAnchor = realpathSync(trustedRoot);
  if (!isWithin(trustAnchor, root) || root !== trustAnchor) {
    throw new Error('Refusing CLI access: repository does not match the configured trusted root');
  }
  if (!statSync(root).isDirectory()) {
    throw new Error('Refusing CLI access: trusted root is not a directory');
  }
  const missing = REQUIRED_MARKERS.filter((marker) => !existsSync(join(root, marker)));
  if (missing.length) {
    throw new Error(`Refusing CLI access: Homecore repository markers are missing (${missing.join(', ')})`);
  }
  const readme = readContainedPrefix(root, join(root, 'README.md'), 131_072);
  if (!/\b(?:RuView|wifi[- ]densepose)\b/i.test(readme)) {
    throw new Error('Refusing CLI access: README does not identify a RuView checkout');
  }
  return root;
}
