// SPDX-License-Identifier: MIT
import { existsSync, realpathSync, readFileSync, statSync } from 'node:fs';
import { isAbsolute, join, relative } from 'node:path';
const REQUIRED_MARKERS = ['.git', 'README.md', 'v2'];
const RUVIEW_MARKERS = ['firmware', 'wifi_densepose'];
function isWithin(parent, child) {
  const rel = relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}
export function assertTrustedRuViewRepo(repoRoot, { trustedRoot = repoRoot } = {}) {
  if (!repoRoot || !trustedRoot) throw new TypeError('repoRoot and trustedRoot are required');
  const root = realpathSync(repoRoot);
  const trustAnchor = realpathSync(trustedRoot);
  if (!isWithin(trustAnchor, root) || root !== trustAnchor) throw new Error('Refusing CLI access: repository does not match the configured trusted root');
  if (!statSync(root).isDirectory()) throw new Error('Refusing CLI access: trusted root is not a directory');
  const missing = REQUIRED_MARKERS.filter((marker) => !existsSync(join(root, marker)));
  if (missing.length || !RUVIEW_MARKERS.some((marker) => existsSync(join(root, marker)))) {
    throw new Error(`Refusing CLI access: RuView repository markers are missing${missing.length ? ` (${missing.join(', ')})` : ''}`);
  }
  const readme = readFileSync(join(root, 'README.md'), 'utf8').slice(0, 131_072);
  if (!/\b(?:RuView|wifi[- ]densepose)\b/i.test(readme)) throw new Error('Refusing CLI access: README does not identify a RuView checkout');
  return root;
}
