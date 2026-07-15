#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// ADR-263 O7: skills/*.md is the single source of truth; the host-projected
// copies (.claude/skills/<name>/SKILL.md) are GENERATED here at pack time.
// Run with --check to verify without writing (used by tests/CI).

import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const SRC = join(ROOT, 'skills');
const DST = join(ROOT, '.claude', 'skills');
const checkOnly = process.argv.includes('--check');

let drift = 0;
for (const f of readdirSync(SRC).filter((f) => f.endsWith('.md'))) {
  const name = f.replace(/\.md$/, '');
  const src = readFileSync(join(SRC, f), 'utf8');
  const dstDir = join(DST, name);
  const dstFile = join(dstDir, 'SKILL.md');
  const current = existsSync(dstFile) ? readFileSync(dstFile, 'utf8') : null;
  if (current === src) continue;
  drift++;
  if (checkOnly) {
    console.error(`DRIFT: .claude/skills/${name}/SKILL.md != skills/${f}`);
  } else {
    mkdirSync(dstDir, { recursive: true });
    writeFileSync(dstFile, src);
    console.error(`synced .claude/skills/${name}/SKILL.md`);
  }
}
if (checkOnly && drift > 0) {
  console.error(`sync-skills --check: ${drift} file(s) out of sync — run \`npm run sync-skills\`.`);
  process.exit(1);
}
console.error(`sync-skills: ${drift === 0 ? 'all in sync' : `${drift} file(s) ${checkOnly ? 'OUT OF SYNC' : 'synced'}`}`);
