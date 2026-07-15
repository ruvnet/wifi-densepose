// Static import/export graph verifier for HOMECORE-UI.
// No deps — parses `import { a, b } from './x.js'` against the named
// exports of x.js. Fails if a panel imports a symbol that doesn't exist.
// Run: node tests/verify-imports.mjs   (from the ui/ dir)
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const files = [
  'js/ui.js', 'js/api.js', 'js/ws.js', 'js/mock.js', 'js/app.js',
  ...readdirSync(resolve(ROOT, 'js/panels')).filter((f) => f.endsWith('.js')).map((f) => 'js/panels/' + f),
];

function namedExports(src) {
  const out = new Set();
  // export function/const/class NAME
  for (const m of src.matchAll(/export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z0-9_$]+)/g)) out.add(m[1]);
  // export { a, b as c }
  for (const m of src.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const part of m[1].split(',')) {
      const name = part.trim().split(/\s+as\s+/).pop().trim();
      if (name) out.add(name);
    }
  }
  if (/export\s+default/.test(src)) out.add('default');
  return out;
}

function imports(src) {
  const res = [];
  for (const m of src.matchAll(/import\s+([^;]+?)\s+from\s+['"]([^'"]+)['"]/g)) {
    const clause = m[1].trim(), spec = m[2];
    const names = [];
    const named = clause.match(/\{([^}]*)\}/);
    if (named) for (const p of named[1].split(',')) { const n = p.trim().split(/\s+as\s+/)[0].trim(); if (n) names.push(n); }
    const def = clause.replace(/\{[^}]*\}/, '').replace(/\*\s+as\s+\w+/, '').replace(/,/g, '').trim();
    if (def) names.push('default');
    if (/\*\s+as\s+/.test(clause)) names.push('*');
    res.push({ spec, names });
  }
  return res;
}

const exportCache = {};
function exportsOf(absPath) {
  if (!exportCache[absPath]) exportCache[absPath] = namedExports(readFileSync(absPath, 'utf8'));
  return exportCache[absPath];
}

let errors = 0;
for (const rel of files) {
  const abs = resolve(ROOT, rel);
  const src = readFileSync(abs, 'utf8');
  for (const imp of imports(src)) {
    if (!imp.spec.startsWith('.')) continue; // skip bare specifiers
    const target = resolve(dirname(abs), imp.spec);
    let exps;
    try { exps = exportsOf(target); } catch { console.error(`✗ ${rel}: cannot resolve ${imp.spec}`); errors++; continue; }
    for (const n of imp.names) {
      if (n === '*') continue;
      if (!exps.has(n)) { console.error(`✗ ${rel}: imports '${n}' from ${imp.spec} which does not export it`); errors++; }
    }
  }
}

if (errors) { console.error(`\nFAILED — ${errors} unresolved import(s)`); process.exit(1); }
console.log(`OK — import/export graph consistent across ${files.length} modules`);
