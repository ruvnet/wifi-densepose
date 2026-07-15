// Benchmark — ADR-131 §8 / ADR-126 §1.1.
// HOMECORE exists partly because HA's frontend is a ~5 MB Lit bundle
// (ADR-126 §1.1). This benchmark enforces a hard bundle budget and
// measures cold render throughput for all 10 panels.
// Run: node tests/benchmark.mjs
import { install } from './dom-shim.mjs';
install();
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const BUDGET_BYTES = 250 * 1024;   // 250 KB total — vs HA's ~5 MB (20× smaller)

function walk(dir) {
  let total = 0; const rows = [];
  for (const name of readdirSync(dir)) {
    if (name === 'tests' || name === 'node_modules') continue;
    const p = resolve(dir, name); const s = statSync(p);
    if (s.isDirectory()) { const sub = walk(p); total += sub.total; rows.push(...sub.rows); }
    else if (/\.(js|css|html|json)$/.test(name)) { total += s.size; rows.push([p.replace(ROOT + '/', ''), s.size]); }
  }
  return { total, rows };
}

const { total, rows } = walk(ROOT);
rows.sort((a, b) => b[1] - a[1]);
console.log('── Bundle size (uncompressed) ──');
for (const [f, sz] of rows.slice(0, 8)) console.log(`  ${(sz / 1024).toFixed(1).padStart(7)} KB  ${f}`);
console.log(`  ${'-'.repeat(40)}`);
console.log(`  ${(total / 1024).toFixed(1).padStart(7)} KB  TOTAL across ${rows.length} files`);
console.log(`  budget ${(BUDGET_BYTES / 1024).toFixed(0)} KB · HA baseline ~5120 KB · ratio ${(5120 * 1024 / total).toFixed(1)}× smaller`);

// ── render throughput ───────────────────────────────────────────────
const { api } = await import('../js/api.js');
const ctx = { api, navigate() {}, params: { id: 'seed-livingroom-a1' }, onEvent() { return () => {}; }, onWs(fn) { fn({ state: 'open', lagged: false }); return () => {}; } };
const PANELS = ['dashboard', 'fleet', 'seed-detail', 'entities', 'rooms', 'cogs', 'calibration', 'events', 'audit', 'settings'];
const mods = {};
for (const p of PANELS) mods[p] = (await import(`../js/panels/${p}.js`)).default;

console.log('\n── Cold render throughput (avg of 50 renders each) ──');
let worst = 0;
for (const p of PANELS) {
  const N = 50; const t0 = performance.now();
  for (let i = 0; i < N; i++) { const root = document.createElement('div'); const c = await mods[p].render(root, ctx); if (typeof c === 'function') c(); }
  const ms = (performance.now() - t0) / N;
  worst = Math.max(worst, ms);
  console.log(`  ${ms.toFixed(3).padStart(7)} ms/render  ${p}`);
}

console.log('');
let exit = 0;
if (total > BUDGET_BYTES) { console.error(`FAIL — bundle ${(total / 1024).toFixed(1)} KB exceeds ${(BUDGET_BYTES / 1024).toFixed(0)} KB budget`); exit = 1; }
else console.log(`OK — bundle within budget; slowest panel ${worst.toFixed(2)} ms/render`);
process.exit(exit);
