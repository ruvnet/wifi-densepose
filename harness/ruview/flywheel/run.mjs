#!/usr/bin/env node
// Human-triggered Darwin exploration. It produces untrusted proposal artifacts;
// it never updates the committed champion or publishes a package.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { evaluateGenome, gateFingerprint, loadEvaluation } from './gate.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const confirmed = args.includes('--confirm');
const genome = JSON.parse(readFileSync(join(ROOT, 'flywheel', 'genome.json'), 'utf8'));
const suites = loadEvaluation();
const report = {
  mode: confirmed ? 'darwin-proposal' : 'dry-run',
  writesChampion: false,
  gateFingerprint: gateFingerprint(),
  baseline: {
    holdout: evaluateGenome(genome, suites.holdout),
    anchor: evaluateGenome(genome, suites.anchor),
  },
  command: ['metaharness-darwin', 'evolve', ROOT, '--generations', '2', '--children', '3', '--concurrency', '2', '--selection', 'pareto', '--seed', '182', '--sandbox', 'real'],
};

if (!confirmed) {
  console.log(JSON.stringify(report, null, 2));
  process.exit(report.baseline.anchor.regressed ? 1 : 0);
}

const cli = join(ROOT, 'node_modules', '@metaharness', 'darwin', 'dist', 'cli.js');
if (!existsSync(cli)) {
  console.error('Pinned Darwin binary missing. Run `npm ci` in harness/ruview.');
  process.exit(2);
}
const child = spawn(process.execPath, [cli, ...report.command.slice(1)], {
  cwd: ROOT,
  shell: false,
  stdio: 'inherit',
  env: { PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE },
});
child.once('exit', (code) => process.exit(code ?? 2));
