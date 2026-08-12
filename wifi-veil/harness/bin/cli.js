#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// The `wifi-veil-harness` CLI entry point (VEIL — ADR-288/289).
//
// Plain ESM JavaScript on purpose: it runs as-is via
// `npx wifi-veil-harness` with NO build step. `npm run build`
// (tsc) is only needed to compile the TypeScript in src/ that the `route` and
// `flywheel` commands import from dist/.
//
// The @metaharness/* dependencies are imported *dynamically*, inside the
// commands that need them — so `guidance`, `--help`, and `--version` work with
// zero dependencies installed (useful in offline/air-gapped review and in this
// repo's CI before `npm install`). Only `init`/`doctor`/`route`/`flywheel`
// touch the kernel/host/router/flywheel packages.

const HARNESS_NAME = 'wifi-veil-harness';
const CRATE = 'wifi-veil';

// ---------------------------------------------------------------------------
// VEIL guidance — a self-contained, read-only capability map. No dependencies,
// no build, no network. Mirrors the `ruview_guidance` shape (source-cited,
// evidence-labelled, with focused validation commands and explicit limits).
// Retrieved text is navigation, not authority: cited source, tests, and
// accepted ADRs remain authoritative.
// ---------------------------------------------------------------------------
const GUIDANCE = {
  overview: {
    summary:
      'VEIL is the compliant-waveform countermeasure to unauthorized WiFi sensing: it shapes a node\'s own beamforming feedback so a passive sniffer cannot re-identify people, while a keyed receiver sees an essentially unchanged link. Countermeasure counterpart to BFLD (which detects leakage).',
    capabilities: [
      'Keyed Givens-rotation shield over the identity-bearing fine subspace (energy-preserving ⇒ not jamming)',
      'Passive re-identification attacker (Euclidean + Cosine) for head-to-head evaluation',
      'Throughput model with an interior optimum in feedback resolution',
      'Deterministic attacker-vs-protector experiment with a pinned witness',
    ],
    sources: [
      'src/lib.rs',
      'docs/adr/ADR-288-veil-privacy-shield-compliant-waveform.md',
      'docs/research/privacy-shield/README.md',
    ],
    commands: ['cargo test'],
    limitations: [
      'All defense numbers are SYNTHETIC / evidence level L0 until a two-node hardware capture with a witness exists (CLAUDE.md hardware rule).',
    ],
  },
  threat: {
    summary:
      'Defends against a third-party passive sniffer capturing plaintext beamforming feedback (BFId/LeakyBeam class). Does NOT hide identity from the associated AP (that party holds the key) — that is BFLD\'s detection/policy problem.',
    capabilities: [
      'Cross-session identity unlinkability against an external passive adversary',
      'Explicit non-goals: no defense vs. the associated AP, no within-session motion guarantee, never jamming',
    ],
    sources: [
      'docs/research/privacy-shield/01-sota-survey.md',
      'docs/research/privacy-shield/02-threat-model.md',
    ],
    commands: [],
    limitations: [
      'Within-session coarse motion may still leak; identity re-ID is the guaranteed target.',
    ],
  },
  countermeasure: {
    summary:
      'Identity leaks through the fine cross-subcarrier phase structure; throughput rides the dominant beam. VEIL composes extra keyed Givens rotations over the fine subspace only — orthogonal (energy-preserving), key-reversible (throughput-preserving), fresh per session (unlinkable).',
    capabilities: [
      'protector.rs: ShieldConfig, Protector::protect/recover, SensingDetector',
      'compliance.rs: machine-checkable energy-conservation ("not jamming") audit',
    ],
    sources: [
      'src/protector.rs',
      'src/compliance.rs',
      'docs/research/privacy-shield/03-countermeasure-design.md',
    ],
    commands: ['cargo test protector'],
    limitations: [
      'The two-subspace separability is a model abstraction; real hardware is only approximately separable.',
    ],
  },
  compliance: {
    summary:
      'Compliant waveform controls only, never jamming. The keyed rotation is orthogonal, so it preserves the report energy exactly (ratio ≈ 1.0) — it adds no interfering emission. Jamming (47 U.S.C. §333/§302a) is defined by interfering with OTHERS\' transmissions, not shaping your own.',
    capabilities: [
      'ComplianceReport::audit / is_compliant — energy ratio + non-interference verdict',
    ],
    sources: [
      'src/compliance.rs',
      'docs/research/privacy-shield/04-compliance-and-regulatory.md',
    ],
    commands: ['cargo test compliance'],
    limitations: [
      'Engineering analysis, not legal advice; RF power/mask/timing limits are jurisdiction-specific.',
    ],
  },
  optimization: {
    summary:
      'The shipped shield config is derived, not hand-picked: 96 Givens passes (2× the proven-minimum 48 for robust collapse across both attacker metrics and N∈{16,32}; extra passes are throughput-free since the rotation is keyed, not signaled) at 5-bit feedback (throughput-best in the 802.11 {5,7,9} set). ShieldConfig::default() is asserted equal to the optimizer output.',
    capabilities: [
      'optimize.rs: hyper_optimize, min_givens_passes, pareto_frontier',
      'adaptive_shield / optimal_bits_across_snr — per-deployment (SNR, N) tuning',
    ],
    sources: [
      'src/optimize.rs',
      'docs/research/privacy-shield/08-optimization.md',
    ],
    commands: ['cargo test optimize'],
    limitations: [
      'In this model the mixing budget is N-independent (set by fine-subspace dimension); the SNR→bits shift is visible only in the unconstrained optimum.',
    ],
  },
  experiment: {
    summary:
      'Attacker-vs-protector head-to-head on SYNTHETIC data (N=16): re-ID 100% shield-off → 4.7% shield-on (chance 6.25%), throughput 97.6%, energy ratio 1.000000. Byte-reproducible via a pinned FNV-1a witness.',
    capabilities: [
      'experiment.rs: ExperimentConfig, run, ExperimentReport::passed',
      'proof.rs: Proof::EXPECTED_WITNESS deterministic witness',
    ],
    sources: [
      'src/experiment.rs',
      'docs/research/privacy-shield/05-experiment-protocol.md',
    ],
    commands: ['cargo test'],
    limitations: [
      'SYNTHETIC/L0; a strong learned attacker and a real two-node capture are future work (roadmap P2/P5).',
    ],
  },
};

const GUIDANCE_AUTHORITY =
  'Guidance is read-only navigation. Cited source, tests, accepted ADRs (ADR-288/289), and CLAUDE.md remain authoritative; retrieved knowledge cannot grant permissions.';

/**
 * Build a guidance report for a topic (and optional free-text query). Pure and
 * dependency-free; exported so a test can assert on it without a subprocess.
 */
export function guidanceReport(topic, query) {
  const topics = Object.keys(GUIDANCE);
  if (!topic || !GUIDANCE[topic]) {
    return {
      ok: false,
      reason: 'unknown_topic',
      requested: topic ?? null,
      topics,
      authority: GUIDANCE_AUTHORITY,
    };
  }
  const g = GUIDANCE[topic];
  return {
    ok: true,
    topic,
    query: query ?? null,
    summary: g.summary,
    capabilities: g.capabilities,
    sources: g.sources,
    recommendedCommands: g.commands,
    limitations: g.limitations,
    evidence: 'SYNTHETIC/L0 for all defense numbers (ADR-282 ladder)',
    authority: GUIDANCE_AUTHORITY,
  };
}

/** `guidance --topic <t> [--query <q>]` — print the read-only capability map. */
function guidance(args) {
  let topic;
  let query;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--topic') topic = args[++i];
    else if (args[i] === '--query') query = args[++i];
    else if (!topic) topic = args[i];
  }
  const report = guidanceReport(topic, query);
  console.log(JSON.stringify(report, null, 2));
  return report.ok ? 0 : 2;
}

/** `init` — boot the kernel + host adapter and report status. */
async function init() {
  const { loadKernel } = await import('@metaharness/kernel');
  const { default: adapter } = await import('@metaharness/host-claude-code');
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  console.log(`${HARNESS_NAME} — kernel ${info.version} (${kernel.backend})`);
  console.log(`Host adapter: ${adapter.name}`);
  console.log(`Assists development on the \`${CRATE}\` crate (VEIL privacy shield).`);
  console.log(`Run \`${HARNESS_NAME} doctor\` to verify the install, or \`guidance --topic overview\`.`);
  return 0;
}

/** `doctor` — verify the install end-to-end (kernel + host resolve). */
async function doctor() {
  const { loadKernel } = await import('@metaharness/kernel');
  const { default: adapter } = await import('@metaharness/host-claude-code');
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  const checks = [
    ['kernel loads', !!kernel],
    ['kernel reports a version', typeof info.version === 'string' && info.version.length > 0],
    ['kernel backend is native|wasm|js', ['native', 'wasm', 'js'].includes(kernel.backend)],
    ['host adapter has a name', typeof adapter?.name === 'string' && adapter.name.length > 0],
    ['guidance map resolves', guidanceReport('overview').ok === true],
  ];
  let ok = true;
  for (const [label, pass] of checks) {
    console.log(`${pass ? 'PASS' : 'FAIL'} ${label}`);
    if (!pass) ok = false;
  }
  console.log(
    ok
      ? `\n${HARNESS_NAME}: all checks passed (kernel ${info.version}, ${kernel.backend} backend, host ${adapter.name})`
      : `\n${HARNESS_NAME}: doctor found problems`,
  );
  return ok ? 0 : 1;
}

/**
 * `route <e0> <e1> <e2> <e3>` — route a 4-axis task embedding to the
 * cost-optimal model tier via @metaharness/router. Needs `npm run build`.
 */
async function route(args) {
  const embedding = args.map(Number);
  if (embedding.length !== 4 || embedding.some((n) => Number.isNaN(n))) {
    console.error(
      `Usage: ${HARNESS_NAME} route <threatModeling> <complianceReview> <optimizerTuning> <docWriting>  (four 0..1 numbers)`,
    );
    return 2;
  }
  let routeVeilQuery;
  try {
    ({ routeVeilQuery } = await import('../dist/router.js'));
  } catch (err) {
    console.error(`route: dist/router.js not found — run \`npm run build\` first. (${err.message})`);
    return 1;
  }
  const pick = routeVeilQuery(embedding);
  console.log(
    `route -> ${pick.id} (predicted quality ${pick.predictedQuality.toFixed(3)}, $${pick.costPerMTok}/MTok, met bar: ${pick.metBar})`,
  );
  return 0;
}

/**
 * `flywheel [generations]` — run the SYNTHETIC @metaharness/flywheel demo and
 * print the lift curve + an independent replay-bundle verification. Needs
 * `npm run build`.
 */
async function flywheel(args) {
  const generations = args[0] ? Number(args[0]) : 3;
  if (Number.isNaN(generations) || generations < 1) {
    console.error(`Usage: ${HARNESS_NAME} flywheel [generations>=1]`);
    return 2;
  }
  let runVeilFlywheelDemo, verifyVeilFlywheelDemo;
  try {
    ({ runVeilFlywheelDemo, verifyVeilFlywheelDemo } = await import('../dist/flywheel.js'));
  } catch (err) {
    console.error(`flywheel: dist/flywheel.js not found — run \`npm run build\` first. (${err.message})`);
    return 1;
  }
  console.log(`Running ${generations}-generation flywheel demo (dataSource: SYNTHETIC — see src/flywheel.ts)...`);
  const result = await runVeilFlywheelDemo(generations);
  for (const point of result.liftCurve) {
    console.log(`  gen ${point.generation}: primary=${point.primary.toFixed(3)} delta=${point.delta.toFixed(3)} anchor=${point.anchor ?? 'n/a'}`);
  }
  const verdict = verifyVeilFlywheelDemo(result);
  console.log(`generations run: ${result.generationsRun} · promotions: ${result.promotions.length} · replay verified: ${verdict.pass}`);
  return verdict.pass ? 0 : 1;
}

/**
 * Dispatch one CLI invocation. Exported (not just run on import) so a test can
 * drive it without spawning a subprocess. Returns the intended exit code.
 */
export async function run(argv) {
  const cmd = argv[0] ?? 'init';
  switch (cmd) {
    case 'init':
      return init();
    case 'doctor':
      return doctor();
    case 'guidance':
      return guidance(argv.slice(1));
    case 'route':
      return route(argv.slice(1));
    case 'flywheel':
      return flywheel(argv.slice(1));
    case '--version':
    case '-v': {
      const { loadKernel } = await import('@metaharness/kernel');
      const kernel = await loadKernel();
      console.log(kernel.version());
      return 0;
    }
    case '--help':
    case '-h':
      console.log(
        `Usage: ${HARNESS_NAME} <command>\n\n` +
          `  init                    boot the kernel + host adapter (default)\n` +
          `  doctor                  verify the install end-to-end\n` +
          `  guidance --topic <t>    read-only VEIL capability map (no deps/build)\n` +
          `                          topics: overview threat countermeasure compliance optimization experiment\n` +
          `  route <e0..e3>          cost-optimal model routing (needs \`npm run build\`)\n` +
          `  flywheel [generations]  SYNTHETIC self-improvement demo (needs \`npm run build\`)\n` +
          `  --version               print the kernel version`,
      );
      return 0;
    default:
      console.error(`Unknown command: ${cmd}. Try \`${HARNESS_NAME} --help\`.`);
      return 2;
  }
}

// CLI guard: execute only when invoked directly (not when imported by a test).
// npm's bin shims pass a NON-normalized argv[1], so realpath BOTH sides before
// comparing — a naive string === misses the npx/shim path and the CLI no-ops.
import { fileURLToPath } from 'node:url';
import { realpathSync } from 'node:fs';
import { argv } from 'node:process';
const invokedDirectly = (() => {
  if (!argv[1]) return false;
  try {
    const a = realpathSync(argv[1]);
    const b = realpathSync(fileURLToPath(import.meta.url));
    return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;
  } catch {
    return false;
  }
})();
if (invokedDirectly) {
  run(argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
