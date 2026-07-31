#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// `npx ruview` — the RuView WiFi-sensing operator harness (minted via metaharness,
// hardened per ADR-182). Plain ESM, no build step: ships and runs as-is.
//
// The `ruview.*` tools (onboard/verify/claim-check/…) and local host adapters are
// pure Node and run with zero runtime dependencies.

import { fileURLToPath } from 'node:url';
import { realpathSync, existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { argv } from 'node:process';
import { TOOLS, runTool, listTools, findRepoRoot, which } from '../src/tools.js';
import { claimCheck, summarize } from '../src/guardrails.js';
import { getHost } from '../src/hosts/index.js';
import { makeProposal, searchBrain, verifyBrain } from '../src/brain.js';

const NAME = 'ruview';
const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const SKILLS_DIR = join(ROOT, 'skills');

// Map friendly CLI verbs → registry tool names (underscore-canonical, ADR-263).
const VERB_TO_TOOL = {
  onboard: 'ruview_onboard',
  verify: 'ruview_verify',
  'claim-check': 'ruview_claim_check',
  calibrate: 'ruview_calibrate',
  monitor: 'ruview_node_monitor',
  flash: 'ruview_node_flash',
  guidance: 'ruview_guidance',
};

function pjson(o) { console.log(JSON.stringify(o, null, 2)); }

function listSkills() {
  if (!existsSync(SKILLS_DIR)) return [];
  return readdirSync(SKILLS_DIR).filter((f) => f.endsWith('.md')).map((f) => f.replace(/\.md$/, ''));
}

async function doctor() {
  const checks = [];
  // Tools layer (always available, no deps).
  checks.push(['tool registry loads', Object.keys(TOOLS).length > 0]);
  checks.push(['claim_check flags a 100% claim',
    !claimCheck('We hit 100% accuracy on poses.').ok]);
  checks.push(['claim_check passes a tagged MEASURED claim',
    claimCheck('Held-out PCK@20 59.5% (MEASURED vs mean-pose baseline, verify.py).').ok]);
  checks.push(['skills present', listSkills().length > 0]);
  checks.push(['Claude Code adapter resolves', getHost('claude-code').name === 'claude-code']);
  checks.push(['Codex adapter resolves', getHost('codex').name === 'codex']);
  const localHosts = [
    which('claude') ? 'claude -p' : null,
    which('codex') ? 'codex exec' : null,
  ].filter(Boolean);
  let ok = true;
  for (const [label, pass] of checks) { console.log(`${pass ? 'PASS' : 'FAIL'} ${label}`); if (!pass) ok = false; }
  console.log(`\n${NAME}: ${ok ? 'all checks passed' : 'doctor found problems'} — local hosts: ${localHosts.join(', ') || 'none on PATH (optional)'}`);
  return ok ? 0 : 1;
}

function help() {
  console.log(`Usage: ${NAME} <command> [options]

Operator tools:
  onboard [--path docker-demo|repo-build|live-esp32]   pick a setup path
  verify [--repo <dir>]                                 run the deterministic proof (VERDICT: PASS)
  claim-check --text "..."  |  --file <path>            lint accuracy claims (the honesty guardrail)
  calibrate --step baseline|enroll|train-room|room-watch
  monitor --port COM8 [--seconds 12]                    assert CSI is flowing on a node
  flash --port COM8 --variant s3-8mb [--confirm]        build+flash firmware (Windows/ESP-IDF)
  guidance [--topic homecore] [--query "Wasmtime"]      source-cited code/capability map

Harness:
  doctor                 verify tools, adapters, and local CLI discovery
  skills                 list bundled skills
  skill <name>           print a skill playbook
  mcp start              run the ruview.* MCP server (stdio)
  install --host <h>     project the harness config into the current repo
  agent run --host claude-code|codex --prompt "..." [--repo <dir>]
  brain search --query "..." | verify | propose
  --version | --help

Hosts implemented and tested locally: claude-code (-p), codex (exec)`);
  return 0;
}

/** tiny flag parser: --k v / --k=v / --flag (boolean) */
function parseFlags(rest) {
  const f = {};
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq !== -1) { f[a.slice(2, eq)] = a.slice(eq + 1); }
      else if (i + 1 < rest.length && !rest[i + 1].startsWith('--')) { f[a.slice(2)] = rest[++i]; }
      else { f[a.slice(2)] = true; }
    }
  }
  return f;
}

export async function run(args) {
  const cmd = args[0] ?? 'onboard';
  const rest = args.slice(1);
  const flags = parseFlags(rest);

  // Direct tool verbs.
  if (VERB_TO_TOOL[cmd]) {
    const toolArgs = { ...flags };
    if (cmd === 'claim-check') {
      if (flags.file) {
        toolArgs.text = readFileSync(flags.file, 'utf8');
        delete toolArgs.file;
      }
      // Fail closed (ADR-263 O1): an honesty gate must never PASS on no input.
      if (typeof toolArgs.text !== 'string' || toolArgs.text.trim().length === 0) {
        console.error('claim-check: no input — pass --text "..." or --file <path> (empty input is an error, not a PASS).');
        return 2;
      }
      const res = await runTool('ruview_claim_check', toolArgs);
      pjson(res);
      return res.ok ? 0 : 1;
    }
    if (cmd === 'monitor' && flags.seconds) toolArgs.seconds = Number(flags.seconds);
    if (cmd === 'guidance' && flags.limit) toolArgs.limit = Number(flags.limit);
    if (cmd === 'calibrate' && typeof flags.args === 'string') toolArgs.args = flags.args.split(',');
    const res = await runTool(VERB_TO_TOOL[cmd], toolArgs);
    pjson(res);
    return res.ok ? 0 : 1;
  }

  switch (cmd) {
    case 'doctor': return doctor();
    case 'skills': console.log(listSkills().join('\n') || '(none)'); return 0;
    case 'skill': {
      const n = rest[0];
      const p = n && join(SKILLS_DIR, `${n}.md`);
      if (!p || !existsSync(p)) { console.error(`No skill "${n}". Try: ${listSkills().join(', ')}`); return 2; }
      console.log(readFileSync(p, 'utf8'));
      return 0;
    }
    case 'mcp': {
      if (rest[0] === 'start' || rest[0] === undefined) {
        const { startMcpServer } = await import('../src/mcp-server.js');
        startMcpServer();
        return new Promise(() => {}); // run until stdin closes
      }
      console.error('Usage: ruview mcp start'); return 2;
    }
    case 'agent': {
      if (rest[0] !== 'run') { console.error('Usage: ruview agent run --host claude-code|codex --prompt "..." [--repo <dir>]'); return 2; }
      const hostName = String(flags.host || 'codex');
      const prompt = String(flags.prompt || '');
      const repo = flags.repo ? resolve(flags.repo) : findRepoRoot();
      if (!repo) { console.error('agent run: trusted RuView repo not found; pass --repo <root>.'); return 2; }
      if (!prompt.trim()) { console.error('agent run: --prompt is required.'); return 2; }
      const allowWrite = flags['allow-write'] === true;
      if (allowWrite && flags.confirm !== true) { console.error('agent run: --allow-write also requires --confirm.'); return 2; }
      try {
        const result = await getHost(hostName).run({
          prompt, repoRoot: repo, trustedRoot: repo, allowWrite, confirm: flags.confirm === true,
        });
        pjson({ ok: true, host: hostName, mode: allowWrite ? 'workspace-write' : 'read-only', stdout: result.stdout, stderr: result.stderr });
        return 0;
      } catch (error) {
        pjson({ ok: false, host: hostName, error: error.message });
        return 1;
      }
    }
    case 'brain': {
      const action = rest[0] || 'search';
      if (action === 'search') {
        const query = String(flags.query || '');
        if (!query.trim()) { console.error('brain search: --query is required.'); return 2; }
        pjson({ ok: true, results: searchBrain(query, { limit: flags.limit }) }); return 0;
      }
      if (action === 'verify') {
        const repo = flags.repo ? resolve(flags.repo) : findRepoRoot();
        if (!repo) { console.error('brain verify: RuView repo not found.'); return 2; }
        const result = verifyBrain({ repo }); pjson(result); return result.ok ? 0 : 1;
      }
      if (action === 'propose') {
        const result = makeProposal(flags); pjson(result); return result.ok ? 0 : 1;
      }
      console.error('Usage: ruview brain search|verify|propose'); return 2;
    }
    case 'install': {
      const host = flags.host || 'claude-code';
      if (!['claude-code', 'codex'].includes(host)) {
        console.error(`Host "${host}" is not implemented. Supported: claude-code, codex.`);
        return 2;
      }
      try {
        const adapter = getHost(host);
        console.log(`Projecting RuView harness for host "${host}" via ${adapter.name}.`);
        console.log('Add to your host config — MCP server command: npx -y ruview mcp start');
        console.log('Skills:', listSkills().join(', '));
        return 0;
      } catch {
        console.error(`Host adapter "${host}" is unavailable.`);
        return 1;
      }
    }
    case 'tools': pjson(listTools()); return 0;
    case '--version': case '-v': {
      const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
      console.log(pkg.version); return 0;
    }
    case '--help': case '-h': return help();
    default:
      console.error(`Unknown command: ${cmd}. Try \`${NAME} --help\`.`);
      return 2;
  }
}

// CLI guard: run only when invoked directly (realpath both sides — npm/npx shims
// pass a non-normalized, possibly case-skewed argv[1] on Windows).
const invokedDirectly = (() => {
  if (!argv[1]) return false;
  try {
    const a = realpathSync(argv[1]);
    const b = realpathSync(fileURLToPath(import.meta.url));
    return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;
  } catch { return false; }
})();
if (invokedDirectly) {
  run(argv.slice(2)).then((code) => process.exit(code)).catch((err) => { console.error(err); process.exit(1); });
}
