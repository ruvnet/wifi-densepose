// SPDX-License-Identifier: MIT
// RuView harness — the `ruview.*` tool registry.
//
// One registry consumed by BOTH the CLI (`npx ruview <tool>`) and the MCP server
// (`npx ruview mcp start`). Every handler returns structured JSON and is
// FAIL-CLOSED: when a prerequisite (the RuView repo, python+pyserial, the
// `wifi-densepose` binary, an ESP32 on a port) is absent, it returns an honest
// negative — never a fabricated success. This mirrors the project's "prove
// everything" rule and the RuField fail-closed posture (ADR-262 §3.3).
//
// ADR-263: handlers are async (promise-based spawn, never spawnSync) so the MCP
// server keeps answering ping/tools/list while a long verify/calibrate runs.
// Canonical tool names use underscores (host tool-name regexes commonly enforce
// ^[a-zA-Z0-9_-]{1,64}$); the historical dotted names are accepted as aliases.

import { spawn } from 'node:child_process';
import { existsSync, accessSync, constants } from 'node:fs';
import { join, dirname, resolve, delimiter } from 'node:path';
import { claimCheck, summarize } from './guardrails.js';

/** Walk up from `start` to find the RuView monorepo root (or null). */
export function findRepoRoot(start = process.cwd()) {
  let dir = resolve(start);
  for (let i = 0; i < 8; i++) {
    const hasProof = existsSync(join(dir, 'archive', 'v1', 'data', 'proof', 'verify.py'));
    const hasV2 = existsSync(join(dir, 'v2', 'Cargo.toml'));
    if (hasProof || hasV2) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

// Dep-free PATH scan (ADR-263 O8) — no shell subprocess per lookup. Only hits
// are memoized: a miss can resolve later in a long-lived MCP session (the
// operator installs python/the CLI mid-run), so misses are re-probed each call.
const whichCache = new Map();
export function which(cmd) {
  if (whichCache.has(cmd)) return whichCache.get(cmd);
  const isWin = process.platform === 'win32';
  const exts = isWin
    ? (process.env.PATHEXT || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean)
    : [''];
  let found = null;
  outer:
  for (const dir of (process.env.PATH || '').split(delimiter)) {
    if (!dir) continue;
    for (const ext of isWin ? ['', ...exts] : exts) {
      const p = join(dir, cmd + ext);
      try {
        accessSync(p, isWin ? constants.F_OK : constants.X_OK);
        found = p;
        break outer;
      } catch { /* keep scanning */ }
    }
  }
  if (found !== null) whichCache.set(cmd, found);
  return found;
}

// Bounded output tails (ADR-263 O4): spawnSync's default 1 MiB maxBuffer killed
// chatty children with ENOBUFS; handlers only ever surface the last few kB, so
// keep rolling tails instead of the full stream.
const STDOUT_TAIL = 65536;
const STDERR_TAIL = 16384;

/** Promise-based spawn with timeout + rolling output tails. */
export function run(cmd, args, opts = {}) {
  const timeout = opts.timeout ?? 120000;
  return new Promise((resolvePromise) => {
    let stdout = '';
    let stderr = '';
    let child;
    try {
      child = spawn(cmd, args, { cwd: opts.cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (e) {
      resolvePromise({ status: null, ok: false, stdout: '', stderr: '', error: e.message });
      return;
    }
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeout);
    child.stdout.on('data', (d) => {
      stdout = (stdout + d).slice(-STDOUT_TAIL);
    });
    child.stderr.on('data', (d) => {
      stderr = (stderr + d).slice(-STDERR_TAIL);
    });
    child.on('error', (e) => {
      clearTimeout(timer);
      resolvePromise({ status: null, ok: false, stdout, stderr, error: e.message });
    });
    child.on('close', (status) => {
      clearTimeout(timer);
      resolvePromise({
        status,
        ok: status === 0,
        stdout,
        stderr,
        error: timedOut ? `timed out after ${timeout} ms` : null,
      });
    });
  });
}

const ONBOARD_PATHS = {
  'docker-demo': 'Fastest. `docker run -p 8000:8000 ruvnet/wifi-densepose` → open the dashboard. No hardware; replays sample CSI. Good for "what does it look like".',
  'repo-build': 'Build from source. `cd v2 && cargo test --workspace --no-default-features` (1,031+ tests). Then `cargo run -p wifi-densepose-cli -- --help`. Good for developers.',
  'live-esp32': 'Real sensing. Flash an ESP32-S3 (see `provision-node` skill), point it at the sensing-server, then `calibrate → enroll → train-room → room-watch` (see `calibrate-room`). Good for an actual install.',
};

// Read-only serial monitor script; the port arrives via sys.argv (ADR-263 O5 —
// never spliced into interpreter source).
const MONITOR_SCRIPT = [
  'import sys,time',
  'try:',
  ' import serial',
  'except Exception as e:',
  " print('NO_PYSERIAL'); sys.exit(3)",
  'port=sys.argv[1]',
  'dur=float(sys.argv[2])',
  'ser=serial.Serial(port,115200,timeout=1)',
  'csi=0; n=0; t=time.time()',
  'while time.time()-t<dur:',
  ' ln=ser.readline()',
  ' if not ln: continue',
  " s=ln.decode('utf-8','replace')",
  ' n+=1',
  " if 'CSI cb' in s or 'csi_collector' in s: csi+=1",
  " if 'MGMT+DATA' in s: print('UPGRADE_MGMT_DATA')",
  'ser.close()',
  "print(f'LINES={n} CSI={csi}')",
].join('\n');

/**
 * The tool registry. Each entry: { title, description, inputSchema, handler }.
 * inputSchema is JSON-Schema (object). handler(args) → JSON-serializable result
 * (sync or promise). Canonical names are underscore-form.
 */
export const TOOLS = {
  ruview_onboard: {
    title: 'Onboard',
    description: 'Pick a RuView setup path (docker-demo | repo-build | live-esp32) and print the next concrete command.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string', enum: Object.keys(ONBOARD_PATHS), description: 'Which setup path. Omit to list all.' } },
    },
    handler(args = {}) {
      const repo = findRepoRoot();
      if (args.path && ONBOARD_PATHS[args.path]) {
        return { ok: true, path: args.path, next: ONBOARD_PATHS[args.path], in_ruview_repo: !!repo };
      }
      return {
        ok: true,
        in_ruview_repo: !!repo,
        repo_root: repo,
        paths: ONBOARD_PATHS,
        recommend: repo ? 'repo-build' : 'docker-demo',
        note: 'WiFi sensing infers coarse pose/presence from CSI — it is not a camera. Accuracy claims must be MEASURED vs a baseline (run `ruview_claim_check`).',
      };
    },
  },

  ruview_claim_check: {
    title: 'Claim check',
    description: 'Static lint: scan text for untagged or overstated accuracy claims (the "prove everything" guardrail). Returns findings. Fail-closed: empty input is an error, not a pass.',
    inputSchema: {
      type: 'object',
      required: ['text'],
      properties: { text: { type: 'string', description: 'The text to lint (a report, README section, PR body, model card).' } },
    },
    handler(args = {}) {
      const text = typeof args.text === 'string' ? args.text : '';
      if (text.trim().length === 0) {
        return { ok: false, reason: 'empty_text', hint: 'Pass the text to lint — an empty input must not pass an honesty gate.' };
      }
      const result = claimCheck(text);
      return { ...result, summary: summarize(result) };
    },
  },

  ruview_verify: {
    title: 'Verify (witness)',
    description: 'Run the deterministic proof (archive/v1/data/proof/verify.py) and report VERDICT. Fail-closed if not in a RuView repo or python is missing.',
    inputSchema: {
      type: 'object',
      properties: { repo: { type: 'string', description: 'RuView repo root. Default: auto-detect from cwd.' } },
    },
    async handler(args = {}) {
      const repo = args.repo ? resolve(args.repo) : findRepoRoot();
      if (!repo) return { ok: false, reason: 'not_in_ruview_repo', hint: 'Run inside the RuView monorepo or pass {repo}.' };
      const proof = join(repo, 'archive', 'v1', 'data', 'proof', 'verify.py');
      if (!existsSync(proof)) return { ok: false, reason: 'proof_missing', path: proof };
      const py = which('python') || which('python3');
      if (!py) return { ok: false, reason: 'python_missing', hint: 'Install python to run the deterministic proof.' };
      const r = await run(py, [proof], { cwd: repo, timeout: 180000 });
      const verdict = /VERDICT:\s*PASS/i.test(r.stdout) ? 'PASS' : (/VERDICT:\s*FAIL/i.test(r.stdout) ? 'FAIL' : 'UNKNOWN');
      return { ok: r.ok && verdict === 'PASS', verdict, exit: r.status, tail: r.stdout.slice(-1200), stderr: r.stderr.slice(-400) };
    },
  },

  ruview_node_monitor: {
    title: 'Node monitor',
    description: 'Open an ESP32 serial port and assert CSI is flowing (MGMT+DATA). Fail-closed if python+pyserial or the port is absent. Read-only.',
    inputSchema: {
      type: 'object',
      properties: {
        port: { type: 'string', description: 'Serial port, e.g. COM8 or /dev/ttyUSB0.' },
        seconds: { type: 'number', description: 'Capture window (default 12).' },
      },
    },
    async handler(args = {}) {
      const port = args.port;
      if (!port || typeof port !== 'string') return { ok: false, reason: 'no_port', hint: 'Pass {port} (e.g. COM8).' };
      const py = which('python') || which('python3');
      if (!py) return { ok: false, reason: 'python_missing' };
      const dur = Number(args.seconds) > 0 ? Number(args.seconds) : 12;
      const r = await run(py, ['-c', MONITOR_SCRIPT, port, String(dur)], { timeout: (dur + 10) * 1000 });
      if (r.stdout.includes('NO_PYSERIAL')) return { ok: false, reason: 'pyserial_missing', hint: 'pip install pyserial' };
      if (!r.ok) return { ok: false, reason: 'port_error', stderr: r.stderr, error: r.error };
      const csi = Number((r.stdout.match(/CSI=(\d+)/) || [])[1] || 0);
      const upgraded = r.stdout.includes('UPGRADE_MGMT_DATA');
      return { ok: csi > 0, csi_callbacks: csi, mgmt_data_upgrade: upgraded, raw: r.stdout.trim() };
    },
  },

  ruview_calibrate: {
    title: 'Calibrate room',
    description: 'Run the ADR-151 room pipeline via the wifi-densepose CLI (baseline→enroll→train-room). Fail-closed if the binary is absent.',
    inputSchema: {
      type: 'object',
      properties: {
        step: { type: 'string', enum: ['baseline', 'enroll', 'train-room', 'room-watch'], description: 'Which calibration step.' },
        args: { type: 'array', items: { type: 'string' }, description: 'Extra CLI args passed through.' },
      },
    },
    async handler(args = {}) {
      const step = args.step || 'baseline';
      const bin = which('wifi-densepose');
      const repo = findRepoRoot();
      if (!bin && !repo) return { ok: false, reason: 'cli_missing', hint: 'Install the wifi-densepose CLI or run in the repo (cargo run -p wifi-densepose-cli).' };
      const passthru = Array.isArray(args.args) ? args.args.map(String) : [];
      // Prefer the installed binary; otherwise cargo-run from the repo.
      const r = bin
        ? await run(bin, [step, ...passthru], { timeout: 300000 })
        : await run('cargo', ['run', '-q', '-p', 'wifi-densepose-cli', '--', step, ...passthru], { cwd: repo, timeout: 600000 });
      return { ok: r.ok, step, via: bin ? 'binary' : 'cargo', exit: r.status, tail: r.stdout.slice(-1500), stderr: r.stderr.slice(-500) };
    },
  },

  ruview_node_flash: {
    title: 'Node flash',
    description: 'Build+flash an ESP32 firmware variant. MUTATING + hardware. Fail-closed off-Windows or without ESP-IDF. Never claims hardware validation without a boot log.',
    inputSchema: {
      type: 'object',
      properties: {
        port: { type: 'string', description: 'Target port, e.g. COM8.' },
        variant: { type: 'string', enum: ['s3-8mb', 's3-4mb', 'c6'], description: 'Firmware variant.' },
        confirm: { type: 'boolean', description: 'Must be true to actually flash (guard).' },
      },
    },
    handler(args = {}) {
      if (process.platform !== 'win32') {
        return { ok: false, reason: 'unsupported_platform', detail: 'The ESP-IDF flash flow is Windows-subprocess-specific today (see CLAUDE.local.md).' };
      }
      if (!args.confirm) {
        return { ok: false, reason: 'not_confirmed', detail: 'Mutating hardware op — re-call with {confirm:true}.', would_flash: { port: args.port, variant: args.variant || 's3-8mb' } };
      }
      return { ok: false, reason: 'manual_step_required', detail: 'Flashing uses the pinned ESP-IDF subprocess in CLAUDE.local.md. This tool returns the exact command rather than running an unattended flash.', see: 'skills/provision-node.md' };
    },
  },
};

// Historical dotted names (pre-ADR-263) accepted as call-time aliases; the
// underscore form is what tools/list advertises.
export const TOOL_ALIASES = Object.fromEntries(
  Object.keys(TOOLS).map((name) => [name.replace(/_/, '.'), name])
);

/** Resolve a canonical or aliased tool name (or null). */
export function resolveToolName(name) {
  if (TOOLS[name]) return name;
  if (TOOL_ALIASES[name]) return TOOL_ALIASES[name];
  return null;
}

/** Run one tool by name (canonical or dotted alias); always resolves to the structured result. */
export async function runTool(name, args) {
  const canonical = resolveToolName(name);
  if (!canonical) return { ok: false, reason: 'unknown_tool', name, available: Object.keys(TOOLS) };
  try {
    return await TOOLS[canonical].handler(args || {});
  } catch (err) {
    return { ok: false, reason: 'tool_threw', name: canonical, error: String(err && err.message || err) };
  }
}

/** MCP-shaped tool list: [{name, description, inputSchema}]. */
export function listTools() {
  return Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.inputSchema }));
}
