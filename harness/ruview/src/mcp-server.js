// SPDX-License-Identifier: MIT
// RuView harness — minimal MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
//
// Dependency-free on purpose: a published `npx ruview` must `mcp start` without
// pulling the full MCP SDK. Implements the subset hosts use: `initialize`,
// `tools/list`, `tools/call`, `ping`, empty `resources/list`/`prompts/list`
// stubs, and the `notifications/initialized` ack. Logs go to stderr ONLY —
// stdout is the JSON-RPC channel and must stay clean.
//
// ADR-263 O2: `tools/call` is dispatched asynchronously — a long-running
// verify/calibrate no longer blocks ping/tools/list, so hosts that health-check
// mid-run see a live server. Responses may therefore arrive out of request
// order, which JSON-RPC permits (ids correlate them).

import { createInterface } from 'node:readline';
import { readFileSync } from 'node:fs';
import { listTools, runTool } from './tools.js';

const PROTOCOL_VERSION = '2024-11-05';
// Single-source the version from package.json (ADR-263 O6).
const PKG = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const SERVER_INFO = { name: 'ruview', version: PKG.version };

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}
function result(id, res) { send({ jsonrpc: '2.0', id, result: res }); }
function error(id, code, message) { send({ jsonrpc: '2.0', id, error: { code, message } }); }
function log(...a) { process.stderr.write('[ruview-mcp] ' + a.join(' ') + '\n'); }

async function handle(msg) {
  const { id, method, params } = msg;
  switch (method) {
    case 'initialize':
      return result(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER_INFO,
        instructions: 'RuView WiFi-sensing operator tools. All results are fail-closed; accuracy claims must pass ruview_claim_check.',
      });
    case 'notifications/initialized':
    case 'initialized':
    case 'notifications/cancelled':
      return; // notifications — no response
    case 'ping':
      return result(id, {});
    case 'tools/list':
      return result(id, { tools: listTools() });
    case 'resources/list':
      return result(id, { resources: [] });
    case 'prompts/list':
      return result(id, { prompts: [] });
    case 'tools/call': {
      const name = params?.name;
      const args = params?.arguments || {};
      const out = await runTool(name, args);
      // MCP content envelope: text block with the JSON, isError reflects ok=false.
      return result(id, {
        content: [{ type: 'text', text: JSON.stringify(out, null, 2) }],
        isError: out && out.ok === false,
      });
    }
    default:
      if (id !== undefined) error(id, -32601, `Method not found: ${method}`);
  }
}

export function startMcpServer() {
  log(`starting v${SERVER_INFO.version} (protocol ${PROTOCOL_VERSION}, ${listTools().length} tools)`);
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

  // tools/call runs are serialized through a FIFO promise chain: hardware/mutating
  // tools (calibrate, serial monitor, flash) must never overlap. ping/tools/list/
  // initialize/resources/prompts stay immediate (ADR-263 O2 — a health check must
  // answer during a long tool run). `toolChain` also lets stdin-close drain the
  // in-flight call so its response is flushed instead of dropped by process.exit.
  let toolChain = Promise.resolve();

  const dispatch = (msg) => handle(msg).catch((err) => {
    if (msg && msg.id !== undefined) error(msg.id, -32603, String(err && err.message || err));
    log('handler error:', String(err));
  });

  rl.on('line', (line) => {
    const s = line.trim();
    if (!s) return;
    let msg;
    try { msg = JSON.parse(s); } catch { return log('bad JSON line dropped'); }
    if (msg && msg.method === 'tools/call') {
      toolChain = toolChain.then(() => dispatch(msg)); // one tool at a time
    } else {
      dispatch(msg); // health/list/handshake answer immediately, even mid tool run
    }
  });

  rl.on('close', () => {
    // Wait for any queued/in-flight tool call to settle (its response written)
    // before exiting — fire-and-forget used to race this and drop the response.
    toolChain.then(() => {
      log('stdin closed — exiting');
      const done = () => process.exit(0);
      // Pipe writes are async; flush buffered stdout before exit.
      if (process.stdout.writableLength) process.stdout.once('drain', done);
      else done();
    });
  });
}
