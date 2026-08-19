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
const MAX_REQUEST_BYTES = 256 * 1024;
const MAX_QUEUED_TOOL_CALLS = 20;
// Single-source the version from package.json (ADR-263 O6).
const PKG = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const SERVER_INFO = { name: 'ruview', version: PKG.version };

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}
function result(id, res) { send({ jsonrpc: '2.0', id, result: res }); }
function error(id, code, message) { send({ jsonrpc: '2.0', id, error: { code, message } }); }
function log(...a) { process.stderr.write('[ruview-mcp] ' + a.join(' ') + '\n'); }

async function handle(msg, context = {}) {
  const { id, method, params } = msg;
  switch (method) {
    case 'initialize':
      return result(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER_INFO,
        instructions: 'RuView WiFi-sensing operator tools. All results are fail-closed; accuracy claims must pass ruview_claim_check. Credentialed external reads are denied without an operator grant; ruview_spaces_list requires credential-use.',
      });
    case 'notifications/initialized':
    case 'initialized':
      return; // notifications — no response
    case 'notifications/cancelled':
      if (context.queuedIds?.has(params?.requestId)) context.cancelled?.add(params.requestId);
      return; // queued requests are cancelled before execution
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
      log('audit', JSON.stringify({ event: 'tools/call', id, name }));
      const out = await runTool(name, args, context);
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
  let queuedToolCalls = 0;
  const cancelled = new Set();
  const queuedIds = new Set();

  const grants = String(process.env.RUVIEW_MCP_GRANTS || '').split(',').map((v) => v.trim()).filter(Boolean);
  const dispatch = (msg) => handle(msg, { source: 'mcp', grants, cancelled, queuedIds }).catch((err) => {
    if (msg && msg.id !== undefined) error(msg.id, -32603, String(err && err.message || err));
    log('handler error:', String(err));
  });

  rl.on('line', (line) => {
    if (Buffer.byteLength(line, 'utf8') > MAX_REQUEST_BYTES) {
      log('oversized JSON-RPC line dropped');
      return;
    }
    const s = line.trim();
    if (!s) return;
    let msg;
    try { msg = JSON.parse(s); } catch { return log('bad JSON line dropped'); }
    if (msg && msg.method === 'tools/call') {
      const validId = typeof msg.id === 'string' || (typeof msg.id === 'number' && Number.isFinite(msg.id));
      if (!validId) {
        error(msg?.id ?? null, -32600, 'tools/call requires a finite string or number id');
        return;
      }
      if (queuedIds.has(msg.id)) {
        error(msg.id, -32600, 'Duplicate in-flight request id');
        return;
      }
      if (queuedToolCalls >= MAX_QUEUED_TOOL_CALLS) {
        if (msg.id !== undefined) error(msg.id, -32000, 'Tool queue is full');
        log('tool queue full:', String(msg.id));
        return;
      }
      queuedToolCalls += 1;
      queuedIds.add(msg.id);
      toolChain = toolChain.then(async () => {
        try {
          if (cancelled.delete(msg.id)) {
            if (msg.id !== undefined) error(msg.id, -32800, 'Request cancelled');
            return;
          }
          await dispatch(msg);
        } finally {
          cancelled.delete(msg.id);
          queuedIds.delete(msg.id);
          queuedToolCalls -= 1;
        }
      }); // one tool at a time
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
