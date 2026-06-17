// SPDX-License-Identifier: MIT
// RuView harness — minimal MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
//
// Dependency-free on purpose: a published `npx ruview` must `mcp start` without
// pulling the full MCP SDK. Implements the subset hosts use: `initialize`,
// `tools/list`, `tools/call`, and the `notifications/initialized` ack. Logs go to
// stderr ONLY — stdout is the JSON-RPC channel and must stay clean.

import { createInterface } from 'node:readline';
import { listTools, runTool } from './tools.js';

const PROTOCOL_VERSION = '2024-11-05';
const SERVER_INFO = { name: 'ruview', version: '0.1.0' };

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}
function result(id, res) { send({ jsonrpc: '2.0', id, result: res }); }
function error(id, code, message) { send({ jsonrpc: '2.0', id, error: { code, message } }); }
function log(...a) { process.stderr.write('[ruview-mcp] ' + a.join(' ') + '\n'); }

function handle(msg) {
  const { id, method, params } = msg;
  switch (method) {
    case 'initialize':
      return result(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER_INFO,
        instructions: 'RuView WiFi-sensing operator tools. All results are fail-closed; accuracy claims must pass ruview.claim_check.',
      });
    case 'notifications/initialized':
    case 'initialized':
      return; // notification — no response
    case 'ping':
      return result(id, {});
    case 'tools/list':
      return result(id, { tools: listTools() });
    case 'tools/call': {
      const name = params?.name;
      const args = params?.arguments || {};
      const out = runTool(name, args);
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
  log(`starting (protocol ${PROTOCOL_VERSION}, ${listTools().length} tools)`);
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on('line', (line) => {
    const s = line.trim();
    if (!s) return;
    let msg;
    try { msg = JSON.parse(s); } catch { return log('bad JSON line dropped'); }
    try { handle(msg); } catch (err) {
      if (msg && msg.id !== undefined) error(msg.id, -32603, String(err && err.message || err));
      log('handler error:', String(err));
    }
  });
  rl.on('close', () => { log('stdin closed — exiting'); process.exit(0); });
}
