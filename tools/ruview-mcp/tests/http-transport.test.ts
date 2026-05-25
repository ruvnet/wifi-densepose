/**
 * ADR-124 §3 Architecture — Streamable HTTP transport security tests.
 *
 * Tests the Origin-validation middleware and bearer-token auth gate.
 * No live MCP server needed for the guard logic — buildHttpApp is tested
 * with a minimal stub McpServer that never actually processes JSON-RPC.
 *
 * Covered:
 *   1. isOriginAllowed() unit tests — the pure function driving the gate
 *   2. POST /mcp with cross-origin Origin → 403
 *   3. POST /mcp with allowed Origin → passes Origin gate (non-403)
 *   4. POST /mcp with no Origin header → passes Origin gate (non-403)
 *   5. Bearer token required, wrong token → 401
 *   6. Bearer token required, correct token + wildcard origin → passes (non-401)
 */

import * as http from "node:http";
import { isOriginAllowed, buildHttpApp } from "../src/http-transport.js";
import { Server as McpServer } from "@modelcontextprotocol/sdk/server/index.js";

// ── helpers ────────────────────────────────────────────────────────────────

function makeMockMcpServer(): McpServer {
  return new McpServer(
    { name: "test-rvagent", version: "0.0.0" },
    { capabilities: { tools: {} } }
  );
}

async function post(
  port: number,
  path: string,
  headers: Record<string, string>,
  body: string
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        method: "POST",
        path,
        headers: { "Content-Type": "application/json", ...headers },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk: Buffer) => { data += chunk.toString(); });
        res.on("end", () => resolve({ status: res.statusCode ?? 0, body: data }));
      }
    );
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

async function startServer(
  opts: Parameters<typeof buildHttpApp>[1],
  basePort: number
): Promise<{ port: number; close: () => Promise<void> }> {
  const port = basePort + Math.floor(Math.random() * 100);
  const { httpServer } = buildHttpApp(makeMockMcpServer(), opts);
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(port, "127.0.0.1", () => resolve());
  });
  const close = () =>
    new Promise<void>((res, rej) =>
      httpServer.close((e) => (e ? rej(e) : res()))
    );
  return { port, close };
}

const MCP_BODY = JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" });

// ── 1. isOriginAllowed unit tests ──────────────────────────────────────────

describe("isOriginAllowed()", () => {
  const allow = ["http://localhost", "http://127.0.0.1"];

  it("allows undefined origin (non-browser request, no Origin header)", () => {
    expect(isOriginAllowed(undefined, allow)).toBe(true);
  });

  it("allows an origin in the allowlist", () => {
    expect(isOriginAllowed("http://localhost", allow)).toBe(true);
    expect(isOriginAllowed("http://127.0.0.1", allow)).toBe(true);
  });

  it("rejects an origin NOT in the allowlist", () => {
    expect(isOriginAllowed("https://evil.example.com", allow)).toBe(false);
  });

  it("allows anything when allowedOrigins includes '*'", () => {
    expect(isOriginAllowed("https://evil.example.com", ["*"])).toBe(true);
  });

  it("is case-sensitive per RFC 6454", () => {
    expect(isOriginAllowed("HTTP://localhost", allow)).toBe(false);
  });
});

// ── 2-4. Origin-validation integration tests ───────────────────────────────

describe("HTTP transport Origin-validation middleware", () => {
  let port: number;
  let close: () => Promise<void>;

  beforeAll(async () => {
    const srv = await startServer(
      { allowedOrigins: ["http://localhost", "http://127.0.0.1"] },
      49200
    );
    port = srv.port;
    close = srv.close;
  });

  afterAll(async () => { await close(); });

  it("rejects cross-origin POST /mcp with 403", async () => {
    const r = await post(port, "/mcp", { Origin: "https://evil.example.com" }, MCP_BODY);
    expect(r.status).toBe(403);
    const body = JSON.parse(r.body) as Record<string, unknown>;
    expect(body["error"]).toMatch(/cross-origin/i);
  });

  it("passes Origin gate for http://localhost — status is not 403", async () => {
    const r = await post(port, "/mcp", { Origin: "http://localhost" }, MCP_BODY);
    expect(r.status).not.toBe(403);
  });

  it("passes Origin gate with no Origin header — status is not 403", async () => {
    const r = await post(port, "/mcp", {}, MCP_BODY);
    expect(r.status).not.toBe(403);
  });
});

// ── 5-6. Bearer-token auth integration tests ──────────────────────────────

describe("HTTP transport bearer-token auth gate", () => {
  const SECRET = "test-secret-token-xyz";
  let port: number;
  let close: () => Promise<void>;

  beforeAll(async () => {
    const srv = await startServer({ allowedOrigins: ["*"], bearerToken: SECRET }, 49400);
    port = srv.port;
    close = srv.close;
  });

  afterAll(async () => { await close(); });

  it("rejects missing Authorization header with 401", async () => {
    const r = await post(port, "/mcp", {}, MCP_BODY);
    expect(r.status).toBe(401);
  });

  it("rejects wrong bearer token with 401", async () => {
    const r = await post(port, "/mcp", { Authorization: "Bearer wrong" }, MCP_BODY);
    expect(r.status).toBe(401);
  });

  it("passes auth gate with correct bearer token — status is not 401", async () => {
    const r = await post(port, "/mcp", { Authorization: `Bearer ${SECRET}` }, MCP_BODY);
    expect(r.status).not.toBe(401);
  });
});
