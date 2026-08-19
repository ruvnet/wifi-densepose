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
  // Factory, not instance: each Streamable-HTTP session gets its own MCP
  // Server (ADR-264 F7/O3).
  const { httpServer } = buildHttpApp(() => makeMockMcpServer(), opts);
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

  // ADR-264 F7: real browser origins carry ports — localhost must match on
  // hostname, any port, even with an empty allowlist.
  it("allows localhost origins on any port", () => {
    expect(isOriginAllowed("http://localhost:5173", [])).toBe(true);
    expect(isOriginAllowed("http://127.0.0.1:8080", [])).toBe(true);
    expect(isOriginAllowed("https://localhost:3001", [])).toBe(true);
  });

  it("rejects non-local origins even with a localhost-looking prefix", () => {
    expect(isOriginAllowed("http://localhost.evil.example.com", [])).toBe(false);
    expect(isOriginAllowed("https://evil.example.com:443", [])).toBe(false);
  });

  // ADR-264 F7 hardening: an EXPLICIT allowlist means exact matching only. The
  // any-port-localhost convenience applies solely to the empty-allowlist case,
  // so an operator who pins an allowlist actually gets it.
  it("with an explicit allowlist, rejects a localhost origin on an unlisted port", () => {
    expect(isOriginAllowed("http://localhost:5173", allow)).toBe(false);
    expect(isOriginAllowed("http://127.0.0.1:8080", allow)).toBe(false);
  });

  it("with an explicit allowlist, still accepts an exactly-listed localhost origin", () => {
    expect(isOriginAllowed("http://localhost", allow)).toBe(true);
    expect(isOriginAllowed("http://127.0.0.1", allow)).toBe(true);
  });

  it("is case-sensitive for non-local allowlist entries per RFC 6454", () => {
    expect(isOriginAllowed("HTTPS://Partner.Example.com", ["https://partner.example.com"])).toBe(false);
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

// ── 7. ADR-264 F7/O3 hardening: body cap + per-session routing ─────────────

describe("HTTP transport session + body-cap hardening (ADR-264 F7)", () => {
  let port: number;
  let close: () => Promise<void>;

  beforeAll(async () => {
    const srv = await startServer({ allowedOrigins: ["*"], maxBodyBytes: 64 * 1024 }, 49600);
    port = srv.port;
    close = srv.close;
  });

  afterAll(async () => { await close(); });

  it("rejects oversized request bodies with 413", async () => {
    const huge = JSON.stringify({ jsonrpc: "2.0", id: 1, method: "x", params: { pad: "y".repeat(128 * 1024) } });
    const r = await post(port, "/mcp", {}, huge);
    expect(r.status).toBe(413);
  });

  it("rejects a non-initialize POST without a session id with 400 (never a shared transport)", async () => {
    const r = await post(port, "/mcp", {}, MCP_BODY); // tools/list, no mcp-session-id
    expect(r.status).toBe(400);
    const body = JSON.parse(r.body) as Record<string, unknown>;
    expect(body["error"]).toMatch(/initialize/i);
  });

  it("rejects a POST with an unknown session id with 404", async () => {
    const r = await post(port, "/mcp", { "mcp-session-id": "no-such-session" }, MCP_BODY);
    expect(r.status).toBe(404);
  });

  it("creates a fresh session (and MCP server) per initialize request", async () => {
    const init = JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test-client", version: "0.0.0" },
      },
    });
    const r = await post(port, "/mcp", { Accept: "application/json, text/event-stream" }, init);
    expect([200, 406]).not.toContain(0); // sanity
    expect(r.status).toBe(200);
  });
});

// ── 8. ADR-264 F7: session-map bounds (cap + idle TTL sweep) ───────────────

describe("HTTP transport session bounds (ADR-264 F7)", () => {
  const initBody = (id: number): string =>
    JSON.stringify({
      jsonrpc: "2.0",
      id,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test-client", version: "0.0.0" },
      },
    });

  // Build directly (not via startServer) so we can inspect the sessions map.
  async function startWithApp(
    opts: Parameters<typeof buildHttpApp>[1],
    basePort: number
  ): Promise<{
    port: number;
    sessions: ReturnType<typeof buildHttpApp>["sessions"];
    close: () => Promise<void>;
  }> {
    const { httpServer, sessions } = buildHttpApp(() => makeMockMcpServer(), opts);
    const port = basePort + Math.floor(Math.random() * 100);
    await new Promise<void>((resolve, reject) => {
      httpServer.once("error", reject);
      httpServer.listen(port, "127.0.0.1", () => resolve());
    });
    const close = () =>
      new Promise<void>((res, rej) => httpServer.close((e) => (e ? rej(e) : res())));
    return { port, sessions, close };
  }

  const ACCEPT = { Accept: "application/json, text/event-stream" };

  it("never exceeds maxSessions — evicts the oldest-idle session at capacity", async () => {
    const srv = await startWithApp({ allowedOrigins: ["*"], maxSessions: 2 }, 49800);
    try {
      for (let i = 0; i < 5; i++) {
        await post(srv.port, "/mcp", ACCEPT, initBody(i));
      }
      expect(srv.sessions.size).toBeLessThanOrEqual(2);
    } finally {
      await srv.close();
    }
  });

  it("sweeps sessions idle beyond sessionIdleMs", async () => {
    const srv = await startWithApp(
      { allowedOrigins: ["*"], sessionIdleMs: 20, sweepIntervalMs: 10 },
      49900
    );
    try {
      await post(srv.port, "/mcp", ACCEPT, initBody(1));
      expect(srv.sessions.size).toBe(1);
      await new Promise((r) => setTimeout(r, 150));
      expect(srv.sessions.size).toBe(0);
    } finally {
      await srv.close();
    }
  });
});
