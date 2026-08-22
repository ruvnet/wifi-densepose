import { createServer, type RequestListener } from "node:http";
import { sensingGet } from "../src/http.js";

async function withServer(
  handler: RequestListener,
  test: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("test server did not bind");
  try {
    await test(`http://127.0.0.1:${address.port}`);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

describe("sensingGet response bounds", () => {
  it("accepts a small JSON response", async () => {
    await withServer((_req, res) => {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ ok: true }));
    }, async (baseUrl) => {
      const result = await sensingGet<{ ok: boolean }>(baseUrl, "/health", undefined);
      expect(result).toEqual({ ok: true, data: { ok: true } });
    });
  });

  it("rejects a chunked response larger than one MiB", async () => {
    await withServer((_req, res) => {
      res.setHeader("Content-Type", "application/json");
      res.write(`{"data":"${"x".repeat(700_000)}`);
      res.end(`${"x".repeat(700_000)}"}`);
    }, async (baseUrl) => {
      const result = await sensingGet(baseUrl, "/large", undefined);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.error).toContain("exceeds 1048576 bytes");
    });
  });
});
