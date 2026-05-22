/**
 * Lightweight HTTP client (re-used in CLI commands).
 * Identical to tools/ruview-mcp/src/http.ts but kept separate to avoid a
 * workspace dependency — both packages are standalone and independently publishable.
 */

const REQUEST_TIMEOUT_MS = 10_000;

export type Ok<T> = { ok: true; data: T };
export type Err = { ok: false; error: string };
export type Result<T> = Ok<T> | Err;

export function ok<T>(data: T): Ok<T> {
  return { ok: true, data };
}

export function err(error: string): Err {
  return { ok: false, error };
}

export async function sensingGet<T>(
  baseUrl: string,
  path: string,
  token: string | undefined
): Promise<Result<T>> {
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) {
      return err(`HTTP ${res.status} from ${url}: ${await res.text().catch(() => "(no body)")}`);
    }
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      return err(`Non-JSON response from ${url}`);
    }
    return ok(body as T);
  } catch (e: unknown) {
    clearTimeout(timer);
    if (e instanceof Error && e.name === "AbortError") {
      return err(`Request to ${url} timed out after ${REQUEST_TIMEOUT_MS}ms`);
    }
    return err(`Network error fetching ${url}: ${String(e)}`);
  }
}
