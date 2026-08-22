/**
 * Lightweight HTTP client for the RuView sensing-server.
 *
 * Uses Node's built-in `fetch` (available since Node 18).  All requests respect
 * the optional RUVIEW_API_TOKEN bearer header and a 10-second hard timeout.
 *
 * Failure model: every public function returns a typed `Result<T>` tuple to
 * avoid try/catch proliferation in callers.
 */

const REQUEST_TIMEOUT_MS = 10_000;
const MAX_RESPONSE_BYTES = 1024 * 1024;

export type Ok<T> = { ok: true; data: T };
export type Err = { ok: false; error: string };
export type Result<T> = Ok<T> | Err;

export function ok<T>(data: T): Ok<T> {
  return { ok: true, data };
}

export function err(error: string): Err {
  return { ok: false, error };
}

async function readResponseBody(res: Response, url: string): Promise<Result<string>> {
  const declared = Number(res.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
    await res.body?.cancel();
    return err(`Response from ${url} exceeds ${MAX_RESPONSE_BYTES} bytes`);
  }
  if (!res.body) return ok("");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let body = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_RESPONSE_BYTES) {
      await reader.cancel();
      return err(`Response from ${url} exceeds ${MAX_RESPONSE_BYTES} bytes`);
    }
    body += decoder.decode(value, { stream: true });
  }
  body += decoder.decode();
  return ok(body);
}

/**
 * Perform an authenticated GET against the sensing-server.
 */
export async function sensingGet<T>(
  baseUrl: string,
  path: string,
  token: string | undefined
): Promise<Result<T>> {
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      headers,
      signal: controller.signal,
    });
    const responseBody = await readResponseBody(res, url);
    if (!responseBody.ok) return responseBody;
    if (!res.ok) return err(`HTTP ${res.status} from ${url}: ${responseBody.data}`);

    let body: unknown;
    try {
      body = JSON.parse(responseBody.data);
    } catch {
      return err(`Non-JSON response from ${url}`);
    }

    return ok(body as T);
  } catch (e: unknown) {
    if (e instanceof Error && e.name === "AbortError") {
      return err(`Request to ${url} timed out after ${REQUEST_TIMEOUT_MS} ms`);
    }
    return err(`Network error fetching ${url}: ${String(e)}`);
  } finally {
    clearTimeout(timer);
  }
}
