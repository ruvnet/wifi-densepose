import { Platform } from 'react-native';
import * as Crypto from 'expo-crypto';
import * as SecureStore from 'expo-secure-store';
import * as WebBrowser from 'expo-web-browser';
import type {
  CognitumBoundary,
  CognitumSessionSummary,
  CognitumSpatialPage,
  CognitumSpatialResource,
  SpatialInsight,
  SpatialIntelligenceInput,
  SpatialKind,
} from '@/types/cognitum';

WebBrowser.maybeCompleteAuthSession();

const AUTH_ORIGIN = 'https://auth.cognitum.one';
const API_ORIGIN = 'https://api.cognitum.one';
const SPACES_CLIENT_ID = 'ruview';
const INFERENCE_CLIENT_ID = 'meta-proxy';
const OOB_REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob';
const SPACES_SESSION_KEY = 'ruview.cognitum.oauth.v1';
const INFERENCE_SESSION_KEY = 'ruview.cognitum.inference.oauth.v1';
const REQUIRED_EXCLUSIONS = ['raw_csi', 'cir', 'rf_tensors', 'recordings', 'pose_frames', 'vital_waveforms', 'identity_observations'];
const SPACES_SCOPE = ['sensing:read', 'spaces:read'];
const INFERENCE_SCOPE = ['inference'];
const MAX_RESPONSE_BYTES = 1024 * 1024;

interface TokenSession {
  accessToken: string;
  refreshToken?: string;
  expiresAt: number;
  scope: string;
  accountId?: string;
  workspaceId?: string;
}

type SessionKind = 'spaces' | 'inference';
const memorySessions: Record<SessionKind, TokenSession | null> = { spaces: null, inference: null };
let pendingAuthorization: { verifier: string; scopes: string[]; clientId: string; kind: SessionKind } | null = null;

const sessionKey = (kind: SessionKind): string => kind === 'spaces' ? SPACES_SESSION_KEY : INFERENCE_SESSION_KEY;
const clientId = (kind: SessionKind): string => kind === 'spaces' ? SPACES_CLIENT_ID : INFERENCE_CLIENT_ID;

const base64Url = (bytes: Uint8Array): string => {
  let binary = '';
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return globalThis.btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
};

const randomValue = async (bytes = 32): Promise<string> => base64Url(await Crypto.getRandomBytesAsync(bytes));
const hasScope = (scope: string, required: string): boolean => scope.split(/\s+/).includes(required);

const decodeClaims = (token: string): Record<string, unknown> => {
  try {
    const part = token.split('.')[1];
    if (!part) return {};
    const padded = part.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(part.length / 4) * 4, '=');
    return JSON.parse(globalThis.atob(padded)) as Record<string, unknown>;
  } catch { return {}; }
};

const sessionStorageAvailable = (): boolean => Platform.OS === 'web' && typeof globalThis.sessionStorage !== 'undefined';
const readStored = async (kind: SessionKind): Promise<TokenSession | null> => {
  try {
    const raw = sessionStorageAvailable() ? globalThis.sessionStorage.getItem(sessionKey(kind)) : await SecureStore.getItemAsync(sessionKey(kind));
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<TokenSession>;
    if (typeof value.accessToken !== 'string' || typeof value.expiresAt !== 'number' || typeof value.scope !== 'string') return null;
    return value as TokenSession;
  } catch { return null; }
};
const writeStored = async (kind: SessionKind, session: TokenSession | null): Promise<void> => {
  if (sessionStorageAvailable()) {
    if (session) globalThis.sessionStorage.setItem(sessionKey(kind), JSON.stringify(session));
    else globalThis.sessionStorage.removeItem(sessionKey(kind));
    return;
  }
  if (session) await SecureStore.setItemAsync(sessionKey(kind), JSON.stringify(session), { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY });
  else await SecureStore.deleteItemAsync(sessionKey(kind));
};

const consumeTokenResponse = async (response: Response, kind: SessionKind, requiredScopes: string[] = []): Promise<TokenSession> => {
  const text = await response.text();
  let value: Record<string, unknown>;
  try { value = JSON.parse(text) as Record<string, unknown>; } catch { throw new Error(`Cognitum OAuth returned HTTP ${response.status} with invalid JSON`); }
  if (!response.ok || typeof value.access_token !== 'string') {
    throw new Error(typeof value.error_description === 'string' ? value.error_description : `Cognitum OAuth returned HTTP ${response.status}`);
  }
  const claims = decodeClaims(value.access_token);
  const expiresIn = typeof value.expires_in === 'number' && value.expires_in > 0 ? Math.min(value.expires_in, 3600) : 900;
  const session: TokenSession = {
    accessToken: value.access_token,
    refreshToken: typeof value.refresh_token === 'string' ? value.refresh_token : undefined,
    expiresAt: Date.now() + expiresIn * 1000,
    scope: typeof value.scope === 'string' ? value.scope : typeof claims.scope === 'string' ? claims.scope : '',
    accountId: typeof claims.account_id === 'string' ? claims.account_id : undefined,
    workspaceId: typeof claims.workspace_id === 'string' ? claims.workspace_id : undefined,
  };
  if (requiredScopes.some((required) => !hasScope(session.scope, required))) throw new Error('Cognitum granted fewer scopes than the requested capability requires');
  memorySessions[kind] = session;
  await writeStored(kind, session);
  return session;
};

const tokenRequest = async (kind: SessionKind, parameters: Record<string, string>): Promise<TokenSession> => consumeTokenResponse(await fetch(`${AUTH_ORIGIN}/oauth/token`, {
  method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams(parameters).toString(),
}), kind);

const beginAuthorization = async (kind: SessionKind, scopes: string[]): Promise<void> => {
  const verifier = await randomValue(48);
  const state = await randomValue(24);
  const digest = await Crypto.digestStringAsync(Crypto.CryptoDigestAlgorithm.SHA256, verifier, { encoding: Crypto.CryptoEncoding.BASE64 });
  const challenge = digest.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  const url = new URL(`${AUTH_ORIGIN}/oauth/authorize`);
  url.searchParams.set('response_type', 'code');
  url.searchParams.set('client_id', clientId(kind));
  url.searchParams.set('redirect_uri', OOB_REDIRECT_URI);
  url.searchParams.set('state', state);
  url.searchParams.set('code_challenge', challenge);
  url.searchParams.set('code_challenge_method', 'S256');
  url.searchParams.set('scope', scopes.join(' '));
  pendingAuthorization = { verifier, scopes, clientId: clientId(kind), kind };
  await WebBrowser.openBrowserAsync(url.toString(), { presentationStyle: WebBrowser.WebBrowserPresentationStyle.FORM_SHEET });
};

const completeAuthorization = async (rawCode: string): Promise<{ kind: SessionKind; session: TokenSession }> => {
  const code = rawCode.trim();
  const pending = pendingAuthorization;
  if (!pending) throw new Error('Start Cognitum authorization before entering a one-time code');
  if (!code || code.length > 4096 || /\s/.test(code)) throw new Error('Enter the one-time Cognitum authorization code without spaces');
  const response = await fetch(`${AUTH_ORIGIN}/v1/oauth/code-exchange`, {
    method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, code_verifier: pending.verifier, client_id: pending.clientId }),
  });
  const session = await consumeTokenResponse(response, pending.kind, pending.scopes);
  const kind = pending.kind;
  pendingAuthorization = null;
  return { kind, session };
};

const refresh = async (kind: SessionKind, session: TokenSession): Promise<TokenSession> => {
  if (!session.refreshToken) throw new Error('Cognitum session expired; sign in again');
  const next = await tokenRequest(kind, { grant_type: 'refresh_token', refresh_token: session.refreshToken, client_id: clientId(kind) });
  if (!next.refreshToken) {
    next.refreshToken = session.refreshToken;
    await writeStored(kind, next);
  }
  return next;
};

const accessToken = async (kind: SessionKind, requiredScope: string): Promise<string> => {
  let session = memorySessions[kind] ?? await readStored(kind);
  if (!session) throw new Error('Sign in with Cognitum to continue');
  if (session.expiresAt <= Date.now() + 60_000) session = await refresh(kind, session);
  if (!hasScope(session.scope, requiredScope)) throw new Error(`Cognitum authorization is missing ${requiredScope}`);
  memorySessions[kind] = session;
  return session.accessToken;
};

const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const apiError = (value: unknown, fallback: string): string => {
  if (!isRecord(value)) return fallback;
  if (typeof value.error === 'string') return value.error;
  if (typeof value.message === 'string') return value.message;
  if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
  return fallback;
};
const boundedJson = async (response: Response): Promise<unknown> => {
  const text = await response.text();
  if (text.length > MAX_RESPONSE_BYTES) throw new Error('Cognitum response exceeded the 1 MB safety limit');
  try { return JSON.parse(text) as unknown; } catch { throw new Error(`Cognitum returned HTTP ${response.status} with invalid JSON`); }
};

const parseBoundary = (value: unknown): CognitumBoundary => {
  if (!isRecord(value) || typeof value.authoritativeState !== 'string' || typeof value.cloudRole !== 'string' || !Array.isArray(value.excluded) || !value.excluded.every((item) => typeof item === 'string')) throw new Error('Cognitum Spaces boundary is invalid');
  if (value.authoritativeState !== 'HomeCore Edge' || REQUIRED_EXCLUSIONS.some((item) => !(value.excluded as string[]).includes(item))) throw new Error('Cognitum Spaces did not preserve the required edge privacy boundary');
  return value as unknown as CognitumBoundary;
};

export const parseSpatialPage = (value: unknown, kind: SpatialKind): CognitumSpatialPage => {
  if (!isRecord(value) || value.object !== 'list' || value.kind !== kind || value.schemaVersion !== '1.0' || !Array.isArray(value.data) || value.data.length > 100) throw new Error(`Cognitum ${kind} response failed schema validation`);
  const data = value.data.map((item): CognitumSpatialResource => {
    if (!isRecord(item) || typeof item.id !== 'string' || item.kind !== kind || (item.privacy !== 'P2' && item.privacy !== 'P3') || typeof item.observedAt !== 'string' || !isRecord(item.attributes) || !isRecord(item.provenance)) throw new Error(`Cognitum ${kind} resource failed schema validation`);
    if (typeof item.confidence === 'number' && (!Number.isFinite(item.confidence) || item.confidence < 0 || item.confidence > 1)) throw new Error(`Cognitum ${kind} confidence is invalid`);
    return item as unknown as CognitumSpatialResource;
  });
  return { kind, data, boundary: parseBoundary(value.boundary) };
};

export const parseSpatialInsight = (value: unknown): SpatialInsight => {
  if (!isRecord(value) || !Array.isArray(value.choices) || !isRecord(value.choices[0]) || !isRecord(value.choices[0].message) || typeof value.choices[0].message.content !== 'string') throw new Error('Cognitum meta-LLM response failed schema validation');
  const usage = isRecord(value.usage) ? value.usage : {};
  const receipt = isRecord(value.x_cognitum) ? value.x_cognitum : {};
  return {
    text: value.choices[0].message.content,
    model: typeof receipt.resolved_model === 'string' ? receipt.resolved_model : typeof value.model === 'string' ? value.model : 'cognitum-auto',
    provider: typeof value.provider === 'string' ? value.provider : undefined,
    generatedAt: Date.now(),
    requestId: typeof receipt.request_id === 'string' ? receipt.request_id : undefined,
    resolvedTier: typeof receipt.resolved_tier === 'string' ? receipt.resolved_tier : undefined,
    escalated: typeof receipt.escalated === 'boolean' ? receipt.escalated : undefined,
    priceUsd: typeof receipt.price_usd === 'number' && Number.isFinite(receipt.price_usd) ? receipt.price_usd : undefined,
    promptTokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : undefined,
    completionTokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : undefined,
  };
};

export const cognitumService = {
  async sessionSummary(): Promise<CognitumSessionSummary> {
    const session = memorySessions.spaces ?? await readStored('spaces');
    if (!session) return { signedIn: false, scopes: [] };
    memorySessions.spaces = session;
    return { signedIn: true, accountId: session.accountId, workspaceId: session.workspaceId, scopes: session.scope.split(/\s+/).filter(Boolean), expiresAt: session.expiresAt };
  },
  async inferenceSessionSummary(): Promise<CognitumSessionSummary> {
    const session = memorySessions.inference ?? await readStored('inference');
    if (!session) return { signedIn: false, scopes: [] };
    memorySessions.inference = session;
    return { signedIn: true, accountId: session.accountId, workspaceId: session.workspaceId, scopes: session.scope.split(/\s+/).filter(Boolean), expiresAt: session.expiresAt };
  },
  async beginSpacesSignIn(): Promise<void> { await beginAuthorization('spaces', SPACES_SCOPE); },
  async beginCloudSignIn(): Promise<void> { await beginAuthorization('inference', INFERENCE_SCOPE); },
  async completeSignIn(code: string): Promise<{ kind: SessionKind; session: CognitumSessionSummary }> {
    const result = await completeAuthorization(code);
    const session = result.kind === 'spaces' ? await this.sessionSummary() : await this.inferenceSessionSummary();
    return { kind: result.kind, session };
  },
  async signOut(): Promise<void> {
    memorySessions.spaces = null; memorySessions.inference = null; pendingAuthorization = null;
    await Promise.all([writeStored('spaces', null), writeStored('inference', null)]);
  },
  async listSpatial(kind: SpatialKind): Promise<CognitumSpatialPage> {
    const token = await accessToken('spaces', 'spaces:read');
    const response = await fetch(`${API_ORIGIN}/v1/spatial/${kind}?limit=50`, { headers: { Accept: 'application/json', Authorization: `Bearer ${token}` } });
    const value = await boundedJson(response);
    if (!response.ok) throw new Error(apiError(value, `Cognitum ${kind} returned HTTP ${response.status}`));
    return parseSpatialPage(value, kind);
  },
  async analyzeSpatial(input: SpatialIntelligenceInput): Promise<SpatialInsight> {
    const token = await accessToken('inference', 'inference');
    const response = await fetch(`${API_ORIGIN}/v1/chat/completions`, {
      method: 'POST',
      headers: { Accept: 'application/json', Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'cognitum-auto', temperature: 0.15, max_tokens: 320,
        messages: [
          { role: 'system', content: 'You are RuView spatial intelligence. Interpret only the supplied anonymous semantic aggregates. State uncertainty, distinguish local RF evidence from Cognitum semantic context, and never infer identity, exact pose, health, or events absent from the data. Return a concise operational brief.' },
          { role: 'user', content: JSON.stringify(input) },
        ],
      }),
    });
    const value = await boundedJson(response);
    if (!response.ok) throw new Error(apiError(value, `Cognitum meta-LLM returned HTTP ${response.status}`));
    return parseSpatialInsight(value);
  },
};
