jest.mock('expo-web-browser', () => ({ __esModule: true, maybeCompleteAuthSession: jest.fn(), openBrowserAsync: jest.fn(), WebBrowserPresentationStyle: { FORM_SHEET: 'formSheet' } }));
jest.mock('expo-crypto', () => ({ getRandomBytesAsync: jest.fn(), digestStringAsync: jest.fn(), CryptoDigestAlgorithm: { SHA256: 'SHA256' }, CryptoEncoding: { BASE64: 'base64' } }));
jest.mock('expo-secure-store', () => ({ getItemAsync: jest.fn(), setItemAsync: jest.fn(), deleteItemAsync: jest.fn(), WHEN_UNLOCKED_THIS_DEVICE_ONLY: 1 }));

import * as Crypto from 'expo-crypto';
import * as WebBrowser from 'expo-web-browser';
import { cognitumService, parseSpatialInsight, parseSpatialPage } from '@/services/cognitum.service';

const mockOpenBrowser = WebBrowser.openBrowserAsync as jest.Mock;

const exclusions = ['raw_csi', 'cir', 'rf_tensors', 'recordings', 'pose_frames', 'vital_waveforms', 'identity_observations'];
const page = {
  object: 'list', kind: 'spaces', schemaVersion: '1.0', nextCursor: null,
  boundary: { authoritativeState: 'HomeCore Edge', cloudRole: 'semantic context only', excluded: exclusions },
  data: [{ id: 'space-1', kind: 'spaces', privacy: 'P2', observedAt: '2026-08-24T12:00:00Z', name: 'Lab', confidence: .82, attributes: {}, provenance: {} }],
};

describe('Cognitum spatial boundary validation', () => {
  beforeEach(() => { jest.clearAllMocks(); });

  it('accepts a bounded semantic Spaces page', () => {
    const result = parseSpatialPage(page, 'spaces');
    expect(result.data[0].name).toBe('Lab');
    expect(result.boundary.authoritativeState).toBe('HomeCore Edge');
  });

  it('fails closed when a raw-sensing exclusion is absent', () => {
    const unsafe = { ...page, boundary: { ...page.boundary, excluded: exclusions.filter((item) => item !== 'raw_csi') } };
    expect(() => parseSpatialPage(unsafe, 'spaces')).toThrow('privacy boundary');
  });

  it('rejects identity-class and malformed resource data', () => {
    expect(() => parseSpatialPage({ ...page, data: [{ ...page.data[0], privacy: 'P1' }] }, 'spaces')).toThrow('resource failed schema validation');
    expect(() => parseSpatialPage({ ...page, data: new Array(101).fill(page.data[0]) }, 'spaces')).toThrow('schema validation');
  });

  it('uses Cognitum OOB PKCE and exchanges the one-time code', async () => {
    (Crypto.getRandomBytesAsync as jest.Mock).mockResolvedValue(new Uint8Array(48).fill(7));
    (Crypto.digestStringAsync as jest.Mock).mockResolvedValue('Y2hhbGxlbmdl');
    mockOpenBrowser.mockResolvedValue({ type: 'opened' });
    await cognitumService.beginSpacesSignIn();
    const authorizeUrl = new URL(mockOpenBrowser.mock.calls[0][0]);
    expect(authorizeUrl.searchParams.get('redirect_uri')).toBe('urn:ietf:wg:oauth:2.0:oob');
    expect(authorizeUrl.searchParams.get('scope')).toBe('sensing:read spaces:read');

    const token = ['x', btoa(JSON.stringify({ scope: 'sensing:read spaces:read', account_id: 'acct-1' })), 'y'].join('.');
    global.fetch = jest.fn().mockResolvedValue(new Response(JSON.stringify({ access_token: token, expires_in: 900, scope: 'sensing:read spaces:read' }), { status: 200 }));
    const result = await cognitumService.completeSignIn('one-time-code');
    expect(global.fetch).toHaveBeenCalledWith('https://auth.cognitum.one/v1/oauth/code-exchange', expect.objectContaining({ method: 'POST' }));
    expect(result.kind).toBe('spaces');
    expect(result.session.signedIn).toBe(true);
    expect(result.session.scopes).toContain('spaces:read');
  });

  it('uses Cognitum’s registered Meta-LLM client and inference scope', async () => {
    (Crypto.getRandomBytesAsync as jest.Mock).mockResolvedValue(new Uint8Array(48).fill(9));
    (Crypto.digestStringAsync as jest.Mock).mockResolvedValue('aW5mZXJlbmNl');
    mockOpenBrowser.mockResolvedValue({ type: 'opened' });
    await cognitumService.beginCloudSignIn();
    const authorizeUrl = new URL(mockOpenBrowser.mock.calls[0][0]);
    expect(authorizeUrl.searchParams.get('client_id')).toBe('meta-proxy');
    expect(authorizeUrl.searchParams.get('scope')).toBe('inference');
    expect(authorizeUrl.searchParams.get('redirect_uri')).toBe('urn:ietf:wg:oauth:2.0:oob');
  });

  it('preserves the governed Meta-LLM routing receipt', () => {
    const insight = parseSpatialInsight({
      model: 'cognitum-auto',
      choices: [{ message: { role: 'assistant', content: 'Two anonymous occupied zones; confidence remains bounded.' } }],
      usage: { prompt_tokens: 84, completion_tokens: 17 },
      x_cognitum: { request_id: 'req-1', resolved_tier: 'low', resolved_model: 'glm-5.2', escalated: false, price_usd: 0.0012 },
    });
    expect(insight).toMatchObject({ model: 'glm-5.2', requestId: 'req-1', resolvedTier: 'low', escalated: false, priceUsd: 0.0012, promptTokens: 84, completionTokens: 17 });
  });
});
