/**
 * Unit tests for HomecoreClient REST methods.
 * Mocks global `fetch` and asserts correct URL + Authorization header.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { HomecoreClient } from '../api/client.js';

describe('HomecoreClient', () => {
  const token = 'test-bearer-token';
  let client: HomecoreClient;
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    client = new HomecoreClient({ token });
    fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([]),
    } as Response);
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('getStates() GETs /api/states with the bearer header', async () => {
    await client.getStates();

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('/api/states');
    expect((init.headers as Record<string, string>)['Authorization']).toBe(`Bearer ${token}`);
    expect(init.method).toBe('GET');
  });

  it('getState() GETs /api/states/:entity_id with the bearer header', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ entity_id: 'light.living', state: 'on', attributes: {}, last_changed: '', last_updated: '', context: { id: 'x', user_id: null, parent_id: null } }),
    } as Response);

    await client.getState('light.living');

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/states/light.living');
  });

  it('getConfig() GETs /api/config', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ location_name: 'Home', version: '0.1.0', state: 'RUNNING', components: [] }),
    } as Response);

    await client.getConfig();

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/config');
  });

  it('throws on non-OK response', async () => {
    fetchSpy.mockResolvedValueOnce({ ok: false, status: 401, statusText: 'Unauthorized' } as Response);

    await expect(client.getStates()).rejects.toThrow('401');
  });
});
