import { createSyntheticNlosFrame } from '@/services/nlos.service';
import { useNlosStore } from '@/stores/nlosStore';
import { createLiveNlosFrameFixture } from '@/testUtils/nlosFixtures';
import { NLOS_STALE_AFTER_MS } from '@/types/nlos';

const NOW = 1_700_000_000_100;

describe('useNlosStore', () => {
  beforeEach(() => useNlosStore.getState().reset());

  it('accepts authenticated live frames and starts fresh', () => {
    const frame = createLiveNlosFrameFixture();
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    expect(useNlosStore.getState()).toMatchObject({ frame, freshness: 'fresh', rejectedFrameCount: 0 });
  });

  it('rejects a live frame delivered over a replay channel', () => {
    useNlosStore.getState().ingestFrame({
      frame: createLiveNlosFrameFixture(),
      channel: 'deterministic_replay',
      receivedAtUnixMs: NOW,
    });
    expect(useNlosStore.getState()).toMatchObject({
      frame: null,
      freshness: 'unknown',
      lastRejectedReason: 'unauthenticated',
      rejectedFrameCount: 1,
    });
  });

  it('rejects replayed sequence numbers in the same session', () => {
    const frame = createLiveNlosFrameFixture({ sequence: 8 });
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW + 1 });
    expect(useNlosStore.getState().rejectedFrameCount).toBe(1);
    expect(useNlosStore.getState().lastRejectedReason).toBe('out_of_order');
  });

  it('clears fresh frames immediately when they become stale', () => {
    const frame = createLiveNlosFrameFixture({ expiresAtUnixMs: NOW + 4_900 });
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    useNlosStore.getState().refreshFreshness(NOW + NLOS_STALE_AFTER_MS + 1);
    expect(useNlosStore.getState()).toMatchObject({ frame: null, freshness: 'stale' });
  });

  it('fails closed on wall clock rollback', () => {
    const frame = createLiveNlosFrameFixture();
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    useNlosStore.getState().refreshFreshness(NOW - 1);
    expect(useNlosStore.getState()).toMatchObject({ frame: null, freshness: 'stale' });
  });

  it('clears a previously accepted frame when transport validation rejects input', () => {
    const frame = createLiveNlosFrameFixture();
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    useNlosStore.getState().recordRejection('malformed_json');
    expect(useNlosStore.getState()).toMatchObject({
      frame: null,
      freshness: 'unknown',
      lastRejectedReason: 'malformed_json',
    });
  });

  it('clears a previously accepted frame immediately when transport closes', () => {
    const frame = createLiveNlosFrameFixture();
    useNlosStore.getState().ingestFrame({ frame, channel: 'authenticated_stream', receivedAtUnixMs: NOW });
    useNlosStore.getState().setStreamStatus('error');
    expect(useNlosStore.getState()).toMatchObject({
      frame: null,
      freshness: 'unknown',
      streamStatus: 'error',
    });
  });

  it('keeps synthetic replay explicitly synthetic', () => {
    const frame = createSyntheticNlosFrame(0, NOW);
    useNlosStore.getState().ingestFrame({ frame, channel: 'deterministic_replay', receivedAtUnixMs: NOW });
    expect(useNlosStore.getState().frame?.source).toBe('synthetic');
    expect(useNlosStore.getState().frame?.evidenceLevel).toBe('l0_synthetic');
  });
});
