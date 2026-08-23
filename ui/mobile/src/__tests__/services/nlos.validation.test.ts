import { parseNlosTrackFrame, utf8ByteLength, validateNlosTrackFrame } from '@/services/nlos.validation';
import { createSyntheticNlosFrame } from '@/services/nlos.service';
import { createLiveNlosFrameFixture } from '@/testUtils/nlosFixtures';
import { NLOS_MAX_MESSAGE_BYTES, NLOS_MAX_TRACKS } from '@/types/nlos';

describe('NLOS track validation', () => {
  it('accepts the canonical measured live contract', () => {
    const frame = createLiveNlosFrameFixture();
    expect(validateNlosTrackFrame(frame)).toEqual({ ok: true, value: frame });
    expect(parseNlosTrackFrame(JSON.stringify(frame))).toEqual({ ok: true, value: frame });
  });

  it('rejects malformed JSON and messages over the 256 KiB limit', () => {
    expect(parseNlosTrackFrame('{bad-json')).toEqual({ ok: false, reason: 'malformed_json' });
    const oversized = `{"padding":"${'x'.repeat(NLOS_MAX_MESSAGE_BYTES)}"}`;
    expect(utf8ByteLength(oversized)).toBeGreaterThan(NLOS_MAX_MESSAGE_BYTES);
    expect(parseNlosTrackFrame(oversized)).toEqual({ ok: false, reason: 'message_too_large' });
  });

  it('rejects excessive track counts and spatial bounds', () => {
    const oneTrack = createLiveNlosFrameFixture().tracks[0];
    const tooMany = createLiveNlosFrameFixture({
      tracks: Array.from({ length: NLOS_MAX_TRACKS + 1 }, (_, index) => ({
        ...oneTrack,
        trackId: `target-${index}`,
      })),
    });
    expect(validateNlosTrackFrame(tooMany)).toEqual({ ok: false, reason: 'invalid_bounds' });

    const outOfBounds = createLiveNlosFrameFixture({
      tracks: [{ ...oneTrack, positionM: { x: 100.01, y: 0, z: 0 } }],
    });
    expect(validateNlosTrackFrame(outOfBounds)).toEqual({ ok: false, reason: 'invalid_shape' });
  });

  it('never promotes depth only or histogram free data to live NLOS', () => {
    const frame = createLiveNlosFrameFixture({
      provenance: {
        ...createLiveNlosFrameFixture().provenance,
        transientKind: 'depth_only',
        histogramPreserved: false,
      },
    });
    expect(validateNlosTrackFrame(frame)).toEqual({ ok: false, reason: 'invalid_provenance' });
    expect(validateNlosTrackFrame(createLiveNlosFrameFixture({
      provenance: {
        ...createLiveNlosFrameFixture().provenance,
        transport: 'replay',
      },
    }))).toEqual({ ok: false, reason: 'invalid_provenance' });
  });

  it('rejects captured replay evidence when timing histograms were discarded', () => {
    const live = createLiveNlosFrameFixture();
    const replay = {
      ...live,
      source: 'replay' as const,
      provenance: {
        ...live.provenance,
        transientKind: 'replay' as const,
        histogramPreserved: false,
        transport: 'replay' as const,
      },
    };

    expect(validateNlosTrackFrame(replay)).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
    expect(validateNlosTrackFrame({
      ...replay,
      provenance: { ...replay.provenance, histogramPreserved: true },
    }).ok).toBe(true);
  });

  it('accepts deterministic synthetic frames only with L0 and the zero calibration hash', () => {
    const synthetic = createSyntheticNlosFrame(0, 1_700_000_000_000);
    expect(validateNlosTrackFrame(synthetic).ok).toBe(true);
    expect(validateNlosTrackFrame({ ...synthetic, evidenceLevel: 'l1_measured' })).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
    expect(validateNlosTrackFrame({ ...synthetic, calibrationHash: 'b'.repeat(64) })).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
    expect(validateNlosTrackFrame({
      ...synthetic,
      provenance: { ...synthetic.provenance, histogramPreserved: true },
    }).ok).toBe(true);
    expect(validateNlosTrackFrame({
      ...synthetic,
      provenance: { ...synthetic.provenance, transientKind: 'raw_histogram' },
    })).toEqual({ ok: false, reason: 'invalid_provenance' });
  });

  it('enforces calibrated hashes, unique tracks, and normalized modality weights', () => {
    const frame = createLiveNlosFrameFixture();
    expect(validateNlosTrackFrame({ ...frame, calibrationHash: '0'.repeat(64) })).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
    expect(validateNlosTrackFrame({ ...frame, tracks: [frame.tracks[0], frame.tracks[0]] })).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
    expect(validateNlosTrackFrame({
      ...frame,
      tracks: [{
        ...frame.tracks[0],
        modalityContributions: { lidar: 0.8, csi: 0.8 },
      }],
    })).toEqual({ ok: false, reason: 'invalid_shape' });
    expect(validateNlosTrackFrame({ ...frame, evidenceLevel: 'l3_corroborated' })).toEqual({
      ok: false,
      reason: 'invalid_provenance',
    });
  });

  it('rejects unknown fields for a schema version instead of interpreting them ambiguously', () => {
    expect(validateNlosTrackFrame({ ...createLiveNlosFrameFixture(), trustMe: true })).toEqual({
      ok: false,
      reason: 'invalid_shape',
    });
  });
});
