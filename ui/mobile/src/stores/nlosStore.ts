import { create } from 'zustand';
import { validateNlosTrackFrame } from '@/services/nlos.validation';
import {
  NLOS_STALE_AFTER_MS,
  type NlosFrameEvent,
  type NlosFreshness,
  type NlosRejectReason,
  type NlosStreamStatus,
  type NlosTrackFrame,
} from '@/types/nlos';

export interface NlosState {
  streamStatus: NlosStreamStatus;
  freshness: NlosFreshness;
  frame: NlosTrackFrame | null;
  lastReceivedAtUnixMs: number | null;
  lastRejectedReason: NlosRejectReason | null;
  rejectedFrameCount: number;
  ingestFrame: (event: NlosFrameEvent) => void;
  setStreamStatus: (status: NlosStreamStatus) => void;
  recordRejection: (reason: NlosRejectReason) => void;
  refreshFreshness: (now: number) => void;
  reset: () => void;
}

const initialState = {
  streamStatus: 'idle' as NlosStreamStatus,
  freshness: 'unknown' as NlosFreshness,
  frame: null as NlosTrackFrame | null,
  lastReceivedAtUnixMs: null as number | null,
  lastRejectedReason: null as NlosRejectReason | null,
  rejectedFrameCount: 0,
};

export const useNlosStore = create<NlosState>((set) => ({
  ...initialState,

  ingestFrame: (event) => {
    const validation = validateNlosTrackFrame(event.frame);
    if (!validation.ok) {
      set((state) => ({
        lastRejectedReason: validation.reason,
        rejectedFrameCount: state.rejectedFrameCount + 1,
      }));
      return;
    }

    const frame = validation.value;
    const channelValid =
      (event.channel === 'authenticated_stream' &&
        (frame.source === 'live' || frame.source === 'synthetic')) ||
      (event.channel === 'deterministic_replay' && frame.source === 'synthetic');
    if (!channelValid) {
      set((state) => ({
        lastRejectedReason: frame.source === 'live' ? 'unauthenticated' : 'invalid_provenance',
        rejectedFrameCount: state.rejectedFrameCount + 1,
      }));
      return;
    }

    set((state) => {
      if (
        state.frame?.sessionId === frame.sessionId &&
        frame.sequence <= state.frame.sequence
      ) {
        return {
          lastRejectedReason: 'out_of_order',
          rejectedFrameCount: state.rejectedFrameCount + 1,
        };
      }
      if (frame.expiresAtUnixMs <= event.receivedAtUnixMs) {
        return {
          lastRejectedReason: 'expired',
          rejectedFrameCount: state.rejectedFrameCount + 1,
        };
      }

      return {
        frame,
        freshness: 'fresh',
        lastReceivedAtUnixMs: event.receivedAtUnixMs,
        lastRejectedReason: null,
      };
    });
  },

  setStreamStatus: (streamStatus) =>
    set((state) => {
      if (streamStatus === 'live' || streamStatus === 'synthetic_replay') {
        return { streamStatus };
      }
      if (
        state.frame === null &&
        state.freshness === 'unknown' &&
        state.lastReceivedAtUnixMs === null
      ) {
        return { streamStatus };
      }
      return {
        streamStatus,
        frame: null,
        freshness: 'unknown',
        lastReceivedAtUnixMs: null,
      };
    }),

  recordRejection: (reason) =>
    set((state) => ({
      frame: null,
      freshness: 'unknown',
      lastReceivedAtUnixMs: null,
      lastRejectedReason: reason,
      rejectedFrameCount: state.rejectedFrameCount + 1,
    })),

  refreshFreshness: (now) =>
    set((state) => {
      if (!state.frame || state.lastReceivedAtUnixMs === null) {
        return state.freshness === 'unknown' ? {} : { freshness: 'unknown' };
      }
      const stale =
        now < state.lastReceivedAtUnixMs ||
        now >= state.frame.expiresAtUnixMs ||
        now - state.lastReceivedAtUnixMs > NLOS_STALE_AFTER_MS;
      const nextFreshness: NlosFreshness = stale ? 'stale' : 'fresh';
      if (stale) {
        return {
          frame: null,
          freshness: nextFreshness,
          lastReceivedAtUnixMs: null,
        };
      }
      return state.freshness === nextFreshness ? {} : { freshness: nextFreshness };
    }),

  reset: () => set(initialState),
}));
