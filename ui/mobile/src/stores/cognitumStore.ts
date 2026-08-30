import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { cognitumService } from '@/services/cognitum.service';
import type { CognitumAuthStatus, CognitumDataStatus, CognitumSessionSummary, CognitumSpatialResource, SpatialInsight, SpatialIntelligenceInput } from '@/types/cognitum';

interface CognitumState {
  authStatus: CognitumAuthStatus;
  dataStatus: CognitumDataStatus;
  session: CognitumSessionSummary;
  inferenceAuthorized: boolean;
  spaces: CognitumSpatialResource[];
  zones: CognitumSpatialResource[];
  boundary: string | null;
  error: string | null;
  cloudConsentAt: number | null;
  insight: SpatialInsight | null;
  bootstrap: () => Promise<void>;
  connectSpaces: () => Promise<void>;
  completeAuthorization: (code: string) => Promise<void>;
  disconnect: () => Promise<void>;
  refreshSpatial: () => Promise<void>;
  setCloudConsent: (enabled: boolean) => void;
  analyze: (input: SpatialIntelligenceInput) => Promise<void>;
}

const message = (error: unknown) => error instanceof Error ? error.message : 'Cognitum request failed';

export const useCognitumStore = create<CognitumState>()(persist((set, get) => ({
  authStatus: 'checking', dataStatus: 'idle', session: { signedIn: false, scopes: [] }, inferenceAuthorized: false, spaces: [], zones: [], boundary: null, error: null, cloudConsentAt: null, insight: null,
  bootstrap: async () => {
    try {
      const [session, inference] = await Promise.all([cognitumService.sessionSummary(), cognitumService.inferenceSessionSummary()]);
      set({ session, inferenceAuthorized: inference.signedIn && inference.scopes.includes('inference'), authStatus: session.signedIn ? 'signed_in' : 'signed_out', error: null });
      if (session.signedIn && session.scopes.includes('spaces:read')) await get().refreshSpatial();
    } catch (error) { set({ authStatus: 'error', error: message(error) }); }
  },
  connectSpaces: async () => {
    set({ authStatus: 'authorizing', error: null });
    try {
      await cognitumService.beginSpacesSignIn();
      set({ authStatus: 'awaiting_code' });
    } catch (error) { set({ authStatus: 'error', error: message(error) }); }
  },
  completeAuthorization: async (code) => {
    set({ authStatus: 'authorizing', error: null });
    try {
      const result = await cognitumService.completeSignIn(code);
      if (result.kind === 'spaces') {
        set({ session: result.session, authStatus: 'signed_in' });
        if (result.session.scopes.includes('spaces:read')) await get().refreshSpatial();
      } else {
        set({ inferenceAuthorized: result.session.scopes.includes('inference'), authStatus: get().session.signedIn ? 'signed_in' : 'signed_out' });
      }
    } catch (error) { set({ authStatus: 'awaiting_code', error: message(error) }); }
  },
  disconnect: async () => {
    await cognitumService.signOut();
    set({ authStatus: 'signed_out', dataStatus: 'idle', session: { signedIn: false, scopes: [] }, inferenceAuthorized: false, spaces: [], zones: [], boundary: null, error: null, insight: null, cloudConsentAt: null });
  },
  refreshSpatial: async () => {
    set({ dataStatus: 'loading', error: null });
    try {
      const [spaces, zones] = await Promise.all([cognitumService.listSpatial('spaces'), cognitumService.listSpatial('zones')]);
      set({ spaces: spaces.data, zones: zones.data, boundary: `${spaces.boundary.authoritativeState} / ${spaces.boundary.cloudRole}`, dataStatus: 'live' });
    } catch (error) { set({ dataStatus: 'error', error: message(error) }); }
  },
  setCloudConsent: (enabled) => set({ cloudConsentAt: enabled ? Date.now() : null, insight: enabled ? get().insight : null }),
  analyze: async (input) => {
    if (!get().cloudConsentAt) { set({ error: 'Enable cloud interpretation before sending a semantic summary' }); return; }
    set({ dataStatus: 'loading', error: null });
    try {
      if (!get().inferenceAuthorized) {
        set({ authStatus: 'authorizing' });
        await cognitumService.beginCloudSignIn();
        set({ authStatus: 'awaiting_code', error: 'Authorize Cognitum inference with the one-time code, then generate the brief again.' });
        return;
      }
      const insight = await cognitumService.analyzeSpatial(input);
      set({ insight, dataStatus: 'live' });
    } catch (error) { set({ dataStatus: 'error', authStatus: get().session.signedIn ? 'signed_in' : 'error', error: message(error) }); }
  },
}), {
  name: 'ruview-cognitum-consent', storage: createJSONStorage(() => AsyncStorage),
  partialize: (state) => ({ cloudConsentAt: state.cloudConsentAt }),
}));
