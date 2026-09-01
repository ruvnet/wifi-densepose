import { useCallback, useEffect, useState } from 'react';
import {
  configureNlosBearerToken,
  hasConfiguredNlosBearerToken,
  nlosService,
} from '@/services/nlos.service';
import { useNlosStore } from '@/stores/nlosStore';
import { useSettingsStore } from '@/stores/settingsStore';

const FRESHNESS_POLL_MS = 250;

export const useNlosStream = () => {
  const nlosServerUrl = useSettingsStore((state) => state.nlosServerUrl);
  const frame = useNlosStore((state) => state.frame);
  const freshness = useNlosStore((state) => state.freshness);
  const streamStatus = useNlosStore((state) => state.streamStatus);
  const lastRejectedReason = useNlosStore((state) => state.lastRejectedReason);
  const rejectedFrameCount = useNlosStore((state) => state.rejectedFrameCount);
  const [liveCredentialAvailable, setLiveCredentialAvailable] = useState(
    hasConfiguredNlosBearerToken,
  );

  useEffect(() => {
    useNlosStore.getState().reset();
    const unsubscribeFrame = nlosService.subscribe((event) => {
      useNlosStore.getState().ingestFrame(event);
    });
    const unsubscribeStatus = nlosService.subscribeStatus((status) => {
      useNlosStore.getState().setStreamStatus(status);
    });
    const unsubscribeRejected = nlosService.subscribeRejected((reason) => {
      useNlosStore.getState().recordRejection(reason);
    });
    const freshnessTimer = setInterval(() => {
      useNlosStore.getState().refreshFreshness(Date.now());
    }, FRESHNESS_POLL_MS);

    if (hasConfiguredNlosBearerToken()) {
      void nlosService.connectConfiguredLive(nlosServerUrl);
    }

    return () => {
      clearInterval(freshnessTimer);
      unsubscribeFrame();
      unsubscribeStatus();
      unsubscribeRejected();
      nlosService.disconnect();
    };
  }, [nlosServerUrl]);

  const startReplay = useCallback(() => {
    useNlosStore.getState().reset();
    nlosService.startDeterministicReplay();
  }, []);

  const connectLive = useCallback(() => {
    void nlosService.connectConfiguredLive(nlosServerUrl);
  }, [nlosServerUrl]);

  const configureCredential = useCallback((token: string): boolean => {
    const configured = configureNlosBearerToken(token);
    if (configured) setLiveCredentialAvailable(true);
    return configured;
  }, []);

  const forgetCredential = useCallback(() => {
    configureNlosBearerToken(null);
    setLiveCredentialAvailable(false);
    nlosService.disconnect();
    useNlosStore.getState().reset();
  }, []);

  return {
    frame,
    freshness,
    streamStatus,
    lastRejectedReason,
    rejectedFrameCount,
    liveCredentialAvailable,
    configureCredential,
    forgetCredential,
    startReplay,
    connectLive,
  };
};
