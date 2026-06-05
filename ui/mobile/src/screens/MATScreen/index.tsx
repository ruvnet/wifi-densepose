import { useEffect } from 'react';
import { useWindowDimensions, View } from 'react-native';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { ThemedView } from '@/components/ThemedView';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { usePoseStream } from '@/hooks/usePoseStream';
import { useMatStore } from '@/stores/matStore';
import { type ConnectionStatus } from '@/types/sensing';
import { Alert, type Survivor } from '@/types/mat';
import { AlertList } from './AlertList';
import { MatWebView } from './MatWebView';
import { SurvivorCounter } from './SurvivorCounter';
import { useMatBridge } from './useMatBridge';

const isAlert = (value: unknown): value is Alert => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;
  return typeof record.id === 'string' && typeof record.message === 'string';
};

const isSurvivor = (value: unknown): value is Survivor => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;
  return typeof record.id === 'string' && typeof record.zone_id === 'string';
};

const resolveBannerState = (status: ConnectionStatus): 'connected' | 'simulated' | 'disconnected' => {
  if (status === 'connecting' || status === 'simulated') {
    return 'disconnected';
  }

  return status;
};

export const MATScreen = () => {
  const { connectionStatus, lastFrame } = usePoseStream();

  const survivors = useMatStore((state) => state.survivors);
  const alerts = useMatStore((state) => state.alerts);
  const upsertSurvivor = useMatStore((state) => state.upsertSurvivor);
  const addAlert = useMatStore((state) => state.addAlert);
  const setDataSource = useMatStore((state) => state.setDataSource);

  // Sync dataSource from connection status
  useEffect(() => {
    setDataSource(connectionStatus === 'connected' ? 'real' : 'unavailable');
  }, [connectionStatus, setDataSource]);

  const { webViewRef, ready, onMessage, sendFrameUpdate } = useMatBridge({
    onSurvivorDetected: (survivor) => {
      if (isSurvivor(survivor)) {
        upsertSurvivor(survivor);
      }
    },
    onAlertGenerated: (alert) => {
      if (isAlert(alert)) {
        addAlert(alert);
      }
    },
  });

  useEffect(() => {
    if (ready && lastFrame) {
      sendFrameUpdate(lastFrame);
    }
  }, [lastFrame, ready, sendFrameUpdate]);

  const { height } = useWindowDimensions();
  const webHeight = Math.max(240, Math.floor(height * 0.5));

  return (
    <ThemedView style={{ flex: 1, backgroundColor: colors.bg, padding: spacing.md }}>
      <ConnectionBanner status={resolveBannerState(connectionStatus)} />
      <View style={{ marginTop: 20 }}>
        <SurvivorCounter survivors={survivors} />
      </View>
      <View style={{ height: webHeight }}>
        <MatWebView
          webViewRef={webViewRef}
          onMessage={onMessage}
          style={{ flex: 1, borderRadius: 12, overflow: 'hidden', backgroundColor: colors.surface }}
        />
      </View>
      <View style={{ flex: 1, marginTop: spacing.md }}>
        <AlertList alerts={alerts} />
      </View>
    </ThemedView>
  );
};

export default MATScreen;
