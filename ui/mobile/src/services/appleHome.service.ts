import { Linking, Platform } from 'react-native';
import { requireOptionalNativeModule, type NativeModule } from 'expo-modules-core';
import { apiService } from '@/services/api.service';
import type { AppleHomeDiscoveryState, RuViewSemanticEvents, RuViewVitalsApiFrame } from '@/types/appleHome';

type NativeAppleHomeModule = NativeModule & {
  addListener(eventName: 'onAppleHomeDiscovery', listener: (state: AppleHomeDiscoveryState) => void): { remove(): void };
  startDiscovery(): Promise<AppleHomeDiscoveryState>;
  stopDiscovery(): Promise<AppleHomeDiscoveryState>;
  getDiscoveredBridges(): Promise<AppleHomeDiscoveryState['bridges']>;
};

const nativeAppleHome = requireOptionalNativeModule<NativeAppleHomeModule>('RuViewAppleHome');

export const appleHomeService = {
  nativeAvailable: Boolean(nativeAppleHome),
  events: nativeAppleHome,
  startDiscovery: (): Promise<AppleHomeDiscoveryState> => nativeAppleHome?.startDiscovery()
    ?? Promise.resolve({ state: 'unavailable', bridges: [], error: 'Apple Home discovery requires the native iOS development build.' }),
  stopDiscovery: (): Promise<AppleHomeDiscoveryState> => nativeAppleHome?.stopDiscovery()
    ?? Promise.resolve({ state: 'unavailable', bridges: [] }),
  getDiscoveredBridges: () => nativeAppleHome?.getDiscoveredBridges() ?? Promise.resolve([]),
  getLiveVitals: (nodeId: string | number) => apiService.get<RuViewVitalsApiFrame>(`/api/v1/vitals/${encodeURIComponent(String(nodeId))}/latest`),
  getSemanticEvents: (nodeId: string | number) => apiService.get<RuViewSemanticEvents>(`/api/v1/semantic-events/${encodeURIComponent(String(nodeId))}/latest`),
  openAppleHome: async () => {
    if (Platform.OS !== 'ios') return false;
    try {
      // `canOpenURL` returns false unless every queried scheme is declared in
      // LSApplicationQueriesSchemes. Opening the system app directly avoids a
      // false-negative while still failing closed if iOS rejects the URL.
      await Linking.openURL('com.apple.home://');
      return true;
    } catch {
      return false;
    }
  },
};
