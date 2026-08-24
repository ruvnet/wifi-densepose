import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

export type Theme = 'light' | 'dark' | 'system';
export type RssiScanIntervalSeconds = 1 | 2 | 5;

export interface SettingsState {
  serverUrl: string;
  nlosServerUrl: string;
  rssiScanEnabled: boolean;
  rssiScanIntervalSeconds: RssiScanIntervalSeconds;
  theme: Theme;
  alertSoundEnabled: boolean;
  setServerUrl: (url: string) => void;
  setNlosServerUrl: (url: string) => void;
  setRssiScanEnabled: (value: boolean) => void;
  setRssiScanIntervalSeconds: (value: RssiScanIntervalSeconds) => void;
  setTheme: (theme: Theme) => void;
  setAlertSoundEnabled: (value: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      serverUrl: 'http://localhost:3000',
      nlosServerUrl: 'http://localhost:3000',
      rssiScanEnabled: false,
      rssiScanIntervalSeconds: 2,
      theme: 'system',
      alertSoundEnabled: true,

      setServerUrl: (url) => {
        set({ serverUrl: url });
      },

      setNlosServerUrl: (url) => {
        const validation = normalizeNlosServerUrl(url);
        if (validation.valid && validation.normalized) {
          set({ nlosServerUrl: validation.normalized });
        }
      },

      setRssiScanEnabled: (value) => {
        set({ rssiScanEnabled: value });
      },

      setRssiScanIntervalSeconds: (value) => {
        set({ rssiScanIntervalSeconds: value });
      },

      setTheme: (theme) => {
        set({ theme });
      },

      setAlertSoundEnabled: (value) => {
        set({ alertSoundEnabled: value });
      },
    }),
    {
      name: 'wifi-densepose-settings',
      storage: createJSONStorage(() => AsyncStorage),
    },
  ),
);
