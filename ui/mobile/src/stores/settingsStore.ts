import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

export type Theme = 'light' | 'dark' | 'system';
export type RssiScanIntervalSeconds = 1 | 2 | 5;

const configuredServerUrl = process.env.EXPO_PUBLIC_DEFAULT_SERVER_URL?.trim().replace(/\/+$/, '');
export const DEFAULT_SERVER_URL = configuredServerUrl || 'http://localhost:3000';

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
      serverUrl: DEFAULT_SERVER_URL,
      nlosServerUrl: DEFAULT_SERVER_URL,
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
      version: 2,
      migrate: (persisted, version) => {
        const previous = (persisted ?? {}) as Partial<SettingsState>;
        if (version < 2 && DEFAULT_SERVER_URL !== 'http://localhost:3000') {
          const wasLoopbackDefault = (value: string | undefined) => value === 'http://localhost:3000' || value === 'http://localhost:8080';
          return {
            ...previous,
            serverUrl: wasLoopbackDefault(previous.serverUrl) ? DEFAULT_SERVER_URL : previous.serverUrl,
            nlosServerUrl: wasLoopbackDefault(previous.nlosServerUrl) ? DEFAULT_SERVER_URL : previous.nlosServerUrl,
          } as SettingsState;
        }
        return previous as SettingsState;
      },
    },
  ),
);
