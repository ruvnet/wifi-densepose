import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

export type Theme = 'light' | 'dark' | 'system';

export interface SettingsState {
  serverUrl: string;
  nlosServerUrl: string;
  rssiScanEnabled: boolean;
  theme: Theme;
  alertSoundEnabled: boolean;
  setServerUrl: (url: string) => void;
  setNlosServerUrl: (url: string) => void;
  setRssiScanEnabled: (value: boolean) => void;
  setTheme: (theme: Theme) => void;
  setAlertSoundEnabled: (value: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      serverUrl: 'http://localhost:3000',
      nlosServerUrl: 'http://localhost:3000',
      rssiScanEnabled: false,
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
