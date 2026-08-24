import { useSettingsStore } from '@/stores/settingsStore';

describe('useSettingsStore', () => {
  beforeEach(() => {
    // Reset to defaults by manually setting all values
    useSettingsStore.setState({
      serverUrl: 'http://localhost:3000',
      nlosServerUrl: 'http://localhost:3000',
      rssiScanEnabled: false,
      rssiScanIntervalSeconds: 2,
      theme: 'system',
      alertSoundEnabled: true,
    });
  });

  describe('default values', () => {
    it('has default serverUrl as http://localhost:3000', () => {
      expect(useSettingsStore.getState().serverUrl).toBe('http://localhost:3000');
    });

    it('has rssiScanEnabled false by default', () => {
      expect(useSettingsStore.getState().rssiScanEnabled).toBe(false);
    });

    it('has a two second RSSI scan interval by default', () => {
      expect(useSettingsStore.getState().rssiScanIntervalSeconds).toBe(2);
    });

    it('has theme as system by default', () => {
      expect(useSettingsStore.getState().theme).toBe('system');
    });

    it('has alertSoundEnabled true by default', () => {
      expect(useSettingsStore.getState().alertSoundEnabled).toBe(true);
    });
  });

  describe('setServerUrl', () => {
    it('updates the server URL', () => {
      useSettingsStore.getState().setServerUrl('http://10.0.0.1:8080');
      expect(useSettingsStore.getState().serverUrl).toBe('http://10.0.0.1:8080');
    });

    it('handles empty string', () => {
      useSettingsStore.getState().setServerUrl('');
      expect(useSettingsStore.getState().serverUrl).toBe('');
    });
  });

  describe('setNlosServerUrl', () => {
    it('updates NLOS independently from the CSI server URL', () => {
      useSettingsStore.getState().setNlosServerUrl('https://nlos.example');
      expect(useSettingsStore.getState().nlosServerUrl).toBe('https://nlos.example');
      expect(useSettingsStore.getState().serverUrl).toBe('http://localhost:3000');
    });

    it('never persists credentials or URL components outside the server origin', () => {
      const initial = useSettingsStore.getState().nlosServerUrl;
      for (const unsafe of [
        'https://user:secret@nlos.example',
        'https://nlos.example/path',
        'https://nlos.example?token=secret',
        'https://nlos.example#secret',
      ]) {
        useSettingsStore.getState().setNlosServerUrl(unsafe);
        expect(useSettingsStore.getState().nlosServerUrl).toBe(initial);
      }
      useSettingsStore.getState().setNlosServerUrl('https://nlos.example:443/');
      expect(useSettingsStore.getState().nlosServerUrl).toBe('https://nlos.example');
    });
  });

  describe('setRssiScanEnabled', () => {
    it('toggles to true', () => {
      useSettingsStore.getState().setRssiScanEnabled(true);
      expect(useSettingsStore.getState().rssiScanEnabled).toBe(true);
    });

    it('toggles back to false', () => {
      useSettingsStore.getState().setRssiScanEnabled(true);
      useSettingsStore.getState().setRssiScanEnabled(false);
      expect(useSettingsStore.getState().rssiScanEnabled).toBe(false);
    });
  });

  describe('setRssiScanIntervalSeconds', () => {
    it('updates the persisted scan interval', () => {
      useSettingsStore.getState().setRssiScanIntervalSeconds(5);
      expect(useSettingsStore.getState().rssiScanIntervalSeconds).toBe(5);
    });
  });

  describe('setTheme', () => {
    it('sets theme to dark', () => {
      useSettingsStore.getState().setTheme('dark');
      expect(useSettingsStore.getState().theme).toBe('dark');
    });

    it('sets theme to light', () => {
      useSettingsStore.getState().setTheme('light');
      expect(useSettingsStore.getState().theme).toBe('light');
    });

    it('sets theme back to system', () => {
      useSettingsStore.getState().setTheme('dark');
      useSettingsStore.getState().setTheme('system');
      expect(useSettingsStore.getState().theme).toBe('system');
    });
  });

  describe('setAlertSoundEnabled', () => {
    it('disables alert sound', () => {
      useSettingsStore.getState().setAlertSoundEnabled(false);
      expect(useSettingsStore.getState().alertSoundEnabled).toBe(false);
    });

    it('re-enables alert sound', () => {
      useSettingsStore.getState().setAlertSoundEnabled(false);
      useSettingsStore.getState().setAlertSoundEnabled(true);
      expect(useSettingsStore.getState().alertSoundEnabled).toBe(true);
    });
  });
});
