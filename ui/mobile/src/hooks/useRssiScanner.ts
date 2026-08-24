import { useEffect, useState } from 'react';
import { rssiService, type WifiNetwork } from '@/services/rssi.service';
import { useSettingsStore } from '@/stores/settingsStore';

export function useRssiScanner(): { networks: WifiNetwork[]; isScanning: boolean } {
  const enabled = useSettingsStore((state) => state.rssiScanEnabled);
  const intervalSeconds = useSettingsStore((state) => state.rssiScanIntervalSeconds);
  const [networks, setNetworks] = useState<WifiNetwork[]>([]);
  const [isScanning, setIsScanning] = useState(false);

  useEffect(() => {
    if (!enabled) {
      rssiService.stopScanning();
      setIsScanning(false);
      return;
    }

    const unsubscribe = rssiService.subscribe((result) => {
      setNetworks(result);
    });
    rssiService.startScanning(intervalSeconds * 1000);
    setIsScanning(true);

    return () => {
      unsubscribe();
      rssiService.stopScanning();
      setIsScanning(false);
    };
  }, [enabled, intervalSeconds]);

  return { networks, isScanning };
}
