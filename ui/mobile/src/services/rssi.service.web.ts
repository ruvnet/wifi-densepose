import type { RssiService, WifiNetwork } from './rssi.service';

class WebRssiService implements RssiService {
  private listeners = new Set<(networks: WifiNetwork[]) => void>();

  startScanning(): void {
    console.warn('Web RSSI scanning not available; no generated network data will be emitted.');
    this.stopScanning();
    this.broadcast([]);
  }

  stopScanning(): void {
  }

  subscribe(listener: (networks: WifiNetwork[]) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private broadcast(networks: WifiNetwork[]): void {
    this.listeners.forEach((listener) => {
      try {
        listener(networks);
      } catch {
        // listener safety
      }
    });
  }
}

export const rssiService = new WebRssiService();
