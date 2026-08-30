export interface AppleHomeBridge {
  id: string;
  name: string;
  hostName?: string;
  port: number;
  model?: string;
  categoryIdentifier?: string;
  paired?: boolean;
  serviceType: '_hap._tcp.' | string;
  domain: string;
  source: 'bonjour_hap';
}

export interface AppleHomeDiscoveryState {
  state: 'idle' | 'searching' | 'error' | 'unavailable';
  bridges: AppleHomeBridge[];
  capturedAtUnixMs?: number;
  source?: 'bonjour_hap';
  error?: string;
}

export interface RuViewVitalsApiFrame {
  node_id: string | number;
  timestamp_ms: number;
  presence: boolean;
  n_persons: number;
  confidence: number;
  breathing_rate_bpm?: number | null;
  heartrate_bpm?: number | null;
  motion: number;
}

export interface RuViewSemanticEvents {
  node_id: string | number;
  privacy_class: number;
  events: Record<string, { active?: boolean; source?: string; ts?: number }>;
}
