import { apiService } from './api.service';
import type {
  Alert, AlertListResponse, DisasterEvent, EventListResponse, MatPipelineStatus,
  MatStreamMessage, ScanZone, Survivor, SurvivorListResponse, ZoneListResponse,
} from '@/types/mat';

const ROOT = '/api/v1/mat';

export interface MatSnapshot {
  events: DisasterEvent[];
  selectedEventId: string | null;
  zones: ScanZone[];
  survivors: Survivor[];
  alerts: Alert[];
  pipeline: MatPipelineStatus;
}

const toWebSocketUrl = (serverUrl: string): string => {
  const parsed = new URL(serverUrl);
  parsed.protocol = parsed.protocol === 'https:' || parsed.protocol === 'wss:' ? 'wss:' : 'ws:';
  parsed.pathname = '/ws/mat/stream';
  parsed.search = '';
  parsed.hash = '';
  return parsed.toString();
};

class MatService {
  configure(serverUrl: string): void { apiService.setBaseUrl(serverUrl); }

  async fetchSnapshot(preferredEventId?: string | null): Promise<MatSnapshot> {
    const [eventList, pipeline] = await Promise.all([
      apiService.get<EventListResponse>(`${ROOT}/events?page_size=100`),
      apiService.get<MatPipelineStatus>(`${ROOT}/scan/status`),
    ]);
    const selectedEventId = eventList.events.some((event) => event.id === preferredEventId)
      ? preferredEventId! : (eventList.events.find((event) => event.status === 'Active')?.id ?? eventList.events[0]?.id ?? null);
    if (!selectedEventId) return { events: eventList.events, selectedEventId, zones: [], survivors: [], alerts: [], pipeline };

    const encoded = encodeURIComponent(selectedEventId);
    const [zoneList, survivorList, alertList] = await Promise.all([
      apiService.get<ZoneListResponse>(`${ROOT}/events/${encoded}/zones`),
      apiService.get<SurvivorListResponse>(`${ROOT}/events/${encoded}/survivors`),
      apiService.get<AlertListResponse>(`${ROOT}/events/${encoded}/alerts?active_only=true`),
    ]);
    return {
      events: eventList.events, selectedEventId, zones: zoneList.zones,
      survivors: survivorList.survivors, alerts: alertList.alerts, pipeline,
    };
  }

  async setScanning(action: 'start' | 'stop' | 'pause' | 'resume' | 'clear_buffer'): Promise<void> {
    await apiService.post(`${ROOT}/scan/control`, { action });
  }

  async acknowledgeAlert(alertId: string, acknowledgedBy = 'RuView mobile operator'): Promise<Alert> {
    const response = await apiService.post<{ success: boolean; alert: Alert }>(
      `${ROOT}/alerts/${encodeURIComponent(alertId)}/acknowledge`, { acknowledged_by: acknowledgedBy },
    );
    if (!response.success) throw new Error('MAT API did not acknowledge the alert');
    return response.alert;
  }

  openStream(serverUrl: string, eventId: string, onMessage: (message: MatStreamMessage) => void, onError: (message: string) => void): () => void {
    const socket = new WebSocket(toWebSocketUrl(serverUrl));
    socket.onopen = () => socket.send(JSON.stringify({ action: 'subscribe', event_id: eventId }));
    socket.onmessage = (event) => {
      try {
        if (typeof event.data === 'string') onMessage(JSON.parse(event.data) as MatStreamMessage);
      } catch { onError('MAT stream returned invalid JSON'); }
    };
    socket.onerror = () => onError('MAT real-time stream disconnected');
    return () => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: 'unsubscribe', event_id: eventId }));
      socket.close();
    };
  }
}

export const matService = new MatService();
export { toWebSocketUrl as buildMatStreamUrl };
