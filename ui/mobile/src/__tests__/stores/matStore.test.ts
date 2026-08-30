import { useMatStore } from '@/stores/matStore';
import type { Alert, DisasterEvent, Survivor } from '@/types/mat';

const event: DisasterEvent = {
  id: 'event-1', event_type: 'BuildingCollapse', status: 'Active', start_time: '2026-01-01T00:00:00Z',
  latitude: 1, longitude: 2, description: 'Warehouse response', zone_count: 1, survivor_count: 0,
  triage_summary: { immediate: 0, delayed: 0, minor: 0, deceased: 0, unknown: 0 },
};
const survivor: Survivor = {
  id: 'survivor-1', zone_id: 'zone-1', status: 'Active', triage_status: 'Unknown', confidence: .82,
  vital_signs: { has_heartbeat: false, has_movement: true, timestamp: '2026-01-01T00:00:01Z' },
  first_detected: '2026-01-01T00:00:00Z', last_updated: '2026-01-01T00:00:01Z', is_deteriorating: false,
};
const alert: Alert = {
  id: 'alert-1', survivor_id: survivor.id, priority: 'High', status: 'Pending', title: 'Movement changed',
  message: 'Source-reported change', triage_status: 'Unknown', created_at: '2026-01-01T00:00:02Z', escalation_count: 0,
};

describe('useMatStore', () => {
  beforeEach(() => useMatStore.getState().reset());

  it('starts fail-closed with no generated operational data', () => {
    const state = useMatStore.getState();
    expect(state.apiStatus).toBe('idle');
    expect(state.events).toEqual([]);
    expect(state.survivors).toEqual([]);
    expect(state.alerts).toEqual([]);
  });

  it('replaces one authoritative MAT snapshot', () => {
    useMatStore.getState().replaceSnapshot({ events: [event], selectedEventId: event.id, zones: [], survivors: [survivor], alerts: [alert], pipeline: null });
    const state = useMatStore.getState();
    expect(state.apiStatus).toBe('live');
    expect(state.selectedEventId).toBe(event.id);
    expect(state.survivors).toEqual([survivor]);
  });

  it('applies survivor and alert stream updates by stable id', () => {
    useMatStore.getState().upsertSurvivor(survivor);
    useMatStore.getState().upsertSurvivor({ ...survivor, confidence: .94 });
    useMatStore.getState().upsertAlert(alert);
    useMatStore.getState().upsertAlert({ ...alert, status: 'Acknowledged' });
    expect(useMatStore.getState().survivors).toHaveLength(1);
    expect(useMatStore.getState().survivors[0].confidence).toBe(.94);
    expect(useMatStore.getState().alerts[0].status).toBe('Acknowledged');
  });

  it('stores WorldGraph source state separately from MAT records', () => {
    const graph = { schema_version: 2, nodes: [], edges: [] };
    useMatStore.getState().setWorldGraph(graph, 'epoch-a', 7);
    expect(useMatStore.getState()).toMatchObject({ worldGraph: graph, worldGraphEpoch: 'epoch-a', worldGraphSeq: 7 });
    expect(useMatStore.getState().survivors).toEqual([]);
  });
});
