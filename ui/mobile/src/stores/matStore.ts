import { create } from 'zustand';
import type { Alert, DisasterEvent, MatPipelineStatus, ScanZone, Survivor } from '@/types/mat';
import type { WorldGraphSnapshot, WorldGraphStreamStatus } from '@/types/worldGraph';

export type MatApiStatus = 'idle' | 'loading' | 'live' | 'error';
export interface MatState {
  events: DisasterEvent[]; zones: ScanZone[]; survivors: Survivor[]; alerts: Alert[];
  selectedEventId: string | null; apiStatus: MatApiStatus; apiError: string | null;
  pipeline: MatPipelineStatus | null; lastUpdatedAt: string | null;
  worldGraph: WorldGraphSnapshot | null; worldGraphStatus: WorldGraphStreamStatus;
  worldGraphError: string | null; worldGraphEpoch: string | null; worldGraphSeq: number | null;
  replaceSnapshot: (data: { events: DisasterEvent[]; zones: ScanZone[]; survivors: Survivor[]; alerts: Alert[]; pipeline: MatPipelineStatus | null; selectedEventId: string | null }) => void;
  upsertEvent: (event: DisasterEvent) => void; upsertZone: (zone: ScanZone) => void;
  upsertSurvivor: (survivor: Survivor) => void; removeSurvivor: (id: string) => void;
  setSurvivors: (survivors: Survivor[]) => void; upsertAlert: (alert: Alert) => void;
  setSelectedEvent: (id: string | null) => void; setApiStatus: (status: MatApiStatus, error?: string | null) => void;
  setPipeline: (pipeline: MatPipelineStatus | null) => void;
  setWorldGraph: (graph: WorldGraphSnapshot | null, epoch?: string | null, seq?: number | null) => void;
  setWorldGraphStatus: (status: WorldGraphStreamStatus, error?: string | null) => void; reset: () => void;
}
const initial = {
  events: [] as DisasterEvent[], zones: [] as ScanZone[], survivors: [] as Survivor[], alerts: [] as Alert[],
  selectedEventId: null, apiStatus: 'idle' as MatApiStatus, apiError: null, pipeline: null,
  lastUpdatedAt: null, worldGraph: null, worldGraphStatus: 'idle' as WorldGraphStreamStatus,
  worldGraphError: null, worldGraphEpoch: null, worldGraphSeq: null,
};
const upsertById = <T extends { id: string }>(items: T[], item: T): T[] =>
  items.some((candidate) => candidate.id === item.id)
    ? items.map((candidate) => candidate.id === item.id ? item : candidate) : [...items, item];
export const useMatStore = create<MatState>((set) => ({
  ...initial,
  replaceSnapshot: (data) => set({ ...data, apiStatus: 'live', apiError: null, lastUpdatedAt: new Date().toISOString() }),
  upsertEvent: (event) => set((state) => ({ events: upsertById(state.events, event) })),
  upsertZone: (zone) => set((state) => ({ zones: upsertById(state.zones, zone) })),
  upsertSurvivor: (survivor) => set((state) => ({ survivors: upsertById(state.survivors, survivor), lastUpdatedAt: new Date().toISOString() })),
  removeSurvivor: (id) => set((state) => ({ survivors: state.survivors.filter((item) => item.id !== id), lastUpdatedAt: new Date().toISOString() })),
  setSurvivors: (survivors) => set({ survivors, lastUpdatedAt: new Date().toISOString() }),
  upsertAlert: (alert) => set((state) => ({ alerts: upsertById(state.alerts, alert), lastUpdatedAt: new Date().toISOString() })),
  setSelectedEvent: (selectedEventId) => set({ selectedEventId }),
  setApiStatus: (apiStatus, apiError = null) => set({ apiStatus, apiError }), setPipeline: (pipeline) => set({ pipeline }),
  setWorldGraph: (worldGraph, worldGraphEpoch = null, worldGraphSeq = null) => set({ worldGraph, worldGraphEpoch, worldGraphSeq }),
  setWorldGraphStatus: (worldGraphStatus, worldGraphError = null) => set({ worldGraphStatus, worldGraphError }),
  reset: () => set(initial),
}));
