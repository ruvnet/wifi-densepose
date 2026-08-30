/** Wire types for `wifi-densepose-mat`'s `/api/v1/mat` API. */
export type DisasterType =
  | 'BuildingCollapse' | 'Earthquake' | 'Landslide' | 'Avalanche' | 'Flood'
  | 'MineCollapse' | 'Industrial' | 'TunnelCollapse' | 'Unknown';
export type EventStatus = 'Initializing' | 'Active' | 'Suspended' | 'SecondarySearch' | 'Closed';
export type ZoneStatus = 'Active' | 'Paused' | 'Complete' | 'Inaccessible' | 'Deactivated';
export type SurvivorStatus = 'Active' | 'Rescued' | 'Lost' | 'Deceased' | 'FalsePositive';
export type TriageStatus = 'Immediate' | 'Delayed' | 'Minor' | 'Deceased' | 'Unknown';
export type AlertPriority = 'Critical' | 'High' | 'Medium' | 'Low';
export type AlertStatus = 'Pending' | 'Acknowledged' | 'InProgress' | 'Resolved' | 'Cancelled' | 'Expired';

export interface TriageSummary { immediate: number; delayed: number; minor: number; deceased: number; unknown: number }

export interface DisasterEvent {
  id: string; event_type: DisasterType; status: EventStatus; start_time: string;
  latitude: number; longitude: number; description: string; zone_count: number;
  survivor_count: number; triage_summary: TriageSummary;
  metadata?: { estimated_occupancy?: number; confirmed_rescued: number; confirmed_deceased: number; weather?: string; lead_agency?: string };
}

export type ZoneBounds =
  | { type: 'rectangle'; min_x: number; min_y: number; max_x: number; max_y: number }
  | { type: 'circle'; center_x: number; center_y: number; radius: number }
  | { type: 'polygon'; vertices: Array<[number, number]> };

export interface ScanZone {
  id: string; name: string; status: ZoneStatus; bounds: ZoneBounds; area: number;
  parameters: { sensitivity: number; max_depth: number; resolution: 'quick' | 'standard' | 'high' | 'maximum'; enhanced_breathing: boolean; heartbeat_detection: boolean };
  last_scan?: string; scan_count: number; detections_count: number;
}

export interface MatLocation { x: number; y: number; z: number; depth: number; uncertainty_radius: number; confidence: number }
export interface VitalSignsSummary {
  breathing_rate?: number; breathing_type?: string; heart_rate?: number; has_heartbeat: boolean;
  has_movement: boolean; movement_type?: string; timestamp: string;
}

export interface Survivor {
  id: string; zone_id: string; status: SurvivorStatus; triage_status: TriageStatus;
  location?: MatLocation; vital_signs: VitalSignsSummary; confidence: number; first_detected: string;
  last_updated: string; is_deteriorating: boolean;
  metadata?: { estimated_age_category?: string; assigned_team?: string; notes: string[]; tags: string[] };
}

export interface Alert {
  id: string; survivor_id: string; priority: AlertPriority; status: AlertStatus; title: string;
  message: string; triage_status: TriageStatus; location?: MatLocation; recommended_action?: string;
  created_at: string; acknowledged_at?: string; acknowledged_by?: string; escalation_count: number;
}

export interface MatPipelineStatus {
  scanning: boolean; buffer_duration_secs: number; ml_enabled: boolean; ml_ready: boolean;
  sample_rate: number; heartbeat_enabled: boolean; min_confidence: number;
}

export interface EventListResponse { events: DisasterEvent[]; total: number; page: number; page_size: number }
export interface ZoneListResponse { zones: ScanZone[]; total: number }
export interface SurvivorListResponse { survivors: Survivor[]; total: number; triage_summary: TriageSummary }
export interface AlertListResponse { alerts: Alert[]; total: number; priority_counts: Record<Lowercase<AlertPriority>, number> }
export type MatStreamMessage =
  | { type: 'survivor_detected' | 'survivor_updated'; event_id: string; survivor: Survivor }
  | { type: 'survivor_lost'; event_id: string; survivor_id: string }
  | { type: 'alert_created' | 'alert_updated'; event_id: string; alert: Alert }
  | { type: 'zone_scan_complete'; event_id: string; zone_id: string; detections: number }
  | { type: 'event_status_changed'; event_id: string; old_status: EventStatus; new_status: EventStatus }
  | { type: 'heartbeat'; timestamp: string }
  | { type: 'error'; code: string; message: string };
