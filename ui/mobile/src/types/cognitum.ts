export type CognitumAuthStatus = 'checking' | 'signed_out' | 'authorizing' | 'awaiting_code' | 'signed_in' | 'error';
export type CognitumDataStatus = 'idle' | 'loading' | 'live' | 'error';
export type SpatialKind = 'spaces' | 'zones';

export interface CognitumSessionSummary {
  signedIn: boolean;
  accountId?: string;
  workspaceId?: string;
  scopes: string[];
  expiresAt?: number;
}

export interface CognitumBoundary {
  authoritativeState: string;
  cloudRole: string;
  excluded: string[];
}

export interface CognitumSpatialResource {
  id: string;
  kind: SpatialKind;
  name?: string;
  privacy: 'P2' | 'P3';
  siteId?: string;
  spaceId?: string;
  confidence?: number;
  observedAt: string;
  status?: string;
  attributes: Record<string, unknown>;
  provenance: Record<string, unknown>;
}

export interface CognitumSpatialPage {
  kind: SpatialKind;
  data: CognitumSpatialResource[];
  boundary: CognitumBoundary;
}

export interface SpatialIntelligenceInput {
  generatedAt: string;
  local: {
    connection: string;
    anonymousOccupancy: number;
    presence: boolean;
    motionLevel: string;
    confidence: number | null;
    signalQuality: number | null;
  };
  spaces: Array<{
    id: string;
    name: string | null;
    status: string | null;
    confidence: number | null;
    observedAt: string;
  }>;
  zones: Array<{
    id: string;
    name: string | null;
    status: string | null;
    confidence: number | null;
    observedAt: string;
  }>;
}

export interface SpatialInsight {
  text: string;
  model: string;
  provider?: string;
  generatedAt: number;
  requestId?: string;
  resolvedTier?: string;
  escalated?: boolean;
  priceUsd?: number;
  promptTokens?: number;
  completionTokens?: number;
}
