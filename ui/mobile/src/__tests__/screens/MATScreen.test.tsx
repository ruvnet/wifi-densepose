import React from 'react';
import { render, waitFor } from '@testing-library/react-native';
import { ThemeProvider } from '@/theme/ThemeContext';
import { useMatStore } from '@/stores/matStore';

jest.mock('react-native-svg', () => {
  const { View } = require('react-native');
  return { __esModule: true, default: View, Circle: View, G: View, Line: View, Polygon: View, Rect: View, Text: View };
});
jest.mock('@expo/vector-icons', () => {
  const { View } = require('react-native');
  return { Ionicons: View };
});
const mockFetchSnapshot = jest.fn(async () => ({
  events: [], selectedEventId: null, zones: [], survivors: [], alerts: [],
  pipeline: { scanning: false, buffer_duration_secs: 0, ml_enabled: true, ml_ready: false, sample_rate: 1000, heartbeat_enabled: false, min_confidence: .7 },
}));
jest.mock('@/services/mat.service', () => ({
  matService: { configure: jest.fn(), fetchSnapshot: mockFetchSnapshot, openStream: jest.fn(() => jest.fn()), setScanning: jest.fn(), acknowledgeAlert: jest.fn() },
}));
jest.mock('@/services/worldGraph.service', () => ({ worldGraphService: {
  connect: jest.fn(), disconnect: jest.fn(), fetchSnapshot: jest.fn(async () => ({
    graph: { schema_version: 1, nodes: [], edges: [] }, epoch: 'test', seq: 0,
  })),
} }));

describe('MATScreen', () => {
  beforeEach(() => { useMatStore.getState().reset(); mockFetchSnapshot.mockClear(); });

  it('renders the source-honest incident dashboard', async () => {
    const { MATScreen } = require('@/screens/MATScreen');
    const view = render(<ThemeProvider><MATScreen /></ThemeProvider>);
    expect(view.getByText('MISSION-AWARE TRIAGE / VERIFIED INPUTS')).toBeTruthy();
    expect(view.getByTestId('worldgraph-map')).toBeTruthy();
    await waitFor(() => expect(mockFetchSnapshot).toHaveBeenCalled());
    expect(view.getByText('No incident events returned. Create an event through the MAT API before scanning.')).toBeTruthy();
  });

  it('does not seed training incidents or simulated detections', async () => {
    const { MATScreen } = require('@/screens/MATScreen');
    const view = render(<ThemeProvider><MATScreen /></ThemeProvider>);
    expect(view.queryByText('Training Scenario')).toBeNull();
    expect(view.queryByText('SIMULATED DATA')).toBeNull();
    expect(useMatStore.getState().survivors).toEqual([]);
    await waitFor(() => expect(mockFetchSnapshot).toHaveBeenCalled());
  });
});
