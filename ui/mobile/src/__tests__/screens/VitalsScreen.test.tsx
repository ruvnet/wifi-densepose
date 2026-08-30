import React from 'react';
import { act, render, screen } from '@testing-library/react-native';
import { ThemeProvider } from '@/theme/ThemeContext';
import { usePoseStore } from '@/stores/poseStore';
import type { SensingFrame } from '@/types/sensing';

jest.mock('@/hooks/usePoseStream', () => ({
  usePoseStream: () => {
    const state = require('@/stores/poseStore').usePoseStore.getState();
    return { connectionStatus: state.connectionStatus, lastFrame: state.lastFrame, isSimulated: state.isSimulated };
  },
}));

jest.mock('@expo/vector-icons', () => {
  const { View } = require('react-native');
  return { Ionicons: View };
});

jest.mock('react-native-svg', () => {
  const { View } = require('react-native');
  return {
    __esModule: true,
    default: View,
    Svg: View,
    Circle: View,
    G: View,
    Text: View,
    Rect: View,
    Line: View,
    Path: View,
    Defs: View,
    LinearGradient: View,
    Stop: View,
  };
});

const frame = (overrides: Partial<SensingFrame> = {}): SensingFrame => ({
  timestamp: Date.now(),
  source: 'esp32-csi',
  nodes: [{ node_id: 7, rssi_dbm: -54, position: [0, 0, 0] }],
  features: { mean_rssi: -54, variance: 2.4, motion_band_power: 0.42, breathing_band_power: 0.18, spectral_entropy: 0.61 },
  classification: { motion_level: 'present_moving', presence: true, confidence: 0.91 },
  signal_field: { grid_size: [1, 1, 1], values: [0.2] },
  signal_quality_score: 0.87,
  ...overrides,
});

const renderVitals = () => {
  const VitalsScreen = require('@/screens/VitalsScreen').default;
  return render(<ThemeProvider><VitalsScreen /></ThemeProvider>);
};

describe('VitalsScreen', () => {
  beforeEach(() => usePoseStore.getState().reset());

  it('exports and renders the redesigned screen', () => {
    const mod = require('@/screens/VitalsScreen');
    expect(typeof mod.default).toBe('function');
    expect(renderVitals().toJSON()).not.toBeNull();
    expect(screen.getByText('The room has a pulse.')).toBeTruthy();
    expect(screen.getByText('Apple Home / local HAP bridge')).toBeTruthy();
  });

  it('fails closed for simulated input instead of showing invented vitals', () => {
    const simulated = frame({ source: 'simulated', vital_signs: { breathing_bpm: 19, hr_proxy_bpm: 73 } });
    usePoseStore.setState({ connectionStatus: 'simulated', isSimulated: true, lastFrame: simulated, features: simulated.features, classification: simulated.classification });
    renderVitals();
    expect(screen.getByText('SIMULATION HIDDEN')).toBeTruthy();
    expect(screen.queryByText('19.0 BPM')).toBeNull();
    expect(screen.queryByText('73.0 BPM')).toBeNull();
    expect(screen.getAllByLabelText('WAITING: --')).toHaveLength(2);
  });

  it('shows only explicit vitals from a fresh connected frame', () => {
    const live = frame({ vital_signs: { breathing_rate_bpm: 18, breathing_confidence: 0.82, heart_rate_bpm: 72, heart_confidence: 0.76 } });
    usePoseStore.setState({ connectionStatus: 'connected', isSimulated: false, lastFrame: live, features: live.features, classification: live.classification });
    renderVitals();
    expect(screen.getByText('MEASURED / FRESH')).toBeTruthy();
    expect(screen.getByLabelText('RF ESTIMATE: 18.0 BPM')).toBeTruthy();
    expect(screen.getByLabelText('RF PROXY: 72.0 BPM')).toBeTruthy();
    expect(screen.getByText('82%')).toBeTruthy();
    expect(screen.getByText('76%')).toBeTruthy();
  });

  it('automatically withdraws measurements when a live stream stops updating', () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-08-24T18:00:00.000Z'));
    const live = frame({ vital_signs: { breathing_rate_bpm: 18, heart_rate_bpm: 72 } });
    usePoseStore.setState({ connectionStatus: 'connected', isSimulated: false, lastFrame: live, features: live.features, classification: live.classification });
    const view = renderVitals();

    expect(screen.getByText('MEASURED / FRESH')).toBeTruthy();
    act(() => jest.advanceTimersByTime(4_000));
    expect(screen.getByText('NO FRESH EVIDENCE')).toBeTruthy();
    expect(screen.queryByLabelText('RF ESTIMATE: 18.0 BPM')).toBeNull();
    expect(screen.queryByLabelText('RF PROXY: 72.0 BPM')).toBeNull();

    view.unmount();
    jest.useRealTimers();
  });

  it('does not derive BPM from signal bands when explicit vitals are absent', () => {
    const live = frame();
    usePoseStore.setState({ connectionStatus: 'connected', isSimulated: false, lastFrame: live, features: live.features, classification: live.classification });
    renderVitals();
    expect(screen.queryByText(/\d+\.\d BPM/)).toBeNull();
    expect(screen.getByText('No measured rate')).toBeTruthy();
    expect(screen.getByText('No measured proxy')).toBeTruthy();
  });

  it('states the real Apple Home privacy boundary', () => {
    renderVitals();
    expect(screen.getByText('HomePod or Apple TV acts as the Home Hub. RuView remains the sensor.')).toBeTruthy();
    expect(screen.getByText(/Breathing, heart-rate proxy, pose, raw CSI, and identity scores never cross this boundary/)).toBeTruthy();
  });
});
