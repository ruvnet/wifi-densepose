import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react-native';
import { ThemeProvider } from '@/theme/ThemeContext';
import { usePoseStore } from '@/stores/poseStore';

const mockConnectSpaces = jest.fn();
const mockBootstrap = jest.fn(async () => undefined);
const mockRefreshSpatial = jest.fn(async () => undefined);
const mockAnalyze = jest.fn(async () => undefined);
const mockCompleteAuthorization = jest.fn(async () => undefined);
const mockCognitumState = {
  authStatus: 'signed_out', dataStatus: 'idle', session: { signedIn: false, scopes: [] }, inferenceAuthorized: false, spaces: [], zones: [], boundary: null, error: null, cloudConsentAt: null, insight: null,
  bootstrap: mockBootstrap, connectSpaces: mockConnectSpaces, completeAuthorization: mockCompleteAuthorization, disconnect: jest.fn(), refreshSpatial: mockRefreshSpatial, setCloudConsent: jest.fn(), analyze: mockAnalyze,
};

jest.mock('@expo/vector-icons', () => { const { View } = require('react-native'); return { Ionicons: View }; });
jest.mock('react-native-svg', () => { const { View } = require('react-native'); return { __esModule: true, default: View, Svg: View, Circle: View, G: View, Text: View, Rect: View, Line: View, Path: View, Polygon: View }; });
jest.mock('@/stores/cognitumStore', () => ({ useCognitumStore: () => mockCognitumState }));
jest.mock('@/services/worldGraph.service', () => ({ worldGraphService: { fetchSnapshot: jest.fn(() => new Promise(() => undefined)) } }));
jest.mock('@/screens/ZonesScreen/FloorPlanSvg', () => { const { View } = require('react-native'); return { FloorPlanSvg: (props: any) => require('react').createElement(View, { testID: 'floor-plan', ...props }) }; });
jest.mock('@/screens/ZonesScreen/ZoneLegend', () => { const { View } = require('react-native'); return { ZoneLegend: () => require('react').createElement(View, { testID: 'zone-legend' }) }; });
jest.mock('@/screens/ZonesScreen/useOccupancyGrid', () => ({ useOccupancyGrid: () => ({ gridValues: new Array(400).fill(0), personPositions: [] }) }));

const renderScreen = () => {
  const { ZonesScreen } = require('@/screens/ZonesScreen');
  return render(<ThemeProvider><ZonesScreen /></ThemeProvider>);
};

describe('ZonesScreen', () => {
  beforeEach(() => { usePoseStore.getState().reset(); jest.clearAllMocks(); mockCognitumState.authStatus = 'signed_out'; });

  it('exports both component forms', () => {
    const mod = require('@/screens/ZonesScreen');
    expect(typeof mod.ZonesScreen).toBe('function'); expect(mod.default).toBeDefined();
  });

  it('renders the source-honest spatial workspace and local field', async () => {
    renderScreen();
    expect(screen.getByTestId('zones-hero')).toBeTruthy();
    expect(screen.getByText(/A room is more than/)).toBeTruthy();
    expect(screen.getByText('ANONYMOUS OCCUPANCY FIELD')).toBeTruthy();
    expect(screen.getByText('LOCAL CSI · NO CAMERA · NO IDENTITY')).toBeTruthy();
    expect(screen.getByTestId('floor-plan')).toBeTruthy();
    expect(screen.getByText('Waiting for a verified sensing frame. The field remains empty rather than synthesizing occupancy.')).toBeTruthy();
    await waitFor(() => expect(mockBootstrap).toHaveBeenCalled());
  });

  it('switches between the functional room layers', () => {
    renderScreen();
    fireEvent.press(screen.getByText('TOPOLOGY'));
    expect(screen.getByTestId('zones-topology-layer')).toBeTruthy();
    fireEvent.press(screen.getByRole('button', { name: 'SPACES layer' }));
    expect(screen.getByTestId('zones-spaces-layer')).toBeTruthy();
    expect(screen.getByText('SPACES IS PRIVATE BY DEFAULT')).toBeTruthy();
  });

  it('offers Cognitum OAuth and keeps cloud interpretation disabled by default', () => {
    renderScreen();
    fireEvent.press(screen.getByText('SIGN IN FOR SPACES'));
    expect(mockConnectSpaces).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText('Cloud spatial interpretation').props.value).toBe(false);
    expect(screen.getByRole('button', { name: 'AUTHORIZE COGNITUM INFERENCE' }).props.accessibilityState.disabled).toBe(true);
  });

  it('completes Cognitum PKCE authorization from the one-time code', async () => {
    mockCognitumState.authStatus = 'awaiting_code';
    renderScreen();
    fireEvent.changeText(screen.getByLabelText('Cognitum one-time code'), 'code-123');
    fireEvent.press(screen.getByRole('button', { name: 'Complete Cognitum sign-in' }));
    await waitFor(() => expect(mockCompleteAuthorization).toHaveBeenCalledWith('code-123'));
  });

  it('makes the semantic privacy boundary explicit', () => {
    renderScreen();
    expect(screen.getByText(/Raw CSI, pose frames, vital waveforms, recordings, and identity observations remain excluded/)).toBeTruthy();
    expect(screen.getByText('NONE')).toBeTruthy();
    expect(screen.getByText('IDENTITY')).toBeTruthy();
  });
});
