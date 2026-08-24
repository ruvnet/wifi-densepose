import React from 'react';
import { Linking, StyleSheet } from 'react-native';
import { fireEvent, render, screen } from '@testing-library/react-native';
import { createSyntheticNlosFrame } from '@/services/nlos.service';
import { createLiveNlosFrameFixture } from '@/testUtils/nlosFixtures';
import { ThemeProvider } from '@/theme/ThemeContext';
import { typography } from '@/theme/typography';
import {
  getBetaPlatformGuidance,
  NLOS_EXPLAINER_URL,
  NLOS_FEEDBACK_URL,
} from '@/screens/NLOSScreen/BetaSetupCard';
import { resolveNlosEvidenceState } from '@/screens/NLOSScreen/ProvenancePanel';
import {
  buildLidarPointCloud,
  LIDAR_POINTS_PER_TRACK,
  LIDAR_RELAY_POINT_COUNT,
  resolveLidarTrackCenter,
} from '@/screens/NLOSScreen/lidarPointCloudData';

const mockSafeAreaInsets = { top: 0, right: 0, bottom: 0, left: 0 };

jest.mock('react-native-safe-area-context', () => ({
  ...jest.requireActual('react-native-safe-area-context'),
  useSafeAreaInsets: () => mockSafeAreaInsets,
}));

const syntheticFrame = createSyntheticNlosFrame(0, 1_700_000_000_000);
const mockNlosResult: Record<string, any> = {
  frame: syntheticFrame,
  freshness: 'fresh' as const,
  streamStatus: 'synthetic_replay' as const,
  lastRejectedReason: null,
  rejectedFrameCount: 0,
  liveCredentialAvailable: false,
  configureCredential: jest.fn(() => true),
  forgetCredential: jest.fn(),
  startReplay: jest.fn(),
  connectLive: jest.fn(),
};

jest.mock('@/hooks/useNlosStream', () => ({
  useNlosStream: () => mockNlosResult,
}));

jest.mock('react-native-svg', () => {
  const { View, Text } = require('react-native');
  return {
    __esModule: true,
    default: View,
    Circle: View,
    Ellipse: View,
    Line: View,
    Polygon: View,
    Rect: View,
    Text,
  };
});

describe('NLOSScreen', () => {
  beforeEach(() => {
    Object.assign(mockSafeAreaInsets, { top: 0, right: 0, bottom: 0, left: 0 });
    Object.assign(mockNlosResult, {
      frame: syntheticFrame,
      freshness: 'fresh',
      streamStatus: 'synthetic_replay',
      lastRejectedReason: null,
      rejectedFrameCount: 0,
      liveCredentialAvailable: false,
    });
    mockNlosResult.configureCredential.mockClear();
    mockNlosResult.forgetCredential.mockClear();
    mockNlosResult.startReplay.mockClear();
    mockNlosResult.connectLive.mockClear();
  });

  it('uses the pinned Cognitum typography roles', () => {
    expect(typography.displayLg.fontFamily).toBe('Outfit_700Bold');
    expect(typography.bodyMd.fontFamily).toBe('Outfit_400Regular');
    expect(typography.mono.fontFamily).toBe('JetBrainsMono_500Medium');
  });

  it('renders the RuView NLOS screen and iPhone API boundary', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByText('Consumer NLOS / field viewer')).toBeTruthy();
    expect(screen.getByText(/does not access raw iPhone LiDAR timing data/)).toBeTruthy();
    expect(screen.getByText(/web client cannot capture ARKit LiDAR or raw timing data/)).toBeTruthy();
  });

  it('keeps the instrument content inside supplied iPhone safe area insets', () => {
    Object.assign(mockSafeAreaInsets, { top: 47, right: 3, bottom: 34, left: 3 });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    const contentStyle = StyleSheet.flatten(
      screen.getByTestId('nlos-scroll-view').props.contentContainerStyle,
    );
    expect(contentStyle.paddingTop).toBe(16);
    expect(contentStyle.paddingRight).toBe(19);
    expect(contentStyle.paddingBottom).toBe(106);
    expect(contentStyle.paddingLeft).toBe(19);
  });

  it('provides platform-specific beta setup without browser API assumptions', () => {
    const ios = getBetaPlatformGuidance('ios');
    const web = getBetaPlatformGuidance('web');

    expect(ios.label).toBe('NATIVE IOS BETA');
    expect(ios.showTestFlightButton).toBe(true);
    expect(ios.steps.join(' ')).toMatch(/TestFlight/);
    expect(web.label).toBe('IPHONE WEB BETA');
    expect(web.showTestFlightButton).toBe(false);
    expect(web.steps.join(' ')).toMatch(/Safari/);
    expect(web.steps.join(' ')).toMatch(/Add to Home Screen/);
  });

  it('exposes the explainer, feedback issue, and evidence boundary', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(NLOS_EXPLAINER_URL).toBe('https://ruview-nlos.ruv.chatgpt.site');
    expect(NLOS_FEEDBACK_URL).toBe('https://github.com/ruvnet/RuView/issues/1690');
    expect(screen.getByRole('link', { name: 'OPEN EXPLAINER' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'TEST STEPS AND FEEDBACK' })).toBeTruthy();
    expect(screen.getByText(/Depth only input is never physical NLOS evidence/)).toBeTruthy();
    expect(screen.getByText(/No credentials are saved by setup/)).toBeTruthy();
  });

  it('opens only the fixed explainer and feedback links', () => {
    const openUrl = jest.spyOn(Linking, 'openURL').mockResolvedValue(undefined);
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    fireEvent.press(screen.getByRole('link', { name: 'OPEN EXPLAINER' }));
    fireEvent.press(screen.getByRole('link', { name: 'TEST STEPS AND FEEDBACK' }));

    expect(openUrl).toHaveBeenNthCalledWith(1, NLOS_EXPLAINER_URL);
    expect(openUrl).toHaveBeenNthCalledWith(2, NLOS_FEEDBACK_URL);
    openUrl.mockRestore();
  });

  it('always watermarks synthetic replay', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByTestId('nlos-synthetic-watermark')).toBeTruthy();
    expect(screen.getByTestId('nlos-provenance-badge').props.children).toBe('SYNTHETIC');
    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('SYNTHETIC');
  });

  it('starts deterministic replay from the primary synthetic control', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    fireEvent.press(screen.getByTestId('nlos-start-synthetic'));

    expect(mockNlosResult.startReplay).toHaveBeenCalledTimes(1);
  });

  it('does not enable live without an ephemeral credential', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    const button = screen.getByRole('button', { name: 'CONNECT AUTHENTICATED LIVE' });
    expect(button.props.accessibilityState?.disabled ?? button.props.disabled).toBeTruthy();
    expect(screen.getByText(/never stored by this client/)).toBeTruthy();
  });

  it('keeps a manually entered pairing credential bounded and masked', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    const input = screen.getByTestId('nlos-credential-input');
    expect(input.props.secureTextEntry).toBe(true);
    expect(input.props.maxLength).toBe(512);
    const unlock = screen.getByRole('button', { name: 'UNLOCK AUTHENTICATED LIVE' });
    expect(unlock.props.accessibilityState?.disabled ?? unlock.props.disabled).toBeTruthy();

    const token = 'p'.repeat(32);
    fireEvent.changeText(input, token);
    fireEvent.press(screen.getByRole('button', { name: 'UNLOCK AUTHENTICATED LIVE' }));
    expect(mockNlosResult.configureCredential).toHaveBeenCalledWith(token);
    expect(screen.getByTestId('nlos-credential-input').props.value).toBe('');
  });

  it('renders unknown evidence without a live or synthetic claim', () => {
    Object.assign(mockNlosResult, { frame: null, freshness: 'unknown', streamStatus: 'idle' });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByTestId('nlos-provenance-badge').props.children).toBe('UNKNOWN');
    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('DISCONNECTED');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.queryByText('target-1')).toBeNull();
    expect(screen.queryByTestId('nlos-synthetic-watermark')).toBeNull();
    expect(screen.getByText(/Unknown evidence is never promoted to live/)).toBeTruthy();
  });

  it('keeps stale measured frames visibly stale', () => {
    const live = createLiveNlosFrameFixture();
    Object.assign(mockNlosResult, {
      frame: live,
      freshness: 'stale',
      streamStatus: 'error',
      liveCredentialAvailable: true,
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByTestId('nlos-stale-overlay')).toBeTruthy();
    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('STALE');
    expect(screen.getByTestId('nlos-freshness-badge').props.children).toBe('STALE');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.getByTestId('nlos-mean-confidence').props.children).toBe('N/A');
    expect(screen.queryByText(live.tracks[0].trackId)).toBeNull();
    expect(screen.queryByTestId('nlos-synthetic-watermark')).toBeNull();
  });

  it('never draws or counts unknown target hypotheses', () => {
    const live = createLiveNlosFrameFixture();
    Object.assign(mockNlosResult, {
      frame: {
        ...live,
        tracks: [{ ...live.tracks[0], state: 'unknown' }],
      },
      freshness: 'fresh',
      streamStatus: 'live',
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.getByTestId('nlos-mean-confidence').props.children).toBe('N/A');
    expect(screen.queryByText(live.tracks[0].trackId)).toBeNull();
  });

  it('distinguishes verified and unverified live evidence without changing source labels', () => {
    const calibrated = createLiveNlosFrameFixture();
    const measured = createLiveNlosFrameFixture({ evidenceLevel: 'l1_measured' });

    expect(resolveNlosEvidenceState(calibrated, 'fresh', 'live')).toBe('LIVE VERIFIED');
    expect(resolveNlosEvidenceState(measured, 'fresh', 'live')).toBe('LIVE UNVERIFIED');
    expect(resolveNlosEvidenceState(calibrated, 'stale', 'error')).toBe('STALE');
    expect(resolveNlosEvidenceState(null, 'unknown', 'idle')).toBe('DISCONNECTED');
    expect(resolveNlosEvidenceState(null, 'unknown', 'connecting')).toBe('LIVE UNVERIFIED');
  });

  it('withholds geometry for unverified live and replay provenance', () => {
    const live = createLiveNlosFrameFixture({ evidenceLevel: 'l1_measured' });
    Object.assign(mockNlosResult, {
      frame: live,
      freshness: 'fresh',
      streamStatus: 'live',
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    const view = render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('LIVE UNVERIFIED');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.queryByText(live.tracks[0].trackId)).toBeNull();
    fireEvent.press(screen.getByTestId('nlos-view-cloud'));
    expect(screen.getByTestId('nlos-cloud-target-count').props.children).toBe(0);

    Object.assign(mockNlosResult, {
      frame: { ...live, source: 'replay' },
      freshness: 'fresh',
      streamStatus: 'live',
    });
    view.rerender(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('DISCONNECTED');
    expect(screen.getByTestId('nlos-provenance-badge').props.children).toBe('REPLAY');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.queryByText(live.tracks[0].trackId)).toBeNull();
  });

  it('projects a no-frame live attempt as unverified while withholding geometry', () => {
    Object.assign(mockNlosResult, {
      frame: null,
      freshness: 'unknown',
      streamStatus: 'connecting',
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('LIVE UNVERIFIED');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.queryByText('target-1')).toBeNull();
  });

  it('shows geometry only after a fresh live frame passes the verification gate', () => {
    const live = createLiveNlosFrameFixture();
    Object.assign(mockNlosResult, {
      frame: live,
      freshness: 'fresh',
      streamStatus: 'live',
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-evidence-state').props.children).toBe('LIVE VERIFIED');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(1);
    expect(screen.getByText(live.tracks[0].trackId)).toBeTruthy();
  });

  it('does not override validator accepted USB sensor provenance in the UI', () => {
    const base = createLiveNlosFrameFixture();
    const usbFrame = createLiveNlosFrameFixture({
      provenance: { ...base.provenance, transport: 'usb_serial' },
    });

    expect(resolveNlosEvidenceState(usbFrame, 'fresh', 'live')).toBe('LIVE VERIFIED');
  });

  it('builds a deterministic bounded LiDAR reconstruction cloud from gated tracks', () => {
    const first = buildLidarPointCloud(syntheticFrame.tracks);
    const second = buildLidarPointCloud(syntheticFrame.tracks);

    expect(first.relayPointCount).toBe(LIDAR_RELAY_POINT_COUNT);
    expect(first.targetPointCount).toBe(syntheticFrame.tracks.length * LIDAR_POINTS_PER_TRACK);
    expect(first.totalPointCount).toBe(first.relayPointCount + first.targetPointCount);
    expect(Array.from(first.positions)).toEqual(Array.from(second.positions));
    expect(Array.from(first.colors)).toEqual(Array.from(second.colors));
    expect(Array.from(first.positions).every(Number.isFinite)).toBe(true);
    expect(resolveLidarTrackCenter(syntheticFrame.tracks[0]).every(Number.isFinite)).toBe(true);
  });

  it('keeps privacy, setup, explainer, and feedback controls visible in the screen tree', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-privacy-legend')).toBeTruthy();
    expect(screen.getByTestId('nlos-beta-setup')).toBeTruthy();
    expect(screen.getByTestId('nlos-explainer-link')).toBeTruthy();
    expect(screen.getByTestId('nlos-feedback-link')).toBeTruthy();
    expect(screen.getByText(/Viewer retention: raw RF off, audio off/)).toBeTruthy();
  });

  it('switches between plan and perspective instrument views', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);

    expect(screen.getByTestId('nlos-view-plan').props.accessibilityState.selected).toBe(true);
    fireEvent.press(screen.getByTestId('nlos-view-perspective'));
    expect(screen.getByTestId('nlos-view-perspective').props.accessibilityState.selected).toBe(true);
    fireEvent.press(screen.getByTestId('nlos-view-cloud'));
    expect(screen.getByTestId('nlos-view-cloud').props.accessibilityState.selected).toBe(true);
    expect(screen.getByTestId('nlos-lidar-point-cloud')).toBeTruthy();
    expect(screen.getByText('RECONSTRUCTION / NOT RAW SCAN')).toBeTruthy();
    expect(screen.getByTestId('nlos-cloud-target-count').props.children).toBe(
      syntheticFrame.tracks.length * LIDAR_POINTS_PER_TRACK,
    );
  });
});
