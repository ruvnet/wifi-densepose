import React from 'react';
import { Linking } from 'react-native';
import { fireEvent, render, screen } from '@testing-library/react-native';
import { createSyntheticNlosFrame } from '@/services/nlos.service';
import { createLiveNlosFrameFixture } from '@/testUtils/nlosFixtures';
import { ThemeProvider } from '@/theme/ThemeContext';
import {
  getBetaPlatformGuidance,
  NLOS_EXPLAINER_URL,
  NLOS_FEEDBACK_URL,
} from '@/screens/NLOSScreen/BetaSetupCard';

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
  });

  it('renders the RuView NLOS screen and iPhone API boundary', () => {
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByText('RuView NLOS')).toBeTruthy();
    expect(screen.getByText(/does not access raw iPhone LiDAR timing data/)).toBeTruthy();
    expect(screen.getByText(/web client cannot capture ARKit LiDAR or raw timing data/)).toBeTruthy();
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
    expect(screen.queryByTestId('nlos-synthetic-watermark')).toBeNull();
    expect(screen.getByText(/Unknown evidence is never promoted to live/)).toBeTruthy();
  });

  it('keeps stale measured frames visibly stale', () => {
    Object.assign(mockNlosResult, {
      frame: createLiveNlosFrameFixture(),
      freshness: 'stale',
      streamStatus: 'error',
      liveCredentialAvailable: true,
    });
    const { NLOSScreen } = require('@/screens/NLOSScreen');
    render(<ThemeProvider><NLOSScreen /></ThemeProvider>);
    expect(screen.getByTestId('nlos-stale-overlay')).toBeTruthy();
    expect(screen.getByTestId('nlos-freshness-badge').props.children).toBe('STALE');
    expect(screen.getByTestId('nlos-track-count').props.children).toBe(0);
    expect(screen.getByTestId('nlos-mean-confidence').props.children).toBe('N/A');
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
});
