import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react-native';
import { ThemeProvider } from '@/theme/ThemeContext';
import { apiService } from '@/services/api.service';
import { wsService } from '@/services/ws.service';
import { useSettingsStore } from '@/stores/settingsStore';

jest.mock('@/services/ws.service', () => ({
  wsService: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    subscribe: jest.fn(() => jest.fn()),
    getStatus: jest.fn(() => 'disconnected'),
  },
}));

jest.mock('@/services/api.service', () => ({
  apiService: {
    setBaseUrl: jest.fn(),
    get: jest.fn(),
    post: jest.fn(),
    getStatus: jest.fn(),
  },
}));

describe('SettingsScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useSettingsStore.setState({
      serverUrl: 'http://localhost:3000',
      nlosServerUrl: 'http://localhost:3000',
      rssiScanEnabled: false,
      theme: 'system',
      alertSoundEnabled: true,
    });
  });

  it('module exports SettingsScreen component', () => {
    const mod = require('@/screens/SettingsScreen');
    expect(mod.SettingsScreen).toBeDefined();
    expect(typeof mod.SettingsScreen).toBe('function');
  });

  it('default export is also available', () => {
    const mod = require('@/screens/SettingsScreen');
    expect(mod.default).toBeDefined();
  });

  it('renders without crashing', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    const { toJSON } = render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    expect(toJSON()).not.toBeNull();
  });

  it('renders the SERVER section', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    expect(screen.getByText('SERVER')).toBeTruthy();
  });

  it('renders the SENSING section', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    expect(screen.getByText('SENSING')).toBeTruthy();
  });

  it('renders the ABOUT section with version', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    expect(screen.getByText('ABOUT')).toBeTruthy();
    expect(screen.getByText('WiFi-DensePose Mobile v1.0.0')).toBeTruthy();
  });

  it('saves the sensing server and reconnects dependent services', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );

    const [serverInput] = screen.getAllByDisplayValue('http://localhost:3000');
    fireEvent.changeText(serverInput, 'http://192.168.1.10:8080');
    fireEvent.press(screen.getByText('Save'));

    expect(useSettingsStore.getState().serverUrl).toBe('http://192.168.1.10:8080');
    expect(wsService.disconnect).toHaveBeenCalledTimes(1);
    expect(wsService.connect).toHaveBeenCalledWith('http://192.168.1.10:8080');
    expect(apiService.setBaseUrl).toHaveBeenCalledWith('http://192.168.1.10:8080');
  });

  it('saves a separate NLOS endpoint', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );

    fireEvent.changeText(screen.getByTestId('nlos-server-url-input'), 'https://nlos.example.com');
    fireEvent.press(screen.getByText('Save NLOS server'));

    expect(useSettingsStore.getState().nlosServerUrl).toBe('https://nlos.example.com');
  });

  it('updates sensing and appearance controls', () => {
    const { SettingsScreen } = require('@/screens/SettingsScreen');
    render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );

    fireEvent(screen.getByRole('switch'), 'valueChange', true);
    fireEvent.press(screen.getByText('5s'));
    fireEvent.press(screen.getByText('DARK'));

    expect(useSettingsStore.getState().rssiScanEnabled).toBe(true);
    expect(useSettingsStore.getState().theme).toBe('dark');
    expect(screen.getByText('Active interval: 5s')).toBeTruthy();
  });
});
