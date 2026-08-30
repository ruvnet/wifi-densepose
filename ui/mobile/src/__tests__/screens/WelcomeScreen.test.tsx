import React from 'react';
import { fireEvent, render } from '@testing-library/react-native';
import { ThemeProvider } from '@/theme/ThemeContext';
import { usePoseStore } from '@/stores/poseStore';
import { useMatStore } from '@/stores/matStore';

const mockNavigate = jest.fn();
jest.mock('@react-navigation/native', () => ({ useNavigation: () => ({ navigate: mockNavigate }) }));
jest.mock('@expo/vector-icons', () => {
  const { View } = require('react-native');
  return { Ionicons: View };
});

describe('WelcomeScreen', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    usePoseStore.getState().reset();
    useMatStore.getState().reset();
  });

  it('presents every real workspace and starts with calibration', () => {
    const { WelcomeScreen } = require('@/screens/WelcomeScreen');
    const view = render(<ThemeProvider><WelcomeScreen /></ThemeProvider>);
    expect(view.getByTestId('welcome-hero')).toBeTruthy();
    for (const route of ['Calibration', 'Live', 'Vitals', 'Zones', 'MAT', 'Settings']) expect(view.getByTestId(`welcome-open-${route.toLowerCase()}`)).toBeTruthy();
    fireEvent.press(view.getByTestId('welcome-primary-action'));
    expect(mockNavigate).toHaveBeenCalledWith('Calibration');
  });

  it('reports source state instead of inventing readiness', () => {
    const { WelcomeScreen } = require('@/screens/WelcomeScreen');
    const view = render(<ThemeProvider><WelcomeScreen /></ThemeProvider>);
    expect(view.getByText('RF DISCONNECTED')).toBeTruthy();
    expect(view.getByText('MAT IDLE')).toBeTruthy();
  });
});
