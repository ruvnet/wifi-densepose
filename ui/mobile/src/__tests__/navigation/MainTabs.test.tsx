import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react-native';
import { AppTabBar } from '@/navigation/MainTabs';

jest.mock('@expo/vector-icons', () => ({
  Ionicons: require('react-native').View,
}));

jest.mock('@/screens/LiveScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));
jest.mock('@/screens/NLOSScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));
jest.mock('@/screens/VitalsScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));
jest.mock('@/screens/ZonesScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));
jest.mock('@/screens/MATScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));
jest.mock('@/screens/SettingsScreen', () => ({
  __esModule: true,
  default: require('react-native').View,
}));

const routes = ['Live', 'NLOS', 'Vitals', 'Zones', 'MAT', 'Settings'].map((name) => ({
  key: `${name}-key`,
  name,
  params: undefined,
}));

const descriptors = Object.fromEntries(
  routes.map((route) => [route.key, { options: {} }]),
);

const renderTabBar = (activeIndex = 1, defaultPrevented = false) => {
  const navigation = {
    emit: jest.fn(() => ({ defaultPrevented })),
    navigate: jest.fn(),
  };

  render(
    <AppTabBar
      state={{ index: activeIndex, routes } as any}
      descriptors={descriptors as any}
      navigation={navigation as any}
      insets={{ top: 0, right: 0, bottom: 34, left: 0 }}
      matAlertCount={3}
    />,
  );

  return navigation;
};

describe('AppTabBar', () => {
  it('renders a full touch target for every route', () => {
    renderTabBar();

    for (const route of routes) {
      expect(screen.getByTestId(`tab-${route.name.toLowerCase()}`)).toBeTruthy();
    }
    expect(screen.getByText('3')).toBeTruthy();
  });

  it('emits tabPress and navigates when an inactive tab is pressed', () => {
    const navigation = renderTabBar();

    fireEvent.press(screen.getByTestId('tab-live'));

    expect(navigation.emit).toHaveBeenCalledWith({
      type: 'tabPress',
      target: 'Live-key',
      canPreventDefault: true,
    });
    expect(navigation.navigate).toHaveBeenCalledWith('Live', undefined);
  });

  it('does not navigate when tabPress is prevented', () => {
    const navigation = renderTabBar(1, true);

    fireEvent.press(screen.getByTestId('tab-settings'));

    expect(navigation.navigate).not.toHaveBeenCalled();
  });

  it('does not re-navigate when the active tab is pressed', () => {
    const navigation = renderTabBar(1);

    fireEvent.press(screen.getByTestId('tab-nlos'));

    expect(navigation.emit).toHaveBeenCalled();
    expect(navigation.navigate).not.toHaveBeenCalled();
  });
});
