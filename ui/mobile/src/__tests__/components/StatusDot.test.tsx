import React from 'react';
import { render } from '@testing-library/react-native';
import { StatusDot } from '@/components/StatusDot';
import { ThemeProvider } from '@/theme/ThemeContext';

const renderWithTheme = (ui: React.ReactElement) =>
  render(<ThemeProvider>{ui}</ThemeProvider>);

describe('StatusDot', () => {
  it('renders without crashing for connected status', () => {
    const { toJSON } = renderWithTheme(<StatusDot status="connected" />);
    expect(toJSON()).not.toBeNull();
  });

  it('renders without crashing for disconnected status', () => {
    const { toJSON } = renderWithTheme(<StatusDot status="disconnected" />);
    expect(toJSON()).not.toBeNull();
  });

  it('renders without crashing for connecting status', () => {
    const { toJSON } = renderWithTheme(<StatusDot status="connecting" />);
    expect(toJSON()).not.toBeNull();
  });

  it('renders with custom size', () => {
    const { toJSON } = renderWithTheme(
      <StatusDot status="connected" size={20} />,
    );
    expect(toJSON()).not.toBeNull();
  });

  it('renders all statuses without error', () => {
    const statuses: Array<'connected' | 'disconnected' | 'connecting'> = [
      'connected',
      'disconnected',
      'connecting',
    ];
    for (const status of statuses) {
      const { unmount } = renderWithTheme(<StatusDot status={status} />);
      unmount();
    }
  });
});
