jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
);

jest.mock('react-native-wifi-reborn', () => ({
  loadWifiList: jest.fn(async () => []),
}));

jest.mock('react-native/Libraries/Components/Switch/Switch', () => {
  const React = require('react');
  const { View } = require('react-native');

  const MockSwitch = (props: Record<string, unknown>) => {
    return React.createElement(View, { accessibilityRole: 'switch', ...props });
  };

  return {
    __esModule: true,
    default: MockSwitch,
  };
});

jest.mock('react-native-gesture-handler', () => {
  const React = require('react');
  const { View } = require('react-native');

  const Passthrough = ({ children, ...props }: { children?: unknown; [key: string]: unknown }) =>
    React.createElement(View, props, children);

  const createGesture = (): Record<string, jest.Mock> => {
    const gesture: Record<string, jest.Mock> = {};
    gesture.onStart = jest.fn(() => gesture);
    gesture.onUpdate = jest.fn(() => gesture);
    gesture.onEnd = jest.fn(() => gesture);
    return gesture;
  };

  return {
    GestureHandlerRootView: Passthrough,
    GestureDetector: Passthrough,
    Gesture: {
      Pan: jest.fn(createGesture),
    },
  };
});

jest.mock('react-native-reanimated', () => {
  const { View } = require('react-native');

  const createAnimatedComponent = (component: unknown) => component;
  const animated = {
    View,
    createAnimatedComponent,
  };
  const identity = (value: unknown) => value;

  return {
    __esModule: true,
    default: animated,
    cancelAnimation: jest.fn(),
    createAnimatedComponent,
    Easing: {
      cubic: identity,
      in: identity,
      inOut: identity,
      linear: identity,
      out: identity,
      quad: identity,
    },
    interpolateColor: (_value: unknown, _input: unknown, output: string[]) => output[0],
    runOnJS: (fn: (...args: unknown[]) => unknown) => fn,
    useAnimatedReaction: jest.fn(),
    useAnimatedProps: (factory: () => unknown) => factory(),
    useAnimatedStyle: (factory: () => unknown) => factory(),
    useDerivedValue: (factory: () => unknown) => factory(),
    useSharedValue: (value: unknown) => ({ value }),
    withRepeat: identity,
    withSequence: (...values: unknown[]) => values[values.length - 1],
    withSpring: identity,
    withTiming: identity,
  };
});

jest.mock('react-native-webview', () => {
  const React = require('react');
  const { View } = require('react-native');

  const MockWebView = (props: unknown) => React.createElement(View, props);

  return {
    __esModule: true,
    default: MockWebView,
    WebView: MockWebView,
  };
});
