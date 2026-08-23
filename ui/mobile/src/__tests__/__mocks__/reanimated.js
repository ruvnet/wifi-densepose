const ReactNative = require('react-native');

const identity = (value) => value;
const noop = () => undefined;
const createSharedValue = (initialValue) => ({
  value: initialValue,
  get: () => initialValue,
  set(nextValue) {
    this.value = typeof nextValue === 'function' ? nextValue(this.value) : nextValue;
  },
});
const evaluate = (updater) => updater();
const createAnimatedComponent = (Component) => Component;

const Easing = {
  linear: identity,
  ease: identity,
  quad: identity,
  cubic: identity,
  in: identity,
  out: identity,
  inOut: identity,
};

const Animated = {
  View: ReactNative.View,
  Text: ReactNative.Text,
  Image: ReactNative.Image,
  ScrollView: ReactNative.ScrollView,
  createAnimatedComponent,
};

module.exports = {
  __esModule: true,
  default: Animated,
  Easing,
  cancelAnimation: noop,
  createAnimatedComponent,
  interpolateColor: (_value, _input, output) => output[0],
  runOnJS: identity,
  useAnimatedProps: evaluate,
  useAnimatedReaction: noop,
  useAnimatedStyle: evaluate,
  useDerivedValue: (updater) => createSharedValue(updater()),
  useSharedValue: createSharedValue,
  withRepeat: identity,
  withSequence: (...values) => values[values.length - 1],
  withSpring: identity,
  withTiming: identity,
};
