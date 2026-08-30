import React, { useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, Animated, Easing, Pressable, StyleSheet, View } from 'react-native';
import {
  createBottomTabNavigator,
  type BottomTabBarProps,
} from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedText } from '../components/ThemedText';
import { colors } from '../theme/colors';
import { useMatStore } from '../stores/matStore';
import { useTabScrollStore } from '../stores/tabScrollStore';
import { MainTabsParamList } from './types';
import LiveScreen from '../screens/LiveScreen';
import NLOSScreen from '../screens/NLOSScreen';
import VitalsScreen from '../screens/VitalsScreen';
import ZonesScreen from '../screens/ZonesScreen';
import MATScreen from '../screens/MATScreen';
import SettingsScreen from '../screens/SettingsScreen';
import WelcomeScreen from '../screens/WelcomeScreen';

const toIconName = (routeName: keyof MainTabsParamList) => {
  switch (routeName) {
    case 'Welcome':
      return 'home';
    case 'Live':
      return 'wifi';
    case 'Calibration':
      return 'scan';
    case 'Vitals':
      return 'heart';
    case 'Zones':
      return 'grid';
    case 'MAT':
      return 'shield-checkmark';
    case 'Settings':
      return 'settings';
    default:
      return 'ellipse';
  }
};

const screens: ReadonlyArray<{ name: keyof MainTabsParamList; component: React.ComponentType }> = [
  { name: 'Welcome', component: WelcomeScreen },
  { name: 'Live', component: LiveScreen },
  { name: 'Calibration', component: NLOSScreen },
  { name: 'Vitals', component: VitalsScreen },
  { name: 'Zones', component: ZonesScreen },
  { name: 'MAT', component: MATScreen },
  { name: 'Settings', component: SettingsScreen },
];

const Tab = createBottomTabNavigator<MainTabsParamList>();

const displayName = (routeName: keyof MainTabsParamList) => routeName;

export const HeaderRadar = () => {
  const pulse = useRef(new Animated.Value(0)).current;
  const sweep = useRef(new Animated.Value(0)).current;
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    let mounted = true;
    void AccessibilityInfo.isReduceMotionEnabled().then((enabled) => { if (mounted) setReduceMotion(enabled); });
    const subscription = AccessibilityInfo.addEventListener('reduceMotionChanged', setReduceMotion);
    return () => { mounted = false; subscription.remove(); };
  }, []);

  useEffect(() => {
    pulse.stopAnimation();
    sweep.stopAnimation();
    if (reduceMotion) {
      pulse.setValue(0.42);
      sweep.setValue(0.12);
      return;
    }
    pulse.setValue(0);
    sweep.setValue(0);
    const pulseLoop = Animated.loop(Animated.timing(pulse, { toValue: 1, duration: 1900, easing: Easing.out(Easing.quad), useNativeDriver: true }));
    const sweepLoop = Animated.loop(Animated.timing(sweep, { toValue: 1, duration: 2800, easing: Easing.linear, useNativeDriver: true }));
    pulseLoop.start();
    sweepLoop.start();
    return () => { pulseLoop.stop(); sweepLoop.stop(); };
  }, [pulse, reduceMotion, sweep]);

  return (
    <View testID="header-radar" pointerEvents="none" accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={styles.radarShell}>
      <View style={styles.radarInnerRing} />
      <Animated.View style={[styles.radarPulse, { opacity: pulse.interpolate({ inputRange: [0, .65, 1], outputRange: [.62, .24, 0] }), transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [.42, 1.34] }) }] }]} />
      <Animated.View testID="header-radar-sweep" style={[styles.radarSweep, { transform: [{ rotate: sweep.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] }) }] }]}><View style={styles.radarSweepLine} /></Animated.View>
      <View style={styles.radarBlip} />
      <View style={styles.statusCore} />
    </View>
  );
};

type AppTabBarProps = BottomTabBarProps & {
  matAlertCount: number;
};

export const AppTabBar = ({
  state,
  descriptors,
  navigation,
  insets,
  matAlertCount,
}: AppTabBarProps) => (
  <View
    testID="main-tab-bar"
    style={[styles.tabBar, { height: 64 + insets.bottom, paddingBottom: insets.bottom }]}
  >
    {state.routes.map((route, index) => {
      const routeName = route.name as keyof MainTabsParamList;
      if (routeName === 'Welcome') return null;
      const isFocused = state.index === index;
      const options = descriptors[route.key]?.options ?? {};
      const color = isFocused ? colors.accent : colors.textSecondary;
      const badge = routeName === 'MAT' && matAlertCount > 0 ? matAlertCount : null;

      const onPress = () => {
        const event = navigation.emit({
          type: 'tabPress',
          target: route.key,
          canPreventDefault: true,
        });

        if (!event.defaultPrevented) {
          useTabScrollStore.getState().requestTop(routeName);
          if (!isFocused) navigation.navigate(routeName, route.params);
        }
      };

      const onLongPress = () => {
        navigation.emit({ type: 'tabLongPress', target: route.key });
      };

      return (
        <Pressable
          key={route.key}
          testID={`tab-${routeName.toLowerCase()}`}
          accessibilityRole="button"
          accessibilityLabel={options.tabBarAccessibilityLabel ?? `${displayName(routeName)} tab`}
          accessibilityState={isFocused ? { selected: true } : {}}
          hitSlop={{ top: 6, bottom: 6, left: 2, right: 2 }}
          onPress={onPress}
          onLongPress={onLongPress}
          style={({ pressed }) => [styles.tabButton, pressed && styles.tabButtonPressed]}
        >
          <View pointerEvents="none" style={styles.tabIconWrap}>
            <Ionicons name={toIconName(routeName)} size={25} color={color} />
            {badge !== null && (
              <View style={styles.tabBadge}>
                <ThemedText preset="mono" style={styles.tabBadgeText}>{badge}</ThemedText>
              </View>
            )}
          </View>
          <ThemedText
            pointerEvents="none"
            preset="mono"
            style={[styles.tabLabel, { color }]}
          >
            {displayName(routeName)}
          </ThemedText>
        </Pressable>
      );
    })}
  </View>
);

export const AppHeader = ({ section, onHome }: { section: keyof MainTabsParamList; onHome: () => void }) => (
  <SafeAreaView edges={['top']} style={styles.headerSafeArea}>
    <View style={styles.headerRow}>
      <Pressable testID="header-home-logo" accessibilityRole="button" accessibilityLabel="Open RuView welcome" onPress={onHome} style={({ pressed }) => [styles.headerIdentity, pressed && styles.headerPressed]}>
        <HeaderRadar />
        <View>
          <ThemedText preset="labelLg" style={styles.headerTitle}>{section === 'Calibration' ? 'RuView Calibration' : 'RuView'}</ThemedText>
          <ThemedText preset="mono" style={styles.headerCaption}>MOBILE INSTRUMENT / 01</ThemedText>
        </View>
      </Pressable>
      <Pressable testID="header-home-nav" accessibilityRole="button" accessibilityLabel="Return to welcome" onPress={onHome} style={({ pressed }) => [styles.sectionBadge, pressed && styles.headerPressed]}><ThemedText preset="mono" style={styles.sectionBadgeText}>{section === 'Welcome' ? 'HOME' : displayName(section).toUpperCase()}</ThemedText></Pressable>
    </View>
  </SafeAreaView>
);

export const MainTabs = () => {
  const matAlertCount = useMatStore((state) => state.alerts.length);

  return (
    <Tab.Navigator
      initialRouteName="Welcome"
      tabBar={(props) => <AppTabBar {...props} matAlertCount={matAlertCount} />}
      screenOptions={({ route, navigation }) => ({
        headerShown: true,
        header: () => <AppHeader section={route.name} onHome={() => { useTabScrollStore.getState().requestTop('Welcome'); navigation.navigate('Welcome'); }} />,
      })}
    >
      {screens.map(({ name, component }) => (
        <Tab.Screen
          key={name}
          name={name}
          options={{
            tabBarBadge: name === 'MAT' ? (matAlertCount > 0 ? matAlertCount : undefined) : undefined,
          }}
          component={component}
        />
      ))}
    </Tab.Navigator>
  );
};

const styles = StyleSheet.create({
  headerSafeArea: {
    backgroundColor: '#080D14',
    borderBottomColor: colors.border,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerRow: {
    height: 62,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerIdentity: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerPressed: { opacity: 0.62 },
  radarShell: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: `${colors.accent}88`,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  radarInnerRing: { position: 'absolute', width: 23, height: 23, borderRadius: 12, borderWidth: 1, borderColor: `${colors.accent}38` },
  radarPulse: { position: 'absolute', width: 28, height: 28, borderRadius: 14, borderWidth: 1, borderColor: colors.success },
  radarSweep: { position: 'absolute', width: 36, height: 36, borderRadius: 18 },
  radarSweepLine: { position: 'absolute', left: 18, top: 17.5, width: 16, height: 1, backgroundColor: colors.success, shadowColor: colors.success, shadowOpacity: .7, shadowRadius: 3 },
  radarBlip: { position: 'absolute', right: 7, top: 9, width: 3, height: 3, borderRadius: 2, backgroundColor: colors.success, shadowColor: colors.success, shadowOpacity: 1, shadowRadius: 4 },
  statusCore: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: colors.success,
    shadowColor: colors.success,
    shadowOpacity: 0.8,
    shadowRadius: 8,
  },
  headerTitle: {
    color: colors.textPrimary,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  headerCaption: {
    color: colors.textSecondary,
    fontSize: 8,
    letterSpacing: 1.1,
  },
  sectionBadge: {
    borderWidth: 1,
    borderColor: `${colors.accent}88`,
    borderRadius: 14,
    paddingHorizontal: 11,
    paddingVertical: 7,
    overflow: 'hidden',
  },
  sectionBadgeText: { color: colors.accent, fontSize: 9 },
  tabBar: {
    position: 'relative',
    zIndex: 1000,
    elevation: 24,
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#0D1117',
    borderTopColor: colors.border,
    borderTopWidth: 1,
    paddingTop: 5,
  },
  tabButton: {
    flex: 1,
    minHeight: 58,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },
  tabButtonPressed: {
    opacity: 0.62,
  },
  tabIconWrap: {
    position: 'relative',
  },
  tabLabel: {
    fontSize: 10,
    lineHeight: 14,
  },
  tabBadge: {
    position: 'absolute',
    top: -5,
    right: -10,
    minWidth: 16,
    height: 16,
    paddingHorizontal: 3,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.danger,
  },
  tabBadgeText: {
    color: '#FFFFFF',
    fontSize: 8,
    lineHeight: 10,
  },
});
