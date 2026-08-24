import React from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import {
  createBottomTabNavigator,
  type BottomTabBarProps,
} from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedText } from '../components/ThemedText';
import { colors } from '../theme/colors';
import { useMatStore } from '../stores/matStore';
import { MainTabsParamList } from './types';
import LiveScreen from '../screens/LiveScreen';
import NLOSScreen from '../screens/NLOSScreen';
import VitalsScreen from '../screens/VitalsScreen';
import ZonesScreen from '../screens/ZonesScreen';
import MATScreen from '../screens/MATScreen';
import SettingsScreen from '../screens/SettingsScreen';

const toIconName = (routeName: keyof MainTabsParamList) => {
  switch (routeName) {
    case 'Live':
      return 'wifi';
    case 'NLOS':
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
  { name: 'Live', component: LiveScreen },
  { name: 'NLOS', component: NLOSScreen },
  { name: 'Vitals', component: VitalsScreen },
  { name: 'Zones', component: ZonesScreen },
  { name: 'MAT', component: MATScreen },
  { name: 'Settings', component: SettingsScreen },
];

const Tab = createBottomTabNavigator<MainTabsParamList>();

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

        if (!isFocused && !event.defaultPrevented) {
          navigation.navigate(routeName, route.params);
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
          accessibilityLabel={options.tabBarAccessibilityLabel ?? `${routeName} tab`}
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
            {routeName}
          </ThemedText>
        </Pressable>
      );
    })}
  </View>
);

const AppHeader = ({ section }: { section: keyof MainTabsParamList }) => (
  <SafeAreaView edges={['top']} style={styles.headerSafeArea}>
    <View style={styles.headerRow}>
      <View style={styles.headerIdentity}>
        <View style={styles.statusRing}>
          <View style={styles.statusCore} />
        </View>
        <View>
          <ThemedText preset="labelLg" style={styles.headerTitle}>RuView NLOS</ThemedText>
          <ThemedText preset="mono" style={styles.headerCaption}>MOBILE INSTRUMENT / 01</ThemedText>
        </View>
      </View>
      <ThemedText preset="mono" style={styles.sectionBadge}>{section.toUpperCase()}</ThemedText>
    </View>
  </SafeAreaView>
);

export const MainTabs = () => {
  const matAlertCount = useMatStore((state) => state.alerts.length);

  return (
    <Tab.Navigator
      initialRouteName="NLOS"
      tabBar={(props) => <AppTabBar {...props} matAlertCount={matAlertCount} />}
      screenOptions={({ route }) => ({
        headerShown: true,
        header: () => <AppHeader section={route.name} />,
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
  statusRing: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: `${colors.accent}88`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusCore: {
    width: 11,
    height: 11,
    borderRadius: 6,
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
    color: colors.accent,
    borderWidth: 1,
    borderColor: `${colors.accent}88`,
    borderRadius: 14,
    paddingHorizontal: 11,
    paddingVertical: 7,
    fontSize: 9,
    overflow: 'hidden',
  },
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
