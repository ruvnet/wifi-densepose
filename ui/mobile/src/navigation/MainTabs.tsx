import React from 'react';
import { StyleSheet, View } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
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
      screenOptions={({ route }) => ({
        headerShown: true,
        header: () => <AppHeader section={route.name} />,
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.textSecondary,
        tabBarStyle: {
          backgroundColor: '#0D1117',
          borderTopColor: colors.border,
          borderTopWidth: 1,
        },
        tabBarIcon: ({ color, size }) => <Ionicons name={toIconName(route.name)} size={size} color={color} />,
        tabBarLabelStyle: {
          fontFamily: 'JetBrainsMono_500Medium',
          fontSize: 10,
        },
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
});
