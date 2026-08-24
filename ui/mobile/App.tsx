import { useEffect } from 'react';
import { View } from 'react-native';
import { useFonts } from 'expo-font';
import { Outfit_400Regular } from '@expo-google-fonts/outfit/400Regular';
import { Outfit_500Medium } from '@expo-google-fonts/outfit/500Medium';
import { Outfit_600SemiBold } from '@expo-google-fonts/outfit/600SemiBold';
import { Outfit_700Bold } from '@expo-google-fonts/outfit/700Bold';
import { JetBrainsMono_400Regular } from '@expo-google-fonts/jetbrains-mono/400Regular';
import { JetBrainsMono_500Medium } from '@expo-google-fonts/jetbrains-mono/500Medium';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { apiService } from '@/services/api.service';
import { rssiService } from '@/services/rssi.service';
import { wsService } from '@/services/ws.service';
import { ThemeProvider } from './src/theme/ThemeContext';
import { usePoseStore } from './src/stores/poseStore';
import { useSettingsStore } from './src/stores/settingsStore';
import { RootNavigator } from './src/navigation/RootNavigator';

export default function App() {
  const [fontsLoaded, fontError] = useFonts({
    Outfit_400Regular,
    Outfit_500Medium,
    Outfit_600SemiBold,
    Outfit_700Bold,
    JetBrainsMono_400Regular,
    JetBrainsMono_500Medium,
  });
  const serverUrl = useSettingsStore((state) => state.serverUrl);
  const rssiScanEnabled = useSettingsStore((state) => state.rssiScanEnabled);
  const rssiScanIntervalSeconds = useSettingsStore((state) => state.rssiScanIntervalSeconds);

  useEffect(() => {
    apiService.setBaseUrl(serverUrl);
    const unsubscribe = wsService.subscribe(usePoseStore.getState().handleFrame);
    wsService.connect(serverUrl);

    return () => {
      unsubscribe();
      wsService.disconnect();
    };
  }, [serverUrl]);

  useEffect(() => {
    if (!rssiScanEnabled) {
      rssiService.stopScanning();
      return;
    }

    const unsubscribe = rssiService.subscribe(() => {
      // Consumers can subscribe elsewhere for RSSI events.
    });
    rssiService.startScanning(rssiScanIntervalSeconds * 1000);

    return () => {
      unsubscribe();
      rssiService.stopScanning();
    };
  }, [rssiScanEnabled, rssiScanIntervalSeconds]);

  useEffect(() => {
    (globalThis as { __appStartTime?: number }).__appStartTime = Date.now();
  }, []);

  if (!fontsLoaded && !fontError) {
    return <View testID="font-load-gate" style={{ flex: 1, backgroundColor: '#0B0E13' }} />;
  }

  const navigationTheme = {
    ...DarkTheme,
    colors: {
      ...DarkTheme.colors,
      background: '#0B0E13',
      card: '#14181F',
      text: '#E7EBEF',
      border: '#272C35',
      primary: '#19D4E6',
    },
  };

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <ThemeProvider>
          <NavigationContainer theme={navigationTheme}>
            <RootNavigator />
          </NavigationContainer>
        </ThemeProvider>
      </SafeAreaProvider>
      <StatusBar style="light" />
    </GestureHandlerRootView>
  );
}
