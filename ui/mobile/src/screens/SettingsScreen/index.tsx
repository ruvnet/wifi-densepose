import { useEffect, useState } from 'react';
import { Alert, Linking, Platform, Pressable, ScrollView, Switch, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { WS_PATH } from '@/constants/websocket';
import { apiService } from '@/services/api.service';
import { wsService } from '@/services/ws.service';
import { type RssiScanIntervalSeconds, useSettingsStore } from '@/stores/settingsStore';
import { ThemePicker } from './ThemePicker';
import { RssiToggle } from './RssiToggle';
import { ServerUrlInput } from './ServerUrlInput';
import { NlosServerUrlInput } from './NlosServerUrlInput';

type GlowCardProps = {
  title: string;
  children: React.ReactNode;
};

const GlowCard = ({ title, children }: GlowCardProps) => {
  return (
    <View
      style={{
        backgroundColor: '#0F141E',
        borderRadius: 14,
        borderWidth: 1,
        borderColor: `${colors.accent}35`,
        padding: spacing.md,
        marginBottom: spacing.md,
      }}
    >
      <ThemedText preset="labelMd" style={{ marginBottom: spacing.sm, color: colors.textPrimary }}>
        {title}
      </ThemedText>
      {children}
    </View>
  );
};

const ScanIntervalPicker = ({
  value,
  onChange,
}: {
  value: RssiScanIntervalSeconds;
  onChange: (value: RssiScanIntervalSeconds) => void;
}) => {
  const options: RssiScanIntervalSeconds[] = [1, 2, 5];

  return (
    <View style={{ flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm }}>
      {options.map((interval) => {
        const isActive = interval === value;
        return (
          <Pressable
            key={interval}
            onPress={() => onChange(interval)}
            style={{
              flex: 1,
              borderWidth: 1,
              borderColor: isActive ? colors.accent : colors.border,
              borderRadius: 8,
              backgroundColor: isActive ? `${colors.accent}20` : colors.surface,
              alignItems: 'center',
            }}
          >
            <ThemedText
              preset="bodySm"
              style={{
                color: isActive ? colors.accent : colors.textSecondary,
                paddingVertical: 8,
              }}
            >
              {interval}s
            </ThemedText>
          </Pressable>
        );
      })}
    </View>
  );
};

export const SettingsScreen = () => {
  const serverUrl = useSettingsStore((state) => state.serverUrl);
  const nlosServerUrl = useSettingsStore((state) => state.nlosServerUrl);
  const rssiScanEnabled = useSettingsStore((state) => state.rssiScanEnabled);
  const rssiScanIntervalSeconds = useSettingsStore((state) => state.rssiScanIntervalSeconds);
  const theme = useSettingsStore((state) => state.theme);
  const alertSoundEnabled = useSettingsStore((state) => state.alertSoundEnabled);
  const setServerUrl = useSettingsStore((state) => state.setServerUrl);
  const setNlosServerUrl = useSettingsStore((state) => state.setNlosServerUrl);
  const setRssiScanEnabled = useSettingsStore((state) => state.setRssiScanEnabled);
  const setRssiScanIntervalSeconds = useSettingsStore((state) => state.setRssiScanIntervalSeconds);
  const setTheme = useSettingsStore((state) => state.setTheme);
  const setAlertSoundEnabled = useSettingsStore((state) => state.setAlertSoundEnabled);

  const [draftUrl, setDraftUrl] = useState(serverUrl);
  const [draftNlosUrl, setDraftNlosUrl] = useState(nlosServerUrl);

  useEffect(() => {
    setDraftUrl(serverUrl);
  }, [serverUrl]);

  useEffect(() => {
    setDraftNlosUrl(nlosServerUrl);
  }, [nlosServerUrl]);

  const handleSaveUrl = () => {
    const newUrl = draftUrl.trim().replace(/\/+$/, '');
    setDraftUrl(newUrl);
    setServerUrl(newUrl);
    wsService.disconnect();
    wsService.connect(newUrl);
    apiService.setBaseUrl(newUrl);
  };

  const handleSaveNlosUrl = () => {
    setNlosServerUrl(draftNlosUrl.trim());
  };

  const handleOpenGitHub = async () => {
    const handled = await Linking.canOpenURL('https://github.com');
    if (!handled) {
      Alert.alert('Unable to open link', 'Please open https://github.com manually in your browser.');
      return;
    }

    await Linking.openURL('https://github.com');
  };

  return (
    <SafeAreaView edges={['top']} style={{ flex: 1, backgroundColor: colors.bg }}>
      <ThemedView style={{ flex: 1, backgroundColor: colors.bg }}>
        <ScrollView
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={{
            paddingHorizontal: spacing.md,
            paddingBottom: spacing.xxl,
          }}
        >
        <View style={{ paddingVertical: spacing.md }}>
          <ThemedText preset="displayMd">Settings</ThemedText>
          <ThemedText preset="bodySm" style={{ color: colors.textSecondary, marginTop: spacing.xs }}>
            Configure sensing, NLOS access, alerts, and appearance.
          </ThemedText>
        </View>
        <GlowCard title="SERVER">
          <ServerUrlInput value={draftUrl} onChange={setDraftUrl} onSave={handleSaveUrl} />
        </GlowCard>

        <GlowCard title="RUVIEW NLOS SERVER">
          <NlosServerUrlInput
            value={draftNlosUrl}
            onChange={setDraftNlosUrl}
            onSave={handleSaveNlosUrl}
          />
        </GlowCard>

        <GlowCard title="SENSING">
          <RssiToggle enabled={rssiScanEnabled} onChange={setRssiScanEnabled} />
          <ThemedText preset="bodyMd" style={{ marginTop: spacing.md }}>
            Scan interval
          </ThemedText>
          <ScanIntervalPicker value={rssiScanIntervalSeconds} onChange={setRssiScanIntervalSeconds} />
          <ThemedText preset="bodySm" style={{ color: colors.textSecondary, marginTop: spacing.sm }}>
            Active interval: {rssiScanIntervalSeconds}s
          </ThemedText>
          {Platform.OS === 'ios' && (
            <ThemedText preset="bodySm" style={{ color: colors.textSecondary, marginTop: spacing.sm }}>
              iOS: RSSI scanning uses stubbed telemetry in this build.
            </ThemedText>
          )}
        </GlowCard>

        <GlowCard title="ALERTS">
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
            <View style={{ flex: 1, paddingRight: spacing.md }}>
              <ThemedText preset="bodyMd">MAT alert sounds</ThemedText>
              <ThemedText preset="bodySm" style={{ color: colors.textSecondary, marginTop: spacing.xs }}>
                Play an audible notification for new triage alerts.
              </ThemedText>
            </View>
            <Switch
              accessibilityLabel="MAT alert sounds"
              value={alertSoundEnabled}
              onValueChange={setAlertSoundEnabled}
              trackColor={{ true: colors.accent, false: colors.surfaceAlt }}
              thumbColor={colors.textPrimary}
            />
          </View>
        </GlowCard>

        <GlowCard title="APPEARANCE">
          <ThemePicker value={theme} onChange={setTheme} />
        </GlowCard>

        <GlowCard title="ABOUT">
          <ThemedText preset="bodyMd" style={{ marginBottom: spacing.xs }}>
            WiFi-DensePose Mobile v1.0.0
          </ThemedText>
          <ThemedText
            preset="bodySm"
            style={{ color: colors.accent, marginBottom: spacing.sm }}
            onPress={handleOpenGitHub}
          >
            View on GitHub
          </ThemedText>
          <ThemedText preset="bodySm">WebSocket: {WS_PATH}</ThemedText>
          <ThemedText preset="bodySm" style={{ color: colors.textSecondary }}>
            Triage priority mapping: Immediate/Delayed/Minor/Deceased/Unknown
          </ThemedText>
        </GlowCard>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
};

export default SettingsScreen;
