import { Pressable, TextInput, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

interface NlosServerUrlInputProps {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
}

export const NlosServerUrlInput = ({ value, onChange, onSave }: NlosServerUrlInputProps) => {
  const validation = normalizeNlosServerUrl(value);

  return (
    <View>
      <ThemedText preset="labelMd" style={{ marginBottom: spacing.sm }}>
        RuView NLOS server URL
      </ThemedText>
      <TextInput
        testID="nlos-server-url-input"
        value={value}
        onChangeText={onChange}
        autoCapitalize="none"
        autoCorrect={false}
        placeholder="https://ruview.example.com"
        keyboardType="url"
        placeholderTextColor={colors.textSecondary}
        style={{
          borderWidth: 1,
          borderColor: validation.valid ? colors.border : colors.danger,
          borderRadius: 10,
          backgroundColor: colors.surface,
          color: colors.textPrimary,
          padding: spacing.sm,
          marginBottom: spacing.sm,
        }}
      />
      {!validation.valid && (
        <ThemedText preset="bodySm" style={{ color: colors.danger, marginBottom: spacing.sm }}>
          {validation.error}
        </ThemedText>
      )}
      <ThemedText preset="bodySm" style={{ color: colors.textSecondary, marginBottom: spacing.sm }}>
        Separate from the CSI endpoint. Live access requires an ephemeral Bearer credential.
      </ThemedText>
      <Pressable
        accessibilityRole="button"
        onPress={onSave}
        disabled={!validation.valid}
        style={{
          paddingVertical: 10,
          borderRadius: 8,
          backgroundColor: validation.valid ? colors.success : colors.surfaceAlt,
          alignItems: 'center',
        }}
      >
        <ThemedText preset="labelMd" style={{ color: colors.textPrimary }}>
          Save NLOS server
        </ThemedText>
      </Pressable>
    </View>
  );
};
