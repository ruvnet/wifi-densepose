import { useState, useEffect, useCallback } from "react";
import type { AppSettings } from "../types";

const DEFAULT_SETTINGS: AppSettings = {
  server_http_port: 8080,
  server_ws_port: 8765,
  server_udp_port: 5005,
  bind_address: "0.0.0.0",
  ui_path: "",
  ota_psk: "",
  auto_discover: true,
  discover_interval_ms: 10_000,
  theme: "dark",
};

export function Settings() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [showPsk, setShowPsk] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load persisted settings on mount
  useEffect(() => {
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const persisted = await invoke<AppSettings | null>("get_settings");
        if (persisted) {
          setSettings(persisted);
        }
      } catch {
        // Settings command may not exist yet -- use defaults
      }
    })();
  }, []);

  const update = useCallback(
    <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
      setSaved(false);
    },
    []
  );

  const save = async () => {
    setError(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("save_settings", { settings });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const reset = () => {
    setSettings(DEFAULT_SETTINGS);
    setSaved(false);
  };

  return (
    <div style={{ padding: "24px", maxWidth: "600px" }}>
      <h1
        style={{
          fontSize: "22px",
          fontWeight: 700,
          color: "var(--text-primary, #e2e8f0)",
          margin: "0 0 4px",
        }}
      >
        Settings
      </h1>
      <p
        style={{
          fontSize: "13px",
          color: "var(--text-secondary, #94a3b8)",
          marginBottom: "24px",
        }}
      >
        Configure server, network, and application preferences
      </p>

      {/* Error */}
      {error && (
        <div
          style={{
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: "6px",
            padding: "12px 16px",
            marginBottom: "16px",
            fontSize: "13px",
            color: "#fca5a5",
          }}
        >
          {error}
        </div>
      )}

      {/* Saved toast */}
      {saved && (
        <div
          style={{
            background: "rgba(34, 197, 94, 0.1)",
            border: "1px solid rgba(34, 197, 94, 0.3)",
            borderRadius: "6px",
            padding: "12px 16px",
            marginBottom: "16px",
            fontSize: "13px",
            color: "#86efac",
          }}
        >
          Settings saved.
        </div>
      )}

      {/* Sensing Server */}
      <Section title="Sensing Server">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "16px",
          }}
        >
          <Field label="HTTP Port">
            <NumberInput
              value={settings.server_http_port}
              onChange={(v) => update("server_http_port", v)}
              min={1}
              max={65535}
            />
          </Field>
          <Field label="WebSocket Port">
            <NumberInput
              value={settings.server_ws_port}
              onChange={(v) => update("server_ws_port", v)}
              min={1}
              max={65535}
            />
          </Field>
          <Field label="UDP Port">
            <NumberInput
              value={settings.server_udp_port}
              onChange={(v) => update("server_udp_port", v)}
              min={1}
              max={65535}
            />
          </Field>
          <Field label="Bind Address">
            <TextInput
              value={settings.bind_address}
              onChange={(v) => update("bind_address", v)}
              placeholder="0.0.0.0"
            />
          </Field>
        </div>
        <div style={{ marginTop: "16px" }}>
          <Field label="UI Static Files Path">
            <TextInput
              value={settings.ui_path}
              onChange={(v) => update("ui_path", v)}
              placeholder="Leave empty for default"
            />
          </Field>
        </div>
      </Section>

      {/* Security */}
      <Section title="Security">
        <Field label="OTA Pre-Shared Key (PSK)">
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              type={showPsk ? "text" : "password"}
              value={settings.ota_psk}
              onChange={(e) => update("ota_psk", e.target.value)}
              placeholder="Enter PSK for OTA authentication"
              style={{ ...inputStyle, flex: 1 }}
            />
            <button
              onClick={() => setShowPsk((prev) => !prev)}
              style={secondaryBtnStyle}
            >
              {showPsk ? "Hide" : "Show"}
            </button>
          </div>
          <p
            style={{
              fontSize: "11px",
              color: "var(--text-muted, #64748b)",
              marginTop: "4px",
            }}
          >
            Used for authenticating OTA firmware updates to nodes.
          </p>
        </Field>
      </Section>

      {/* Discovery */}
      <Section title="Network Discovery">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "16px",
          }}
        >
          <Field label="Auto-Discover">
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={settings.auto_discover}
                onChange={(e) => update("auto_discover", e.target.checked)}
                style={{ accentColor: "var(--accent, #6366f1)" }}
              />
              <span
                style={{
                  fontSize: "13px",
                  color: "var(--text-secondary, #94a3b8)",
                }}
              >
                Enable periodic scanning
              </span>
            </label>
          </Field>
          <Field label="Scan Interval (ms)">
            <NumberInput
              value={settings.discover_interval_ms}
              onChange={(v) => update("discover_interval_ms", v)}
              min={1000}
              max={120_000}
              step={1000}
              disabled={!settings.auto_discover}
            />
          </Field>
        </div>
      </Section>

      {/* Actions */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "24px",
        }}
      >
        <button onClick={reset} style={secondaryBtnStyle}>
          Reset to Defaults
        </button>
        <button onClick={save} style={primaryBtnStyle}>
          Save Settings
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: "var(--card-bg, #1e1e2e)",
        border: "1px solid var(--border, #2e2e3e)",
        borderRadius: "8px",
        padding: "20px",
        marginBottom: "16px",
      }}
    >
      <h2
        style={{
          fontSize: "14px",
          fontWeight: 600,
          color: "var(--text-primary, #e2e8f0)",
          margin: "0 0 16px",
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        style={{
          display: "block",
          fontSize: "12px",
          fontWeight: 600,
          color: "var(--text-secondary, #94a3b8)",
          marginBottom: "6px",
        }}
      >
        {label}
      </label>
      {children}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  disabled = false,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      style={{
        ...inputStyle,
        opacity: disabled ? 0.5 : 1,
      }}
    />
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled = false,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => {
        const n = parseInt(e.target.value, 10);
        if (!isNaN(n)) onChange(n);
      }}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      style={{
        ...inputStyle,
        opacity: disabled ? 0.5 : 1,
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 12px",
  border: "1px solid var(--border, #2e2e3e)",
  borderRadius: "6px",
  background: "var(--input-bg, #12121a)",
  color: "var(--text-primary, #e2e8f0)",
  fontSize: "13px",
  outline: "none",
  boxSizing: "border-box",
};

const primaryBtnStyle: React.CSSProperties = {
  padding: "8px 20px",
  border: "none",
  borderRadius: "6px",
  background: "var(--accent, #6366f1)",
  color: "#fff",
  fontSize: "13px",
  fontWeight: 600,
  cursor: "pointer",
};

const secondaryBtnStyle: React.CSSProperties = {
  padding: "8px 16px",
  border: "1px solid var(--border, #2e2e3e)",
  borderRadius: "6px",
  background: "transparent",
  color: "var(--text-secondary, #94a3b8)",
  fontSize: "13px",
  fontWeight: 500,
  cursor: "pointer",
};
