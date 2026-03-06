import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { SerialPort, Chip, FlashProgress, FlashPhase } from "../types";

type WizardStep = 1 | 2 | 3;

export function FlashFirmware() {
  // --- State ---
  const [step, setStep] = useState<WizardStep>(1);
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [selectedPort, setSelectedPort] = useState<string>("");
  const [firmwarePath, setFirmwarePath] = useState<string>("");
  const [chip, setChip] = useState<Chip>("esp32s3");
  const [baud, setBaud] = useState<number>(460800);
  const [isLoadingPorts, setIsLoadingPorts] = useState(false);
  const [progress, setProgress] = useState<FlashProgress | null>(null);
  const [isFlashing, setIsFlashing] = useState(false);
  const [flashResult, setFlashResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Load serial ports ---
  const loadPorts = useCallback(async () => {
    setIsLoadingPorts(true);
    setError(null);
    try {
      const result = await invoke<SerialPort[]>("list_serial_ports");
      setPorts(result);
      if (result.length === 1) {
        setSelectedPort(result[0].name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoadingPorts(false);
    }
  }, []);

  useEffect(() => {
    loadPorts();
  }, [loadPorts]);

  // --- Listen for flash progress events ---
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    listen<FlashProgress>("flash-progress", (event) => {
      setProgress(event.payload);
    }).then((fn) => {
      unlisten = fn;
    });

    return () => {
      unlisten?.();
    };
  }, []);

  // --- File picker ---
  const pickFirmware = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        filters: [
          { name: "Firmware Binary", extensions: ["bin"] },
          { name: "All Files", extensions: ["*"] },
        ],
      });
      if (selected && typeof selected === "string") {
        setFirmwarePath(selected);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // --- Flash ---
  const startFlash = async () => {
    if (!selectedPort || !firmwarePath) return;

    setIsFlashing(true);
    setFlashResult(null);
    setProgress(null);
    setError(null);

    try {
      await invoke("flash_firmware", {
        port: selectedPort,
        firmwarePath,
        chip,
        baud,
      });
      setFlashResult({
        success: true,
        message: "Firmware flashed successfully.",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setFlashResult({ success: false, message: msg });
    } finally {
      setIsFlashing(false);
    }
  };

  // --- Step validation ---
  const canProceed = (s: WizardStep): boolean => {
    if (s === 1) return selectedPort !== "";
    if (s === 2) return firmwarePath !== "";
    return false;
  };

  return (
    <div style={{ padding: "24px", maxWidth: "700px" }}>
      <h1
        style={{
          fontSize: "22px",
          fontWeight: 700,
          color: "var(--text-primary, #e2e8f0)",
          margin: "0 0 4px",
        }}
      >
        Flash Firmware
      </h1>
      <p
        style={{
          fontSize: "13px",
          color: "var(--text-secondary, #94a3b8)",
          marginBottom: "24px",
        }}
      >
        Flash firmware to an ESP32 via serial connection
      </p>

      {/* Step indicator */}
      <StepIndicator current={step} />

      {/* Error banner */}
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

      {/* Step 1: Select Serial Port */}
      {step === 1 && (
        <div style={cardStyle}>
          <h2 style={stepTitle}>Step 1: Select Serial Port</h2>
          <p style={stepDesc}>
            Connect your ESP32 via USB and select the serial port.
          </p>

          <div style={{ marginBottom: "16px" }}>
            <label style={labelStyle}>Serial Port</label>
            <div style={{ display: "flex", gap: "8px" }}>
              <select
                value={selectedPort}
                onChange={(e) => setSelectedPort(e.target.value)}
                style={{ ...inputStyle, flex: 1 }}
                disabled={isLoadingPorts}
              >
                <option value="">
                  {isLoadingPorts
                    ? "Loading..."
                    : ports.length === 0
                      ? "No ports detected"
                      : "Select a port..."}
                </option>
                {ports.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                    {p.description ? ` - ${p.description}` : ""}
                    {p.chip ? ` (${p.chip.toUpperCase()})` : ""}
                  </option>
                ))}
              </select>
              <button onClick={loadPorts} style={secondaryBtnStyle} disabled={isLoadingPorts}>
                Refresh
              </button>
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={() => setStep(2)}
              disabled={!canProceed(1)}
              style={canProceed(1) ? primaryBtnStyle : disabledBtnStyle}
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Select Firmware */}
      {step === 2 && (
        <div style={cardStyle}>
          <h2 style={stepTitle}>Step 2: Select Firmware</h2>
          <p style={stepDesc}>
            Choose the firmware binary file and chip configuration.
          </p>

          <div style={{ marginBottom: "16px" }}>
            <label style={labelStyle}>Firmware Binary (.bin)</label>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                type="text"
                value={firmwarePath}
                readOnly
                placeholder="No file selected"
                style={{ ...inputStyle, flex: 1 }}
              />
              <button onClick={pickFirmware} style={secondaryBtnStyle}>
                Browse
              </button>
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "16px",
              marginBottom: "16px",
            }}
          >
            <div>
              <label style={labelStyle}>Chip</label>
              <select
                value={chip}
                onChange={(e) => setChip(e.target.value as Chip)}
                style={inputStyle}
              >
                <option value="esp32">ESP32</option>
                <option value="esp32s3">ESP32-S3</option>
                <option value="esp32c3">ESP32-C3</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Baud Rate</label>
              <select
                value={baud}
                onChange={(e) => setBaud(Number(e.target.value))}
                style={inputStyle}
              >
                <option value={115200}>115200</option>
                <option value={230400}>230400</option>
                <option value={460800}>460800</option>
                <option value={921600}>921600</option>
              </select>
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setStep(1)} style={secondaryBtnStyle}>
              Back
            </button>
            <button
              onClick={() => setStep(3)}
              disabled={!canProceed(2)}
              style={canProceed(2) ? primaryBtnStyle : disabledBtnStyle}
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Flash */}
      {step === 3 && (
        <div style={cardStyle}>
          <h2 style={stepTitle}>Step 3: Flash</h2>

          {/* Summary */}
          <div
            style={{
              background: "rgba(0,0,0,0.15)",
              borderRadius: "6px",
              padding: "12px 16px",
              marginBottom: "16px",
              fontSize: "12px",
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "8px",
            }}
          >
            <SummaryField label="Port" value={selectedPort} />
            <SummaryField
              label="Firmware"
              value={firmwarePath.split(/[\\/]/).pop() ?? firmwarePath}
            />
            <SummaryField label="Chip" value={chip.toUpperCase()} />
            <SummaryField label="Baud" value={String(baud)} />
          </div>

          {/* Progress */}
          {(isFlashing || progress) && !flashResult && (
            <div style={{ marginBottom: "16px" }}>
              <ProgressBar progress={progress} />
            </div>
          )}

          {/* Result */}
          {flashResult && (
            <div
              style={{
                background: flashResult.success
                  ? "rgba(34, 197, 94, 0.1)"
                  : "rgba(239, 68, 68, 0.1)",
                border: `1px solid ${flashResult.success ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)"}`,
                borderRadius: "6px",
                padding: "12px 16px",
                marginBottom: "16px",
                fontSize: "13px",
                color: flashResult.success ? "#86efac" : "#fca5a5",
              }}
            >
              {flashResult.message}
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <button
              onClick={() => {
                setStep(2);
                setFlashResult(null);
                setProgress(null);
              }}
              style={secondaryBtnStyle}
              disabled={isFlashing}
            >
              Back
            </button>
            {flashResult ? (
              <button
                onClick={() => {
                  setStep(1);
                  setFlashResult(null);
                  setProgress(null);
                  setFirmwarePath("");
                  setSelectedPort("");
                }}
                style={primaryBtnStyle}
              >
                Flash Another
              </button>
            ) : (
              <button
                onClick={startFlash}
                disabled={isFlashing}
                style={isFlashing ? disabledBtnStyle : primaryBtnStyle}
              >
                {isFlashing ? "Flashing..." : "Start Flash"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StepIndicator({ current }: { current: WizardStep }) {
  const steps = [
    { n: 1, label: "Select Port" },
    { n: 2, label: "Select Firmware" },
    { n: 3, label: "Flash" },
  ];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0",
        marginBottom: "24px",
      }}
    >
      {steps.map(({ n, label }, i) => {
        const isActive = n === current;
        const isDone = n < current;
        return (
          <div key={n} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div
                style={{
                  width: "28px",
                  height: "28px",
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "12px",
                  fontWeight: 700,
                  background: isActive
                    ? "var(--accent, #6366f1)"
                    : isDone
                      ? "rgba(34, 197, 94, 0.2)"
                      : "var(--border, #2e2e3e)",
                  color: isActive
                    ? "#fff"
                    : isDone
                      ? "#22c55e"
                      : "var(--text-muted, #64748b)",
                }}
              >
                {isDone ? "\u2713" : n}
              </div>
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: isActive ? 600 : 400,
                  color: isActive
                    ? "var(--text-primary, #e2e8f0)"
                    : "var(--text-muted, #64748b)",
                }}
              >
                {label}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div
                style={{
                  width: "40px",
                  height: "1px",
                  background: "var(--border, #2e2e3e)",
                  margin: "0 12px",
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

const PHASE_LABELS: Record<FlashPhase, string> = {
  connecting: "Connecting...",
  erasing: "Erasing flash...",
  writing: "Writing firmware...",
  verifying: "Verifying...",
  done: "Complete",
  error: "Error",
};

function ProgressBar({ progress }: { progress: FlashProgress | null }) {
  const pct = progress?.progress_pct ?? 0;
  const phase = progress?.phase ?? "connecting";
  const speed = progress?.speed_bps ?? 0;
  const speedKB = speed > 0 ? `${(speed / 1024).toFixed(1)} KB/s` : "";

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "12px",
          marginBottom: "6px",
        }}
      >
        <span style={{ color: "var(--text-secondary, #94a3b8)" }}>
          {PHASE_LABELS[phase]}
        </span>
        <span style={{ color: "var(--text-muted, #64748b)" }}>
          {pct.toFixed(1)}% {speedKB && `| ${speedKB}`}
        </span>
      </div>
      <div
        style={{
          width: "100%",
          height: "8px",
          background: "var(--border, #2e2e3e)",
          borderRadius: "4px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.min(pct, 100)}%`,
            height: "100%",
            background:
              phase === "error"
                ? "#ef4444"
                : phase === "done"
                  ? "#22c55e"
                  : "var(--accent, #6366f1)",
            borderRadius: "4px",
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

function SummaryField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        style={{
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-muted, #64748b)",
          marginBottom: "1px",
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "var(--text-secondary, #94a3b8)",
          fontFamily: "monospace",
          fontSize: "12px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

const cardStyle: React.CSSProperties = {
  background: "var(--card-bg, #1e1e2e)",
  border: "1px solid var(--border, #2e2e3e)",
  borderRadius: "8px",
  padding: "20px",
};

const stepTitle: React.CSSProperties = {
  fontSize: "16px",
  fontWeight: 600,
  color: "var(--text-primary, #e2e8f0)",
  margin: "0 0 4px",
};

const stepDesc: React.CSSProperties = {
  fontSize: "13px",
  color: "var(--text-secondary, #94a3b8)",
  marginBottom: "16px",
};

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: "12px",
  fontWeight: 600,
  color: "var(--text-secondary, #94a3b8)",
  marginBottom: "6px",
};

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

const disabledBtnStyle: React.CSSProperties = {
  ...primaryBtnStyle,
  background: "var(--border, #2e2e3e)",
  color: "var(--text-muted, #64748b)",
  cursor: "not-allowed",
};
