import type { HealthStatus } from "../types";

interface StatusBadgeProps {
  status: HealthStatus;
  /** Optional size variant. Default: "sm" */
  size?: "sm" | "md" | "lg";
}

const STATUS_STYLES: Record<HealthStatus, { bg: string; text: string; label: string }> = {
  online: {
    bg: "rgba(34, 197, 94, 0.15)",
    text: "#22c55e",
    label: "Online",
  },
  offline: {
    bg: "rgba(239, 68, 68, 0.15)",
    text: "#ef4444",
    label: "Offline",
  },
  degraded: {
    bg: "rgba(234, 179, 8, 0.15)",
    text: "#eab308",
    label: "Degraded",
  },
  unknown: {
    bg: "rgba(148, 163, 184, 0.15)",
    text: "#94a3b8",
    label: "Unknown",
  },
};

const SIZE_STYLES: Record<string, { fontSize: string; padding: string; dot: string }> = {
  sm: { fontSize: "11px", padding: "2px 8px", dot: "6px" },
  md: { fontSize: "13px", padding: "4px 12px", dot: "8px" },
  lg: { fontSize: "15px", padding: "6px 16px", dot: "10px" },
};

export function StatusBadge({ status, size = "sm" }: StatusBadgeProps) {
  const style = STATUS_STYLES[status];
  const sizeStyle = SIZE_STYLES[size];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        backgroundColor: style.bg,
        color: style.text,
        fontSize: sizeStyle.fontSize,
        fontWeight: 600,
        padding: sizeStyle.padding,
        borderRadius: "9999px",
        lineHeight: 1,
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: sizeStyle.dot,
          height: sizeStyle.dot,
          borderRadius: "50%",
          backgroundColor: style.text,
          flexShrink: 0,
        }}
      />
      {style.label}
    </span>
  );
}
