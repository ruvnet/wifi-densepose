import type { Node } from "../types";
import { StatusBadge } from "./StatusBadge";

interface NodeCardProps {
  node: Node;
  onClick?: (node: Node) => void;
}

function formatUptime(secs: number | null): string {
  if (secs == null) return "--";
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  return `${Math.floor(secs / 86400)}d ${Math.floor((secs % 86400) / 3600)}h`;
}

function formatLastSeen(iso: string): string {
  try {
    const d = new Date(iso);
    const now = Date.now();
    const diffMs = now - d.getTime();
    if (diffMs < 60_000) return "just now";
    if (diffMs < 3_600_000) return `${Math.floor(diffMs / 60_000)}m ago`;
    if (diffMs < 86_400_000) return `${Math.floor(diffMs / 3_600_000)}h ago`;
    return d.toLocaleDateString();
  } catch {
    return "--";
  }
}

export function NodeCard({ node, onClick }: NodeCardProps) {
  const isOnline = node.health === "online";

  return (
    <div
      onClick={() => onClick?.(node)}
      style={{
        background: "var(--card-bg, #1e1e2e)",
        border: "1px solid var(--border, #2e2e3e)",
        borderRadius: "8px",
        padding: "16px",
        cursor: onClick ? "pointer" : "default",
        opacity: isOnline ? 1 : 0.6,
        transition: "border-color 0.15s, box-shadow 0.15s",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--accent, #6366f1)";
        e.currentTarget.style.boxShadow = "0 0 0 1px var(--accent, #6366f1)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--border, #2e2e3e)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "12px",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "14px",
              fontWeight: 700,
              color: "var(--text-primary, #e2e8f0)",
              marginBottom: "2px",
            }}
          >
            {node.friendly_name || node.hostname || `Node ${node.node_id}`}
          </div>
          <div
            style={{
              fontSize: "12px",
              color: "var(--text-secondary, #94a3b8)",
              fontFamily: "monospace",
            }}
          >
            {node.ip}
          </div>
        </div>
        <StatusBadge status={node.health} />
      </div>

      {/* Details grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px 16px",
          fontSize: "12px",
        }}
      >
        <DetailRow label="MAC" value={node.mac ?? "--"} mono />
        <DetailRow label="Firmware" value={node.firmware_version ?? "--"} />
        <DetailRow label="Chip" value={node.chip.toUpperCase()} />
        <DetailRow label="Role" value={node.mesh_role} />
        <DetailRow
          label="TDM"
          value={
            node.tdm_slot != null && node.tdm_total != null
              ? `${node.tdm_slot}/${node.tdm_total}`
              : "--"
          }
        />
        <DetailRow
          label="Edge Tier"
          value={node.edge_tier != null ? String(node.edge_tier) : "--"}
        />
        <DetailRow label="Uptime" value={formatUptime(node.uptime_secs)} />
        <DetailRow label="Seen" value={formatLastSeen(node.last_seen)} />
      </div>
    </div>
  );
}

function DetailRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div
        style={{
          color: "var(--text-muted, #64748b)",
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "1px",
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "var(--text-secondary, #94a3b8)",
          fontFamily: mono ? "monospace" : "inherit",
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
