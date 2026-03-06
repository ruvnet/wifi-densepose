import { useState } from "react";
import { useNodes } from "../hooks/useNodes";
import { StatusBadge } from "../components/StatusBadge";
import type { Node } from "../types";

export function Nodes() {
  const { nodes, isScanning, scan, error } = useNodes({
    pollInterval: 10_000,
    autoScan: true,
  });
  const [expandedMac, setExpandedMac] = useState<string | null>(null);

  const toggleExpand = (node: Node) => {
    const key = node.mac ?? node.ip;
    setExpandedMac((prev) => (prev === key ? null : key));
  };

  return (
    <div style={{ padding: "24px", maxWidth: "1200px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <div>
          <h1
            style={{
              fontSize: "22px",
              fontWeight: 700,
              color: "var(--text-primary, #e2e8f0)",
              margin: 0,
            }}
          >
            Nodes
          </h1>
          <p
            style={{
              fontSize: "13px",
              color: "var(--text-secondary, #94a3b8)",
              marginTop: "4px",
            }}
          >
            {nodes.length} node{nodes.length !== 1 ? "s" : ""} in registry
          </p>
        </div>
        <button
          onClick={scan}
          disabled={isScanning}
          style={{
            padding: "8px 16px",
            border: "none",
            borderRadius: "6px",
            background: isScanning
              ? "var(--border, #2e2e3e)"
              : "var(--accent, #6366f1)",
            color: isScanning ? "var(--text-muted, #64748b)" : "#fff",
            fontSize: "13px",
            fontWeight: 600,
            cursor: isScanning ? "not-allowed" : "pointer",
          }}
        >
          {isScanning ? "Scanning..." : "Refresh"}
        </button>
      </div>

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

      {/* Table */}
      {nodes.length === 0 ? (
        <div
          style={{
            background: "var(--card-bg, #1e1e2e)",
            border: "1px solid var(--border, #2e2e3e)",
            borderRadius: "8px",
            padding: "48px",
            textAlign: "center",
            color: "var(--text-muted, #64748b)",
            fontSize: "13px",
          }}
        >
          {isScanning ? "Scanning for nodes..." : "No nodes found. Run a scan to discover ESP32 devices."}
        </div>
      ) : (
        <div
          style={{
            background: "var(--card-bg, #1e1e2e)",
            border: "1px solid var(--border, #2e2e3e)",
            borderRadius: "8px",
            overflow: "hidden",
          }}
        >
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "13px",
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "1px solid var(--border, #2e2e3e)",
                  textAlign: "left",
                }}
              >
                <Th>Status</Th>
                <Th>MAC</Th>
                <Th>IP</Th>
                <Th>Firmware</Th>
                <Th>Chip</Th>
                <Th>Last Seen</Th>
              </tr>
            </thead>
            <tbody>
              {nodes.map((node) => {
                const key = node.mac ?? node.ip;
                const isExpanded = expandedMac === key;
                return (
                  <NodeRow
                    key={key}
                    node={node}
                    isExpanded={isExpanded}
                    onToggle={() => toggleExpand(node)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th
      style={{
        padding: "10px 16px",
        fontSize: "10px",
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.05em",
        color: "var(--text-muted, #64748b)",
      }}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  mono = false,
}: {
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <td
      style={{
        padding: "10px 16px",
        color: "var(--text-secondary, #94a3b8)",
        fontFamily: mono ? "monospace" : "inherit",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </td>
  );
}

function formatLastSeen(iso: string): string {
  try {
    const d = new Date(iso);
    const diff = Date.now() - d.getTime();
    if (diff < 60_000) return "just now";
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return d.toLocaleDateString();
  } catch {
    return "--";
  }
}

function NodeRow({
  node,
  isExpanded,
  onToggle,
}: {
  node: Node;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        style={{
          borderBottom: isExpanded ? "none" : "1px solid var(--border, #2e2e3e)",
          cursor: "pointer",
          transition: "background 0.1s",
        }}
        onMouseEnter={(e) =>
          (e.currentTarget.style.background = "rgba(255,255,255,0.02)")
        }
        onMouseLeave={(e) =>
          (e.currentTarget.style.background = "transparent")
        }
      >
        <Td>
          <StatusBadge status={node.health} />
        </Td>
        <Td mono>{node.mac ?? "--"}</Td>
        <Td mono>{node.ip}</Td>
        <Td>{node.firmware_version ?? "--"}</Td>
        <Td>{node.chip.toUpperCase()}</Td>
        <Td>{formatLastSeen(node.last_seen)}</Td>
      </tr>
      {isExpanded && (
        <tr style={{ borderBottom: "1px solid var(--border, #2e2e3e)" }}>
          <td colSpan={6} style={{ padding: "0 16px 16px" }}>
            <ExpandedDetails node={node} />
          </td>
        </tr>
      )}
    </>
  );
}

function ExpandedDetails({ node }: { node: Node }) {
  return (
    <div
      style={{
        background: "rgba(0,0,0,0.15)",
        borderRadius: "6px",
        padding: "16px",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
        gap: "12px 24px",
        fontSize: "12px",
      }}
    >
      <DetailField label="Hostname" value={node.hostname ?? "--"} />
      <DetailField label="Node ID" value={String(node.node_id)} />
      <DetailField label="Mesh Role" value={node.mesh_role} />
      <DetailField
        label="TDM Slot"
        value={
          node.tdm_slot != null && node.tdm_total != null
            ? `${node.tdm_slot} / ${node.tdm_total}`
            : "--"
        }
      />
      <DetailField
        label="Edge Tier"
        value={node.edge_tier != null ? String(node.edge_tier) : "--"}
      />
      <DetailField
        label="Uptime"
        value={
          node.uptime_secs != null
            ? `${Math.floor(node.uptime_secs / 3600)}h ${Math.floor((node.uptime_secs % 3600) / 60)}m`
            : "--"
        }
      />
      <DetailField label="Discovery" value={node.discovery_method} />
      <DetailField
        label="Capabilities"
        value={
          node.capabilities
            ? Object.entries(node.capabilities)
                .filter(([, v]) => v)
                .map(([k]) => k)
                .join(", ") || "none"
            : "--"
        }
      />
      {node.friendly_name && (
        <DetailField label="Name" value={node.friendly_name} />
      )}
      {node.notes && <DetailField label="Notes" value={node.notes} />}
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        style={{
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-muted, #64748b)",
          marginBottom: "2px",
        }}
      >
        {label}
      </div>
      <div style={{ color: "var(--text-secondary, #94a3b8)" }}>{value}</div>
    </div>
  );
}
