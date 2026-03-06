import React, { useEffect, useState } from "react";
import { StatusBadge } from "../components/StatusBadge";
import type { HealthStatus } from "../types";

interface DiscoveredNode {
  ip: string;
  mac: string | null;
  hostname: string | null;
  node_id: number;
  firmware_version: string | null;
  health: HealthStatus;
  last_seen: string;
}

interface ServerStatus {
  running: boolean;
  pid: number | null;
  http_port: number | null;
  ws_port: number | null;
}

const Dashboard: React.FC = () => {
  const [nodes, setNodes] = useState<DiscoveredNode[]>([]);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [scanning, setScanning] = useState(false);

  const handleScan = async () => {
    setScanning(true);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const found = await invoke<DiscoveredNode[]>("discover_nodes", {
        timeoutMs: 3000,
      });
      setNodes(found);
    } catch (err) {
      console.error("Discovery failed:", err);
    } finally {
      setScanning(false);
    }
  };

  const fetchServerStatus = async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const status = await invoke<ServerStatus>("server_status");
      setServerStatus(status);
    } catch (err) {
      console.error("Server status check failed:", err);
    }
  };

  useEffect(() => {
    handleScan();
    fetchServerStatus();
  }, []);

  const onlineCount = nodes.filter((n) => n.health === "online").length;

  return (
    <div style={{ padding: "var(--space-5)" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "var(--space-5)",
        }}
      >
        <h2 className="heading-lg">Dashboard</h2>
        <button
          onClick={handleScan}
          disabled={scanning}
          style={{
            padding: "var(--space-2) var(--space-4)",
            background: scanning ? "var(--bg-active)" : "var(--accent)",
            color: scanning ? "var(--text-muted)" : "#fff",
            borderRadius: 6,
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          {scanning ? "Scanning..." : "Scan Network"}
        </button>
      </div>

      {/* Stats row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "var(--space-4)",
          marginBottom: "var(--space-5)",
        }}
      >
        <StatCard label="Total Nodes" value={String(nodes.length)} />
        <StatCard label="Online" value={String(onlineCount)} color="var(--status-online)" />
        <StatCard label="Offline" value={String(nodes.length - onlineCount)} color="var(--status-error)" />
        <StatCard
          label="Server"
          value={serverStatus?.running ? "Running" : "Stopped"}
          color={serverStatus?.running ? "var(--status-online)" : "var(--status-error)"}
        />
      </div>

      {/* Server status panel */}
      <div
        style={{
          background: "var(--bg-surface)",
          borderRadius: 8,
          padding: "var(--space-4)",
          marginBottom: "var(--space-5)",
          border: "1px solid var(--border)",
        }}
      >
        <h3 className="heading-sm" style={{ marginBottom: "var(--space-2)" }}>
          Sensing Server
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: serverStatus?.running ? "var(--status-online)" : "var(--status-error)",
            }}
          />
          <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            {serverStatus?.running
              ? `Running (PID ${serverStatus.pid})`
              : "Stopped"}
          </span>
          {serverStatus?.running && serverStatus.http_port && (
            <span
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
                marginLeft: "var(--space-2)",
              }}
            >
              :{serverStatus.http_port}
            </span>
          )}
        </div>
      </div>

      {/* Node list */}
      <h3
        className="heading-sm"
        style={{
          marginBottom: "var(--space-3)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        Discovered Nodes ({nodes.length})
      </h3>

      {nodes.length === 0 ? (
        <div
          style={{
            background: "var(--bg-surface)",
            borderRadius: 8,
            padding: "var(--space-6)",
            textAlign: "center",
            color: "var(--text-muted)",
            border: "1px solid var(--border)",
            fontSize: 13,
          }}
        >
          No nodes discovered. Click "Scan Network" to search.
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "var(--space-4)",
          }}
        >
          {nodes.map((node, i) => (
            <div
              key={node.mac || i}
              style={{
                background: "var(--bg-surface)",
                borderRadius: 8,
                padding: "var(--space-4)",
                border: "1px solid var(--border)",
                transition: "border-color 0.15s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.borderColor = "var(--bg-active)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.borderColor = "var(--border)")
              }
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "start",
                  marginBottom: "var(--space-3)",
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14 }}>
                    {node.hostname || `Node ${node.node_id}`}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-muted)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {node.ip}
                  </div>
                </div>
                <StatusBadge status={node.health} />
              </div>

              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                  <span style={{ color: "var(--text-muted)" }}>MAC</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{node.mac || "--"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                  <span style={{ color: "var(--text-muted)" }}>Firmware</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{node.firmware_version || "--"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Node ID</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{node.node_id}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "var(--space-4)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-muted)",
          marginBottom: "var(--space-1)",
          fontFamily: "var(--font-sans)",
        }}
      >
        {label}
      </div>
      <div
        className="data-lg"
        style={{ color: color || "var(--text-primary)" }}
      >
        {value}
      </div>
    </div>
  );
}

export default Dashboard;
