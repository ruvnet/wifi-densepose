import React, { useEffect, useState } from "react";

interface DiscoveredNode {
  ip: string;
  mac: string | null;
  hostname: string | null;
  node_id: number;
  firmware_version: string | null;
  health: string;
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

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 24,
        }}
      >
        <h2 style={{ fontSize: 24 }}>Dashboard</h2>
        <button
          onClick={handleScan}
          disabled={scanning}
          style={{
            padding: "8px 16px",
            background: scanning ? "#475569" : "#38bdf8",
            color: "#0f172a",
            border: "none",
            borderRadius: 6,
            cursor: scanning ? "not-allowed" : "pointer",
            fontWeight: 600,
          }}
        >
          {scanning ? "Scanning..." : "Scan Network"}
        </button>
      </div>

      {/* Server status panel */}
      <div
        style={{
          background: "#1e293b",
          borderRadius: 8,
          padding: 16,
          marginBottom: 24,
          border: "1px solid #334155",
        }}
      >
        <h3 style={{ fontSize: 14, color: "#94a3b8", marginBottom: 8 }}>
          Sensing Server
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: serverStatus?.running ? "#22c55e" : "#ef4444",
              display: "inline-block",
            }}
          />
          <span>
            {serverStatus?.running
              ? `Running (PID ${serverStatus.pid})`
              : "Stopped"}
          </span>
        </div>
      </div>

      {/* Node grid */}
      <h3
        style={{
          fontSize: 14,
          color: "#94a3b8",
          marginBottom: 12,
          textTransform: "uppercase",
          letterSpacing: 1,
        }}
      >
        Discovered Nodes ({nodes.length})
      </h3>

      {nodes.length === 0 ? (
        <div
          style={{
            background: "#1e293b",
            borderRadius: 8,
            padding: 32,
            textAlign: "center",
            color: "#64748b",
            border: "1px solid #334155",
          }}
        >
          No nodes discovered. Click "Scan Network" to search.
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 16,
          }}
        >
          {nodes.map((node, i) => (
            <div
              key={node.mac || i}
              style={{
                background: "#1e293b",
                borderRadius: 8,
                padding: 16,
                border: "1px solid #334155",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "start",
                  marginBottom: 12,
                }}
              >
                <div>
                  <div style={{ fontWeight: 600 }}>
                    {node.hostname || `Node ${node.node_id}`}
                  </div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>
                    {node.ip}
                  </div>
                </div>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: 12,
                    fontSize: 11,
                    fontWeight: 600,
                    background:
                      node.health === "online" ? "#064e3b" : "#7f1d1d",
                    color:
                      node.health === "online" ? "#34d399" : "#fca5a5",
                  }}
                >
                  {node.health}
                </span>
              </div>

              <div style={{ fontSize: 13, color: "#94a3b8" }}>
                <div>MAC: {node.mac || "unknown"}</div>
                <div>Firmware: {node.firmware_version || "unknown"}</div>
                <div>Node ID: {node.node_id}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
