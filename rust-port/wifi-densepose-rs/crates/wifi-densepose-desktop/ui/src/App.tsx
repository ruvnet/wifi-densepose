import React, { useState } from "react";
import Dashboard from "./pages/Dashboard";
import { Nodes } from "./pages/Nodes";
import { FlashFirmware } from "./pages/FlashFirmware";
import { Settings } from "./pages/Settings";

type Page =
  | "dashboard"
  | "nodes"
  | "flash"
  | "ota"
  | "wasm"
  | "sensing"
  | "mesh"
  | "settings";

interface NavItem {
  id: Page;
  label: string;
  shortcut: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: "dashboard", label: "Dashboard", shortcut: "D" },
  { id: "nodes", label: "Nodes", shortcut: "N" },
  { id: "flash", label: "Flash", shortcut: "F" },
  { id: "ota", label: "OTA", shortcut: "O" },
  { id: "wasm", label: "WASM", shortcut: "W" },
  { id: "sensing", label: "Sensing", shortcut: "S" },
  { id: "mesh", label: "Mesh", shortcut: "M" },
  { id: "settings", label: "Settings", shortcut: "G" },
];

const App: React.FC = () => {
  const [activePage, setActivePage] = useState<Page>("dashboard");

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <nav
        style={{
          width: 200,
          background: "#1e293b",
          borderRight: "1px solid #334155",
          display: "flex",
          flexDirection: "column",
          padding: "16px 0",
        }}
      >
        <div
          style={{
            padding: "0 16px 16px",
            borderBottom: "1px solid #334155",
            marginBottom: 8,
          }}
        >
          <h1
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: "#38bdf8",
            }}
          >
            RuView
          </h1>
          <span style={{ fontSize: 11, color: "#64748b" }}>
            WiFi DensePose Desktop
          </span>
        </div>

        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setActivePage(item.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              width: "100%",
              padding: "10px 16px",
              border: "none",
              background:
                activePage === item.id ? "#334155" : "transparent",
              color:
                activePage === item.id ? "#f1f5f9" : "#94a3b8",
              cursor: "pointer",
              fontSize: 14,
              textAlign: "left",
              borderLeft:
                activePage === item.id
                  ? "3px solid #38bdf8"
                  : "3px solid transparent",
            }}
          >
            <span
              style={{
                width: 20,
                height: 20,
                borderRadius: 4,
                background:
                  activePage === item.id ? "#38bdf8" : "#475569",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 11,
                fontWeight: 700,
                color:
                  activePage === item.id ? "#0f172a" : "#94a3b8",
              }}
            >
              {item.shortcut}
            </span>
            {item.label}
          </button>
        ))}

        <div style={{ flex: 1 }} />
        <div
          style={{
            padding: "8px 16px",
            fontSize: 11,
            color: "#475569",
            borderTop: "1px solid #334155",
          }}
        >
          v0.3.0
        </div>
      </nav>

      {/* Main content */}
      <main style={{ flex: 1, overflow: "auto", padding: 24 }}>
        {activePage === "dashboard" && <Dashboard />}
        {activePage === "nodes" && <Nodes />}
        {activePage === "flash" && <FlashFirmware />}
        {activePage === "settings" && <Settings />}
        {!["dashboard", "nodes", "flash", "settings"].includes(activePage) && (
          <div>
            <h2 style={{ fontSize: 24, marginBottom: 8 }}>
              {NAV_ITEMS.find((n) => n.id === activePage)?.label}
            </h2>
            <p style={{ color: "#64748b" }}>
              This page is not yet implemented.
            </p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
