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
  { id: "wasm", label: "Edge Modules", shortcut: "W" },
  { id: "sensing", label: "Sensing", shortcut: "S" },
  { id: "mesh", label: "Mesh View", shortcut: "M" },
  { id: "settings", label: "Settings", shortcut: "G" },
];

const App: React.FC = () => {
  const [activePage, setActivePage] = useState<Page>("dashboard");

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Sidebar */}
        <nav
          style={{
            width: "var(--sidebar-width)",
            minWidth: "var(--sidebar-width)",
            background: "var(--bg-surface)",
            borderRight: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Brand */}
          <div
            style={{
              padding: "var(--space-4)",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <h1
              style={{
                fontSize: 18,
                fontWeight: 700,
                color: "var(--accent)",
                fontFamily: "var(--font-sans)",
                margin: 0,
              }}
            >
              RuView
            </h1>
            <span
              style={{
                fontSize: 11,
                color: "var(--text-muted)",
                fontFamily: "var(--font-sans)",
              }}
            >
              WiFi DensePose Desktop
            </span>
          </div>

          {/* Nav items */}
          <div style={{ flex: 1, paddingTop: "var(--space-2)" }}>
            {NAV_ITEMS.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "var(--space-2)",
                    width: "100%",
                    padding: "10px var(--space-4)",
                    background: isActive ? "var(--bg-active)" : "transparent",
                    color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                    fontSize: 13,
                    fontWeight: isActive ? 600 : 400,
                    textAlign: "left",
                    borderLeft: isActive
                      ? "3px solid var(--accent)"
                      : "3px solid transparent",
                    fontFamily: "var(--font-sans)",
                  }}
                >
                  <span
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: 4,
                      background: isActive ? "var(--accent)" : "var(--bg-hover)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 10,
                      fontWeight: 700,
                      fontFamily: "var(--font-mono)",
                      color: isActive ? "#fff" : "var(--text-muted)",
                    }}
                  >
                    {item.shortcut}
                  </span>
                  {item.label}
                </button>
              );
            })}
          </div>

          {/* Version */}
          <div
            style={{
              padding: "var(--space-2) var(--space-4)",
              fontSize: 11,
              color: "var(--text-muted)",
              borderTop: "1px solid var(--border)",
              fontFamily: "var(--font-mono)",
            }}
          >
            v0.3.0
          </div>
        </nav>

        {/* Main content */}
        <main
          style={{
            flex: 1,
            overflow: "auto",
            background: "var(--bg-base)",
          }}
        >
          {activePage === "dashboard" && <Dashboard />}
          {activePage === "nodes" && <Nodes />}
          {activePage === "flash" && <FlashFirmware />}
          {activePage === "settings" && <Settings />}
          {!["dashboard", "nodes", "flash", "settings"].includes(activePage) && (
            <div style={{ padding: "var(--space-5)" }}>
              <h2
                style={{
                  fontSize: 20,
                  fontWeight: 600,
                  marginBottom: "var(--space-2)",
                }}
              >
                {NAV_ITEMS.find((n) => n.id === activePage)?.label}
              </h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 13 }}>
                This page is not yet implemented.
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Status Bar */}
      <footer
        style={{
          height: "var(--statusbar-height)",
          minHeight: "var(--statusbar-height)",
          background: "var(--bg-surface)",
          borderTop: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          padding: "0 var(--space-4)",
          gap: "var(--space-4)",
          fontSize: 11,
          fontFamily: "var(--font-sans)",
          color: "var(--text-muted)",
        }}
      >
        <span style={{ color: "var(--text-muted)" }}>Powered by rUv</span>
        <span style={{ color: "var(--border)" }}>|</span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "var(--status-online)",
              marginRight: 4,
              verticalAlign: "middle",
            }}
          />
          0 nodes online
        </span>
        <span style={{ color: "var(--border)" }}>|</span>
        <span>Server: stopped</span>
        <span style={{ color: "var(--border)" }}>|</span>
        <span style={{ fontFamily: "var(--font-mono)" }}>Port: 8080</span>
      </footer>
    </div>
  );
};

export default App;
