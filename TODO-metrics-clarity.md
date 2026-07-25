# TODO: Clarify /health Metrics (Issue #1125 Follow-up)

## Problem
The `/health` endpoint returns `"clients": 5` which conflates:
- **UI browser connections** (dashboard, observatory tabs)
- **Actual ESP32 nodes** (hardware sending real CSI data)

Users can't tell from the metric how many real nodes are connected vs. how many people are viewing the dashboard.

## Current Implementation
```json
GET /health
{
  "clients": 5,           // WebSocket broadcast subscribers (browsers + nodes)
  "source": "esp32",      // Data source (esp32, wifi, waiting_for_hardware)
  "status": "ok",
  "tick": 6904
}
```

## Proposed Fix
Split into two metrics:

```json
{
  "ui_clients": 2,        // Active browser UI connections (dashboard, observatory, etc)
  "active_nodes": 3,      // ESP32/WiFi nodes currently sending CSI frames
  "source": "esp32",      // Aggregated source
  "status": "ok",
  "tick": 6904
}
```

## Implementation Notes
- `ui_clients`: Count WebSocket connections from `localhost` or specific known UI origins
- `active_nodes`: Count ESP32 nodes from `node_features` array in latest sensing data
- Requires filtering by connection origin or source IP in the server's health handler
- See `v2/crates/wifi-densepose-sensing-server/src/main.rs` line 3427 (health handler)

## Impact
- Users immediately see how many real nodes are connected
- Dashboard clearly shows "3 active nodes, 2 UI viewers" vs. misleading "5 clients"
- Helps debug connectivity issues (missing nodes, extra browser tabs, etc.)
