# WiFi DensePose UI

A modular, modern web interface for the WiFi DensePose human tracking system. Provides real-time monitoring, WiFi sensing visualization, 3D pose estimation, vital sign tracking, and model training — all driven by live CSI (Channel State Information) from ESP32 hardware or the macOS WiFi bridge.

## Architecture

```
ui/
├── app.js                    # Main application entry point
├── index.html                # HTML shell with tab structure (System dashboard)
├── home.html                 # Consumer-friendly home page (presence, vitals, signal quality)
├── viz.html                  # 3D pose visualization (Three.js + WebSocket integration)
├── sensing-dashboard.html    # Technical sensing dashboard (spectrum, waterfall, heatmap)
├── style.css                 # Complete CSS design system
├── start-ui.sh               # Quick-start shell script
├── config/
│   └── api.config.js         # API endpoints and configuration
├── services/
│   ├── api.service.js         # HTTP API client
│   ├── websocket.service.js   # WebSocket connection manager (sensing tab)
│   ├── websocket-client.js    # WebSocket client for 3D viz (pose stream, source tracking)
│   ├── pose.service.js        # Pose estimation API wrapper
│   ├── sensing.service.js     # WiFi sensing data service (live + simulation fallback)
│   ├── health.service.js      # Health monitoring API wrapper
│   ├── stream.service.js      # Streaming API wrapper
│   ├── data-processor.js      # Signal data processing (keypoint normalization, heatmaps)
│   ├── model.service.js       # RVF model management (load, list, LoRA profiles)
│   └── training.service.js    # Training lifecycle, CSI recording, progress streaming
├── components/
│   ├── TabManager.js          # Tab navigation component
│   ├── DashboardTab.js        # Dashboard with live system metrics
│   ├── SensingTab.js          # WiFi sensing visualization (3D signal field, metrics)
│   ├── LiveDemoTab.js         # Live pose detection with setup guide
│   ├── HardwareTab.js         # Hardware configuration
│   ├── SettingsPanel.js       # Settings panel
│   ├── ModelPanel.js          # Model management panel (list, load, LoRA profiles)
│   ├── TrainingPanel.js       # Training panel (record CSI, train, progress charts)
│   ├── PoseDetectionCanvas.js # Canvas-based pose skeleton renderer
│   ├── gaussian-splats.js     # 3D Gaussian splat signal field renderer (Three.js)
│   ├── body-model.js          # 3D body model (COCO 17-keypoint skeleton)
│   ├── scene.js               # Three.js scene management (renderer, camera, controls)
│   ├── signal-viz.js          # Signal visualization (Doppler, amplitude, phase rings)
│   ├── environment.js         # Environment/room visualization (APs, zones, heatmap)
│   └── dashboard-hud.js       # HUD overlay (FPS, connection, source, confidence)
├── utils/
│   ├── backend-detector.js    # Auto-detect backend availability
│   ├── mock-server.js         # Mock server for testing
│   └── pose-renderer.js       # Pose rendering utilities
├── mobile/                    # React Native mobile app (Expo)
│   ├── App.tsx                # Mobile entry point
│   ├── e2e/                   # Maestro E2E test flows
│   └── ...
└── tests/
    ├── test-runner.html       # Test runner UI
    ├── test-runner.js         # Test framework and cases
    └── integration-test.html  # Integration testing page
```

## Pages

### Home (`home.html`)
Consumer-friendly dashboard designed for non-technical users.
- **Presence hero**: large animated indicator showing room occupancy status
- **Vital signs**: breathing rate (animated lung ring) and heart rate (pulsing heart ring) with confidence bars
- **Signal quality**: 5-bar strength meter, RSSI value, and data source indicator
- **Environment details**: motion energy, dominant frequency, variability, change events
- **Activity timeline**: color-coded motion bar chart (blue=calm, green=moving, amber=walking)
- Live WebSocket connection to `/ws/sensing`

### 3D Visualization (`viz.html`)
Full 3D pose visualization powered by Three.js with real server integration.
- **Three.js scene**: room environment with access point models, zone overlays, confidence heatmap
- **Body model**: COCO 17-keypoint skeleton driven by live pose data from the server
- **Signal visualization**: Doppler spectrum, amplitude rings, phase indicators
- **Dashboard HUD overlay**: connection status, FPS, person count, confidence, sensing mode
- **Auto data source detection**: connects to `ws://<host>/api/v1/stream/pose`, automatically switches from demo mode to live server data when pose frames arrive
- **Sensing mode display**: shows actual source (CSI, Simulated, WiFi) instead of hardcoded labels
- **Keypoint normalization**: auto-detects pixel coordinates from server and normalizes to [0,1] for the body model
- Keyboard shortcuts: `R` reset camera, `D` toggle demo, `C` force reconnect

### Sensing Dashboard (`sensing-dashboard.html`)
Technical dashboard for signal engineers.
- **RSSI chart**: live signal strength timeline
- **Subcarrier spectrum**: 56-channel amplitude bar chart
- **Vital signs chart**: overlaid breathing + heart rate timelines
- **Subcarrier waterfall**: time-frequency spectrogram
- **Motion timeline**: color-coded motion energy bars with walking threshold
- **Signal field heatmap**: 20x20 spatial grid with peak marker
- **Classification badge**: ABSENT / STILL / MOVING / WALKING with confidence

### System Dashboard (`index.html`)
Multi-tab system overview.
- **Dashboard**: system health, API status, live statistics, zone occupancy, benefit cards
- **Hardware**: interactive 3x3 antenna array, CSI data display, WiFi config
- **Live Demo**: WebSocket pose skeleton with setup guide and debug mode
- **Architecture**: pipeline flow diagram (CSI -> Phase Sanitization -> CNN -> DensePose-RCNN)
- **Performance**: AP metrics comparison (WiFi vs image-based)
- **Applications**: use case cards (elderly care, security, healthcare, smart building, AR/VR)
- **Sensing**: WiFi sensing visualization tab
- **Training**: CSI recording, model training with progress charts, RVF model management

## Data Flow

### Real Integration (ESP32 / WiFi Bridge)

```
ESP32/Bridge ──UDP:5005──> Rust Server ──WS──> UI
                              │
                              ├── /ws/sensing           → home.html, sensing-dashboard.html
                              └── /api/v1/stream/pose   → viz.html (pose_data messages)
```

The server converts raw `sensing_update` broadcasts into `pose_data` messages for the 3D viz:
1. `udp_receiver_task` parses ESP32 CSI frames, extracts features, classifies presence
2. `broadcast_tick_task` sends sensing updates via broadcast channel
3. `ws_pose_handler` subscribes, converts to COCO 17-keypoint pose, sends to viz clients
4. `data-processor.js` normalizes keypoints (pixel coords -> [0,1]) and extracts metadata
5. `viz.html` switches from demo mode to live mode when persons are detected

### Simulation Fallback

When no hardware is detected, the server generates simulated CSI data. The UDP listener still runs in the background, so if real frames arrive later (e.g. WiFi bridge starts after the server), the simulation automatically yields and real data takes over.

### Demo Mode (Client-Side)

If the WebSocket connection fails entirely, `viz.html` falls back to client-side demo mode with pre-recorded pose cycles (standing, walking, arms raised, sitting, waving).

## Data Sources

| Source | HUD Label | Description |
|--------|-----------|-------------|
| **ESP32 CSI** | CSI | Real CSI frames from ESP32 hardware via UDP |
| **macOS WiFi Bridge** | CSI | `macos_wifi_bridge.py` captures native WiFi frames, sends as ESP32-format UDP |
| **Simulated** | Simulated | Server-generated synthetic CSI (fallback when no hardware) |
| **Demo** | Demo | Client-side pre-recorded poses (WebSocket disconnected) |

## Backends

### Rust Sensing Server (primary)
The Rust-based `wifi-densepose-sensing-server` serves the UI and provides:
- `GET /health` — server health
- `GET /api/v1/sensing/latest` — latest sensing features
- `GET /api/v1/vital-signs` — vital sign estimates (HR/RR)
- `GET /api/v1/model/info` — RVF model container info
- `GET /api/v1/models` — list discovered RVF models
- `GET /api/v1/models/active` — currently loaded model
- `GET /api/v1/pose/latest` — latest pose detection
- `GET /api/v1/pose/stats` — pose detection statistics
- `GET /api/v1/pose/zones/summary` — zone occupancy summary
- `WS /ws/sensing` — real-time sensing data stream (features, classification, vitals)
- `WS /api/v1/stream/pose` — real-time pose keypoint stream (COCO 17-keypoint format)

### Python FastAPI (legacy)
The original Python backend on port 8000 is still supported. The UI auto-detects which backend is available via `backend-detector.js`.

## Quick Start

### With the run_all.sh script
```bash
# Builds the Rust server, starts it on port 8080, optionally starts the WiFi bridge
bash run_all.sh
```
Open http://localhost:8080/ui/home.html

### With Docker
```bash
cd docker/

# Default: auto-detects ESP32 on UDP 5005, falls back to simulation
docker-compose up

# Force real ESP32 data
CSI_SOURCE=esp32 docker-compose up

# Force simulation (no hardware needed)
CSI_SOURCE=simulated docker-compose up
```
Open http://localhost:3000/ui/home.html

### With local Rust binary
```bash
cd rust-port/wifi-densepose-rs
cargo build --release -p wifi-densepose-sensing-server --no-default-features

# Auto-detect (simulation + background UDP listener for real hardware)
./target/release/sensing-server --ui-path ../../ui

# Force ESP32 mode
./target/release/sensing-server --source esp32 --ui-path ../../ui

# With a trained model for full pose inference
./target/release/sensing-server --source esp32 --model path/to/model.rvf --ui-path ../../ui
```
Open http://localhost:8080/ui/home.html

### With macOS WiFi Bridge (no ESP32 hardware needed)
```bash
# Terminal 1: start the server
./target/release/sensing-server --ui-path ../../ui

# Terminal 2: start the WiFi bridge (captures native macOS WiFi frames)
python3 scripts/macos_wifi_bridge.py --mac-wifi ./mac_wifi --port 5005
```
The server auto-detects bridge frames on UDP:5005 and switches from simulation to real data.

## Pose Estimation Modes

| Mode | HUD Badge | Requirements | Accuracy |
|------|-----------|-------------|----------|
| **Signal-Derived** | Simulated/CSI | 1+ ESP32, no model needed | Presence, breathing, gross motion, signal-derived skeleton |
| **Model Inference** | CSI | 4+ ESP32s + trained `.rvf` model | Full 17-keypoint COCO pose with limb tracking |

## Key Services

### `websocket-client.js`
Low-level WebSocket client for the 3D viz page. Features:
- Auto-reconnect with exponential backoff (500ms to 30s, up to 15 attempts)
- Heartbeat ping/pong every 25s
- `dataSource` property tracks actual server source string (`"esp32"`, `"simulated"`, etc.)
- `isRealData` flag properly detects server data via `payload.metadata.source`
- Connection metrics: message count, latency, bytes received, uptime

### `data-processor.js`
Transforms server pose messages into Three.js-ready data. Features:
- Handles both `data` and `payload` message formats
- Auto-normalizes pixel coordinates to [0,1] (detects values >1.5 as pixel coords)
- Generates confidence heatmaps from person positions
- Demo mode with smoothly interpolated pre-recorded COCO poses
- Source-to-mode mapping: `esp32`->`CSI`, `simulated`->`Simulated`, `wifi`->`WiFi`

### `model.service.js` / `training.service.js`
Model management and training lifecycle:
- List, load, and switch between RVF models
- LoRA profile management
- CSI recording start/stop with server-side persistence
- Training progress streaming via WebSocket
- Training configuration (epochs, learning rate, batch size)

## Configuration

### API Configuration
Edit `config/api.config.js`:

```javascript
export const API_CONFIG = {
  BASE_URL: window.location.origin,
  API_VERSION: '/api/v1',
  WS_CONFIG: {
    RECONNECT_DELAY: 5000,
    MAX_RECONNECT_ATTEMPTS: 20,
    PING_INTERVAL: 30000
  }
};
```

### Server CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--http-port` | 8080 | HTTP server port |
| `--ws-port` | 8765 | Dedicated WebSocket port |
| `--udp-port` | 5005 | UDP port for ESP32 CSI frames |
| `--ui-path` | `../../ui` | Path to UI static files |
| `--source` | `auto` | Data source: `auto`, `esp32`, `wifi`, `simulate` |
| `--tick-ms` | 100 | Tick interval (100ms = 10 fps) |
| `--model` | — | Path to trained `.rvf` model file |

## Testing

Open `tests/test-runner.html` to run the test suite:

```bash
cd ui/
python -m http.server 3000
# Open http://localhost:3000/tests/test-runner.html
```

Test categories: API configuration, API service, WebSocket, pose service, health service, UI components, integration.

## Mobile App

A React Native (Expo) companion app lives in `mobile/`. See `mobile/README.md` for setup. Includes Maestro E2E tests for live, MAT, vitals, zones, settings, and offline screens.

## Styling

Uses a CSS design system with custom properties, dark/light mode, responsive layout, and component-based styling. Key variables in `:root` of `style.css`. The `home.html` and `sensing-dashboard.html` pages use self-contained inline styles for standalone operation.

## License

Part of the WiFi-DensePose system. See the main project LICENSE file.
