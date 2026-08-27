# RuView Operator UI

A mobile-first, function-first web console for the WiFi-DensePose sensing system. Real-time presence, vital-sign, signal-field and pose monitoring from CSI (Channel State Information), plus node health and model/training control.

## Architecture

Presentation is a small vanilla-JS SPA styled with a **compiled Tailwind build** (no runtime CDN — works offline on the Pi appliance). The proven data layer in `services/` is reused as-is; views bind directly to the live `/api/v1/*` REST endpoints and the `/ws/sensing` stream.

```
ui/
├── index.html                # Responsive shell (sidebar / bottom tab bar)
├── package.json              # Tailwind + jsdom dev deps, build scripts
├── tailwind.config.js        # Theme tokens (brand teal, ink surfaces)
├── src/input.css             # Tailwind directives + component layer (source)
├── assets/app.css            # COMPILED CSS — committed, served at runtime
├── app/
│   ├── main.js               # Entry: layout, hash router, status header
│   ├── lib.js                # DOM + formatting helpers, toast, sparkline
│   ├── icons.js              # Inline SVG icon set
│   └── views/
│       ├── dashboard.js      # Live operator overview (presence, vitals, system)
│       ├── sensing.js        # CSI features + classification + field heatmap
│       ├── nodes.js          # Per-node health, mesh, hardware reference
│       ├── demo.js           # Live pose / detection canvas
│       ├── training.js       # Models, recordings, training-run control
│       └── about.js          # Folded Architecture/Performance/Applications
├── services/                 # Reused data layer (sensing/health/pose/api/ws…)
└── sw.js, manifest.json      # PWA (network-first service worker)
```

Navigation is function-first: **Dashboard · Sensing · Nodes · Live Demo · Training**, with **About** (the former marketing/info pages) demoted to the end. The standalone Three.js pages — `pose-fusion.html` and `observatory.html` — are linked from the nav, not re-implemented.

## Build

```bash
cd ui
npm install            # tailwindcss + jsdom (dev only)
npm run build:css      # src/input.css -> assets/app.css (commit the output)
npm run watch:css      # rebuild on change during development
```

`assets/app.css` is checked in so the server can serve the UI without a build step on the target.

## Features

### WiFi Sensing Tab
- 3D Gaussian-splat signal field visualization (Three.js)
- Real-time RSSI, variance, motion band, breathing band metrics
- Presence/motion classification with confidence scores
- **Data source banner**: green "LIVE - ESP32", yellow "RECONNECTING...", or red "SIMULATED DATA"
- Sparkline RSSI history graph
- "About This Data" card explaining CSI capabilities per sensor count

### Live Demo Tab
- WebSocket-based real-time pose skeleton rendering
- **Estimation Mode badge**: green "Signal-Derived" or blue "Model Inference"
- **Setup Guide panel** showing what each ESP32 count provides:
  - 1 ESP32: presence, breathing, gross motion
  - 2-3 ESP32s: body localization, motion direction
  - 4+ ESP32s + trained model: individual limb tracking, full pose
- Debug mode with log export
- Zone selection and force-reconnect controls
- Performance metrics sidebar (frames, uptime, errors)

### Dashboard
- Live system health monitoring
- Real-time pose detection statistics
- Zone occupancy tracking
- System metrics (CPU, memory, disk)
- API status indicators

### Hardware Configuration
- Interactive antenna array visualization
- Real-time CSI data display
- Configuration panels
- Hardware status monitoring

## Data Sources

The sensing service (`sensing.service.js`) supports three connection states:

| State | Banner Color | Description |
|-------|-------------|-------------|
| **LIVE - ESP32** | Green | Connected to the Rust sensing server receiving real CSI data |
| **RECONNECTING** | Yellow (pulsing) | WebSocket disconnected, retrying (up to 20 attempts) |
| **SIMULATED DATA** | Red | Fallback to client-side simulation after 5+ failed reconnects |

Simulated frames include a `_simulated: true` marker so code can detect synthetic data.

## Backends

### Rust Sensing Server (primary)
The Rust-based `wifi-densepose-sensing-server` serves the UI and provides:
- `GET /health` — server health
- `GET /api/v1/sensing/latest` — latest sensing features
- `GET /api/v1/vital-signs` — vital sign estimates (HR/RR)
- `GET /api/v1/model/info` — RVF model container info
- `WS /ws/sensing` — real-time sensing data stream
- `WS /api/v1/stream/pose` — real-time pose keypoint stream

### Python FastAPI (legacy)
The original Python backend on port 8000 is still supported. The UI auto-detects which backend is available via `backend-detector.js`.

## Quick Start

### With Docker (recommended)
```bash
cd docker/

# Default: auto-detects ESP32 on UDP 5005, falls back to simulation
docker-compose up

# Force real ESP32 data
CSI_SOURCE=esp32 docker-compose up

# Force simulation (no hardware needed)
CSI_SOURCE=simulated docker-compose up
```
Open http://localhost:3000/ui/index.html

### With local Rust binary
```bash
cd v2
cargo build -p wifi-densepose-sensing-server --no-default-features

# Run with simulated data
../../target/debug/sensing-server --source simulated --tick-ms 100 --ui-path ../../ui --http-port 3000

# Run with real ESP32
../../target/debug/sensing-server --source esp32 --tick-ms 100 --ui-path ../../ui --http-port 3000
```
Open http://localhost:3000/ui/index.html

### With Python HTTP server (legacy)
```bash
# Start FastAPI backend on port 8000
wifi-densepose start

# Serve the UI on port 3000
cd ui/
python -m http.server 3000
```
Open http://localhost:3000

## Pose Estimation Modes

| Mode | Badge | Requirements | Accuracy |
|------|-------|-------------|----------|
| **Signal-Derived** | Green | 1+ ESP32, no model needed | Presence, breathing, gross motion |
| **Model Inference** | Blue | 4+ ESP32s + trained `.rvf` model | Full 17-keypoint COCO pose |

To use model inference, start the server with a trained model:
```bash
sensing-server --source esp32 --model path/to/model.rvf --ui-path ./ui
```

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

## Testing

Open `tests/test-runner.html` to run the test suite:

```bash
cd ui/
python -m http.server 3000
# Open http://localhost:3000/tests/test-runner.html
```

Test categories: API configuration, API service, WebSocket, pose service, health service, UI components, integration.

## Styling

Tailwind CSS, dark-first. Theme tokens (brand teal ramp, `ink` surface ramp, status colours) live in `tailwind.config.js`; reusable component classes (`.card`, `.btn-*`, `.nav-link`, `.stat`, `.badge-*`, `.meter`) are defined in the `@layer components` block of `src/input.css`. Edit those, then `npm run build:css`. The layout is mobile-first: a desktop sidebar (`md:` and up) collapses to a fixed bottom tab bar on small screens, with `env(safe-area-inset-*)` padding for notched devices.

## License

Part of the WiFi-DensePose system. See the main project LICENSE file.
