# Pull Request: Flipper Zero WiFi Dev Board Support & Unified CLI

## Description
This PR adds official build support for the **Flipper Zero WiFi Dev Board (ESP32-S2)** and introduces a unified All-in-One CLI to streamline node management, scanning, and benchmarking. It also performs a structural cleanup by archiving legacy components.

## Key Changes
1.  **Flipper Zero Support:** Added `firmware/esp32-csi-node/sdkconfig.defaults.esp32s2` to enable seamless compilation for the Flipper WiFi Dev board.
2.  **Unified CLI (`ruview.py`):** Created a comprehensive entry point that replaces multiple fragmented scripts.
    *   `node build/flash/provision`: Manage firmware lifecycle.
    *   `scan live/wide`: Visualize RF spectrum.
    *   `sense server`: Launch the V2 Rust engine.
    *   `bench report`: Generate ADR-031 metric acceptance summaries.
3.  **Project Organization:**
    *   Moved legacy Python 1.1.0 code (`wifi_densepose/`) to `archive/v1/`.
    *   Consolidated design references into `archive/references/`.
4.  **Optimizations:**
    *   Pre-wired the CLI to use **Port 5005** for production sensing.
    *   Added support for **HE (WiFi 6) Mesh** configuration via CLI.

## Verification Results
- **Connectivity:** Flipper board verified active at `192.168.1.129`.
- **CSI Stream:** Confirmed UDP frames flowing to Port 5005.
- **Build Test:** `./ruview.py node build --chip esp32s2` successfully generates the S2 binary.
- **V2 Engine:** Rust sensing server confirmed operational with the Flipper board.

## Checklist
- [x] Firmware builds for ESP32-S3, ESP32-C6, and ESP32-S2.
- [x] CLI `ruview.py` is executable and documentation is updated.
- [x] Legacy folders archived without breaking main build paths.
- [x] README.md updated with Flipper Zero specific instructions.

*Prepared by Gemini CLI on behalf of the user.*
