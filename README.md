# RuView — Unified WiFi Sensing Platform

RuView is a state-of-the-art platform for passive human sensing using WiFi CSI (Channel State Information). It transforms standard WiFi signals into high-resolution pose estimation, presence detection, and vital sign monitoring.

## 🚀 Quick Start (All-in-One CLI)

The `ruview.py` tool is your unified entry point for everything in the ecosystem.

```bash
# 1. Check if your hardware is connected
./ruview.py status

# 2. Start the sensing server (V2 Rust Engine)
./ruview.py sense server

# 3. View the live RF spectrum
./ruview.py scan live

# 4. Generate a Full ADR-031 System Report
./ruview.py bench report
```

---

## 📂 Project Organization

| Folder | Contents |
| :--- | :--- |
| **`v2/`** | **The Core Engine.** High-performance Rust implementation (ADR-117+). Includes the sensing server, pose estimation, and CLI. |
| **`models/`** | Pre-trained Hugging Face models (DensePose, CSI Encoders, Edge-optimized variants). |
| **`scripts/`** | Python and Node.js utilities for provisioning, data capture, and legacy watching. |
| **`ui/`** | Modern React-based desktop and web dashboard. |
| **`firmware/`** | ESP32-S3 and ESP32-C6 firmware source (Wi-Fi 6, iTWT, and mesh sync support). |
| **`docs/`** | Full architectural decision records (ADRs), release notes, and user guides. |
| **`aether-arena/`** | Official SOTA benchmarking and room calibration tools. |

---

## 📡 Hardware: Flipper Zero WiFi Dev Board

Your dev board is currently **active and streaming**. 
*   **IP Address**: `192.168.1.129`
*   **Target Port**: `5005` (Production Sensing)
*   **Status**: Good and Going.

### Optimization Commands
To get the most out of your hardware:
```bash
# Calibrate for your specific room (30s capture)
./ruview.py tune calibrate

# Setup HE (WiFi 6) Mesh (C6 Boards only)
./ruview.py node mesh
```

---

## 🛠️ V2 Support & Features

Yes, this project fully supports **RuView V2** (ADR-117). The V2 engine is built in Rust for 100x lower latency and includes:

*   **BFLD (Privacy Layer)**: Structural prevention of identity leakage.
*   **Multistatic Fusion**: High-accuracy 3D pose estimation from multiple nodes.
*   **Apple Home / Matter**: Native integration without needing Home Assistant.
*   **75K-param Transformer**: Beats academic SOTA while running on a single CPU thread.

---

## 📖 Commands Reference

### **`node`**
*   `flash`: Flash the latest firmware to your board.
*   `provision`: Set WiFi SSID/Pass and Target IP.
*   `mesh`: Enable high-resolution HE (WiFi 6) mesh mode.

### **`scan`**
*   `live`: Real-time spectrum visualization.
*   `wide`: Wideband multi-frequency environment scanning.

### **`sense`**
*   `server`: Start the V2 Rust Sensing Engine (Port 5005).
*   `watcher`: Simple Python-based presence alert watcher.

### **`bench`**
*   `report`: Generate a full SOTA performance report.
*   `aa`: Run the AetherArena official score runner.

### **`tune`**
*   `calibrate`: Record a room-baseline for improved accuracy.
*   `adapt`: Apply few-shot LoRA adaptation to your room.

---

*Built by [rUv](https://github.com/ruvnet). Released under MIT/Apache-2.0.*
