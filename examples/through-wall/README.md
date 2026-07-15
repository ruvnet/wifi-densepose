# WiFlow Browser Trainer (`wiflow_browser.html`)

A **single self-contained HTML page** that does the entire camera-supervised
WiFi-pose loop **in your browser, in your laptop camera's coordinate frame**, as
a **4-stage gated flow** with a progress stepper (each stage unlocks the next):

0. **CALIBRATE** *(ADR-151 empty-room baseline)* — you step OUT of the space; the
   page captures ~10 s of the quiescent CSI and computes a per-feature running
   **mean + std (Welford)** over the 410-d vector. Every CSI vector afterwards is
   expressed as **deviation from baseline**
   (`x_norm = (x − base_mean) / (base_std + ε)`), so a body's perturbation stands
   out from the static channel. Persisted to IndexedDB. *Can't capture without it.*
1. **CAPTURE** — MediaPipe Pose runs on your laptop camera → 17 COCO keypoints
   (the *label*), paired with the **baseline-normalized** 410-d ESP32 CSI vector
   (the *input*). A **guided, balanced routine** cycles big on-screen prompts
   (stand / turn / walk / arms / crouch / sit / reach) with a countdown, and a
   **per-pose coverage meter** so you build a balanced dataset, not 2 000 frames
   of standing.
2. **TRAIN** — a TensorFlow.js MLP learns `CSI → pose` in-browser. Honest
   held-out PCK@0.10 / PCK@0.05 / MPJPE, plus a **mean-pose baseline** the model
   must beat (the project's whole ethos — no baseline-beating signal, it says so).
   *Can't train with <200 samples.*
3. **INFER** — the trained model drives a skeleton **from WiFi CSI only**
   (baseline-normalized → standardized → model), drawn over the **same** camera
   frame it trained in — so the inferred skeleton **aligns** with the camera
   image. That alignment is the entire point of doing this in-browser instead of
   with a separate Python camera. *Can't infer without a model.*

## Why in-browser

The Python pipeline (`wiflow_capture.py` → `wiflow_train.py` → `wiflow_infer.py`)
proved the signal is real (held-out PCK@0.10 ≈ 59.5% vs a 50% mean-pose baseline
= +9.4 pp). But it trained in a *different* camera's frame, so the inferred
skeleton never lined up with the laptop camera. Doing capture + train + infer all
in the browser with the **same** camera makes the training frame and the
inference frame identical → the skeleton aligns.

## Compute backends (WebGPU / WASM / WebGL)

Training and inference run on TensorFlow.js. The page selects the backend at
startup, preferring the fastest available:

- **WebGPU** (Chrome / Edge, secure context — `localhost` qualifies) — GPU compute.
- **WASM-SIMD** fallback (`tfjs-backend-wasm`, SIMD enabled, `.wasm` from the CDN).
- **WebGL** last-resort fallback (ships inside tfjs core).

The **active backend is shown as a badge in the header** (`compute: WebGPU` /
`WASM-SIMD` / `WebGL`) so it's honest about what's actually running. The model
code is backend-agnostic — tf.js abstracts the device.

## Honesty (baked in)

- The **CAPTURE** skeleton (blue) is the camera = ground truth, labeled as such.
- The **INFER** skeleton (green) is **CSI-only**, labeled, and **coarse** — the
  real measured held-out PCK is shown, not a marketing number.
- The **mean-pose baseline** is always computed and shown in TRAIN; the verdict
  states plainly whether the model **beats** it (real signal) or **does not**
  (no usable signal). This guards against the project's retracted 92.9% that
  failed exactly this check.
- Status banner is strict and mutually exclusive:
  **LIVE** (real `source: "esp32"`) / **SIMULATED — not real** (any other source)
  / **NO-CSI-SERVER**. The page never invents frames.

## How to run

### 1. Start the real sensing-server (provides the CSI WebSocket on :8765)

```bash
cd v2
cargo build -p wifi-densepose-sensing-server
./target/debug/sensing-server.exe --ws-port 8765 --udp-port 5005
```

A real ESP32-S3 must be provisioned and streaming for `source` to read `esp32`
(see `CLAUDE.local.md` for the firmware build/provision steps). The page expects
the verified live endpoint **`ws://localhost:8765/ws/sensing`** with
`source:"esp32"`, nodes `[9, 13]`, `features.*`, `node_features[].features.*`,
and `signal_field.values` (400 floats).

### 2. Serve this page over localhost (camera + WebGPU need a localhost/secure origin)

Any static localhost server works. For example:

```bash
python -m http.server 8099
# then open: http://localhost:8099/examples/through-wall/wiflow_browser.html
```

(8099 is just the static file server — 8765 is a separate process, the CSI
WebSocket.) Allow camera access when the browser prompts.

Point at a CSI server on another host with `?ws=`:

```
http://localhost:8099/examples/through-wall/wiflow_browser.html?ws=ws://192.168.1.20:8765/ws/sensing
```

### 3. Use it

1. **CAPTURE** tab → *enable laptop camera* → *start recording*. Follow the guided
   routine (stand / turn / walk / arms / crouch / sit). A pair is stored only when
   a confident pose AND a fresh live `esp32` CSI frame coexist. Aim for a few
   thousand samples. Samples persist in IndexedDB across refreshes.
2. **TRAIN** tab → *train model*. Watch the live loss curve, held-out PCK, and the
   baseline verdict. The model saves to IndexedDB.
3. **INFER** tab → the green skeleton is now driven by WiFi CSI only, aligned over
   your camera. Toggle *hide camera* to see the CSI-only skeleton on black.

## The 410-d CSI vector (matches the Python pipeline exactly)

```
[ mean_rssi, variance, motion_band_power, breathing_band_power ]   # 4  (features.*)
+ for node 9 then node 13: [ mean_rssi, variance, motion_band_power ]  # 6 (node_features[].features.*)
+ signal_field.values, padded / truncated to 400                  # 400
= 410-d
```

Verified against a real live frame: the in-browser `csiVector()` produces the
identical 410 vector as `wiflow_capture.py`'s `csi_vector()` (node 9 first, then
node 13; field zero-padded).

## Libraries (CDN only, no bundler)

| Library | CDN |
|---|---|
| TensorFlow.js core | `@tensorflow/tfjs@4.22.0/dist/tf.min.js` |
| TF.js WebGPU backend | `@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js` |
| TF.js WASM backend | `@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js` |
| MediaPipe Pose 0.5 (legacy solutions) | `@mediapipe/pose@0.5/pose.js` |

## Scope / honesty caveats

Same person, same room, same session. **Not** validated cross-day, cross-room, or
through-wall. The inferred pose is coarse (PCK@0.05 is typically weak). If the
model does not beat the mean-pose baseline, the page says so — that is a feature.
