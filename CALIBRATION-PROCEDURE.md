# ESP32 Calibration Procedure

## Overview
Each ESP32 node must be calibrated to establish an empty-room baseline. This removes static environment interference (walls, furniture, reflections) so the sensing system can accurately detect dynamic motion and people.

**Duration**: ~5 minutes per node (30 seconds capture + data processing)
**Requirement**: Room must be completely empty during calibration

---

## Step-by-Step Calibration

### Phase 1: Prepare (1 minute)

1. **Position your 4 ESP32 nodes** in their final sensing locations
   - Node 1, 2, 3, 4 should be placed where they will stay
   - DO NOT move them after calibration (requires recalibration)

2. **Verify sensing-server is running** and receiving CSI from all 4 nodes
   ```bash
   curl http://localhost:8080/api/v1/sensing/latest | python3 -m json.tool
   ```
   You should see `node_id: 1`, `2`, `3`, `4` in the response

3. **Clear the room completely**
   - No people inside
   - No pets, fans, HVAC running if possible
   - Wait 30 seconds for any residual motion to settle

### Phase 2: Capture Baselines (2-3 minutes)

4. **Run calibration for Node 1**
   ```bash
   cd /home/austin/Documents/Software/Ruview/v2
   cargo run --release -p wifi-densepose-cli -- calibrate \
     --tier A \
     --duration 30 \
     --output /tmp/calib_node_1.toml
   ```
   - Watch the terminal; it will count frames and show progress
   - Should capture ~600 frames in 30 seconds
   - Output: `/tmp/calib_node_1.toml` (~7 KB)

5. **Repeat for Nodes 2, 3, 4**
   ```bash
   # Node 2 (wait for Node 1 to complete, room still empty)
   cargo run --release -p wifi-densepose-cli -- calibrate \
     --tier A \
     --duration 30 \
     --output /tmp/calib_node_2.toml
   
   # Node 3
   cargo run --release -p wifi-densepose-cli -- calibrate \
     --tier A \
     --duration 30 \
     --output /tmp/calib_node_3.toml
   
   # Node 4
   cargo run --release -p wifi-densepose-cli -- calibrate \
     --tier A \
     --duration 30 \
     --output /tmp/calib_node_4.toml
   ```

### Phase 3: Verify Baselines (1 minute)

6. **Check that all 4 baseline files were created**
   ```bash
   ls -lh /tmp/calib_*.toml
   ```
   Should show 4 files, each ~7 KB

7. **Inspect a baseline file** to verify reasonable values
   ```bash
   head -20 /tmp/calib_node_1.toml
   ```
   You should see:
   - `captured_at_utc`: timestamp when you ran it
   - `device_id`: which ESP32 (COM port)
   - `frame_count`: should be ~600
   - `n_subcarriers`: 52 (HT20)
   - `amp_mean`, `amp_variance`, `phase_*` arrays with numeric values

---

## Known Issues & Workarounds

### Issue: "cannot bind UDP socket"
**Cause**: sensing-server is already listening on the UDP port
**Solution**: Kill the old server first
```bash
pkill sensing-server
sleep 2
# Then re-run calibration
```

### Issue: Baseline file is empty or has NaN values
**Cause**: No CSI frames were received during calibration
**Solution**: 
1. Verify sensing-server is running and sending frames: `curl http://localhost:8080/health`
2. Check ESP32 nodes are transmitting (look at logs or LED)
3. Recalibrate with the server running and ESP32s active

### Issue: Amplitude values are much different between nodes
**Cause**: Normal — hardware differences in antenna gain, cable loss, regulatory domains
**Solution**: This is why per-node calibration exists. Each baseline captures its own hardware characteristics.

### Issue: Phase values look all zeros
**Cause**: Phase is captured as sin/cos components (not angles), so may look strange
**Solution**: This is normal. Phase stats are captured as circular means of sin(φ) and cos(φ).

---

## When to Recalibrate

You **must** recalibrate if:
- ✅ You move an ESP32 node to a different location
- ✅ You rearrange furniture significantly  
- ✅ You change the WiFi channel the nodes are listening on
- ✅ The baseline file is older than 1-2 weeks (seasonal thermal drift)

You **do NOT** need to recalibrate if:
- ❌ You restart the sensing-server
- ❌ Different people are in the room (calibration is environment, not people)
- ❌ You just turned on an ESP32 that was off

---

## Next: Integrate Baselines into Sensing-Server

Currently the sensing-server does NOT load pre-captured baselines automatically.

**TODO** (for future work):
- Implement `--calibration-dir /tmp/` flag in sensing-server
- Load baseline TOML files for each node on startup
- Apply baseline subtraction to all incoming CSI frames
- Store per-node baseline in memory

Without this integration, the baselines are captured but not used. The server's motion detection will work but with higher false-positive rates in drift-prone environments.

---

## Calibration File Format

Each TOML file contains:

```toml
[meta]
schema_version = 1
captured_at_utc = "2026-07-25T21:55:00Z"  # When you ran calibration
device_id = "esp32s3-com9"                 # Which ESP32
bandwidth_mhz = 20                         # HT20 (52 subcarriers)
tier = "A"                                 # ESP32-S3 tier
n_streams = 1                              # Spatial streams
n_subcarriers = 52                         # Active subcarriers
frame_count = 600                          # Frames captured

[[stream]]
stream_idx = 0

[stream.amp_mean]
values = [0.421, 0.418, 0.425, ...]  # Baseline amplitude per subcarrier

[stream.amp_variance]
values = [0.0012, 0.0009, 0.0015, ...]  # Amplitude variance per subcarrier

[stream.phase_cos_mean]
values = [0.871, 0.864, 0.879, ...]  # cos(φ) circular mean

[stream.phase_sin_mean]
values = [0.122, 0.134, 0.105, ...]  # sin(φ) circular mean

[stream.phase_circular_variance]
values = [0.031, 0.028, 0.035, ...]  # Phase spread (0=concentrated, 1=dispersed)
```

The sensing-server will subtract `amp_mean` from each incoming CSI frame to remove the static environment.

---

## Reference

- **ADR-135**: Empty-Room Baseline Calibration (`docs/adr/ADR-135-*`)
- **CLI Help**: `wifi-densepose calibrate --help`
- **Duration**: Default 30 seconds (recommended for presence detection)
