# USRP / SDR Integration

For direct X310 RF sensing, prefer
[USRP X310 RF-Direct Integration](x310-rf-direct.md). This page documents the
older compatibility bridge that maps SDR features into WiFi CSI-like
amplitude/phase vectors.

RuView can run without ESP32 hardware by accepting CSI-like feature frames from
a USRP, UHD, GNU Radio, or other SDR pipeline.

This integration is a bridge contract, not a native UHD driver. USRP devices
produce raw IQ samples; RuView's sensing server expects per-frame amplitude and
phase vectors similar to WiFi CSI subcarriers. Put waveform-specific demodulation,
OFDM synchronization, or channel-estimation logic in a small bridge process, then
send JSON datagrams to RuView.

## Start RuView

```bash
cd v2
cargo run -p wifi-densepose-sensing-server -- \
  --source usrp \
  --usrp-udp-port 5010 \
  --http-port 3000 \
  --ws-port 3001 \
  --ui-path ../ui
```

Docker:

```bash
cd docker
CSI_SOURCE=usrp USRP_UDP_PORT=5010 docker compose up
```

Open `http://localhost:3000/ui/index.html`.

## UDP JSON Frame

Send one JSON object per UDP datagram to `127.0.0.1:5010`.

Minimum with amplitude/phase:

```json
{
  "node_id": 1,
  "sequence": 42,
  "freq_mhz": 2450,
  "sample_rate_hz": 20.0,
  "rssi_dbm": -52.0,
  "noise_floor_dbm": -95.0,
  "amplitudes": [0.91, 1.04, 0.98],
  "phases": [0.02, -0.04, 0.01]
}
```

Or send complex bins directly:

```json
{
  "node_id": 1,
  "sequence": 42,
  "freq_mhz": 2450,
  "sample_rate_hz": 20.0,
  "iq_pairs": [[0.91, 0.02], [1.04, -0.04], [0.98, 0.01]]
}
```

Fields:

| Field | Required | Notes |
|---|---:|---|
| `amplitudes` or `iq_pairs` | yes | CSI-like bins/features for one frame. |
| `phases` | no | Must match `amplitudes` length; omitted phases default to zero. |
| `sample_rate_hz` | no | Used for breathing-band estimation; defaults to 20 Hz. |
| `rssi_dbm` | no | Defaults to a rough power-derived value. |
| `noise_floor_dbm` | no | Defaults to -95 dBm. |
| `freq_mhz` | no | Metadata surfaced in the stream; defaults to 2400. |
| `node_id` | no | Defaults to 1. |
| `sequence` | no | Defaults to 0. |

## What This Enables

- Live UI, REST, and WebSocket paths without ESP32 hardware.
- USRP as a higher-fidelity RF front end when another process extracts features.
- Gradual migration from synthetic data to real SDR captures.
- A compatibility route for existing CSI-shaped RuView processing.

## What It Does Not Do Yet

- It does not decode 802.11 OFDM packets or compute CSI directly from raw USRP
  IQ inside RuView.
- It does not compensate for USRP clocking, antenna geometry, or multi-radio
  synchronization.
- It does not make ESP32-trained models automatically valid for your USRP
  waveform. Treat model output as experimental until calibrated on matched data.

Recommended next step for real WiFi CSI is a GNU Radio/UHD bridge that performs
packet detection, CFO correction, OFDM FFT, pilot/channel estimation, and emits
the resulting per-subcarrier channel vector using the JSON contract above.

If you do not need WiFi packet CSI, use `--source rf-direct` instead. The
RF-direct path accepts continuous-wave, multi-tone, FMCW, or other RF-native
features without pretending they are WiFi subcarriers.
