#!/usr/bin/env python3
"""
macOS WiFi → UDP bridge for the WiFi-DensePose sensing server.

Reads real RSSI/noise/tx_rate from the compiled mac_wifi Swift helper and
packs each reading into the ESP32 binary frame format expected by the
sensing server's UDP listener on port 5005.

The server auto-detects these frames and switches from simulation to live
WiFi data (hot-plug).

Usage:
    python3 scripts/macos_wifi_bridge.py [--mac-wifi ./mac_wifi] [--port 5005]
"""

import argparse
import json
import math
import socket
import struct
import subprocess
import sys
import time


MAGIC = 0xC511_0001
NODE_ID = 1
N_ANTENNAS = 1
N_SUBCARRIERS = 56  # match simulated frame size

UDP_HOST = "127.0.0.1"


def build_esp32_frame(seq: int, rssi: float, noise: float, tx_rate: float) -> bytes:
    """Pack a WiFi reading into the binary ESP32 frame format.

    Layout (little-endian):
        [0:4]   u32  magic        0xC5110001
        [4]     u8   node_id
        [5]     u8   n_antennas
        [6]     u8   n_subcarriers
        [7]     u8   (reserved)
        [8:10]  u16  freq_mhz
        [10:14] u32  sequence
        [14]    i8   rssi
        [15]    i8   noise_floor
        [16:20] (reserved / padding)
        [20..]  i8 pairs (I, Q) × n_antennas × n_subcarriers

    We synthesize per-subcarrier I/Q values from the RSSI + noise so the
    server's feature extractor has plausible amplitude/phase distributions.
    """
    rssi_i8 = max(-128, min(127, int(rssi)))
    noise_i8 = max(-128, min(127, int(noise)))

    # Derive a base amplitude from RSSI (higher RSSI → larger amplitude)
    snr = max(rssi - noise, 1.0)
    base_amp = snr / 2.0  # scale into a reasonable I/Q range

    # 20-byte header matching parse_esp32_frame() layout exactly:
    #   [0:4]  u32 LE magic, [4] node_id, [5] n_antennas, [6] n_subcarriers,
    #   [7] reserved, [8:10] u16 LE freq_mhz, [10:14] u32 LE sequence,
    #   [14] i8 rssi, [15] i8 noise_floor, [16:20] reserved
    header = struct.pack(
        "<IBBBBHIbb4x",
        MAGIC,
        NODE_ID,
        N_ANTENNAS,
        N_SUBCARRIERS,
        0,              # reserved
        2437,           # freq_mhz (channel 6, 2.4 GHz)
        seq,
        rssi_i8,
        noise_i8,
    )

    t = time.time()
    rate_factor = tx_rate / 400.0 if tx_rate > 0 else 0.5

    iq_data = bytearray()
    for i in range(N_SUBCARRIERS):
        phase = math.sin(i * 0.2 + t * 0.5) * math.pi
        amp = base_amp * (0.8 + 0.4 * math.sin(i * 0.15 + t * 0.3)) * rate_factor
        amp = max(1.0, min(127.0, amp))
        i_val = int(amp * math.cos(phase))
        q_val = int(amp * math.sin(phase))
        i_val = max(-128, min(127, i_val))
        q_val = max(-128, min(127, q_val))
        iq_data.append(i_val & 0xFF)
        iq_data.append(q_val & 0xFF)

    return header + bytes(iq_data)


def main():
    parser = argparse.ArgumentParser(description="macOS WiFi → sensing server bridge")
    parser.add_argument("--mac-wifi", default="./mac_wifi", help="Path to mac_wifi binary")
    parser.add_argument("--port", type=int, default=5005, help="UDP port for sensing server")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[bridge] Starting mac_wifi helper: {args.mac_wifi}")
    print(f"[bridge] Sending ESP32 frames to {UDP_HOST}:{args.port}")

    proc = subprocess.Popen(
        [args.mac_wifi],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**__import__("os").environ, "NSUnbufferedIO": "YES"},
    )

    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    seq = 0
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            rssi = data.get("rssi", -70)
            noise = data.get("noise", -90)
            tx_rate = data.get("tx_rate", 0.0)
            seq += 1

            frame = build_esp32_frame(seq, rssi, noise, tx_rate)
            sock.sendto(frame, (UDP_HOST, args.port))

            if seq % 10 == 1:
                print(f"[bridge] #{seq:>5d}  RSSI={rssi:>4d} dBm  noise={noise:>4d} dBm  "
                      f"tx_rate={tx_rate:>6.1f} Mbps  frame={len(frame)} bytes")

    except KeyboardInterrupt:
        print("\n[bridge] Stopped.")
    finally:
        proc.terminate()
        proc.wait()
        sock.close()


if __name__ == "__main__":
    main()
