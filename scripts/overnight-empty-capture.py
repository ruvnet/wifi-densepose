#!/usr/bin/env python3
"""Segmented overnight empty-room CSI capture (ADR-135 baseline / MAE corpus).

Binds UDP once and writes fixed-duration JSONL segments with explicit names —
no post-hoc renaming, no glob collisions with other recordings.

Usage:
    python scripts/overnight-empty-capture.py --segments 8 --segment-seconds 3300
"""

import argparse
import json
import os
import socket
import struct
import time


def parse_csi_packet(data):
    """ADR-018 binary CSI packet → dict (same layout as record-csi-udp.py)."""
    if len(data) < 8:
        return None
    node_id = data[4]
    rssi = struct.unpack("b", bytes([data[6]]))[0]
    channel = data[7]
    iq = data[8:]
    amplitudes = []
    for i in range(0, len(iq) - 1, 2):
        I = struct.unpack("b", bytes([iq[i]]))[0]
        Q = struct.unpack("b", bytes([iq[i + 1]]))[0]
        amplitudes.append(round((I * I + Q * Q) ** 0.5, 2))
    return {
        "type": "raw_csi",
        "ts_ns": time.time_ns(),
        "node_id": node_id,
        "rssi": rssi,
        "channel": channel,
        "subcarriers": len(iq) // 2,
        "amplitudes": amplitudes,
        "iq_hex": iq.hex(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--segment-seconds", type=int, default=3300)
    ap.add_argument("--output", default="data/recordings")
    ap.add_argument("--prefix", default="overnight-empty")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(2.0)

    for seg in range(1, args.segments + 1):
        path = os.path.join(
            args.output, f"{args.prefix}-seg{seg}-{int(time.time())}.csi.jsonl"
        )
        n = 0
        t_end = time.time() + args.segment_seconds
        with open(path, "w", encoding="utf-8") as f:
            while time.time() < t_end:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                rec = parse_csi_packet(data)
                if rec is not None:
                    f.write(json.dumps(rec) + "\n")
                    n += 1
        print(f"segment {seg}: {n} frames -> {path}", flush=True)

    print("capture complete", flush=True)


if __name__ == "__main__":
    main()
