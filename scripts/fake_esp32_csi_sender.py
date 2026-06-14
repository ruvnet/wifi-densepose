#!/usr/bin/env python3
"""Fake ESP32 CSI UDP sender — emits ADR-018 binary frames to the sensing-server.

Wire format (matches parse_esp32_frame in wifi-densepose-sensing-server/src/main.rs
and firmware csi_collector.c):

    [0..3]   magic = 0xC511_0001 (u32 LE)
    [4]      node_id (u8)
    [5]      n_antennas (u8)
    [6..7]   n_subcarriers (u16 LE)
    [8..11]  freq_mhz (u32 LE)
    [12..15] sequence (u32 LE)
    [16]     rssi (i8)
    [17]     noise_floor (i8)
    [18]     ppdu_type (u8, 0 = HT/legacy)
    [19]     reserved
    [20..]   I/Q data: n_antennas * n_subcarriers pairs of (i8 I, i8 Q)

Use this to exercise the --source esp32 ingestion path without real hardware.
"""
import argparse
import math
import socket
import struct
import time

MAGIC = 0xC511_0001


def build_frame(node_id, n_antennas, n_subcarriers, freq_mhz, sequence,
                rssi, noise_floor, t, breath_hz, motion):
    hdr = struct.pack(
        "<IBBHIIbbBB",
        MAGIC, node_id, n_antennas, n_subcarriers, freq_mhz, sequence,
        rssi, noise_floor, 0, 0,
    )
    # Synthesize I/Q: a steady carrier + a slow breathing oscillation + motion noise.
    body = bytearray()
    breath = math.sin(2 * math.pi * breath_hz * t)
    for a in range(n_antennas):
        for k in range(n_subcarriers):
            base = 18 + (k % 30)
            amp = base + 8 * breath + motion * math.sin(k * 0.4 + t * 6.0)
            phase = 0.3 * k + t * 1.5
            i = int(max(-127, min(127, amp * math.cos(phase))))
            q = int(max(-127, min(127, amp * math.sin(phase))))
            body += struct.pack("bb", i, q)
    return hdr + bytes(body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--nodes", type=int, default=2, help="number of nodes (node_id 1..N)")
    p.add_argument("--antennas", type=int, default=1)
    p.add_argument("--subcarriers", type=int, default=64)
    p.add_argument("--freq-mhz", type=int, default=2462)
    p.add_argument("--rate", type=float, default=20.0, help="frames/sec per node")
    p.add_argument("--rssi", type=int, default=-47)
    p.add_argument("--noise", type=int, default=-90)
    p.add_argument("--breath-hz", type=float, default=0.25, help="~15 BPM")
    p.add_argument("--motion", type=float, default=4.0, help="0 = still, higher = more motion")
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = forever")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / args.rate
    seq = 0
    start = time.time()
    print(f"Sending CSI frames -> {args.host}:{args.port} | "
          f"{args.nodes} node(s), {args.subcarriers} subcarriers, {args.rate} Hz")
    try:
        while True:
            t = time.time() - start
            for node_id in range(1, args.nodes + 1):
                frame = build_frame(
                    node_id, args.antennas, args.subcarriers, args.freq_mhz,
                    seq, args.rssi, args.noise, t, args.breath_hz, args.motion,
                )
                sock.sendto(frame, (args.host, args.port))
            seq += 1
            if seq % int(args.rate) == 0:
                print(f"  t={t:5.1f}s  sent seq={seq} ({len(frame)} B/frame)")
            if args.duration and t >= args.duration:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    print(f"Done. Sent {seq} sequences to {args.nodes} node(s).")


if __name__ == "__main__":
    main()
