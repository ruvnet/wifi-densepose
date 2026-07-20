#!/usr/bin/env python3
"""Send USRP/SDR CSI-like feature frames to RuView over UDP.

This is a protocol shim, not a UHD receiver. Use `--demo` to generate a stable
test stream, or pipe JSONL frames produced by GNU Radio/UHD with `--jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path
from typing import Iterable


def demo_frames(rate_hz: float, bins: int) -> Iterable[dict]:
    seq = 0
    period = 1.0 / rate_hz
    while True:
        t = seq * period
        amplitudes = []
        phases = []
        for i in range(bins):
            carrier = 1.0 + 0.08 * math.sin(i * 0.17)
            breathing = 0.035 * math.sin(2.0 * math.pi * 0.25 * t + i * 0.03)
            motion = 0.12 * math.sin(2.0 * math.pi * 1.2 * t + i * 0.11)
            amplitudes.append(max(0.001, carrier + breathing + motion))
            phases.append(0.08 * math.sin(2.0 * math.pi * 0.25 * t + i * 0.07))
        yield {
            "node_id": 1,
            "sequence": seq,
            "freq_mhz": 2450,
            "sample_rate_hz": rate_hz,
            "rssi_dbm": -48.0 + 1.5 * math.sin(t * 0.5),
            "noise_floor_dbm": -95.0,
            "amplitudes": amplitudes,
            "phases": phases,
        }
        seq += 1
        time.sleep(period)


def jsonl_frames(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--bins", type=int, default=56)
    parser.add_argument("--demo", action="store_true", help="send generated feature frames")
    parser.add_argument("--jsonl", type=Path, help="send one JSON frame per input line")
    args = parser.parse_args()

    if not args.demo and args.jsonl is None:
        parser.error("choose --demo or --jsonl PATH")

    frames = demo_frames(args.rate_hz, args.bins) if args.demo else jsonl_frames(args.jsonl)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    for frame in frames:
        payload = json.dumps(frame, separators=(",", ":")).encode("utf-8")
        sock.sendto(payload, target)
        if args.jsonl is not None:
            time.sleep(1.0 / args.rate_hz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
