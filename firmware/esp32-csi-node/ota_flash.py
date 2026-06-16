#!/usr/bin/env python3
"""Upload a RuView ESP32 app image through the node's HTTP OTA endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "build" / "esp32-csi-node.bin"


def run_curl(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["curl", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def curl_common(host: str, interface: str | None, timeout: int) -> list[str]:
    args = ["--fail-with-body", "--show-error", "--silent", "--max-time", str(timeout)]
    if interface:
        args.extend(["--interface", interface])
    return args


def get_status(host: str, interface: str | None, timeout: int) -> dict[str, object]:
    url = f"http://{host}:8032/ota/status"
    result = run_curl([*curl_common(host, interface, timeout), url], timeout + 2)
    if result.returncode != 0:
        raise RuntimeError(
            f"status failed for {url}: {result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"status returned non-JSON: {result.stdout[:200]!r}") from exc


def upload(host: str, image: Path, token: str, interface: str | None, timeout: int) -> str:
    url = f"http://{host}:8032/ota"
    result = run_curl(
        [
            *curl_common(host, interface, timeout),
            "--request",
            "POST",
            "--header",
            f"Authorization: Bearer {token}",
            "--header",
            "Content-Type: application/octet-stream",
            "--data-binary",
            f"@{image}",
            url,
        ],
        timeout + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"upload failed for {url}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flash a running RuView node over HTTP OTA on port 8032."
    )
    parser.add_argument("host", help="Node IP address or hostname, for example 192.168.1.161")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"RuView app image to upload (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--psk",
        default=os.environ.get("RUVIEW_OTA_PSK"),
        help="OTA bearer token. Defaults to RUVIEW_OTA_PSK.",
    )
    parser.add_argument(
        "--interface",
        default=os.environ.get("RUVIEW_OTA_IFACE", "wlan0"),
        help="Network interface for curl, or empty string to let routing choose (default: wlan0).",
    )
    parser.add_argument("--timeout", type=int, default=30, help="curl timeout in seconds")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only print /ota/status; do not upload firmware.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.expanduser().resolve()
    interface = args.interface or None

    try:
        status = get_status(args.host, interface, args.timeout)
        print(json.dumps(status, indent=2, sort_keys=True))

        if args.status_only:
            return 0

        if not args.psk:
            print("error: --psk or RUVIEW_OTA_PSK is required for upload", file=sys.stderr)
            return 2
        if not image.is_file():
            print(f"error: image not found: {image}", file=sys.stderr)
            return 2

        max_size = int(status.get("max_size") or 0)
        image_size = image.stat().st_size
        if max_size and image_size > max_size:
            print(
                f"error: image is too large for next OTA partition "
                f"({image_size} > {max_size} bytes)",
                file=sys.stderr,
            )
            return 2

        print(f"uploading {image} ({image_size} bytes) to {args.host}:8032 ...")
        print(upload(args.host, image, args.psk, interface, args.timeout))
        return 0
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
