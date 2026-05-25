#!/usr/bin/env python3
"""Fix safetensors files whose header zone is padded with NUL bytes.

The safetensors spec (https://github.com/huggingface/safetensors#format)
requires the header zone — the N bytes following the 8-byte u64 length
prefix — to be either valid JSON or JSON followed by ASCII space (0x20)
padding. Some writers (notably the JS SafeTensorsWriter in
vendor/ruvector/.../export.js) emit NUL (0x00) padding instead, which
strict readers (Rust safetensors crate, Candle, safetensors.torch.load_file)
reject with `SafetensorError: trailing characters at line 1 column N+1`.

This utility opens a .safetensors file, detects NUL padding in the header
zone, and rewrites the padding bytes in-place as ASCII spaces. The
declared header length, the JSON content, and every tensor byte are
preserved unchanged — only the padding bytes flip.

See docs/huggingface/SAFETENSORS-HEADER-BUG.md for the full bug analysis.

Usage:
    python scripts/fix-safetensors-header.py path/to/model.safetensors
    python scripts/fix-safetensors-header.py path/to/model.safetensors --dry-run
    python scripts/fix-safetensors-header.py models/*.safetensors

Exits:
    0  — file already clean, OR file was patched successfully
    1  — file is not a valid safetensors layout / could not be opened
    2  — bad CLI arguments
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


# Bytes that the spec accepts as valid header content. JSON is permitted
# to contain any printable ASCII; pad bytes are restricted to 0x20.
PAD_GOOD = 0x20  # ASCII space — the spec-required padding byte
PAD_BAD = 0x00   # NUL — what the buggy JS writer emits


def inspect_header(path: Path) -> tuple[int, int, bytes]:
    """Return (declared_header_len, json_end_offset_in_header, padding_bytes).

    `json_end_offset_in_header` is the offset within the header zone (i.e.
    relative to byte 8 of the file) where the JSON document ends. Anything
    after that and before `declared_header_len` is padding.

    Raises ValueError if the file is too short or the header is not JSON.
    """
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: file shorter than 8 bytes")
        (declared,) = struct.unpack("<Q", prefix)
        if declared <= 0 or declared > 100 * 1024 * 1024:
            raise ValueError(
                f"{path}: declared header length {declared} is implausible"
            )
        header = f.read(declared)
        if len(header) < declared:
            raise ValueError(
                f"{path}: file truncated — declared header len {declared}, "
                f"only {len(header)} bytes available"
            )

    # Find where the JSON document ends. The spec mandates the header start
    # with `{`, so we scan from the right for the matching `}` then check that
    # everything after is padding-class bytes.
    if not header or header[0] != ord("{"):
        raise ValueError(
            f"{path}: header does not start with '{{' (byte 0x{header[0]:02x}) — "
            "not a safetensors file"
        )

    # Walk from the end, skipping known padding-class bytes (NUL or space).
    json_end = len(header)
    while json_end > 0 and header[json_end - 1] in (PAD_GOOD, PAD_BAD):
        json_end -= 1
    if json_end == 0 or header[json_end - 1] != ord("}"):
        raise ValueError(
            f"{path}: could not locate end of JSON header (last non-pad byte "
            f"is 0x{header[json_end - 1]:02x} at offset {json_end - 1})"
        )

    padding = header[json_end:]
    return declared, json_end, padding


def classify(padding: bytes) -> str:
    """Return one of 'empty', 'spaces', 'nuls', 'mixed'."""
    if not padding:
        return "empty"
    has_nul = any(b == PAD_BAD for b in padding)
    has_space = any(b == PAD_GOOD for b in padding)
    if has_nul and not has_space:
        return "nuls"
    if has_space and not has_nul:
        return "spaces"
    return "mixed"


def fix_file(path: Path, dry_run: bool = False) -> bool:
    """Rewrite the header padding zone of `path` to use 0x20.

    Returns True if the file was modified (or would be, in dry-run mode);
    False if it was already clean. Raises ValueError on malformed input.
    """
    declared, json_end, padding = inspect_header(path)
    cls = classify(padding)

    if cls in ("empty", "spaces"):
        print(f"  [ok]      {path}  ({len(padding)} pad bytes already clean)")
        return False

    new_padding = bytes([PAD_GOOD] * len(padding))
    print(
        f"  [{'would patch' if dry_run else 'patched   '}] {path}  "
        f"({len(padding)} {cls} pad bytes -> spaces, declared header "
        f"length {declared} unchanged)"
    )

    if dry_run:
        return True

    # Open in r+b and overwrite only the padding bytes. This preserves
    # every other byte (declared length, JSON header, tensor payload) and
    # the file's overall size.
    with path.open("r+b") as f:
        f.seek(8 + json_end)
        f.write(new_padding)
        f.flush()

    # Re-inspect to confirm the rewrite landed.
    _, _, after = inspect_header(path)
    if classify(after) not in ("empty", "spaces"):
        raise RuntimeError(
            f"{path}: post-write inspection shows padding is still '{classify(after)}'"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See docs/huggingface/SAFETENSORS-HEADER-BUG.md for full context.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more .safetensors files to inspect / fix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without rewriting any bytes",
    )
    args = parser.parse_args()

    any_error = False
    any_changed = False
    print(f"Inspecting {len(args.paths)} file(s){' (dry-run)' if args.dry_run else ''}...")
    for path in args.paths:
        if not path.exists():
            print(f"  [error]   {path}  (does not exist)", file=sys.stderr)
            any_error = True
            continue
        try:
            changed = fix_file(path, dry_run=args.dry_run)
        except ValueError as exc:
            print(f"  [error]   {exc}", file=sys.stderr)
            any_error = True
            continue
        except OSError as exc:
            print(f"  [error]   {path}  ({exc})", file=sys.stderr)
            any_error = True
            continue
        any_changed = any_changed or changed

    if any_error:
        return 1
    if args.dry_run and any_changed:
        print("\nDry-run finished. Re-run without --dry-run to apply the fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
