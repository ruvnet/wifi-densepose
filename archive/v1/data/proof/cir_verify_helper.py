#!/usr/bin/env python3
"""
CIR Verification Helper (ADR-134)

Optional Python comparator — invokes the Rust cir_proof_runner binary and
checks its output against expected_cir_features.sha256.

Usage:
  python cir_verify_helper.py              # verify against stored hash
  python cir_verify_helper.py --generate  # regenerate hash via Rust binary

This script is a thin wrapper; all cryptographic work is done in the Rust
binary. It exists to integrate the CIR proof step into the Python verify.py
flow if needed.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))


def find_binary() -> str:
    """Locate the cir_proof_runner binary."""
    candidates = [
        os.path.join(REPO_ROOT, "v2", "target", "release", "cir_proof_runner"),
        os.path.join(REPO_ROOT, "v2", "target", "release", "cir_proof_runner.exe"),
        os.path.join(REPO_ROOT, "v2", "target", "debug", "cir_proof_runner"),
        os.path.join(REPO_ROOT, "v2", "target", "debug", "cir_proof_runner.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def build_binary() -> bool:
    """Build the release binary via cargo."""
    print("Building cir_proof_runner (release)...")
    result = subprocess.run(
        [
            "cargo", "build",
            "-p", "wifi-densepose-signal",
            "--bin", "cir_proof_runner",
            "--release",
            "--no-default-features",
        ],
        cwd=os.path.join(REPO_ROOT, "v2"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Build failed:", result.stderr[-2000:])
        return False
    return True


def run_generate(binary: str) -> str:
    """Run the binary with --generate-hash; return the hex hash."""
    result = subprocess.run(
        [binary, "--generate-hash"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error running binary:", result.stderr)
        return ""
    return result.stdout.strip()


def run_verify(binary: str) -> bool:
    """Run the binary in verify mode; return True on PASS."""
    result = subprocess.run(
        [binary],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CIR verification helper (ADR-134)")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate expected_cir_features.sha256 via Rust binary",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Build the binary before running (default: use cached binary)",
    )
    args = parser.parse_args()

    binary = find_binary()

    if args.build or not binary:
        if not build_binary():
            sys.exit(1)
        binary = find_binary()

    if not binary:
        print("ERROR: cir_proof_runner binary not found. Run with --build.")
        sys.exit(1)

    if args.generate:
        hash_val = run_generate(binary)
        if not hash_val:
            sys.exit(1)
        hash_file = os.path.join(SCRIPT_DIR, "expected_cir_features.sha256")
        with open(hash_file, "w") as f:
            f.write(hash_val + "\n")
        print(f"Wrote CIR hash to {hash_file}")
        print(f"Hash: {hash_val}")
    else:
        ok = run_verify(binary)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
