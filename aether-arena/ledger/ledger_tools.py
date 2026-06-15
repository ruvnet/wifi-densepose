#!/usr/bin/env python3
"""AetherArena append-only, tamper-evident results ledger (ADR-149 §2.3/§2.4).

Each row is hash-chained to the previous one: ``row_hash = sha256(canonical_row
+ prev_hash)``. Any silent edit to an earlier row breaks every subsequent
``prev_hash`` link, so the ledger is append-only and verifiable by anyone — no
trust in the maintainer required. (Ed25519 row signing is the next hardening;
the chain already makes tampering detectable.)

Usage:
    python ledger_tools.py seed        # (re)build ledger.jsonl with genesis + baseline
    python ledger_tools.py verify      # verify the whole chain  -> exit 0 / 1
    python ledger_tools.py append '<json-row>'   # append one scored row
"""
import hashlib
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).parent / "ledger.jsonl"
GENESIS_PREV = "0" * 64


def canonical(row: dict) -> bytes:
    # Stable key order, no whitespace -> deterministic bytes for hashing.
    body = {k: row[k] for k in sorted(row) if k != "row_hash"}
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


def row_hash(row: dict) -> str:
    return hashlib.sha256(canonical(row)).hexdigest()


def read_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def append(entry: dict) -> dict:
    rows = read_rows()
    prev = rows[-1]["row_hash"] if rows else GENESIS_PREV
    entry = dict(entry)
    entry["seq"] = len(rows)
    entry["prev_hash"] = prev
    entry["row_hash"] = row_hash(entry)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify() -> bool:
    rows = read_rows()
    prev = GENESIS_PREV
    for i, r in enumerate(rows):
        if r.get("seq") != i:
            print(f"FAIL: row {i} seq mismatch ({r.get('seq')})")
            return False
        if r.get("prev_hash") != prev:
            print(f"FAIL: row {i} prev_hash broken — ledger was edited")
            return False
        if r.get("row_hash") != row_hash(r):
            print(f"FAIL: row {i} row_hash mismatch — row was tampered")
            return False
        prev = r["row_hash"]
    print(f"OK: {len(rows)} rows, chain intact")
    return True


def seed():
    """Rebuild with the genesis row only — an EMPTY board.

    Benchmark-first: no placeholder/hand-entered numbers ever sit on the
    leaderboard. Every result row is produced by the real scoring pipeline
    (load model -> run inference -> score against the private eval split ->
    proof hash). The board starts empty and awaits the first real harness score,
    including RuView's own — which gets no special seeding.
    """
    if LEDGER.exists():
        LEDGER.unlink()
    append({
        "kind": "genesis",
        "benchmark": "AetherArena",
        "spec": "ADR-149",
        "note": "Official Spatial-Intelligence Benchmark — append-only signed ledger. "
                "Entries are real harness scores only; no seeded numbers.",
        "created": "2026-05-30",
    })


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "seed":
        seed(); verify()
    elif cmd == "verify":
        sys.exit(0 if verify() else 1)
    elif cmd == "append":
        print(json.dumps(append(json.loads(sys.argv[2])), indent=2))
    else:
        print(__doc__); sys.exit(2)
