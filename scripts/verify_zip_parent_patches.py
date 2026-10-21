#!/usr/bin/env python3
"""Verify checked-in ZIP parent patch provenance/integrity; no network or writes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "v2" / "dependency-patches"
manifest = json.loads((root / "patches.json").read_text())
for package in manifest["packages"]:
    directory = root / package["package"]
    expected = {entry["path"]: entry for entry in package["files"]}
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    assert actual == set(expected), "Patched package inventory changed"
    changed = []
    for name, entry in expected.items():
        path = directory / name
        assert not path.is_symlink() and directory.resolve() in path.resolve().parents
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], "Patched package digest changed: " + name
        if digest != entry["upstreamSha256"]:
            changed.append(name)
    assert sorted(changed) == sorted(package["changedFiles"])
print("ZIP parent patch inventory and SHA256: PASS")
