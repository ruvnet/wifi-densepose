"""
Phase 3 — RuViewOccDataset: WorldGraph history → OccWorld-format tensors.

Replaces OccWorld's nuScenesSceneDatasetLidar with a loader that reads
WorldGraph JSON snapshots produced by wifi-densepose-worldgraph and returns
(B, F, H, W, D) occupancy tensors in the same format OccWorld expects.

Class mapping (18-class OccWorld schema):
    RuView class    → OccWorld index   nuScenes label
    free / unknown  → 17               free
    person          → 7                pedestrian
    wall / ceiling  → 11               other-flat (closest structural)
    floor           → 9                terrain
    furniture       → 16               other-object
    door / window   → 14               bicycle (repurposed for portals)

Ego-pose: indoor fixed sensor has no ego-motion. rel_poses are all zeros,
which suppresses the pose-prediction head without affecting occupancy output.

Usage (standalone validation):
    python3 scripts/ruview_occ_dataset.py --snapshots /tmp/snapshots/ --check

Usage (as OccWorld dataset replacement):
    from ruview_occ_dataset import RuViewOccDataset
    ds = RuViewOccDataset(snapshot_dir="/tmp/snapshots", return_len=16)
    sample = ds[0]  # dict with keys: img_metas, target_occs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

# ── OccWorld voxel grid constants ───────────────────────────────────────────
GRID_H = 200        # X (east)
GRID_W = 200        # Y (north)
GRID_D = 16         # Z (up)

NUM_CLASSES = 18
FREE_CLASS = 17
PERSON_CLASS = 7
FLOOR_CLASS = 9
WALL_CLASS = 11
FURNITURE_CLASS = 16
DOOR_CLASS = 14

# Default spatial extent matching nuScenes at 0.4 m/voxel
DEFAULT_VOXEL_M = 0.4        # metres per voxel
DEFAULT_X_MIN = -40.0        # east min (m)
DEFAULT_Y_MIN = -40.0        # north min (m)
DEFAULT_Z_MIN = -1.0         # up min (m)
DEFAULT_Z_STEP = 0.4         # metres per depth slice


# ── WorldGraph snapshot format ───────────────────────────────────────────────

def _load_snapshot(path: Path) -> dict:
    """Load a WorldGraph JSON snapshot from disk."""
    with open(path) as f:
        return json.load(f)


def _extract_persons(snapshot: dict) -> list[tuple[float, float, float]]:
    """Return list of (east_m, north_m, up_m) for all PersonTrack nodes."""
    persons = []
    nodes = snapshot.get("nodes", {})
    if isinstance(nodes, dict):
        items = nodes.values()
    elif isinstance(nodes, list):
        items = nodes
    else:
        return persons

    for node in items:
        kind = node.get("kind") or node.get("type") or ""
        if "person" in kind.lower() or "PersonTrack" in kind:
            pos = node.get("last_position") or node.get("position") or {}
            e = float(pos.get("east_m", pos.get("e", 0.0)))
            n = float(pos.get("north_m", pos.get("n", 0.0)))
            u = float(pos.get("up_m", pos.get("u", 0.0)))
            persons.append((e, n, u))

    return persons


def _extract_room_bounds(snapshot: dict) -> dict[str, float] | None:
    """Try to extract room bounds from a ZoneBoundsEnu node, else return None."""
    nodes = snapshot.get("nodes", {})
    if isinstance(nodes, dict):
        items = nodes.values()
    elif isinstance(nodes, list):
        items = nodes
    else:
        return None

    for node in items:
        kind = node.get("kind") or node.get("type") or ""
        if "room" in kind.lower() or "zone" in kind.lower():
            bounds = node.get("bounds") or {}
            if "min_e" in bounds:
                return {
                    "x_min": float(bounds["min_e"]),
                    "x_max": float(bounds["max_e"]),
                    "y_min": float(bounds["min_n"]),
                    "y_max": float(bounds["max_n"]),
                }
    return None


def snapshot_to_voxels(
    snapshot: dict,
    voxel_m: float = DEFAULT_VOXEL_M,
    x_min: float = DEFAULT_X_MIN,
    y_min: float = DEFAULT_Y_MIN,
    z_min: float = DEFAULT_Z_MIN,
    z_step: float = DEFAULT_Z_STEP,
) -> np.ndarray:
    """
    Convert a WorldGraph snapshot to a (H, W, D) uint8 occupancy voxel grid.

    Parameters
    ----------
    snapshot : WorldGraph JSON dict
    voxel_m  : metres per horizontal voxel
    x_min, y_min, z_min : spatial origin in ENU metres
    z_step   : metres per depth slice

    Returns
    -------
    np.ndarray of shape (GRID_H, GRID_W, GRID_D), dtype uint8, values in [0,17]
    """
    grid = np.full((GRID_H, GRID_W, GRID_D), FREE_CLASS, dtype=np.uint8)

    # Mark floor slice (D=0) as terrain
    grid[:, :, 0] = FLOOR_CLASS

    persons = _extract_persons(snapshot)
    for (e, n, u) in persons:
        xi = int((e - x_min) / voxel_m)
        yi = int((n - y_min) / voxel_m)
        zi = int((u - z_min) / z_step)
        # Person occupies a 2-voxel vertical column (standing height ≈ 1.8 m)
        for dz in range(min(5, GRID_D)):
            zz = zi + dz
            if 0 <= xi < GRID_H and 0 <= yi < GRID_W and 0 <= zz < GRID_D:
                grid[xi, yi, zz] = PERSON_CLASS

    return grid


# ── Dataset class ────────────────────────────────────────────────────────────

class RuViewOccDataset:
    """
    OccWorld-compatible dataset backed by WorldGraph JSON snapshots.

    Expected directory layout::

        snapshot_dir/
            scene_000/
                frame_000.json
                frame_001.json
                ...
            scene_001/
                ...

    Each frame_NNN.json is a WorldGraph JSON snapshot (as produced by
    wifi-densepose-worldgraph's to_json() method or the sensing server's
    /api/v1/worldgraph/snapshot endpoint).

    Parameters
    ----------
    snapshot_dir : root directory containing scene sub-directories
    return_len   : number of consecutive frames per sample (matches OccWorld num_frames+offset)
    voxel_m      : metres per horizontal voxel
    x_min, y_min, z_min, z_step : spatial grid parameters
    test_mode    : if True, disable augmentation (always True for inference)
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        return_len: int = 16,
        voxel_m: float = DEFAULT_VOXEL_M,
        x_min: float = DEFAULT_X_MIN,
        y_min: float = DEFAULT_Y_MIN,
        z_min: float = DEFAULT_Z_MIN,
        z_step: float = DEFAULT_Z_STEP,
        test_mode: bool = True,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.return_len = return_len
        self.voxel_m = voxel_m
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.z_step = z_step
        self.test_mode = test_mode

        self._scenes: list[list[Path]] = self._index()

    def _index(self) -> list[list[Path]]:
        """Walk snapshot_dir and build a list of frame-path sequences."""
        scenes: list[list[Path]] = []
        root = self.snapshot_dir

        if not root.exists():
            return scenes

        # Support flat layout (root/*.json) and scene layout (root/scene/*/*.json)
        json_files = sorted(root.glob("*.json"))
        if json_files:
            # Flat layout — treat as a single scene
            scenes.append(json_files)
        else:
            for scene_dir in sorted(root.iterdir()):
                if scene_dir.is_dir():
                    frames = sorted(scene_dir.glob("*.json"))
                    if frames:
                        scenes.append(frames)

        return scenes

    def _sliding_windows(self) -> list[tuple[int, int]]:
        """Return (scene_idx, frame_start) pairs for all valid windows."""
        windows = []
        for si, frames in enumerate(self._scenes):
            for fi in range(len(frames) - self.return_len + 1):
                windows.append((si, fi))
        return windows

    def __len__(self) -> int:
        return sum(
            max(0, len(f) - self.return_len + 1) for f in self._scenes
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return a dict compatible with OccWorld's data loader expectations::

            {
              "img_metas": [{"scene_token": ..., "frame_idx": ...}],
              "target_occs": np.ndarray (F, H, W, D) uint8,
              "rel_poses": np.ndarray (F, 3, 4) float32  — all zeros,
            }
        """
        windows = self._sliding_windows()
        if idx >= len(windows):
            raise IndexError(idx)

        si, fi = windows[idx]
        frame_paths = self._scenes[si][fi : fi + self.return_len]

        voxels_seq = []
        for fp in frame_paths:
            snap = _load_snapshot(fp)
            v = snapshot_to_voxels(
                snap,
                voxel_m=self.voxel_m,
                x_min=self.x_min,
                y_min=self.y_min,
                z_min=self.z_min,
                z_step=self.z_step,
            )
            voxels_seq.append(v)

        target_occs = np.stack(voxels_seq, axis=0)  # (F, H, W, D)

        # Zero ego-poses: indoor fixed sensor has no ego-motion
        rel_poses = np.zeros((self.return_len, 3, 4), dtype=np.float32)

        return {
            "img_metas": [{
                "scene_token": self._scenes[si][fi].parent.name,
                "frame_idx": fi,
                "source": "ruview_worldgraph",
            }],
            "target_occs": target_occs,
            "rel_poses": rel_poses,
        }


# ── Snapshot recorder helper ─────────────────────────────────────────────────

def record_snapshot(worldgraph_json: dict, out_dir: Path, frame_idx: int) -> Path:
    """
    Save a WorldGraph JSON snapshot to out_dir/frame_NNN.json.

    Call this from the sensing server or a WorldGraph event listener to
    accumulate training data for Phase 5 VQVAE retraining.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{frame_idx:06d}.json"
    with open(out_path, "w") as f:
        json.dump(worldgraph_json, f)
    return out_path


# ── CLI validation ───────────────────────────────────────────────────────────

def _make_synthetic_snapshot(
    person_pos: tuple[float, float, float] = (1.0, 1.0, 0.0)
) -> dict:
    """Create a minimal synthetic WorldGraph snapshot for testing."""
    return {
        "nodes": [
            {
                "kind": "PersonTrack",
                "id": 1,
                "last_position": {
                    "east_m": person_pos[0],
                    "north_m": person_pos[1],
                    "up_m": person_pos[2],
                },
            }
        ],
        "edges": [],
    }


def _cli_check() -> None:
    """Validate RuViewOccDataset with synthetic data."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        scene_dir = Path(tmpdir) / "scene_000"
        scene_dir.mkdir()

        # Write 20 synthetic snapshots: person walks east at 0.5 m/frame
        for i in range(20):
            snap = _make_synthetic_snapshot(person_pos=(float(i) * 0.5, 2.0, 0.0))
            (scene_dir / f"frame_{i:06d}.json").write_text(json.dumps(snap))

        ds = RuViewOccDataset(tmpdir, return_len=16)
        print(f"Dataset length: {len(ds)} windows")
        assert len(ds) == 5, f"Expected 5 windows, got {len(ds)}"

        sample = ds[0]
        occ = sample["target_occs"]
        print(f"target_occs shape: {occ.shape}  dtype: {occ.dtype}")
        assert occ.shape == (16, GRID_H, GRID_W, GRID_D)

        # Check person voxels present in first frame
        assert (occ[0] == PERSON_CLASS).any(), "No person voxels in frame 0"
        print(f"Person voxels in frame 0: {(occ[0] == PERSON_CLASS).sum()}")

        # Check floor voxels
        assert (occ[0, :, :, 0] == FLOOR_CLASS).any(), "No floor in frame 0"

        # Check rel_poses are zeros
        assert (sample["rel_poses"] == 0).all(), "rel_poses should be all zeros"

        print("rel_poses shape:", sample["rel_poses"].shape, "— all zeros:", (sample["rel_poses"] == 0).all())
        print("\nVALIDATION PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuViewOccDataset — Phase 3 domain adapter")
    parser.add_argument("--snapshots", type=str, default=None, help="Snapshot directory")
    parser.add_argument("--check", action="store_true", help="Run synthetic validation")
    args = parser.parse_args()

    if args.check:
        _cli_check()
    elif args.snapshots:
        ds = RuViewOccDataset(args.snapshots)
        print(f"Loaded {len(ds)} windows from {args.snapshots}")
        if len(ds) > 0:
            s = ds[0]
            print(f"  target_occs: {s['target_occs'].shape}")
            print(f"  rel_poses:   {s['rel_poses'].shape}")
    else:
        parser.print_help()
