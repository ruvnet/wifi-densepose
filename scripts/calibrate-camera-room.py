#!/usr/bin/env python3
"""Two-checkerboard camera-room calibration for WiFi pose training (ADR-152 S2.1.3).

Aligns the ADR-079 ground-truth camera and the ESP32 WiFi transceivers in
one shared 3D room frame -- the PerceptAlign (arXiv 2601.12252) defense
against "coordinate overfitting", where CSI-to-camera-coordinate regression
memorizes the deployment layout and collapses cross-layout.

Procedure (<5 minutes):
  1. Print a checkerboard (default 9x6 inner corners, 25 mm squares).
  2. Tape one board flat on the ORIGIN WALL, tape-measure its top-left inner
     corner position in room coordinates (+x along wall, +y into room, +z up).
  3. Lay the second board flat on the FLOOR, measure its near-left inner corner.
  4. With the collection camera in its final position, photograph each board.
  5. Run this script; tape-measure each ESP32 node position when prompted
     (or pass --geometry nodes.json).

Output: a calibration bundle JSON consumed by
    scripts/collect-ground-truth.py --calibration <bundle.json>

Usage:
    python scripts/calibrate-camera-room.py \\
        --wall-image photos/wall.jpg --wall-origin 0.50,0.0,1.60 \\
        --floor-image photos/floor.jpg --floor-origin 1.00,1.00,0.0 \\
        --calib-images "photos/intrinsics/*.jpg" \\
        --geometry config/transceivers.json \\
        --output data/calibration/camera-room.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_lib as cal  # noqa: E402

INTRINSICS_CACHE = Path("data") / ".cache" / "camera_intrinsics.json"


def parse_vec3(text: str) -> np.ndarray:
    parts = [float(p) for p in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated numbers, got {text!r}")
    return np.array(parts, dtype=np.float64)


def detect_corners(image_path: Path, cols: int, rows: int) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Cannot read image {image_path}", file=sys.stderr)
        sys.exit(1)
    corners = cal.find_board_corners(image, cols, rows)
    if corners is None:
        print(
            f"ERROR: No {cols}x{rows} checkerboard found in {image_path}. "
            "Check lighting, focus, and the --board-cols/--board-rows flags.",
            file=sys.stderr,
        )
        sys.exit(1)
    h, w = image.shape[:2]
    return corners, (w, h)


def resolve_intrinsics(args, repo_root: Path, board_args: tuple[int, int, float]) -> dict:
    """Pre-computed file > cached > computed from --calib-images >
    last-resort 2-view estimate from the wall+floor photos themselves."""
    cols, rows, square_m = board_args

    if args.intrinsics:
        print(f"Intrinsics: loading {args.intrinsics}")
        return cal.load_intrinsics(Path(args.intrinsics))

    cache_path = repo_root / INTRINSICS_CACHE
    if cache_path.exists() and not args.recalibrate_intrinsics:
        print(f"Intrinsics: using cached {cache_path} (pass --recalibrate-intrinsics to redo)")
        intr = cal.load_intrinsics(cache_path)
        intr["source"] = "cached"
        return intr

    if args.calib_images:
        paths = sorted(glob.glob(args.calib_images))
        if len(paths) < 3:
            print(
                f"ERROR: --calib-images matched only {len(paths)} file(s); "
                "need >= 3 checkerboard views for stable intrinsics.",
                file=sys.stderr,
            )
            sys.exit(1)
        corner_sets, image_size = [], None
        for p in paths:
            corners, size = detect_corners(Path(p), cols, rows)
            if image_size is None:
                image_size = size
            elif size != image_size:
                print(f"ERROR: {p} has size {size}, expected {image_size}.", file=sys.stderr)
                sys.exit(1)
            corner_sets.append(corners)
            print(f"  corners found: {p}")
        intr = cal.compute_intrinsics(corner_sets, image_size, cols, rows, square_m)
        print(f"Intrinsics: computed from {len(paths)} views, "
              f"reprojection RMS {intr['reprojection_error_px']:.3f} px")
        cal.save_bundle(intr, cache_path)  # plain JSON write; reused on next run
        print(f"  cached to {cache_path}")
        return intr

    # Last resort: 2-view calibration from the extrinsic photos. Workable but
    # weak -- warn loudly and recommend a proper multi-view pass.
    print(
        "WARNING: no --intrinsics / cache / --calib-images; estimating intrinsics "
        "from the wall+floor photos alone (2 views, low quality). Prefer "
        "--calib-images with 5-10 varied board views.",
        file=sys.stderr,
    )
    corner_sets, image_size = [], None
    for p in (args.wall_image, args.floor_image):
        corners, size = detect_corners(Path(p), cols, rows)
        image_size = image_size or size
        corner_sets.append(corners)
    intr = cal.compute_intrinsics(corner_sets, image_size, cols, rows, square_m)
    intr["source"] = "two-view-fallback"
    return intr


def prompt_transceiver_geometry() -> dict:
    """Tape-measure entry of ESP32 node positions in room coordinates."""
    print()
    print("Transceiver geometry -- enter one node per line:")
    print("  <node-id> <x> <y> <z> [yaw_deg]     (meters, room frame; blank line to finish)")
    print("  example:  esp32-s3-a 0.10 2.40 1.10 180")
    nodes = []
    while True:
        try:
            line = input("node> ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.split()
        if len(parts) not in (4, 5):
            print("  expected: <node-id> <x> <y> <z> [yaw_deg]", file=sys.stderr)
            continue
        try:
            node = {"id": parts[0], "position_m": [float(parts[1]), float(parts[2]), float(parts[3])]}
            if len(parts) == 5:
                node["antenna_yaw_deg"] = float(parts[4])
        except ValueError:
            print("  positions must be numeric", file=sys.stderr)
            continue
        nodes.append(node)
    if not nodes:
        print("WARNING: no transceiver nodes entered; bundle will carry empty geometry.",
              file=sys.stderr)
    return {"nodes": nodes, "units": "meters", "source": "tape-measure-prompt"}


def load_geometry_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", data if isinstance(data, list) else None)
    if nodes is None:
        raise ValueError(f"{path}: expected {{'nodes': [...]}} or a top-level list")
    for node in nodes:
        if "id" not in node or "position_m" not in node:
            raise ValueError(f"{path}: each node needs 'id' and 'position_m' [x,y,z]")
    return {"nodes": nodes, "units": "meters", "source": "file"}


def main():
    parser = argparse.ArgumentParser(
        description="Two-checkerboard camera-room calibration (ADR-152 S2.1.3 / ADR-079)."
    )
    parser.add_argument("--wall-image", required=True,
                        help="Photo of the checkerboard on the origin wall")
    parser.add_argument("--floor-image", required=True,
                        help="Photo of the checkerboard on the floor (camera NOT moved)")
    parser.add_argument("--wall-origin", type=parse_vec3, default="0.5,0.0,1.6",
                        help="Room xyz (m) of the wall board's first inner corner "
                             "(default: 0.5,0.0,1.6)")
    parser.add_argument("--floor-origin", type=parse_vec3, default="1.0,1.0,0.0",
                        help="Room xyz (m) of the floor board's first inner corner "
                             "(default: 1.0,1.0,0.0)")
    parser.add_argument("--wall-axes", default="+x,-z",
                        help="Wall board column,row directions in room frame (default: +x,-z)")
    parser.add_argument("--floor-axes", default="+x,+y",
                        help="Floor board column,row directions in room frame (default: +x,+y)")
    parser.add_argument("--board-cols", type=int, default=cal.DEFAULT_BOARD_COLS,
                        help=f"Inner corners per row (default: {cal.DEFAULT_BOARD_COLS})")
    parser.add_argument("--board-rows", type=int, default=cal.DEFAULT_BOARD_ROWS,
                        help=f"Inner corners per column (default: {cal.DEFAULT_BOARD_ROWS})")
    parser.add_argument("--square-size-mm", type=float, default=cal.DEFAULT_SQUARE_SIZE_MM,
                        help=f"Checkerboard square size in mm (default: {cal.DEFAULT_SQUARE_SIZE_MM})")
    parser.add_argument("--intrinsics", help="Pre-computed intrinsics JSON (skips computation)")
    parser.add_argument("--calib-images",
                        help="Glob of >=3 checkerboard photos for intrinsics computation")
    parser.add_argument("--recalibrate-intrinsics", action="store_true",
                        help="Ignore the cached intrinsics and recompute")
    parser.add_argument("--geometry",
                        help="Transceiver geometry JSON ({nodes:[{id,position_m,[antenna_yaw_deg]}]}); "
                             "omit to be prompted for tape-measure entry")
    parser.add_argument("--output", default=None,
                        help="Bundle output path (default: data/calibration/camera-room-<ts>.json)")
    args = parser.parse_args()

    if isinstance(args.wall_origin, str):
        args.wall_origin = parse_vec3(args.wall_origin)
    if isinstance(args.floor_origin, str):
        args.floor_origin = parse_vec3(args.floor_origin)

    repo_root = Path(__file__).resolve().parent.parent
    cols, rows = args.board_cols, args.board_rows
    square_m = args.square_size_mm / 1000.0

    # --- Intrinsics ---
    intrinsics = resolve_intrinsics(args, repo_root, (cols, rows, square_m))
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64)

    # --- Corner detection on the two placed boards ---
    wall_corners, wall_size = detect_corners(Path(args.wall_image), cols, rows)
    floor_corners, floor_size = detect_corners(Path(args.floor_image), cols, rows)
    if wall_size != floor_size:
        print(f"ERROR: wall image {wall_size} and floor image {floor_size} differ in size; "
              "both must come from the fixed collection camera.", file=sys.stderr)
        sys.exit(1)
    print(f"Corners detected: wall + floor boards ({cols}x{rows}, {args.square_size_mm} mm)")

    # Re-scale intrinsics if they were computed at a different resolution
    # than the extrinsic photos (the bundle always stores K at wall_size).
    intr_size = tuple(intrinsics["image_size"])
    if intr_size != wall_size:
        sx, sy = wall_size[0] / intr_size[0], wall_size[1] / intr_size[1]
        camera_matrix[0, 0] *= sx
        camera_matrix[0, 2] *= sx
        camera_matrix[1, 1] *= sy
        camera_matrix[1, 2] *= sy
        print(f"  intrinsics scaled {intr_size} -> {wall_size}")
    intrinsics = {**intrinsics, "camera_matrix": camera_matrix.tolist(),
                  "image_size": list(wall_size)}

    # --- Room-frame corner positions from the measured placements ---
    wall_u, wall_v = (cal.parse_axis(t) for t in args.wall_axes.split(","))
    floor_u, floor_v = (cal.parse_axis(t) for t in args.floor_axes.split(","))
    wall_room = cal.board_room_points(cols, rows, square_m, args.wall_origin, wall_u, wall_v)
    floor_room = cal.board_room_points(cols, rows, square_m, args.floor_origin, floor_u, floor_v)

    # --- Extrinsics: joint two-board solve (resolves per-board corner-order
    # ambiguity -- a single planar board is centrosymmetric; the pair is not) ---
    extrinsics = cal.solve_two_board_extrinsics(
        wall_room, wall_corners, floor_room, floor_corners, camera_matrix, dist_coeffs
    )
    wall_rmse = extrinsics["per_board"]["wall"]["rmse_px"]
    floor_rmse = extrinsics["per_board"]["floor"]["rmse_px"]
    print(f"  joint solve: RMSE {extrinsics['rmse_px']:.3f} px "
          f"(wall {wall_rmse:.3f} / floor {floor_rmse:.3f})")
    print(f"  camera at room {np.round(extrinsics['translation_m'], 3).tolist()} m")
    if max(wall_rmse, floor_rmse) > 3.0:
        print(
            "WARNING: high per-board reprojection error -- re-check the measured "
            "board origins/axes and that the camera did not move between photos.",
            file=sys.stderr,
        )

    # --- Transceiver geometry ---
    if args.geometry:
        geometry = load_geometry_file(Path(args.geometry))
        print(f"Transceiver geometry: {len(geometry['nodes'])} node(s) from {args.geometry}")
    else:
        geometry = prompt_transceiver_geometry()

    # --- Bundle ---
    bundle = cal.make_bundle(
        camera_intrinsics=intrinsics,
        camera_to_room_extrinsics=extrinsics,
        checkerboard_spec={"cols": cols, "rows": rows, "square_size_mm": args.square_size_mm},
        transceiver_geometry=geometry,
    )
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = repo_root / "data" / "calibration" / f"camera-room-{ts}.json"
    cal.save_bundle(bundle, out_path)

    print()
    print("=== Calibration bundle written ===")
    print(f"  path:           {out_path}")
    print(f"  calibration_id: {cal.calibration_id(bundle)}")
    print(f"  next: python scripts/collect-ground-truth.py --calibration {out_path}")


if __name__ == "__main__":
    main()
