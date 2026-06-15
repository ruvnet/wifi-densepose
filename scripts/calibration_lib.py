#!/usr/bin/env python3
"""Camera-room calibration library for WiFi pose ground truth (ADR-152 S2.1.3).

Implements the PerceptAlign-style two-checkerboard alignment adopted in
ADR-152 S2.1.3 to defend the ADR-079 camera-supervised pipeline against
"coordinate overfitting" (arXiv 2601.12252, MobiCom'26): models regressing
CSI to raw camera-frame coordinates memorize the deployment layout and
collapse cross-layout. The fix is to express camera AND WiFi transceivers
in one shared 3D room frame, and stamp every training label with the
calibration + transceiver geometry that produced it.

Used by:
    scripts/calibrate-camera-room.py   (produces the calibration bundle)
    scripts/collect-ground-truth.py    (consumes it via --calibration)

Room frame convention (right-handed, meters):
    origin = a designated wall/floor corner of the room
    +x     = along the origin wall
    +y     = into the room (away from the origin wall)
    +z     = up

No-depth limitation (IMPORTANT): a single 2D camera keypoint constrains
only a *ray* in the room frame, not a 3D point. The transform helpers here
therefore return unit bearing rays from the camera center -- a projective
alignment. Consumers that need metric 3D points must supply a depth
assumption downstream (floor-plane intersection, known subject height,
multi-view triangulation, ...). Raw image coordinates are always preserved
alongside the room-frame rays so training can choose either representation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_METHOD = "two-checkerboard"

# Default checkerboard: 9x6 inner corners, 25 mm squares (a common print).
DEFAULT_BOARD_COLS = 9
DEFAULT_BOARD_ROWS = 6
DEFAULT_SQUARE_SIZE_MM = 25.0

_AXIS_TOKENS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def parse_axis(token: str) -> np.ndarray:
    """Parse an axis token like '+x' or '-z' into a room-frame unit vector."""
    key = token.strip().lower()
    if key in _AXIS_TOKENS:
        return np.array(_AXIS_TOKENS[key], dtype=np.float64)
    raise ValueError(f"Invalid axis token {token!r}; expected one of {sorted(_AXIS_TOKENS)}")


# ---------------------------------------------------------------------------
# Checkerboard geometry
# ---------------------------------------------------------------------------

def board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Inner-corner positions in the board's own frame (z=0 plane), row-major.

    Matches the corner ordering of cv2.findChessboardCorners for a
    (cols, rows) pattern: cols varies fastest.
    """
    pts = np.zeros((rows * cols, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # (rows*cols, 2), cols fastest
    pts[:, :2] = grid * square_size_m
    return pts


def board_room_points(
    cols: int,
    rows: int,
    square_size_m: float,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    """Inner-corner positions in ROOM coordinates for a board placed at a
    known position: first corner at `origin`, columns stepping along
    `u_axis`, rows stepping along `v_axis` (both room-frame unit vectors).
    """
    local = board_object_points(cols, rows, square_size_m)
    origin = np.asarray(origin, dtype=np.float64)
    u = np.asarray(u_axis, dtype=np.float64)
    v = np.asarray(v_axis, dtype=np.float64)
    return origin[None, :] + local[:, 0:1] * u[None, :] + local[:, 1:2] * v[None, :]


def find_board_corners(image: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Detect and sub-pixel-refine checkerboard inner corners.

    Returns (cols*rows, 2) float64 pixel coordinates, or None if not found.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def compute_intrinsics(
    corner_sets: list[np.ndarray],
    image_size: tuple[int, int],
    cols: int,
    rows: int,
    square_size_m: float,
) -> dict:
    """Camera intrinsics from N checkerboard views via cv2.calibrateCamera.

    corner_sets: list of (cols*rows, 2) pixel corner arrays.
    image_size:  (width, height) of the calibration images.
    """
    obj = board_object_points(cols, rows, square_size_m).astype(np.float32)
    obj_pts = [obj for _ in corner_sets]
    img_pts = [c.reshape(-1, 1, 2).astype(np.float32) for c in corner_sets]
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, tuple(image_size), None, None
    )
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "reprojection_error_px": float(rms),
        "source": "computed",
    }


def load_intrinsics(path: Path) -> dict:
    """Load a pre-computed intrinsics JSON ({camera_matrix, dist_coeffs, image_size})."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either a bare intrinsics dict or a full calibration bundle.
    intr = data.get("camera_intrinsics", data)
    for key in ("camera_matrix", "dist_coeffs", "image_size"):
        if key not in intr:
            raise ValueError(f"Intrinsics file {path} missing key {key!r}")
    intr = dict(intr)
    intr["source"] = "file"
    return intr


# ---------------------------------------------------------------------------
# Extrinsics (camera -> room rigid transform)
# ---------------------------------------------------------------------------

def reprojection_rmse(
    room_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(room_points, rvec, tvec, camera_matrix, dist_coeffs)
    err = proj.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def _solve_pnp(
    room_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict | None:
    """One solvePnP run (room->camera), inverted to camera->room. Returns
    {rotation (3x3 camera->room), translation_m (camera center in room
    frame), rmse_px} or None on failure.
    """
    ok, rvec, tvec = cv2.solvePnP(
        room_points.reshape(-1, 1, 3),
        image_points.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmse = reprojection_rmse(room_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    r_room_to_cam, _ = cv2.Rodrigues(rvec)
    r_cam_to_room = r_room_to_cam.T
    camera_center_room = (-r_cam_to_room @ tvec).ravel()
    return {
        "rotation": r_cam_to_room.tolist(),
        "translation_m": camera_center_room.tolist(),
        "rmse_px": rmse,
    }


def solve_extrinsics(
    room_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    """Solve the camera->room rigid transform from 3D room-frame points and
    their 2D pixel observations.

    NOTE: the corner grid of a single planar checkerboard is centrosymmetric,
    so the corner ordering returned by findChessboardCorners (which may
    enumerate from either board end) cannot be disambiguated from one board
    alone -- the reversed ordering fits a ghost pose with identical
    reprojection error. Use solve_two_board_extrinsics for the full
    two-checkerboard procedure, where the joint point set breaks the symmetry.
    """
    ext = _solve_pnp(room_points, image_points, camera_matrix, dist_coeffs)
    if ext is None:
        raise RuntimeError("solvePnP failed")
    return ext


def solve_two_board_extrinsics(
    wall_room: np.ndarray,
    wall_image: np.ndarray,
    floor_room: np.ndarray,
    floor_image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    """Joint camera->room solve over both checkerboards (the ADR-152 S2.1.3
    two-checkerboard method).

    Tries all 4 per-board corner-ordering combinations: each board's ordering
    is individually ambiguous (centrosymmetric grid), but the combined
    wall+floor point set is not, so exactly one combination reaches minimal
    reprojection error. Returns the solve_extrinsics dict plus
    {wall_flipped, floor_flipped, per_board: {wall|floor: {rmse_px}}}.
    """
    best = None
    for wall_flipped in (False, True):
        for floor_flipped in (False, True):
            wi = wall_image[::-1].copy() if wall_flipped else wall_image
            fi = floor_image[::-1].copy() if floor_flipped else floor_image
            room = np.concatenate([wall_room, floor_room], axis=0)
            img = np.concatenate([wi, fi], axis=0)
            ext = _solve_pnp(room, img, camera_matrix, dist_coeffs)
            if ext is None:
                continue
            if best is None or ext["rmse_px"] < best[0]["rmse_px"]:
                ext["wall_flipped"] = wall_flipped
                ext["floor_flipped"] = floor_flipped
                rvec, _ = cv2.Rodrigues(np.asarray(ext["rotation"]).T)
                tvec = -np.asarray(ext["rotation"]).T @ np.asarray(ext["translation_m"])
                ext["per_board"] = {
                    "wall": {"rmse_px": reprojection_rmse(
                        wall_room, wi, rvec, tvec, camera_matrix, dist_coeffs)},
                    "floor": {"rmse_px": reprojection_rmse(
                        floor_room, fi, rvec, tvec, camera_matrix, dist_coeffs)},
                }
                best = (ext,)
    if best is None:
        raise RuntimeError("solvePnP failed for all corner-ordering combinations")
    return best[0]


def extrinsics_consistency(ext_a: dict, ext_b: dict) -> dict:
    """Angular + translational disagreement between two extrinsic solutions
    (the two single-board solves). Large values mean a mis-entered board
    placement or a bad corner detection.
    """
    ra = np.asarray(ext_a["rotation"])
    rb = np.asarray(ext_b["rotation"])
    r_delta = ra.T @ rb
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0))))
    t_delta = float(
        np.linalg.norm(np.asarray(ext_a["translation_m"]) - np.asarray(ext_b["translation_m"]))
    )
    return {"rotation_deg": angle, "translation_m": t_delta}


# ---------------------------------------------------------------------------
# Calibration bundle (the artifact written to disk)
# ---------------------------------------------------------------------------

def make_bundle(
    camera_intrinsics: dict,
    camera_to_room_extrinsics: dict,
    checkerboard_spec: dict,
    transceiver_geometry: dict,
) -> dict:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "method": BUNDLE_METHOD,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "room_frame": {
            "description": "right-handed; origin at wall/floor corner; "
            "+x along origin wall, +y into room, +z up",
            "units": "meters",
        },
        "checkerboard_spec": checkerboard_spec,
        "camera_intrinsics": camera_intrinsics,
        "camera_to_room_extrinsics": camera_to_room_extrinsics,
        "transceiver_geometry": transceiver_geometry,
    }


def calibration_id(bundle: dict) -> str:
    """Stable content hash of a bundle -- stamped onto every emitted sample
    so a label can always be traced to the exact calibration that framed it.
    """
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_bundle(bundle: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
        f.write("\n")


def load_bundle(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    for key in ("camera_intrinsics", "camera_to_room_extrinsics", "transceiver_geometry"):
        if key not in bundle:
            raise ValueError(f"Calibration bundle {path} missing key {key!r}")
    return bundle


# ---------------------------------------------------------------------------
# Keypoint transform (image -> room-frame bearing rays)
# ---------------------------------------------------------------------------

class CalibrationContext:
    """Pre-computed transform state for a collection session.

    Scales the bundle's intrinsics to the live capture resolution (MediaPipe
    keypoints are normalized [0,1], so we need the actual frame size to get
    back to pixels before undistorting).
    """

    def __init__(self, bundle: dict, frame_w: int, frame_h: int):
        self.bundle = bundle
        self.calibration_id = calibration_id(bundle)
        self.transceiver_geometry = bundle["transceiver_geometry"]
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)

        intr = bundle["camera_intrinsics"]
        k = np.asarray(intr["camera_matrix"], dtype=np.float64)
        cal_w, cal_h = intr["image_size"]
        sx = self.frame_w / float(cal_w)
        sy = self.frame_h / float(cal_h)
        k = k.copy()
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        self.camera_matrix = k
        self.dist_coeffs = np.asarray(intr["dist_coeffs"], dtype=np.float64)

        ext = bundle["camera_to_room_extrinsics"]
        self.r_cam_to_room = np.asarray(ext["rotation"], dtype=np.float64)
        self.origin_room = np.asarray(ext["translation_m"], dtype=np.float64)

    def transform_keypoints(self, keypoints_norm: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """Normalized [0,1] image keypoints -> unit bearing rays in the room
        frame, anchored at the camera center.

        Projective alignment ONLY (no depth): each returned ray is the locus
        of room positions consistent with the 2D observation. Returns
        (camera_origin_room (3,), ray_dirs (N, 3) unit vectors).
        """
        pts = np.asarray(keypoints_norm, dtype=np.float64)
        pts_px = pts * np.array([self.frame_w, self.frame_h], dtype=np.float64)
        undist = cv2.undistortPoints(
            pts_px.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs
        ).reshape(-1, 2)
        rays_cam = np.concatenate([undist, np.ones((len(undist), 1))], axis=1)
        rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
        rays_room = (self.r_cam_to_room @ rays_cam.T).T
        return self.origin_room, rays_room


def load_calibration_context(path: Path, frame_w: int, frame_h: int) -> CalibrationContext:
    return CalibrationContext(load_bundle(path), frame_w, frame_h)


def augment_record(record: dict, ctx: CalibrationContext | None) -> dict:
    """Stamp a ground-truth record with room-frame rays + calibration metadata.

    With ctx=None this is the identity -- the record (and hence the emitted
    JSONL line) is byte-identical to the pre-calibration ADR-079 format.
    Raw image-coordinate keypoints are kept untouched in both cases; the
    room-frame representation is ADDED, never substituted, so training can
    choose either (ADR-152 S2.1.3).
    """
    if ctx is None:
        return record
    if record.get("keypoints"):
        _, rays = ctx.transform_keypoints(record["keypoints"])
        record["keypoints_room"] = [[round(float(v), 5) for v in ray] for ray in rays]
    else:
        record["keypoints_room"] = []
    record["camera_origin_room"] = [round(float(v), 5) for v in ctx.origin_room]
    record["calibration_id"] = ctx.calibration_id
    record["transceiver_geometry"] = ctx.transceiver_geometry
    return record
