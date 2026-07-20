#!/usr/bin/env python3
"""Headless tests for the camera-room calibration pipeline (ADR-152 S2.1.3).

Covers calibration_lib.py end to end on synthetic data -- no camera, no
display, no MediaPipe:
  * known extrinsics recovered from synthetic two-checkerboard corners
  * calibration bundle JSON round-trip + stable content hash
  * image->room keypoint transform correctness (rays pass through the
    original 3D points -- the projective, no-depth alignment of ADR-079
    labels into the shared room frame)
  * collect-ground-truth's no-calibration record path is byte-identical
    (augment_record with ctx=None is the identity)

Run:  python -m pytest scripts/tests/ -q
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import calibration_lib as cal

# ---------------------------------------------------------------------------
# Synthetic scene fixtures
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 1280, 720
K_GT = np.array(
    [[800.0, 0.0, 640.0],
     [0.0, 800.0, 360.0],
     [0.0, 0.0, 1.0]]
)
DIST_ZERO = np.zeros(5)
DIST_MILD = np.array([-0.10, 0.02, 0.001, -0.001, 0.0])

BOARD_COLS, BOARD_ROWS = 9, 6
SQUARE_M = 0.025


def look_at_pose(camera_pos, target):
    """Ground-truth camera pose: returns (R_cam_to_room, camera_center_room).

    Camera convention: +z forward (optical axis), +x right, +y down.
    """
    c = np.asarray(camera_pos, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - c
    fwd /= np.linalg.norm(fwd)
    up_room = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(fwd, -up_room)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(fwd, x_cam)
    r_cam_to_room = np.stack([x_cam, y_cam, fwd], axis=1)  # columns = camera axes in room
    return r_cam_to_room, c


def room_to_cam(r_cam_to_room, center):
    """Invert to the solvePnP (room->camera) convention: rvec, tvec."""
    r_room_to_cam = r_cam_to_room.T
    tvec = -r_room_to_cam @ center
    rvec, _ = cv2.Rodrigues(r_room_to_cam)
    return rvec, tvec.reshape(3, 1)


def project_room_points(points_room, r_cam_to_room, center, k=K_GT, dist=DIST_ZERO):
    rvec, tvec = room_to_cam(r_cam_to_room, center)
    proj, _ = cv2.projectPoints(np.asarray(points_room, dtype=np.float64), rvec, tvec, k, dist)
    return proj.reshape(-1, 2)


@pytest.fixture
def scene():
    """A camera in the room looking at the wall + floor checkerboards."""
    r_gt, c_gt = look_at_pose(camera_pos=[1.5, 3.0, 1.3], target=[1.0, 0.5, 0.8])
    wall_room = cal.board_room_points(
        BOARD_COLS, BOARD_ROWS, SQUARE_M,
        origin=[0.5, 0.0, 1.6], u_axis=cal.parse_axis("+x"), v_axis=cal.parse_axis("-z"),
    )
    floor_room = cal.board_room_points(
        BOARD_COLS, BOARD_ROWS, SQUARE_M,
        origin=[1.0, 1.0, 0.0], u_axis=cal.parse_axis("+x"), v_axis=cal.parse_axis("+y"),
    )
    return r_gt, c_gt, wall_room, floor_room


def make_bundle(r_gt, c_gt, dist=DIST_ZERO):
    return cal.make_bundle(
        camera_intrinsics={
            "image_size": [IMG_W, IMG_H],
            "camera_matrix": K_GT.tolist(),
            "dist_coeffs": dist.tolist(),
            "reprojection_error_px": 0.0,
            "source": "synthetic",
        },
        camera_to_room_extrinsics={
            "rotation": r_gt.tolist(),
            "translation_m": c_gt.tolist(),
            "rmse_px": 0.0,
        },
        checkerboard_spec={"cols": BOARD_COLS, "rows": BOARD_ROWS, "square_size_mm": 25.0},
        transceiver_geometry={
            "nodes": [
                {"id": "esp32-s3-a", "position_m": [0.1, 2.4, 1.1], "antenna_yaw_deg": 180.0},
                {"id": "esp32-c6-b", "position_m": [3.2, 0.3, 0.9]},
            ],
            "units": "meters",
            "source": "file",
        },
    )


# ---------------------------------------------------------------------------
# Extrinsics recovery from synthetic checkerboard corners
# ---------------------------------------------------------------------------

class TestExtrinsicsRecovery:
    def test_two_board_combined_recovers_known_pose(self, scene):
        r_gt, c_gt, wall_room, floor_room = scene
        room_pts = np.concatenate([wall_room, floor_room], axis=0)
        img_pts = project_room_points(room_pts, r_gt, c_gt)

        ext = cal.solve_extrinsics(room_pts, img_pts, K_GT, DIST_ZERO)

        assert ext["rmse_px"] < 1e-3
        np.testing.assert_allclose(np.asarray(ext["translation_m"]), c_gt, atol=1e-4)
        r_delta = np.asarray(ext["rotation"]).T @ r_gt
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(r_delta) - 1) / 2, -1, 1)))
        assert angle_deg < 0.01

    def test_single_board_solves_agree(self, scene):
        # With correct corner ordering, each board alone recovers the same pose.
        r_gt, c_gt, wall_room, floor_room = scene
        ext_wall = cal.solve_extrinsics(
            wall_room, project_room_points(wall_room, r_gt, c_gt), K_GT, DIST_ZERO)
        ext_floor = cal.solve_extrinsics(
            floor_room, project_room_points(floor_room, r_gt, c_gt), K_GT, DIST_ZERO)
        consistency = cal.extrinsics_consistency(ext_wall, ext_floor)
        assert consistency["rotation_deg"] < 0.1
        assert consistency["translation_m"] < 1e-3

    def test_reversed_corner_order_auto_recovered(self, scene):
        # findChessboardCorners may enumerate from either board end. A single
        # board cannot disambiguate that flip (centrosymmetric grid), but the
        # joint two-board solve can -- feed it a reversed wall ordering and
        # require the true pose back.
        r_gt, c_gt, wall_room, floor_room = scene
        wall_img = project_room_points(wall_room, r_gt, c_gt)
        floor_img = project_room_points(floor_room, r_gt, c_gt)
        ext = cal.solve_two_board_extrinsics(
            wall_room, wall_img[::-1].copy(), floor_room, floor_img,
            K_GT, DIST_ZERO)
        assert ext["wall_flipped"] is True
        assert ext["floor_flipped"] is False
        assert ext["rmse_px"] < 1e-3
        np.testing.assert_allclose(np.asarray(ext["translation_m"]), c_gt, atol=1e-3)

    def test_joint_solver_matches_unflipped(self, scene):
        r_gt, c_gt, wall_room, floor_room = scene
        ext = cal.solve_two_board_extrinsics(
            wall_room, project_room_points(wall_room, r_gt, c_gt),
            floor_room, project_room_points(floor_room, r_gt, c_gt),
            K_GT, DIST_ZERO)
        assert ext["wall_flipped"] is False and ext["floor_flipped"] is False
        assert ext["per_board"]["wall"]["rmse_px"] < 1e-3
        assert ext["per_board"]["floor"]["rmse_px"] < 1e-3

    def test_intrinsics_recovered_from_synthetic_views(self):
        # Several board views from different poses -> calibrateCamera should
        # get focal length / principal point close to ground truth.
        obj = cal.board_object_points(BOARD_COLS, BOARD_ROWS, SQUARE_M)
        poses = [
            ([0.05, 1.2, 0.05], [0.10, 0.0, 0.06]),
            ([-0.25, 1.0, 0.20], [0.10, 0.0, 0.06]),
            ([0.45, 0.9, -0.15], [0.10, 0.0, 0.06]),
            ([0.10, 1.4, 0.30], [0.10, 0.0, 0.06]),
            ([-0.15, 0.8, -0.20], [0.10, 0.0, 0.06]),
        ]
        corner_sets = []
        for cam_pos, target in poses:
            r, c = look_at_pose(cam_pos, target)
            # Embed the board rigidly in the y=0 plane (u=+x, v=+z) and view it.
            board_in_room = np.column_stack([obj[:, 0], obj[:, 2], obj[:, 1]])
            corner_sets.append(project_room_points(board_in_room, r, c))
        intr = cal.compute_intrinsics(corner_sets, (IMG_W, IMG_H),
                                      BOARD_COLS, BOARD_ROWS, SQUARE_M)
        k = np.asarray(intr["camera_matrix"])
        assert abs(k[0, 0] - K_GT[0, 0]) / K_GT[0, 0] < 0.05
        assert abs(k[1, 1] - K_GT[1, 1]) / K_GT[1, 1] < 0.05
        assert intr["reprojection_error_px"] < 1.0


# ---------------------------------------------------------------------------
# Bundle round-trip + content hash
# ---------------------------------------------------------------------------

class TestBundle:
    def test_save_load_roundtrip(self, scene, tmp_path):
        r_gt, c_gt, _, _ = scene
        bundle = make_bundle(r_gt, c_gt)
        path = tmp_path / "camera-room.json"
        cal.save_bundle(bundle, path)
        loaded = cal.load_bundle(path)
        assert loaded == bundle
        assert cal.calibration_id(loaded) == cal.calibration_id(bundle)

    def test_bundle_schema_fields(self, scene):
        r_gt, c_gt, _, _ = scene
        bundle = make_bundle(r_gt, c_gt)
        for key in ("schema_version", "method", "calibrated_at", "room_frame",
                    "checkerboard_spec", "camera_intrinsics",
                    "camera_to_room_extrinsics", "transceiver_geometry"):
            assert key in bundle
        assert bundle["method"] == "two-checkerboard"

    def test_calibration_id_changes_with_content(self, scene):
        r_gt, c_gt, _, _ = scene
        bundle_a = make_bundle(r_gt, c_gt)
        bundle_b = json.loads(json.dumps(bundle_a))
        bundle_b["transceiver_geometry"]["nodes"][0]["position_m"] = [0.2, 2.4, 1.1]
        assert cal.calibration_id(bundle_a) != cal.calibration_id(bundle_b)
        assert cal.calibration_id(bundle_a).startswith("sha256:")

    def test_load_bundle_rejects_missing_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"camera_intrinsics": {}}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing key"):
            cal.load_bundle(path)


# ---------------------------------------------------------------------------
# Keypoint transform: image -> room-frame bearing rays (projective alignment)
# ---------------------------------------------------------------------------

class TestKeypointTransform:
    PERSON_POINTS = np.array([
        [1.2, 1.5, 1.7],   # head height
        [1.1, 1.5, 1.4],   # shoulder
        [1.3, 1.6, 0.9],   # hip
        [1.2, 1.5, 0.1],   # ankle
    ])

    @pytest.mark.parametrize("dist", [DIST_ZERO, DIST_MILD], ids=["no-distortion", "mild-distortion"])
    def test_rays_pass_through_original_points(self, scene, dist):
        r_gt, c_gt, _, _ = scene
        img = project_room_points(self.PERSON_POINTS, r_gt, c_gt, dist=dist)
        kps_norm = (img / np.array([IMG_W, IMG_H])).tolist()

        ctx = cal.CalibrationContext(make_bundle(r_gt, c_gt, dist=dist), IMG_W, IMG_H)
        origin, rays = ctx.transform_keypoints(kps_norm)

        np.testing.assert_allclose(origin, c_gt, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-9)
        for point, ray in zip(self.PERSON_POINTS, rays):
            v = point - origin
            # Distance from the true 3D point to the recovered ray ~ 0, and
            # the point sits in FRONT of the camera along the ray.
            dist_to_ray = np.linalg.norm(v - np.dot(v, ray) * ray)
            assert dist_to_ray < 1e-4
            assert np.dot(v, ray) > 0

    def test_resolution_scaling(self, scene):
        # Collection camera runs 640x360 while the bundle was made at
        # 1280x720 -- normalized keypoints must land on the same rays.
        r_gt, c_gt, _, _ = scene
        img = project_room_points(self.PERSON_POINTS, r_gt, c_gt)
        kps_norm = (img / np.array([IMG_W, IMG_H])).tolist()

        ctx = cal.CalibrationContext(make_bundle(r_gt, c_gt), 640, 360)
        origin, rays = ctx.transform_keypoints(kps_norm)
        for point, ray in zip(self.PERSON_POINTS, rays):
            v = point - origin
            assert np.linalg.norm(v - np.dot(v, ray) * ray) < 1e-4


# ---------------------------------------------------------------------------
# collect-ground-truth record path (import-level; no camera loop)
# ---------------------------------------------------------------------------

class TestRecordAugmentation:
    LEGACY_RECORD = {
        "ts_ns": 1775300000000000000,
        "keypoints": [[0.45, 0.12]] * 17,
        "confidence": 0.92,
        "n_visible": 14,
        "n_persons": 1,
    }

    def test_no_calibration_is_byte_identical(self):
        # The collector's no---calibration path must emit exactly the
        # original ADR-079 JSONL line (back-compat guarantee).
        record = json.loads(json.dumps(self.LEGACY_RECORD))
        before = json.dumps(record)
        out = cal.augment_record(record, None)
        assert out is record
        assert json.dumps(out) == before
        assert set(out.keys()) == {"ts_ns", "keypoints", "confidence",
                                   "n_visible", "n_persons"}

    def test_calibrated_record_gains_room_fields(self, scene):
        r_gt, c_gt, _, _ = scene
        bundle = make_bundle(r_gt, c_gt)
        ctx = cal.CalibrationContext(bundle, IMG_W, IMG_H)

        record = json.loads(json.dumps(self.LEGACY_RECORD))
        out = cal.augment_record(record, ctx)

        # Raw image coords preserved untouched; room representation added.
        assert out["keypoints"] == self.LEGACY_RECORD["keypoints"]
        assert len(out["keypoints_room"]) == 17
        assert all(len(ray) == 3 for ray in out["keypoints_room"])
        assert out["calibration_id"] == cal.calibration_id(bundle)
        assert out["transceiver_geometry"] == bundle["transceiver_geometry"]
        assert len(out["camera_origin_room"]) == 3
        json.dumps(out)  # remains JSONL-serializable

    def test_empty_keypoints_record(self, scene):
        r_gt, c_gt, _, _ = scene
        ctx = cal.CalibrationContext(make_bundle(r_gt, c_gt), IMG_W, IMG_H)
        record = {"ts_ns": 1, "keypoints": [], "confidence": 0.0,
                  "n_visible": 0, "n_persons": 0}
        out = cal.augment_record(record, ctx)
        assert out["keypoints_room"] == []
        assert "calibration_id" in out
