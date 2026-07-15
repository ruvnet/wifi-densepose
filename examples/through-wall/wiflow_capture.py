#!/usr/bin/env python3
"""WiFlow-style camera-supervised capture (ADR-079 / ADR-180).

Runs on a box with BOTH a camera (ground truth) and reachable live CSI:
  - opens a camera, runs MediaPipe Pose -> 17 COCO keypoints (the LABEL),
  - subscribes to the sensing-server /ws/sensing (the INPUT: CSI features +
    20x20 signal-field),
  - writes timestamp-aligned (csi -> pose) pairs to a JSONL dataset.

This is the *collect* phase of camera-supervised CSI->pose training. The camera
and the CSI nodes MUST see the same person in the same space at the same time,
or the pairs are meaningless. Honest by construction: we only emit a pair when
BOTH a confident camera pose AND a live (source=esp32) CSI frame are present in
the same ~100 ms window.

Usage (on ruvultra, with the CSI tunneled to localhost:8765):
    python3 wiflow_capture.py --ws ws://localhost:8765/ws/sensing \
        --cam 0 --out ~/wiflow-room/dataset.jsonl --seconds 180
"""
import argparse, asyncio, json, time, threading, sys, os
from collections import deque

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import websockets

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

def ensure_model(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[capture] downloading pose model -> {path}", flush=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path

# MediaPipe Pose (33 landmarks) -> 17 COCO keypoints (same mapping as
# scripts/collect-ground-truth.py, ADR-079).
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COCO_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_sho","r_sho","l_elb",
              "r_elb","l_wri","r_wri","l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]

# ---- shared state between the CSI (async) thread and the camera (sync) loop ----
_latest_csi = {"t": 0.0, "frame": None}
_csi_lock = threading.Lock()
_stop = threading.Event()


def csi_thread(ws_url: str):
    """Background thread: keep the most recent LIVE csi frame in _latest_csi."""
    async def run():
        while not _stop.is_set():
            try:
                async with websockets.connect(ws_url, open_timeout=8, ping_interval=20) as ws:
                    while not _stop.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=8)
                        d = json.loads(msg)
                        with _csi_lock:
                            _latest_csi["t"] = time.time()
                            _latest_csi["frame"] = d
            except Exception as e:
                print(f"[csi] reconnect ({e})", flush=True)
                await asyncio.sleep(1.0)
    asyncio.new_event_loop().run_until_complete(run())


def csi_vector(frame: dict):
    """Flatten a csi frame to a fixed-length input vector: features + field."""
    f = frame.get("features", {}) or {}
    feats = [f.get("mean_rssi", 0.0), f.get("variance", 0.0),
             f.get("motion_band_power", 0.0), f.get("breathing_band_power", 0.0)]
    # per-node mean_rssi/variance/motion for up to the 2 nodes (9, 13)
    pernode = {nf.get("node_id"): (nf.get("features") or {}) for nf in (frame.get("node_features") or [])}
    for nid in (9, 13):
        nf = pernode.get(nid, {})
        feats += [nf.get("mean_rssi", 0.0), nf.get("variance", 0.0), nf.get("motion_band_power", 0.0)]
    field = (frame.get("signal_field", {}) or {}).get("values") or []
    field = (field + [0.0] * 400)[:400]
    return feats + field   # 4 + 6 + 400 = 410-d


def main():
    ap = argparse.ArgumentParser(description="WiFlow camera-supervised CSI<->pose capture (ADR-180).")
    ap.add_argument("--ws", default="ws://localhost:8765/ws/sensing")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/wiflow-room/dataset.jsonl"))
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--min-vis", type=float, default=0.5, help="min mean landmark visibility to accept a pose label")
    ap.add_argument("--max-skew-ms", type=float, default=150, help="max csi/pose time skew to pair")
    ap.add_argument("--require-esp32", action="store_true", default=True,
                    help="only pair when csi source==esp32 (real). Default on.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    th = threading.Thread(target=csi_thread, args=(args.ws,), daemon=True)
    th.start()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.cam}", file=sys.stderr); sys.exit(2)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    model_path = ensure_model(os.path.expanduser("~/wiflow-room/pose_landmarker_lite.task"))
    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE, min_pose_detection_confidence=0.5))

    n_pairs = 0; n_nopose = 0; n_nocsi = 0; n_skew = 0; n_sim = 0
    t0 = time.time()
    print(f"[capture] camera {args.cam} {W}x{H} -> {args.out} for {args.seconds}s")
    print("[capture] stand in view AND in the CSI field; move/walk so poses vary. Ctrl-C to stop.")
    with open(args.out, "a") as out:
        try:
            while time.time() - t0 < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not res.pose_landmarks:
                    n_nopose += 1; continue
                lm = res.pose_landmarks[0]
                kps = [[lm[i].x, lm[i].y, lm[i].visibility] for i in COCO_FROM_MP]
                vis = float(np.mean([k[2] for k in kps]))
                if vis < args.min_vis:
                    n_nopose += 1; continue
                with _csi_lock:
                    ct = _latest_csi["t"]; cf = _latest_csi["frame"]
                if cf is None:
                    n_nocsi += 1; continue
                if (now - ct) * 1000.0 > args.max_skew_ms:
                    n_skew += 1; continue
                if args.require_esp32 and cf.get("source") != "esp32":
                    n_sim += 1; continue
                rec = {"t": now, "vis": round(vis, 3),
                       "kps": [[round(x, 4), round(y, 4), round(v, 3)] for x, y, v in kps],
                       "csi": csi_vector(cf),
                       "src": cf.get("source"),
                       "nodes": sorted(n.get("node_id") for n in cf.get("nodes", []) if n.get("node_id") is not None)}
                out.write(json.dumps(rec) + "\n")
                n_pairs += 1
                if n_pairs % 30 == 0:
                    out.flush()
                    el = int(now - t0)
                    print(f"[capture] t+{el:3d}s pairs={n_pairs} (skip: nopose={n_nopose} nocsi={n_nocsi} skew={n_skew} sim={n_sim})", flush=True)
        except KeyboardInterrupt:
            print("\n[capture] stopped by user")
    _stop.set(); cap.release()
    print(f"[capture] DONE. wrote {n_pairs} paired samples to {args.out}")
    print(f"[capture] skipped: no-pose={n_nopose} no-csi={n_nocsi} skew={n_skew} simulated={n_sim}")
    if n_pairs == 0:
        print("[capture] WARNING: 0 pairs — check camera sees you AND csi source==esp32 (live).")


if __name__ == "__main__":
    main()
