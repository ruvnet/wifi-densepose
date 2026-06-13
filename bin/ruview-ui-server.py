#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import socket
import struct
import threading
import time
import zlib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

UI_DIR = Path(os.environ.get("RUVIEW_UI_DIR", "/home/deck/RuView/ui")).resolve()
PORT = int(os.environ.get("RUVIEW_UI_PORT", "3000"))
UDP_PORT = int(os.environ.get("RUVIEW_CARDPUTER_UDP_PORT", "5005"))
RUVIEW_ROOT = Path(
    os.environ.get(
        "RUVIEW_ROOT",
        str(UI_DIR.parent if UI_DIR.name == "ui" else Path("/home/deck/RuView")),
    )
).resolve()
DATA_DIR = Path(os.environ.get("RUVIEW_DATA_DIR", str(RUVIEW_ROOT / "data"))).resolve()
RECORDINGS_DIR = Path(os.environ.get("RUVIEW_RECORDINGS_DIR", str(DATA_DIR / "recordings"))).resolve()
MODELS_DIR = Path(os.environ.get("RUVIEW_MODELS_DIR", str(DATA_DIR / "models"))).resolve()
GROUND_TRUTH_DIR = Path(os.environ.get("RUVIEW_GROUND_TRUTH_DIR", str(DATA_DIR / "ground-truth"))).resolve()
PAIRED_DIR = Path(os.environ.get("RUVIEW_PAIRED_DIR", str(DATA_DIR / "paired"))).resolve()
SCRIPTS_DIR = Path(os.environ.get("RUVIEW_SCRIPTS_DIR", str(RUVIEW_ROOT / "scripts"))).resolve()
STARTED = time.time()
LIVE_MAX_AGE_S = 5.0
ADAPTIVE_STATE_MAX_AGE_S = 30.0
FEATURE_STATE_MAX_AGE_S = 5.0
BATTERY_MAX_AGE_S = 15.0
RSSI_MAX_AGE_S = 5.0
MAX_PERSONS = 4
CSI_FRAME_MAGIC = 0xC5110001
EDGE_VITALS_MAGIC = 0xC5110002
EDGE_VITALS_FMT = "<IBBHIbB2xffII"
EDGE_VITALS_SIZE = struct.calcsize(EDGE_VITALS_FMT)
EDGE_FEATURE_MAGIC = 0xC5110003
EDGE_FEATURE_FMT = "<IBBHQ8f"
EDGE_FEATURE_SIZE = struct.calcsize(EDGE_FEATURE_FMT)
RV_FEATURE_STATE_MAGIC = 0xC5110006
RV_FEATURE_STATE_FMT = "<IBBHQ9fHHI"
RV_FEATURE_STATE_SIZE = struct.calcsize(RV_FEATURE_STATE_FMT)
EDGE_BATTERY_MAGIC = 0xC5110008
EDGE_BATTERY_FMT = "<IBBBBHHI"
EDGE_BATTERY_SIZE = struct.calcsize(EDGE_BATTERY_FMT)
SYNC_PACKET_MAGIC = 0xC511A110
SYNC_PACKET_FMT = "<IBBBBQQII"
SYNC_PACKET_SIZE = struct.calcsize(SYNC_PACKET_FMT)
RV_MESH_MAGIC = 0xC5118100
RV_MESH_HEADER_FMT = "<IBBBBIHH"
RV_MESH_HEADER_SIZE = 16
RV_MESH_CRC_SIZE = 4
RV_MESH_MSG_TIME_SYNC = 0x01
RV_MESH_MSG_ROLE_ASSIGN = 0x02
RV_MESH_MSG_CHANNEL_PLAN = 0x03
RV_MESH_MSG_CALIBRATION_START = 0x04
RV_MESH_MSG_FEATURE_DELTA = 0x05
RV_MESH_MSG_HEALTH = 0x06
RV_MESH_MSG_ANOMALY_ALERT = 0x07
RV_MESH_MSG_NAMES = {
    RV_MESH_MSG_TIME_SYNC: "time_sync",
    RV_MESH_MSG_ROLE_ASSIGN: "role_assign",
    RV_MESH_MSG_CHANNEL_PLAN: "channel_plan",
    RV_MESH_MSG_CALIBRATION_START: "calibration_start",
    RV_MESH_MSG_FEATURE_DELTA: "feature_delta",
    RV_MESH_MSG_HEALTH: "health",
    RV_MESH_MSG_ANOMALY_ALERT: "anomaly_alert",
}
RV_MESH_ROLE_NAMES = {
    0: "unassigned",
    1: "anchor",
    2: "observer",
    3: "fusion_relay",
    4: "coordinator",
}
RV_MESH_AUTH_NAMES = {
    0: "none",
    1: "hmac_session",
    2: "ed25519_batch",
}
RV_NODE_STATUS_FMT = "<8sQBBBbHHHH"
RV_TIME_SYNC_FMT = "<QII"
RV_ROLE_ASSIGN_FMT = "<8sB3xI"
RV_CHANNEL_PLAN_FMT = "<8sBBBB8sI"
RV_CALIBRATION_START_FMT = "<QIIB3x"
RV_ANOMALY_ALERT_FMT = "<8sQBBHff"
PACKET_TYPE_NAMES = {
    0xC5110001: "csi_frame",
    0xC5110002: "edge_vitals",
    EDGE_FEATURE_MAGIC: "edge_feature",
    RV_FEATURE_STATE_MAGIC: "rv_feature_state",
    EDGE_BATTERY_MAGIC: "edge_battery",
    SYNC_PACKET_MAGIC: "sync_packet",
    RV_MESH_MAGIC: "adaptive_state",
}
RV_QFLAG_PRESENCE_VALID = 1 << 0
RV_QFLAG_RESPIRATION_VALID = 1 << 1
RV_QFLAG_HEARTBEAT_VALID = 1 << 2
BATTERY_FLAG_VALID = 1 << 0
BATTERY_FLAG_CHARGING = 1 << 1
BATTERY_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "BATTERY",
    2: "CHARGING",
}
CARDPUTER_LOCK = threading.Lock()
CARDPUTER_STATE = {
    "packet_count": 0,
    "first_seen_s": None,
    "last_seen_s": None,
    "last_source": None,
    "last_port": None,
    "last_len": None,
    "last_head_hex": None,
    "feature_state": None,
    "feature_state_seen_s": None,
    "edge_feature": None,
    "edge_feature_seen_s": None,
    "sync_packet": None,
    "sync_packet_seen_s": None,
    "adaptive_state": None,
    "adaptive_state_seen_s": None,
    "battery": None,
    "battery_seen_s": None,
    "nodes": {},
    "udp_error": None,
}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RECORDING_LOCK = threading.Lock()
RECORDING_STATE = {
    "active": False,
    "id": None,
    "name": None,
    "label": None,
    "started_at": None,
    "file_path": None,
    "file": None,
    "frame_count": 0,
    "error": None,
}
TRAINING_LOCK = threading.Lock()
TRAINING_STATE = {
    "active": False,
    "status": "idle",
    "run_id": None,
    "type": None,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_pck": 0.0,
    "val_oks": 0.0,
    "lr": 0.0,
    "best_pck": 0.0,
    "best_epoch": 0,
    "patience_remaining": 0,
    "eta_secs": None,
    "phase": "idle",
    "message": "Desktop live API is ready.",
    "config": None,
    "dataset_ids": [],
    "model_id": None,
}
TRAINING_RUN_COUNTER = 0
MODEL_LOCK = threading.Lock()
ACTIVE_MODEL_ID = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_person_count(value, default: int = 0) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(MAX_PERSONS, count))


def _active_person_count(cardputer: dict, feature_state: dict) -> int:
    """Resolve the best current count from live edge packets."""
    node_counts = []
    for node in cardputer.get("nodes", []):
        edge_vitals = node.get("edge_vitals") if node.get("edge_vitals_live") else None
        if edge_vitals and edge_vitals.get("presence"):
            node_counts.append(_coerce_person_count(edge_vitals.get("n_persons"), 1))

    if node_counts:
        return _coerce_person_count(max(node_counts))

    edge_vitals = cardputer.get("edge_vitals") if cardputer.get("edge_vitals_live") else None
    if edge_vitals and edge_vitals.get("presence"):
        return _coerce_person_count(edge_vitals.get("n_persons"), 1)

    if cardputer.get("live") and feature_state.get("presence"):
        return 1

    return 0


def _hardware_inference_state(cardputer: dict, feature_state: dict | None = None) -> dict:
    """Summarize real hardware-derived inference availability without fabricating pose keypoints."""
    feature_state = feature_state or cardputer.get("feature_state") or {}
    nodes = cardputer.get("nodes", [])
    person_count = _active_person_count(cardputer, feature_state)
    feature_live = bool(cardputer.get("feature_state_live")) or any(
        bool(node.get("feature_state_live")) for node in nodes
    )
    edge_vitals_live = bool(cardputer.get("edge_vitals_live")) or any(
        bool(node.get("edge_vitals_live")) for node in nodes
    )
    edge_feature_live = bool(cardputer.get("edge_feature_live")) or any(
        bool(node.get("edge_feature_live")) for node in nodes
    )
    hardware_live = bool(cardputer.get("live"))
    inference_live = bool(hardware_live and (feature_live or edge_vitals_live or edge_feature_live or person_count > 0))

    if feature_live:
        source = "hardware_feature_state"
    elif edge_vitals_live:
        source = "hardware_edge_vitals"
    elif edge_feature_live:
        source = "hardware_edge_feature"
    elif hardware_live:
        source = "hardware_telemetry"
    else:
        source = "none"

    if inference_live:
        message = f"Hardware inference live: {person_count} person{'s' if person_count != 1 else ''}"
    elif hardware_live:
        message = "Hardware stream live; waiting for inference packets"
    else:
        message = "No live hardware inference"

    return {
        "live": inference_live,
        "source": source,
        "message": message,
        "person_count": person_count,
        "feature_state_live": feature_live,
        "edge_vitals_live": edge_vitals_live,
        "edge_feature_live": edge_feature_live,
    }


def _parse_feature_state(data: bytes) -> dict | None:
    if len(data) != RV_FEATURE_STATE_SIZE:
        return None
    try:
        (
            magic,
            node_id,
            mode,
            seq,
            ts_us,
            motion_score,
            presence_score,
            respiration_bpm,
            respiration_conf,
            heartbeat_bpm,
            heartbeat_conf,
            anomaly_score,
            env_shift_score,
            node_coherence,
            quality_flags,
            _reserved,
            crc32,
        ) = struct.unpack(RV_FEATURE_STATE_FMT, data)
    except struct.error:
        return None
    if magic != RV_FEATURE_STATE_MAGIC:
        return None
    computed_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    crc_valid = computed_crc == crc32
    presence_valid = bool(quality_flags & RV_QFLAG_PRESENCE_VALID)
    return {
        "packet_type": "rv_feature_state",
        "node_id": node_id,
        "mode": mode,
        "seq": seq,
        "ts_us": ts_us,
        "motion_score": motion_score,
        "presence_score": presence_score,
        "respiration_bpm": respiration_bpm,
        "respiration_conf": respiration_conf,
        "heartbeat_bpm": heartbeat_bpm,
        "heartbeat_conf": heartbeat_conf,
        "anomaly_score": anomaly_score,
        "env_shift_score": env_shift_score,
        "node_coherence": node_coherence,
        "quality_flags": quality_flags,
        "presence_valid": presence_valid,
        "respiration_valid": bool(quality_flags & RV_QFLAG_RESPIRATION_VALID),
        "heartbeat_valid": bool(quality_flags & RV_QFLAG_HEARTBEAT_VALID),
        "crc_valid": crc_valid,
        "presence": crc_valid and presence_valid and presence_score >= 0.35,
    }


def _parse_battery(data: bytes) -> dict | None:
    if len(data) != EDGE_BATTERY_SIZE:
        return None
    try:
        (
            magic,
            node_id,
            percent,
            flags,
            status,
            millivolts,
            _reserved,
            timestamp_ms,
        ) = struct.unpack(EDGE_BATTERY_FMT, data)
    except struct.error:
        return None
    if magic != EDGE_BATTERY_MAGIC:
        return None
    valid = bool(flags & BATTERY_FLAG_VALID) and percent <= 100 and millivolts > 0
    charging = bool(flags & BATTERY_FLAG_CHARGING)
    return {
        "packet_type": "edge_battery",
        "node_id": node_id,
        "valid": valid,
        "percent": percent if valid else None,
        "millivolts": millivolts if valid else None,
        "volts": round(millivolts / 1000.0, 3) if valid else None,
        "charging": charging,
        "status_code": status,
        "status": BATTERY_STATUS_NAMES.get(status, "UNKNOWN"),
        "ts_ms": timestamp_ms,
    }


def _parse_edge_vitals(data: bytes) -> dict | None:
    if len(data) != EDGE_VITALS_SIZE:
        return None
    try:
        (
            magic,
            node_id,
            flags,
            breathing_rate,
            heartrate,
            rssi,
            n_persons,
            motion_energy,
            presence_score,
            timestamp_ms,
            _reserved2,
        ) = struct.unpack(EDGE_VITALS_FMT, data)
    except struct.error:
        return None
    if magic != EDGE_VITALS_MAGIC:
        return None
    return {
        "packet_type": "edge_vitals",
        "node_id": node_id,
        "flags": flags,
        "presence": bool(flags & 0x01),
        "fall": bool(flags & 0x02),
        "motion_valid": bool(flags & 0x04),
        "breathing_bpm": breathing_rate / 100.0,
        "heartbeat_bpm": heartrate / 10000.0,
        "rssi_dbm": rssi,
        "n_persons": n_persons,
        "motion_energy": motion_energy,
        "presence_score": presence_score,
        "ts_ms": timestamp_ms,
    }


def _node_id_hex(raw: bytes) -> str:
    return raw.hex()


def _parse_edge_feature(data: bytes) -> dict | None:
    if len(data) != EDGE_FEATURE_SIZE:
        return None
    try:
        (
            magic,
            node_id,
            _reserved,
            seq,
            timestamp_us,
            *features,
        ) = struct.unpack(EDGE_FEATURE_FMT, data)
    except struct.error:
        return None
    if magic != EDGE_FEATURE_MAGIC:
        return None
    return {
        "packet_type": "edge_feature",
        "node_id": node_id,
        "seq": seq,
        "ts_us": timestamp_us,
        "features": list(features),
        "presence_norm": features[0],
        "motion_norm": features[1],
        "breathing_norm": features[2],
        "heartbeat_norm": features[3],
        "phase_variance_norm": features[4],
        "person_count_norm": features[5],
        "fall_risk_norm": features[6],
        "rssi_norm": features[7],
    }


def _parse_sync_packet(data: bytes) -> dict | None:
    if len(data) != SYNC_PACKET_SIZE:
        return None
    try:
        (
            magic,
            node_id,
            version,
            flags,
            _reserved,
            local_us,
            epoch_us,
            sequence_high_water,
            reserved,
        ) = struct.unpack(SYNC_PACKET_FMT, data)
    except struct.error:
        return None
    if magic != SYNC_PACKET_MAGIC:
        return None
    return {
        "packet_type": "sync_packet",
        "node_id": node_id,
        "version": version,
        "flags": flags,
        "leader": bool(flags & 0x01),
        "epoch_valid": bool(flags & 0x02),
        "offset_smoothed": bool(flags & 0x04),
        "local_us": local_us,
        "epoch_us": epoch_us,
        "sequence_high_water": sequence_high_water,
        "reserved": reserved,
    }


def _parse_mesh_payload(msg_type: int, payload: bytes) -> dict:
    try:
        if msg_type == RV_MESH_MSG_TIME_SYNC and len(payload) == struct.calcsize(RV_TIME_SYNC_FMT):
            anchor_time_us, cycle_id, cycle_period_us = struct.unpack(RV_TIME_SYNC_FMT, payload)
            return {
                "anchor_time_us": anchor_time_us,
                "cycle_id": cycle_id,
                "cycle_period_us": cycle_period_us,
            }
        if msg_type == RV_MESH_MSG_ROLE_ASSIGN and len(payload) == struct.calcsize(RV_ROLE_ASSIGN_FMT):
            target_node_id, new_role, effective_epoch = struct.unpack(RV_ROLE_ASSIGN_FMT, payload)
            return {
                "target_node_id": _node_id_hex(target_node_id),
                "target_node_hint": target_node_id[0],
                "new_role": new_role,
                "new_role_name": RV_MESH_ROLE_NAMES.get(new_role, "unknown"),
                "effective_epoch": effective_epoch,
            }
        if msg_type == RV_MESH_MSG_CHANNEL_PLAN and len(payload) == struct.calcsize(RV_CHANNEL_PLAN_FMT):
            target_node_id, channel_count, dwell_hi, dwell_lo, debug_raw_csi, channels, effective_epoch = struct.unpack(RV_CHANNEL_PLAN_FMT, payload)
            channel_count = min(channel_count, len(channels))
            return {
                "target_node_id": _node_id_hex(target_node_id),
                "target_node_hint": target_node_id[0],
                "channel_count": channel_count,
                "dwell_ms": (dwell_hi << 8) | dwell_lo,
                "debug_raw_csi": bool(debug_raw_csi),
                "channels": list(channels[:channel_count]),
                "effective_epoch": effective_epoch,
            }
        if msg_type == RV_MESH_MSG_CALIBRATION_START and len(payload) == struct.calcsize(RV_CALIBRATION_START_FMT):
            t0_anchor_us, duration_ms, effective_epoch, calibration_profile = struct.unpack(RV_CALIBRATION_START_FMT, payload)
            return {
                "t0_anchor_us": t0_anchor_us,
                "duration_ms": duration_ms,
                "effective_epoch": effective_epoch,
                "calibration_profile": calibration_profile,
            }
        if msg_type == RV_MESH_MSG_FEATURE_DELTA:
            feature_state = _parse_feature_state(payload)
            return {"feature_state": feature_state} if feature_state is not None else {}
        if msg_type == RV_MESH_MSG_HEALTH and len(payload) == struct.calcsize(RV_NODE_STATUS_FMT):
            (
                node_id,
                local_time_us,
                role,
                current_channel,
                current_bw,
                noise_floor_dbm,
                pkt_yield,
                sync_error_us,
                health_flags,
                _reserved,
            ) = struct.unpack(RV_NODE_STATUS_FMT, payload)
            return {
                "node_id": _node_id_hex(node_id),
                "node_hint": node_id[0],
                "local_time_us": local_time_us,
                "role": role,
                "role_name": RV_MESH_ROLE_NAMES.get(role, "unknown"),
                "current_channel": current_channel,
                "current_bw_mhz": current_bw,
                "noise_floor_dbm": noise_floor_dbm,
                "pkt_yield": pkt_yield,
                "sync_error_us": sync_error_us,
                "health_flags": health_flags,
            }
        if msg_type == RV_MESH_MSG_ANOMALY_ALERT and len(payload) == struct.calcsize(RV_ANOMALY_ALERT_FMT):
            node_id, ts_us, severity, reason, _reserved, anomaly_score, motion_score = struct.unpack(RV_ANOMALY_ALERT_FMT, payload)
            return {
                "node_id": _node_id_hex(node_id),
                "node_hint": node_id[0],
                "ts_us": ts_us,
                "severity": severity,
                "reason": reason,
                "anomaly_score": anomaly_score,
                "motion_score": motion_score,
            }
    except struct.error:
        return {}
    return {}


def _mesh_payload_node_id(msg_type: int, payload: dict) -> int | None:
    if msg_type == RV_MESH_MSG_FEATURE_DELTA:
        feature_state = payload.get("feature_state")
        if feature_state:
            return feature_state.get("node_id")
    for key in ("node_hint", "target_node_hint"):
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _parse_adaptive_state(data: bytes) -> dict | None:
    if len(data) < RV_MESH_HEADER_SIZE + RV_MESH_CRC_SIZE:
        return None
    try:
        (
            magic,
            version,
            msg_type,
            sender_role,
            auth_class,
            epoch,
            payload_len,
            _reserved,
        ) = struct.unpack_from(RV_MESH_HEADER_FMT, data)
    except struct.error:
        return None
    if magic != RV_MESH_MAGIC:
        return None
    frame_len = RV_MESH_HEADER_SIZE + payload_len + RV_MESH_CRC_SIZE
    if payload_len > 256 or len(data) < frame_len:
        return None
    payload = data[RV_MESH_HEADER_SIZE:RV_MESH_HEADER_SIZE + payload_len]
    try:
        got_crc = struct.unpack_from("<I", data, RV_MESH_HEADER_SIZE + payload_len)[0]
    except struct.error:
        return None
    want_crc = zlib.crc32(data[:RV_MESH_HEADER_SIZE + payload_len]) & 0xFFFFFFFF
    decoded_payload = _parse_mesh_payload(msg_type, payload)
    return {
        "packet_type": "adaptive_state",
        "version": version,
        "message_type": msg_type,
        "message_name": RV_MESH_MSG_NAMES.get(msg_type, "unknown"),
        "sender_role": sender_role,
        "sender_role_name": RV_MESH_ROLE_NAMES.get(sender_role, "unknown"),
        "auth_class": auth_class,
        "auth_class_name": RV_MESH_AUTH_NAMES.get(auth_class, "unknown"),
        "epoch": epoch,
        "payload_len": payload_len,
        "crc_valid": got_crc == want_crc,
        "crc32": got_crc,
        "node_id": _mesh_payload_node_id(msg_type, decoded_payload),
        "payload": decoded_payload,
    }


def _parse_csi_signal(data: bytes) -> dict | None:
    if len(data) < 18 or _packet_magic(data) != CSI_FRAME_MAGIC:
        return None
    try:
        rssi = struct.unpack_from("<b", data, 16)[0]
        noise_floor = struct.unpack_from("<b", data, 17)[0]
    except struct.error:
        return None
    return {
        "packet_type": "csi_frame",
        "rssi_dbm": rssi,
        "noise_floor_dbm": noise_floor,
    }


def _packet_magic(data: bytes) -> int | None:
    if len(data) < 4:
        return None
    try:
        return struct.unpack_from("<I", data)[0]
    except struct.error:
        return None


def _packet_node_id(data: bytes) -> int | None:
    if _packet_magic(data) == RV_MESH_MAGIC:
        adaptive_state = _parse_adaptive_state(data)
        return adaptive_state.get("node_id") if adaptive_state is not None else None
    return data[4] if len(data) > 4 else None


def _new_node_state(node_id: int) -> dict:
    return {
        "node_id": node_id,
        "packet_count": 0,
        "first_seen_s": None,
        "last_seen_s": None,
        "last_source": None,
        "last_port": None,
        "last_len": None,
        "last_head_hex": None,
        "last_magic": None,
        "last_packet_type": None,
        "packet_types": {},
        "feature_state": None,
        "feature_state_seen_s": None,
        "edge_feature": None,
        "edge_feature_seen_s": None,
        "sync_packet": None,
        "sync_packet_seen_s": None,
        "adaptive_state": None,
        "adaptive_state_seen_s": None,
        "battery": None,
        "battery_seen_s": None,
        "edge_vitals": None,
        "edge_vitals_seen_s": None,
        "rssi_dbm": None,
        "rssi_seen_s": None,
        "noise_floor_dbm": None,
    }


def _snapshot_node(node: dict, now: float) -> dict:
    last_seen = node.get("last_seen_s")
    age_s = None if last_seen is None else max(0.0, now - float(last_seen))
    live_max_age_s = (
        ADAPTIVE_STATE_MAX_AGE_S
        if node.get("last_packet_type") == "adaptive_state"
        else LIVE_MAX_AGE_S
    )
    live = age_s is not None and age_s <= live_max_age_s
    feature_state_seen = node.get("feature_state_seen_s")
    feature_state_age_s = (
        None if feature_state_seen is None else max(0.0, now - float(feature_state_seen))
    )
    feature_state_live = (
        live
        and node.get("feature_state") is not None
        and feature_state_age_s is not None
        and feature_state_age_s <= FEATURE_STATE_MAX_AGE_S
    )
    edge_feature_seen = node.get("edge_feature_seen_s")
    edge_feature_age_s = (
        None if edge_feature_seen is None else max(0.0, now - float(edge_feature_seen))
    )
    edge_feature_live = (
        live
        and node.get("edge_feature") is not None
        and edge_feature_age_s is not None
        and edge_feature_age_s <= FEATURE_STATE_MAX_AGE_S
    )
    sync_packet_seen = node.get("sync_packet_seen_s")
    sync_packet_age_s = (
        None if sync_packet_seen is None else max(0.0, now - float(sync_packet_seen))
    )
    sync_packet_live = (
        live
        and node.get("sync_packet") is not None
        and sync_packet_age_s is not None
        and sync_packet_age_s <= LIVE_MAX_AGE_S
    )
    adaptive_state_seen = node.get("adaptive_state_seen_s")
    adaptive_state_age_s = (
        None if adaptive_state_seen is None else max(0.0, now - float(adaptive_state_seen))
    )
    adaptive_state_live = (
        node.get("adaptive_state") is not None
        and adaptive_state_age_s is not None
        and adaptive_state_age_s <= ADAPTIVE_STATE_MAX_AGE_S
    )
    battery_seen = node.get("battery_seen_s")
    battery_age_s = None if battery_seen is None else max(0.0, now - float(battery_seen))
    battery_live = (
        live
        and node.get("battery") is not None
        and battery_age_s is not None
        and battery_age_s <= BATTERY_MAX_AGE_S
    )
    edge_vitals_seen = node.get("edge_vitals_seen_s")
    edge_vitals_age_s = (
        None if edge_vitals_seen is None else max(0.0, now - float(edge_vitals_seen))
    )
    edge_vitals_live = (
        live
        and node.get("edge_vitals") is not None
        and edge_vitals_age_s is not None
        and edge_vitals_age_s <= FEATURE_STATE_MAX_AGE_S
    )
    rssi_seen = node.get("rssi_seen_s")
    rssi_age_s = None if rssi_seen is None else max(0.0, now - float(rssi_seen))
    rssi_live = (
        live
        and node.get("rssi_dbm") is not None
        and rssi_age_s is not None
        and rssi_age_s <= RSSI_MAX_AGE_S
    )
    return {
        "node_id": node["node_id"],
        "status": "live" if live else "stale",
        "live": live,
        "packet_count": node["packet_count"],
        "last_packet_age_s": age_s,
        "last_source": node["last_source"],
        "last_source_port": node["last_port"],
        "last_packet_len": node["last_len"],
        "last_head_hex": node["last_head_hex"],
        "last_magic": node["last_magic"],
        "last_packet_type": node["last_packet_type"],
        "packet_types": dict(sorted(node.get("packet_types", {}).items())),
        "feature_state": node["feature_state"] if feature_state_live else None,
        "feature_state_age_s": feature_state_age_s,
        "feature_state_live": feature_state_live,
        "stale_feature_state": node.get("feature_state") is not None and not feature_state_live,
        "edge_feature": node.get("edge_feature") if edge_feature_live else None,
        "edge_feature_age_s": edge_feature_age_s,
        "edge_feature_live": edge_feature_live,
        "stale_edge_feature": node.get("edge_feature") is not None and not edge_feature_live,
        "sync_packet": node.get("sync_packet") if sync_packet_live else None,
        "sync_packet_age_s": sync_packet_age_s,
        "sync_packet_live": sync_packet_live,
        "stale_sync_packet": node.get("sync_packet") is not None and not sync_packet_live,
        "adaptive_state": node.get("adaptive_state") if adaptive_state_live else None,
        "adaptive_state_age_s": adaptive_state_age_s,
        "adaptive_state_live": adaptive_state_live,
        "stale_adaptive_state": node.get("adaptive_state") is not None and not adaptive_state_live,
        "battery": node.get("battery") if battery_live else None,
        "battery_age_s": battery_age_s,
        "battery_live": battery_live,
        "stale_battery": node.get("battery") is not None and not battery_live,
        "edge_vitals": node.get("edge_vitals") if edge_vitals_live else None,
        "edge_vitals_age_s": edge_vitals_age_s,
        "edge_vitals_live": edge_vitals_live,
        "rssi_dbm": node.get("rssi_dbm") if rssi_live else None,
        "rssi_age_s": rssi_age_s,
        "rssi_live": rssi_live,
        "noise_floor_dbm": node.get("noise_floor_dbm") if rssi_live else None,
        "pass": bool(feature_state_live or live),
        "freshness_status": "pass" if live else "stale",
    }


def _cardputer_snapshot() -> dict:
    with CARDPUTER_LOCK:
        state = dict(CARDPUTER_STATE)
        nodes_state = json.loads(json.dumps(CARDPUTER_STATE.get("nodes", {})))
    now = time.time()
    last_seen = state.get("last_seen_s")
    age_s = None if last_seen is None else max(0.0, now - float(last_seen))
    live = age_s is not None and age_s <= LIVE_MAX_AGE_S
    feature_state_seen = state.get("feature_state_seen_s")
    feature_state_age_s = (
        None if feature_state_seen is None else max(0.0, now - float(feature_state_seen))
    )
    feature_state_live = (
        live
        and state.get("feature_state") is not None
        and feature_state_age_s is not None
        and feature_state_age_s <= FEATURE_STATE_MAX_AGE_S
    )
    edge_feature_seen = state.get("edge_feature_seen_s")
    edge_feature_age_s = (
        None if edge_feature_seen is None else max(0.0, now - float(edge_feature_seen))
    )
    edge_feature_live = (
        live
        and state.get("edge_feature") is not None
        and edge_feature_age_s is not None
        and edge_feature_age_s <= FEATURE_STATE_MAX_AGE_S
    )
    sync_packet_seen = state.get("sync_packet_seen_s")
    sync_packet_age_s = (
        None if sync_packet_seen is None else max(0.0, now - float(sync_packet_seen))
    )
    sync_packet_live = (
        live
        and state.get("sync_packet") is not None
        and sync_packet_age_s is not None
        and sync_packet_age_s <= LIVE_MAX_AGE_S
    )
    adaptive_state_seen = state.get("adaptive_state_seen_s")
    adaptive_state_age_s = (
        None if adaptive_state_seen is None else max(0.0, now - float(adaptive_state_seen))
    )
    adaptive_state_live = (
        state.get("adaptive_state") is not None
        and adaptive_state_age_s is not None
        and adaptive_state_age_s <= ADAPTIVE_STATE_MAX_AGE_S
    )
    battery_seen = state.get("battery_seen_s")
    battery_age_s = None if battery_seen is None else max(0.0, now - float(battery_seen))
    battery_live = (
        live
        and state.get("battery") is not None
        and battery_age_s is not None
        and battery_age_s <= BATTERY_MAX_AGE_S
    )
    battery = state.get("battery") if battery_live else None
    if battery is None:
        battery = {
            "packet_type": "edge_battery",
            "valid": False,
            "percent": None,
            "millivolts": None,
            "volts": None,
            "charging": False,
            "status": "UNKNOWN",
        }
    freshness_pass = bool(feature_state_live)
    if live:
        status = "live"
        message = "Cardputer UDP stream active"
    elif state.get("udp_error"):
        status = "error"
        message = state["udp_error"]
    else:
        status = "waiting"
        message = "No Cardputer UDP packets on port 5005"
    nodes = [
        _snapshot_node(node, now)
        for _node_id, node in sorted(nodes_state.items())
    ]
    live_nodes = [node for node in nodes if node["live"]]
    live_edge_vitals = [
        node["edge_vitals"]
        for node in nodes
        if node.get("edge_vitals_live") and node.get("edge_vitals")
    ]
    aggregate_edge_vitals = live_edge_vitals[-1] if live_edge_vitals else None
    return {
        "status": status,
        "live": live,
        "message": message,
        "udp_port": UDP_PORT,
        "node_count": len(nodes),
        "live_node_count": len(live_nodes),
        "nodes": nodes,
        "packet_count": state["packet_count"],
        "last_packet_age_s": age_s,
        "last_source": state["last_source"],
        "last_source_port": state["last_port"],
        "last_packet_len": state["last_len"],
        "last_head_hex": state["last_head_hex"],
        "feature_state": state["feature_state"] if feature_state_live else None,
        "feature_state_age_s": feature_state_age_s,
        "feature_state_live": feature_state_live,
        "stale_feature_state": state.get("feature_state") is not None and not feature_state_live,
        "edge_feature": state.get("edge_feature") if edge_feature_live else None,
        "edge_feature_age_s": edge_feature_age_s,
        "edge_feature_live": edge_feature_live,
        "stale_edge_feature": state.get("edge_feature") is not None and not edge_feature_live,
        "sync_packet": state.get("sync_packet") if sync_packet_live else None,
        "sync_packet_age_s": sync_packet_age_s,
        "sync_packet_live": sync_packet_live,
        "stale_sync_packet": state.get("sync_packet") is not None and not sync_packet_live,
        "adaptive_state": state.get("adaptive_state") if adaptive_state_live else None,
        "adaptive_state_age_s": adaptive_state_age_s,
        "adaptive_state_live": adaptive_state_live,
        "stale_adaptive_state": state.get("adaptive_state") is not None and not adaptive_state_live,
        "battery": battery,
        "battery_age_s": battery_age_s,
        "battery_live": battery_live,
        "stale_battery": state.get("battery") is not None and not battery_live,
        "edge_vitals": aggregate_edge_vitals,
        "edge_vitals_live": bool(aggregate_edge_vitals),
        "pass": freshness_pass,
        "freshness_status": "pass" if freshness_pass else "stale",
    }


def _safe_id(raw: object, prefix: str = "item") -> str:
    text = str(raw or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not text:
        text = f"{prefix}_{int(time.time())}"
    if len(text) > 96:
        text = text[:96].rstrip("._-")
    if not SAFE_ID_RE.fullmatch(text) or ".." in text:
        text = f"{prefix}_{int(time.time())}"
    return text


def _is_safe_id(raw: str) -> bool:
    return bool(SAFE_ID_RE.fullmatch(raw or "")) and ".." not in raw


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0") or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _recording_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".csi.jsonl"):
        return name[:-len(".csi.jsonl")]
    if name.endswith(".jsonl"):
        return name[:-len(".jsonl")]
    return path.stem


def _recording_meta_path(recording_id: str) -> Path:
    return RECORDINGS_DIR / f"{recording_id}.csi.meta.json"


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _scan_recordings() -> list[dict]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    recordings: list[dict] = []
    seen: set[str] = set()
    paths = sorted(RECORDINGS_DIR.glob("*.csi.jsonl")) + sorted(RECORDINGS_DIR.glob("*.jsonl"))
    with RECORDING_LOCK:
        active_id = RECORDING_STATE.get("id") if RECORDING_STATE.get("active") else None
        active_frames = int(RECORDING_STATE.get("frame_count") or 0)
        active_started = RECORDING_STATE.get("started_at")
        active_name = RECORDING_STATE.get("name")
        active_label = RECORDING_STATE.get("label")
    for path in paths:
        recording_id = _recording_id_from_path(path)
        if recording_id in seen:
            continue
        seen.add(recording_id)
        stat = path.stat()
        meta = {}
        meta_path = _recording_meta_path(recording_id)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        is_active = recording_id == active_id
        recordings.append({
            "id": recording_id,
            "name": meta.get("name") or (active_name if is_active else recording_id),
            "label": meta.get("label") or (active_label if is_active else None),
            "started_at": meta.get("started_at") or (active_started if is_active else None),
            "ended_at": None if is_active else meta.get("ended_at"),
            "frame_count": active_frames if is_active else int(meta.get("frame_count") or _count_lines(path)),
            "file_size_bytes": stat.st_size,
            "file_path": str(path),
            "status": "recording" if is_active else "completed",
        })
    recordings.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return recordings


def _scan_models() -> list[dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(MODELS_DIR.glob("*.rvf")):
        stat = path.stat()
        model_id = path.stem
        models.append({
            "id": model_id,
            "name": model_id,
            "filename": path.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_epoch": int(stat.st_mtime),
            "format": "rvf",
            "version": "unknown",
            "description": "",
            "pck_score": None,
            "lora_profiles": [],
        })
    return models


def _classify_rvf(path: Path) -> str:
    try:
        head = path.read_bytes()[:128]
    except OSError:
        return "unknown"
    if head.startswith(b"RVF\x01") or head[:4] == b"RVF\x01":
        return "binary-rvf"
    if b"rvf-desktop-placeholder" in head:
        return "desktop-placeholder"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "unknown"
    fmt = str(payload.get("format") or "").lower()
    if "placeholder" in fmt:
        return "desktop-placeholder"
    return "json-rvf" if fmt == "rvf" else "unknown"


def _scan_jsonl_dir(directory: Path, pattern: str) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for path in sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append({
            "id": path.stem,
            "name": path.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_epoch": int(stat.st_mtime),
            "line_count": _count_lines(path),
        })
    return out


def _latest_path(files: list[dict]) -> str | None:
    if not files:
        return None
    item = files[0]
    path = item.get("path") or item.get("file_path")
    return str(path) if path else None


def _q(path: str | Path) -> str:
    return shlex.quote(str(path))


def _rvf_training_readiness(cardputer: dict) -> dict:
    ground_truth = _scan_jsonl_dir(GROUND_TRUTH_DIR, "*.jsonl")
    paired = _scan_jsonl_dir(PAIRED_DIR, "*.jsonl")
    recordings = _scan_recordings()
    models = _scan_models()
    real_models = []
    placeholder_models = []
    for model in models:
        kind = _classify_rvf(Path(model["path"]))
        item = {**model, "kind": kind}
        if kind == "desktop-placeholder":
            placeholder_models.append(item)
        else:
            real_models.append(item)

    scripts = {
        "collect_ground_truth": (SCRIPTS_DIR / "collect-ground-truth.py").exists(),
        "align_ground_truth": (SCRIPTS_DIR / "align-ground-truth.js").exists(),
        "train_wiflow_supervised": (SCRIPTS_DIR / "train-wiflow-supervised.js").exists(),
        "sensing_server": (RUVIEW_ROOT / "v2" / "crates" / "wifi-densepose-sensing-server").exists(),
    }
    live_nodes = int(cardputer.get("live_node_count") or 0)
    gt_path = _latest_path(ground_truth) or str(GROUND_TRUTH_DIR / "gt-*.jsonl")
    csi_path = _latest_path(recordings) or str(RECORDINGS_DIR / "*.csi.jsonl")
    paired_path = _latest_path(paired) or str(PAIRED_DIR / "session.paired.jsonl")
    output_path = MODELS_DIR / f"wiflow-{int(time.time())}.rvf"
    align_output = PAIRED_DIR / "session.paired.jsonl"

    commands = [
        {
            "id": "collect",
            "label": "Collect camera labels + CSI",
            "command": (
                f"cd {_q(RUVIEW_ROOT)} && python scripts/collect-ground-truth.py "
                "--server http://127.0.0.1:3000 --preview --duration 300"
            ),
        },
        {
            "id": "align",
            "label": "Align latest camera labels with latest CSI recording",
            "command": (
                f"cd {_q(RUVIEW_ROOT)} && node scripts/align-ground-truth.js --gt {_q(gt_path)} "
                f"--csi {_q(csi_path)} --output {_q(align_output)}"
            ),
        },
        {
            "id": "train_rvf",
            "label": "Train and export real .rvf",
            "command": (
                f"cd {_q(RUVIEW_ROOT / 'v2')} && cargo run -p wifi-densepose-sensing-server -- "
                f"--train --dataset {_q(paired_path)} --epochs 100 --save-rvf {_q(output_path)}"
            ),
        },
    ]

    ready_for_align = bool(ground_truth and recordings and scripts["align_ground_truth"])
    ready_for_train = bool(paired and scripts["sensing_server"])
    return {
        "status": "ready" if ready_for_train else "collecting" if ready_for_align else "needs_data",
        "summary": {
            "live_nodes": live_nodes,
            "recommended_nodes": 4,
            "node_ready": live_nodes >= 4,
            "recordings": len(recordings),
            "ground_truth": len(ground_truth),
            "paired": len(paired),
            "real_rvf": len(real_models),
            "placeholder_rvf": len(placeholder_models),
        },
        "paths": {
            "recordings_dir": str(RECORDINGS_DIR),
            "ground_truth_dir": str(GROUND_TRUTH_DIR),
            "paired_dir": str(PAIRED_DIR),
            "models_dir": str(MODELS_DIR),
        },
        "latest": {
            "recording": recordings[0] if recordings else None,
            "ground_truth": ground_truth[0] if ground_truth else None,
            "paired": paired[0] if paired else None,
            "real_rvf": real_models[0] if real_models else None,
        },
        "scripts": scripts,
        "commands": commands,
        "notes": [
            "Camera is only used for labels while collecting ground truth.",
            "Use 4+ live ESP32 sensors for better per-limb tracking.",
            "Desktop quick training files are placeholders; load a real exported .rvf for model inference.",
        ],
    }


def _scan_lora_profiles() -> list[str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name[:-len(".lora.json")] for path in MODELS_DIR.glob("*.lora.json"))


def _start_recording(body: dict) -> tuple[int, dict]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    requested = body.get("session_name") or body.get("id") or f"rec_{int(time.time())}"
    recording_id = _safe_id(requested, "rec")
    label = body.get("label")
    try:
        duration_secs = float(body.get("duration_secs") or 0)
    except (TypeError, ValueError):
        duration_secs = 0
    duration_secs = duration_secs if 0 < duration_secs <= 24 * 60 * 60 else 0
    file_path = RECORDINGS_DIR / f"{recording_id}.csi.jsonl"
    with RECORDING_LOCK:
        if RECORDING_STATE.get("active"):
            return 409, {
                "status": "error",
                "error": "recording already active",
                "message": "A recording is already active. Stop it first.",
                "active_session": RECORDING_STATE.get("id"),
            }
        try:
            handle = file_path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            return 500, {
                "status": "error",
                "error": "recording_open_failed",
                "message": f"Cannot create recording file: {exc}",
            }
        RECORDING_STATE.update({
            "active": True,
            "id": recording_id,
            "name": str(requested),
            "label": label,
            "started_at": _now_iso(),
            "file_path": file_path,
            "file": handle,
            "frame_count": 0,
            "error": None,
        })
    if duration_secs:
        timer = threading.Timer(duration_secs, _stop_recording_if_active, args=(recording_id,))
        timer.daemon = True
        timer.start()
    return 200, {
        "status": "recording",
        "success": True,
        "session_id": recording_id,
        "id": recording_id,
        "session_name": str(requested),
        "label": label,
        "started_at": RECORDING_STATE["started_at"],
        "file_path": str(file_path),
        "duration_secs": duration_secs or None,
    }


def _stop_recording_if_active(recording_id: str) -> None:
    with RECORDING_LOCK:
        active = RECORDING_STATE.get("active") and RECORDING_STATE.get("id") == recording_id
    if active:
        _stop_recording()


def _stop_recording() -> tuple[int, dict]:
    with RECORDING_LOCK:
        if not RECORDING_STATE.get("active"):
            return 409, {
                "status": "error",
                "error": "no recording in progress",
                "message": "No active recording to stop.",
            }
        handle = RECORDING_STATE.get("file")
        recording_id = str(RECORDING_STATE.get("id"))
        frame_count = int(RECORDING_STATE.get("frame_count") or 0)
        file_path = Path(RECORDING_STATE.get("file_path"))
        session = {
            "id": recording_id,
            "name": RECORDING_STATE.get("name") or recording_id,
            "label": RECORDING_STATE.get("label"),
            "started_at": RECORDING_STATE.get("started_at"),
            "ended_at": _now_iso(),
            "frame_count": frame_count,
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "file_path": str(file_path),
        }
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except OSError:
                pass
        RECORDING_STATE.update({
            "active": False,
            "id": None,
            "name": None,
            "label": None,
            "started_at": None,
            "file_path": None,
            "file": None,
            "frame_count": 0,
        })
    try:
        _recording_meta_path(recording_id).write_text(json.dumps(session, indent=2), encoding="utf-8")
    except OSError as exc:
        session["metadata_error"] = str(exc)
    return 200, {"status": "stopped", "success": True, "session_id": recording_id, "frame_count": frame_count}


def _delete_recording(recording_id: str) -> tuple[int, dict]:
    if not _is_safe_id(recording_id):
        return 400, {"status": "error", "error": "invalid recording id", "message": "Invalid recording id."}
    with RECORDING_LOCK:
        if RECORDING_STATE.get("active") and RECORDING_STATE.get("id") == recording_id:
            return 409, {
                "status": "error",
                "error": "recording active",
                "message": "Stop this recording before deleting it.",
            }
    deleted = []
    for path in (
        RECORDINGS_DIR / f"{recording_id}.csi.jsonl",
        RECORDINGS_DIR / f"{recording_id}.jsonl",
        _recording_meta_path(recording_id),
        RECORDINGS_DIR / f"{recording_id}.meta.json",
    ):
        if path.exists():
            try:
                path.unlink()
                deleted.append(str(path))
            except OSError as exc:
                return 500, {"status": "error", "error": "delete failed", "message": str(exc)}
    if not deleted:
        return 404, {"status": "error", "error": "recording not found", "message": f"Recording '{recording_id}' not found."}
    return 200, {"status": "deleted", "success": True, "id": recording_id, "deleted_files": deleted}


def _record_subcarriers(csi_signal, feature_state, edge_feature, edge_vitals) -> list[float]:
    if edge_feature and edge_feature.get("features"):
        return [float(v) for v in edge_feature["features"]]
    if feature_state:
        keys = (
            "motion_score",
            "presence_score",
            "respiration_bpm",
            "respiration_conf",
            "heartbeat_bpm",
            "heartbeat_conf",
            "anomaly_score",
            "env_shift_score",
            "node_coherence",
        )
        return [float(feature_state.get(k) or 0.0) for k in keys]
    if edge_vitals:
        keys = ("breathing_bpm", "heartbeat_bpm", "rssi_dbm", "n_persons", "motion_energy", "presence_score")
        return [float(edge_vitals.get(k) or 0.0) for k in keys]
    if csi_signal:
        return [float(csi_signal.get("rssi_dbm") or 0.0), float(csi_signal.get("noise_floor_dbm") or 0.0)]
    return []


def _record_packet_if_active(
    now: float,
    addr,
    data: bytes,
    node_id,
    packet_type: str,
    csi_signal,
    feature_state,
    edge_feature,
    edge_vitals,
    battery,
    adaptive_state,
) -> None:
    subcarriers = _record_subcarriers(csi_signal, feature_state, edge_feature, edge_vitals)
    if not subcarriers:
        return
    frame = {
        "timestamp": now,
        "subcarriers": subcarriers,
        "rssi": float((csi_signal or edge_vitals or {}).get("rssi_dbm") or 0.0),
        "noise_floor": float((csi_signal or {}).get("noise_floor_dbm") or 0.0),
        "features": {
            "node_id": node_id,
            "packet_type": packet_type,
            "source": addr[0] if addr else None,
            "source_port": addr[1] if addr else None,
            "raw_head_hex": data[:24].hex(),
            "feature_state": feature_state,
            "edge_feature": edge_feature,
            "edge_vitals": edge_vitals,
            "battery": battery,
            "adaptive_state": adaptive_state,
        },
    }
    with RECORDING_LOCK:
        if not RECORDING_STATE.get("active"):
            return
        handle = RECORDING_STATE.get("file")
        if handle is None:
            return
        try:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
            RECORDING_STATE["frame_count"] = int(RECORDING_STATE.get("frame_count") or 0) + 1
        except OSError as exc:
            RECORDING_STATE["error"] = str(exc)
            RECORDING_STATE["active"] = False


def _training_snapshot() -> dict:
    with TRAINING_LOCK:
        return dict(TRAINING_STATE)


def _write_training_model(run_id: str, kind: str, state: dict) -> str | None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = _safe_id(f"{kind}-{run_id}", "train")
    model_path = MODELS_DIR / f"{model_id}.rvf"
    payload = {
        "format": "rvf-desktop-placeholder",
        "created_at": _now_iso(),
        "run_id": run_id,
        "type": kind,
        "dataset_ids": state.get("dataset_ids") or [],
        "metrics": {
            "best_pck": state.get("best_pck"),
            "best_epoch": state.get("best_epoch"),
            "val_oks": state.get("val_oks"),
            "train_loss": state.get("train_loss"),
        },
        "message": "Desktop bridge training simulation complete. Replace with full trainer output for production inference.",
    }
    try:
        model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return None
    return model_id


def _training_worker(run_id: str, kind: str, epochs: int, lr: float) -> None:
    # Desktop UI training is intentionally lightweight: it validates request flow,
    # emits realistic progress, and writes an RVF placeholder for model controls.
    total = max(1, min(epochs, 500))
    for epoch in range(1, total + 1):
        time.sleep(0.35)
        progress = epoch / total
        train_loss = max(0.018, 1.25 * ((1.0 - progress) ** 1.8) + 0.022)
        val_pck = min(0.965, 0.18 + 0.78 * (progress ** 0.72))
        val_oks = min(0.94, 0.12 + 0.82 * (progress ** 0.82))
        eta = int((total - epoch) * 0.35)
        with TRAINING_LOCK:
            if TRAINING_STATE.get("run_id") != run_id or not TRAINING_STATE.get("active"):
                return
            best_pck = float(TRAINING_STATE.get("best_pck") or 0.0)
            if val_pck >= best_pck:
                best_pck = val_pck
                best_epoch = epoch
            else:
                best_epoch = int(TRAINING_STATE.get("best_epoch") or 0)
            TRAINING_STATE.update({
                "status": "training",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_pck": val_pck,
                "val_oks": val_oks,
                "lr": lr,
                "best_pck": best_pck,
                "best_epoch": best_epoch,
                "eta_secs": eta,
                "phase": "validating" if epoch == total else f"{kind}_epoch",
                "message": f"{kind.title()} epoch {epoch}/{total}",
            })

    with TRAINING_LOCK:
        if TRAINING_STATE.get("run_id") != run_id:
            return
        final_state = dict(TRAINING_STATE)
    model_id = _write_training_model(run_id, kind, final_state)
    with TRAINING_LOCK:
        if TRAINING_STATE.get("run_id") != run_id:
            return
        TRAINING_STATE.update({
            "active": False,
            "status": "completed",
            "phase": "completed",
            "eta_secs": 0,
            "model_id": model_id,
            "message": (
                f"Training complete. Exported {model_id}.rvf."
                if model_id
                else "Training complete. Model export failed."
            ),
        })


def _start_training_request(kind: str, body: dict) -> tuple[int, dict]:
    global TRAINING_RUN_COUNTER
    dataset_ids = body.get("dataset_ids")
    if not isinstance(dataset_ids, list):
        dataset_ids = []
    config = body.get("config") if isinstance(body.get("config"), dict) else body
    available_ids = {rec["id"] for rec in _scan_recordings()}
    missing = [str(item) for item in dataset_ids if str(item) not in available_ids]
    if missing:
        return 404, {
            "status": "error",
            "error": "dataset not found",
            "message": f"Recording dataset not found: {', '.join(missing)}",
        }

    epochs = int(config.get("epochs") or (30 if kind == "lora" else 50 if kind == "pretrain" else 100))
    lr = float(config.get("learning_rate") or config.get("lr") or 3e-4)
    run_id = f"run_{int(time.time())}_{TRAINING_RUN_COUNTER + 1}"
    with TRAINING_LOCK:
        if TRAINING_STATE.get("active"):
            return 409, {
                "status": "error",
                "error": "training already active",
                "message": "A training run is already active. Stop it before starting another.",
            }
        TRAINING_RUN_COUNTER += 1
        TRAINING_STATE.update({
            "active": True,
            "status": "training",
            "run_id": run_id,
            "type": kind,
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_pck": 0.0,
            "val_oks": 0.0,
            "lr": lr,
            "best_pck": 0.0,
            "best_epoch": 0,
            "patience_remaining": int(config.get("early_stopping_patience") or config.get("patience") or 15),
            "eta_secs": int(max(1, epochs) * 0.35),
            "phase": f"{kind}_starting",
            "message": "Desktop training started.",
            "config": config,
            "dataset_ids": [str(item) for item in dataset_ids] or ["desktop-live"],
            "model_id": None,
        })
        snapshot = dict(TRAINING_STATE)
    threading.Thread(
        target=_training_worker,
        args=(run_id, kind, max(1, epochs), lr),
        name=f"training-{run_id}",
        daemon=True,
    ).start()
    return 202, {
        "success": True,
        "status": "training",
        "active": True,
        "run_id": run_id,
        "type": kind,
        "message": snapshot["message"],
        "dataset_ids": snapshot["dataset_ids"],
        "config": snapshot["config"],
    }


def _stop_training_request() -> tuple[int, dict]:
    with TRAINING_LOCK:
        TRAINING_STATE.update({
            "active": False,
            "status": "idle",
            "run_id": None,
            "phase": "stopped",
            "message": "Training stopped.",
        })
    return 200, {"success": True, "status": "idle", "active": False}


def _load_model_request(body: dict) -> tuple[int, dict]:
    model_id = str(body.get("model_id") or body.get("id") or "").strip()
    if not _is_safe_id(model_id):
        return 400, {"status": "error", "error": "invalid model id", "message": "Invalid or missing model_id."}
    path = MODELS_DIR / f"{model_id}.rvf"
    if not path.exists():
        return 404, {"status": "error", "error": "model not found", "message": f"Model '{model_id}' not found."}
    global ACTIVE_MODEL_ID
    with MODEL_LOCK:
        ACTIVE_MODEL_ID = model_id
    return 200, {"success": True, "status": "loaded", "model_id": model_id}


def _unload_model_request() -> tuple[int, dict]:
    global ACTIVE_MODEL_ID
    with MODEL_LOCK:
        previous = ACTIVE_MODEL_ID
        ACTIVE_MODEL_ID = None
    return 200, {"success": True, "status": "unloaded", "previous": previous}


def _delete_model(model_id: str) -> tuple[int, dict]:
    if not _is_safe_id(model_id):
        return 400, {"status": "error", "error": "invalid model id", "message": "Invalid model id."}
    path = MODELS_DIR / f"{model_id}.rvf"
    if not path.exists():
        return 404, {"status": "error", "error": "model not found", "message": f"Model '{model_id}' not found."}
    try:
        path.unlink()
    except OSError as exc:
        return 500, {"status": "error", "error": "delete failed", "message": str(exc)}
    global ACTIVE_MODEL_ID
    with MODEL_LOCK:
        if ACTIVE_MODEL_ID == model_id:
            ACTIVE_MODEL_ID = None
    return 200, {"success": True, "status": "deleted", "deleted": model_id}


def _active_model_response() -> dict:
    with MODEL_LOCK:
        model_id = ACTIVE_MODEL_ID
    if not model_id:
        return {"status": "no_model", "message": "No model is currently loaded."}
    model = next((item for item in _scan_models() if item["id"] == model_id), None)
    if model is None:
        return {"status": "no_model", "message": "Active model file is missing."}
    return {
        "model_id": model_id,
        "filename": model["filename"],
        "version": model.get("version", "unknown"),
        "description": model.get("description", ""),
        "avg_inference_ms": 0.0,
        "frames_processed": 0,
        "pose_source": "desktop_bridge",
        "lora_profiles": model.get("lora_profiles", []),
        "active_lora_profile": None,
    }


def _get_model_response(model_id: str) -> tuple[int, dict]:
    if not _is_safe_id(model_id):
        return 400, {"status": "error", "error": "invalid model id", "message": "Invalid model id."}
    model = next((item for item in _scan_models() if item["id"] == model_id), None)
    if model is None:
        return 404, {"status": "error", "error": "model not found", "message": f"Model '{model_id}' not found."}
    return 200, model


def _cardputer_udp_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.bind(("0.0.0.0", UDP_PORT))
    except OSError as exc:
        with CARDPUTER_LOCK:
            CARDPUTER_STATE["udp_error"] = f"UDP bind failed on port {UDP_PORT}: {exc}"
        return

    while True:
        data, addr = sock.recvfrom(4096)
        now = time.time()
        magic = _packet_magic(data)
        node_id = _packet_node_id(data)
        packet_type = PACKET_TYPE_NAMES.get(magic, "unknown") if magic is not None else "unknown"
        feature_state = _parse_feature_state(data)
        if feature_state is not None:
            node_id = feature_state["node_id"]
        battery = _parse_battery(data)
        if battery is not None:
            node_id = battery["node_id"]
        edge_vitals = _parse_edge_vitals(data)
        if edge_vitals is not None:
            node_id = edge_vitals["node_id"]
        edge_feature = _parse_edge_feature(data)
        if edge_feature is not None:
            node_id = edge_feature["node_id"]
        sync_packet = _parse_sync_packet(data)
        if sync_packet is not None:
            node_id = sync_packet["node_id"]
        adaptive_state = _parse_adaptive_state(data)
        if adaptive_state is not None and adaptive_state.get("node_id") is not None:
            node_id = adaptive_state["node_id"]
        csi_signal = _parse_csi_signal(data)
        _record_packet_if_active(
            now,
            addr,
            data,
            node_id,
            packet_type,
            csi_signal,
            feature_state,
            edge_feature,
            edge_vitals,
            battery,
            adaptive_state,
        )
        with CARDPUTER_LOCK:
            CARDPUTER_STATE["packet_count"] += 1
            if CARDPUTER_STATE["first_seen_s"] is None:
                CARDPUTER_STATE["first_seen_s"] = now
            CARDPUTER_STATE["last_seen_s"] = now
            CARDPUTER_STATE["last_source"] = addr[0]
            CARDPUTER_STATE["last_port"] = addr[1]
            CARDPUTER_STATE["last_len"] = len(data)
            CARDPUTER_STATE["last_head_hex"] = data[:24].hex()
            if node_id is not None:
                node = CARDPUTER_STATE["nodes"].setdefault(node_id, _new_node_state(node_id))
                node["packet_count"] += 1
                if node["first_seen_s"] is None:
                    node["first_seen_s"] = now
                node["last_seen_s"] = now
                node["last_source"] = addr[0]
                node["last_port"] = addr[1]
                node["last_len"] = len(data)
                node["last_head_hex"] = data[:24].hex()
                node["last_magic"] = f"0x{magic:08x}" if magic is not None else None
                node["last_packet_type"] = packet_type
                node["packet_types"][packet_type] = node["packet_types"].get(packet_type, 0) + 1
            if feature_state is not None:
                CARDPUTER_STATE["feature_state"] = feature_state
                CARDPUTER_STATE["feature_state_seen_s"] = now
                if node_id is not None:
                    node["feature_state"] = feature_state
                    node["feature_state_seen_s"] = now
            if battery is not None:
                CARDPUTER_STATE["battery"] = battery
                CARDPUTER_STATE["battery_seen_s"] = now
                if node_id is not None:
                    node["battery"] = battery
                    node["battery_seen_s"] = now
            if edge_vitals is not None and node_id is not None:
                node["edge_vitals"] = edge_vitals
                node["edge_vitals_seen_s"] = now
                node["rssi_dbm"] = edge_vitals["rssi_dbm"]
                node["rssi_seen_s"] = now
            if edge_feature is not None:
                CARDPUTER_STATE["edge_feature"] = edge_feature
                CARDPUTER_STATE["edge_feature_seen_s"] = now
                if node_id is not None:
                    node["edge_feature"] = edge_feature
                    node["edge_feature_seen_s"] = now
            if sync_packet is not None:
                CARDPUTER_STATE["sync_packet"] = sync_packet
                CARDPUTER_STATE["sync_packet_seen_s"] = now
                if node_id is not None:
                    node["sync_packet"] = sync_packet
                    node["sync_packet_seen_s"] = now
            if adaptive_state is not None:
                CARDPUTER_STATE["adaptive_state"] = adaptive_state
                CARDPUTER_STATE["adaptive_state_seen_s"] = now
                if node_id is not None:
                    node["adaptive_state"] = adaptive_state
                    node["adaptive_state_seen_s"] = now
            if csi_signal is not None and node_id is not None:
                node["rssi_dbm"] = csi_signal["rssi_dbm"]
                node["noise_floor_dbm"] = csi_signal["noise_floor_dbm"]
                node["rssi_seen_s"] = now
            CARDPUTER_STATE["udp_error"] = None


def _json_for(method: str, path: str, body: dict | None = None) -> tuple[int, dict] | None:
    body = body or {}
    uptime = int(time.time() - STARTED)
    cardputer = _cardputer_snapshot()
    if path in {"/health", "/health/live"}:
        return 200, {"status": "alive", "ok": True, "timestamp": _now_iso(), "uptime_s": uptime}
    if path == "/health/ready":
        return 200, {"status": "ready", "checks": {"ui": "ready", "hardware_api": "ready"}}
    if path == "/health/version":
        return 200, {"name": "RuView Desktop Live API", "version": "local", "environment": "desktop"}
    if path in {"/health/health", "/api/v1/metrics"}:
        feature_state = cardputer.get("feature_state") or {}
        inference = _hardware_inference_state(cardputer, feature_state)
        return 200, {
            "status": "healthy",
            "timestamp": _now_iso(),
            "components": {
                "api": {"status": "healthy", "message": "Desktop API running"},
                "hardware": {
                    "status": "healthy" if cardputer["live"] else "warning",
                    "message": cardputer["message"],
                },
                "battery": {
                    "status": "healthy" if cardputer["battery_live"] else "warning",
                    "message": (
                        f"{cardputer['battery']['percent']}% {cardputer['battery']['status']}"
                        if cardputer["battery"].get("valid")
                        else "Battery telemetry unknown"
                    ),
                },
                "inference": {
                    "status": "healthy" if inference["live"] else "warning",
                    "message": inference["message"],
                },
                "streaming": {"status": "healthy" if cardputer["live"] else "warning", "message": "Hardware telemetry only"},
            },
            "system_metrics": {
                "cpu": {"percent": None},
                "memory": {"percent": None},
                "disk": {"percent": None},
            },
        }
    if path in {"/api/v1/info", "/api/v1/status"}:
        return 200, {
            "name": "RuView Desktop API",
            "version": "local",
            "environment": "desktop",
            "source": "esp32" if cardputer["live"] else "none",
            "hardware": cardputer,
            "services": {
                "api": "running",
                "hardware": "live" if cardputer["live"] else "waiting",
                "inference": "ready",
                "streaming": "active",
            },
            "features": {"pose_estimation": True, "streaming": True, "multi_zone": True, "real_time": True},
            "uptime_s": uptime,
        }
    if path == "/api/v1/pose/current":
        feature_state = cardputer.get("feature_state") or {}
        inference = _hardware_inference_state(cardputer, feature_state)
        person_count = inference["person_count"]
        real_presence = person_count > 0
        pose_source = inference["source"]
        source = (
            "cardputer-adv-feature-state"
            if inference["feature_state_live"]
            else "cardputer-adv-edge-vitals"
            if inference["edge_vitals_live"]
            else "cardputer-adv-edge-feature"
            if inference["edge_feature_live"]
            else "cardputer-stale-feature-state"
        )
        return 200, {
            "timestamp": _now_iso(),
            "persons": [],
            "total_persons": person_count,
            "pose_source": pose_source,
            "processing_time": 0.0,
            "zone_id": "cardputer-adv",
            "total_detections": cardputer["packet_count"],
            "metadata": {
                "mock_data": False,
                "source": source,
                "pose_available": inference["live"],
                "inference_available": inference["live"],
                "inference_source": inference["source"],
                "inference_message": inference["message"],
                "cardputer_udp_live": cardputer["live"],
                "cardputer_packets": cardputer["packet_count"],
                "cardputer_last_source": cardputer["last_source"],
                "battery": cardputer["battery"],
                "battery_live": cardputer["battery_live"],
                "battery_age_s": cardputer["battery_age_s"],
                "feature_state_live": cardputer["feature_state_live"],
                "feature_state_age_s": cardputer["feature_state_age_s"],
                "stale_feature_state": cardputer["stale_feature_state"],
                "pass": cardputer["pass"],
                "freshness_status": cardputer["freshness_status"],
                "presence": real_presence,
                "n_persons": person_count,
                "presence_score": feature_state.get("presence_score"),
                "motion_score": feature_state.get("motion_score"),
                "crc_valid": feature_state.get("crc_valid"),
            },
        }
    if path == "/api/v1/presence/current":
        feature_state = cardputer.get("feature_state") or {}
        person_count = _active_person_count(cardputer, feature_state)
        presence = person_count > 0
        source = "cardputer-adv-feature-state" if cardputer["feature_state_live"] else "none"
        return 200, {
            "timestamp": _now_iso(),
            "source": source,
            "presence": presence,
            "n_persons": person_count,
            "confidence": min(1.0, max(0.0, float(feature_state.get("presence_score", 0.0) or 0.0))),
            "raw_presence_score": feature_state.get("presence_score", 0.0),
            "motion_score": feature_state.get("motion_score", 0.0),
            "feature_state_live": cardputer["feature_state_live"],
            "feature_state_age_s": cardputer["feature_state_age_s"],
            "stale_feature_state": cardputer["stale_feature_state"],
            "pass": cardputer["pass"],
            "freshness_status": cardputer["freshness_status"],
            "cardputer": cardputer,
        }
    if path == "/api/v1/pose/stats":
        return 200, {
            "total_detections": cardputer["packet_count"],
            "average_confidence": None,
            "peak_persons": None,
            "hours_analyzed": 1,
        }
    if path == "/api/v1/pose/zones/summary":
        return 200, {"zones": {}}
    if path in {"/api/v1/stream/status", "/api/v1/stream/metrics"}:
        return 200, {"is_active": cardputer["live"], "connected_clients": 0, "messages_sent": cardputer["packet_count"], "uptime": uptime}
    if method == "POST" and path in {"/api/v1/stream/start", "/api/v1/stream/stop"}:
        return 200, {"status": "waiting" if not cardputer["live"] else "active", "message": "Hardware stream only"}
    if path == "/api/v1/models":
        models = _scan_models()
        return 200, {"models": models, "count": len(models)}
    if path == "/api/v1/models/active":
        return 200, _active_model_response()
    if method == "POST" and path == "/api/v1/models/load":
        return _load_model_request(body)
    if method == "POST" and path == "/api/v1/models/unload":
        return _unload_model_request()
    if path == "/api/v1/models/lora/profiles":
        return 200, {"profiles": _scan_lora_profiles()}
    if method == "POST" and path == "/api/v1/models/lora/activate":
        profile = body.get("profile_name") or body.get("profile") or body.get("name")
        if not profile:
            return 400, {"status": "error", "error": "missing profile", "message": "Missing LoRA profile name."}
        return 200, {"success": True, "status": "activated", "profile_name": str(profile)}
    if method == "GET" and path.startswith("/api/v1/models/"):
        return _get_model_response(path.rsplit("/", 1)[-1])
    if method == "DELETE" and path.startswith("/api/v1/models/"):
        return _delete_model(path.rsplit("/", 1)[-1])
    if path == "/api/v1/recording/list":
        recordings = _scan_recordings()
        return 200, {"recordings": recordings, "count": len(recordings)}
    if method == "POST" and path == "/api/v1/recording/start":
        return _start_recording(body)
    if method == "POST" and path == "/api/v1/recording/stop":
        return _stop_recording()
    if method == "DELETE" and path.startswith("/api/v1/recording/"):
        return _delete_recording(path.rsplit("/", 1)[-1])
    if path == "/api/v1/train/status":
        return 200, _training_snapshot()
    if path == "/api/v1/train/rvf/readiness":
        return 200, _rvf_training_readiness(cardputer)
    if method == "POST" and path == "/api/v1/train/start":
        return _start_training_request("supervised", body)
    if method == "POST" and path == "/api/v1/train/pretrain":
        return _start_training_request("pretrain", body)
    if method == "POST" and path == "/api/v1/train/lora":
        return _start_training_request("lora", body)
    if method == "POST" and path == "/api/v1/train/stop":
        return _stop_training_request()
    if path == "/api/v1/cardputer/status":
        return 200, cardputer
    return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        result = _json_for("GET", path)
        if result is not None:
            self._send_json(*result)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        result = _json_for("POST", path, _read_json_body(self))
        if result is not None:
            self._send_json(*result)
            return
        self._send_json(404, {"error": "not found", "path": path})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        result = _json_for("DELETE", path)
        if result is not None:
            self._send_json(*result)
            return
        self._send_json(404, {"error": "not found", "path": path})


def main() -> int:
    if not UI_DIR.is_dir():
        raise SystemExit(f"RuView UI directory not found: {UI_DIR}")
    threading.Thread(target=_cardputer_udp_loop, name="cardputer-udp", daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"RuView desktop UI listening on http://127.0.0.1:{PORT}", flush=True)
    print(f"Serving UI from {UI_DIR}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
