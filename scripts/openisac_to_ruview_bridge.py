#!/usr/bin/env python3
"""Bridge OpenISAC sensing UDP frames into RuView rf-direct JSON frames.

The bridge can run without USRP hardware in ``--demo`` mode. In live mode it
listens for OpenISAC chunked sensing payloads, decodes dense range-Doppler or
metadata sidecar frames, summarizes them, and forwards compact JSON to RuView's
``--source rf-direct`` UDP input.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import socket
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


CTRL_HEADER = b"CTRL"
PARAMS_COMMAND = b"PARM"
READY_COMMAND = b"RDY "

HEADER_SIZE = 12
MAX_DATAGRAM_SIZE = 65535
METADATA_CHUNK_FLAG = 0x80000000

DEFAULT_MAX_CHUNKS = 4096
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PARTIAL_FRAMES = 32
DEFAULT_PARTIAL_TTL_SECONDS = 2.0
DEFAULT_MAX_PENDING_PAIRS = 32
DEFAULT_PAIR_TTL_SECONDS = 2.0

COMPACT_MAGIC_VERSION = 0x43534D31  # "CSM1"
AGGREGATE_MAGIC_VERSION = 0x41534731  # "ASG1"

FRAME_FORMAT_DENSE_CHANNEL_BUFFER = 0
FRAME_FORMAT_COMPACT_RAW = 1
FRAME_FORMAT_DENSE_RANGE_DOPPLER = 2
FRAME_FORMAT_COMPACT_SPARSE = 3

FLAG_COMPACT_MASK = 1 << 0
FLAG_BACKEND_SENSING_PROCESSING = 1 << 6
FLAG_SENSING_METADATA_SIDECAR = 1 << 7

WIRE_DATA_FORMAT_COMPLEX_FLOAT32 = 0
WIRE_DATA_FORMAT_COMPLEX_FLOAT16 = 1

PARAMS_PACKET_STRUCT_V1 = struct.Struct("!4s4s11I")
PARAMS_PACKET_STRUCT_V3 = struct.Struct("!4s4s13I")
PARAMS_PACKET_STRUCT_V4 = struct.Struct("!4s4s14I")
PARAMS_PACKET_STRUCT = struct.Struct("!4s4s17I")
REQUEST_PACKET_STRUCT = struct.Struct("!4s4si")
COMPACT_HEADER_STRUCT = struct.Struct("!IIIQ")
AGGREGATE_HEADER_STRUCT = struct.Struct("!IIIIQ")
SENSING_METADATA_HEADER_STRUCT = struct.Struct("<4s11I9fQ")
AGGREGATE_METADATA_HEADER_STRUCT = struct.Struct("<4sIIIQ")
SENSING_CLUSTER_DTYPE = np.dtype(
    [
        ("peak_doppler_idx", "<i4"),
        ("peak_range_idx", "<i4"),
        ("peak_strength_db", "<f4"),
        ("cluster_size", "<u4"),
        ("centroid_doppler_idx", "<f4"),
        ("centroid_range_idx", "<f4"),
    ]
)


@dataclass(frozen=True)
class ViewerRuntimeParams:
    version: int = 0
    flags: int = 0
    frame_format: int = FRAME_FORMAT_DENSE_CHANNEL_BUFFER
    wire_rows: int = 100
    wire_cols: int = 1024
    active_rows: int = 100
    active_cols: int = 1024
    frame_symbol_period: int = 100
    range_fft_size: int = 1024
    doppler_fft_size: int = 100
    compact_mask_hash: int = 0
    wire_data_format: int = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
    stream_channel_count: int = 1
    stream_channel_mask: int = 1
    backend_os_rank_percent: float = 75.0
    backend_os_suppress_doppler: int = 2
    backend_os_suppress_range: int = 2

    def wire_complex_bytes(self) -> int:
        return 4 if self.wire_data_format == WIRE_DATA_FORMAT_COMPLEX_FLOAT16 else 8

    def is_compact_raw(self) -> bool:
        return self.frame_format == FRAME_FORMAT_COMPACT_RAW


@dataclass(frozen=True)
class DecodedSensingFrame:
    frame_id: int
    matrix: np.ndarray
    compact_mask_hash: int = 0
    used_compact_header: bool = False


@dataclass(frozen=True)
class DecodedSensingMetadata:
    frame_id: int
    cfar_points: np.ndarray
    cfar_hits: int
    cfar_shown_hits: int
    cfar_stats: dict
    target_clusters: list[dict]
    md_spectrum: Optional[np.ndarray] = None
    md_extent: Optional[list[float]] = None


@dataclass(frozen=True)
class CompletedPayload:
    frame_id: int
    payload: bytes
    is_metadata: bool


@dataclass
class _PartialFrame:
    total_chunks: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    received: int = 0
    is_metadata: bool = False
    total_bytes: int = 0
    updated_at: float = 0.0


@dataclass
class AssemblerStats:
    accepted_datagrams: int = 0
    rejected_datagrams: int = 0
    duplicate_chunks: int = 0
    expired_frames: int = 0
    evicted_frames: int = 0
    configuration_resets: int = 0
    discarded_on_configuration_reset: int = 0


@dataclass
class PairingStats:
    paired_frames: int = 0
    pair_timeouts: int = 0
    evicted_pairs: int = 0
    duplicate_payloads: int = 0
    duplicate_frames: int = 0
    out_of_order_frames: int = 0
    configuration_resets: int = 0
    discarded_on_configuration_reset: int = 0


@dataclass
class _PendingPair:
    raw: Optional[dict] = None
    metadata: Optional[dict] = None
    updated_at: float = 0.0


@dataclass
class BridgeStats:
    received_datagrams: int = 0
    completed_payloads: int = 0
    forwarded_frames: int = 0
    malformed_datagrams: int = 0
    decode_errors: int = 0


class FrameRecorder:
    """Optional recorder for forwarded JSONL frames and raw OpenISAC payloads."""

    def __init__(
        self,
        jsonl_path: Optional[str | Path] = None,
        raw_dir: Optional[str | Path] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._jsonl_file = None
        self._raw_dir = Path(raw_dir) if raw_dir else None
        self.run_id = run_id or f"{time.time_ns()}-{uuid.uuid4().hex}"
        self._raw_run_dir = self._raw_dir / self.run_id if self._raw_dir else None
        self._manifest_file = None
        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = path.open("a", encoding="utf-8")
        if self._raw_run_dir:
            self._raw_run_dir.mkdir(parents=True, exist_ok=True)
            self._manifest_file = (self._raw_run_dir / "manifest.jsonl").open("a", encoding="utf-8")

    def record_frame(self, frame: dict) -> None:
        if self._jsonl_file is None:
            return
        self._jsonl_file.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        self._jsonl_file.write("\n")
        self._jsonl_file.flush()

    def record_payload(
        self,
        completed: CompletedPayload,
        *,
        sender: tuple[str, int],
        config_epoch: int,
        config_hash: str,
    ) -> Optional[Path]:
        if self._raw_run_dir is None:
            return None
        kind = "metadata" if completed.is_metadata else "raw"
        hash_token = config_hash.removeprefix("sha256:")[:12]
        sender_text = f"{sender[0]}:{sender[1]}"
        sender_token = "".join(
            character if character.isalnum() or character in {".", "-", "_"} else "_"
            for character in sender_text
        )
        evidence_dir = (
            self._raw_run_dir
            / f"epoch_{int(config_epoch):06d}_{hash_token}"
            / f"sender_{sender_token}"
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{completed.frame_id:09d}_{kind}"
        collision_index = 0
        while True:
            suffix = "" if collision_index == 0 else f"_collision_{collision_index:03d}"
            path = evidence_dir / f"{stem}{suffix}.bin"
            try:
                with path.open("xb") as payload_file:
                    payload_file.write(completed.payload)
                break
            except FileExistsError:
                collision_index += 1

        manifest_entry = {
            "run_id": self.run_id,
            "config_epoch": int(config_epoch),
            "config_hash": config_hash,
            "sender": sender_text,
            "frame_id": int(completed.frame_id),
            "kind": kind,
            "collision_index": collision_index,
            "payload_bytes": len(completed.payload),
            "payload_sha256": hashlib.sha256(completed.payload).hexdigest(),
            "relative_path": path.relative_to(self._raw_run_dir).as_posix(),
        }
        assert self._manifest_file is not None
        self._manifest_file.write(
            json.dumps(manifest_entry, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        )
        self._manifest_file.write("\n")
        self._manifest_file.flush()
        return path

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._manifest_file is not None:
            self._manifest_file.close()
            self._manifest_file = None


class FrameAssembler:
    """Reassemble OpenISAC UDP payload chunks.

    Each OpenISAC datagram starts with ``!III``: frame id, total chunks, chunk
    id. The high bit of total chunks marks metadata sidecar chunks.
    """

    def __init__(
        self,
        *,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_partial_frames: int = DEFAULT_MAX_PARTIAL_FRAMES,
        partial_ttl_seconds: float = DEFAULT_PARTIAL_TTL_SECONDS,
    ) -> None:
        if min(max_chunks, max_payload_bytes, max_partial_frames) <= 0:
            raise ValueError("frame assembler limits must be positive")
        if partial_ttl_seconds <= 0.0:
            raise ValueError("partial frame TTL must be positive")
        self.max_chunks = int(max_chunks)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_partial_frames = int(max_partial_frames)
        self.partial_ttl_seconds = float(partial_ttl_seconds)
        self.stats = AssemblerStats()
        self._frames: dict[tuple[tuple[str, int], int, bool], _PartialFrame] = {}

    @property
    def partial_frame_count(self) -> int:
        return len(self._frames)

    def expire(self, *, now: Optional[float] = None) -> None:
        current = time.monotonic() if now is None else float(now)
        expired = [
            key
            for key, frame in self._frames.items()
            if current - frame.updated_at >= self.partial_ttl_seconds
        ]
        for key in expired:
            del self._frames[key]
        self.stats.expired_frames += len(expired)

    def _evict_oldest(self) -> None:
        if not self._frames:
            return
        oldest_key = min(self._frames, key=lambda key: self._frames[key].updated_at)
        del self._frames[oldest_key]
        self.stats.evicted_frames += 1

    def reset_for_configuration(self) -> int:
        discarded = len(self._frames)
        self._frames.clear()
        self.stats.configuration_resets += 1
        self.stats.discarded_on_configuration_reset += discarded
        return discarded

    def add_datagram(
        self,
        data: bytes,
        *,
        sender: Optional[tuple[str, int]] = None,
        now: Optional[float] = None,
    ) -> Optional[CompletedPayload]:
        current = time.monotonic() if now is None else float(now)
        sender_key = sender or ("unknown", 0)
        self.expire(now=current)
        if len(data) < HEADER_SIZE:
            self.stats.rejected_datagrams += 1
            return None
        frame_id, total_chunks_raw, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
        is_metadata = bool(total_chunks_raw & METADATA_CHUNK_FLAG)
        total_chunks = int(total_chunks_raw & ~METADATA_CHUNK_FLAG)
        chunk = data[HEADER_SIZE:]
        if (
            total_chunks <= 0
            or total_chunks > self.max_chunks
            or chunk_id >= total_chunks
            or len(chunk) > self.max_payload_bytes
        ):
            self.stats.rejected_datagrams += 1
            return None

        key = (sender_key, int(frame_id), is_metadata)
        frame = self._frames.get(key)
        if frame is None:
            if len(self._frames) >= self.max_partial_frames:
                self._evict_oldest()
            frame = _PartialFrame(
                total_chunks=total_chunks,
                is_metadata=is_metadata,
                updated_at=current,
            )
            self._frames[key] = frame
        elif frame.total_chunks != total_chunks:
            self.stats.rejected_datagrams += 1
            return None

        if chunk_id in frame.chunks:
            self.stats.duplicate_chunks += 1
            return None

        if frame.total_bytes + len(chunk) > self.max_payload_bytes:
            del self._frames[key]
            self.stats.rejected_datagrams += 1
            return None

        frame.chunks[int(chunk_id)] = chunk
        frame.received += 1
        frame.total_bytes += len(chunk)
        frame.updated_at = current
        self.stats.accepted_datagrams += 1
        if frame.received != frame.total_chunks:
            return None

        payload = b"".join(frame.chunks[index] for index in range(frame.total_chunks))
        del self._frames[key]
        return CompletedPayload(frame_id=int(frame_id), payload=payload, is_metadata=is_metadata)


class FramePairer:
    """Pair decoded raw and metadata payloads before emitting an observation."""

    def __init__(
        self,
        *,
        params: ViewerRuntimeParams,
        source_instance_id: Optional[str] = None,
        max_pending_pairs: int = DEFAULT_MAX_PENDING_PAIRS,
        pair_ttl_seconds: float = DEFAULT_PAIR_TTL_SECONDS,
    ) -> None:
        if max_pending_pairs <= 0 or pair_ttl_seconds <= 0.0:
            raise ValueError("frame pairing limits must be positive")
        self.params = params
        self.source_instance_id = source_instance_id or uuid.uuid4().hex
        if (
            len(self.source_instance_id) != 32
            or any(character not in "0123456789abcdef" for character in self.source_instance_id)
        ):
            raise ValueError("source_instance_id must be 32 lowercase hexadecimal characters")
        self.config_epoch = 0
        self.max_pending_pairs = int(max_pending_pairs)
        self.pair_ttl_seconds = float(pair_ttl_seconds)
        self.stats = PairingStats()
        self._pending: dict[tuple[tuple[str, int], int], _PendingPair] = {}
        self._last_forwarded: dict[tuple[str, int], int] = {}

    @property
    def pending_pair_count(self) -> int:
        return len(self._pending)

    def expire(self, *, now: Optional[float] = None) -> None:
        current = time.monotonic() if now is None else float(now)
        expired = [
            key
            for key, pair in self._pending.items()
            if current - pair.updated_at >= self.pair_ttl_seconds
        ]
        for key in expired:
            del self._pending[key]
        self.stats.pair_timeouts += len(expired)

    def _evict_oldest(self) -> None:
        if not self._pending:
            return
        oldest_key = min(self._pending, key=lambda key: self._pending[key].updated_at)
        del self._pending[oldest_key]
        self.stats.evicted_pairs += 1

    def update_params(self, params: ViewerRuntimeParams) -> int:
        discarded = len(self._pending)
        self._pending.clear()
        self.params = params
        self.config_epoch += 1
        self.stats.configuration_resets += 1
        self.stats.discarded_on_configuration_reset += discarded
        return discarded

    def add(
        self,
        summary: dict,
        *,
        sender: Optional[tuple[str, int]] = None,
        now: Optional[float] = None,
        received_at_ns: Optional[int] = None,
    ) -> Optional[dict]:
        current = time.monotonic() if now is None else float(now)
        sender_key = sender or ("unknown", 0)
        self.expire(now=current)
        kind = summary.get("kind")
        if kind not in {"range_doppler", "metadata"}:
            raise ValueError(f"unknown decoded payload kind: {kind!r}")
        frame_id = int(summary["frame_id"])
        key = (sender_key, frame_id)
        pair = self._pending.get(key)
        if pair is None:
            if len(self._pending) >= self.max_pending_pairs:
                self._evict_oldest()
            pair = _PendingPair(updated_at=current)
            self._pending[key] = pair

        attribute = "raw" if kind == "range_doppler" else "metadata"
        if getattr(pair, attribute) is not None:
            self.stats.duplicate_payloads += 1
            return None
        setattr(pair, attribute, summary)
        pair.updated_at = current
        if pair.raw is None or pair.metadata is None:
            return None

        del self._pending[key]
        previous = self._last_forwarded.get(sender_key)
        if previous is not None and frame_id <= previous:
            if frame_id == previous:
                self.stats.duplicate_frames += 1
            else:
                self.stats.out_of_order_frames += 1
            return None

        observation = build_rf_observation(
            pair.raw,
            pair.metadata,
            params=self.params,
            source_instance_id=self.source_instance_id,
            config_epoch=self.config_epoch,
            received_at_ns=time.time_ns() if received_at_ns is None else int(received_at_ns),
        )
        self._last_forwarded[sender_key] = frame_id
        self.stats.paired_frames += 1
        return observation


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_params_request(value: int = 0) -> bytes:
    return REQUEST_PACKET_STRUCT.pack(b"REQ ", PARAMS_COMMAND, int(value))


def parse_params_packet(data: bytes) -> Optional[ViewerRuntimeParams]:
    if len(data) < PARAMS_PACKET_STRUCT_V1.size:
        return None
    if len(data) >= PARAMS_PACKET_STRUCT.size:
        unpacked = PARAMS_PACKET_STRUCT.unpack_from(data)
        (
            header,
            command,
            version,
            flags,
            frame_format,
            wire_rows,
            wire_cols,
            active_rows,
            active_cols,
            frame_symbol_period,
            range_fft_size,
            doppler_fft_size,
            compact_mask_hash,
            wire_data_format,
            stream_channel_count,
            stream_channel_mask,
            os_cfar_rank_percent_x100,
            os_cfar_suppress_doppler,
            os_cfar_suppress_range,
        ) = unpacked
    elif len(data) >= PARAMS_PACKET_STRUCT_V4.size:
        (
            header,
            command,
            version,
            flags,
            frame_format,
            wire_rows,
            wire_cols,
            active_rows,
            active_cols,
            frame_symbol_period,
            range_fft_size,
            doppler_fft_size,
            compact_mask_hash,
            wire_data_format,
            stream_channel_count,
            stream_channel_mask,
        ) = PARAMS_PACKET_STRUCT_V4.unpack_from(data)
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2
    elif len(data) >= PARAMS_PACKET_STRUCT_V3.size:
        (
            header,
            command,
            version,
            flags,
            frame_format,
            wire_rows,
            wire_cols,
            active_rows,
            active_cols,
            frame_symbol_period,
            range_fft_size,
            doppler_fft_size,
            compact_mask_hash,
            stream_channel_count,
            stream_channel_mask,
        ) = PARAMS_PACKET_STRUCT_V3.unpack_from(data)
        wire_data_format = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2
    else:
        (
            header,
            command,
            version,
            flags,
            frame_format,
            wire_rows,
            wire_cols,
            active_rows,
            active_cols,
            frame_symbol_period,
            range_fft_size,
            doppler_fft_size,
            compact_mask_hash,
        ) = PARAMS_PACKET_STRUCT_V1.unpack_from(data)
        wire_data_format = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
        stream_channel_count = 1
        stream_channel_mask = 1
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2

    if header != CTRL_HEADER or command != PARAMS_COMMAND:
        return None

    return ViewerRuntimeParams(
        version=int(version),
        flags=int(flags),
        frame_format=int(frame_format),
        wire_rows=max(1, int(wire_rows)),
        wire_cols=max(1, int(wire_cols)),
        active_rows=max(1, int(active_rows)),
        active_cols=max(1, int(active_cols)),
        frame_symbol_period=max(1, int(frame_symbol_period)),
        range_fft_size=max(1, int(range_fft_size)),
        doppler_fft_size=max(1, int(doppler_fft_size)),
        compact_mask_hash=int(compact_mask_hash),
        wire_data_format=int(wire_data_format),
        stream_channel_count=max(1, int(stream_channel_count)),
        stream_channel_mask=int(stream_channel_mask) or 1,
        backend_os_rank_percent=float(int(os_cfar_rank_percent_x100)) / 100.0,
        backend_os_suppress_doppler=max(0, int(os_cfar_suppress_doppler)),
        backend_os_suppress_range=max(0, int(os_cfar_suppress_range)),
    )


def _decode_wire_complex_payload(payload: bytes, expected_count: int, wire_data_format: int) -> np.ndarray:
    if wire_data_format == WIRE_DATA_FORMAT_COMPLEX_FLOAT16:
        scalar = np.frombuffer(payload, dtype=np.float16)
        if scalar.size != expected_count * 2:
            raise ValueError("complex float16 payload scalar count mismatch")
        pairs = scalar.astype(np.float32, copy=False).reshape((expected_count, 2))
        return (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64, copy=False)

    if len(payload) != expected_count * np.dtype(np.complex64).itemsize:
        raise ValueError("complex float32 payload byte size mismatch")
    return np.frombuffer(payload, dtype=np.complex64)


def decode_sensing_payload(frame_id_hint: int, payload: bytes, params: ViewerRuntimeParams) -> DecodedSensingFrame:
    if len(payload) >= COMPACT_HEADER_STRUCT.size:
        magic, mask_hash, re_count, frame_start = COMPACT_HEADER_STRUCT.unpack_from(payload)
        if magic == COMPACT_MAGIC_VERSION:
            if not params.is_compact_raw():
                raise ValueError("compact payload received while params are not compact raw")
            expected_count = int(params.active_rows) * int(params.active_cols)
            if int(re_count) != expected_count:
                raise ValueError("compact payload RE count mismatch")
            body = payload[COMPACT_HEADER_STRUCT.size:]
            data = _decode_wire_complex_payload(body, expected_count, params.wire_data_format)
            return DecodedSensingFrame(
                frame_id=int(frame_start),
                matrix=data.reshape((int(params.active_rows), int(params.active_cols))),
                compact_mask_hash=int(mask_hash),
                used_compact_header=True,
            )

    expected_count = int(params.wire_rows) * int(params.wire_cols)
    data = _decode_wire_complex_payload(payload, expected_count, params.wire_data_format)
    return DecodedSensingFrame(frame_id=int(frame_id_hint), matrix=data.reshape((params.wire_rows, params.wire_cols)))


def _expand_channel_ids(channel_count: int, channel_mask: int) -> list[int]:
    if channel_mask:
        ids = [bit for bit in range(32) if channel_mask & (1 << bit)]
        if len(ids) == channel_count:
            return ids
    return list(range(channel_count))


def decode_aggregate_sensing_payload(
    frame_id_hint: int,
    payload: bytes,
    params: ViewerRuntimeParams,
) -> tuple[int, list[tuple[int, DecodedSensingFrame]]]:
    if len(payload) < AGGREGATE_HEADER_STRUCT.size:
        raise ValueError("aggregate payload shorter than header")
    magic, channel_count, channel_payload_bytes, channel_mask, frame_start = (
        AGGREGATE_HEADER_STRUCT.unpack_from(payload)
    )
    if int(magic) != AGGREGATE_MAGIC_VERSION:
        raise ValueError("unexpected aggregate payload magic")
    channel_count = int(channel_count)
    channel_payload_bytes = int(channel_payload_bytes)
    if channel_count <= 0 or channel_payload_bytes <= 0:
        raise ValueError("aggregate payload has no channel data")
    expected_size = AGGREGATE_HEADER_STRUCT.size + channel_count * channel_payload_bytes
    if len(payload) != expected_size:
        raise ValueError("aggregate payload byte size mismatch")

    decoded: list[tuple[int, DecodedSensingFrame]] = []
    channel_ids = _expand_channel_ids(channel_count, int(channel_mask))
    offset = AGGREGATE_HEADER_STRUCT.size
    for ch_id in channel_ids:
        ch_payload = payload[offset:offset + channel_payload_bytes]
        ch_frame = decode_sensing_payload(int(frame_id_hint), ch_payload, params)
        if not ch_frame.used_compact_header:
            ch_frame = DecodedSensingFrame(
                frame_id=int(frame_start),
                matrix=ch_frame.matrix,
                compact_mask_hash=ch_frame.compact_mask_hash,
                used_compact_header=False,
            )
        decoded.append((int(ch_id), ch_frame))
        offset += channel_payload_bytes
    return int(frame_start), decoded


def decode_metadata_payload(payload: bytes) -> DecodedSensingMetadata:
    if len(payload) < SENSING_METADATA_HEADER_STRUCT.size:
        raise ValueError("metadata payload shorter than header")

    (
        magic,
        total_bytes,
        flags,
        cfar_point_count,
        cluster_count,
        md_rows,
        md_cols,
        cfar_hits,
        cfar_shown_hits,
        invalid_cells,
        nonfinite_cells,
        nonpositive_cells,
        noise_min,
        noise_max,
        thresh_min,
        thresh_max,
        power_min_db,
        md_t0,
        md_t1,
        md_f0,
        md_f1,
        frame_start,
    ) = SENSING_METADATA_HEADER_STRUCT.unpack_from(payload)

    if magic != b"SMD1":
        raise ValueError(f"unexpected metadata magic {magic!r}")
    if int(total_bytes) != len(payload):
        raise ValueError("metadata payload size mismatch")

    offset = SENSING_METADATA_HEADER_STRUCT.size
    if int(cfar_point_count) > 0:
        point_bytes = int(cfar_point_count) * 8
        point_slice = payload[offset:offset + point_bytes]
        if len(point_slice) != point_bytes:
            raise ValueError("metadata CFAR points truncated")
        cfar_points = np.frombuffer(point_slice, dtype="<i4").reshape((-1, 2)).astype(np.int32, copy=False)
        offset += point_bytes
    else:
        cfar_points = np.empty((0, 2), dtype=np.int32)

    target_clusters: list[dict] = []
    if int(cluster_count) > 0:
        cluster_bytes = int(cluster_count) * SENSING_CLUSTER_DTYPE.itemsize
        cluster_slice = payload[offset:offset + cluster_bytes]
        if len(cluster_slice) != cluster_bytes:
            raise ValueError("metadata clusters truncated")
        clusters = np.frombuffer(cluster_slice, dtype=SENSING_CLUSTER_DTYPE)
        target_clusters = [
            {
                "peak_doppler_idx": int(item["peak_doppler_idx"]),
                "peak_range_idx": int(item["peak_range_idx"]),
                "peak_strength_db": float(item["peak_strength_db"]),
                "cluster_size": int(item["cluster_size"]),
                "centroid_doppler_idx": float(item["centroid_doppler_idx"]),
                "centroid_range_idx": float(item["centroid_range_idx"]),
            }
            for item in clusters
        ]
        offset += cluster_bytes

    md_spectrum = None
    md_extent = None
    total_md_values = int(md_rows) * int(md_cols)
    if total_md_values > 0:
        md_bytes = total_md_values * np.dtype("<f4").itemsize
        md_slice = payload[offset:offset + md_bytes]
        if len(md_slice) != md_bytes:
            raise ValueError("metadata micro-Doppler truncated")
        md_spectrum = np.frombuffer(md_slice, dtype="<f4").reshape((int(md_rows), int(md_cols)))
        md_extent = [float(md_t0), float(md_t1), float(md_f0), float(md_f1)]

    return DecodedSensingMetadata(
        frame_id=int(frame_start),
        cfar_points=cfar_points,
        cfar_hits=int(cfar_hits),
        cfar_shown_hits=int(cfar_shown_hits),
        cfar_stats={
            "noise_min": float(noise_min),
            "noise_max": float(noise_max),
            "thresh_min": float(thresh_min),
            "thresh_max": float(thresh_max),
            "power_min_db": float(power_min_db),
            "invalid_cells": int(invalid_cells),
            "nonfinite_cells": int(nonfinite_cells),
            "nonpositive_cells": int(nonpositive_cells),
            "backend_flags": int(flags),
        },
        target_clusters=target_clusters,
        md_spectrum=md_spectrum,
        md_extent=md_extent,
    )


def decode_aggregate_metadata_payload(
    payload: bytes,
) -> tuple[int, list[tuple[int, DecodedSensingMetadata]]]:
    if len(payload) < AGGREGATE_METADATA_HEADER_STRUCT.size:
        raise ValueError("aggregate metadata payload shorter than header")
    magic, channel_count, channel_mask, _, frame_start = AGGREGATE_METADATA_HEADER_STRUCT.unpack_from(payload)
    if magic != b"ASM1":
        raise ValueError(f"unexpected aggregate metadata magic {magic!r}")
    channel_count = int(channel_count)
    if channel_count <= 0:
        raise ValueError("aggregate metadata has no channels")

    decoded: list[tuple[int, DecodedSensingMetadata]] = []
    channel_ids = _expand_channel_ids(channel_count, int(channel_mask))
    offset = AGGREGATE_METADATA_HEADER_STRUCT.size
    for ch_id in channel_ids:
        if offset + SENSING_METADATA_HEADER_STRUCT.size > len(payload):
            raise ValueError("aggregate metadata truncated before channel header")
        _, total_bytes, *_ = SENSING_METADATA_HEADER_STRUCT.unpack_from(payload, offset)
        total_bytes = int(total_bytes)
        channel_payload = payload[offset:offset + total_bytes]
        if len(channel_payload) != total_bytes:
            raise ValueError("aggregate metadata channel payload truncated")
        decoded.append((int(ch_id), decode_metadata_payload(channel_payload)))
        offset += total_bytes

    return int(frame_start), decoded


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _diagnostic_peaks(power: np.ndarray, count: int) -> list[dict]:
    if power.size == 0:
        return []
    flat = power.reshape(-1)
    top_count = min(max(1, int(count)), flat.size)
    indices = np.argpartition(flat, -top_count)[-top_count:]
    ordered = sorted(indices, key=lambda idx: float(flat[idx]), reverse=True)
    peaks: list[dict] = []
    cols = power.shape[1]
    for idx in ordered:
        row = int(idx // cols)
        col = int(idx % cols)
        strength = float(flat[idx])
        if strength <= 0.0:
            continue
        peaks.append(
            {
                "kind": "unclassified_peak",
                "range_bin": col,
                "doppler_bin": row,
                "strength_db": _db(strength),
            }
        )
    return peaks


def summarize_range_doppler(
    matrix: np.ndarray,
    *,
    frame_id: int,
    params: ViewerRuntimeParams,
    source: str,
    center_freq_hz: Optional[float],
    sample_rate_hz: Optional[float],
    feature_rate_hz: float,
    top_peaks: int = 8,
) -> dict:
    rd = np.asarray(matrix, dtype=np.complex64)
    magnitude = np.abs(rd).astype(np.float64, copy=False)
    if magnitude.size == 0:
        magnitude = np.zeros((1, 1), dtype=np.float64)

    peak = float(np.max(magnitude))
    mean = float(np.mean(magnitude))
    median = float(np.median(magnitude))
    noise = max(median, 1e-12)
    snr_db = 20.0 * math.log10(max(peak, 1e-12) / noise)
    range_profile = np.max(magnitude, axis=0)
    max_range = float(np.max(range_profile)) if range_profile.size else 1.0
    normalized_range = (range_profile / max(max_range, 1e-12)).astype(float)
    return {
        "kind": "range_doppler",
        "source": source,
        "frame_id": int(frame_id),
        "center_freq_hz": center_freq_hz,
        "sample_rate_hz": sample_rate_hz,
        "feature_rate_hz": float(feature_rate_hz),
        "range_doppler": {
            "amplitude": peak,
            "snr_db": max(0.0, snr_db),
            "range_profile": normalized_range.tolist(),
            "peaks": _diagnostic_peaks(magnitude, top_peaks),
            "frame_format": int(params.frame_format),
            "range_fft_size": int(params.range_fft_size),
            "doppler_fft_size": int(params.doppler_fft_size),
            "rd_shape": [int(magnitude.shape[0]), int(magnitude.shape[1])],
            "mean_amplitude": mean,
        },
    }


def metadata_to_ruview_frame(
    metadata: DecodedSensingMetadata,
    *,
    source: str,
    center_freq_hz: Optional[float],
    sample_rate_hz: Optional[float],
    feature_rate_hz: float,
) -> dict:
    candidate_clusters = [
        {
            "range_bin": int(item["peak_range_idx"]),
            "doppler_bin": int(item["peak_doppler_idx"]),
            "strength_db": float(item["peak_strength_db"]),
            "cluster_size": int(item["cluster_size"]),
            "centroid_range_bin": float(item["centroid_range_idx"]),
            "centroid_doppler_bin": float(item["centroid_doppler_idx"]),
        }
        for item in metadata.target_clusters
    ]
    frame = {
        "kind": "metadata",
        "source": source,
        "frame_id": int(metadata.frame_id),
        "center_freq_hz": center_freq_hz,
        "sample_rate_hz": sample_rate_hz,
        "feature_rate_hz": float(feature_rate_hz),
        "cfar": {
            "hits": int(metadata.cfar_hits),
            "shown_hits": int(metadata.cfar_shown_hits),
            "stats": metadata.cfar_stats,
            "candidate_clusters": candidate_clusters,
        },
    }
    if metadata.md_spectrum is not None:
        frame["micro_doppler"] = {
            "rows": int(metadata.md_spectrum.shape[0]),
            "cols": int(metadata.md_spectrum.shape[1]),
            "extent": metadata.md_extent,
            "peak": float(np.max(metadata.md_spectrum)) if metadata.md_spectrum.size else 0.0,
        }
    return frame


def _configuration_hash(params: ViewerRuntimeParams) -> str:
    encoded = json.dumps(asdict(params), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_rf_observation(
    raw: dict,
    metadata: dict,
    *,
    params: ViewerRuntimeParams,
    source_instance_id: str,
    config_epoch: int,
    received_at_ns: int,
) -> dict:
    raw_frame_id = int(raw["frame_id"])
    metadata_frame_id = int(metadata["frame_id"])
    if raw_frame_id != metadata_frame_id:
        raise ValueError("raw and metadata frame IDs do not match")
    if raw.get("source") != metadata.get("source"):
        raise ValueError("raw and metadata sources do not match")

    observation = {
        "schema": "ruview.rf_observation",
        "protocol_version": 2,
        "source": str(raw["source"]),
        "source_instance_id": source_instance_id,
        "config_epoch": int(config_epoch),
        "frame_id": raw_frame_id,
        "sequence": raw_frame_id,
        "source_timestamp_ns": None,
        "received_at_ns": int(received_at_ns),
        "config_hash": _configuration_hash(params),
        "freshness": "fresh",
        "center_freq_hz": raw.get("center_freq_hz"),
        "sample_rate_hz": raw.get("sample_rate_hz"),
        "feature_rate_hz": raw.get("feature_rate_hz"),
        "observation": {
            "range_doppler": raw["range_doppler"],
            "cfar": metadata["cfar"],
            "micro_doppler": metadata.get("micro_doppler"),
        },
    }
    openisac = {}
    if raw.get("openisac"):
        openisac["raw"] = raw["openisac"]
    if metadata.get("openisac"):
        openisac["metadata"] = metadata["openisac"]
    if openisac:
        observation["observation"]["openisac"] = openisac
    return observation


def send_json(sock: socket.socket, target: tuple[str, int], frame: dict) -> None:
    payload = json.dumps(frame, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sock.sendto(payload, target)


def handle_payload(
    completed: CompletedPayload,
    *,
    params: ViewerRuntimeParams,
    source: str,
    center_freq_hz: Optional[float],
    sample_rate_hz: Optional[float],
    feature_rate_hz: float,
) -> dict:
    if completed.is_metadata:
        if len(completed.payload) >= 4 and completed.payload[:4] == b"ASM1":
            frame_id, decoded = decode_aggregate_metadata_payload(completed.payload)
            if not decoded:
                raise ValueError("aggregate metadata decoded to no channels")
            channel_summaries = [
                metadata_to_ruview_frame(
                    metadata,
                    source=f"{source}-ch{ch_id}",
                    center_freq_hz=center_freq_hz,
                    sample_rate_hz=sample_rate_hz,
                    feature_rate_hz=feature_rate_hz,
                )
                for ch_id, metadata in decoded
            ]
            def peak_strength(summary: dict) -> float:
                clusters = summary.get("cfar", {}).get("candidate_clusters", [])
                return max((float(item.get("strength_db", 0.0)) for item in clusters), default=0.0)

            best = max(channel_summaries, key=peak_strength)
            best["source"] = source
            best["frame_id"] = int(frame_id)
            best["openisac"] = {
                **best.get("openisac", {}),
                "aggregate_metadata_channels": [
                    {
                        "channel_id": ch_id,
                        "cfar": summary.get("cfar"),
                        "micro_doppler": summary.get("micro_doppler"),
                    }
                    for (ch_id, _), summary in zip(decoded, channel_summaries)
                ],
            }
            return best
        metadata = decode_metadata_payload(completed.payload)
        return metadata_to_ruview_frame(
            metadata,
            source=source,
            center_freq_hz=center_freq_hz,
            sample_rate_hz=sample_rate_hz,
            feature_rate_hz=feature_rate_hz,
        )
    if len(completed.payload) >= 4 and struct.unpack("!I", completed.payload[:4])[0] == AGGREGATE_MAGIC_VERSION:
        frame_id, decoded = decode_aggregate_sensing_payload(completed.frame_id, completed.payload, params)
        if not decoded:
            raise ValueError("aggregate payload decoded to no channels")
        channel_summaries = [
            summarize_range_doppler(
                frame.matrix,
                frame_id=frame.frame_id,
                params=params,
                source=f"{source}-ch{ch_id}",
                center_freq_hz=center_freq_hz,
                sample_rate_hz=sample_rate_hz,
                feature_rate_hz=feature_rate_hz,
            )
            for ch_id, frame in decoded
        ]
        best = max(
            channel_summaries,
            key=lambda item: float(item.get("range_doppler", {}).get("amplitude", 0.0)),
        )
        best["source"] = source
        best["frame_id"] = int(frame_id)
        best["openisac"] = {
            **best.get("openisac", {}),
            "aggregate_channels": [
                {
                    "channel_id": idx,
                    "range_doppler": summary.get("range_doppler"),
                }
                for idx, summary in zip([ch_id for ch_id, _ in decoded], channel_summaries)
            ],
        }
        return best
    decoded = decode_sensing_payload(completed.frame_id, completed.payload, params)
    return summarize_range_doppler(
        decoded.matrix,
        frame_id=decoded.frame_id,
        params=params,
        source=source,
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        feature_rate_hz=feature_rate_hz,
    )


def _handle_payload(
    completed: CompletedPayload,
    *,
    params: ViewerRuntimeParams,
    source: str,
    center_freq_hz: Optional[float],
    sample_rate_hz: Optional[float],
    feature_rate_hz: float,
) -> dict:
    return handle_payload(
        completed,
        params=params,
        source=source,
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        feature_rate_hz=feature_rate_hz,
    )


def _frame_id_from_path(path: Path, fallback: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if digits:
        return int(digits[-9:])
    return fallback


def replay_payload_files(
    paths: list[str | Path],
    *,
    params: ViewerRuntimeParams,
    source: str,
    center_freq_hz: Optional[float],
    sample_rate_hz: Optional[float],
    feature_rate_hz: float,
    record_jsonl: Optional[str | Path] = None,
) -> list[dict]:
    recorder = FrameRecorder(record_jsonl, None)
    frames: list[dict] = []
    pairer = FramePairer(params=params)
    try:
        for fallback, raw_path in enumerate(paths):
            path = Path(raw_path)
            is_metadata = "metadata" in path.stem.lower()
            completed = CompletedPayload(
                frame_id=_frame_id_from_path(path, fallback),
                payload=path.read_bytes(),
                is_metadata=is_metadata,
            )
            summary = _handle_payload(
                completed,
                params=params,
                source=source,
                center_freq_hz=center_freq_hz,
                sample_rate_hz=sample_rate_hz,
                feature_rate_hz=feature_rate_hz,
            )
            frame = pairer.add(
                summary,
                sender=("replay", 0),
                received_at_ns=path.stat().st_mtime_ns,
            )
            if frame is None:
                continue
            recorder.record_frame(frame)
            frames.append(frame)
    finally:
        recorder.close()
    return frames


def demo_frames(
    rate_hz: float,
    *,
    source_instance_id: Optional[str] = None,
) -> list[dict]:
    frames = []
    instance_id = source_instance_id or uuid.uuid4().hex
    params = ViewerRuntimeParams(
        frame_format=FRAME_FORMAT_DENSE_RANGE_DOPPLER,
        wire_rows=32,
        wire_cols=64,
        range_fft_size=64,
        doppler_fft_size=32,
    )
    for seq in range(3):
        rd = np.zeros((32, 64), dtype=np.complex64)
        rd[16, 4] = 0.25 + 0.0j
        rd[12 + seq, 18 + seq] = 1.0 + 0.0j
        rd[21, 36] = 0.6 + 0.0j
        raw = summarize_range_doppler(
            rd,
            frame_id=seq,
            params=params,
            source="openisac-rd-demo",
            center_freq_hz=3.1e9,
            sample_rate_hz=50e6,
            feature_rate_hz=rate_hz,
        )
        metadata = {
            "kind": "metadata",
            "source": "openisac-rd-demo",
            "frame_id": seq,
            "cfar": {
                "hits": 0,
                "shown_hits": 0,
                "stats": {},
                "candidate_clusters": [],
            },
        }
        frames.append(
            build_rf_observation(
                raw,
                metadata,
                params=params,
                source_instance_id=instance_id,
                config_epoch=0,
                received_at_ns=time.time_ns(),
            )
        )
    return frames


def validate_openisac_bind_host(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(f"OpenISAC bind host must be a loopback IP address: {host!r}") from exc
    if not address.is_loopback:
        raise ValueError(
            "OpenISAC UDP must bind to loopback until source authentication and integrity protection exist"
        )


def bridge_udp_loop(args: argparse.Namespace) -> int:
    validate_openisac_bind_host(args.openisac_host)
    listen_addr = (args.openisac_host, args.openisac_port)
    ruview_target = (args.ruview_host, args.ruview_port)
    params = ViewerRuntimeParams(
        frame_format=args.frame_format,
        wire_rows=args.wire_rows,
        wire_cols=args.wire_cols,
        range_fft_size=args.range_fft_size,
        doppler_fft_size=args.doppler_fft_size,
    )
    assembler = FrameAssembler()
    pairer = FramePairer(params=params)
    stats = BridgeStats()
    recorder = FrameRecorder(args.record_jsonl, args.record_raw_dir)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(listen_addr)
    rx.settimeout(0.5)
    start = time.monotonic()
    last_sender: Optional[tuple[str, int]] = None

    print(
        f"openisac bridge: listening on {listen_addr[0]}:{listen_addr[1]}, "
        f"forwarding to {ruview_target[0]}:{ruview_target[1]}",
        flush=True,
    )

    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            try:
                data, sender = rx.recvfrom(MAX_DATAGRAM_SIZE)
            except socket.timeout:
                assembler.expire()
                pairer.expire()
                continue

            stats.received_datagrams += 1
            last_sender = sender

            if len(data) >= 8 and data[:4] == CTRL_HEADER:
                command = data[4:8]
                parsed = parse_params_packet(data)
                if parsed is not None:
                    params = parsed
                    assembler.reset_for_configuration()
                    pairer.update_params(params)
                    if args.verbose:
                        print(f"openisac bridge: params updated {params}", flush=True)
                elif command == READY_COMMAND:
                    tx.sendto(build_params_request(0), (sender[0], args.control_port))
                continue

            completed = assembler.add_datagram(data, sender=sender)
            if completed is None:
                stats.malformed_datagrams += int(len(data) < HEADER_SIZE)
                continue

            stats.completed_payloads += 1
            recorder.record_payload(
                completed,
                sender=sender,
                config_epoch=pairer.config_epoch,
                config_hash=_configuration_hash(params),
            )
            try:
                summary = _handle_payload(
                    completed,
                    params=params,
                    source=args.source,
                    center_freq_hz=args.center_freq_hz,
                    sample_rate_hz=args.sample_rate_hz,
                    feature_rate_hz=args.feature_rate_hz,
                )
            except Exception as exc:
                stats.decode_errors += 1
                if args.verbose:
                    print(f"openisac bridge: decode error: {exc}", flush=True)
                if last_sender:
                    tx.sendto(build_params_request(0), (last_sender[0], args.control_port))
                continue

            frame = pairer.add(summary, sender=sender)
            if frame is None:
                continue
            send_json(tx, ruview_target, frame)
            recorder.record_frame(frame)
            stats.forwarded_frames += 1
            if args.verbose:
                print(
                    f"seq={frame.get('sequence')} "
                    f"cfar_candidates={len(frame['observation']['cfar'].get('candidate_clusters', []))}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        tx.close()
        recorder.close()

    if args.verbose:
        print(
            f"openisac bridge: stats={stats}, assembler={assembler.stats}, pairing={pairer.stats}",
            flush=True,
        )
    return 0


def run_demo(args: argparse.Namespace) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recorder = FrameRecorder(args.record_jsonl, None)
    target = (args.ruview_host, args.ruview_port)
    period = 1.0 / max(0.1, float(args.feature_rate_hz))
    start = time.monotonic()
    seq_offset = 0
    source_instance_id = uuid.uuid4().hex
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            for frame in demo_frames(
                args.feature_rate_hz,
                source_instance_id=source_instance_id,
            ):
                frame["sequence"] = int(frame["sequence"]) + seq_offset
                frame["frame_id"] = int(frame["frame_id"]) + seq_offset
                send_json(sock, target, frame)
                recorder.record_frame(frame)
                if args.verbose:
                    print(
                        f"demo seq={frame['sequence']} cfar_candidates=0",
                        flush=True,
                    )
                time.sleep(period)
            seq_offset += 3
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        recorder.close()
    return 0


def run_replay(args: argparse.Namespace) -> int:
    params = ViewerRuntimeParams(
        frame_format=args.frame_format,
        wire_rows=args.wire_rows,
        wire_cols=args.wire_cols,
        range_fft_size=args.range_fft_size,
        doppler_fft_size=args.doppler_fft_size,
    )
    frames = replay_payload_files(
        [Path(path) for path in args.replay_payload],
        params=params,
        source=args.source,
        center_freq_hz=args.center_freq_hz,
        sample_rate_hz=args.sample_rate_hz,
        feature_rate_hz=args.feature_rate_hz,
        record_jsonl=args.record_jsonl,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        target = (args.ruview_host, args.ruview_port)
        for frame in frames:
            send_json(sock, target, frame)
            if args.verbose:
                print(
                    f"replay seq={frame.get('sequence')} "
                    f"cfar_candidates={len(frame['observation']['cfar'].get('candidate_clusters', []))}",
                    flush=True,
                )
    finally:
        sock.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="emit synthetic OpenISAC RD summaries")
    parser.add_argument(
        "--replay-payload",
        action="append",
        default=[],
        help="decode a completed raw payload file captured with --record-raw-dir; can be repeated",
    )
    parser.add_argument(
        "--openisac-host",
        default="127.0.0.1",
        help="OpenISAC sensing UDP listen host (loopback only until authenticated transport exists)",
    )
    parser.add_argument("--openisac-port", type=int, default=8888, help="OpenISAC sensing UDP listen port")
    parser.add_argument("--control-port", type=int, default=9999, help="OpenISAC runtime control port")
    parser.add_argument("--ruview-host", default="127.0.0.1", help="RuView rf-direct UDP host")
    parser.add_argument("--ruview-port", type=int, default=5020, help="RuView rf-direct UDP port")
    parser.add_argument("--source", default="openisac-rd", help="source label sent to RuView")
    parser.add_argument("--center-freq-hz", type=float, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    parser.add_argument("--feature-rate-hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means forever")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record-jsonl", help="append forwarded RuView JSON frames to this JSONL file")
    parser.add_argument("--record-raw-dir", help="write completed raw OpenISAC payloads into this directory")
    parser.add_argument("--frame-format", type=int, default=FRAME_FORMAT_DENSE_RANGE_DOPPLER)
    parser.add_argument("--wire-rows", type=int, default=100)
    parser.add_argument("--wire-cols", type=int, default=1024)
    parser.add_argument("--range-fft-size", type=int, default=1024)
    parser.add_argument("--doppler-fft-size", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.feature_rate_hz <= 0:
        raise SystemExit("--feature-rate-hz must be positive")
    if args.replay_payload:
        return run_replay(args)
    if args.demo:
        return run_demo(args)
    return bridge_udp_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
