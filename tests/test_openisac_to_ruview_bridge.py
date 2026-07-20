import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openisac_to_ruview_bridge.py"


def load_bridge():
    spec = importlib.util.spec_from_file_location("openisac_to_ruview_bridge", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_metadata_payload(bridge, frame_id, *, range_bin, doppler_bin, strength_db):
    cfar_points = np.array([[doppler_bin, range_bin]], dtype="<i4")
    clusters = np.array(
        [
            (
                doppler_bin,
                range_bin,
                strength_db,
                2,
                float(doppler_bin) + 0.25,
                float(range_bin) + 0.5,
            )
        ],
        dtype=bridge.SENSING_CLUSTER_DTYPE,
    )
    md = np.array([[0.0, 0.5]], dtype="<f4")
    total_bytes = (
        bridge.SENSING_METADATA_HEADER_STRUCT.size
        + cfar_points.nbytes
        + clusters.nbytes
        + md.nbytes
    )
    header = bridge.SENSING_METADATA_HEADER_STRUCT.pack(
        b"SMD1",
        total_bytes,
        0,
        1,
        1,
        1,
        2,
        3,
        1,
        0,
        0,
        0,
        -60.0,
        -20.0,
        -50.0,
        -10.0,
        -70.0,
        0.0,
        1.0,
        -0.5,
        0.5,
        frame_id,
    )
    return header + cfar_points.tobytes() + clusters.tobytes() + md.tobytes()


def test_summarize_range_doppler_emits_diagnostics_without_inference():
    bridge = load_bridge()
    rd = np.zeros((8, 12), dtype=np.complex64)
    rd[4, 2] = 1.0 + 0.0j
    rd[6, 7] = 4.0 + 0.0j

    frame = bridge.summarize_range_doppler(
        rd,
        frame_id=17,
        params=bridge.ViewerRuntimeParams(
            frame_format=bridge.FRAME_FORMAT_DENSE_RANGE_DOPPLER,
            wire_rows=8,
            wire_cols=12,
            range_fft_size=12,
            doppler_fft_size=8,
        ),
        source="unit-rd",
        center_freq_hz=3.1e9,
        sample_rate_hz=50e6,
        feature_rate_hz=10.0,
    )

    assert frame["source"] == "unit-rd"
    assert frame["frame_id"] == 17
    assert frame["center_freq_hz"] == 3.1e9
    assert frame["sample_rate_hz"] == 50e6
    assert frame["feature_rate_hz"] == 10.0
    assert "motion_energy" not in frame
    assert "targets" not in frame
    diagnostics = frame["range_doppler"]
    assert diagnostics["amplitude"] == 4.0
    assert diagnostics["snr_db"] > 0.0
    assert diagnostics["range_profile"][7] == 1.0
    assert diagnostics["peaks"][0]["kind"] == "unclassified_peak"
    assert diagnostics["peaks"][0]["range_bin"] == 7
    assert diagnostics["peaks"][0]["doppler_bin"] == 6
    assert diagnostics["peaks"][0]["strength_db"] >= diagnostics["peaks"][1]["strength_db"]


def test_static_zero_doppler_peak_never_becomes_motion_or_target():
    bridge = load_bridge()
    rd = np.zeros((8, 12), dtype=np.complex64)
    rd[4, 8] = 1000.0 + 0.0j

    frame = bridge.summarize_range_doppler(
        rd,
        frame_id=18,
        params=bridge.ViewerRuntimeParams(
            frame_format=bridge.FRAME_FORMAT_DENSE_RANGE_DOPPLER,
            wire_rows=8,
            wire_cols=12,
            range_fft_size=12,
            doppler_fft_size=8,
        ),
        source="unit-static",
        center_freq_hz=3.1e9,
        sample_rate_hz=50e6,
        feature_rate_hz=10.0,
    )

    assert "motion_energy" not in frame
    assert "targets" not in frame
    assert frame["range_doppler"]["peaks"][0] == {
        "kind": "unclassified_peak",
        "range_bin": 8,
        "doppler_bin": 4,
        "strength_db": 60.0,
    }


def test_metadata_to_ruview_frame_maps_clusters_and_cfar_stats():
    bridge = load_bridge()
    metadata = bridge.DecodedSensingMetadata(
        frame_id=23,
        cfar_points=np.array([[6, 7], [4, 2]], dtype=np.int32),
        cfar_hits=9,
        cfar_shown_hits=2,
        cfar_stats={"power_min_db": -70.0, "noise_max": -20.0},
        target_clusters=[
            {
                "peak_doppler_idx": 6,
                "peak_range_idx": 7,
                "peak_strength_db": 18.5,
                "cluster_size": 3,
                "centroid_doppler_idx": 5.75,
                "centroid_range_idx": 7.25,
            }
        ],
        md_spectrum=np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32),
        md_extent=[0.0, 1.0, -0.5, 0.5],
    )

    frame = bridge.metadata_to_ruview_frame(
        metadata,
        source="unit-meta",
        center_freq_hz=3.1e9,
        sample_rate_hz=50e6,
        feature_rate_hz=5.0,
    )

    assert frame["source"] == "unit-meta"
    assert frame["frame_id"] == 23
    assert "motion_energy" not in frame
    assert "confidence" not in frame
    assert "targets" not in frame
    assert frame["cfar"]["candidate_clusters"] == [
        {
            "range_bin": 7,
            "doppler_bin": 6,
            "strength_db": 18.5,
            "cluster_size": 3,
            "centroid_range_bin": 7.25,
            "centroid_doppler_bin": 5.75,
        }
    ]
    assert frame["cfar"]["hits"] == 9
    assert frame["micro_doppler"]["rows"] == 2


def test_decode_aggregate_metadata_payload_returns_channel_metadata():
    bridge = load_bridge()
    ch0 = build_metadata_payload(bridge, 51, range_bin=4, doppler_bin=2, strength_db=12.0)
    ch1 = build_metadata_payload(bridge, 51, range_bin=8, doppler_bin=5, strength_db=24.0)
    payload = bridge.AGGREGATE_METADATA_HEADER_STRUCT.pack(b"ASM1", 2, 0b11, 0, 51) + ch0 + ch1

    frame_id, decoded = bridge.decode_aggregate_metadata_payload(payload)

    assert frame_id == 51
    assert [ch_id for ch_id, _ in decoded] == [0, 1]
    assert decoded[0][1].target_clusters[0]["peak_range_idx"] == 4
    assert decoded[1][1].target_clusters[0]["peak_strength_db"] == np.float32(24.0)


def test_handle_aggregate_metadata_payload_uses_strongest_channel():
    bridge = load_bridge()
    ch0 = build_metadata_payload(bridge, 52, range_bin=4, doppler_bin=2, strength_db=12.0)
    ch1 = build_metadata_payload(bridge, 52, range_bin=8, doppler_bin=5, strength_db=24.0)
    payload = bridge.AGGREGATE_METADATA_HEADER_STRUCT.pack(b"ASM1", 2, 0b11, 0, 52) + ch0 + ch1
    completed = bridge.CompletedPayload(frame_id=52, payload=payload, is_metadata=True)

    frame = bridge.handle_payload(
        completed,
        params=bridge.ViewerRuntimeParams(),
        source="unit-asm",
        center_freq_hz=3.1e9,
        sample_rate_hz=50e6,
        feature_rate_hz=10.0,
    )

    assert frame["source"] == "unit-asm"
    assert frame["frame_id"] == 52
    assert frame["cfar"]["candidate_clusters"][0]["range_bin"] == 8
    assert frame["openisac"]["aggregate_metadata_channels"][0]["channel_id"] == 0
    assert frame["openisac"]["aggregate_metadata_channels"][1]["channel_id"] == 1


def test_frame_assembler_reassembles_raw_and_metadata_chunks():
    bridge = load_bridge()
    assembler = bridge.FrameAssembler()
    raw_payload = b"abcdefgh"
    meta_payload = b"metadata"

    assert assembler.add_datagram(struct.pack("!III", 5, 2, 1) + raw_payload[4:]) is None
    completed = assembler.add_datagram(struct.pack("!III", 5, 2, 0) + raw_payload[:4])
    assert completed == bridge.CompletedPayload(frame_id=5, payload=raw_payload, is_metadata=False)

    meta_total = 0x80000000 | 1
    completed_meta = assembler.add_datagram(struct.pack("!III", 6, meta_total, 0) + meta_payload)
    assert completed_meta == bridge.CompletedPayload(frame_id=6, payload=meta_payload, is_metadata=True)


def test_frame_assembler_rejects_attacker_controlled_chunk_count_before_allocation():
    bridge = load_bridge()
    assembler = bridge.FrameAssembler(max_chunks=4)

    packet = struct.pack("!III", 5, 5, 0) + b"x"

    assert assembler.add_datagram(packet, sender=("127.0.0.1", 9000), now=0.0) is None
    assert assembler.partial_frame_count == 0
    assert assembler.stats.rejected_datagrams == 1


def test_frame_assembler_bounds_payload_partial_count_and_ttl():
    bridge = load_bridge()
    assembler = bridge.FrameAssembler(
        max_chunks=4,
        max_payload_bytes=4,
        max_partial_frames=2,
        partial_ttl_seconds=1.0,
    )
    sender = ("127.0.0.1", 9000)

    assert assembler.add_datagram(struct.pack("!III", 1, 2, 0) + b"abc", sender=sender, now=0.0) is None
    assert assembler.add_datagram(struct.pack("!III", 1, 2, 1) + b"de", sender=sender, now=0.1) is None
    assert assembler.stats.rejected_datagrams == 1
    assert assembler.partial_frame_count == 0

    assert assembler.add_datagram(struct.pack("!III", 2, 2, 0) + b"a", sender=sender, now=0.2) is None
    assert assembler.add_datagram(struct.pack("!III", 3, 2, 0) + b"b", sender=sender, now=0.3) is None
    assert assembler.add_datagram(struct.pack("!III", 4, 2, 0) + b"c", sender=sender, now=0.4) is None
    assert assembler.partial_frame_count == 2
    assert assembler.stats.evicted_frames == 1

    assembler.expire(now=2.0)
    assert assembler.partial_frame_count == 0
    assert assembler.stats.expired_frames == 2


def test_frame_assembler_isolates_senders_and_counts_duplicate_chunks():
    bridge = load_bridge()
    assembler = bridge.FrameAssembler()
    sender_a = ("127.0.0.1", 9000)
    sender_b = ("127.0.0.1", 9001)

    first = struct.pack("!III", 7, 2, 0) + b"a"
    second = struct.pack("!III", 7, 2, 1) + b"b"
    assert assembler.add_datagram(first, sender=sender_a, now=0.0) is None
    assert assembler.add_datagram(first, sender=sender_a, now=0.1) is None
    assert assembler.stats.duplicate_chunks == 1
    assert assembler.add_datagram(second, sender=sender_b, now=0.2) is None
    assert assembler.partial_frame_count == 2

    completed = assembler.add_datagram(second, sender=sender_a, now=0.3)
    assert completed == bridge.CompletedPayload(frame_id=7, payload=b"ab", is_metadata=False)
    assert assembler.partial_frame_count == 1


def test_frame_pairer_emits_versioned_observation_only_after_both_halves():
    bridge = load_bridge()
    params = bridge.ViewerRuntimeParams()
    pairer = bridge.FramePairer(params=params)
    sender = ("127.0.0.1", 9000)
    raw = {
        "kind": "range_doppler",
        "source": "unit",
        "frame_id": 42,
        "center_freq_hz": 3.1e9,
        "sample_rate_hz": 50e6,
        "feature_rate_hz": 10.0,
        "range_doppler": {"range_profile": [0.0, 1.0], "peaks": []},
    }
    metadata = {
        "kind": "metadata",
        "source": "unit",
        "frame_id": 42,
        "cfar": {"hits": 1, "shown_hits": 1, "stats": {}, "candidate_clusters": []},
    }

    assert pairer.add(raw, sender=sender, now=0.0, received_at_ns=100) is None
    observation = pairer.add(metadata, sender=sender, now=0.1, received_at_ns=200)

    assert observation["schema"] == "ruview.rf_observation"
    assert observation["protocol_version"] == 1
    assert observation["source"] == "unit"
    assert observation["frame_id"] == 42
    assert observation["sequence"] == 42
    assert observation["source_timestamp_ns"] is None
    assert observation["received_at_ns"] == 200
    assert observation["config_hash"].startswith("sha256:")
    assert observation["freshness"] == "fresh"
    assert observation["observation"]["range_doppler"] == raw["range_doppler"]
    assert observation["observation"]["cfar"] == metadata["cfar"]
    assert "presence" not in observation
    assert "estimated_persons" not in observation
    assert pairer.stats.paired_frames == 1


def test_frame_pairer_expires_missing_half_and_rejects_duplicate_or_out_of_order():
    bridge = load_bridge()
    pairer = bridge.FramePairer(params=bridge.ViewerRuntimeParams(), pair_ttl_seconds=1.0)
    sender = ("127.0.0.1", 9000)

    assert pairer.add({"kind": "range_doppler", "source": "unit", "frame_id": 9,
                       "range_doppler": {}}, sender=sender, now=0.0) is None
    pairer.expire(now=2.0)
    assert pairer.stats.pair_timeouts == 1

    def paired(frame_id):
        raw = {"kind": "range_doppler", "source": "unit", "frame_id": frame_id,
               "range_doppler": {}}
        meta = {"kind": "metadata", "source": "unit", "frame_id": frame_id,
                "cfar": {"candidate_clusters": []}}
        assert pairer.add(raw, sender=sender, now=3.0) is None
        return pairer.add(meta, sender=sender, now=3.1)

    assert paired(10) is not None
    assert paired(10) is None
    assert paired(8) is None
    assert pairer.stats.duplicate_frames == 1
    assert pairer.stats.out_of_order_frames == 1


def test_frame_recorder_writes_jsonl_and_raw_payloads(tmp_path):
    bridge = load_bridge()
    recorder = bridge.FrameRecorder(
        jsonl_path=tmp_path / "frames.jsonl",
        raw_dir=tmp_path / "raw",
    )
    payload = bridge.CompletedPayload(frame_id=9, payload=b"raw-bytes", is_metadata=True)
    frame = {"schema": "ruview.rf_observation", "source": "unit", "sequence": 9}

    recorder.record_payload(payload)
    recorder.record_frame(frame)
    recorder.close()

    assert json.loads((tmp_path / "frames.jsonl").read_text(encoding="utf-8")) == frame
    assert (tmp_path / "raw" / "frame_000000009_metadata.bin").read_bytes() == b"raw-bytes"


def test_decode_aggregate_payload_returns_channel_frames():
    bridge = load_bridge()
    params = bridge.ViewerRuntimeParams(
        frame_format=bridge.FRAME_FORMAT_DENSE_RANGE_DOPPLER,
        wire_rows=2,
        wire_cols=3,
        range_fft_size=3,
        doppler_fft_size=2,
        stream_channel_count=2,
        stream_channel_mask=0b11,
    )
    ch0 = np.zeros((2, 3), dtype=np.complex64)
    ch1 = np.zeros((2, 3), dtype=np.complex64)
    ch0[0, 1] = 1.0 + 0.0j
    ch1[1, 2] = 2.0 + 0.0j
    ch_bytes = ch0.tobytes() + ch1.tobytes()
    payload = bridge.AGGREGATE_HEADER_STRUCT.pack(
        bridge.AGGREGATE_MAGIC_VERSION,
        2,
        ch0.nbytes,
        0b11,
        41,
    ) + ch_bytes

    frame_id, decoded = bridge.decode_aggregate_sensing_payload(41, payload, params)

    assert frame_id == 41
    assert [ch_id for ch_id, _ in decoded] == [0, 1]
    assert decoded[0][1].matrix[0, 1] == np.complex64(1.0 + 0.0j)
    assert decoded[1][1].matrix[1, 2] == np.complex64(2.0 + 0.0j)


def test_replay_payload_file_decodes_and_records_jsonl(tmp_path):
    bridge = load_bridge()
    params = bridge.ViewerRuntimeParams(
        frame_format=bridge.FRAME_FORMAT_DENSE_RANGE_DOPPLER,
        wire_rows=2,
        wire_cols=3,
        range_fft_size=3,
        doppler_fft_size=2,
    )
    rd = np.zeros((2, 3), dtype=np.complex64)
    rd[1, 2] = 3.0 + 0.0j
    raw_path = tmp_path / "frame_000000041_raw.bin"
    raw_path.write_bytes(rd.tobytes())
    metadata_path = tmp_path / "frame_000000041_metadata.bin"
    metadata_path.write_bytes(
        build_metadata_payload(bridge, 41, range_bin=2, doppler_bin=1, strength_db=12.0)
    )
    jsonl_path = tmp_path / "replay.jsonl"

    frames = bridge.replay_payload_files(
        [raw_path, metadata_path],
        params=params,
        source="unit-replay",
        center_freq_hz=3.1e9,
        sample_rate_hz=50e6,
        feature_rate_hz=10.0,
        record_jsonl=jsonl_path,
    )

    assert len(frames) == 1
    assert frames[0]["schema"] == "ruview.rf_observation"
    assert frames[0]["source"] == "unit-replay"
    assert frames[0]["sequence"] == 41
    assert frames[0]["observation"]["cfar"]["candidate_clusters"][0]["range_bin"] == 2
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["sequence"] == 41


def test_replay_missing_metadata_fails_closed(tmp_path):
    bridge = load_bridge()
    params = bridge.ViewerRuntimeParams(
        frame_format=bridge.FRAME_FORMAT_DENSE_RANGE_DOPPLER,
        wire_rows=2,
        wire_cols=3,
        range_fft_size=3,
        doppler_fft_size=2,
    )
    raw_path = tmp_path / "frame_000000042_raw.bin"
    raw_path.write_bytes(np.zeros((2, 3), dtype=np.complex64).tobytes())

    assert bridge.replay_payload_files(
        [raw_path],
        params=params,
        source="unit-replay",
        center_freq_hz=None,
        sample_rate_hz=None,
        feature_rate_hz=10.0,
    ) == []


def test_parser_defaults_openisac_listener_to_loopback():
    bridge = load_bridge()

    args = bridge.build_parser().parse_args([])

    assert args.openisac_host == "127.0.0.1"


def test_bridge_rejects_non_loopback_openisac_listener():
    bridge = load_bridge()

    with pytest.raises(ValueError, match="loopback"):
        bridge.validate_openisac_bind_host("0.0.0.0")


def test_compose_does_not_publish_rf_udp_ports_by_default():
    compose = (MODULE_PATH.parents[1] / "docker" / "docker-compose.yml").read_text(encoding="utf-8")

    assert '"5010:5010/udp"' not in compose
    assert '"5020:5020/udp"' not in compose


def test_ui_accepts_only_explicit_rf_source_labels():
    service = (MODULE_PATH.parents[1] / "ui" / "services" / "sensing.service.js").read_text(
        encoding="utf-8"
    )

    assert "rawSource.startsWith('rf-')" not in service
