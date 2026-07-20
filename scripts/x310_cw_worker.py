#!/usr/bin/env python3
"""USRP X310 continuous-wave RF worker for RuView rf-direct mode.

The worker sends RF-native feature JSON to the Rust sensing server on UDP
port 5020 by default. Use --demo on machines without UHD hardware. On the
USRP host, omit --demo to transmit a low-amplitude CW tone on one channel,
receive it on another channel, estimate the complex channel, and stream
motion/breathing observables to RuView.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_phase(delta: float) -> float:
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def send_json(sock: socket.socket, target: tuple[str, int], frame: dict) -> None:
    payload = json.dumps(frame, separators=(",", ":")).encode("utf-8")
    sock.sendto(payload, target)


def demo_frames(rate_hz: float) -> Iterable[dict]:
    seq = 0
    period = 1.0 / rate_hz
    phase = 0.0
    last_phase = 0.0
    while True:
        t = seq * period
        breathing_hz = 0.24
        slow_body_motion = 0.18 * math.sin(2.0 * math.pi * 0.035 * t)
        phase = 0.7 * math.sin(2.0 * math.pi * breathing_hz * t) + slow_body_motion
        phase_delta = wrap_phase(phase - last_phase)
        last_phase = phase
        amplitude = 0.78 + 0.03 * math.sin(2.0 * math.pi * breathing_hz * t + 0.7)
        motion_energy = clamp(abs(phase_delta) * 8.0 + abs(slow_body_motion) * 0.12, 0.0, 1.0)
        snr_db = 31.0 + 1.5 * math.sin(2.0 * math.pi * 0.05 * t)
        yield {
            "source": "x310-cw-demo",
            "node_id": 1,
            "sequence": seq,
            "center_freq_hz": 2_450_000_000.0,
            "sample_rate_hz": 1_000_000.0,
            "feature_rate_hz": rate_hz,
            "motion_energy": motion_energy,
            "breathing_bpm": breathing_hz * 60.0,
            "breathing_confidence": 0.78,
            "phase_track_rad": phase,
            "phase_delta_rad": phase_delta,
            "amplitude": amplitude,
            "snr_db": snr_db,
            "confidence": clamp(snr_db / 40.0, 0.0, 1.0),
            "range_bins": [
                amplitude,
                motion_energy,
                abs(phase_delta),
                0.78,
            ],
        }
        seq += 1
        time.sleep(period)


@dataclass
class FeatureEstimate:
    phase: float
    phase_delta: float
    amplitude: float
    motion_energy: float
    snr_db: float
    breathing_bpm: Optional[float]
    breathing_confidence: float


class ChannelEstimator:
    def __init__(self, sample_rate_hz: float, tone_hz: float, feature_rate_hz: float):
        self.sample_rate_hz = sample_rate_hz
        self.tone_hz = tone_hz
        self.feature_rate_hz = feature_rate_hz
        self.sample_index = 0
        self.last_phase: Optional[float] = None
        self.last_amplitude: Optional[float] = None
        self.unwrapped_phase = 0.0
        self.phase_history: Deque[float] = deque(maxlen=max(64, int(feature_rate_hz * 30)))

    def estimate(self, samples) -> FeatureEstimate:
        import numpy as np

        n = np.arange(samples.size, dtype=np.float64) + float(self.sample_index)
        lo = np.exp(-1j * 2.0 * np.pi * self.tone_hz * n / self.sample_rate_hz)
        mixed = samples * lo.astype(np.complex64)
        channel = np.mean(mixed)
        residual = mixed - channel
        signal_power = float(abs(channel) ** 2)
        noise_power = float(np.mean(np.abs(residual) ** 2) + 1e-12)
        snr_db = 10.0 * math.log10(max(signal_power, 1e-12) / noise_power)

        phase = float(np.angle(channel))
        amplitude = float(abs(channel))
        if self.last_phase is None:
            phase_delta = 0.0
        else:
            phase_delta = wrap_phase(phase - self.last_phase)
        self.last_phase = phase
        self.unwrapped_phase += phase_delta
        self.phase_history.append(self.unwrapped_phase)

        amp_delta = 0.0
        if self.last_amplitude is not None:
            amp_delta = abs(amplitude - self.last_amplitude) / max(self.last_amplitude, 1e-6)
        self.last_amplitude = amplitude

        motion_energy = clamp(abs(phase_delta) * 6.0 + amp_delta * 2.0, 0.0, 1.0)
        breathing_bpm, breathing_confidence = self._estimate_breathing()
        self.sample_index += samples.size
        return FeatureEstimate(
            phase=phase,
            phase_delta=phase_delta,
            amplitude=amplitude,
            motion_energy=motion_energy,
            snr_db=snr_db,
            breathing_bpm=breathing_bpm,
            breathing_confidence=breathing_confidence,
        )

    def _estimate_breathing(self) -> tuple[Optional[float], float]:
        import numpy as np

        min_len = max(64, int(self.feature_rate_hz * 8))
        if len(self.phase_history) < min_len:
            return None, 0.0

        data = np.asarray(self.phase_history, dtype=np.float64)
        data = data - np.mean(data)
        if not np.any(np.isfinite(data)):
            return None, 0.0

        window = np.hanning(data.size)
        spectrum = np.abs(np.fft.rfft(data * window)) ** 2
        freqs = np.fft.rfftfreq(data.size, d=1.0 / self.feature_rate_hz)
        band = (freqs >= 0.10) & (freqs <= 0.60)
        if not np.any(band):
            return None, 0.0

        band_power = spectrum[band]
        band_freqs = freqs[band]
        idx = int(np.argmax(band_power))
        peak = float(band_power[idx])
        floor = float(np.median(band_power) + 1e-12)
        confidence = clamp((peak / floor - 1.0) / 8.0, 0.0, 1.0)
        if confidence < 0.15:
            return None, confidence
        return float(band_freqs[idx] * 60.0), confidence


def configure_usrp(args):
    import uhd

    usrp = uhd.usrp.MultiUSRP(args.device_args)
    if args.clock_source:
        usrp.set_clock_source(args.clock_source)
    if args.time_source:
        usrp.set_time_source(args.time_source)

    usrp.set_tx_rate(args.rate, args.tx_chan)
    usrp.set_rx_rate(args.rate, args.rx_chan)
    tune = uhd.types.TuneRequest(args.center_freq)
    usrp.set_tx_freq(tune, args.tx_chan)
    usrp.set_rx_freq(tune, args.rx_chan)
    usrp.set_tx_gain(args.tx_gain, args.tx_chan)
    usrp.set_rx_gain(args.rx_gain, args.rx_chan)
    usrp.set_tx_antenna(args.tx_ant, args.tx_chan)
    usrp.set_rx_antenna(args.rx_ant, args.rx_chan)
    time.sleep(args.settle_time)
    return usrp, uhd


def make_stream_args(uhd, channels: list[int], stream_args: str):
    args = uhd.usrp.StreamArgs("fc32", "sc16")
    args.channels = channels
    if stream_args:
        args.args = uhd.types.DeviceAddr(stream_args)
    return args


def tx_worker(uhd, tx_streamer, args, stop_event: threading.Event) -> None:
    import numpy as np

    num_channels = tx_streamer.get_num_channels()
    block_len = min(max(1024, int(args.rate / 200)), tx_streamer.get_max_num_samps())
    block_len = max(256, block_len)
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    phase = 0.0
    phase_inc = 2.0 * np.pi * args.tone_hz / args.rate

    while not stop_event.is_set():
        phases = phase + phase_inc * np.arange(block_len, dtype=np.float32)
        tone = (args.tx_amplitude * np.exp(1j * phases)).astype(np.complex64)
        phase = float((phase + phase_inc * block_len) % (2.0 * np.pi))
        buffer = tone.reshape(1, -1) if num_channels == 1 else np.tile(tone, (num_channels, 1))
        try:
            tx_streamer.send(buffer, metadata)
        except RuntimeError as exc:
            print(f"x310_cw_worker: TX runtime error: {exc}", file=sys.stderr)
            stop_event.set()
            break
        metadata.start_of_burst = False

    metadata.end_of_burst = True
    try:
        tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), metadata)
    except RuntimeError:
        pass


def recv_with_optional_timeout(rx_streamer, buffer, metadata, timeout: float):
    try:
        return rx_streamer.recv(buffer, metadata, timeout)
    except TypeError:
        return rx_streamer.recv(buffer, metadata)


def rx_metadata_ok(uhd, metadata) -> bool:
    ok_code = getattr(uhd.types.RXMetadataErrorCode, "none", None)
    if ok_code is not None:
        return metadata.error_code == ok_code
    return "none" in str(metadata.error_code).lower()


def rx_metadata_error(metadata) -> str:
    strerror = getattr(metadata, "strerror", None)
    if callable(strerror):
        return strerror()
    return str(getattr(metadata, "error_code", "unknown"))


def run_real(args) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for real UHD mode; install numpy or use --demo") from exc

    try:
        usrp, uhd = configure_usrp(args)
    except ImportError as exc:
        raise SystemExit(
            "UHD Python API is not importable on this host. Run on the USRP host "
            "or use --demo on this laptop."
        ) from exc

    rx_streamer = usrp.get_rx_stream(make_stream_args(uhd, [args.rx_chan], args.stream_args))
    tx_streamer = usrp.get_tx_stream(make_stream_args(uhd, [args.tx_chan], args.stream_args))
    rx_metadata = uhd.types.RXMetadata()
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)

    stop_event = threading.Event()
    tx_thread = threading.Thread(target=tx_worker, args=(uhd, tx_streamer, args, stop_event), daemon=True)
    tx_thread.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)
    estimator = ChannelEstimator(args.rate, args.tone_hz, args.feature_rate_hz)
    frame_len = max(256, int(args.rate / args.feature_rate_hz))
    chunk_len = min(rx_streamer.get_max_num_samps(), frame_len)
    recv_buffer = np.empty((1, chunk_len), dtype=np.complex64)
    frame_buffer = np.empty(frame_len, dtype=np.complex64)
    filled = 0
    sequence = 0
    start = time.monotonic()

    print(
        "x310_cw_worker: streaming RF-direct frames to "
        f"{args.host}:{args.port} at {args.feature_rate_hz:.1f} Hz",
        flush=True,
    )
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            got = int(recv_with_optional_timeout(rx_streamer, recv_buffer, rx_metadata, 1.0))
            if got <= 0:
                continue
            if not rx_metadata_ok(uhd, rx_metadata):
                print(f"x310_cw_worker: RX metadata error: {rx_metadata_error(rx_metadata)}", file=sys.stderr)
                continue

            chunk = recv_buffer[0, :got]
            offset = 0
            while offset < got:
                take = min(frame_len - filled, got - offset)
                frame_buffer[filled : filled + take] = chunk[offset : offset + take]
                filled += take
                offset += take
                if filled < frame_len:
                    continue

                estimate = estimator.estimate(frame_buffer)
                frame = {
                    "source": "x310-cw",
                    "node_id": 1,
                    "sequence": sequence,
                    "center_freq_hz": args.center_freq,
                    "sample_rate_hz": args.rate,
                    "feature_rate_hz": args.feature_rate_hz,
                    "motion_energy": estimate.motion_energy,
                    "breathing_bpm": estimate.breathing_bpm,
                    "breathing_confidence": estimate.breathing_confidence,
                    "phase_track_rad": estimate.phase,
                    "phase_delta_rad": estimate.phase_delta,
                    "amplitude": estimate.amplitude,
                    "snr_db": estimate.snr_db,
                    "confidence": clamp(estimate.snr_db / 40.0, 0.0, 1.0),
                    "range_bins": [
                        estimate.amplitude,
                        estimate.motion_energy,
                        abs(estimate.phase_delta),
                        estimate.breathing_confidence,
                    ],
                }
                send_json(sock, target, frame)
                if args.verbose and sequence % int(max(1, args.feature_rate_hz)) == 0:
                    bpm = "-" if estimate.breathing_bpm is None else f"{estimate.breathing_bpm:.1f}"
                    print(
                        f"seq={sequence} motion={estimate.motion_energy:.3f} "
                        f"snr={estimate.snr_db:.1f}dB br={bpm}",
                        flush=True,
                    )
                sequence += 1
                filled = 0
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        tx_thread.join(timeout=2.0)
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    return 0


def run_demo(args) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)
    start = time.monotonic()
    for frame in demo_frames(args.feature_rate_hz):
        send_json(sock, target, frame)
        if args.verbose and frame["sequence"] % int(max(1, args.feature_rate_hz)) == 0:
            print(
                f"seq={frame['sequence']} motion={frame['motion_energy']:.3f} "
                f"snr={frame['snr_db']:.1f}dB br={frame['breathing_bpm']:.1f}",
                flush=True,
            )
        if args.duration > 0 and time.monotonic() - start >= args.duration:
            break
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="RuView sensing-server host")
    p.add_argument("--port", type=int, default=5020, help="RuView rf-direct UDP port")
    p.add_argument("--demo", action="store_true", help="send synthetic RF-direct frames")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means forever")
    p.add_argument("--verbose", action="store_true", help="print one-line feature summaries")
    p.add_argument("--device-args", default="addr=192.168.10.2", help="UHD device args")
    p.add_argument("--center-freq", type=float, default=2.45e9, help="RF center frequency in Hz")
    p.add_argument("--rate", type=float, default=1e6, help="USRP sample rate in samples/sec")
    p.add_argument("--feature-rate-hz", type=float, default=20.0, help="feature frame rate")
    p.add_argument("--tone-hz", type=float, default=25_000.0, help="baseband TX tone offset")
    p.add_argument("--tx-chan", type=int, default=0)
    p.add_argument("--rx-chan", type=int, default=1)
    p.add_argument("--tx-ant", default="TX/RX")
    p.add_argument("--rx-ant", default="RX2")
    p.add_argument("--tx-gain", type=float, default=0.0)
    p.add_argument("--rx-gain", type=float, default=10.0)
    p.add_argument("--tx-amplitude", type=float, default=0.05, help="complex baseband amplitude [0,1]")
    p.add_argument("--clock-source", choices=["internal", "external", "gpsdo"], help="optional UHD clock source")
    p.add_argument("--time-source", choices=["internal", "external", "gpsdo"], help="optional UHD time source")
    p.add_argument("--settle-time", type=float, default=0.2)
    p.add_argument("--stream-args", default="", help="optional UHD stream args, e.g. spp=200")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.feature_rate_hz <= 0:
        raise SystemExit("--feature-rate-hz must be positive")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    if not 0.0 < args.tx_amplitude <= 1.0:
        raise SystemExit("--tx-amplitude must be in (0, 1]")
    if args.demo:
        return run_demo(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
