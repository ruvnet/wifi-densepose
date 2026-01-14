"""Vital signs detection from CSI signals.

This module provides breathing and heartbeat detection capabilities
mirroring the Rust wifi-densepose-mat crate functionality.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from datetime import datetime, timezone
import scipy.signal
import scipy.fft


class BreathingType(Enum):
    """Types of breathing patterns."""
    NORMAL = "normal"
    SHALLOW = "shallow"
    DEEP = "deep"
    RAPID = "rapid"
    IRREGULAR = "irregular"
    APNEA = "apnea"


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class BreathingPattern:
    """Detected breathing pattern."""
    rate_bpm: float
    amplitude: float
    regularity: float
    pattern_type: BreathingType
    confidence: float
    timestamp: datetime


@dataclass
class HeartbeatSignature:
    """Detected heartbeat signature."""
    rate_bpm: float
    signal_strength: SignalStrength
    hrv_estimate: Optional[float]
    confidence: float
    timestamp: datetime


@dataclass
class VitalSignsReading:
    """Combined vital signs reading."""
    breathing: Optional[BreathingPattern]
    heartbeat: Optional[HeartbeatSignature]
    motion_detected: bool
    overall_confidence: float
    timestamp: datetime


@dataclass
class BreathingDetectorConfig:
    """Configuration for breathing detection."""
    min_rate_bpm: float = 4.0  # Very slow breathing
    max_rate_bpm: float = 40.0  # Fast breathing (distressed)
    min_amplitude: float = 0.1
    window_size: int = 512
    window_overlap: float = 0.5
    confidence_threshold: float = 0.3


@dataclass
class HeartbeatDetectorConfig:
    """Configuration for heartbeat detection."""
    min_rate_bpm: float = 30.0  # Bradycardia
    max_rate_bpm: float = 200.0  # Extreme tachycardia
    min_signal_strength: float = 0.05
    window_size: int = 1024
    enhanced_processing: bool = True
    confidence_threshold: float = 0.4


class BreathingDetector:
    """Detector for breathing patterns in CSI signals.

    Breathing causes periodic chest movement that modulates the WiFi signal.
    We detect this by looking for periodic variations in the 0.1-0.67 Hz range
    (corresponding to 6-40 breaths per minute).
    """

    def __init__(self, config: Optional[BreathingDetectorConfig] = None):
        """Initialize breathing detector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or BreathingDetectorConfig()

    def detect(self, csi_amplitudes: np.ndarray, sample_rate: float) -> Optional[BreathingPattern]:
        """Detect breathing pattern from CSI amplitude variations.

        Args:
            csi_amplitudes: Array of CSI amplitude values.
            sample_rate: Sampling rate in Hz.

        Returns:
            Detected BreathingPattern or None if not detected.
        """
        if len(csi_amplitudes) < self.config.window_size:
            return None

        # Calculate the frequency spectrum
        spectrum = self._compute_spectrum(csi_amplitudes)

        # Find the dominant frequency in the breathing range
        min_freq = self.config.min_rate_bpm / 60.0
        max_freq = self.config.max_rate_bpm / 60.0

        result = self._find_dominant_frequency(
            spectrum, sample_rate, min_freq, max_freq
        )

        if result is None:
            return None

        dominant_freq, amplitude = result

        # Convert to BPM
        rate_bpm = dominant_freq * 60.0

        # Check amplitude threshold
        if amplitude < self.config.min_amplitude:
            return None

        # Calculate regularity
        regularity = self._calculate_regularity(spectrum, dominant_freq, sample_rate)

        # Determine breathing type
        pattern_type = self._classify_pattern(rate_bpm, regularity)

        # Calculate confidence
        confidence = self._calculate_confidence(amplitude, regularity)

        if confidence < self.config.confidence_threshold:
            return None

        return BreathingPattern(
            rate_bpm=rate_bpm,
            amplitude=amplitude,
            regularity=regularity,
            pattern_type=pattern_type,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
        )

    def _compute_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Compute frequency spectrum using FFT."""
        # Apply window
        window = scipy.signal.windows.hamming(len(signal))
        windowed = signal * window

        # Compute FFT
        spectrum = np.abs(scipy.fft.rfft(windowed))
        return spectrum

    def _find_dominant_frequency(
        self,
        spectrum: np.ndarray,
        sample_rate: float,
        min_freq: float,
        max_freq: float
    ) -> Optional[Tuple[float, float]]:
        """Find the dominant frequency in a given range."""
        n = len(spectrum) * 2  # Original signal length
        freqs = scipy.fft.rfftfreq(n, 1.0 / sample_rate)

        # Find indices in the frequency range
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(mask):
            return None

        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0

        # Find peak
        peak_idx = np.argmax(masked_spectrum)
        if masked_spectrum[peak_idx] == 0:
            return None

        return freqs[peak_idx], spectrum[peak_idx]

    def _calculate_regularity(
        self,
        spectrum: np.ndarray,
        dominant_freq: float,
        sample_rate: float
    ) -> float:
        """Calculate how regular the breathing pattern is."""
        n = len(spectrum) * 2
        freqs = scipy.fft.rfftfreq(n, 1.0 / sample_rate)

        # Look at energy concentration around dominant frequency
        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        peak_idx = int(dominant_freq / freq_resolution) if freq_resolution > 0 else 0

        # Calculate energy in narrow band around peak
        half_bandwidth = 3  # bins on each side
        start_idx = max(0, peak_idx - half_bandwidth)
        end_idx = min(len(spectrum), peak_idx + half_bandwidth + 1)

        peak_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
        total_energy = np.sum(spectrum ** 2) + 1e-10

        regularity = float(peak_energy / total_energy)
        return min(1.0, regularity * 2.0)  # Scale to 0-1

    def _classify_pattern(self, rate_bpm: float, regularity: float) -> BreathingType:
        """Classify breathing pattern based on rate and regularity."""
        if regularity < 0.3:
            return BreathingType.IRREGULAR

        if rate_bpm < 6:
            return BreathingType.APNEA
        elif rate_bpm < 12:
            return BreathingType.SHALLOW
        elif rate_bpm <= 20:
            return BreathingType.NORMAL
        elif rate_bpm <= 25:
            return BreathingType.DEEP
        else:
            return BreathingType.RAPID

    def _calculate_confidence(self, amplitude: float, regularity: float) -> float:
        """Calculate detection confidence."""
        # Combine amplitude and regularity factors
        amp_factor = min(1.0, amplitude / 0.5)
        confidence = 0.6 * amp_factor + 0.4 * regularity
        return float(np.clip(confidence, 0.0, 1.0))


class HeartbeatDetector:
    """Detector for heartbeat signatures using micro-Doppler analysis.

    Heartbeats cause very small chest wall movements (~0.5mm) that can be
    detected through careful analysis of CSI phase variations at higher
    frequencies than breathing (0.8-3.3 Hz for 48-200 BPM).
    """

    def __init__(self, config: Optional[HeartbeatDetectorConfig] = None):
        """Initialize heartbeat detector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or HeartbeatDetectorConfig()

    def detect(
        self,
        csi_phase: np.ndarray,
        sample_rate: float,
        breathing_rate: Optional[float] = None
    ) -> Optional[HeartbeatSignature]:
        """Detect heartbeat from CSI phase data.

        Args:
            csi_phase: Array of CSI phase values in radians.
            sample_rate: Sampling rate in Hz.
            breathing_rate: Known breathing rate in Hz (optional).

        Returns:
            Detected HeartbeatSignature or None if not detected.
        """
        if len(csi_phase) < self.config.window_size:
            return None

        # Remove breathing component if known
        if breathing_rate is not None:
            filtered = self._remove_breathing_component(csi_phase, sample_rate, breathing_rate)
        else:
            filtered = self._highpass_filter(csi_phase, sample_rate, 0.8)

        # Compute micro-Doppler spectrum
        spectrum = self._compute_micro_doppler_spectrum(filtered, sample_rate)

        # Find heartbeat frequency
        min_freq = self.config.min_rate_bpm / 60.0
        max_freq = self.config.max_rate_bpm / 60.0

        result = self._find_heartbeat_frequency(
            spectrum, sample_rate, min_freq, max_freq
        )

        if result is None:
            return None

        heart_freq, strength = result

        if strength < self.config.min_signal_strength:
            return None

        rate_bpm = heart_freq * 60.0

        # Classify signal strength
        signal_strength = self._classify_signal_strength(strength)

        # Estimate HRV if we have enough data
        hrv_estimate = self._estimate_hrv(csi_phase, sample_rate, heart_freq)

        # Calculate confidence
        confidence = self._calculate_confidence(strength, signal_strength)

        if confidence < self.config.confidence_threshold:
            return None

        return HeartbeatSignature(
            rate_bpm=rate_bpm,
            signal_strength=signal_strength,
            hrv_estimate=hrv_estimate,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
        )

    def _remove_breathing_component(
        self,
        phase: np.ndarray,
        sample_rate: float,
        breathing_rate: float
    ) -> np.ndarray:
        """Remove breathing frequency component from phase signal."""
        # Design notch filter at breathing frequency
        quality_factor = 30.0
        b, a = scipy.signal.iirnotch(breathing_rate, quality_factor, sample_rate)

        # Also remove harmonics (2x, 3x)
        filtered = scipy.signal.filtfilt(b, a, phase)

        for harmonic in [2, 3]:
            notch_freq = breathing_rate * harmonic
            if notch_freq < sample_rate / 2:
                b, a = scipy.signal.iirnotch(notch_freq, quality_factor, sample_rate)
                filtered = scipy.signal.filtfilt(b, a, filtered)

        return filtered

    def _highpass_filter(
        self,
        signal: np.ndarray,
        sample_rate: float,
        cutoff: float
    ) -> np.ndarray:
        """Apply highpass filter to remove low-frequency components."""
        nyquist = sample_rate / 2
        if cutoff >= nyquist:
            return signal

        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='high')
        return scipy.signal.filtfilt(b, a, signal)

    def _compute_micro_doppler_spectrum(
        self,
        signal: np.ndarray,
        sample_rate: float
    ) -> np.ndarray:
        """Compute micro-Doppler spectrum for heartbeat detection."""
        # Use shorter window for better time resolution
        window_size = min(len(signal), self.config.window_size)

        if self.config.enhanced_processing:
            # Use STFT for better frequency resolution
            f, t, Zxx = scipy.signal.stft(
                signal,
                sample_rate,
                nperseg=window_size,
                noverlap=window_size // 2
            )
            # Average over time
            spectrum = np.mean(np.abs(Zxx), axis=1)
        else:
            # Simple FFT
            window = scipy.signal.windows.hamming(window_size)
            windowed = signal[:window_size] * window
            spectrum = np.abs(scipy.fft.rfft(windowed))

        return spectrum

    def _find_heartbeat_frequency(
        self,
        spectrum: np.ndarray,
        sample_rate: float,
        min_freq: float,
        max_freq: float
    ) -> Optional[Tuple[float, float]]:
        """Find heartbeat frequency in the spectrum."""
        n = len(spectrum) * 2
        freqs = scipy.fft.rfftfreq(n, 1.0 / sample_rate)

        # Find indices in the frequency range
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(mask):
            return None

        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0

        # Find peak
        peak_idx = np.argmax(masked_spectrum)
        if masked_spectrum[peak_idx] == 0:
            return None

        return freqs[peak_idx], spectrum[peak_idx]

    def _classify_signal_strength(self, strength: float) -> SignalStrength:
        """Classify signal strength level."""
        if strength > 0.3:
            return SignalStrength.STRONG
        elif strength > 0.15:
            return SignalStrength.MODERATE
        elif strength > 0.08:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _estimate_hrv(
        self,
        phase: np.ndarray,
        sample_rate: float,
        heart_freq: float
    ) -> Optional[float]:
        """Estimate heart rate variability."""
        # Simple HRV estimation based on spectral width
        # In practice, would use peak detection and RR interval analysis
        n = len(phase)
        if n < self.config.window_size * 2:
            return None

        # Placeholder - would require more sophisticated analysis
        return None

    def _calculate_confidence(
        self,
        strength: float,
        signal_class: SignalStrength
    ) -> float:
        """Calculate detection confidence."""
        strength_factor = min(1.0, strength / 0.2)

        class_weights = {
            SignalStrength.STRONG: 1.0,
            SignalStrength.MODERATE: 0.7,
            SignalStrength.WEAK: 0.4,
            SignalStrength.VERY_WEAK: 0.2,
        }
        class_factor = class_weights[signal_class]

        confidence = 0.5 * strength_factor + 0.5 * class_factor
        return float(np.clip(confidence, 0.0, 1.0))


class VitalSignsDetector:
    """Combined vital signs detector for breathing and heartbeat."""

    def __init__(
        self,
        breathing_config: Optional[BreathingDetectorConfig] = None,
        heartbeat_config: Optional[HeartbeatDetectorConfig] = None
    ):
        """Initialize combined detector.

        Args:
            breathing_config: Breathing detector configuration.
            heartbeat_config: Heartbeat detector configuration.
        """
        self.breathing_detector = BreathingDetector(breathing_config)
        self.heartbeat_detector = HeartbeatDetector(heartbeat_config)
        self._motion_threshold = 0.5

    def detect(
        self,
        csi_amplitude: np.ndarray,
        csi_phase: np.ndarray,
        sample_rate: float
    ) -> VitalSignsReading:
        """Detect vital signs from CSI data.

        Args:
            csi_amplitude: CSI amplitude values.
            csi_phase: CSI phase values in radians.
            sample_rate: Sampling rate in Hz.

        Returns:
            Combined VitalSignsReading.
        """
        # Detect breathing
        breathing = self.breathing_detector.detect(csi_amplitude, sample_rate)

        # Detect heartbeat (using breathing rate if available)
        breathing_rate = (breathing.rate_bpm / 60.0) if breathing else None
        heartbeat = self.heartbeat_detector.detect(csi_phase, sample_rate, breathing_rate)

        # Detect motion
        motion_detected = self._detect_motion(csi_amplitude)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            breathing, heartbeat, motion_detected
        )

        return VitalSignsReading(
            breathing=breathing,
            heartbeat=heartbeat,
            motion_detected=motion_detected,
            overall_confidence=overall_confidence,
            timestamp=datetime.now(timezone.utc)
        )

    def _detect_motion(self, amplitude: np.ndarray) -> bool:
        """Detect significant motion from amplitude variance."""
        if len(amplitude) < 10:
            return False
        variance = np.var(amplitude)
        return variance > self._motion_threshold

    def _calculate_overall_confidence(
        self,
        breathing: Optional[BreathingPattern],
        heartbeat: Optional[HeartbeatSignature],
        motion_detected: bool
    ) -> float:
        """Calculate overall detection confidence."""
        confidences = []

        if breathing:
            confidences.append(breathing.confidence)
        if heartbeat:
            confidences.append(heartbeat.confidence)

        if not confidences:
            return 0.0

        base_confidence = np.mean(confidences)

        # Motion can either help (confirms presence) or hurt (noise)
        if motion_detected:
            # Strong motion reduces confidence in subtle vital sign detection
            if base_confidence > 0.7:
                base_confidence *= 0.9

        return float(np.clip(base_confidence, 0.0, 1.0))
