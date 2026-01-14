"""CSI data extraction from WiFi hardware using Test-Driven Development approach."""

import asyncio
import struct
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, Protocol, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging


class CSIParseError(Exception):
    """Exception raised for CSI parsing errors."""
    pass


class CSIValidationError(Exception):
    """Exception raised for CSI validation errors."""
    pass


@dataclass
class CSIData:
    """Data structure for CSI measurements."""
    timestamp: datetime
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: float
    bandwidth: float
    num_subcarriers: int
    num_antennas: int
    snr: float
    metadata: Dict[str, Any]


class CSIParser(Protocol):
    """Protocol for CSI data parsers."""

    def parse(self, raw_data: bytes) -> CSIData:
        """Parse raw CSI data into structured format."""
        ...


class ESP32CSIParser:
    """Parser for ESP32 CSI data format.

    ESP32 CSI data format (from esp-csi library):
    - Header: 'CSI_DATA:' prefix
    - Fields: timestamp,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
              not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,
              ampdu_cnt,channel,secondary_channel,local_timestamp,
              ant,sig_len,rx_state,len,first_word,data[...]

    The actual CSI data is in the 'data' field as complex I/Q values.
    """

    def __init__(self):
        """Initialize ESP32 CSI parser with default configuration."""
        self.htltf_subcarriers = 56  # HT-LTF subcarriers for 20MHz
        self.antenna_count = 1  # Most ESP32 have 1 antenna

    def parse(self, raw_data: bytes) -> CSIData:
        """Parse ESP32 CSI data format.

        Args:
            raw_data: Raw bytes from ESP32 serial/network

        Returns:
            Parsed CSI data

        Raises:
            CSIParseError: If data format is invalid
        """
        if not raw_data:
            raise CSIParseError("Empty data received")

        try:
            data_str = raw_data.decode('utf-8').strip()

            # Handle ESP-CSI library format
            if data_str.startswith('CSI_DATA,'):
                return self._parse_esp_csi_format(data_str)
            # Handle simplified format for testing
            elif data_str.startswith('CSI_DATA:'):
                return self._parse_simple_format(data_str)
            else:
                raise CSIParseError("Invalid ESP32 CSI data format")

        except UnicodeDecodeError:
            # Binary format - parse as raw bytes
            return self._parse_binary_format(raw_data)
        except (ValueError, IndexError) as e:
            raise CSIParseError(f"Failed to parse ESP32 data: {e}")

    def _parse_esp_csi_format(self, data_str: str) -> CSIData:
        """Parse ESP-CSI library CSV format.

        Format: CSI_DATA,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bw>,<smoothing>,
                <not_sounding>,<aggregation>,<stbc>,<fec>,<sgi>,<noise>,
                <ampdu_cnt>,<channel>,<sec_chan>,<timestamp>,<ant>,<sig_len>,
                <rx_state>,<len>,[csi_data...]
        """
        parts = data_str.split(',')

        if len(parts) < 22:
            raise CSIParseError(f"Incomplete ESP-CSI data: expected >= 22 fields, got {len(parts)}")

        # Extract metadata
        mac_addr = parts[1]
        rssi = int(parts[2])
        rate = int(parts[3])
        sig_mode = int(parts[4])
        mcs = int(parts[5])
        bandwidth = int(parts[6])  # 0=20MHz, 1=40MHz
        channel = int(parts[15])
        timestamp_us = int(parts[17])
        csi_len = int(parts[21])

        # Parse CSI I/Q data (remaining fields are the CSI values)
        csi_raw = [int(x) for x in parts[22:22 + csi_len]]

        # Convert I/Q pairs to complex numbers
        # ESP32 CSI format: [I0, Q0, I1, Q1, ...] as signed 8-bit integers
        amplitude, phase = self._iq_to_amplitude_phase(csi_raw)

        # Determine frequency from channel
        if channel <= 14:
            frequency = 2.412e9 + (channel - 1) * 5e6  # 2.4 GHz band
        else:
            frequency = 5.0e9 + (channel - 36) * 5e6  # 5 GHz band

        bw_hz = 20e6 if bandwidth == 0 else 40e6
        num_subcarriers = len(amplitude) // self.antenna_count

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp_us / 1e6, tz=timezone.utc),
            amplitude=amplitude.reshape(self.antenna_count, -1),
            phase=phase.reshape(self.antenna_count, -1),
            frequency=frequency,
            bandwidth=bw_hz,
            num_subcarriers=num_subcarriers,
            num_antennas=self.antenna_count,
            snr=float(rssi + 100),  # Approximate SNR from RSSI
            metadata={
                'source': 'esp32',
                'mac': mac_addr,
                'rssi': rssi,
                'mcs': mcs,
                'channel': channel,
                'sig_mode': sig_mode,
            }
        )

    def _parse_simple_format(self, data_str: str) -> CSIData:
        """Parse simplified CSI format for testing/development.

        Format: CSI_DATA:timestamp,antennas,subcarriers,freq,bw,snr,[amp_values],[phase_values]
        """
        content = data_str[9:]  # Remove 'CSI_DATA:' prefix

        # Split the main fields and array data
        if '[' in content:
            main_part, arrays_part = content.split('[', 1)
            parts = main_part.rstrip(',').split(',')

            # Parse amplitude and phase arrays
            arrays_str = '[' + arrays_part
            amp_str, phase_str = self._split_arrays(arrays_str)
            amplitude = np.array([float(x) for x in amp_str.strip('[]').split(',')])
            phase = np.array([float(x) for x in phase_str.strip('[]').split(',')])
        else:
            parts = content.split(',')
            # No array data provided, need to return error or minimal data
            raise CSIParseError("No CSI array data in simple format")

        timestamp_ms = int(parts[0])
        num_antennas = int(parts[1])
        num_subcarriers = int(parts[2])
        frequency_mhz = float(parts[3])
        bandwidth_mhz = float(parts[4])
        snr = float(parts[5])

        # Reshape arrays
        expected_size = num_antennas * num_subcarriers
        if len(amplitude) != expected_size:
            # Interpolate or pad
            amplitude = np.interp(
                np.linspace(0, 1, expected_size),
                np.linspace(0, 1, len(amplitude)),
                amplitude
            )
            phase = np.interp(
                np.linspace(0, 1, expected_size),
                np.linspace(0, 1, len(phase)),
                phase
            )

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc),
            amplitude=amplitude.reshape(num_antennas, num_subcarriers),
            phase=phase.reshape(num_antennas, num_subcarriers),
            frequency=frequency_mhz * 1e6,
            bandwidth=bandwidth_mhz * 1e6,
            num_subcarriers=num_subcarriers,
            num_antennas=num_antennas,
            snr=snr,
            metadata={'source': 'esp32', 'format': 'simple'}
        )

    def _parse_binary_format(self, raw_data: bytes) -> CSIData:
        """Parse binary CSI format from ESP32.

        Binary format (struct packed):
        - 4 bytes: timestamp (uint32)
        - 1 byte: num_antennas (uint8)
        - 1 byte: num_subcarriers (uint8)
        - 2 bytes: channel (uint16)
        - 4 bytes: frequency (float32)
        - 4 bytes: bandwidth (float32)
        - 4 bytes: snr (float32)
        - Remaining: CSI I/Q data as int8 pairs
        """
        if len(raw_data) < 20:
            raise CSIParseError("Binary data too short")

        header_fmt = '<IBBHfff'
        header_size = struct.calcsize(header_fmt)

        timestamp, num_antennas, num_subcarriers, channel, freq, bw, snr = \
            struct.unpack(header_fmt, raw_data[:header_size])

        # Parse I/Q data
        iq_data = raw_data[header_size:]
        csi_raw = list(struct.unpack(f'{len(iq_data)}b', iq_data))

        amplitude, phase = self._iq_to_amplitude_phase(csi_raw)

        # Adjust dimensions
        expected_size = num_antennas * num_subcarriers
        if len(amplitude) < expected_size:
            amplitude = np.pad(amplitude, (0, expected_size - len(amplitude)))
            phase = np.pad(phase, (0, expected_size - len(phase)))
        elif len(amplitude) > expected_size:
            amplitude = amplitude[:expected_size]
            phase = phase[:expected_size]

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            amplitude=amplitude.reshape(num_antennas, num_subcarriers),
            phase=phase.reshape(num_antennas, num_subcarriers),
            frequency=float(freq),
            bandwidth=float(bw),
            num_subcarriers=num_subcarriers,
            num_antennas=num_antennas,
            snr=float(snr),
            metadata={'source': 'esp32', 'format': 'binary', 'channel': channel}
        )

    def _iq_to_amplitude_phase(self, iq_data: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert I/Q pairs to amplitude and phase.

        Args:
            iq_data: List of interleaved I, Q values (signed 8-bit)

        Returns:
            Tuple of (amplitude, phase) arrays
        """
        if len(iq_data) % 2 != 0:
            iq_data = iq_data[:-1]  # Trim odd value

        i_vals = np.array(iq_data[0::2], dtype=np.float64)
        q_vals = np.array(iq_data[1::2], dtype=np.float64)

        # Calculate amplitude (magnitude) and phase
        complex_vals = i_vals + 1j * q_vals
        amplitude = np.abs(complex_vals)
        phase = np.angle(complex_vals)

        # Normalize amplitude to [0, 1] range
        max_amp = np.max(amplitude)
        if max_amp > 0:
            amplitude = amplitude / max_amp

        return amplitude, phase

    def _split_arrays(self, arrays_str: str) -> Tuple[str, str]:
        """Split concatenated array strings."""
        # Find the boundary between two arrays
        depth = 0
        split_idx = 0
        for i, c in enumerate(arrays_str):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    split_idx = i + 1
                    break

        amp_str = arrays_str[:split_idx]
        phase_str = arrays_str[split_idx:].lstrip(',')
        return amp_str, phase_str


class RouterCSIParser:
    """Parser for router CSI data formats (Atheros, Intel, etc.).

    Supports:
    - Atheros CSI Tool format (ath9k/ath10k)
    - Intel 5300 CSI Tool format
    - Nexmon CSI format (Broadcom)
    """

    def __init__(self):
        """Initialize router CSI parser."""
        self.default_subcarriers = 56  # 20MHz HT
        self.default_antennas = 3

    def parse(self, raw_data: bytes) -> CSIData:
        """Parse router CSI data format.

        Args:
            raw_data: Raw bytes from router

        Returns:
            Parsed CSI data

        Raises:
            CSIParseError: If data format is invalid
        """
        if not raw_data:
            raise CSIParseError("Empty data received")

        # Try to decode as text first
        try:
            data_str = raw_data.decode('utf-8')
            if data_str.startswith('ATHEROS_CSI:'):
                return self._parse_atheros_text_format(data_str)
            elif data_str.startswith('INTEL_CSI:'):
                return self._parse_intel_text_format(data_str)
        except UnicodeDecodeError:
            pass

        # Binary format detection based on header
        if len(raw_data) >= 4:
            magic = struct.unpack('<I', raw_data[:4])[0]
            if magic == 0x11111111:  # Atheros CSI Tool magic
                return self._parse_atheros_binary_format(raw_data)
            elif magic == 0xBB:  # Intel 5300 magic byte pattern
                return self._parse_intel_binary_format(raw_data)

        raise CSIParseError("Unknown router CSI format")

    def _parse_atheros_text_format(self, data_str: str) -> CSIData:
        """Parse Atheros CSI text format.

        Format: ATHEROS_CSI:timestamp,rssi,rate,channel,bw,nr,nc,num_tones,[csi_data...]
        """
        content = data_str[12:]  # Remove 'ATHEROS_CSI:' prefix
        parts = content.split(',')

        if len(parts) < 8:
            raise CSIParseError("Incomplete Atheros CSI data")

        timestamp = int(parts[0])
        rssi = int(parts[1])
        rate = int(parts[2])
        channel = int(parts[3])
        bandwidth = int(parts[4])  # MHz
        nr = int(parts[5])  # Rx antennas
        nc = int(parts[6])  # Tx antennas (usually 1 for probe)
        num_tones = int(parts[7])  # Subcarriers

        # Parse CSI matrix data
        csi_values = [float(x) for x in parts[8:] if x.strip()]

        # CSI data is complex: [real, imag, real, imag, ...]
        amplitude, phase = self._parse_complex_csi(csi_values, nr, num_tones)

        # Calculate frequency from channel
        if channel <= 14:
            frequency = 2.412e9 + (channel - 1) * 5e6
        else:
            frequency = 5.18e9 + (channel - 36) * 5e6

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth * 1e6,
            num_subcarriers=num_tones,
            num_antennas=nr,
            snr=float(rssi + 95),
            metadata={
                'source': 'atheros_router',
                'rssi': rssi,
                'rate': rate,
                'channel': channel,
                'tx_antennas': nc,
            }
        )

    def _parse_atheros_binary_format(self, raw_data: bytes) -> CSIData:
        """Parse Atheros CSI Tool binary format.

        Based on ath9k/ath10k CSI Tool structure:
        - 4 bytes: magic (0x11111111)
        - 8 bytes: timestamp
        - 2 bytes: channel
        - 1 byte: bandwidth (0=20MHz, 1=40MHz, 2=80MHz)
        - 1 byte: nr (rx antennas)
        - 1 byte: nc (tx antennas)
        - 1 byte: num_tones
        - 2 bytes: rssi
        - Remaining: CSI payload (complex int16 per subcarrier per antenna pair)
        """
        if len(raw_data) < 20:
            raise CSIParseError("Atheros binary data too short")

        header_fmt = '<IQHBBBBB'  # Q is 8-byte timestamp
        header_size = struct.calcsize(header_fmt)

        magic, timestamp, channel, bw, nr, nc, num_tones, rssi = \
            struct.unpack(header_fmt, raw_data[:header_size])

        if magic != 0x11111111:
            raise CSIParseError("Invalid Atheros magic number")

        # Parse CSI payload
        csi_data = raw_data[header_size:]

        # Each subcarrier has complex value per antenna pair: int16 real + int16 imag
        expected_bytes = nr * nc * num_tones * 4
        if len(csi_data) < expected_bytes:
            # Adjust num_tones based on available data
            num_tones = len(csi_data) // (nr * nc * 4)

        csi_complex = np.zeros((nr, num_tones), dtype=np.complex128)

        for ant in range(nr):
            for tone in range(num_tones):
                offset = (ant * nc * num_tones + tone) * 4
                if offset + 4 <= len(csi_data):
                    real, imag = struct.unpack('<hh', csi_data[offset:offset+4])
                    csi_complex[ant, tone] = complex(real, imag)

        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)

        # Normalize amplitude
        max_amp = np.max(amplitude)
        if max_amp > 0:
            amplitude = amplitude / max_amp

        # Calculate frequency
        if channel <= 14:
            frequency = 2.412e9 + (channel - 1) * 5e6
        else:
            frequency = 5.18e9 + (channel - 36) * 5e6

        bandwidth_hz = [20e6, 40e6, 80e6][bw] if bw < 3 else 20e6

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc),
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth_hz,
            num_subcarriers=num_tones,
            num_antennas=nr,
            snr=float(rssi),
            metadata={
                'source': 'atheros_router',
                'format': 'binary',
                'channel': channel,
                'tx_antennas': nc,
            }
        )

    def _parse_intel_text_format(self, data_str: str) -> CSIData:
        """Parse Intel 5300 CSI text format."""
        content = data_str[10:]  # Remove 'INTEL_CSI:' prefix
        parts = content.split(',')

        if len(parts) < 6:
            raise CSIParseError("Incomplete Intel CSI data")

        timestamp = int(parts[0])
        rssi = int(parts[1])
        channel = int(parts[2])
        bandwidth = int(parts[3])
        num_antennas = int(parts[4])
        num_tones = int(parts[5])

        csi_values = [float(x) for x in parts[6:] if x.strip()]
        amplitude, phase = self._parse_complex_csi(csi_values, num_antennas, num_tones)

        frequency = 5.18e9 + (channel - 36) * 5e6 if channel > 14 else 2.412e9 + (channel - 1) * 5e6

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth * 1e6,
            num_subcarriers=num_tones,
            num_antennas=num_antennas,
            snr=float(rssi + 95),
            metadata={'source': 'intel_5300', 'channel': channel}
        )

    def _parse_intel_binary_format(self, raw_data: bytes) -> CSIData:
        """Parse Intel 5300 CSI Tool binary format."""
        # Intel format is more complex with BFEE (beamforming feedback) structure
        if len(raw_data) < 25:
            raise CSIParseError("Intel binary data too short")

        # BFEE header structure
        timestamp = struct.unpack('<Q', raw_data[0:8])[0]
        rssi_a, rssi_b, rssi_c = struct.unpack('<bbb', raw_data[8:11])
        noise = struct.unpack('<b', raw_data[11:12])[0]
        agc = struct.unpack('<B', raw_data[12:13])[0]
        antenna_sel = struct.unpack('<B', raw_data[13:14])[0]
        perm = struct.unpack('<BBB', raw_data[14:17])
        num_tones = struct.unpack('<B', raw_data[17:18])[0]
        nc = struct.unpack('<B', raw_data[18:19])[0]
        nr = struct.unpack('<B', raw_data[19:20])[0]

        # Parse CSI matrix
        csi_data = raw_data[20:]

        # Intel stores CSI in a packed format with variable bit width
        csi_complex = self._unpack_intel_csi(csi_data, nr, nc, num_tones)

        # Use first TX stream
        amplitude = np.abs(csi_complex[:, 0, :])
        phase = np.angle(csi_complex[:, 0, :])

        # Normalize
        max_amp = np.max(amplitude)
        if max_amp > 0:
            amplitude = amplitude / max_amp

        rssi_avg = (rssi_a + rssi_b + rssi_c) / 3

        return CSIData(
            timestamp=datetime.fromtimestamp(timestamp / 1e6, tz=timezone.utc),
            amplitude=amplitude,
            phase=phase,
            frequency=5.32e9,  # Default Intel channel
            bandwidth=40e6,
            num_subcarriers=num_tones,
            num_antennas=nr,
            snr=float(rssi_avg - noise),
            metadata={
                'source': 'intel_5300',
                'format': 'binary',
                'noise_floor': noise,
                'agc': agc,
            }
        )

    def _unpack_intel_csi(self, data: bytes, nr: int, nc: int, num_tones: int) -> np.ndarray:
        """Unpack Intel CSI data with bit manipulation."""
        csi = np.zeros((nr, nc, num_tones), dtype=np.complex128)

        # Intel uses packed 10-bit values
        bits_per_sample = 10
        samples_needed = nr * nc * num_tones * 2  # real + imag

        # Simple unpacking (actual Intel format is more complex)
        idx = 0
        for tone in range(num_tones):
            for nc_idx in range(nc):
                for nr_idx in range(nr):
                    if idx + 2 <= len(data):
                        # Approximate unpacking
                        real = int.from_bytes(data[idx:idx+1], 'little', signed=True)
                        imag = int.from_bytes(data[idx+1:idx+2], 'little', signed=True)
                        csi[nr_idx, nc_idx, tone] = complex(real, imag)
                        idx += 2

        return csi

    def _parse_complex_csi(
        self,
        values: List[float],
        num_antennas: int,
        num_tones: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse complex CSI values from real/imag pairs."""
        expected_len = num_antennas * num_tones * 2

        if len(values) < expected_len:
            # Pad with zeros
            values = values + [0.0] * (expected_len - len(values))

        csi_complex = np.zeros((num_antennas, num_tones), dtype=np.complex128)

        for ant in range(num_antennas):
            for tone in range(num_tones):
                idx = (ant * num_tones + tone) * 2
                if idx + 1 < len(values):
                    csi_complex[ant, tone] = complex(values[idx], values[idx + 1])

        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)

        # Normalize
        max_amp = np.max(amplitude)
        if max_amp > 0:
            amplitude = amplitude / max_amp

        return amplitude, phase


class CSIExtractor:
    """Main CSI data extractor supporting multiple hardware types."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize CSI extractor.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.hardware_type = config['hardware_type']
        self.sampling_rate = config['sampling_rate']
        self.buffer_size = config['buffer_size']
        self.timeout = config['timeout']
        self.validation_enabled = config.get('validation_enabled', True)
        self.retry_attempts = config.get('retry_attempts', 3)

        # State management
        self.is_connected = False
        self.is_streaming = False
        self._connection = None

        # Create appropriate parser
        if self.hardware_type == 'esp32':
            self.parser = ESP32CSIParser()
        elif self.hardware_type in ('router', 'atheros', 'intel'):
            self.parser = RouterCSIParser()
        else:
            raise ValueError(f"Unsupported hardware type: {self.hardware_type}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_fields = ['hardware_type', 'sampling_rate', 'buffer_size', 'timeout']
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")

        if config['sampling_rate'] <= 0:
            raise ValueError("sampling_rate must be positive")

        if config['buffer_size'] <= 0:
            raise ValueError("buffer_size must be positive")

        if config['timeout'] <= 0:
            raise ValueError("timeout must be positive")

    async def connect(self) -> bool:
        """Establish connection to CSI hardware."""
        try:
            success = await self._establish_hardware_connection()
            self.is_connected = success
            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to hardware: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from CSI hardware."""
        if self.is_connected:
            await self._close_hardware_connection()
            self.is_connected = False

    async def extract_csi(self) -> CSIData:
        """Extract CSI data from hardware."""
        if not self.is_connected:
            raise CSIParseError("Not connected to hardware")

        for attempt in range(self.retry_attempts):
            try:
                raw_data = await self._read_raw_data()
                csi_data = self.parser.parse(raw_data)

                if self.validation_enabled:
                    self.validate_csi_data(csi_data)

                return csi_data

            except ConnectionError as e:
                if attempt < self.retry_attempts - 1:
                    self.logger.warning(f"Extraction attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(0.1)
                else:
                    raise CSIParseError(f"Extraction failed after {self.retry_attempts} attempts: {e}")

    def validate_csi_data(self, csi_data: CSIData) -> bool:
        """Validate CSI data structure and values."""
        if csi_data.amplitude.size == 0:
            raise CSIValidationError("Empty amplitude data")

        if csi_data.phase.size == 0:
            raise CSIValidationError("Empty phase data")

        if csi_data.frequency <= 0:
            raise CSIValidationError("Invalid frequency")

        if csi_data.bandwidth <= 0:
            raise CSIValidationError("Invalid bandwidth")

        if csi_data.num_subcarriers <= 0:
            raise CSIValidationError("Invalid number of subcarriers")

        if csi_data.num_antennas <= 0:
            raise CSIValidationError("Invalid number of antennas")

        if csi_data.snr < -50 or csi_data.snr > 100:
            raise CSIValidationError("Invalid SNR value")

        return True

    async def start_streaming(self, callback: Callable[[CSIData], None]) -> None:
        """Start streaming CSI data."""
        self.is_streaming = True

        try:
            while self.is_streaming:
                csi_data = await self.extract_csi()
                callback(csi_data)
                await asyncio.sleep(1.0 / self.sampling_rate)
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
        finally:
            self.is_streaming = False

    def stop_streaming(self) -> None:
        """Stop streaming CSI data."""
        self.is_streaming = False

    async def _establish_hardware_connection(self) -> bool:
        """Establish connection to hardware."""
        connection_config = self.config.get('connection', {})

        if self.hardware_type == 'esp32':
            # Serial or network connection for ESP32
            port = connection_config.get('port', '/dev/ttyUSB0')
            baudrate = connection_config.get('baudrate', 115200)

            try:
                import serial_asyncio
                reader, writer = await serial_asyncio.open_serial_connection(
                    url=port, baudrate=baudrate
                )
                self._connection = (reader, writer)
                return True
            except ImportError:
                self.logger.warning("serial_asyncio not available, using mock connection")
                return True
            except Exception as e:
                self.logger.error(f"Serial connection failed: {e}")
                return False

        elif self.hardware_type in ('router', 'atheros', 'intel'):
            # Network connection for router
            host = connection_config.get('host', '192.168.1.1')
            port = connection_config.get('port', 5500)

            try:
                reader, writer = await asyncio.open_connection(host, port)
                self._connection = (reader, writer)
                return True
            except Exception as e:
                self.logger.error(f"Network connection failed: {e}")
                return False

        return False

    async def _close_hardware_connection(self) -> None:
        """Close hardware connection."""
        if self._connection:
            try:
                reader, writer = self._connection
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None

    async def _read_raw_data(self) -> bytes:
        """Read raw data from hardware."""
        if self._connection:
            reader, writer = self._connection
            try:
                # Read until newline or buffer size
                data = await asyncio.wait_for(
                    reader.readline(),
                    timeout=self.timeout
                )
                return data
            except asyncio.TimeoutError:
                raise ConnectionError("Read timeout")
        else:
            # Mock data for testing when no real connection
            raise ConnectionError("No active connection")