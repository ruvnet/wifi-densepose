"""
Router interface for WiFi CSI data collection.

Supports multiple router types:
- OpenWRT routers with Atheros CSI Tool
- DD-WRT routers with custom CSI extraction
- Custom firmware routers with raw CSI access
"""

import logging
import asyncio
import struct
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import numpy as np

try:
    import asyncssh
    HAS_ASYNCSSH = True
except ImportError:
    HAS_ASYNCSSH = False

logger = logging.getLogger(__name__)


class RouterInterface:
    """Interface for connecting to WiFi routers and collecting CSI data."""
    
    def __init__(
        self,
        router_id: str,
        host: str,
        port: int = 22,
        username: str = "admin",
        password: str = "",
        interface: str = "wlan0",
        mock_mode: bool = False
    ):
        """Initialize router interface.
        
        Args:
            router_id: Unique identifier for the router
            host: Router IP address or hostname
            port: SSH port for connection
            username: SSH username
            password: SSH password
            interface: WiFi interface name
            mock_mode: Whether to use mock data instead of real connection
        """
        self.router_id = router_id
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.interface = interface
        self.mock_mode = mock_mode
        
        self.logger = logging.getLogger(f"{__name__}.{router_id}")
        
        # Connection state
        self.is_connected = False
        self.connection = None
        self.last_error = None
        
        # Data collection state
        self.last_data_time = None
        self.error_count = 0
        self.sample_count = 0
        
        # Mock data generation
        self.mock_data_generator = None
        if mock_mode:
            self._initialize_mock_generator()
    
    def _initialize_mock_generator(self):
        """Initialize mock data generator."""
        self.mock_data_generator = {
            'phase': 0,
            'amplitude_base': 1.0,
            'frequency': 0.1,
            'noise_level': 0.1
        }
    
    async def connect(self):
        """Connect to the router via SSH."""
        if self.mock_mode:
            self.is_connected = True
            self.logger.info(f"Mock connection established to router {self.router_id}")
            return

        if not HAS_ASYNCSSH:
            self.logger.warning("asyncssh not available, falling back to mock mode")
            self.mock_mode = True
            self._initialize_mock_generator()
            self.is_connected = True
            return

        try:
            self.logger.info(f"Connecting to router {self.router_id} at {self.host}:{self.port}")

            # Establish SSH connection
            self.connection = await asyncssh.connect(
                self.host,
                port=self.port,
                username=self.username,
                password=self.password if self.password else None,
                known_hosts=None,  # Disable host key checking for embedded devices
                connect_timeout=10
            )

            # Verify connection by checking router type
            await self._detect_router_type()

            self.is_connected = True
            self.error_count = 0
            self.logger.info(f"Connected to router {self.router_id}")

        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            self.logger.error(f"Failed to connect to router {self.router_id}: {e}")
            raise

    async def _detect_router_type(self):
        """Detect router firmware type and CSI capabilities."""
        if not self.connection:
            return

        try:
            # Check for OpenWRT
            result = await self.connection.run('cat /etc/openwrt_release 2>/dev/null || echo ""', check=False)
            if 'OpenWrt' in result.stdout:
                self.router_type = 'openwrt'
                self.logger.info(f"Detected OpenWRT router: {self.router_id}")
                return

            # Check for DD-WRT
            result = await self.connection.run('nvram get DD_BOARD 2>/dev/null || echo ""', check=False)
            if result.stdout.strip():
                self.router_type = 'ddwrt'
                self.logger.info(f"Detected DD-WRT router: {self.router_id}")
                return

            # Check for Atheros CSI Tool
            result = await self.connection.run('which csi_tool 2>/dev/null || echo ""', check=False)
            if result.stdout.strip():
                self.csi_tool_path = result.stdout.strip()
                self.router_type = 'atheros_csi'
                self.logger.info(f"Detected Atheros CSI Tool on router: {self.router_id}")
                return

            # Default to generic Linux
            self.router_type = 'generic'
            self.logger.info(f"Generic Linux router: {self.router_id}")

        except Exception as e:
            self.logger.warning(f"Could not detect router type: {e}")
            self.router_type = 'unknown'
    
    async def disconnect(self):
        """Disconnect from the router."""
        try:
            if self.connection:
                # Close SSH connection
                self.connection = None
            
            self.is_connected = False
            self.logger.info(f"Disconnected from router {self.router_id}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from router {self.router_id}: {e}")
    
    async def reconnect(self):
        """Reconnect to the router."""
        await self.disconnect()
        await asyncio.sleep(1)  # Wait before reconnecting
        await self.connect()
    
    async def get_csi_data(self) -> Optional[np.ndarray]:
        """Get CSI data from the router.
        
        Returns:
            CSI data as numpy array, or None if no data available
        """
        if not self.is_connected:
            raise RuntimeError(f"Router {self.router_id} is not connected")
        
        try:
            if self.mock_mode:
                csi_data = self._generate_mock_csi_data()
            else:
                csi_data = await self._collect_real_csi_data()
            
            if csi_data is not None:
                self.last_data_time = datetime.now()
                self.sample_count += 1
                self.error_count = 0
            
            return csi_data
            
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            self.logger.error(f"Error getting CSI data from router {self.router_id}: {e}")
            return None
    
    def _generate_mock_csi_data(self) -> np.ndarray:
        """Generate mock CSI data for testing."""
        # Simulate CSI data with realistic characteristics
        num_subcarriers = 64
        num_antennas = 4
        num_samples = 100
        
        # Update mock generator state
        self.mock_data_generator['phase'] += self.mock_data_generator['frequency']
        
        # Generate amplitude and phase data
        time_axis = np.linspace(0, 1, num_samples)
        
        # Create realistic CSI patterns
        csi_data = np.zeros((num_antennas, num_subcarriers, num_samples), dtype=complex)
        
        for antenna in range(num_antennas):
            for subcarrier in range(num_subcarriers):
                # Base signal with some variation per antenna/subcarrier
                amplitude = (
                    self.mock_data_generator['amplitude_base'] * 
                    (1 + 0.2 * np.sin(2 * np.pi * subcarrier / num_subcarriers)) *
                    (1 + 0.1 * antenna)
                )
                
                # Phase with spatial and frequency variation
                phase_offset = (
                    self.mock_data_generator['phase'] +
                    2 * np.pi * subcarrier / num_subcarriers +
                    np.pi * antenna / num_antennas
                )
                
                # Add some movement simulation
                movement_freq = 0.5  # Hz
                movement_amplitude = 0.3
                movement = movement_amplitude * np.sin(2 * np.pi * movement_freq * time_axis)
                
                # Generate complex signal
                signal_amplitude = amplitude * (1 + movement)
                signal_phase = phase_offset + movement * 0.5
                
                # Add noise
                noise_real = np.random.normal(0, self.mock_data_generator['noise_level'], num_samples)
                noise_imag = np.random.normal(0, self.mock_data_generator['noise_level'], num_samples)
                noise = noise_real + 1j * noise_imag
                
                # Create complex signal
                signal = signal_amplitude * np.exp(1j * signal_phase) + noise
                csi_data[antenna, subcarrier, :] = signal
        
        return csi_data
    
    async def _collect_real_csi_data(self) -> Optional[np.ndarray]:
        """Collect real CSI data from router via SSH.

        Supports multiple CSI extraction methods:
        - Atheros CSI Tool (ath9k/ath10k)
        - Custom kernel module reading
        - Proc filesystem access
        - Raw device file reading

        Returns:
            Numpy array of complex CSI values or None on failure
        """
        if not self.connection:
            self.logger.error("No SSH connection available")
            return None

        try:
            router_type = getattr(self, 'router_type', 'unknown')

            if router_type == 'atheros_csi':
                return await self._collect_atheros_csi()
            elif router_type == 'openwrt':
                return await self._collect_openwrt_csi()
            else:
                return await self._collect_generic_csi()

        except Exception as e:
            self.logger.error(f"Error collecting CSI data: {e}")
            self.error_count += 1
            return None

    async def _collect_atheros_csi(self) -> Optional[np.ndarray]:
        """Collect CSI using Atheros CSI Tool."""
        csi_tool = getattr(self, 'csi_tool_path', '/usr/bin/csi_tool')

        try:
            # Read single CSI sample
            result = await self.connection.run(
                f'{csi_tool} -i {self.interface} -c 1 -f /tmp/csi_sample.dat && '
                f'cat /tmp/csi_sample.dat | base64',
                check=True,
                timeout=5
            )

            # Decode base64 CSI data
            import base64
            csi_bytes = base64.b64decode(result.stdout.strip())

            return self._parse_atheros_csi_bytes(csi_bytes)

        except Exception as e:
            self.logger.error(f"Atheros CSI collection failed: {e}")
            return None

    async def _collect_openwrt_csi(self) -> Optional[np.ndarray]:
        """Collect CSI from OpenWRT with CSI support."""
        try:
            # Try reading from debugfs (common CSI location)
            result = await self.connection.run(
                f'cat /sys/kernel/debug/ieee80211/phy0/ath9k/csi 2>/dev/null | head -c 4096 | base64',
                check=False,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                import base64
                csi_bytes = base64.b64decode(result.stdout.strip())
                return self._parse_atheros_csi_bytes(csi_bytes)

            # Try alternate location
            result = await self.connection.run(
                f'cat /proc/csi 2>/dev/null | head -c 4096 | base64',
                check=False,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                import base64
                csi_bytes = base64.b64decode(result.stdout.strip())
                return self._parse_generic_csi_bytes(csi_bytes)

            self.logger.warning("No CSI data available from OpenWRT paths")
            return None

        except Exception as e:
            self.logger.error(f"OpenWRT CSI collection failed: {e}")
            return None

    async def _collect_generic_csi(self) -> Optional[np.ndarray]:
        """Collect CSI using generic Linux methods."""
        try:
            # Try iw command for station info (not real CSI but channel info)
            result = await self.connection.run(
                f'iw dev {self.interface} survey dump 2>/dev/null || echo ""',
                check=False,
                timeout=5
            )

            if result.stdout.strip():
                # Parse survey data for channel metrics
                return self._parse_survey_data(result.stdout)

            self.logger.warning("No CSI data available via generic methods")
            return None

        except Exception as e:
            self.logger.error(f"Generic CSI collection failed: {e}")
            return None

    def _parse_atheros_csi_bytes(self, data: bytes) -> Optional[np.ndarray]:
        """Parse Atheros CSI Tool binary format.

        Format:
        - 4 bytes: magic (0x11111111)
        - 8 bytes: timestamp
        - 2 bytes: channel
        - 1 byte: bandwidth
        - 1 byte: num_rx_antennas
        - 1 byte: num_tx_antennas
        - 1 byte: num_tones
        - 2 bytes: RSSI
        - Remaining: CSI matrix as int16 I/Q pairs
        """
        if len(data) < 20:
            return None

        try:
            magic = struct.unpack('<I', data[0:4])[0]
            if magic != 0x11111111:
                # Try different offset or format
                return self._parse_generic_csi_bytes(data)

            # Parse header
            timestamp = struct.unpack('<Q', data[4:12])[0]
            channel = struct.unpack('<H', data[12:14])[0]
            bw = struct.unpack('<B', data[14:15])[0]
            nr = struct.unpack('<B', data[15:16])[0]
            nc = struct.unpack('<B', data[16:17])[0]
            num_tones = struct.unpack('<B', data[17:18])[0]

            if nr == 0 or num_tones == 0:
                return None

            # Parse CSI matrix
            csi_data = data[20:]
            csi_matrix = np.zeros((nr, num_tones), dtype=complex)

            for ant in range(nr):
                for tone in range(num_tones):
                    offset = (ant * num_tones + tone) * 4
                    if offset + 4 <= len(csi_data):
                        real, imag = struct.unpack('<hh', csi_data[offset:offset+4])
                        csi_matrix[ant, tone] = complex(real, imag)

            return csi_matrix

        except Exception as e:
            self.logger.error(f"Error parsing Atheros CSI: {e}")
            return None

    def _parse_generic_csi_bytes(self, data: bytes) -> Optional[np.ndarray]:
        """Parse generic binary CSI format."""
        if len(data) < 8:
            return None

        try:
            # Assume simple format: int16 I/Q pairs
            num_samples = len(data) // 4
            if num_samples == 0:
                return None

            # Default to 56 subcarriers (20MHz), adjust antennas
            num_tones = min(56, num_samples)
            num_antennas = max(1, num_samples // num_tones)

            csi_matrix = np.zeros((num_antennas, num_tones), dtype=complex)

            for i in range(min(num_samples, num_antennas * num_tones)):
                offset = i * 4
                if offset + 4 <= len(data):
                    real, imag = struct.unpack('<hh', data[offset:offset+4])
                    ant = i // num_tones
                    tone = i % num_tones
                    if ant < num_antennas and tone < num_tones:
                        csi_matrix[ant, tone] = complex(real, imag)

            return csi_matrix

        except Exception as e:
            self.logger.error(f"Error parsing generic CSI: {e}")
            return None

    def _parse_survey_data(self, survey_output: str) -> Optional[np.ndarray]:
        """Parse iw survey dump output to extract channel metrics.

        This isn't true CSI but provides per-channel noise and activity data
        that can be used as a fallback.
        """
        try:
            lines = survey_output.strip().split('\n')
            noise_values = []
            busy_values = []

            for line in lines:
                if 'noise:' in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'dBm' and i > 0:
                            try:
                                noise_values.append(float(parts[i-1]))
                            except ValueError:
                                pass
                elif 'channel busy time:' in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'ms' and i > 0:
                            try:
                                busy_values.append(float(parts[i-1]))
                            except ValueError:
                                pass

            if noise_values:
                # Create pseudo-CSI from noise measurements
                num_channels = len(noise_values)
                csi_matrix = np.zeros((1, max(56, num_channels)), dtype=complex)

                for i, noise in enumerate(noise_values):
                    # Convert noise dBm to amplitude (simplified)
                    amplitude = 10 ** (noise / 20)
                    phase = 0 if i >= len(busy_values) else busy_values[i] / 1000 * np.pi
                    csi_matrix[0, i] = amplitude * np.exp(1j * phase)

                return csi_matrix

            return None

        except Exception as e:
            self.logger.error(f"Error parsing survey data: {e}")
            return None
    
    async def check_health(self) -> bool:
        """Check if the router connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # In mock mode, always healthy
            if self.mock_mode:
                return True
            
            # For real connections, we could ping the router or check SSH connection
            # For now, consider healthy if error count is low
            return self.error_count < 5
            
        except Exception as e:
            self.logger.error(f"Error checking health of router {self.router_id}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get router status information.
        
        Returns:
            Dictionary containing router status
        """
        return {
            "router_id": self.router_id,
            "connected": self.is_connected,
            "mock_mode": self.mock_mode,
            "last_data_time": self.last_data_time.isoformat() if self.last_data_time else None,
            "error_count": self.error_count,
            "sample_count": self.sample_count,
            "last_error": self.last_error,
            "configuration": {
                "host": self.host,
                "port": self.port,
                "username": self.username,
                "interface": self.interface
            }
        }
    
    async def get_router_info(self) -> Dict[str, Any]:
        """Get router hardware information.
        
        Returns:
            Dictionary containing router information
        """
        if self.mock_mode:
            return {
                "model": "Mock Router",
                "firmware": "1.0.0-mock",
                "wifi_standard": "802.11ac",
                "antennas": 4,
                "supported_bands": ["2.4GHz", "5GHz"],
                "csi_capabilities": {
                    "max_subcarriers": 64,
                    "max_antennas": 4,
                    "sampling_rate": 1000
                }
            }
        
        # For real routers, this would query the actual hardware
        return {
            "model": "Unknown",
            "firmware": "Unknown",
            "wifi_standard": "Unknown",
            "antennas": 1,
            "supported_bands": ["Unknown"],
            "csi_capabilities": {
                "max_subcarriers": 64,
                "max_antennas": 1,
                "sampling_rate": 100
            }
        }
    
    async def configure_csi_collection(self, config: Dict[str, Any]) -> bool:
        """Configure CSI data collection parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration successful, False otherwise
        """
        try:
            if self.mock_mode:
                # Update mock generator parameters
                if 'sampling_rate' in config:
                    self.mock_data_generator['frequency'] = config['sampling_rate'] / 1000.0
                
                if 'noise_level' in config:
                    self.mock_data_generator['noise_level'] = config['noise_level']
                
                self.logger.info(f"Mock CSI collection configured for router {self.router_id}")
                return True
            
            # For real routers, this would send configuration commands
            self.logger.warning("Real CSI configuration not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error configuring CSI collection for router {self.router_id}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router interface metrics.
        
        Returns:
            Dictionary containing metrics
        """
        uptime = 0
        if self.last_data_time:
            uptime = (datetime.now() - self.last_data_time).total_seconds()
        
        success_rate = 0
        if self.sample_count > 0:
            success_rate = (self.sample_count - self.error_count) / self.sample_count
        
        return {
            "router_id": self.router_id,
            "sample_count": self.sample_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "uptime_seconds": uptime,
            "is_connected": self.is_connected,
            "mock_mode": self.mock_mode
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.error_count = 0
        self.sample_count = 0
        self.last_error = None
        self.logger.info(f"Statistics reset for router {self.router_id}")