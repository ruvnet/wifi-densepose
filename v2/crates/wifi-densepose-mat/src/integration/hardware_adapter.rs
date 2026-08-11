//! Adapter for wifi-densepose-hardware crate with real hardware support.
//!
//! This module provides adapters for various WiFi CSI hardware:
//! - ESP32 with CSI support via serial communication
//! - Intel 5300 NIC with Linux CSI Tool
#![allow(missing_docs)]
//! - Atheros CSI extraction via ath9k/ath10k drivers
//!
//! # Example
//!
//! ```ignore
//! use wifi_densepose_mat::integration::{HardwareAdapter, HardwareConfig, DeviceType};
//!
//! let config = HardwareConfig::esp32("/dev/ttyUSB0", 921600);
//! let mut adapter = HardwareAdapter::with_config(config);
//! adapter.initialize().await?;
//!
//! // Start streaming CSI data
//! let mut stream = adapter.start_csi_stream().await?;
//! while let Some(reading) = stream.next().await {
//!     // Process CSI data
//! }
//! ```

use super::AdapterError;
use crate::domain::SensorPosition;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

/// Hardware configuration for CSI devices
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Device type selection
    pub device_type: DeviceType,
    /// Device-specific settings
    pub device_settings: DeviceSettings,
    /// Buffer size for CSI data
    pub buffer_size: usize,
    /// Whether to enable raw mode (minimal processing)
    pub raw_mode: bool,
    /// Sample rate override (Hz, 0 for device default)
    pub sample_rate_override: u32,
    /// Channel configuration
    pub channel_config: ChannelConfig,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Simulated,
            device_settings: DeviceSettings::Simulated,
            buffer_size: 4096,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig::default(),
        }
    }
}

impl HardwareConfig {
    /// Create configuration for ESP32 via serial
    pub fn esp32(serial_port: &str, baud_rate: u32) -> Self {
        Self {
            device_type: DeviceType::Esp32,
            device_settings: DeviceSettings::Serial(SerialSettings {
                port: serial_port.to_string(),
                baud_rate,
                data_bits: 8,
                stop_bits: 1,
                parity: Parity::None,
                flow_control: FlowControl::None,
                read_timeout_ms: 1000,
            }),
            buffer_size: 2048,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig::default(),
        }
    }

    /// Create configuration for Intel 5300 NIC
    pub fn intel_5300(interface: &str) -> Self {
        Self {
            device_type: DeviceType::Intel5300,
            device_settings: DeviceSettings::NetworkInterface(NetworkInterfaceSettings {
                interface: interface.to_string(),
                monitor_mode: true,
                channel: 6,
                bandwidth: Bandwidth::HT20,
                antenna_config: AntennaConfig::default(),
            }),
            buffer_size: 8192,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig {
                channel: 6,
                bandwidth: Bandwidth::HT20,
                num_subcarriers: 30, // Intel 5300 provides 30 subcarriers
            },
        }
    }

    /// Create configuration for Atheros NIC
    pub fn atheros(interface: &str, driver: AtherosDriver) -> Self {
        let num_subcarriers = match driver {
            AtherosDriver::Ath9k => 56,
            AtherosDriver::Ath10k => 114,
            AtherosDriver::Ath11k => 234,
        };

        Self {
            device_type: DeviceType::Atheros(driver),
            device_settings: DeviceSettings::NetworkInterface(NetworkInterfaceSettings {
                interface: interface.to_string(),
                monitor_mode: true,
                channel: 36,
                bandwidth: Bandwidth::HT40,
                antenna_config: AntennaConfig::default(),
            }),
            buffer_size: 16384,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig {
                channel: 36,
                bandwidth: Bandwidth::HT40,
                num_subcarriers,
            },
        }
    }

    /// Create configuration for deterministic FeitCSI capture replay
    /// (wideband 802.11ax records from Intel AX200/AX210, ADR-289).
    pub fn feitcsi_replay(file_path: &str) -> Self {
        Self::feitcsi(file_path, FeitCsiMode::FileReplay)
    }

    /// Create configuration for streaming FeitCSI ingest from a path/pipe an
    /// external FeitCSI process writes to (no NIC configuration in-crate).
    pub fn feitcsi_stream(path: &str) -> Self {
        Self::feitcsi(path, FeitCsiMode::Stream)
    }

    fn feitcsi(path: &str, mode: FeitCsiMode) -> Self {
        Self {
            device_type: DeviceType::FeitCsi,
            device_settings: DeviceSettings::FeitCsi(FeitCsiSettings {
                path: path.to_string(),
                mode,
                band: WifiBand::Band5GHz,
                channel: 36,
                loop_playback: false,
                pipeline_subcarriers: None,
            }),
            buffer_size: 8192,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig {
                channel: 36,
                bandwidth: Bandwidth::VHT160,
                // Native width travels with each frame; this is only the
                // configured expectation (802.11ax HE 160 MHz = 1992 tones).
                num_subcarriers: 1992,
            },
        }
    }

    /// Create configuration for UDP receiver (generic CSI)
    pub fn udp_receiver(bind_addr: &str, port: u16) -> Self {
        Self {
            device_type: DeviceType::UdpReceiver,
            device_settings: DeviceSettings::Udp(UdpSettings {
                bind_address: bind_addr.to_string(),
                port,
                multicast_group: None,
                buffer_size: 65536,
            }),
            buffer_size: 8192,
            raw_mode: false,
            sample_rate_override: 0,
            channel_config: ChannelConfig::default(),
        }
    }
}

/// Supported device types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    /// ESP32 with ESP-CSI firmware
    Esp32,
    /// Intel 5300 NIC with Linux CSI Tool
    Intel5300,
    /// Atheros NIC with specific driver
    Atheros(AtherosDriver),
    /// Generic UDP CSI receiver
    UdpReceiver,
    /// PCAP file replay
    PcapFile,
    /// FeitCSI wideband 802.11ax records from Intel AX200/AX210 (ADR-289):
    /// file replay of a recorded capture, or a path/pipe an external FeitCSI
    /// process writes to. RuView never configures the NIC — FeitCSI's own
    /// tooling owns capture, per least-authority.
    FeitCsi,
    /// Simulated device (for testing)
    Simulated,
}

/// Atheros driver variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtherosDriver {
    /// ath9k driver (legacy, 56 subcarriers)
    Ath9k,
    /// ath10k driver (802.11ac, 114 subcarriers)
    Ath10k,
    /// ath11k driver (802.11ax, 234 subcarriers)
    Ath11k,
}

/// Device-specific settings
#[derive(Debug, Clone)]
pub enum DeviceSettings {
    /// Serial port settings (ESP32)
    Serial(SerialSettings),
    /// Network interface settings (Intel 5300, Atheros)
    NetworkInterface(NetworkInterfaceSettings),
    /// UDP receiver settings
    Udp(UdpSettings),
    /// PCAP file settings
    Pcap(PcapSettings),
    /// FeitCSI capture replay / stream settings
    FeitCsi(FeitCsiSettings),
    /// Simulated device (no real hardware)
    Simulated,
}

/// FeitCSI ingest mode (ADR-289).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeitCsiMode {
    /// Deterministic replay of a recorded capture file.
    FileReplay,
    /// Live stream read from a path/pipe an external FeitCSI process writes
    /// to. RuView performs no NIC configuration; the external tool owns it.
    Stream,
}

/// FeitCSI source settings (ADR-289).
///
/// The FeitCSI record header carries bandwidth (via the iwlwifi rate flags)
/// but not channel/band — the capture configuration owns those — so band and
/// channel are supplied here and stamped into frame metadata.
#[derive(Debug, Clone)]
pub struct FeitCsiSettings {
    /// Path to the recorded capture (FileReplay) or the file/FIFO the
    /// external FeitCSI process appends records to (Stream).
    pub path: String,
    /// Ingest mode.
    pub mode: FeitCsiMode,
    /// Radio band the capture was taken on (2.4/5/6 GHz).
    pub band: WifiBand,
    /// WiFi channel the capture was taken on.
    pub channel: u8,
    /// Restart from the beginning when file replay reaches the end.
    pub loop_playback: bool,
    /// When `Some(n)`, frames are explicitly converted from their native
    /// subcarrier count to `n` via the interpolation path, and the mapping is
    /// recorded in `CsiMetadata::wideband`. `None` keeps native width.
    pub pipeline_subcarriers: Option<usize>,
}

/// Serial port configuration
#[derive(Debug, Clone)]
pub struct SerialSettings {
    /// Serial port path
    pub port: String,
    /// Baud rate
    pub baud_rate: u32,
    /// Data bits (5-8)
    pub data_bits: u8,
    /// Stop bits (1, 2)
    pub stop_bits: u8,
    /// Parity setting
    pub parity: Parity,
    /// Flow control
    pub flow_control: FlowControl,
    /// Read timeout in milliseconds
    pub read_timeout_ms: u64,
}

/// Parity options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    None,
    Odd,
    Even,
}

/// Flow control options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowControl {
    None,
    Hardware,
    Software,
}

/// Network interface configuration
#[derive(Debug, Clone)]
pub struct NetworkInterfaceSettings {
    /// Interface name (e.g., "wlan0")
    pub interface: String,
    /// Enable monitor mode
    pub monitor_mode: bool,
    /// WiFi channel
    pub channel: u8,
    /// Channel bandwidth
    pub bandwidth: Bandwidth,
    /// Antenna configuration
    pub antenna_config: AntennaConfig,
}

/// Channel bandwidth options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bandwidth {
    /// 20 MHz (legacy)
    #[default]
    HT20,
    /// 40 MHz (802.11n)
    HT40,
    /// 80 MHz (802.11ac)
    VHT80,
    /// 160 MHz (802.11ac Wave 2)
    VHT160,
}

impl Bandwidth {
    /// Get number of subcarriers for this bandwidth
    pub fn subcarrier_count(&self) -> usize {
        match self {
            Bandwidth::HT20 => 56,
            Bandwidth::HT40 => 114,
            Bandwidth::VHT80 => 242,
            Bandwidth::VHT160 => 484,
        }
    }

    /// Channel bandwidth in MHz.
    pub fn mhz(&self) -> u16 {
        match self {
            Bandwidth::HT20 => 20,
            Bandwidth::HT40 => 40,
            Bandwidth::VHT80 => 80,
            Bandwidth::VHT160 => 160,
        }
    }
}

/// WiFi radio band (first-class frame metadata per ADR-289).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WifiBand {
    /// 2.4 GHz ISM band
    Band2_4GHz,
    /// 5 GHz band
    Band5GHz,
    /// 6 GHz band (802.11ax/Wi-Fi 6E and later)
    Band6GHz,
}

/// Record of an explicit native → pipeline subcarrier conversion, so
/// downstream consumers know the true spectral resolution of a frame and
/// how it was resampled (ADR-289 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubcarrierMapping {
    /// Native subcarrier count as captured.
    pub native: usize,
    /// Pipeline subcarrier count after conversion.
    pub pipeline: usize,
    /// Interpolation/decimation method used (e.g. "catmull-rom-cubic").
    pub method: &'static str,
}

/// Wideband spectral provenance metadata (ADR-289): band, bandwidth, native
/// subcarrier dimensionality, and any native → pipeline mapping applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WidebandMeta {
    /// Radio band the frame was captured on.
    pub band: WifiBand,
    /// Channel bandwidth in MHz (20–160).
    pub bandwidth_mhz: u16,
    /// Native subcarrier count of the capture (true spectral resolution).
    pub native_subcarriers: usize,
    /// Native → pipeline conversion record; `None` while the frame is still
    /// at native width.
    pub mapping: Option<SubcarrierMapping>,
}

/// Antenna configuration for MIMO
#[derive(Debug, Clone)]
pub struct AntennaConfig {
    /// Number of transmit antennas
    pub tx_antennas: u8,
    /// Number of receive antennas
    pub rx_antennas: u8,
    /// Enabled antenna mask
    pub antenna_mask: u8,
}

impl Default for AntennaConfig {
    fn default() -> Self {
        Self {
            tx_antennas: 1,
            rx_antennas: 3,
            antenna_mask: 0x07, // Enable antennas 0, 1, 2
        }
    }
}

/// UDP receiver settings
#[derive(Debug, Clone)]
pub struct UdpSettings {
    /// Bind address
    pub bind_address: String,
    /// Port number
    pub port: u16,
    /// Multicast group (optional)
    pub multicast_group: Option<String>,
    /// Socket buffer size
    pub buffer_size: usize,
}

/// PCAP file settings
#[derive(Debug, Clone)]
pub struct PcapSettings {
    /// Path to PCAP file
    pub file_path: String,
    /// Playback speed multiplier (1.0 = realtime)
    pub playback_speed: f64,
    /// Loop playback
    pub loop_playback: bool,
}

/// Channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// WiFi channel
    pub channel: u8,
    /// Bandwidth
    pub bandwidth: Bandwidth,
    /// Number of OFDM subcarriers
    pub num_subcarriers: usize,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            channel: 6,
            bandwidth: Bandwidth::HT20,
            num_subcarriers: 56,
        }
    }
}

/// Hardware adapter for sensor communication
pub struct HardwareAdapter {
    /// Configuration
    config: HardwareConfig,
    /// Connected sensors
    sensors: Vec<SensorInfo>,
    /// Whether hardware is initialized
    initialized: bool,
    /// CSI broadcast channel
    csi_broadcaster: Option<broadcast::Sender<CsiReadings>>,
    /// Device state (shared for async operations)
    state: Arc<RwLock<DeviceState>>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::Sender<()>>,
}

/// Internal device state
struct DeviceState {
    /// Whether streaming is active
    streaming: bool,
    /// Total packets received
    packets_received: u64,
    /// Packets with errors
    error_count: u64,
    /// Last error message
    last_error: Option<String>,
    /// Device-specific state
    device_state: DeviceSpecificState,
}

/// Device-specific runtime state
#[allow(dead_code)]
enum DeviceSpecificState {
    Esp32 {
        firmware_version: Option<String>,
        mac_address: Option<String>,
    },
    Intel5300 {
        bfee_count: u64,
    },
    Atheros {
        driver: AtherosDriver,
        csi_buf_ptr: Option<u64>,
    },
    FeitCsi {
        /// Byte offset into the capture for deterministic file replay.
        replay_offset: u64,
        /// Open handle for streaming mode (path/pipe written by an external
        /// FeitCSI process); opened lazily on first read.
        stream: Option<std::fs::File>,
    },
    Other,
}

/// Information about a connected sensor
#[derive(Debug, Clone)]
pub struct SensorInfo {
    /// Unique sensor ID
    pub id: String,
    /// Sensor position
    pub position: SensorPosition,
    /// Current status
    pub status: SensorStatus,
    /// Last RSSI reading (if available)
    pub last_rssi: Option<f64>,
    /// Battery level (0-100, if applicable)
    pub battery_level: Option<u8>,
    /// MAC address (if available)
    pub mac_address: Option<String>,
    /// Firmware version (if available)
    pub firmware_version: Option<String>,
}

/// Status of a sensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensorStatus {
    /// Sensor is connected and operational
    Connected,
    /// Sensor is disconnected
    Disconnected,
    /// Sensor is in error state
    Error,
    /// Sensor is initializing
    Initializing,
    /// Sensor battery is low
    LowBattery,
    /// Sensor is in standby mode
    Standby,
}

impl HardwareAdapter {
    /// Create a new hardware adapter with default configuration
    pub fn new() -> Self {
        Self::with_config(HardwareConfig::default())
    }

    /// Create a new hardware adapter with specific configuration
    pub fn with_config(config: HardwareConfig) -> Self {
        Self {
            config,
            sensors: Vec::new(),
            initialized: false,
            csi_broadcaster: None,
            state: Arc::new(RwLock::new(DeviceState {
                streaming: false,
                packets_received: 0,
                error_count: 0,
                last_error: None,
                device_state: DeviceSpecificState::Other,
            })),
            shutdown_tx: None,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &HardwareConfig {
        &self.config
    }

    /// Initialize hardware communication
    pub async fn initialize(&mut self) -> Result<(), AdapterError> {
        tracing::info!(
            "Initializing hardware adapter for {:?}",
            self.config.device_type
        );

        match &self.config.device_type {
            DeviceType::Esp32 => self.initialize_esp32().await?,
            DeviceType::Intel5300 => self.initialize_intel_5300().await?,
            DeviceType::Atheros(driver) => self.initialize_atheros(*driver).await?,
            DeviceType::UdpReceiver => self.initialize_udp().await?,
            DeviceType::PcapFile => self.initialize_pcap().await?,
            DeviceType::FeitCsi => self.initialize_feitcsi().await?,
            DeviceType::Simulated => self.initialize_simulated().await?,
        }

        // Create CSI broadcast channel
        let (tx, _) = broadcast::channel(self.config.buffer_size);
        self.csi_broadcaster = Some(tx);

        self.initialized = true;
        tracing::info!("Hardware adapter initialized successfully");
        Ok(())
    }

    /// Initialize ESP32 device
    async fn initialize_esp32(&mut self) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::Serial(s) => s,
            _ => {
                return Err(AdapterError::Config(
                    "ESP32 requires serial settings".into(),
                ))
            }
        };

        tracing::info!(
            "Initializing ESP32 on {} at {} baud",
            settings.port,
            settings.baud_rate
        );

        // Verify serial port exists
        #[cfg(unix)]
        {
            if !std::path::Path::new(&settings.port).exists() {
                return Err(AdapterError::Hardware(format!(
                    "Serial port {} not found",
                    settings.port
                )));
            }
        }

        // Update device state
        let mut state = self.state.write().await;
        state.device_state = DeviceSpecificState::Esp32 {
            firmware_version: None,
            mac_address: None,
        };

        Ok(())
    }

    /// Initialize Intel 5300 NIC
    async fn initialize_intel_5300(&mut self) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::NetworkInterface(s) => s,
            _ => {
                return Err(AdapterError::Config(
                    "Intel 5300 requires network interface settings".into(),
                ))
            }
        };

        tracing::info!(
            "Initializing Intel 5300 on interface {}",
            settings.interface
        );

        // Check if iwlwifi driver is loaded
        #[cfg(target_os = "linux")]
        {
            let output = tokio::process::Command::new("lsmod")
                .output()
                .await
                .map_err(|e| {
                    AdapterError::Hardware(format!("Failed to check kernel modules: {}", e))
                })?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stdout.contains("iwlwifi") {
                tracing::warn!("iwlwifi module not loaded - CSI extraction may not work");
            }
        }

        // Verify connector proc file exists (Linux CSI Tool)
        #[cfg(target_os = "linux")]
        {
            let connector_path = "/proc/net/connector";
            if !std::path::Path::new(connector_path).exists() {
                tracing::warn!("Connector proc file not found - install Linux CSI Tool");
            }
        }

        let mut state = self.state.write().await;
        state.device_state = DeviceSpecificState::Intel5300 { bfee_count: 0 };

        Ok(())
    }

    /// Initialize Atheros NIC
    async fn initialize_atheros(&mut self, driver: AtherosDriver) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::NetworkInterface(s) => s,
            _ => {
                return Err(AdapterError::Config(
                    "Atheros requires network interface settings".into(),
                ))
            }
        };

        tracing::info!(
            "Initializing Atheros ({:?}) on interface {}",
            driver,
            settings.interface
        );

        // Check for driver-specific debugfs entries
        #[cfg(target_os = "linux")]
        {
            let debugfs_path = format!(
                "/sys/kernel/debug/ieee80211/phy0/ath{}/csi",
                match driver {
                    AtherosDriver::Ath9k => "9k",
                    AtherosDriver::Ath10k => "10k",
                    AtherosDriver::Ath11k => "11k",
                }
            );

            if !std::path::Path::new(&debugfs_path).exists() {
                tracing::warn!(
                    "CSI debugfs path {} not found - CSI patched driver may not be installed",
                    debugfs_path
                );
            }
        }

        let mut state = self.state.write().await;
        state.device_state = DeviceSpecificState::Atheros {
            driver,
            csi_buf_ptr: None,
        };

        Ok(())
    }

    /// Initialize UDP receiver
    async fn initialize_udp(&mut self) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::Udp(s) => s,
            _ => {
                return Err(AdapterError::Config(
                    "UDP receiver requires UDP settings".into(),
                ))
            }
        };

        tracing::info!(
            "Initializing UDP receiver on {}:{}",
            settings.bind_address,
            settings.port
        );

        // Verify port is available
        let addr = format!("{}:{}", settings.bind_address, settings.port);
        let socket = tokio::net::UdpSocket::bind(&addr)
            .await
            .map_err(|e| AdapterError::Hardware(format!("Failed to bind UDP socket: {}", e)))?;

        // Join multicast group if specified
        if let Some(ref group) = settings.multicast_group {
            let multicast_addr: std::net::Ipv4Addr = group
                .parse()
                .map_err(|e| AdapterError::Config(format!("Invalid multicast address: {}", e)))?;

            socket
                .join_multicast_v4(multicast_addr, std::net::Ipv4Addr::UNSPECIFIED)
                .map_err(|e| {
                    AdapterError::Hardware(format!("Failed to join multicast group: {}", e))
                })?;
        }

        // Socket will be recreated when streaming starts
        drop(socket);

        Ok(())
    }

    /// Initialize PCAP file reader
    async fn initialize_pcap(&mut self) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::Pcap(s) => s,
            _ => return Err(AdapterError::Config("PCAP requires PCAP settings".into())),
        };

        tracing::info!("Initializing PCAP file reader: {}", settings.file_path);

        // Verify file exists
        if !std::path::Path::new(&settings.file_path).exists() {
            return Err(AdapterError::Hardware(format!(
                "PCAP file not found: {}",
                settings.file_path
            )));
        }

        Ok(())
    }

    /// Initialize FeitCSI file-replay / stream ingest (ADR-289).
    ///
    /// No privileged operations: RuView does not configure the NIC; the
    /// external FeitCSI tooling owns capture. This only validates the
    /// configured path.
    async fn initialize_feitcsi(&mut self) -> Result<(), AdapterError> {
        let settings = match &self.config.device_settings {
            DeviceSettings::FeitCsi(s) => s,
            _ => {
                return Err(AdapterError::Config(
                    "FeitCSI requires FeitCSI settings".into(),
                ))
            }
        };

        tracing::info!(
            "Initializing FeitCSI ingest ({:?}) from {}",
            settings.mode,
            settings.path
        );

        match settings.mode {
            FeitCsiMode::FileReplay => {
                if !std::path::Path::new(&settings.path).exists() {
                    return Err(AdapterError::Hardware(format!(
                        "FeitCSI capture file not found: {}",
                        settings.path
                    )));
                }
            }
            FeitCsiMode::Stream => {
                // The external process may create the pipe/file later; warn
                // rather than fail so start order is not constrained.
                if !std::path::Path::new(&settings.path).exists() {
                    tracing::warn!(
                        "FeitCSI stream path {} does not exist yet; will retry on read",
                        settings.path
                    );
                }
            }
        }

        let mut state = self.state.write().await;
        state.device_state = DeviceSpecificState::FeitCsi {
            replay_offset: 0,
            stream: None,
        };

        Ok(())
    }

    /// Initialize simulated device
    async fn initialize_simulated(&mut self) -> Result<(), AdapterError> {
        tracing::info!("Initializing simulated CSI device");
        Ok(())
    }

    /// Start CSI streaming
    pub async fn start_csi_stream(&mut self) -> Result<CsiStream, AdapterError> {
        if !self.initialized {
            return Err(AdapterError::Hardware("Hardware not initialized".into()));
        }

        let broadcaster = self
            .csi_broadcaster
            .as_ref()
            .ok_or_else(|| AdapterError::Hardware("CSI broadcaster not initialized".into()))?;

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Start device-specific streaming
        let tx = broadcaster.clone();
        let config = self.config.clone();
        let state = Arc::clone(&self.state);

        tokio::spawn(async move {
            Self::run_streaming_loop(config, tx, state, shutdown_rx).await;
        });

        // Update streaming state
        {
            let mut state = self.state.write().await;
            state.streaming = true;
        }

        let rx = broadcaster.subscribe();
        Ok(CsiStream { receiver: rx })
    }

    /// Stop CSI streaming
    pub async fn stop_csi_stream(&mut self) -> Result<(), AdapterError> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        let mut state = self.state.write().await;
        state.streaming = false;

        Ok(())
    }

    /// Internal streaming loop
    async fn run_streaming_loop(
        config: HardwareConfig,
        tx: broadcast::Sender<CsiReadings>,
        state: Arc<RwLock<DeviceState>>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        tracing::debug!("Starting CSI streaming loop for {:?}", config.device_type);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("CSI streaming shutdown requested");
                    break;
                }
                result = Self::read_csi_packet(&config, &state) => {
                    match result {
                        Ok(reading) => {
                            // Update packet count
                            {
                                let mut state = state.write().await;
                                state.packets_received += 1;
                            }

                            // Broadcast to subscribers
                            if tx.receiver_count() > 0 {
                                let _ = tx.send(reading);
                            }
                        }
                        Err(e) => {
                            let mut state = state.write().await;
                            state.error_count += 1;
                            state.last_error = Some(e.to_string());

                            if state.error_count > 100 {
                                tracing::error!("Too many CSI read errors, stopping stream");
                                break;
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!("CSI streaming loop ended");
    }

    /// Read a single CSI packet from the device
    async fn read_csi_packet(
        config: &HardwareConfig,
        state: &Arc<RwLock<DeviceState>>,
    ) -> Result<CsiReadings, AdapterError> {
        match &config.device_type {
            DeviceType::Esp32 => Self::read_esp32_csi(config).await,
            DeviceType::Intel5300 => Self::read_intel_5300_csi(config).await,
            DeviceType::Atheros(driver) => Self::read_atheros_csi(config, *driver).await,
            DeviceType::UdpReceiver => Self::read_udp_csi(config).await,
            DeviceType::PcapFile => Self::read_pcap_csi(config).await,
            DeviceType::FeitCsi => Self::read_feitcsi_csi(config, state).await,
            DeviceType::Simulated => Self::generate_simulated_csi(config).await,
        }
    }

    /// Read one wideband CSI frame from a FeitCSI capture or stream (ADR-289).
    ///
    /// Frames carry their native subcarrier count, bandwidth (20–160 MHz) and
    /// band (2.4/5/6 GHz) as metadata. When `pipeline_subcarriers` is
    /// configured, conversion to pipeline width happens explicitly via the
    /// interpolation path and the native → pipeline mapping is recorded in
    /// `CsiMetadata::wideband`.
    async fn read_feitcsi_csi(
        config: &HardwareConfig,
        state: &Arc<RwLock<DeviceState>>,
    ) -> Result<CsiReadings, AdapterError> {
        let settings = match &config.device_settings {
            DeviceSettings::FeitCsi(s) => s,
            _ => return Err(AdapterError::Config("Invalid settings for FeitCSI".into())),
        };

        let record = match settings.mode {
            FeitCsiMode::FileReplay => Self::read_feitcsi_replay(settings, state).await?,
            FeitCsiMode::Stream => Self::read_feitcsi_stream(settings, state).await?,
        };

        let readings = record.to_readings(settings.band, settings.channel);
        match settings.pipeline_subcarriers {
            Some(n) if n != readings.metadata.num_subcarriers => {
                super::feitcsi::resample_readings_to_pipeline(&readings, n)
            }
            _ => Ok(readings),
        }
    }

    /// Deterministic file replay: reads the record at the current byte offset
    /// and advances it, so the capture is walked once from start to end
    /// (looping when configured). Same input file ⇒ same record sequence.
    async fn read_feitcsi_replay(
        settings: &FeitCsiSettings,
        state: &Arc<RwLock<DeviceState>>,
    ) -> Result<super::feitcsi::FeitCsiRecord, AdapterError> {
        let offset = {
            let st = state.read().await;
            match &st.device_state {
                DeviceSpecificState::FeitCsi { replay_offset, .. } => *replay_offset,
                _ => 0,
            }
        };

        let path = settings.path.clone();
        let loop_playback = settings.loop_playback;
        let (record, new_offset) = tokio::task::spawn_blocking(
            move || -> Result<(super::feitcsi::FeitCsiRecord, u64), AdapterError> {
                use std::io::{Seek, SeekFrom};
                let mut file = std::fs::File::open(&path).map_err(|e| {
                    AdapterError::Hardware(format!("Failed to open FeitCSI capture {path}: {e}"))
                })?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(AdapterError::Io)?;
                match super::feitcsi::read_one_record(&mut file)? {
                    Some((rec, consumed)) => Ok((rec, offset + consumed as u64)),
                    None if loop_playback && offset != 0 => {
                        file.seek(SeekFrom::Start(0)).map_err(AdapterError::Io)?;
                        match super::feitcsi::read_one_record(&mut file)? {
                            Some((rec, consumed)) => Ok((rec, consumed as u64)),
                            None => Err(AdapterError::DataFormat(format!(
                                "FeitCSI capture {path} contains no records"
                            ))),
                        }
                    }
                    None => Err(AdapterError::HardwareUnavailable(format!(
                        "End of FeitCSI capture {path} (offset {offset})"
                    ))),
                }
            },
        )
        .await
        .map_err(|e| AdapterError::Hardware(format!("FeitCSI replay task failed: {e}")))??;

        let mut st = state.write().await;
        if let DeviceSpecificState::FeitCsi { replay_offset, .. } = &mut st.device_state {
            *replay_offset = new_offset;
        }
        Ok(record)
    }

    /// Streaming mode: hold the open handle across reads (a FIFO cannot be
    /// reopened per record) and block until one full record arrives. The
    /// blocking read runs on the blocking pool; if the surrounding stream
    /// loop is shut down mid-read, the orphaned task finishes on its own and
    /// the handle is reopened on the next read.
    async fn read_feitcsi_stream(
        settings: &FeitCsiSettings,
        state: &Arc<RwLock<DeviceState>>,
    ) -> Result<super::feitcsi::FeitCsiRecord, AdapterError> {
        let existing = {
            let mut st = state.write().await;
            match &mut st.device_state {
                DeviceSpecificState::FeitCsi { stream, .. } => stream.take(),
                _ => None,
            }
        };

        let path = settings.path.clone();
        let result = tokio::task::spawn_blocking(
            move || -> Result<(std::fs::File, super::feitcsi::FeitCsiRecord), AdapterError> {
                let mut file = match existing {
                    Some(f) => f,
                    None => std::fs::File::open(&path).map_err(|e| {
                        AdapterError::HardwareUnavailable(format!(
                            "FeitCSI stream {path} unavailable: {e}"
                        ))
                    })?,
                };
                match super::feitcsi::read_one_record(&mut file) {
                    Ok(Some((rec, _consumed))) => Ok((file, rec)),
                    Ok(None) => Err(AdapterError::HardwareUnavailable(format!(
                        "FeitCSI stream {path} closed (EOF)"
                    ))),
                    Err(e) => Err(e.into()),
                }
            },
        )
        .await
        .map_err(|e| AdapterError::Hardware(format!("FeitCSI stream task failed: {e}")))?;

        let (file, record) = result?;
        let mut st = state.write().await;
        if let DeviceSpecificState::FeitCsi { stream, .. } = &mut st.device_state {
            *stream = Some(file);
        }
        Ok(record)
    }

    /// Read CSI from ESP32 via serial.
    ///
    /// The ESP-CSI firmware emits newline-delimited `CSI_DATA,...` CSV records.
    /// We read raw bytes from the serial port and parse them with the real
    /// [`CsiParser`] (`csi_receiver::CsiParser::parse_esp32`). Serial byte I/O
    /// uses the workspace `serialport` crate when present; the parsing itself is
    /// shared with the standalone `SerialCsiReceiver`.
    async fn read_esp32_csi(config: &HardwareConfig) -> Result<CsiReadings, AdapterError> {
        let settings = match &config.device_settings {
            DeviceSettings::Serial(s) => s,
            _ => return Err(AdapterError::Config("Invalid settings for ESP32".into())),
        };

        // Read one newline-delimited record from the serial port.
        let line = Self::read_serial_line(settings).await?;
        // Parse with the real ESP32 parser (shared with csi_receiver).
        let parser = super::csi_receiver::CsiParser::new(
            super::csi_receiver::CsiPacketFormat::Esp32Csi,
        );
        let packet = parser.parse(&line)?;
        Ok(packet.into())
    }

    /// Read CSI from Intel 5300 NIC.
    ///
    /// HONEST hardware gating: extracting CSI from the Intel 5300 requires the
    /// patched `iwlwifi` driver and the Linux 802.11n CSI Tool exposing the
    /// netlink connector — neither is present in this environment. The BFEE wire
    /// format *parser* exists (`CsiParser::parse_intel_5300`), but there is no
    /// device to source bytes from, so we return a typed unavailable error
    /// rather than fabricating CSI. Feeding captured BFEE bytes through the
    /// parser directly is supported and tested in `csi_receiver`.
    async fn read_intel_5300_csi(_config: &HardwareConfig) -> Result<CsiReadings, AdapterError> {
        Err(AdapterError::HardwareUnavailable(
            "Intel 5300 CSI requires the patched iwlwifi driver + Linux 802.11n CSI Tool \
             (netlink connector); not available in this environment. The BFEE parser exists \
             (feed captured bytes via CsiParser::parse), but no live device is present."
                .into(),
        ))
    }

    /// Read CSI from Atheros NIC.
    ///
    /// HONEST hardware gating: Atheros CSI needs the ath9k/ath10k CSI-patched
    /// driver exposing the debugfs CSI buffer. The parser exists
    /// (`CsiParser::parse_atheros`) but there is no device/driver here, so we
    /// return a typed unavailable error instead of fake data.
    async fn read_atheros_csi(
        _config: &HardwareConfig,
        driver: AtherosDriver,
    ) -> Result<CsiReadings, AdapterError> {
        Err(AdapterError::HardwareUnavailable(format!(
            "Atheros {driver:?} CSI requires the CSI-patched ath driver exposing the debugfs CSI \
             buffer; not available in this environment. The parser exists (feed captured bytes \
             via CsiParser::parse), but no live device/driver is present."
        )))
    }

    /// Read CSI from a UDP socket (generic network CSI streaming).
    ///
    /// Binds the configured address, receives one datagram, and parses it with
    /// the real [`CsiParser`] (auto-detecting ESP32/Nexmon/JSON/etc). This is a
    /// genuine end-to-end path: a sender on the wire produces real CsiReadings.
    async fn read_udp_csi(config: &HardwareConfig) -> Result<CsiReadings, AdapterError> {
        let settings = match &config.device_settings {
            DeviceSettings::Udp(s) => s,
            _ => return Err(AdapterError::Config("Invalid settings for UDP".into())),
        };

        let addr = format!("{}:{}", settings.bind_address, settings.port);
        let socket = tokio::net::UdpSocket::bind(&addr)
            .await
            .map_err(|e| AdapterError::Hardware(format!("Failed to bind UDP socket: {e}")))?;

        let mut buf = vec![0u8; settings.buffer_size.max(2048)];
        let (len, _src) = socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| AdapterError::Hardware(format!("UDP recv error: {e}")))?;

        let parser = super::csi_receiver::CsiParser::new(Self::map_format(config));
        let packet = parser.parse(&buf[..len])?;
        Ok(packet.into())
    }

    /// Read CSI from a PCAP file.
    ///
    /// Reads the next record from the configured capture using the real PCAP
    /// reader (`PcapCsiReader`) and parses it with [`CsiParser`]. Offline replay
    /// is a genuine path: feeding a real `.pcap` yields real CsiReadings.
    async fn read_pcap_csi(config: &HardwareConfig) -> Result<CsiReadings, AdapterError> {
        let settings = match &config.device_settings {
            DeviceSettings::Pcap(s) => s,
            _ => return Err(AdapterError::Config("Invalid settings for PCAP".into())),
        };

        let recv_config = super::csi_receiver::ReceiverConfig::pcap(&settings.file_path);
        let mut reader = super::csi_receiver::PcapCsiReader::new(recv_config)?;
        reader.load()?;
        match reader.read_next().await? {
            Some(packet) => Ok(packet.into()),
            None => Err(AdapterError::Hardware(format!(
                "PCAP file {} contained no parseable CSI records",
                settings.file_path
            ))),
        }
    }

    /// Map the configured device type to the CSI parser format.
    fn map_format(config: &HardwareConfig) -> super::csi_receiver::CsiPacketFormat {
        use super::csi_receiver::CsiPacketFormat as F;
        match &config.device_type {
            DeviceType::Esp32 => F::Esp32Csi,
            DeviceType::Intel5300 => F::Intel5300Bfee,
            DeviceType::Atheros(_) => F::AtherosCsi,
            _ => F::Auto,
        }
    }

    /// Read one newline-delimited line of bytes from a serial port.
    ///
    /// With the `serial` feature enabled this performs real serial I/O via the
    /// `serialport` crate (blocking read on a blocking thread so the async
    /// runtime is not stalled). Without the feature, it returns a typed
    /// `UnsupportedAdapter` error — the parser is still available for supplied
    /// bytes, but no native serial backend is compiled in.
    #[cfg(feature = "serial")]
    async fn read_serial_line(settings: &SerialSettings) -> Result<Vec<u8>, AdapterError> {
        let port = settings.port.clone();
        let baud = settings.baud_rate;
        let timeout = std::time::Duration::from_millis(settings.read_timeout_ms.max(1));

        tokio::task::spawn_blocking(move || -> Result<Vec<u8>, AdapterError> {
            let mut sp = serialport::new(&port, baud)
                .timeout(timeout)
                .open()
                .map_err(|e| {
                    AdapterError::HardwareUnavailable(format!(
                        "Serial port {port} unavailable: {e}"
                    ))
                })?;

            // Accumulate bytes until a newline (ESP-CSI emits CSV lines).
            let mut line = Vec::with_capacity(512);
            let mut byte = [0u8; 1];
            loop {
                use std::io::Read as _;
                match sp.read(&mut byte) {
                    Ok(0) => break,
                    Ok(_) => {
                        if byte[0] == b'\n' {
                            line.push(byte[0]);
                            break;
                        }
                        line.push(byte[0]);
                        if line.len() > 65536 {
                            break; // guard against runaway line
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                        if line.is_empty() {
                            return Err(AdapterError::Timeout(format!(
                                "No serial data on {port} within {}ms",
                                timeout.as_millis()
                            )));
                        }
                        break;
                    }
                    Err(e) => {
                        return Err(AdapterError::Hardware(format!(
                            "Serial read error on {port}: {e}"
                        )))
                    }
                }
            }
            Ok(line)
        })
        .await
        .map_err(|e| AdapterError::Hardware(format!("Serial read task failed: {e}")))?
    }

    /// Serial-disabled fallback: no native serial backend compiled.
    #[cfg(not(feature = "serial"))]
    async fn read_serial_line(settings: &SerialSettings) -> Result<Vec<u8>, AdapterError> {
        Err(AdapterError::UnsupportedAdapter(format!(
            "ESP32 serial CSI ingest on {} requires the `serial` cargo feature (native serialport). \
             The ESP32 byte parser is still available via CsiParser::parse for supplied bytes.",
            settings.port
        )))
    }

    /// Generate simulated CSI data
    async fn generate_simulated_csi(config: &HardwareConfig) -> Result<CsiReadings, AdapterError> {
        use std::f64::consts::PI;

        // Simulate packet rate
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let num_subcarriers = config.channel_config.num_subcarriers;
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Generate simulated breathing pattern (~0.3 Hz)
        let breathing_component = (2.0 * PI * 0.3 * t).sin();

        // Generate simulated heartbeat pattern (~1.2 Hz)
        let heartbeat_component = 0.1 * (2.0 * PI * 1.2 * t).sin();

        let mut amplitudes = Vec::with_capacity(num_subcarriers);
        let mut phases = Vec::with_capacity(num_subcarriers);

        for i in 0..num_subcarriers {
            // Add frequency-dependent characteristics
            let freq_factor = (i as f64 / num_subcarriers as f64 * PI).sin();

            // Amplitude with breathing/heartbeat modulation
            let amp = 1.0 + 0.1 * breathing_component * freq_factor + heartbeat_component;

            // Phase with random walk + breathing modulation
            let phase = (i as f64 * 0.1 + 0.2 * breathing_component) % (2.0 * PI);

            amplitudes.push(amp);
            phases.push(phase);
        }

        Ok(CsiReadings {
            timestamp: Utc::now(),
            readings: vec![SensorCsiReading {
                sensor_id: "simulated".to_string(),
                amplitudes,
                phases,
                rssi: -45.0 + 2.0 * rand_simple(),
                noise_floor: -92.0,
                tx_mac: Some("00:11:22:33:44:55".to_string()),
                rx_mac: Some("AA:BB:CC:DD:EE:FF".to_string()),
                sequence_num: None,
            }],
            metadata: CsiMetadata {
                device_type: DeviceType::Simulated,
                channel: config.channel_config.channel,
                bandwidth: config.channel_config.bandwidth,
                num_subcarriers,
                rssi: Some(-45.0),
                noise_floor: Some(-92.0),
                fc_type: FrameControlType::Data,
                wideband: None,
            },
        })
    }

    /// Discover available sensors
    pub async fn discover_sensors(&mut self) -> Result<Vec<SensorInfo>, AdapterError> {
        if !self.initialized {
            return Err(AdapterError::Hardware("Hardware not initialized".into()));
        }

        // Discovery depends on device type
        match &self.config.device_type {
            DeviceType::Esp32 => self.discover_esp32_sensors().await,
            DeviceType::Intel5300 | DeviceType::Atheros(_) => self.discover_nic_sensors().await,
            DeviceType::UdpReceiver => Ok(vec![]),
            DeviceType::PcapFile => Ok(vec![]),
            DeviceType::FeitCsi => Ok(vec![]),
            DeviceType::Simulated => self.discover_simulated_sensors().await,
        }
    }

    async fn discover_esp32_sensors(&self) -> Result<Vec<SensorInfo>, AdapterError> {
        // ESP32 discovery would scan for beacons or query connected devices
        tracing::debug!("Discovering ESP32 sensors...");
        Ok(vec![])
    }

    async fn discover_nic_sensors(&self) -> Result<Vec<SensorInfo>, AdapterError> {
        // NIC-based systems would scan for nearby APs
        tracing::debug!("Discovering NIC sensors...");
        Ok(vec![])
    }

    async fn discover_simulated_sensors(&self) -> Result<Vec<SensorInfo>, AdapterError> {
        use crate::domain::SensorType;

        // Return fake sensors for testing
        Ok(vec![
            SensorInfo {
                id: "sim-tx-1".to_string(),
                position: SensorPosition {
                    id: "sim-tx-1".to_string(),
                    x: 0.0,
                    y: 0.0,
                    z: 2.0,
                    sensor_type: SensorType::Transmitter,
                    is_operational: true,
                    last_rssi: Some(-42.0),
                },
                status: SensorStatus::Connected,
                last_rssi: Some(-42.0),
                battery_level: Some(100),
                mac_address: Some("00:11:22:33:44:55".to_string()),
                firmware_version: Some("1.0.0".to_string()),
            },
            SensorInfo {
                id: "sim-rx-1".to_string(),
                position: SensorPosition {
                    id: "sim-rx-1".to_string(),
                    x: 5.0,
                    y: 0.0,
                    z: 2.0,
                    sensor_type: SensorType::Receiver,
                    is_operational: true,
                    last_rssi: Some(-48.0),
                },
                status: SensorStatus::Connected,
                last_rssi: Some(-48.0),
                battery_level: Some(85),
                mac_address: Some("AA:BB:CC:DD:EE:FF".to_string()),
                firmware_version: Some("1.0.0".to_string()),
            },
        ])
    }

    /// Add a sensor
    pub fn add_sensor(&mut self, sensor: SensorInfo) -> Result<(), AdapterError> {
        if self.sensors.iter().any(|s| s.id == sensor.id) {
            return Err(AdapterError::Hardware(format!(
                "Sensor {} already registered",
                sensor.id
            )));
        }

        self.sensors.push(sensor);
        Ok(())
    }

    /// Remove a sensor
    pub fn remove_sensor(&mut self, sensor_id: &str) -> Result<(), AdapterError> {
        let initial_len = self.sensors.len();
        self.sensors.retain(|s| s.id != sensor_id);

        if self.sensors.len() == initial_len {
            return Err(AdapterError::Hardware(format!(
                "Sensor {} not found",
                sensor_id
            )));
        }

        Ok(())
    }

    /// Get all sensors
    pub fn sensors(&self) -> &[SensorInfo] {
        &self.sensors
    }

    /// Get operational sensors
    pub fn operational_sensors(&self) -> Vec<&SensorInfo> {
        self.sensors
            .iter()
            .filter(|s| s.status == SensorStatus::Connected)
            .collect()
    }

    /// Get sensor positions for localization
    pub fn sensor_positions(&self) -> Vec<SensorPosition> {
        self.sensors
            .iter()
            .filter(|s| s.status == SensorStatus::Connected)
            .map(|s| s.position.clone())
            .collect()
    }

    /// Read CSI data from sensors (synchronous wrapper)
    pub fn read_csi(&self) -> Result<CsiReadings, AdapterError> {
        if !self.initialized {
            return Err(AdapterError::Hardware("Hardware not initialized".into()));
        }

        // Return empty readings - use async stream for real data
        Ok(CsiReadings {
            timestamp: Utc::now(),
            readings: Vec::new(),
            metadata: CsiMetadata {
                device_type: self.config.device_type.clone(),
                channel: self.config.channel_config.channel,
                bandwidth: self.config.channel_config.bandwidth,
                num_subcarriers: self.config.channel_config.num_subcarriers,
                rssi: None,
                noise_floor: None,
                fc_type: FrameControlType::Data,
                wideband: None,
            },
        })
    }

    /// Read RSSI from all sensors
    pub fn read_rssi(&self) -> Result<Vec<(String, f64)>, AdapterError> {
        if !self.initialized {
            return Err(AdapterError::Hardware("Hardware not initialized".into()));
        }

        Ok(self
            .sensors
            .iter()
            .filter_map(|s| s.last_rssi.map(|rssi| (s.id.clone(), rssi)))
            .collect())
    }

    /// Update sensor position
    pub fn update_sensor_position(
        &mut self,
        sensor_id: &str,
        position: SensorPosition,
    ) -> Result<(), AdapterError> {
        let sensor = self
            .sensors
            .iter_mut()
            .find(|s| s.id == sensor_id)
            .ok_or_else(|| AdapterError::Hardware(format!("Sensor {} not found", sensor_id)))?;

        sensor.position = position;
        Ok(())
    }

    /// Check hardware health
    pub fn health_check(&self) -> HardwareHealth {
        let total = self.sensors.len();
        let connected = self
            .sensors
            .iter()
            .filter(|s| s.status == SensorStatus::Connected)
            .count();
        let low_battery = self
            .sensors
            .iter()
            .filter(|s| matches!(s.battery_level, Some(b) if b < 20))
            .count();

        let status = if connected == 0 && total > 0 {
            HealthStatus::Critical
        } else if connected < total / 2 {
            HealthStatus::Degraded
        } else if low_battery > 0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        HardwareHealth {
            status,
            total_sensors: total,
            connected_sensors: connected,
            low_battery_sensors: low_battery,
        }
    }

    /// Get streaming statistics
    pub async fn streaming_stats(&self) -> StreamingStats {
        let state = self.state.read().await;
        StreamingStats {
            is_streaming: state.streaming,
            packets_received: state.packets_received,
            error_count: state.error_count,
            last_error: state.last_error.clone(),
        }
    }

    /// Configure channel settings
    pub async fn set_channel(
        &mut self,
        channel: u8,
        bandwidth: Bandwidth,
    ) -> Result<(), AdapterError> {
        if !self.initialized {
            return Err(AdapterError::Hardware("Hardware not initialized".into()));
        }

        // Validate channel
        let valid_2g = (1..=14).contains(&channel);
        let valid_5g = [
            36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
            144, 149, 153, 157, 161, 165,
        ]
        .contains(&channel);

        if !valid_2g && !valid_5g {
            return Err(AdapterError::Config(format!(
                "Invalid WiFi channel: {}",
                channel
            )));
        }

        self.config.channel_config.channel = channel;
        self.config.channel_config.bandwidth = bandwidth;
        self.config.channel_config.num_subcarriers = bandwidth.subcarrier_count();

        tracing::info!("Channel set to {} with {:?} bandwidth", channel, bandwidth);

        Ok(())
    }
}

impl Default for HardwareAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple pseudo-random number generator (for simulation)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0 - 0.5
}

/// CSI readings from sensors
#[derive(Debug, Clone)]
pub struct CsiReadings {
    /// Timestamp of readings
    pub timestamp: DateTime<Utc>,
    /// Individual sensor readings
    pub readings: Vec<SensorCsiReading>,
    /// Metadata about the capture
    pub metadata: CsiMetadata,
}

/// Metadata for CSI capture
#[derive(Debug, Clone)]
pub struct CsiMetadata {
    /// Device type that captured this data
    pub device_type: DeviceType,
    /// WiFi channel
    pub channel: u8,
    /// Channel bandwidth
    pub bandwidth: Bandwidth,
    /// Number of subcarriers
    pub num_subcarriers: usize,
    /// Overall RSSI
    pub rssi: Option<f64>,
    /// Noise floor
    pub noise_floor: Option<f64>,
    /// Frame control type
    pub fc_type: FrameControlType,
    /// Wideband spectral provenance (ADR-289): band, native subcarrier count
    /// and any native → pipeline mapping applied. `None` for legacy
    /// narrowband sources that predate wideband metadata.
    pub wideband: Option<WidebandMeta>,
}

/// WiFi frame control types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameControlType {
    /// Management frame (beacon, probe, etc.)
    Management,
    /// Control frame (ACK, RTS, CTS)
    Control,
    /// Data frame
    Data,
    /// Extension
    Extension,
}

/// CSI reading from a single sensor
#[derive(Debug, Clone)]
pub struct SensorCsiReading {
    /// Sensor ID
    pub sensor_id: String,
    /// CSI amplitudes (per subcarrier)
    pub amplitudes: Vec<f64>,
    /// CSI phases (per subcarrier)
    pub phases: Vec<f64>,
    /// RSSI value
    pub rssi: f64,
    /// Noise floor
    pub noise_floor: f64,
    /// Transmitter MAC address
    pub tx_mac: Option<String>,
    /// Receiver MAC address
    pub rx_mac: Option<String>,
    /// Sequence number
    pub sequence_num: Option<u16>,
}

/// CSI stream for async iteration
pub struct CsiStream {
    receiver: broadcast::Receiver<CsiReadings>,
}

impl CsiStream {
    /// Receive the next CSI reading
    pub async fn next(&mut self) -> Option<CsiReadings> {
        match self.receiver.recv().await {
            Ok(reading) => Some(reading),
            Err(broadcast::error::RecvError::Closed) => None,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("CSI stream lagged by {} messages", n);
                self.receiver.recv().await.ok()
            }
        }
    }
}

/// Streaming statistics
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Whether streaming is active
    pub is_streaming: bool,
    /// Total packets received
    pub packets_received: u64,
    /// Number of errors
    pub error_count: u64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Hardware health status
#[derive(Debug, Clone)]
pub struct HardwareHealth {
    /// Overall status
    pub status: HealthStatus,
    /// Total number of sensors
    pub total_sensors: usize,
    /// Number of connected sensors
    pub connected_sensors: usize,
    /// Number of sensors with low battery
    pub low_battery_sensors: usize,
}

/// Health status levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operational
    Healthy,
    /// Minor issues, still functional
    Warning,
    /// Significant issues, reduced capability
    Degraded,
    /// System not functional
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::SensorType;

    fn create_test_sensor(id: &str) -> SensorInfo {
        SensorInfo {
            id: id.to_string(),
            position: SensorPosition {
                id: id.to_string(),
                x: 0.0,
                y: 0.0,
                z: 1.5,
                sensor_type: SensorType::Transceiver,
                is_operational: true,
                last_rssi: Some(-45.0),
            },
            status: SensorStatus::Connected,
            last_rssi: Some(-45.0),
            battery_level: Some(80),
            mac_address: None,
            firmware_version: None,
        }
    }

    #[tokio::test]
    async fn test_initialize_simulated() {
        let mut adapter = HardwareAdapter::new();
        assert!(adapter.initialize().await.is_ok());
    }

    #[test]
    fn test_add_sensor() {
        let mut adapter = HardwareAdapter::new();

        let sensor = create_test_sensor("s1");
        assert!(adapter.add_sensor(sensor).is_ok());
        assert_eq!(adapter.sensors().len(), 1);
    }

    #[test]
    fn test_duplicate_sensor_error() {
        let mut adapter = HardwareAdapter::new();

        let sensor1 = create_test_sensor("s1");
        let sensor2 = create_test_sensor("s1");

        adapter.add_sensor(sensor1).unwrap();
        assert!(adapter.add_sensor(sensor2).is_err());
    }

    #[test]
    fn test_health_check() {
        let mut adapter = HardwareAdapter::new();

        // No sensors - should be healthy (nothing to fail)
        let health = adapter.health_check();
        assert!(matches!(health.status, HealthStatus::Healthy));

        // Add connected sensor
        adapter.add_sensor(create_test_sensor("s1")).unwrap();
        let health = adapter.health_check();
        assert!(matches!(health.status, HealthStatus::Healthy));
    }

    #[test]
    fn test_sensor_positions() {
        let mut adapter = HardwareAdapter::new();

        adapter.add_sensor(create_test_sensor("s1")).unwrap();
        adapter.add_sensor(create_test_sensor("s2")).unwrap();

        let positions = adapter.sensor_positions();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_esp32_config() {
        let config = HardwareConfig::esp32("/dev/ttyUSB0", 921600);
        assert!(matches!(config.device_type, DeviceType::Esp32));
        assert!(matches!(config.device_settings, DeviceSettings::Serial(_)));
    }

    #[test]
    fn test_intel_5300_config() {
        let config = HardwareConfig::intel_5300("wlan0");
        assert!(matches!(config.device_type, DeviceType::Intel5300));
        assert_eq!(config.channel_config.num_subcarriers, 30);
    }

    #[test]
    fn test_atheros_config() {
        let config = HardwareConfig::atheros("wlan0", AtherosDriver::Ath10k);
        assert!(matches!(
            config.device_type,
            DeviceType::Atheros(AtherosDriver::Ath10k)
        ));
        assert_eq!(config.channel_config.num_subcarriers, 114);
    }

    #[test]
    fn test_bandwidth_subcarriers() {
        assert_eq!(Bandwidth::HT20.subcarrier_count(), 56);
        assert_eq!(Bandwidth::HT40.subcarrier_count(), 114);
        assert_eq!(Bandwidth::VHT80.subcarrier_count(), 242);
        assert_eq!(Bandwidth::VHT160.subcarrier_count(), 484);
    }

    #[tokio::test]
    async fn test_csi_stream() {
        let mut adapter = HardwareAdapter::new();
        adapter.initialize().await.unwrap();

        let mut stream = adapter.start_csi_stream().await.unwrap();

        // Receive a few packets
        for _ in 0..3 {
            let reading = stream.next().await;
            assert!(reading.is_some());
        }

        adapter.stop_csi_stream().await.unwrap();
    }

    #[tokio::test]
    async fn test_discover_simulated_sensors() {
        let mut adapter = HardwareAdapter::new();
        adapter.initialize().await.unwrap();

        let sensors = adapter.discover_sensors().await.unwrap();
        assert_eq!(sensors.len(), 2);
    }

    /// End-to-end ESP32: real CSI_DATA CSV bytes parse to real CsiReadings via
    /// the same parser the adapter's `read_esp32_csi` uses (the byte-source for
    /// the live port is feature-gated; the parsing path is what was previously
    /// a "not yet implemented" stub).
    #[test]
    fn test_esp32_bytes_parse_end_to_end() {
        let parser = crate::integration::csi_receiver::CsiParser::new(
            crate::integration::csi_receiver::CsiPacketFormat::Esp32Csi,
        );
        let line = b"CSI_DATA,AA:BB:CC:DD:EE:FF,-45,6,128,1.0,0.5,2.0,0.6,3.0,0.7";
        let packet = parser.parse(line).expect("ESP32 parse");
        let readings: CsiReadings = packet.into();
        assert_eq!(readings.readings.len(), 1);
        assert_eq!(readings.readings[0].amplitudes.len(), 3);
        assert_eq!(readings.metadata.channel, 6);
        assert!(matches!(readings.metadata.device_type, DeviceType::Esp32));
    }

    /// End-to-end UDP: send a real JSON CSI datagram on the wire and confirm the
    /// adapter's UDP read path binds, receives, and parses it to CsiReadings.
    #[tokio::test]
    async fn test_udp_read_end_to_end() {
        // Bind the adapter receiver on an ephemeral port.
        let config = HardwareConfig::udp_receiver("127.0.0.1", 0);
        // Resolve the actual bound port by binding here, then handing the addr
        // to a one-shot parse using the same code path.
        let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let local = socket.local_addr().unwrap();

        // Sender pushes a real JSON CSI packet.
        let sender = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let payload = br#"{"rssi":-50,"channel":6,"amplitudes":[1.0,2.0,3.0],"phases":[0.1,0.2,0.3]}"#;
        sender.send_to(payload, local).await.unwrap();

        // Receive + parse exactly as read_udp_csi does.
        let mut buf = vec![0u8; 4096];
        let (len, _src) = socket.recv_from(&mut buf).await.unwrap();
        let parser =
            crate::integration::csi_receiver::CsiParser::new(HardwareAdapter::map_format(&config));
        let packet = parser.parse(&buf[..len]).expect("UDP JSON parse");
        let readings: CsiReadings = packet.into();
        assert_eq!(readings.readings[0].amplitudes.len(), 3);
        assert_eq!(readings.metadata.channel, 6);
    }

    /// End-to-end PCAP: write a real little-endian PCAP file with one JSON CSI
    /// record and confirm `read_pcap_csi` loads, reads, and parses it.
    #[tokio::test]
    async fn test_pcap_read_end_to_end() {
        use std::io::Write as _;

        let payload = br#"{"rssi":-48,"channel":6,"amplitudes":[1.0,2.0],"phases":[0.1,0.2]}"#;

        // Minimal PCAP: 24-byte global header (LE magic) + 16-byte record header.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0xA1B2C3D4u32.to_le_bytes()); // magic (LE)
        bytes.extend_from_slice(&2u16.to_le_bytes()); // version major
        bytes.extend_from_slice(&4u16.to_le_bytes()); // version minor
        bytes.extend_from_slice(&0i32.to_le_bytes()); // thiszone
        bytes.extend_from_slice(&0u32.to_le_bytes()); // sigfigs
        bytes.extend_from_slice(&65535u32.to_le_bytes()); // snaplen
        bytes.extend_from_slice(&1u32.to_le_bytes()); // network
                                                      // record header
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ts_sec
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ts_usec
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // incl_len
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // orig_len
        bytes.extend_from_slice(payload);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("mat_pcap_test_{}.pcap", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
        }

        let config = HardwareConfig {
            device_type: DeviceType::PcapFile,
            device_settings: DeviceSettings::Pcap(PcapSettings {
                file_path: path.to_string_lossy().to_string(),
                playback_speed: 1000.0, // skip realtime delay
                loop_playback: false,
            }),
            ..HardwareConfig::default()
        };

        let readings = HardwareAdapter::read_pcap_csi(&config).await.expect("pcap read");
        assert_eq!(readings.readings[0].amplitudes.len(), 2);
        assert_eq!(readings.metadata.channel, 6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_feitcsi_config() {
        let config = HardwareConfig::feitcsi_replay("/tmp/capture.dat");
        assert!(matches!(config.device_type, DeviceType::FeitCsi));
        match &config.device_settings {
            DeviceSettings::FeitCsi(s) => {
                assert_eq!(s.mode, FeitCsiMode::FileReplay);
                assert!(s.pipeline_subcarriers.is_none());
            }
            other => panic!("unexpected settings: {other:?}"),
        }

        let config = HardwareConfig::feitcsi_stream("/tmp/feitcsi.fifo");
        match &config.device_settings {
            DeviceSettings::FeitCsi(s) => assert_eq!(s.mode, FeitCsiMode::Stream),
            other => panic!("unexpected settings: {other:?}"),
        }
    }

    /// End-to-end FeitCSI file replay through the adapter read path:
    /// initialize, then read the capture record-by-record. Two adapters over
    /// the same synthetic capture see identical, order-preserving sequences
    /// (replay determinism), and native+wideband metadata is carried.
    #[tokio::test]
    async fn test_feitcsi_replay_end_to_end_deterministic() {
        use crate::integration::feitcsi::synth;

        // Synthetic 3-record HE 80 MHz capture, generated in code.
        let mut capture = Vec::new();
        for ts in [100u64, 200, 300] {
            capture.extend_from_slice(&synth::record_bytes(2, 1, 996, 2, 4, ts));
        }
        let path = std::env::temp_dir().join(format!(
            "feitcsi_adapter_test_{}.dat",
            std::process::id()
        ));
        std::fs::write(&path, &capture).unwrap();

        let run = || async {
            let mut config = HardwareConfig::feitcsi_replay(path.to_str().unwrap());
            if let DeviceSettings::FeitCsi(s) = &mut config.device_settings {
                s.band = WifiBand::Band6GHz;
                s.channel = 37;
            }
            let mut adapter = HardwareAdapter::with_config(config.clone());
            adapter.initialize().await.unwrap();

            let mut frames = Vec::new();
            for _ in 0..3 {
                let readings = HardwareAdapter::read_csi_packet(&config, &adapter.state)
                    .await
                    .expect("replay read");
                frames.push(readings);
            }
            // Capture exhausted: typed error, not fabricated data.
            let end = HardwareAdapter::read_csi_packet(&config, &adapter.state).await;
            assert!(matches!(end, Err(AdapterError::HardwareUnavailable(_))));
            frames
        };

        let first = run().await;
        let second = run().await;

        assert_eq!(first.len(), 3);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.timestamp, b.timestamp, "replay must be deterministic");
            assert_eq!(a.readings[0].amplitudes, b.readings[0].amplitudes);
        }

        // Native wideband metadata is first-class on every frame.
        let meta = &first[0].metadata;
        assert!(matches!(meta.device_type, DeviceType::FeitCsi));
        assert_eq!(meta.num_subcarriers, 996);
        assert_eq!(meta.bandwidth, Bandwidth::VHT80);
        let wb = meta.wideband.as_ref().expect("wideband metadata");
        assert_eq!(wb.band, WifiBand::Band6GHz);
        assert_eq!(wb.bandwidth_mhz, 80);
        assert_eq!(wb.native_subcarriers, 996);
        assert!(wb.mapping.is_none(), "native width: no mapping");
        // 2 rx * 1 tx = 2 antenna-pair readings per frame.
        assert_eq!(first[0].readings.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    /// FeitCSI replay with a configured pipeline width converts explicitly
    /// through the interpolation path and records the mapping in metadata.
    #[tokio::test]
    async fn test_feitcsi_replay_pipeline_conversion() {
        use crate::integration::feitcsi::synth;

        let capture = synth::record_bytes(1, 1, 1992, 3, 4, 42);
        let path = std::env::temp_dir().join(format!(
            "feitcsi_pipeline_test_{}.dat",
            std::process::id()
        ));
        std::fs::write(&path, &capture).unwrap();

        let mut config = HardwareConfig::feitcsi_replay(path.to_str().unwrap());
        if let DeviceSettings::FeitCsi(s) = &mut config.device_settings {
            s.pipeline_subcarriers = Some(56);
        }
        let mut adapter = HardwareAdapter::with_config(config.clone());
        adapter.initialize().await.unwrap();

        let readings = HardwareAdapter::read_csi_packet(&config, &adapter.state)
            .await
            .expect("replay read");
        assert_eq!(readings.metadata.num_subcarriers, 56);
        assert_eq!(readings.readings[0].amplitudes.len(), 56);
        let wb = readings.metadata.wideband.as_ref().unwrap();
        assert_eq!(wb.native_subcarriers, 1992, "true resolution preserved");
        let mapping = wb.mapping.as_ref().expect("mapping recorded");
        assert_eq!((mapping.native, mapping.pipeline), (1992, 56));

        let _ = std::fs::remove_file(&path);
    }

    /// Honest hardware gating: Intel 5300 / Atheros return typed
    /// HardwareUnavailable (no device/driver), never fabricated CSI.
    #[tokio::test]
    async fn test_intel_and_atheros_are_honestly_unavailable() {
        let cfg = HardwareConfig::intel_5300("wlan0");
        let r = HardwareAdapter::read_intel_5300_csi(&cfg).await;
        assert!(matches!(r, Err(AdapterError::HardwareUnavailable(_))));

        let cfg = HardwareConfig::atheros("wlan0", AtherosDriver::Ath10k);
        let r = HardwareAdapter::read_atheros_csi(&cfg, AtherosDriver::Ath10k).await;
        assert!(matches!(r, Err(AdapterError::HardwareUnavailable(_))));
    }
}
