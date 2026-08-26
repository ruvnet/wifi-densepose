//! Authenticated BLE identity evidence and external Bluetooth Channel Sounding
//! primitives for ADR-341.
//!
//! The module is deliberately narrow. ESP32-S3 contributes ordinary BLE
//! advertisements and RSSI only. A separate capable radio contributes phase
//! and timing primitives over the authenticated RVCS v1 frame. Neither input
//! is presented as a civil identity or clinical vital sign. The deterministic
//! replay is labelled `SYNTHETIC` and exercises expiry, spoof conflict,
//! crossing ambiguity, motion gating and abstention.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

use hmac::{Hmac, Mac};
use ruview_ontology::TrackId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;
type GatewayReplayKey = (u8, u8, u64);
type BleReplayKey = (u64, u8, u8, u32, [u8; 32]);
type BleReplayHighWater = (u32, i64);
type ChannelSoundingReplayKey = (u8, u32, u32);

/// Firmware BLE identity telemetry magic.
pub const BLE_IDENTITY_MAGIC: u32 = 0xC511_00B1;
/// Firmware BLE identity telemetry version.
pub const BLE_IDENTITY_VERSION: u8 = 1;
/// Fixed BLE identity telemetry length.
pub const BLE_IDENTITY_PACKET_SIZE: usize = 36;
/// Authenticated gateway envelope magic (`RVAE`, little endian).
pub const GATEWAY_ENVELOPE_MAGIC: u32 = 0x4541_5652;
/// Authenticated gateway envelope wire version.
pub const GATEWAY_ENVELOPE_VERSION: u8 = 1;
/// Fixed gateway envelope header length.
pub const GATEWAY_ENVELOPE_HEADER_SIZE: usize = 40;
/// Truncated HMAC-SHA256 tag length used by the gateway envelope.
pub const GATEWAY_ENVELOPE_TAG_SIZE: usize = 16;
/// External Channel Sounding UART magic (`RVCS`).
pub const CHANNEL_SOUNDING_MAGIC: u32 = 0x5343_5652;
/// External Channel Sounding wire version.
pub const CHANNEL_SOUNDING_VERSION: u8 = 1;
/// Fixed authenticated Channel Sounding frame length.
pub const CHANNEL_SOUNDING_FRAME_SIZE: usize = 72;
/// Minimum BLE confidence admitted to track association.
pub const BLE_ASSOCIATION_MIN_CONFIDENCE_PERMILLE: u16 = 600;
/// Evidence label applied to every bundled deterministic replay result.
pub const SYNTHETIC_LABEL: &str = "SYNTHETIC";

const BLE_AUTHENTICATED: u8 = 1 << 0;
const BLE_TIME_VERIFIED: u8 = 1 << 1;
const BLE_EXTENDED_ADVERT: u8 = 1 << 2;
const BLE_ALLOWED_FLAGS: u8 = BLE_AUTHENTICATED | BLE_TIME_VERIFIED | BLE_EXTENDED_ADVERT;
const CS_CALIBRATED: u8 = 1 << 0;
const CS_MOTION: u8 = 1 << 1;
const CS_ALLOWED_FLAGS: u8 = CS_CALIBRATED | CS_MOTION;
const CS_SIGNED_PREFIX: usize = 52;
const CS_TAG_OFFSET: usize = 52;
const CS_TAG_SIZE: usize = 16;
const CS_CRC_OFFSET: usize = 68;
const CS_MAC_DOMAIN: &[u8; 12] = b"RuView/CS/v1";
const GATEWAY_MAC_DOMAIN: &[u8; 12] = b"RuView/GW/v1";
const GATEWAY_FLAG_RX_MONOTONIC: u8 = 1 << 0;
const GATEWAY_ALLOWED_FLAGS: u8 = GATEWAY_FLAG_RX_MONOTONIC;
const GATEWAY_PAYLOAD_BLE_IDENTITY: u8 = 1;
const GATEWAY_PAYLOAD_CHANNEL_SOUNDING: u8 = 2;
const MAX_GATEWAY_TIMING_UNCERTAINTY_US: u32 = 1_000_000;
const MAX_BLE_TTL_MS: u16 = 5_000;
const MAX_BLE_TOKEN_SKEW_MIN: u32 = 10;
const MAX_RESPIRATION_DISAGREEMENT_BPM: f64 = 57.0;
const RADIO_REPLAY_SNAPSHOT_VERSION: u8 = 2;
const BLE_REPLAY_FINGERPRINT_DOMAIN: &[u8] = b"RuView/BLEReplay/v1";
const MAX_REPLAY_GATEWAY_SESSIONS: usize = 4_096;
const MAX_REPLAY_BLE_RECORDS: usize = 65_536;
const MAX_REPLAY_CS_SESSIONS: usize = 4_096;

/// Boundary failures for BLE and Channel Sounding ingress.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RadioIngressError {
    /// The frame has an unexpected byte length.
    #[error("unexpected frame length")]
    Length,
    /// Magic does not select the expected contract.
    #[error("wrong frame magic")]
    Magic,
    /// The version is unsupported.
    #[error("unsupported frame version")]
    Version,
    /// Reserved or unknown flag bits were set.
    #[error("invalid flags")]
    Flags,
    /// A reserved field was nonzero.
    #[error("reserved field is nonzero")]
    Reserved,
    /// CRC validation failed.
    #[error("CRC validation failed")]
    Crc,
    /// The expected provisioned key was not selected.
    #[error("unknown key id")]
    KeyId,
    /// The expected provisioned node was not selected.
    #[error("unknown node id")]
    NodeId,
    /// The expected provisioned companion source was not selected.
    #[error("unknown companion source id")]
    SourceId,
    /// The envelope carried the wrong payload kind.
    #[error("unexpected gateway payload type")]
    PayloadType,
    /// HMAC validation failed.
    #[error("authentication failed")]
    Authentication,
    /// A numeric field was outside the bounded contract.
    #[error("measurement field out of bounds")]
    Bounds,
    /// The evidence is stale at the receive boundary.
    #[error("evidence is stale")]
    Stale,
    /// A token epoch is outside the host freshness allowance.
    #[error("token epoch is outside the freshness allowance")]
    Epoch,
    /// A sequence repeated or moved backwards within an authenticated source session.
    #[error("replay or out-of-order sequence")]
    Replay,
    /// The bounded replay-source table is full.
    #[error("replay-source capacity exceeded")]
    Capacity,
    /// A caller supplied policy is unsafe or internally inconsistent.
    #[error("invalid ingress or fusion policy")]
    InvalidPolicy,
    /// A durable replay snapshot is malformed, unsupported, or over capacity.
    #[error("invalid replay snapshot")]
    InvalidSnapshot,
}

/// Payload selector authenticated by the gateway envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatewayPayloadType {
    /// Privacy-minimized BLE identity telemetry.
    BleIdentity,
    /// Authenticated external Channel Sounding primitive.
    ChannelSounding,
}

impl GatewayPayloadType {
    fn from_wire(value: u8) -> Result<Self, RadioIngressError> {
        match value {
            GATEWAY_PAYLOAD_BLE_IDENTITY => Ok(Self::BleIdentity),
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING => Ok(Self::ChannelSounding),
            _ => Err(RadioIngressError::PayloadType),
        }
    }

    fn payload_size(self) -> usize {
        match self {
            Self::BleIdentity => BLE_IDENTITY_PACKET_SIZE,
            Self::ChannelSounding => CHANNEL_SOUNDING_FRAME_SIZE,
        }
    }
}

/// Provisioned gateway identity and host-side envelope bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatewayIngressConfig {
    /// Enrolled ESP32 gateway node identifier.
    pub node_id: u8,
    /// Enrolled gateway envelope key selector.
    pub key_id: u8,
    /// Maximum accepted gateway receive-time uncertainty.
    pub max_timing_uncertainty_us: u32,
}

impl Default for GatewayIngressConfig {
    fn default() -> Self {
        Self {
            node_id: 0,
            key_id: 0,
            max_timing_uncertainty_us: 50_000,
        }
    }
}

/// Authenticated metadata retained from an RVAE v1 gateway envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEnvelopeMetadata {
    /// Authenticated payload selector.
    pub payload_type: GatewayPayloadType,
    /// Provisioned gateway key selector.
    pub key_id: u8,
    /// Provisioned gateway node identifier.
    pub node_id: u8,
    /// Sequence within this gateway boot session.
    pub sequence: u32,
    /// Random nonzero boot-session nonce.
    pub boot_nonce: u64,
    /// Receive timestamp on the gateway's boot-relative clock.
    pub received_at_boot_us: u64,
    /// Gateway-declared receive-time uncertainty.
    pub timing_uncertainty_us: u32,
}

/// An HMAC-verified RVAE v1 envelope whose payload still borrows the wire frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedGatewayEnvelope<'a> {
    /// Authenticated envelope metadata.
    pub metadata: GatewayEnvelopeMetadata,
    payload: &'a [u8],
}

impl<'a> AuthenticatedGatewayEnvelope<'a> {
    /// Validate an RVAE v1 envelope against an enrolled node and gateway key.
    pub fn parse(
        frame: &'a [u8],
        secret: &[u8; 32],
        config: GatewayIngressConfig,
    ) -> Result<Self, RadioIngressError> {
        if config.max_timing_uncertainty_us == 0
            || config.max_timing_uncertainty_us > MAX_GATEWAY_TIMING_UNCERTAINTY_US
        {
            return Err(RadioIngressError::InvalidPolicy);
        }
        if secret.iter().all(|byte| *byte == 0) {
            return Err(RadioIngressError::Authentication);
        }
        if frame.len() < GATEWAY_ENVELOPE_HEADER_SIZE + GATEWAY_ENVELOPE_TAG_SIZE {
            return Err(RadioIngressError::Length);
        }
        if le_u32(frame, 0) != GATEWAY_ENVELOPE_MAGIC {
            return Err(RadioIngressError::Magic);
        }
        if frame[4] != GATEWAY_ENVELOPE_VERSION {
            return Err(RadioIngressError::Version);
        }
        let payload_type = GatewayPayloadType::from_wire(frame[5])?;
        let flags = frame[6];
        if flags & !GATEWAY_ALLOWED_FLAGS != 0 || flags & GATEWAY_FLAG_RX_MONOTONIC == 0 {
            return Err(RadioIngressError::Flags);
        }
        if frame[7] != config.key_id {
            return Err(RadioIngressError::KeyId);
        }
        if frame[12] != config.node_id {
            return Err(RadioIngressError::NodeId);
        }
        if frame[13..16] != [0, 0, 0] {
            return Err(RadioIngressError::Reserved);
        }
        let total_len = usize::from(le_u16(frame, 8));
        let payload_len = usize::from(le_u16(frame, 10));
        if total_len != frame.len()
            || payload_len != payload_type.payload_size()
            || total_len != GATEWAY_ENVELOPE_HEADER_SIZE + payload_len + GATEWAY_ENVELOPE_TAG_SIZE
        {
            return Err(RadioIngressError::Length);
        }
        let sequence = le_u32(frame, 16);
        let boot_nonce = le_u64(frame, 20);
        let timing_uncertainty_us = le_u32(frame, 36);
        if sequence == 0
            || boot_nonce == 0
            || timing_uncertainty_us > config.max_timing_uncertainty_us
        {
            return Err(RadioIngressError::Bounds);
        }
        let signed_len = GATEWAY_ENVELOPE_HEADER_SIZE + payload_len;
        let mut mac =
            HmacSha256::new_from_slice(secret).map_err(|_| RadioIngressError::Authentication)?;
        mac.update(GATEWAY_MAC_DOMAIN);
        mac.update(&frame[..signed_len]);
        mac.verify_truncated_left(&frame[signed_len..total_len])
            .map_err(|_| RadioIngressError::Authentication)?;
        Ok(Self {
            metadata: GatewayEnvelopeMetadata {
                payload_type,
                key_id: frame[7],
                node_id: frame[12],
                sequence,
                boot_nonce,
                received_at_boot_us: le_u64(frame, 28),
                timing_uncertainty_us,
            },
            payload: &frame[GATEWAY_ENVELOPE_HEADER_SIZE..signed_len],
        })
    }

    /// Return the exact authenticated inner payload.
    #[must_use]
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

/// Host parsing policy for BLE identity telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleIngressConfig {
    /// Maximum accepted event lifetime.
    pub max_ttl_ms: u16,
    /// Maximum absolute token clock skew.
    pub max_token_skew_min: u32,
}

impl Default for BleIngressConfig {
    fn default() -> Self {
        Self {
            max_ttl_ms: 5_000,
            max_token_skew_min: 2,
        }
    }
}

/// A short lived, authenticated, pseudonymous BLE anchor observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BleIdentityEvidence {
    /// ESP32 scanner node identifier, cross-checked against the gateway envelope.
    pub node_id: u8,
    /// Provisioned BLE advertiser key selector.
    pub key_id: u8,
    /// BLE scanner sequence carried by the inner telemetry record.
    pub sequence: u32,
    /// Gateway envelope key selector.
    pub gateway_key_id: u8,
    /// Gateway envelope sequence used for replay suppression.
    pub gateway_sequence: u32,
    /// Authenticated gateway boot-session nonce.
    pub gateway_boot_nonce: u64,
    /// Gateway boot relative observation time retained for diagnostics.
    pub observed_at_boot_ms: u32,
    /// Gateway receive timestamp on its boot-relative clock.
    pub gateway_received_at_boot_us: u64,
    /// Gateway receive-time uncertainty.
    pub gateway_timing_uncertainty_us: u32,
    /// Host receive timestamp used for expiry.
    pub received_at_unix_ms: i64,
    /// Host computed expiry timestamp.
    pub expires_at_unix_ms: i64,
    /// Authenticated rotating pseudonym. This is not a person name or device MAC.
    pub pseudonym: [u8; 8],
    /// Advertiser token epoch in Unix minutes.
    pub token_epoch_min: u32,
    /// Received signal strength.
    pub rssi_dbm: i8,
    /// Advertised transmit power, or `127` when absent.
    pub tx_power_dbm: i8,
    /// Evidence quality in permille. This is not identity probability.
    pub confidence_permille: u16,
    /// Whether the scanner had a valid wall clock when it checked freshness.
    pub scanner_time_verified: bool,
}

impl BleIdentityEvidence {
    /// Verify an RVAE envelope, parse its BLE payload, and update replay state.
    pub fn parse_gateway_authenticated(
        frame: &[u8],
        received_at_unix_ms: i64,
        gateway_secret: &[u8; 32],
        gateway_config: GatewayIngressConfig,
        ble_config: BleIngressConfig,
        replay: &mut RadioReplayGuard,
    ) -> Result<Self, RadioIngressError> {
        let envelope = AuthenticatedGatewayEnvelope::parse(frame, gateway_secret, gateway_config)?;
        if envelope.metadata.payload_type != GatewayPayloadType::BleIdentity {
            return Err(RadioIngressError::PayloadType);
        }
        let evidence = Self::parse_payload(
            envelope.payload(),
            received_at_unix_ms,
            ble_config,
            envelope.metadata,
        )?;
        replay.admit_ble(&envelope.metadata, &evidence)?;
        Ok(evidence)
    }

    fn parse_payload(
        packet: &[u8],
        received_at_unix_ms: i64,
        config: BleIngressConfig,
        gateway: GatewayEnvelopeMetadata,
    ) -> Result<Self, RadioIngressError> {
        if config.max_ttl_ms == 0
            || config.max_ttl_ms > MAX_BLE_TTL_MS
            || config.max_token_skew_min > MAX_BLE_TOKEN_SKEW_MIN
        {
            return Err(RadioIngressError::InvalidPolicy);
        }
        if packet.len() != BLE_IDENTITY_PACKET_SIZE {
            return Err(RadioIngressError::Length);
        }
        if le_u32(packet, 0) != BLE_IDENTITY_MAGIC {
            return Err(RadioIngressError::Magic);
        }
        if packet[4] != BLE_IDENTITY_VERSION {
            return Err(RadioIngressError::Version);
        }
        let flags = packet[6];
        if flags & !BLE_ALLOWED_FLAGS != 0 || flags & BLE_AUTHENTICATED == 0 {
            return Err(RadioIngressError::Flags);
        }
        if le_u16(packet, 22) != 0 {
            return Err(RadioIngressError::Reserved);
        }
        if packet[5] != gateway.node_id {
            return Err(RadioIngressError::NodeId);
        }
        let ttl_ms = le_u16(packet, 16);
        let confidence = le_u16(packet, 18);
        let rssi = packet[20] as i8;
        let tx_power = packet[21] as i8;
        if le_u32(packet, 8) == 0
            || ttl_ms == 0
            || ttl_ms > config.max_ttl_ms
            || confidence > 1000
            || !(-127..=20).contains(&rssi)
            || !(tx_power == 127 || (-127..=20).contains(&tx_power))
        {
            return Err(RadioIngressError::Bounds);
        }
        let token_epoch_min = le_u32(packet, 32);
        let host_epoch_min = received_at_unix_ms.div_euclid(60_000);
        if host_epoch_min < 0
            || host_epoch_min.abs_diff(i64::from(token_epoch_min))
                > u64::from(config.max_token_skew_min)
        {
            return Err(RadioIngressError::Epoch);
        }
        let expires_at_unix_ms = received_at_unix_ms
            .checked_add(i64::from(ttl_ms))
            .ok_or(RadioIngressError::Bounds)?;
        let mut pseudonym = [0u8; 8];
        pseudonym.copy_from_slice(&packet[24..32]);
        Ok(Self {
            node_id: packet[5],
            key_id: packet[7],
            sequence: le_u32(packet, 8),
            gateway_key_id: gateway.key_id,
            gateway_sequence: gateway.sequence,
            gateway_boot_nonce: gateway.boot_nonce,
            observed_at_boot_ms: le_u32(packet, 12),
            gateway_received_at_boot_us: gateway.received_at_boot_us,
            gateway_timing_uncertainty_us: gateway.timing_uncertainty_us,
            received_at_unix_ms,
            expires_at_unix_ms,
            pseudonym,
            token_epoch_min,
            rssi_dbm: rssi,
            tx_power_dbm: tx_power,
            confidence_permille: confidence,
            scanner_time_verified: flags & BLE_TIME_VERIFIED != 0,
        })
    }

    /// Whether this evidence is still within its explicit TTL.
    #[must_use]
    pub fn is_live(&self, now_unix_ms: i64) -> bool {
        now_unix_ms >= self.received_at_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GatewayReplaySnapshotEntry {
    node_id: u8,
    key_id: u8,
    boot_nonce: u64,
    sequence: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BleReplaySnapshotEntry {
    gateway_boot_nonce: u64,
    node_id: u8,
    key_id: u8,
    token_epoch_min: u32,
    /// One-way replay fingerprint. The raw eight-byte over-air pseudonym must
    /// never be serialized, even into private replay state.
    pseudonym_fingerprint: [u8; 32],
    sequence: u32,
    retain_until_unix_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChannelSoundingReplaySnapshotEntry {
    key_id: u8,
    source_id: u32,
    source_session_id: u32,
    sequence: u32,
}

/// Versioned, deterministic durable replay state containing no key material.
///
/// Callers may serialize this value, but must pass deserialized values through
/// [`RadioReplaySnapshot::validate`] or [`RadioReplayGuard::restore`] before use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadioReplaySnapshot {
    version: u8,
    max_gateway_sessions: usize,
    max_ble_records: usize,
    max_channel_sounding_sessions: usize,
    gateway: Vec<GatewayReplaySnapshotEntry>,
    ble: Vec<BleReplaySnapshotEntry>,
    channel_sounding: Vec<ChannelSoundingReplaySnapshotEntry>,
}

impl RadioReplaySnapshot {
    /// Validate version, capacities, record bounds, and unique replay keys.
    pub fn validate(&self) -> Result<(), RadioIngressError> {
        if self.version != RADIO_REPLAY_SNAPSHOT_VERSION
            || self.max_gateway_sessions == 0
            || self.max_gateway_sessions > MAX_REPLAY_GATEWAY_SESSIONS
            || self.max_ble_records == 0
            || self.max_ble_records > MAX_REPLAY_BLE_RECORDS
            || self.max_channel_sounding_sessions == 0
            || self.max_channel_sounding_sessions > MAX_REPLAY_CS_SESSIONS
            || self.gateway.len() > self.max_gateway_sessions
            || self.ble.len() > self.max_ble_records
            || self.channel_sounding.len() > self.max_channel_sounding_sessions
        {
            return Err(RadioIngressError::InvalidSnapshot);
        }

        let mut gateway_keys = BTreeSet::new();
        let mut gateway_boots = BTreeSet::new();
        for entry in &self.gateway {
            if entry.boot_nonce == 0
                || entry.sequence == 0
                || !gateway_keys.insert((entry.node_id, entry.key_id, entry.boot_nonce))
            {
                return Err(RadioIngressError::InvalidSnapshot);
            }
            gateway_boots.insert((entry.node_id, entry.boot_nonce));
        }
        let mut ble_keys = BTreeSet::new();
        for entry in &self.ble {
            if entry.gateway_boot_nonce == 0
                || entry.sequence == 0
                || entry.retain_until_unix_ms < 0
                || !gateway_boots.contains(&(entry.node_id, entry.gateway_boot_nonce))
                || !ble_keys.insert((
                    entry.gateway_boot_nonce,
                    entry.node_id,
                    entry.key_id,
                    entry.token_epoch_min,
                    entry.pseudonym_fingerprint,
                ))
                || entry.pseudonym_fingerprint.iter().all(|byte| *byte == 0)
            {
                return Err(RadioIngressError::InvalidSnapshot);
            }
        }
        let mut channel_sounding_keys = BTreeSet::new();
        for entry in &self.channel_sounding {
            if entry.source_id == 0
                || entry.source_session_id == 0
                || !channel_sounding_keys.insert((
                    entry.key_id,
                    entry.source_id,
                    entry.source_session_id,
                ))
            {
                return Err(RadioIngressError::InvalidSnapshot);
            }
        }
        Ok(())
    }

    /// Return the snapshot format version.
    #[must_use]
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Return gateway, BLE-record, and companion-session capacities.
    #[must_use]
    pub fn capacities(&self) -> (usize, usize, usize) {
        (
            self.max_gateway_sessions,
            self.max_ble_records,
            self.max_channel_sounding_sessions,
        )
    }
}

/// Stateful, bounded replay protection for gateway boots, BLE records, and
/// companion Channel Sounding sessions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadioReplayGuard {
    gateway_sequences: BTreeMap<GatewayReplayKey, u32>,
    ble_sequences: BTreeMap<BleReplayKey, BleReplayHighWater>,
    channel_sounding_sequences: BTreeMap<ChannelSoundingReplayKey, u32>,
    max_gateway_sessions: usize,
    max_ble_records: usize,
    max_channel_sounding_sessions: usize,
}

impl RadioReplayGuard {
    /// Construct a fail-closed guard with explicit authenticated-session bounds.
    #[must_use]
    pub fn new(
        max_gateway_sessions: usize,
        max_ble_records: usize,
        max_channel_sounding_sessions: usize,
    ) -> Self {
        Self {
            gateway_sequences: BTreeMap::new(),
            ble_sequences: BTreeMap::new(),
            channel_sounding_sequences: BTreeMap::new(),
            max_gateway_sessions,
            max_ble_records,
            max_channel_sounding_sessions,
        }
    }

    /// Produce validated, deterministic replay state suitable for caller-owned storage.
    pub fn snapshot(&self) -> Result<RadioReplaySnapshot, RadioIngressError> {
        let snapshot = RadioReplaySnapshot {
            version: RADIO_REPLAY_SNAPSHOT_VERSION,
            max_gateway_sessions: self.max_gateway_sessions,
            max_ble_records: self.max_ble_records,
            max_channel_sounding_sessions: self.max_channel_sounding_sessions,
            gateway: self
                .gateway_sequences
                .iter()
                .map(
                    |(&(node_id, key_id, boot_nonce), &sequence)| GatewayReplaySnapshotEntry {
                        node_id,
                        key_id,
                        boot_nonce,
                        sequence,
                    },
                )
                .collect(),
            ble: self
                .ble_sequences
                .iter()
                .map(
                    |(
                        &(
                            gateway_boot_nonce,
                            node_id,
                            key_id,
                            token_epoch_min,
                            pseudonym_fingerprint,
                        ),
                        &(sequence, retain_until_unix_ms),
                    )| BleReplaySnapshotEntry {
                        gateway_boot_nonce,
                        node_id,
                        key_id,
                        token_epoch_min,
                        pseudonym_fingerprint,
                        sequence,
                        retain_until_unix_ms,
                    },
                )
                .collect(),
            channel_sounding: self
                .channel_sounding_sequences
                .iter()
                .map(|(&(key_id, source_id, source_session_id), &sequence)| {
                    ChannelSoundingReplaySnapshotEntry {
                        key_id,
                        source_id,
                        source_session_id,
                        sequence,
                    }
                })
                .collect(),
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Build a guard from validated durable state, dropping expired BLE records.
    pub fn from_snapshot(
        snapshot: RadioReplaySnapshot,
        now_unix_ms: i64,
    ) -> Result<Self, RadioIngressError> {
        if now_unix_ms < 0 {
            return Err(RadioIngressError::InvalidSnapshot);
        }
        snapshot.validate()?;
        let mut guard = Self::new(
            snapshot.max_gateway_sessions,
            snapshot.max_ble_records,
            snapshot.max_channel_sounding_sessions,
        );
        for entry in snapshot.gateway {
            guard.gateway_sequences.insert(
                (entry.node_id, entry.key_id, entry.boot_nonce),
                entry.sequence,
            );
        }
        for entry in snapshot.ble {
            if entry.retain_until_unix_ms > now_unix_ms {
                guard.ble_sequences.insert(
                    (
                        entry.gateway_boot_nonce,
                        entry.node_id,
                        entry.key_id,
                        entry.token_epoch_min,
                        entry.pseudonym_fingerprint,
                    ),
                    (entry.sequence, entry.retain_until_unix_ms),
                );
            }
        }
        for entry in snapshot.channel_sounding {
            guard.channel_sounding_sequences.insert(
                (entry.key_id, entry.source_id, entry.source_session_id),
                entry.sequence,
            );
        }
        Ok(guard)
    }

    /// Atomically replace this guard with validated durable replay state.
    pub fn restore(
        &mut self,
        snapshot: RadioReplaySnapshot,
        now_unix_ms: i64,
    ) -> Result<(), RadioIngressError> {
        let restored = Self::from_snapshot(snapshot, now_unix_ms)?;
        *self = restored;
        Ok(())
    }

    /// Atomically admit outer gateway and inner BLE sequences.
    pub fn admit_ble(
        &mut self,
        gateway: &GatewayEnvelopeMetadata,
        evidence: &BleIdentityEvidence,
    ) -> Result<(), RadioIngressError> {
        let gateway_source = (gateway.node_id, gateway.key_id, gateway.boot_nonce);
        let ble_source = (
            gateway.boot_nonce,
            evidence.node_id,
            evidence.key_id,
            evidence.token_epoch_min,
            ble_replay_fingerprint(
                gateway.boot_nonce,
                evidence.node_id,
                evidence.key_id,
                evidence.token_epoch_min,
                evidence.pseudonym,
            ),
        );
        if evidence.node_id != gateway.node_id {
            return Err(RadioIngressError::NodeId);
        }
        if evidence.gateway_key_id != gateway.key_id
            || evidence.gateway_sequence != gateway.sequence
            || evidence.gateway_boot_nonce != gateway.boot_nonce
        {
            return Err(RadioIngressError::Bounds);
        }
        check_strict_sequence(
            &self.gateway_sequences,
            gateway_source,
            gateway.sequence,
            self.max_gateway_sessions,
        )?;
        let live_ble_record = self
            .ble_sequences
            .get(&ble_source)
            .filter(|(_, retain_until_ms)| *retain_until_ms > evidence.received_at_unix_ms);
        match live_ble_record {
            Some((previous, _)) if evidence.sequence <= *previous => {
                return Err(RadioIngressError::Replay);
            }
            None if self
                .ble_sequences
                .values()
                .filter(|(_, retain_until_ms)| *retain_until_ms > evidence.received_at_unix_ms)
                .count()
                >= self.max_ble_records =>
            {
                return Err(RadioIngressError::Capacity);
            }
            _ => {}
        }
        let retain_until_ms = evidence
            .received_at_unix_ms
            .checked_add(i64::from(MAX_BLE_TOKEN_SKEW_MIN + 1) * 60_000)
            .ok_or(RadioIngressError::Bounds)?;
        self.ble_sequences
            .retain(|_, (_, retain_until_ms)| *retain_until_ms > evidence.received_at_unix_ms);
        record_sequence(
            &mut self.gateway_sequences,
            gateway_source,
            gateway.sequence,
        );
        self.ble_sequences
            .insert(ble_source, (evidence.sequence, retain_until_ms));
        Ok(())
    }

    /// Atomically admit the outer gateway sequence and inner companion sequence.
    pub fn admit_channel_sounding(
        &mut self,
        gateway: &GatewayEnvelopeMetadata,
        measurement: &ChannelSoundingMeasurement,
    ) -> Result<(), RadioIngressError> {
        let gateway_source = (gateway.node_id, gateway.key_id, gateway.boot_nonce);
        let companion_source = (
            measurement.key_id,
            measurement.source_id,
            measurement.source_session_id,
        );
        check_strict_sequence(
            &self.gateway_sequences,
            gateway_source,
            gateway.sequence,
            self.max_gateway_sessions,
        )?;
        check_wrapping_sequence(
            &self.channel_sounding_sequences,
            companion_source,
            measurement.sequence,
            self.max_channel_sounding_sessions,
        )?;
        record_sequence(
            &mut self.gateway_sequences,
            gateway_source,
            gateway.sequence,
        );
        record_sequence(
            &mut self.channel_sounding_sequences,
            companion_source,
            measurement.sequence,
        );
        Ok(())
    }

    /// Retire one gateway session only after a trusted replay-horizon decision.
    pub fn retire_gateway_session(&mut self, node_id: u8, key_id: u8, boot_nonce: u64) {
        self.gateway_sequences
            .remove(&(node_id, key_id, boot_nonce));
    }

    /// Retire one BLE replay record only after its authenticated freshness horizon.
    pub fn retire_ble_record(
        &mut self,
        gateway_boot_nonce: u64,
        node_id: u8,
        key_id: u8,
        token_epoch_min: u32,
        pseudonym: [u8; 8],
    ) {
        self.ble_sequences.remove(&(
            gateway_boot_nonce,
            node_id,
            key_id,
            token_epoch_min,
            ble_replay_fingerprint(
                gateway_boot_nonce,
                node_id,
                key_id,
                token_epoch_min,
                pseudonym,
            ),
        ));
    }

    /// Retire one companion session only after a trusted replay-horizon decision.
    pub fn retire_channel_sounding_session(
        &mut self,
        key_id: u8,
        source_id: u32,
        source_session_id: u32,
    ) {
        self.channel_sounding_sequences
            .remove(&(key_id, source_id, source_session_id));
    }
}

impl Default for RadioReplayGuard {
    fn default() -> Self {
        Self::new(64, 1024, 64)
    }
}

/// Bounded Channel Sounding host verification policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelSoundingIngressConfig {
    /// Expected provisioned companion key selector.
    pub key_id: u8,
    /// Expected provisioned companion source identifier.
    pub source_id: u32,
    /// Maximum admitted companion supplied sample age.
    pub max_sample_age_us: u32,
    /// Minimum quality in permille.
    pub min_quality_permille: u16,
}

impl Default for ChannelSoundingIngressConfig {
    fn default() -> Self {
        Self {
            key_id: 0,
            source_id: 0,
            max_sample_age_us: 2_000_000,
            min_quality_permille: 600,
        }
    }
}

/// An authenticated phase and timing primitive from a capable companion radio.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSoundingMeasurement {
    /// Gateway node that authenticated and forwarded the primitive.
    pub gateway_node_id: u8,
    /// Gateway envelope key selector.
    pub gateway_key_id: u8,
    /// Gateway envelope sequence.
    pub gateway_sequence: u32,
    /// Authenticated gateway boot-session nonce.
    pub gateway_boot_nonce: u64,
    /// Companion key selector.
    pub key_id: u8,
    /// Per source monotonic sequence.
    pub sequence: u32,
    /// Opaque provisioned companion source identifier.
    pub source_id: u32,
    /// Nonzero companion boot or controller session identifier.
    pub source_session_id: u32,
    /// Nonzero Channel Sounding procedure identifier.
    pub procedure_id: u32,
    /// Bluetooth RF channel index.
    pub channel_index: u16,
    /// Zero-based step index within the procedure.
    pub step_index: u16,
    /// Total bounded steps declared for the procedure.
    pub step_count: u16,
    /// Companion supplied age at UART transmission.
    pub sample_age_us: u32,
    /// Gateway receive time on its boot-relative clock.
    pub gateway_received_at_boot_us: u64,
    /// Approximate capture time on the same gateway boot-relative clock.
    pub capture_at_gateway_boot_us: u64,
    /// Independent host wall-clock receipt time; never compared with gateway boot time.
    pub host_received_at_unix_us: i64,
    /// Gateway-declared receive-time uncertainty.
    pub gateway_timing_uncertainty_us: u32,
    /// Timing uncertainty declared by the companion.
    pub timing_uncertainty_us: u16,
    /// Corrected carrier phase in milliradians.
    pub phase_millirad: i32,
    /// Round trip timing estimate in picoseconds.
    pub rtt_picoseconds: i32,
    /// Estimated carrier frequency offset.
    pub frequency_offset_hz: i32,
    /// Measurement quality in permille.
    pub quality_permille: u16,
    /// Companion declared calibration state.
    pub calibrated: bool,
    /// Companion declared gross motion state.
    pub motion: bool,
}

impl ChannelSoundingMeasurement {
    /// Verify the outer RVAE and inner RVCS HMACs, then update replay state.
    pub fn parse_gateway_authenticated(
        frame: &[u8],
        host_received_at_unix_us: i64,
        gateway_secret: &[u8; 32],
        gateway_config: GatewayIngressConfig,
        companion_secret: &[u8; 32],
        companion_config: ChannelSoundingIngressConfig,
        replay: &mut RadioReplayGuard,
    ) -> Result<Self, RadioIngressError> {
        let envelope = AuthenticatedGatewayEnvelope::parse(frame, gateway_secret, gateway_config)?;
        if envelope.metadata.payload_type != GatewayPayloadType::ChannelSounding {
            return Err(RadioIngressError::PayloadType);
        }
        let measurement = Self::parse_companion_payload(
            envelope.payload(),
            host_received_at_unix_us,
            envelope.metadata,
            companion_secret,
            companion_config,
        )?;
        replay.admit_channel_sounding(&envelope.metadata, &measurement)?;
        Ok(measurement)
    }

    fn parse_companion_payload(
        frame: &[u8],
        host_received_at_unix_us: i64,
        gateway: GatewayEnvelopeMetadata,
        secret: &[u8; 32],
        config: ChannelSoundingIngressConfig,
    ) -> Result<Self, RadioIngressError> {
        if config.source_id == 0
            || config.max_sample_age_us == 0
            || config.max_sample_age_us > 10_000_000
            || config.min_quality_permille == 0
            || config.min_quality_permille > 1000
        {
            return Err(RadioIngressError::InvalidPolicy);
        }
        if host_received_at_unix_us < 0 {
            return Err(RadioIngressError::Bounds);
        }
        if secret.iter().all(|byte| *byte == 0) {
            return Err(RadioIngressError::Authentication);
        }
        if frame.len() != CHANNEL_SOUNDING_FRAME_SIZE {
            return Err(RadioIngressError::Length);
        }
        if le_u32(frame, 0) != CHANNEL_SOUNDING_MAGIC {
            return Err(RadioIngressError::Magic);
        }
        if frame[4] != CHANNEL_SOUNDING_VERSION {
            return Err(RadioIngressError::Version);
        }
        let flags = frame[5];
        if flags & !CS_ALLOWED_FLAGS != 0 {
            return Err(RadioIngressError::Flags);
        }
        if frame[7] != 0 || le_u16(frame, 8) as usize != CHANNEL_SOUNDING_FRAME_SIZE {
            return Err(RadioIngressError::Reserved);
        }
        if crc32(&frame[..CS_CRC_OFFSET]) != le_u32(frame, CS_CRC_OFFSET) {
            return Err(RadioIngressError::Crc);
        }
        if frame[6] != config.key_id {
            return Err(RadioIngressError::KeyId);
        }
        let mut mac =
            HmacSha256::new_from_slice(secret).map_err(|_| RadioIngressError::Authentication)?;
        mac.update(CS_MAC_DOMAIN);
        mac.update(&frame[..CS_SIGNED_PREFIX]);
        mac.verify_truncated_left(&frame[CS_TAG_OFFSET..CS_CRC_OFFSET])
            .map_err(|_| RadioIngressError::Authentication)?;

        let sample_age_us = le_u32(frame, 16);
        let source_id = le_u32(frame, 20);
        let quality = le_u16(frame, 24);
        let timing_uncertainty_us = le_u16(frame, 26);
        let channel_index = le_u16(frame, 10);
        let phase = le_i32(frame, 28);
        let rtt = le_i32(frame, 32);
        let offset = le_i32(frame, 36);
        let source_session_id = le_u32(frame, 40);
        let procedure_id = le_u32(frame, 44);
        let step_index = le_u16(frame, 48);
        let step_count = le_u16(frame, 50);
        if source_id != config.source_id {
            return Err(RadioIngressError::SourceId);
        }
        if source_session_id == 0
            || procedure_id == 0
            || !(4..=79).contains(&step_count)
            || step_index >= step_count
            || channel_index > 78
            || sample_age_us > config.max_sample_age_us
            || quality < config.min_quality_permille
            || quality > 1000
            || timing_uncertainty_us > 10_000
            || !(-3142..=3142).contains(&phase)
            || !(0..=250_000).contains(&rtt)
            || !(-500_000..=500_000).contains(&offset)
        {
            return Err(if sample_age_us > config.max_sample_age_us {
                RadioIngressError::Stale
            } else {
                RadioIngressError::Bounds
            });
        }
        let capture_at_gateway_boot_us = gateway
            .received_at_boot_us
            .checked_sub(u64::from(sample_age_us))
            .ok_or(RadioIngressError::Stale)?;
        Ok(Self {
            gateway_node_id: gateway.node_id,
            gateway_key_id: gateway.key_id,
            gateway_sequence: gateway.sequence,
            gateway_boot_nonce: gateway.boot_nonce,
            key_id: frame[6],
            sequence: le_u32(frame, 12),
            source_id,
            source_session_id,
            procedure_id,
            channel_index,
            step_index,
            step_count,
            sample_age_us,
            gateway_received_at_boot_us: gateway.received_at_boot_us,
            capture_at_gateway_boot_us,
            host_received_at_unix_us,
            gateway_timing_uncertainty_us: gateway.timing_uncertainty_us,
            timing_uncertainty_us,
            phase_millirad: phase,
            rtt_picoseconds: rtt,
            frequency_offset_hz: offset,
            quality_permille: quality,
            calibrated: flags & CS_CALIBRATED != 0,
            motion: flags & CS_MOTION != 0,
        })
    }
}

/// Why the respiration path refused to emit a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RespirationAbstainReason {
    /// A caller supplied unsafe or nonsensical fusion policy.
    InvalidPolicy,
    /// Too few valid samples or too short a time window.
    InsufficientCoverage,
    /// Gross motion invalidates micromotion inference.
    Motion,
    /// Sources or RF channels were mixed without calibration.
    IncoherentSource,
    /// The phase series did not contain a dominant respiratory component.
    LowSpectralConfidence,
    /// CSI and Channel Sounding estimates conflict beyond the safety threshold.
    SourceConflict,
    /// Every available estimate expired.
    Expired,
}

/// Evidence origin. This code cannot mint a measured label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RadioEvidenceLabel {
    /// Generated by the deterministic simulator.
    Synthetic,
    /// Produced by an unvalidated hardware path and therefore only claimed.
    Claimed,
}

/// Respiration estimate or explicit abstention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum RespirationDecision {
    /// A bounded, nonclinical rate estimate.
    Estimated {
        /// Breaths per minute.
        bpm: f64,
        /// Variance used for uncertainty weighted fusion.
        variance: f64,
        /// Quality score in `[0, 1]`.
        confidence: f64,
        /// Honest evidence label.
        evidence: RadioEvidenceLabel,
    },
    /// No number was emitted.
    Abstain {
        /// Explicit reason.
        reason: RespirationAbstainReason,
    },
}

/// Deterministic phase window estimator configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RespirationEstimatorConfig {
    /// Minimum primitives retained from complete coherent procedures.
    pub min_samples: usize,
    /// Minimum covered duration.
    pub min_duration_us: u64,
    /// Minimum accepted spectral peak ratio.
    pub min_peak_ratio: f64,
    /// Evidence label assigned by the caller.
    pub evidence: RadioEvidenceLabel,
}

impl Default for RespirationEstimatorConfig {
    fn default() -> Self {
        Self {
            min_samples: 80,
            min_duration_us: 8_000_000,
            min_peak_ratio: 4.0,
            evidence: RadioEvidenceLabel::Claimed,
        }
    }
}

fn complete_coherent_procedure_samples(
    samples: &[ChannelSoundingMeasurement],
) -> Result<Vec<&ChannelSoundingMeasurement>, RespirationAbstainReason> {
    let Some(anchor) = samples.first() else {
        return Ok(Vec::new());
    };
    if samples.iter().any(|sample| {
        sample.source_id != anchor.source_id
            || sample.source_session_id != anchor.source_session_id
            || sample.key_id != anchor.key_id
            || sample.gateway_node_id != anchor.gateway_node_id
            || sample.gateway_key_id != anchor.gateway_key_id
            || sample.gateway_boot_nonce != anchor.gateway_boot_nonce
            || sample.gateway_sequence == 0
            || sample.gateway_boot_nonce == 0
            || sample.source_id == 0
            || sample.source_session_id == 0
            || sample.procedure_id == 0
            || sample.step_index >= sample.step_count
            || sample.channel_index > 78
            || sample.quality_permille > 1000
            || sample.timing_uncertainty_us > 10_000
            || !(-3142..=3142).contains(&sample.phase_millirad)
            || !(0..=250_000).contains(&sample.rtt_picoseconds)
            || !(-500_000..=500_000).contains(&sample.frequency_offset_hz)
            || sample.capture_at_gateway_boot_us > sample.gateway_received_at_boot_us
            || sample.host_received_at_unix_us < 0
            || !sample.calibrated
    }) {
        return Err(RespirationAbstainReason::IncoherentSource);
    }
    if samples.iter().any(|sample| sample.motion) {
        return Err(RespirationAbstainReason::Motion);
    }

    let mut procedures: BTreeMap<u32, Vec<&ChannelSoundingMeasurement>> = BTreeMap::new();
    for sample in samples {
        procedures
            .entry(sample.procedure_id)
            .or_default()
            .push(sample);
    }
    let mut expected_channel_plan: Option<Vec<u16>> = None;
    let mut admitted = Vec::new();
    for procedure in procedures.values() {
        let step_count = procedure[0].step_count;
        if !(4..=79).contains(&step_count)
            || procedure
                .iter()
                .any(|sample| sample.step_count != step_count)
        {
            return Err(RespirationAbstainReason::IncoherentSource);
        }
        let mut by_step = BTreeMap::new();
        let mut channels = BTreeSet::new();
        for sample in procedure {
            if by_step.insert(sample.step_index, *sample).is_some()
                || !channels.insert(sample.channel_index)
            {
                return Err(RespirationAbstainReason::IncoherentSource);
            }
        }
        if by_step.len() < usize::from(step_count) {
            continue;
        }
        if by_step.len() != usize::from(step_count) || channels.len() != usize::from(step_count) {
            return Err(RespirationAbstainReason::IncoherentSource);
        }
        let channel_plan: Vec<_> = by_step
            .values()
            .map(|sample| sample.channel_index)
            .collect();
        if let Some(expected) = &expected_channel_plan {
            if *expected != channel_plan {
                return Err(RespirationAbstainReason::IncoherentSource);
            }
        } else {
            expected_channel_plan = Some(channel_plan);
        }
        admitted.extend(by_step.into_values());
    }
    admitted.sort_by_key(|sample| sample.capture_at_gateway_boot_us);
    Ok(admitted)
}

/// Estimate respiration from complete, coherent authenticated procedures.
#[must_use]
pub fn estimate_channel_sounding_respiration(
    samples: &[ChannelSoundingMeasurement],
    config: RespirationEstimatorConfig,
) -> RespirationDecision {
    if config.min_samples == 0
        || config.min_duration_us == 0
        || !config.min_peak_ratio.is_finite()
        || config.min_peak_ratio <= 1.0
    {
        return abstain(RespirationAbstainReason::InvalidPolicy);
    }
    if samples.len() < config.min_samples {
        return abstain(RespirationAbstainReason::InsufficientCoverage);
    }
    let ordered = match complete_coherent_procedure_samples(samples) {
        Ok(ordered) => ordered,
        Err(reason) => return abstain(reason),
    };
    if ordered.len() < config.min_samples {
        return abstain(RespirationAbstainReason::InsufficientCoverage);
    }
    let duration = ordered
        .last()
        .map(|sample| sample.capture_at_gateway_boot_us)
        .unwrap_or(ordered[0].capture_at_gateway_boot_us)
        - ordered[0].capture_at_gateway_boot_us;
    if duration < config.min_duration_us {
        return abstain(RespirationAbstainReason::InsufficientCoverage);
    }

    let t0 = ordered[0].capture_at_gateway_boot_us;
    let mut channel_phase_sums: BTreeMap<u16, (f64, f64)> = BTreeMap::new();
    for sample in &ordered {
        let phase = f64::from(sample.phase_millirad) / 1000.0;
        let entry = channel_phase_sums
            .entry(sample.channel_index)
            .or_insert((0.0, 0.0));
        entry.0 += phase.sin();
        entry.1 += phase.cos();
    }
    let points: Vec<(f64, f64)> = ordered
        .iter()
        .map(|sample| {
            let phase = f64::from(sample.phase_millirad) / 1000.0;
            let (sin_sum, cos_sum) = channel_phase_sums[&sample.channel_index];
            let channel_mean = sin_sum.atan2(cos_sum);
            let centered_phase = (phase - channel_mean + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0;
            (
                (sample.capture_at_gateway_boot_us - t0) as f64 / 1_000_000.0,
                centered_phase,
            )
        })
        .collect();

    let mut best_frequency = 0.0;
    let mut best_power = 0.0;
    let mut power_sum = 0.0;
    let mut bins = 0usize;
    for bin in 0..=240 {
        let frequency = 0.1 + bin as f64 * 0.0025;
        let mut real = 0.0;
        let mut imag = 0.0;
        for &(time, phase) in &points {
            let angle = TAU * frequency * time;
            real += phase * angle.cos();
            imag -= phase * angle.sin();
        }
        let power = real * real + imag * imag;
        power_sum += power;
        bins += 1;
        if power > best_power {
            best_power = power;
            best_frequency = frequency;
        }
    }
    let background = ((power_sum - best_power) / (bins.saturating_sub(1).max(1) as f64)).max(1e-12);
    let peak_ratio = best_power / background;
    if !peak_ratio.is_finite() || peak_ratio < config.min_peak_ratio {
        return abstain(RespirationAbstainReason::LowSpectralConfidence);
    }
    let mean_quality = ordered
        .iter()
        .map(|sample| f64::from(sample.quality_permille) / 1000.0)
        .sum::<f64>()
        / ordered.len() as f64;
    let confidence = (mean_quality * (1.0 - 1.0 / peak_ratio)).clamp(0.0, 0.99);
    RespirationDecision::Estimated {
        bpm: best_frequency * 60.0,
        variance: 0.25 + (1.0 - confidence) * 4.0,
        confidence,
        evidence: config.evidence,
    }
}

/// One prevalidated rate estimate for cross modality fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RespirationEvidence {
    /// Breaths per minute.
    pub bpm: f64,
    /// Strictly positive variance.
    pub variance: f64,
    /// Expiry on the host time axis.
    pub expires_at_ms: i64,
    /// Whether gross motion was present.
    pub motion: bool,
    /// Evidence label.
    pub evidence: RadioEvidenceLabel,
}

/// Fuse live CSI and Channel Sounding rate evidence, or abstain on conflict.
#[must_use]
pub fn fuse_respiration(
    inputs: &[RespirationEvidence],
    now_ms: i64,
    max_disagreement_bpm: f64,
) -> RespirationDecision {
    if !max_disagreement_bpm.is_finite()
        || max_disagreement_bpm <= 0.0
        || max_disagreement_bpm > MAX_RESPIRATION_DISAGREEMENT_BPM
    {
        return abstain(RespirationAbstainReason::InvalidPolicy);
    }
    let usable: Vec<_> = inputs
        .iter()
        .filter(|input| {
            input.expires_at_ms > now_ms
                && input.bpm.is_finite()
                && (3.0..=60.0).contains(&input.bpm)
                && input.variance.is_finite()
                && input.variance > 0.0
                && (1.0 / input.variance).is_finite()
        })
        .collect();
    if usable.is_empty() {
        return abstain(RespirationAbstainReason::Expired);
    }
    if usable.iter().any(|input| input.motion) {
        return abstain(RespirationAbstainReason::Motion);
    }
    let min = usable
        .iter()
        .map(|input| input.bpm)
        .fold(f64::INFINITY, f64::min);
    let max = usable
        .iter()
        .map(|input| input.bpm)
        .fold(f64::NEG_INFINITY, f64::max);
    if max - min > max_disagreement_bpm {
        return abstain(RespirationAbstainReason::SourceConflict);
    }
    let precision_sum: f64 = usable.iter().map(|input| 1.0 / input.variance).sum();
    if !precision_sum.is_finite() || precision_sum <= 0.0 {
        return abstain(RespirationAbstainReason::InvalidPolicy);
    }
    let bpm = usable
        .iter()
        .map(|input| input.bpm / input.variance)
        .sum::<f64>()
        / precision_sum;
    let evidence = if usable
        .iter()
        .any(|input| input.evidence == RadioEvidenceLabel::Synthetic)
    {
        RadioEvidenceLabel::Synthetic
    } else {
        RadioEvidenceLabel::Claimed
    };
    RespirationDecision::Estimated {
        bpm,
        variance: 1.0 / precision_sum,
        confidence: (1.0 / (1.0 + 1.0 / precision_sum)).clamp(0.0, 0.99),
        evidence,
    }
}

/// Candidate likelihood that a short lived BLE pseudonym is colocated with a track.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackLikelihood {
    /// Candidate persistent pseudonymous track.
    pub track: TrackId,
    /// Bounded spatial likelihood in `[0, 1]` from an upstream localizer.
    pub likelihood: f64,
}

/// BLE anchor evidence plus bounded spatial candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct BleAssociationInput {
    /// Authenticated, expiring BLE evidence.
    pub evidence: BleIdentityEvidence,
    /// Candidate scores derived from geometry. RSSI alone is not position.
    pub candidates: Vec<TrackLikelihood>,
}

/// Why a BLE pseudonym did not bind to a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BleAssociationAbstainReason {
    /// A caller supplied an unsafe or nonsensical association policy.
    InvalidPolicy,
    /// The evidence TTL expired.
    Expired,
    /// No live observation met the minimum authenticated evidence quality.
    LowQuality,
    /// Candidate geometry was absent or weak.
    InsufficientGeometry,
    /// Top candidates were too close to call.
    Ambiguous,
    /// Concurrent observations strongly placed one pseudonym on incompatible tracks.
    SpoofConflict,
    /// Two pseudonyms competed for the same track.
    OneToOneConflict,
}

/// Short horizon pseudonymous anchor association.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum BleAssociationDecision {
    /// Pseudonym bound to a track for no longer than its TTL.
    Bound {
        /// BLE advertiser key selector that scopes the rotating pseudonym.
        key_id: u8,
        /// Token epoch that scopes the rotating pseudonym.
        token_epoch_min: u32,
        /// Rotating pseudonym.
        pseudonym: [u8; 8],
        /// Existing privacy preserving track.
        track: TrackId,
        /// Association confidence.
        confidence: f64,
        /// Binding expiry.
        expires_at_unix_ms: i64,
    },
    /// No binding was emitted.
    Abstain {
        /// BLE advertiser key selector that scopes the rotating pseudonym.
        key_id: u8,
        /// Token epoch that scopes the rotating pseudonym.
        token_epoch_min: u32,
        /// Rotating pseudonym, when present.
        pseudonym: [u8; 8],
        /// Explicit reason.
        reason: BleAssociationAbstainReason,
    },
}

/// Associate authenticated BLE pseudonyms to tracks with fail closed ambiguity.
#[must_use]
pub fn associate_ble_to_tracks(
    inputs: &[BleAssociationInput],
    now_unix_ms: i64,
    min_likelihood: f64,
    min_margin: f64,
) -> Vec<BleAssociationDecision> {
    type IdentityKey = (u8, u32, [u8; 8]);
    let mut grouped: BTreeMap<IdentityKey, Vec<&BleAssociationInput>> = BTreeMap::new();
    for input in inputs {
        grouped
            .entry((
                input.evidence.key_id,
                input.evidence.token_epoch_min,
                input.evidence.pseudonym,
            ))
            .or_default()
            .push(input);
    }
    let mut decisions = Vec::new();
    let mut proposed: Vec<(IdentityKey, TrackId, f64, i64)> = Vec::new();
    let policy_valid = min_likelihood.is_finite()
        && min_margin.is_finite()
        && min_likelihood > 0.0
        && min_likelihood <= 1.0
        && min_margin > 0.0
        && min_margin <= 1.0;
    for (identity, group) in grouped {
        let (key_id, token_epoch_min, pseudonym) = identity;
        if !policy_valid {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::InvalidPolicy,
            });
            continue;
        }
        let live: Vec<_> = group
            .into_iter()
            .filter(|input| input.evidence.is_live(now_unix_ms))
            .collect();
        if live.is_empty() {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::Expired,
            });
            continue;
        }
        let qualified: Vec<_> = live
            .into_iter()
            .filter(|input| {
                input.evidence.confidence_permille >= BLE_ASSOCIATION_MIN_CONFIDENCE_PERMILLE
            })
            .collect();
        if qualified.is_empty() {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::LowQuality,
            });
            continue;
        }

        let strong_tops: BTreeSet<TrackId> = qualified
            .iter()
            .filter_map(|input| best_candidate(&input.candidates))
            .filter(|(_, score, margin)| *score >= min_likelihood && *margin >= min_margin)
            .map(|(track, _, _)| track.clone())
            .collect();
        if strong_tops.len() > 1 {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::SpoofConflict,
            });
            continue;
        }

        let mut scores: BTreeMap<TrackId, (f64, usize)> = BTreeMap::new();
        for input in &qualified {
            for candidate in &input.candidates {
                if candidate.likelihood.is_finite() && (0.0..=1.0).contains(&candidate.likelihood) {
                    let entry = scores.entry(candidate.track.clone()).or_insert((0.0, 0));
                    entry.0 += candidate.likelihood;
                    entry.1 += 1;
                }
            }
        }
        let averaged: Vec<_> = scores
            .into_iter()
            .map(|(track, (sum, count))| TrackLikelihood {
                track,
                likelihood: sum / count as f64,
            })
            .collect();
        let Some((track, score, margin)) = best_candidate(&averaged) else {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::InsufficientGeometry,
            });
            continue;
        };
        if score < min_likelihood {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::InsufficientGeometry,
            });
        } else if margin < min_margin {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::Ambiguous,
            });
        } else {
            let expiry = qualified
                .iter()
                .map(|input| input.evidence.expires_at_unix_ms)
                .min()
                .unwrap_or(now_unix_ms);
            proposed.push((identity, track.clone(), score, expiry));
        }
    }

    let mut track_counts: BTreeMap<TrackId, usize> = BTreeMap::new();
    for (_, track, _, _) in &proposed {
        *track_counts.entry(track.clone()).or_default() += 1;
    }
    for ((key_id, token_epoch_min, pseudonym), track, confidence, expiry) in proposed {
        if track_counts.get(&track).copied().unwrap_or(0) > 1 {
            decisions.push(BleAssociationDecision::Abstain {
                key_id,
                token_epoch_min,
                pseudonym,
                reason: BleAssociationAbstainReason::OneToOneConflict,
            });
        } else {
            decisions.push(BleAssociationDecision::Bound {
                key_id,
                token_epoch_min,
                pseudonym,
                track,
                confidence,
                expires_at_unix_ms: expiry,
            });
        }
    }
    decisions.sort_by_key(|decision| match decision {
        BleAssociationDecision::Bound {
            key_id,
            token_epoch_min,
            pseudonym,
            ..
        }
        | BleAssociationDecision::Abstain {
            key_id,
            token_epoch_min,
            pseudonym,
            ..
        } => (*key_id, *token_epoch_min, *pseudonym),
    });
    decisions
}

/// Deterministic replay acceptance report. Every field is `SYNTHETIC`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticReplayReport {
    /// Required evidence label.
    pub evidence: String,
    /// Both pseudonyms bound before the crossing.
    pub bound_before_crossing: bool,
    /// Exact overlap caused an ambiguity abstention rather than an identity swap.
    pub abstained_at_crossing: bool,
    /// Both pseudonyms rebound to their original tracks after crossing.
    pub stable_after_crossing: bool,
    /// Incompatible duplicate evidence caused spoof abstention.
    pub spoof_abstained: bool,
    /// Expired evidence caused abstention.
    pub expiry_abstained: bool,
    /// Gross motion caused respiration abstention.
    pub motion_abstained: bool,
    /// Synthetic CSI plus Channel Sounding respiration output.
    pub fused_respiration_bpm: Option<f64>,
}

/// Run the built in two track deterministic simulation.
#[must_use]
pub fn run_synthetic_ble_cs_replay() -> SyntheticReplayReport {
    let epoch_min = 30_000_000u32;
    let now_ms = i64::from(epoch_min) * 60_000;
    let track_a = TrackId::new("synthetic-track-a").expect("static id");
    let track_b = TrackId::new("synthetic-track-b").expect("static id");
    let pseudo_a = [0xa1; 8];
    let pseudo_b = [0xb2; 8];

    let before = synthetic_association_frame(
        now_ms,
        epoch_min,
        (&track_a, pseudo_a),
        (&track_b, pseudo_b),
        4.5,
        5.5,
    );
    let overlap = synthetic_association_frame(
        now_ms + 500,
        epoch_min,
        (&track_a, pseudo_a),
        (&track_b, pseudo_b),
        5.0,
        5.0,
    );
    let after = synthetic_association_frame(
        now_ms + 1_000,
        epoch_min,
        (&track_a, pseudo_a),
        (&track_b, pseudo_b),
        5.5,
        4.5,
    );

    let before_decisions = associate_ble_to_tracks(&before, now_ms, 0.6, 0.15);
    let overlap_decisions = associate_ble_to_tracks(&overlap, now_ms + 500, 0.6, 0.15);
    let after_decisions = associate_ble_to_tracks(&after, now_ms + 1_000, 0.6, 0.15);

    let mut spoof = before.clone();
    spoof.push(BleAssociationInput {
        evidence: synthetic_ble_evidence(1, pseudo_a, epoch_min, now_ms),
        candidates: vec![
            TrackLikelihood {
                track: track_b.clone(),
                likelihood: 0.98,
            },
            TrackLikelihood {
                track: track_a.clone(),
                likelihood: 0.02,
            },
        ],
    });
    let spoof_decisions = associate_ble_to_tracks(&spoof, now_ms, 0.6, 0.15);
    let expiry_decisions = associate_ble_to_tracks(&before, now_ms + 10_000, 0.6, 0.15);

    let gateway_secret = [0x24u8; 32];
    let companion_secret = [0x42u8; 32];
    let gateway_config = GatewayIngressConfig {
        node_id: 1,
        key_id: 4,
        ..Default::default()
    };
    let companion_config = ChannelSoundingIngressConfig {
        key_id: 9,
        source_id: 101,
        ..Default::default()
    };
    let mut replay = RadioReplayGuard::new(1, 16, 1);
    let mut cs_samples = Vec::new();
    for index in 0..400u32 {
        let t = f64::from(index) / 20.0;
        let phase_millirad = (900.0 * (TAU * 0.25 * t).sin()).round() as i32;
        let payload = synthetic_cs_frame(index + 1, phase_millirad, false, &companion_secret);
        let gateway_received_at_boot_us = 5_000_000u64 + u64::from(index) * 50_000;
        let frame = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            index + 1,
            0x0102_0304_0506_0708,
            gateway_received_at_boot_us,
            8_000,
            &payload,
            &gateway_secret,
        );
        cs_samples.push(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &frame,
                now_ms * 1_000 + i64::from(index) * 50_000,
                &gateway_secret,
                gateway_config,
                &companion_secret,
                companion_config,
                &mut replay,
            )
            .expect("synthetic frame is valid"),
        );
    }
    let cs_estimate = estimate_channel_sounding_respiration(
        &cs_samples,
        RespirationEstimatorConfig {
            evidence: RadioEvidenceLabel::Synthetic,
            ..Default::default()
        },
    );
    let cs_rate = match cs_estimate {
        RespirationDecision::Estimated { bpm, variance, .. } => Some(RespirationEvidence {
            bpm,
            variance,
            expires_at_ms: now_ms + 2_000,
            motion: false,
            evidence: RadioEvidenceLabel::Synthetic,
        }),
        RespirationDecision::Abstain { .. } => None,
    };
    let mut rates = vec![RespirationEvidence {
        bpm: 14.8,
        variance: 1.0,
        expires_at_ms: now_ms + 2_000,
        motion: false,
        evidence: RadioEvidenceLabel::Synthetic,
    }];
    if let Some(rate) = cs_rate {
        rates.push(rate);
    }
    let fused = fuse_respiration(&rates, now_ms + 1_500, 3.0);
    let fused_respiration_bpm = match fused {
        RespirationDecision::Estimated { bpm, .. } => Some(bpm),
        RespirationDecision::Abstain { .. } => None,
    };

    let mut moving = cs_samples;
    moving[200].motion = true;
    let motion_abstained = matches!(
        estimate_channel_sounding_respiration(
            &moving,
            RespirationEstimatorConfig {
                evidence: RadioEvidenceLabel::Synthetic,
                ..Default::default()
            },
        ),
        RespirationDecision::Abstain {
            reason: RespirationAbstainReason::Motion
        }
    );

    SyntheticReplayReport {
        evidence: SYNTHETIC_LABEL.to_string(),
        bound_before_crossing: count_bound(&before_decisions) == 2,
        abstained_at_crossing: overlap_decisions.iter().all(|decision| matches!(
            decision,
            BleAssociationDecision::Abstain { reason: BleAssociationAbstainReason::Ambiguous, .. }
        )),
        stable_after_crossing: bound_to(&after_decisions, pseudo_a, &track_a)
            && bound_to(&after_decisions, pseudo_b, &track_b),
        spoof_abstained: spoof_decisions.iter().any(|decision| matches!(
            decision,
            BleAssociationDecision::Abstain { pseudonym, reason: BleAssociationAbstainReason::SpoofConflict, .. }
                if *pseudonym == pseudo_a
        )),
        expiry_abstained: expiry_decisions.iter().all(|decision| matches!(
            decision,
            BleAssociationDecision::Abstain { reason: BleAssociationAbstainReason::Expired, .. }
        )),
        motion_abstained,
        fused_respiration_bpm,
    }
}

fn synthetic_association_frame(
    now_ms: i64,
    epoch_min: u32,
    subject_a: (&TrackId, [u8; 8]),
    subject_b: (&TrackId, [u8; 8]),
    x_a: f64,
    x_b: f64,
) -> Vec<BleAssociationInput> {
    let (track_a, pseudo_a) = subject_a;
    let (track_b, pseudo_b) = subject_b;
    let sigma = 0.35f64;
    let score =
        |anchor_x: f64, track_x: f64| (-(anchor_x - track_x).powi(2) / (2.0 * sigma * sigma)).exp();
    vec![
        BleAssociationInput {
            evidence: synthetic_ble_evidence(1, pseudo_a, epoch_min, now_ms),
            candidates: vec![
                TrackLikelihood {
                    track: track_a.clone(),
                    likelihood: score(x_a, x_a),
                },
                TrackLikelihood {
                    track: track_b.clone(),
                    likelihood: score(x_a, x_b),
                },
            ],
        },
        BleAssociationInput {
            evidence: synthetic_ble_evidence(2, pseudo_b, epoch_min, now_ms),
            candidates: vec![
                TrackLikelihood {
                    track: track_a.clone(),
                    likelihood: score(x_b, x_a),
                },
                TrackLikelihood {
                    track: track_b.clone(),
                    likelihood: score(x_b, x_b),
                },
            ],
        },
    ]
}

fn synthetic_ble_evidence(
    sequence: u32,
    pseudonym: [u8; 8],
    epoch_min: u32,
    now_ms: i64,
) -> BleIdentityEvidence {
    BleIdentityEvidence {
        node_id: 1,
        key_id: 7,
        sequence,
        gateway_key_id: 4,
        gateway_sequence: sequence,
        gateway_boot_nonce: 0x0102_0304_0506_0708,
        observed_at_boot_ms: sequence * 100,
        gateway_received_at_boot_us: u64::from(sequence) * 100_000,
        gateway_timing_uncertainty_us: 1_000,
        received_at_unix_ms: now_ms,
        expires_at_unix_ms: now_ms + 3_000,
        pseudonym,
        token_epoch_min: epoch_min,
        rssi_dbm: -60,
        tx_power_dbm: 127,
        confidence_permille: 900,
        scanner_time_verified: true,
    }
}

fn synthetic_cs_frame(
    sequence: u32,
    phase_millirad: i32,
    motion: bool,
    secret: &[u8; 32],
) -> [u8; CHANNEL_SOUNDING_FRAME_SIZE] {
    let mut frame = [0u8; CHANNEL_SOUNDING_FRAME_SIZE];
    let ordinal = sequence.saturating_sub(1);
    let step_index = (ordinal % 20) as u16;
    put_u32(&mut frame, 0, CHANNEL_SOUNDING_MAGIC);
    frame[4] = CHANNEL_SOUNDING_VERSION;
    frame[5] = CS_CALIBRATED | if motion { CS_MOTION } else { 0 };
    frame[6] = 9;
    put_u16(&mut frame, 8, CHANNEL_SOUNDING_FRAME_SIZE as u16);
    put_u16(&mut frame, 10, step_index);
    put_u32(&mut frame, 12, sequence);
    put_u32(&mut frame, 16, 1_000);
    put_u32(&mut frame, 20, 101);
    put_u16(&mut frame, 24, 900);
    put_u16(&mut frame, 26, 100);
    put_i32(&mut frame, 28, phase_millirad);
    put_i32(&mut frame, 32, 50_000);
    put_i32(&mut frame, 36, 0);
    put_u32(&mut frame, 40, 77);
    put_u32(&mut frame, 44, 1 + ordinal / 20);
    put_u16(&mut frame, 48, step_index);
    put_u16(&mut frame, 50, 20);
    let mut mac = HmacSha256::new_from_slice(secret).expect("fixed key length");
    mac.update(CS_MAC_DOMAIN);
    mac.update(&frame[..CS_SIGNED_PREFIX]);
    let digest = mac.finalize().into_bytes();
    frame[CS_TAG_OFFSET..CS_CRC_OFFSET].copy_from_slice(&digest[..CS_TAG_SIZE]);
    let checksum = crc32(&frame[..CS_CRC_OFFSET]);
    put_u32(&mut frame, CS_CRC_OFFSET, checksum);
    frame
}

#[allow(clippy::too_many_arguments)]
fn synthetic_gateway_frame(
    payload_type: u8,
    sequence: u32,
    boot_nonce: u64,
    received_at_boot_us: u64,
    timing_uncertainty_us: u32,
    payload: &[u8],
    secret: &[u8; 32],
) -> Vec<u8> {
    let signed_len = GATEWAY_ENVELOPE_HEADER_SIZE + payload.len();
    let total_len = signed_len + GATEWAY_ENVELOPE_TAG_SIZE;
    let mut frame = vec![0u8; total_len];
    put_u32(&mut frame, 0, GATEWAY_ENVELOPE_MAGIC);
    frame[4] = GATEWAY_ENVELOPE_VERSION;
    frame[5] = payload_type;
    frame[6] = GATEWAY_FLAG_RX_MONOTONIC;
    frame[7] = 4;
    put_u16(&mut frame, 8, total_len as u16);
    put_u16(&mut frame, 10, payload.len() as u16);
    frame[12] = 1;
    put_u32(&mut frame, 16, sequence);
    put_u64(&mut frame, 20, boot_nonce);
    put_u64(&mut frame, 28, received_at_boot_us);
    put_u32(&mut frame, 36, timing_uncertainty_us);
    frame[GATEWAY_ENVELOPE_HEADER_SIZE..signed_len].copy_from_slice(payload);
    let mut mac = HmacSha256::new_from_slice(secret).expect("fixed key length");
    mac.update(GATEWAY_MAC_DOMAIN);
    mac.update(&frame[..signed_len]);
    let digest = mac.finalize().into_bytes();
    frame[signed_len..].copy_from_slice(&digest[..GATEWAY_ENVELOPE_TAG_SIZE]);
    frame
}

fn best_candidate(candidates: &[TrackLikelihood]) -> Option<(&TrackId, f64, f64)> {
    let mut valid: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            candidate.likelihood.is_finite() && (0.0..=1.0).contains(&candidate.likelihood)
        })
        .collect();
    valid.sort_by(|a, b| {
        b.likelihood
            .total_cmp(&a.likelihood)
            .then_with(|| a.track.as_str().cmp(b.track.as_str()))
    });
    let first = valid.first()?;
    let second = valid
        .get(1)
        .map(|candidate| candidate.likelihood)
        .unwrap_or(0.0);
    Some((&first.track, first.likelihood, first.likelihood - second))
}

fn count_bound(decisions: &[BleAssociationDecision]) -> usize {
    decisions
        .iter()
        .filter(|decision| matches!(decision, BleAssociationDecision::Bound { .. }))
        .count()
}

fn bound_to(decisions: &[BleAssociationDecision], pseudonym: [u8; 8], track: &TrackId) -> bool {
    decisions.iter().any(|decision| {
        matches!(
            decision,
            BleAssociationDecision::Bound { pseudonym: value, track: value_track, .. }
                if *value == pseudonym && value_track == track
        )
    })
}

fn abstain(reason: RespirationAbstainReason) -> RespirationDecision {
    RespirationDecision::Abstain { reason }
}

fn ble_replay_fingerprint(
    gateway_boot_nonce: u64,
    node_id: u8,
    key_id: u8,
    token_epoch_min: u32,
    pseudonym: [u8; 8],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(BLE_REPLAY_FINGERPRINT_DOMAIN);
    digest.update(gateway_boot_nonce.to_le_bytes());
    digest.update([node_id, key_id]);
    digest.update(token_epoch_min.to_le_bytes());
    digest.update(pseudonym);
    digest.finalize().into()
}

fn check_strict_sequence<K: Ord + Copy>(
    sequences: &BTreeMap<K, u32>,
    source: K,
    candidate: u32,
    capacity: usize,
) -> Result<(), RadioIngressError> {
    if let Some(previous) = sequences.get(&source) {
        return if candidate > *previous {
            Ok(())
        } else {
            Err(RadioIngressError::Replay)
        };
    }
    if sequences.len() >= capacity {
        return Err(RadioIngressError::Capacity);
    }
    Ok(())
}

fn check_wrapping_sequence<K: Ord + Copy>(
    sequences: &BTreeMap<K, u32>,
    source: K,
    candidate: u32,
    capacity: usize,
) -> Result<(), RadioIngressError> {
    if let Some(previous) = sequences.get(&source) {
        return if wrapping_sequence_is_newer(candidate, *previous) {
            Ok(())
        } else {
            Err(RadioIngressError::Replay)
        };
    }
    if sequences.len() >= capacity {
        return Err(RadioIngressError::Capacity);
    }
    Ok(())
}

fn record_sequence<K: Ord>(sequences: &mut BTreeMap<K, u32>, source: K, candidate: u32) {
    sequences.insert(source, candidate);
}

fn wrapping_sequence_is_newer(candidate: u32, previous: u32) -> bool {
    let delta = candidate.wrapping_sub(previous);
    delta != 0 && delta < 0x8000_0000
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn le_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATEWAY_SECRET: [u8; 32] = [0x24; 32];
    const COMPANION_SECRET: [u8; 32] = [0x42; 32];
    const BOOT_NONCE: u64 = 0x0102_0304_0506_0708;

    fn gateway_config() -> GatewayIngressConfig {
        GatewayIngressConfig {
            node_id: 1,
            key_id: 4,
            ..Default::default()
        }
    }

    fn companion_config() -> ChannelSoundingIngressConfig {
        ChannelSoundingIngressConfig {
            key_id: 9,
            source_id: 101,
            ..Default::default()
        }
    }

    fn ble_payload(epoch: u32, confidence_permille: u16) -> [u8; BLE_IDENTITY_PACKET_SIZE] {
        let mut packet = [0u8; BLE_IDENTITY_PACKET_SIZE];
        put_u32(&mut packet, 0, BLE_IDENTITY_MAGIC);
        packet[4] = BLE_IDENTITY_VERSION;
        packet[5] = 1;
        packet[6] = BLE_AUTHENTICATED | BLE_TIME_VERIFIED;
        packet[7] = 7;
        put_u32(&mut packet, 8, 1);
        put_u32(&mut packet, 12, 500);
        put_u16(&mut packet, 16, 3_000);
        put_u16(&mut packet, 18, confidence_permille);
        packet[20] = (-60i8) as u8;
        packet[21] = 127;
        packet[24..32].copy_from_slice(&[0xaa; 8]);
        put_u32(&mut packet, 32, epoch);
        packet
    }

    fn resign_cs(frame: &mut [u8; CHANNEL_SOUNDING_FRAME_SIZE]) {
        let mut mac = HmacSha256::new_from_slice(&COMPANION_SECRET).unwrap();
        mac.update(CS_MAC_DOMAIN);
        mac.update(&frame[..CS_SIGNED_PREFIX]);
        let digest = mac.finalize().into_bytes();
        frame[CS_TAG_OFFSET..CS_CRC_OFFSET].copy_from_slice(&digest[..CS_TAG_SIZE]);
        let checksum = crc32(&frame[..CS_CRC_OFFSET]);
        put_u32(frame, CS_CRC_OFFSET, checksum);
    }

    fn cs_gateway_frame(
        gateway_sequence: u32,
        companion_sequence: u32,
        gateway_received_at_boot_us: u64,
    ) -> Vec<u8> {
        let payload = synthetic_cs_frame(companion_sequence, 100, false, &COMPANION_SECRET);
        synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            gateway_sequence,
            BOOT_NONCE,
            gateway_received_at_boot_us,
            8_000,
            &payload,
            &GATEWAY_SECRET,
        )
    }

    fn gateway_metadata(sequence: u32) -> GatewayEnvelopeMetadata {
        GatewayEnvelopeMetadata {
            payload_type: GatewayPayloadType::ChannelSounding,
            key_id: 4,
            node_id: 1,
            sequence,
            boot_nonce: BOOT_NONCE,
            received_at_boot_us: 2_000_000 + u64::from(sequence),
            timing_uncertainty_us: 8_000,
        }
    }

    fn cs_measurement(
        procedure_id: u32,
        step_index: u16,
        step_count: u16,
        capture_at_gateway_boot_us: u64,
        phase_millirad: i32,
    ) -> ChannelSoundingMeasurement {
        let sequence = (procedure_id - 1)
            .saturating_mul(u32::from(step_count))
            .saturating_add(u32::from(step_index))
            .saturating_add(1);
        ChannelSoundingMeasurement {
            gateway_node_id: 1,
            gateway_key_id: 4,
            gateway_sequence: sequence,
            gateway_boot_nonce: BOOT_NONCE,
            key_id: 9,
            sequence,
            source_id: 101,
            source_session_id: 77,
            procedure_id,
            channel_index: step_index,
            step_index,
            step_count,
            sample_age_us: 1_000,
            gateway_received_at_boot_us: capture_at_gateway_boot_us + 1_000,
            capture_at_gateway_boot_us,
            host_received_at_unix_us: 1_800_000_000_000_000
                + i64::try_from(capture_at_gateway_boot_us).unwrap(),
            gateway_timing_uncertainty_us: 8_000,
            timing_uncertainty_us: 100,
            phase_millirad,
            rtt_picoseconds: 50_000,
            frequency_offset_hz: 0,
            quality_permille: 900,
            calibrated: true,
            motion: false,
        }
    }

    #[test]
    fn synthetic_replay_covers_crossing_spoof_expiry_motion_and_fusion() {
        let report = run_synthetic_ble_cs_replay();
        assert_eq!(report.evidence, SYNTHETIC_LABEL);
        assert!(report.bound_before_crossing);
        assert!(report.abstained_at_crossing);
        assert!(report.stable_after_crossing);
        assert!(report.spoof_abstained);
        assert!(report.expiry_abstained);
        assert!(report.motion_abstained);
        let bpm = report
            .fused_respiration_bpm
            .expect("synthetic fusion estimate");
        assert!((bpm - 15.0).abs() < 0.3, "unexpected synthetic bpm {bpm}");
    }

    #[test]
    fn gateway_envelope_validates_layout_hmac_enrollment_and_boot_replay() {
        let epoch = 30_000_000u32;
        let payload = ble_payload(epoch, 900);
        let frame = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            1,
            BOOT_NONCE,
            2_000_000,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(frame.len(), 40 + 36 + 16);
        let envelope =
            AuthenticatedGatewayEnvelope::parse(&frame, &GATEWAY_SECRET, gateway_config()).unwrap();
        assert_eq!(
            envelope.metadata.payload_type,
            GatewayPayloadType::BleIdentity
        );
        assert_eq!(envelope.metadata.sequence, 1);
        assert_eq!(envelope.metadata.boot_nonce, BOOT_NONCE);
        assert_eq!(envelope.metadata.received_at_boot_us, 2_000_000);
        assert_eq!(envelope.payload(), payload);

        let mut wrong_node = gateway_config();
        wrong_node.node_id = 2;
        assert_eq!(
            AuthenticatedGatewayEnvelope::parse(&frame, &GATEWAY_SECRET, wrong_node),
            Err(RadioIngressError::NodeId)
        );
        assert_eq!(
            AuthenticatedGatewayEnvelope::parse(&frame, &[1; 32], gateway_config()),
            Err(RadioIngressError::Authentication)
        );
        let mut tampered = frame.clone();
        tampered[GATEWAY_ENVELOPE_HEADER_SIZE + 20] ^= 1;
        assert_eq!(
            AuthenticatedGatewayEnvelope::parse(&tampered, &GATEWAY_SECRET, gateway_config()),
            Err(RadioIngressError::Authentication)
        );

        let mut replay = RadioReplayGuard::new(2, 16, 1);
        let gateway_source = (
            envelope.metadata.node_id,
            envelope.metadata.key_id,
            envelope.metadata.boot_nonce,
        );
        check_strict_sequence(
            &replay.gateway_sequences,
            gateway_source,
            envelope.metadata.sequence,
            replay.max_gateway_sessions,
        )
        .unwrap();
        record_sequence(
            &mut replay.gateway_sequences,
            gateway_source,
            envelope.metadata.sequence,
        );
        assert_eq!(
            check_strict_sequence(
                &replay.gateway_sequences,
                gateway_source,
                envelope.metadata.sequence,
                replay.max_gateway_sessions,
            ),
            Err(RadioIngressError::Replay)
        );
        let second = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            2,
            BOOT_NONCE,
            2_100_000,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        let second =
            AuthenticatedGatewayEnvelope::parse(&second, &GATEWAY_SECRET, gateway_config())
                .unwrap();
        check_strict_sequence(
            &replay.gateway_sequences,
            gateway_source,
            second.metadata.sequence,
            replay.max_gateway_sessions,
        )
        .unwrap();
        record_sequence(
            &mut replay.gateway_sequences,
            gateway_source,
            second.metadata.sequence,
        );
        let next_boot = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            1,
            BOOT_NONCE + 1,
            100,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        let next_boot =
            AuthenticatedGatewayEnvelope::parse(&next_boot, &GATEWAY_SECRET, gateway_config())
                .unwrap();
        let next_boot_source = (
            next_boot.metadata.node_id,
            next_boot.metadata.key_id,
            next_boot.metadata.boot_nonce,
        );
        check_strict_sequence(
            &replay.gateway_sequences,
            next_boot_source,
            next_boot.metadata.sequence,
            replay.max_gateway_sessions,
        )
        .unwrap();
        record_sequence(
            &mut replay.gateway_sequences,
            next_boot_source,
            next_boot.metadata.sequence,
        );
        let over_capacity = GatewayEnvelopeMetadata {
            boot_nonce: BOOT_NONCE + 2,
            ..next_boot.metadata
        };
        assert_eq!(
            check_strict_sequence(
                &replay.gateway_sequences,
                (
                    over_capacity.node_id,
                    over_capacity.key_id,
                    over_capacity.boot_nonce,
                ),
                over_capacity.sequence,
                replay.max_gateway_sessions,
            ),
            Err(RadioIngressError::Capacity)
        );

        let mut no_wrap = BTreeMap::new();
        no_wrap.insert(gateway_source, u32::MAX);
        assert_eq!(
            check_strict_sequence(&no_wrap, gateway_source, 1, 1),
            Err(RadioIngressError::Replay)
        );
    }

    #[test]
    fn channel_sounding_validates_72_byte_contract_clocks_and_authentication() {
        let frame = cs_gateway_frame(1, 1, 2_000_000);
        assert_eq!(frame.len(), 40 + 72 + 16);
        let mut replay = RadioReplayGuard::default();
        let measurement = ChannelSoundingMeasurement::parse_gateway_authenticated(
            &frame,
            1_800_000_000_000_000,
            &GATEWAY_SECRET,
            gateway_config(),
            &COMPANION_SECRET,
            companion_config(),
            &mut replay,
        )
        .unwrap();
        assert_eq!(measurement.source_session_id, 77);
        assert_eq!(measurement.procedure_id, 1);
        assert_eq!(measurement.step_index, 0);
        assert_eq!(measurement.step_count, 20);
        assert_eq!(measurement.gateway_received_at_boot_us, 2_000_000);
        assert_eq!(measurement.capture_at_gateway_boot_us, 1_999_000);
        assert_eq!(measurement.host_received_at_unix_us, 1_800_000_000_000_000);
        assert_eq!(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &frame,
                1_800_000_000_000_001,
                &GATEWAY_SECRET,
                gateway_config(),
                &COMPANION_SECRET,
                companion_config(),
                &mut replay,
            ),
            Err(RadioIngressError::Replay)
        );

        let mut payload = synthetic_cs_frame(2, 100, false, &COMPANION_SECRET);
        payload[28] ^= 1;
        let checksum = crc32(&payload[..CS_CRC_OFFSET]);
        put_u32(&mut payload, CS_CRC_OFFSET, checksum);
        let tampered = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            2,
            BOOT_NONCE,
            2_050_000,
            8_000,
            &payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &tampered,
                1_800_000_000_050_000,
                &GATEWAY_SECRET,
                gateway_config(),
                &COMPANION_SECRET,
                companion_config(),
                &mut RadioReplayGuard::default(),
            ),
            Err(RadioIngressError::Authentication)
        );

        let mut stale_payload = synthetic_cs_frame(2, 100, false, &COMPANION_SECRET);
        put_u32(&mut stale_payload, 16, 3_000_000);
        resign_cs(&mut stale_payload);
        let stale = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            2,
            BOOT_NONCE,
            4_000_000,
            8_000,
            &stale_payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &stale,
                1_800_000_003_000_000,
                &GATEWAY_SECRET,
                gateway_config(),
                &COMPANION_SECRET,
                companion_config(),
                &mut RadioReplayGuard::default(),
            ),
            Err(RadioIngressError::Stale)
        );
    }

    #[test]
    fn channel_sounding_rejects_bad_procedure_and_uses_bounded_session_replay() {
        let mut bad_step = synthetic_cs_frame(1, 100, false, &COMPANION_SECRET);
        put_u16(&mut bad_step, 48, 20);
        resign_cs(&mut bad_step);
        let bad_step = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            1,
            BOOT_NONCE,
            2_000_000,
            8_000,
            &bad_step,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &bad_step,
                1_800_000_000_000_000,
                &GATEWAY_SECRET,
                gateway_config(),
                &COMPANION_SECRET,
                companion_config(),
                &mut RadioReplayGuard::default(),
            ),
            Err(RadioIngressError::Bounds)
        );

        let mut replay = RadioReplayGuard::new(1, 16, 1);
        let first = cs_gateway_frame(1, 1, 2_000_000);
        ChannelSoundingMeasurement::parse_gateway_authenticated(
            &first,
            1_800_000_000_000_000,
            &GATEWAY_SECRET,
            gateway_config(),
            &COMPANION_SECRET,
            companion_config(),
            &mut replay,
        )
        .unwrap();
        let mut new_session_payload = synthetic_cs_frame(1, 100, false, &COMPANION_SECRET);
        put_u32(&mut new_session_payload, 40, 78);
        resign_cs(&mut new_session_payload);
        let new_session = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_CHANNEL_SOUNDING,
            2,
            BOOT_NONCE,
            2_050_000,
            8_000,
            &new_session_payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            ChannelSoundingMeasurement::parse_gateway_authenticated(
                &new_session,
                1_800_000_000_050_000,
                &GATEWAY_SECRET,
                gateway_config(),
                &COMPANION_SECRET,
                companion_config(),
                &mut replay,
            ),
            Err(RadioIngressError::Capacity)
        );
        let next = cs_gateway_frame(2, 2, 2_050_000);
        assert!(ChannelSoundingMeasurement::parse_gateway_authenticated(
            &next,
            1_800_000_000_050_000,
            &GATEWAY_SECRET,
            gateway_config(),
            &COMPANION_SECRET,
            companion_config(),
            &mut replay,
        )
        .is_ok());
    }

    #[test]
    fn gateway_sequence_is_strict_while_companion_sequence_may_wrap() {
        let mut replay = RadioReplayGuard::new(1, 1, 1);
        let mut measurement = cs_measurement(1, 0, 4, 1_000_000, 100);
        measurement.sequence = u32::MAX;
        replay
            .admit_channel_sounding(&gateway_metadata(1), &measurement)
            .unwrap();

        measurement.sequence = 0;
        replay
            .admit_channel_sounding(&gateway_metadata(2), &measurement)
            .unwrap();

        measurement.sequence = 100;
        assert_eq!(
            replay.admit_channel_sounding(&gateway_metadata(1), &measurement),
            Err(RadioIngressError::Replay)
        );
        measurement.sequence = 1;
        assert!(replay
            .admit_channel_sounding(&gateway_metadata(3), &measurement)
            .is_ok());
    }

    #[test]
    fn respiration_admits_only_complete_unique_coherent_procedures() {
        let mut complete = Vec::new();
        for procedure_id in 1..=2 {
            for step_index in 0..4 {
                complete.push(cs_measurement(
                    procedure_id,
                    step_index,
                    4,
                    u64::from(procedure_id) * 1_000_000 + u64::from(step_index) * 1_000,
                    i32::from(step_index) * 10,
                ));
            }
        }
        assert_eq!(
            complete_coherent_procedure_samples(&complete)
                .unwrap()
                .len(),
            8
        );

        let mut incomplete = complete.clone();
        incomplete.pop();
        assert_eq!(
            complete_coherent_procedure_samples(&incomplete)
                .unwrap()
                .len(),
            4
        );
        assert!(matches!(
            estimate_channel_sounding_respiration(
                &incomplete[4..],
                RespirationEstimatorConfig {
                    min_samples: 1,
                    min_duration_us: 1,
                    min_peak_ratio: 2.0,
                    evidence: RadioEvidenceLabel::Synthetic,
                }
            ),
            RespirationDecision::Abstain {
                reason: RespirationAbstainReason::InsufficientCoverage
            }
        ));

        let mut duplicate_step = complete.clone();
        duplicate_step[3].step_index = duplicate_step[2].step_index;
        assert_eq!(
            complete_coherent_procedure_samples(&duplicate_step),
            Err(RespirationAbstainReason::IncoherentSource)
        );
        let mut duplicate_channel = complete.clone();
        duplicate_channel[3].channel_index = duplicate_channel[2].channel_index;
        assert_eq!(
            complete_coherent_procedure_samples(&duplicate_channel),
            Err(RespirationAbstainReason::IncoherentSource)
        );
        let too_few_channels: Vec<_> = (0..3)
            .map(|step| cs_measurement(1, step, 3, 1_000_000 + u64::from(step), 100))
            .collect();
        assert_eq!(
            complete_coherent_procedure_samples(&too_few_channels),
            Err(RespirationAbstainReason::IncoherentSource)
        );
        let too_many_channels = vec![cs_measurement(1, 0, 81, 1_000_000, 100)];
        assert_eq!(
            complete_coherent_procedure_samples(&too_many_channels),
            Err(RespirationAbstainReason::IncoherentSource)
        );
        let mut changed_plan = complete;
        changed_plan[7].channel_index = 10;
        assert_eq!(
            complete_coherent_procedure_samples(&changed_plan),
            Err(RespirationAbstainReason::IncoherentSource)
        );
    }

    #[test]
    fn ble_packet_requires_authenticated_gateway_expires_and_rejects_bad_epoch() {
        let epoch = 30_000_000u32;
        let now = i64::from(epoch) * 60_000;
        let payload = ble_payload(epoch, 900);
        let frame = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            1,
            BOOT_NONCE,
            500_999,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        let mut replay = RadioReplayGuard::default();
        let evidence = BleIdentityEvidence::parse_gateway_authenticated(
            &frame,
            now,
            &GATEWAY_SECRET,
            gateway_config(),
            BleIngressConfig::default(),
            &mut replay,
        )
        .unwrap();
        assert_eq!(evidence.gateway_boot_nonce, BOOT_NONCE);
        assert_eq!(evidence.gateway_received_at_boot_us, 500_999);
        assert!(evidence.is_live(now + 2_999));
        assert!(!evidence.is_live(now + 3_000));
        assert_eq!(
            BleIdentityEvidence::parse_gateway_authenticated(
                &frame,
                now,
                &GATEWAY_SECRET,
                gateway_config(),
                BleIngressConfig::default(),
                &mut replay,
            ),
            Err(RadioIngressError::Replay)
        );

        let stale_payload = ble_payload(epoch - 10, 900);
        let stale = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            2,
            BOOT_NONCE,
            600_000,
            1_000,
            &stale_payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            BleIdentityEvidence::parse_gateway_authenticated(
                &stale,
                now,
                &GATEWAY_SECRET,
                gateway_config(),
                BleIngressConfig::default(),
                &mut replay,
            ),
            Err(RadioIngressError::Epoch)
        );
    }

    #[test]
    fn ble_replay_is_atomic_strict_and_scoped_to_gateway_boot() {
        let epoch = 30_000_000u32;
        let now = i64::from(epoch) * 60_000;
        let make_frame = |gateway_sequence, boot_nonce, ble_sequence| {
            let mut payload = ble_payload(epoch, 900);
            put_u32(&mut payload, 8, ble_sequence);
            synthetic_gateway_frame(
                GATEWAY_PAYLOAD_BLE_IDENTITY,
                gateway_sequence,
                boot_nonce,
                500_000 + u64::from(gateway_sequence),
                1_000,
                &payload,
                &GATEWAY_SECRET,
            )
        };
        let mut replay = RadioReplayGuard::new(2, 4, 1);
        BleIdentityEvidence::parse_gateway_authenticated(
            &make_frame(1, BOOT_NONCE, 10),
            now,
            &GATEWAY_SECRET,
            gateway_config(),
            BleIngressConfig::default(),
            &mut replay,
        )
        .unwrap();

        assert_eq!(
            BleIdentityEvidence::parse_gateway_authenticated(
                &make_frame(2, BOOT_NONCE, 9),
                now + 1,
                &GATEWAY_SECRET,
                gateway_config(),
                BleIngressConfig::default(),
                &mut replay,
            ),
            Err(RadioIngressError::Replay)
        );
        assert!(BleIdentityEvidence::parse_gateway_authenticated(
            &make_frame(2, BOOT_NONCE, 11),
            now + 1,
            &GATEWAY_SECRET,
            gateway_config(),
            BleIngressConfig::default(),
            &mut replay,
        )
        .is_ok());

        assert!(BleIdentityEvidence::parse_gateway_authenticated(
            &make_frame(1, BOOT_NONCE + 1, 1),
            now + 2,
            &GATEWAY_SECRET,
            gateway_config(),
            BleIngressConfig::default(),
            &mut replay,
        )
        .is_ok());
        assert_eq!(replay.ble_sequences.len(), 2);
    }

    #[test]
    fn replay_snapshot_round_trips_all_high_water_state_and_rejects_malformed() {
        let epoch = 30_000_000u32;
        let now = i64::from(epoch) * 60_000;
        let payload = ble_payload(epoch, 900);
        let ble_frame = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            1,
            BOOT_NONCE,
            500_000,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        let mut replay = RadioReplayGuard::new(2, 4, 2);
        BleIdentityEvidence::parse_gateway_authenticated(
            &ble_frame,
            now,
            &GATEWAY_SECRET,
            gateway_config(),
            BleIngressConfig::default(),
            &mut replay,
        )
        .unwrap();
        ChannelSoundingMeasurement::parse_gateway_authenticated(
            &cs_gateway_frame(2, 1, 550_000),
            now * 1_000 + 50_000,
            &GATEWAY_SECRET,
            gateway_config(),
            &COMPANION_SECRET,
            companion_config(),
            &mut replay,
        )
        .unwrap();

        let snapshot = replay.snapshot().unwrap();
        assert_eq!(snapshot.version(), RADIO_REPLAY_SNAPSHOT_VERSION);
        assert_eq!(snapshot.capacities(), (2, 4, 2));
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(!json.contains("secret"));
        let decoded: RadioReplaySnapshot = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();
        let restored = RadioReplayGuard::from_snapshot(decoded.clone(), now).unwrap();
        assert_eq!(restored, replay);

        let after_ble_horizon = RadioReplayGuard::from_snapshot(
            decoded.clone(),
            now + i64::from(MAX_BLE_TOKEN_SKEW_MIN + 1) * 60_000,
        )
        .unwrap();
        assert!(after_ble_horizon.ble_sequences.is_empty());
        assert_eq!(after_ble_horizon.gateway_sequences.len(), 1);
        assert_eq!(after_ble_horizon.channel_sounding_sequences.len(), 1);

        let mut bad_version = decoded.clone();
        bad_version.version += 1;
        assert_eq!(
            bad_version.validate(),
            Err(RadioIngressError::InvalidSnapshot)
        );
        let mut duplicate = decoded.clone();
        duplicate.gateway.push(duplicate.gateway[0].clone());
        assert_eq!(
            duplicate.validate(),
            Err(RadioIngressError::InvalidSnapshot)
        );
        let mut orphan_ble = decoded.clone();
        orphan_ble.ble[0].gateway_boot_nonce += 1;
        assert_eq!(
            orphan_ble.validate(),
            Err(RadioIngressError::InvalidSnapshot)
        );
        let mut over_capacity = decoded.clone();
        over_capacity.max_gateway_sessions = 1;
        let mut extra_gateway = over_capacity.gateway[0].clone();
        extra_gateway.boot_nonce += 1;
        extra_gateway.sequence = 1;
        over_capacity.gateway.push(extra_gateway);
        assert_eq!(
            over_capacity.validate(),
            Err(RadioIngressError::InvalidSnapshot)
        );
        let mut zero_capacity = decoded;
        zero_capacity.max_ble_records = 0;
        let before_failed_restore = replay.clone();
        assert_eq!(
            replay.restore(zero_capacity, now),
            Err(RadioIngressError::InvalidSnapshot)
        );
        assert_eq!(replay, before_failed_restore);
    }

    #[test]
    fn association_scopes_identity_and_rejects_low_quality_or_invalid_policy() {
        let now = 1_800_000_000_000i64;
        let track_a = TrackId::new("track-a").unwrap();
        let track_b = TrackId::new("track-b").unwrap();
        let pseudonym = [0xaa; 8];
        let candidate = |track: &TrackId| {
            vec![TrackLikelihood {
                track: track.clone(),
                likelihood: 0.95,
            }]
        };
        let mut first = synthetic_ble_evidence(1, pseudonym, 30_000_000, now);
        let mut second = synthetic_ble_evidence(2, pseudonym, 30_000_000, now);
        second.key_id = 8;
        let scoped = associate_ble_to_tracks(
            &[
                BleAssociationInput {
                    evidence: first.clone(),
                    candidates: candidate(&track_a),
                },
                BleAssociationInput {
                    evidence: second,
                    candidates: candidate(&track_b),
                },
            ],
            now,
            0.6,
            0.15,
        );
        assert_eq!(count_bound(&scoped), 2);

        first.confidence_permille = BLE_ASSOCIATION_MIN_CONFIDENCE_PERMILLE - 1;
        let low_quality = associate_ble_to_tracks(
            &[BleAssociationInput {
                evidence: first.clone(),
                candidates: candidate(&track_a),
            }],
            now,
            0.6,
            0.15,
        );
        assert!(matches!(
            low_quality.as_slice(),
            [BleAssociationDecision::Abstain {
                reason: BleAssociationAbstainReason::LowQuality,
                ..
            }]
        ));
        let invalid = associate_ble_to_tracks(
            &[BleAssociationInput {
                evidence: first,
                candidates: candidate(&track_a),
            }],
            now,
            f64::NAN,
            0.15,
        );
        assert!(matches!(
            invalid.as_slice(),
            [BleAssociationDecision::Abstain {
                reason: BleAssociationAbstainReason::InvalidPolicy,
                ..
            }]
        ));
    }

    #[test]
    fn public_policies_fail_closed_on_zero_nan_and_expired_motion() {
        assert!(matches!(
            estimate_channel_sounding_respiration(
                &[],
                RespirationEstimatorConfig {
                    min_samples: 0,
                    ..Default::default()
                }
            ),
            RespirationDecision::Abstain {
                reason: RespirationAbstainReason::InvalidPolicy
            }
        ));
        assert!(matches!(
            estimate_channel_sounding_respiration(
                &[],
                RespirationEstimatorConfig {
                    min_peak_ratio: f64::NAN,
                    ..Default::default()
                }
            ),
            RespirationDecision::Abstain {
                reason: RespirationAbstainReason::InvalidPolicy
            }
        ));
        let expired_motion = RespirationEvidence {
            bpm: 15.0,
            variance: 1.0,
            expires_at_ms: 1_000,
            motion: true,
            evidence: RadioEvidenceLabel::Synthetic,
        };
        assert!(matches!(
            fuse_respiration(&[expired_motion], 1_000, 3.0),
            RespirationDecision::Abstain {
                reason: RespirationAbstainReason::Expired
            }
        ));
        assert!(matches!(
            fuse_respiration(&[], 1_000, f64::NAN),
            RespirationDecision::Abstain {
                reason: RespirationAbstainReason::InvalidPolicy
            }
        ));

        let epoch = 30_000_000u32;
        let payload = ble_payload(epoch, 900);
        let frame = synthetic_gateway_frame(
            GATEWAY_PAYLOAD_BLE_IDENTITY,
            1,
            BOOT_NONCE,
            500_000,
            1_000,
            &payload,
            &GATEWAY_SECRET,
        );
        assert_eq!(
            BleIdentityEvidence::parse_gateway_authenticated(
                &frame,
                i64::from(epoch) * 60_000,
                &GATEWAY_SECRET,
                gateway_config(),
                BleIngressConfig {
                    max_ttl_ms: 0,
                    ..Default::default()
                },
                &mut RadioReplayGuard::default(),
            ),
            Err(RadioIngressError::InvalidPolicy)
        );

        let mut invalid_gateway = gateway_config();
        invalid_gateway.max_timing_uncertainty_us = 0;
        assert_eq!(
            AuthenticatedGatewayEnvelope::parse(&frame, &GATEWAY_SECRET, invalid_gateway),
            Err(RadioIngressError::InvalidPolicy)
        );
    }
}
