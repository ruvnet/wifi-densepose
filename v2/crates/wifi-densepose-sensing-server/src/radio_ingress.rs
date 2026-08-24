//! Live ADR-341 radio ingress for the sensing server.
//!
//! The boundary accepts only an HMAC-authenticated RVAE envelope. BLE records
//! expose a short-lived rotating pseudonym and RSSI. Channel Sounding records
//! remain edge-local: only a bounded aggregate decision and opaque provenance
//! leave this module. Replay high-water marks are committed atomically before
//! any update is returned to a caller for publication.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use hmac::{Hmac, Mac};
use ruview_fusion::radio_fusion::{
    estimate_channel_sounding_respiration, AuthenticatedGatewayEnvelope, BleIdentityEvidence,
    BleIngressConfig, ChannelSoundingIngressConfig, ChannelSoundingMeasurement,
    GatewayIngressConfig, GatewayPayloadType, RadioEvidenceLabel, RadioIngressError,
    RadioReplayGuard, RadioReplaySnapshot, RespirationDecision, RespirationEstimatorConfig,
    BLE_IDENTITY_PACKET_SIZE, CHANNEL_SOUNDING_FRAME_SIZE, GATEWAY_ENVELOPE_HEADER_SIZE,
    GATEWAY_ENVELOPE_TAG_SIZE,
};
use serde::Serialize;
use sha2::Sha256;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;
const SECRET_SIZE: usize = 32;
const MAX_REPLAY_FILE_SIZE: u64 = 4 * 1024 * 1024;
const MAX_CS_WINDOW_SAMPLES: usize = 4_096;
const CS_WINDOW_US: u64 = 30_000_000;
const MAX_FINALIZED_CS_PROCEDURES: usize = 1_024;
const MAX_CS_SCOPES: usize = 64;
const RADIO_EVIDENCE_TTL_US: i64 = 5_000_000;
const RADIO_WORKER_QUEUE_CAPACITY: usize = 256;
const RADIO_COMMIT_BATCH_LIMIT: usize = 256;
const RADIO_COMMIT_INTERVAL: Duration = Duration::from_secs(1);

/// Return true only for the two exact RVAE datagram sizes implemented by
/// ADR-341. The shared UDP loop calls this before copying attacker-controlled
/// input into the bounded worker queue.
#[must_use]
pub const fn is_supported_rvae_datagram_len(len: usize) -> bool {
    len == GATEWAY_ENVELOPE_HEADER_SIZE + BLE_IDENTITY_PACKET_SIZE + GATEWAY_ENVELOPE_TAG_SIZE
        || len
            == GATEWAY_ENVELOPE_HEADER_SIZE
                + CHANNEL_SOUNDING_FRAME_SIZE
                + GATEWAY_ENVELOPE_TAG_SIZE
}

/// One independently enrolled ESP32 gateway.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatewayRuntimeOptions {
    pub node_id: u8,
    pub key_id: u8,
    pub secret_file: PathBuf,
}

/// Optional companion-radio configuration. ESP32-S3 cannot produce these
/// primitives itself; this identifies a separately enrolled capable radio.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSoundingRuntimeOptions {
    /// Provisioned companion HMAC key selector.
    pub key_id: u8,
    /// Opaque, nonzero companion source identifier.
    pub source_id: u32,
    /// File containing exactly 32 raw secret bytes.
    pub secret_file: PathBuf,
}

/// Configuration required to enable authenticated radio ingress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadioIngressOptions {
    /// One or more independently enrolled ESP32-S3 gateways.
    pub gateways: Vec<GatewayRuntimeOptions>,
    /// Deployment-scoped key used only to derive the canonical `blep:` host
    /// token. It must be distinct from all transport and companion keys.
    pub host_pseudonym_secret_file: PathBuf,
    /// Durable replay state. Deleting it is an explicit replay reset and must
    /// be paired with key rotation or retirement of every old radio session.
    pub replay_state_file: PathBuf,
    /// One-shot creation of a new empty replay snapshot. The runtime rejects
    /// this flag after creation. State loss requires rotation of every key.
    pub initialize_replay_state: bool,
    /// Optional separately enrolled Bluetooth Channel Sounding companion.
    pub channel_sounding: Option<ChannelSoundingRuntimeOptions>,
}

/// A privacy-minimized update safe for the normal sensing broadcast.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
pub enum RadioIngressUpdate {
    /// Authenticated rotating BLE anchor. It is proximity evidence, not a
    /// civil identity and not proof that a device is physically on a person.
    BleIdentity {
        /// This live path is hardware-unvalidated, so it cannot mint MEASURED.
        evidence: RadioEvidenceLabel,
        gateway_node_id: u8,
        pseudonymous_token: String,
        rssi_dbm: i8,
        confidence_permille: u16,
        expires_at_unix_ms: i64,
        scanner_time_verified: bool,
        gateway_timing_uncertainty_us: u32,
    },
    /// Aggregate from complete, coherent Channel Sounding procedures. Exact
    /// phase, RTT, frequency-offset, and per-step samples remain edge-local.
    ChannelSoundingRespiration {
        evidence: RadioEvidenceLabel,
        gateway_node_id: u8,
        source_id: u32,
        source_session_id: u32,
        trigger_procedure_id: u32,
        complete_steps: u16,
        unique_channels: u16,
        window_procedure_ids: Vec<u32>,
        window_sample_count: u16,
        window_start_gateway_boot_us: u64,
        window_end_gateway_boot_us: u64,
        publish_gateway_sequence: u32,
        valid_from_unix_us: i64,
        expires_at_unix_us: i64,
        respiration: RespirationDecision,
    },
}

/// Privacy classes used by the explicit sensing WebSocket export gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RadioPrivacyClass {
    /// Aggregated biological signal evidence.
    P4,
    /// Pseudonymous identity and association evidence.
    P5,
}

/// Explicit deployment export override for the normal sensing WebSocket. This
/// is not a subject consent receipt. The binary restricts it to an audited,
/// loopback-only deployment. Ingestion and edge-local fusion continue when
/// both fields are false.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RadioExportPolicy {
    pub allow_biological_p4: bool,
    pub allow_identity_p5: bool,
}

impl RadioExportPolicy {
    #[must_use]
    pub fn allows(self, update: &RadioIngressUpdate) -> bool {
        match update.privacy_class() {
            RadioPrivacyClass::P4 => self.allow_biological_p4,
            RadioPrivacyClass::P5 => self.allow_identity_p5,
        }
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.allow_biological_p4 || self.allow_identity_p5
    }
}

impl RadioIngressUpdate {
    #[must_use]
    pub fn privacy_class(&self) -> RadioPrivacyClass {
        match self {
            Self::BleIdentity { .. } => RadioPrivacyClass::P5,
            Self::ChannelSoundingRespiration { .. } => RadioPrivacyClass::P4,
        }
    }
}

/// Fail-closed configuration, authentication, replay, and storage failures.
#[derive(Debug, Error)]
pub enum RadioRuntimeError {
    #[error("radio secret file must be a regular non-symlink file")]
    SecretFileType,
    #[error("radio secret file permissions permit group or other access")]
    SecretPermissions,
    #[error("radio secret must contain exactly 32 raw bytes")]
    SecretLength,
    #[error("radio secret must not be all zero")]
    ZeroSecret,
    #[error("gateway, host pseudonym, and companion secrets must all be distinct")]
    ReusedSecret,
    #[error("at least one nonzero gateway enrollment is required")]
    MissingGateway,
    #[error("gateway enrollment node/key pairs must be unique")]
    DuplicateGateway,
    #[error("RVAE node/key selector is not enrolled")]
    UnknownGateway,
    #[error("replay state file must be a regular non-symlink file")]
    ReplayFileType,
    #[error("replay state file permissions permit group or other access")]
    ReplayPermissions,
    #[error("replay state exceeds the bounded file size")]
    ReplayFileSize,
    #[error(
        "replay state is missing; explicitly initialize it and rotate keys after any state loss"
    )]
    ReplayInitializationRequired,
    #[error("replay state already exists; remove the one-shot initialization option")]
    ReplayAlreadyInitialized,
    #[error("replay state is already locked by another sensing process")]
    ReplayLocked,
    #[error("an export override requires a private audit log")]
    ExportAuditRequired,
    #[error("radio export audit file must be a regular non-symlink file")]
    AuditFileType,
    #[error("radio export audit file permissions permit group or other access")]
    AuditPermissions,
    #[error("radio export audit file is already locked by another process")]
    AuditLocked,
    #[error("radio export audit file aliases replay state, a lock, or an enrolled secret")]
    AuditAliasesProtectedFile,
    #[error("radio export audit log has an incomplete or malformed tail")]
    AuditMalformed,
    #[error("invalid replay state: {0}")]
    ReplayState(String),
    #[error("Channel Sounding payload received without an enrolled companion")]
    ChannelSoundingDisabled,
    #[error("radio ingress rejected the frame: {0}")]
    Ingress(#[from] RadioIngressError),
    #[error("radio ingress storage failure: {0}")]
    Storage(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CsScope {
    gateway_node_id: u8,
    gateway_key_id: u8,
    gateway_boot_nonce: u64,
    source_id: u32,
    source_session_id: u32,
}

#[derive(Clone, Debug, Default)]
struct CsState {
    samples: VecDeque<ChannelSoundingMeasurement>,
    finalized_procedures: VecDeque<(u64, u32, u32)>,
    last_capture_at_gateway_boot_us: u64,
}

struct SecretBytes([u8; SECRET_SIZE]);

impl SecretBytes {
    fn expose(&self) -> &[u8; SECRET_SIZE] {
        &self.0
    }
}

impl Drop for SecretBytes {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

struct EnrolledGateway {
    secret: SecretBytes,
    config: GatewayIngressConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FileIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(not(unix))]
    canonical_path: PathBuf,
}

#[derive(Default)]
struct ProtectedFiles {
    identities: BTreeSet<FileIdentity>,
    canonical_paths: BTreeSet<PathBuf>,
}

impl ProtectedFiles {
    fn insert(&mut self, identity: FileIdentity, path: &Path) -> Result<(), RadioRuntimeError> {
        self.identities.insert(identity);
        self.canonical_paths.insert(
            fs::canonicalize(path)
                .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?,
        );
        Ok(())
    }

    fn contains(&self, file: &File, path: &Path) -> Result<bool, RadioRuntimeError> {
        let identity = file_identity(file, path)?;
        let canonical = fs::canonicalize(path)
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        Ok(self.identities.contains(&identity) || self.canonical_paths.contains(&canonical))
    }
}

/// Single-owner live runtime. Production callers hand it to the bounded radio
/// worker so HMAC, estimation, and durable group commit never block CSI UDP.
pub struct RadioIngressRuntime {
    gateways: Vec<EnrolledGateway>,
    host_pseudonym_secret: SecretBytes,
    companion_secret: Option<SecretBytes>,
    companion_config: Option<ChannelSoundingIngressConfig>,
    replay_state_file: PathBuf,
    _replay_lock: File,
    protected_files: ProtectedFiles,
    replay: RadioReplayGuard,
    cs_states: BTreeMap<CsScope, CsState>,
}

impl RadioIngressRuntime {
    /// Load enrolled secrets and durable replay state while holding an
    /// exclusive process lock. A missing snapshot is fail closed unless the
    /// caller explicitly requests initialization.
    pub fn open(options: RadioIngressOptions, now_unix_ms: i64) -> Result<Self, RadioRuntimeError> {
        if now_unix_ms < 0 {
            return Err(RadioRuntimeError::ReplayState(
                "host wall clock predates the Unix epoch".to_string(),
            ));
        }
        if options.gateways.is_empty()
            || options.gateways.iter().any(|gateway| gateway.node_id == 0)
        {
            return Err(RadioRuntimeError::MissingGateway);
        }

        let mut protected_files = ProtectedFiles::default();
        let mut gateway_pairs = BTreeSet::new();
        let mut enrolled_gateways = Vec::with_capacity(options.gateways.len());
        let mut enrolled_secret_values = Vec::with_capacity(options.gateways.len());
        for gateway in options.gateways {
            if !gateway_pairs.insert((gateway.node_id, gateway.key_id)) {
                return Err(RadioRuntimeError::DuplicateGateway);
            }
            let (secret, identity) = read_secret(&gateway.secret_file)?;
            protected_files.insert(identity, &gateway.secret_file)?;
            if enrolled_secret_values.iter().any(|value| value == &secret) {
                return Err(RadioRuntimeError::ReusedSecret);
            }
            enrolled_secret_values.push(secret);
            enrolled_gateways.push(EnrolledGateway {
                secret: SecretBytes(secret),
                config: GatewayIngressConfig {
                    node_id: gateway.node_id,
                    key_id: gateway.key_id,
                    ..GatewayIngressConfig::default()
                },
            });
        }

        let (host_pseudonym_secret_value, host_identity) =
            read_secret(&options.host_pseudonym_secret_file)?;
        protected_files.insert(host_identity, &options.host_pseudonym_secret_file)?;
        if enrolled_secret_values
            .iter()
            .any(|secret| secret == &host_pseudonym_secret_value)
        {
            return Err(RadioRuntimeError::ReusedSecret);
        }
        let host_pseudonym_secret = SecretBytes(host_pseudonym_secret_value);
        let (companion_secret, companion_config) = match options.channel_sounding {
            Some(companion) => {
                if companion.source_id == 0 {
                    return Err(RadioRuntimeError::ReplayState(
                        "companion source id must be nonzero".to_string(),
                    ));
                }
                let (secret, identity) = read_secret(&companion.secret_file)?;
                protected_files.insert(identity, &companion.secret_file)?;
                if enrolled_secret_values.iter().any(|value| value == &secret)
                    || secret == *host_pseudonym_secret.expose()
                {
                    return Err(RadioRuntimeError::ReusedSecret);
                }
                (
                    Some(SecretBytes(secret)),
                    Some(ChannelSoundingIngressConfig {
                        key_id: companion.key_id,
                        source_id: companion.source_id,
                        ..ChannelSoundingIngressConfig::default()
                    }),
                )
            }
            None => (None, None),
        };

        let parent = replay_parent(&options.replay_state_file);
        if !parent.exists() {
            if !options.initialize_replay_state {
                return Err(RadioRuntimeError::ReplayInitializationRequired);
            }
            fs::create_dir_all(parent)
                .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        }
        let (replay_lock, replay_lock_identity, replay_lock_path) =
            acquire_replay_lock(&options.replay_state_file)?;
        protected_files.insert(replay_lock_identity, &replay_lock_path)?;
        let (replay, replay_identity, needs_initial_snapshot) =
            match read_replay_guard(&options.replay_state_file, now_unix_ms)? {
                Some(_) if options.initialize_replay_state => {
                    return Err(RadioRuntimeError::ReplayAlreadyInitialized)
                }
                Some((replay, identity)) => (replay, Some(identity), false),
                None if options.initialize_replay_state => {
                    (RadioReplayGuard::default(), None, true)
                }
                None => return Err(RadioRuntimeError::ReplayInitializationRequired),
            };
        if let Some(identity) = replay_identity {
            protected_files.insert(identity, &options.replay_state_file)?;
        }
        let mut runtime = Self {
            gateways: enrolled_gateways,
            host_pseudonym_secret,
            companion_secret,
            companion_config,
            replay_state_file: options.replay_state_file,
            _replay_lock: replay_lock,
            protected_files,
            replay,
            cs_states: BTreeMap::new(),
        };
        if needs_initial_snapshot {
            runtime.persist_replay(&runtime.replay)?;
            let replay_file = open_read_nofollow(&runtime.replay_state_file)
                .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
            let identity = file_identity(&replay_file, &runtime.replay_state_file)?;
            runtime
                .protected_files
                .insert(identity, &runtime.replay_state_file)?;
        }
        Ok(runtime)
    }

    /// Verify and stage one RVAE datagram. `Ok(None)` means a valid Channel
    /// Sounding step was buffered but no complete procedure was ready. Callers
    /// must call [`Self::commit`] before publishing any returned update.
    pub fn stage(
        &mut self,
        frame: &[u8],
        host_received_at_unix_us: i64,
    ) -> Result<Option<RadioIngressUpdate>, RadioRuntimeError> {
        let gateway_index = self.gateway_index(frame)?;
        let gateway = &self.gateways[gateway_index];
        let envelope =
            AuthenticatedGatewayEnvelope::parse(frame, gateway.secret.expose(), gateway.config)?;
        match envelope.metadata.payload_type {
            GatewayPayloadType::BleIdentity => {
                let mut candidate_replay = self.replay.clone();
                let evidence = BleIdentityEvidence::parse_gateway_authenticated(
                    frame,
                    host_received_at_unix_us.div_euclid(1_000),
                    gateway.secret.expose(),
                    gateway.config,
                    BleIngressConfig::default(),
                    &mut candidate_replay,
                )?;
                self.replay = candidate_replay;
                Ok(Some(ble_update(
                    evidence,
                    self.host_pseudonym_secret.expose(),
                )))
            }
            GatewayPayloadType::ChannelSounding => {
                let companion_secret = self
                    .companion_secret
                    .as_ref()
                    .ok_or(RadioRuntimeError::ChannelSoundingDisabled)?;
                let companion_config = self
                    .companion_config
                    .ok_or(RadioRuntimeError::ChannelSoundingDisabled)?;
                let mut candidate_replay = self.replay.clone();
                let measurement = ChannelSoundingMeasurement::parse_gateway_authenticated(
                    frame,
                    host_received_at_unix_us,
                    gateway.secret.expose(),
                    gateway.config,
                    companion_secret.expose(),
                    companion_config,
                    &mut candidate_replay,
                )?;
                self.ingest_channel_sounding(measurement, candidate_replay)
            }
        }
    }

    /// Persist all staged replay high-water marks atomically.
    pub fn commit(&self) -> Result<(), RadioRuntimeError> {
        self.persist_replay(&self.replay)
    }

    /// Test and administrative convenience path that stages and commits one
    /// datagram before returning an update.
    pub fn ingest_durable(
        &mut self,
        frame: &[u8],
        host_received_at_unix_us: i64,
    ) -> Result<Option<RadioIngressUpdate>, RadioRuntimeError> {
        let update = self.stage(frame, host_received_at_unix_us)?;
        self.commit()?;
        Ok(update)
    }

    fn gateway_index(&self, frame: &[u8]) -> Result<usize, RadioRuntimeError> {
        if frame.len() <= 12 {
            return Err(RadioRuntimeError::Ingress(RadioIngressError::Length));
        }
        self.gateways
            .iter()
            .position(|gateway| {
                gateway.config.key_id == frame[7] && gateway.config.node_id == frame[12]
            })
            .ok_or(RadioRuntimeError::UnknownGateway)
    }

    /// Path whose atomic snapshot protects publication across process restarts.
    #[must_use]
    pub fn replay_state_file(&self) -> &Path {
        &self.replay_state_file
    }

    fn ingest_channel_sounding(
        &mut self,
        measurement: ChannelSoundingMeasurement,
        candidate_replay: RadioReplayGuard,
    ) -> Result<Option<RadioIngressUpdate>, RadioRuntimeError> {
        let scope = CsScope {
            gateway_node_id: measurement.gateway_node_id,
            gateway_key_id: measurement.gateway_key_id,
            gateway_boot_nonce: measurement.gateway_boot_nonce,
            source_id: measurement.source_id,
            source_session_id: measurement.source_session_id,
        };
        let candidate_state = self.cs_states.get(&scope).cloned().unwrap_or_default();
        let mut candidate_samples = candidate_state.samples;
        let mut candidate_finalized = candidate_state.finalized_procedures;
        let procedure_key = (
            measurement.gateway_boot_nonce,
            measurement.source_session_id,
            measurement.procedure_id,
        );
        if candidate_finalized.contains(&procedure_key) {
            self.install_cs_state(
                scope,
                candidate_replay,
                candidate_samples,
                candidate_finalized,
                measurement.capture_at_gateway_boot_us,
            );
            return Ok(None);
        }

        candidate_samples.push_back(measurement.clone());
        let newest_capture = measurement.capture_at_gateway_boot_us;
        candidate_samples.retain(|sample| {
            newest_capture.saturating_sub(sample.capture_at_gateway_boot_us) <= CS_WINDOW_US
        });
        while candidate_samples.len() > MAX_CS_WINDOW_SAMPLES {
            candidate_samples.pop_front();
        }

        let procedure_samples: Vec<_> = candidate_samples
            .iter()
            .filter(|sample| sample.procedure_id == measurement.procedure_id)
            .collect();
        let declared_count = usize::from(measurement.step_count);
        let unique_steps: BTreeSet<_> = procedure_samples
            .iter()
            .map(|sample| sample.step_index)
            .collect();
        let unique_channels: BTreeSet<_> = procedure_samples
            .iter()
            .map(|sample| sample.channel_index)
            .collect();
        let invalid = !(4..=79).contains(&measurement.step_count)
            || procedure_samples
                .iter()
                .any(|sample| sample.step_count != measurement.step_count)
            || procedure_samples.len() > declared_count
            || unique_steps.len() != procedure_samples.len()
            || unique_channels.len() != procedure_samples.len();
        if invalid {
            candidate_samples.retain(|sample| sample.procedure_id != measurement.procedure_id);
            remember_finalized(&mut candidate_finalized, procedure_key);
            self.install_cs_state(
                scope,
                candidate_replay,
                candidate_samples,
                candidate_finalized,
                newest_capture,
            );
            return Ok(None);
        }
        if procedure_samples.len() < declared_count {
            self.install_cs_state(
                scope,
                candidate_replay,
                candidate_samples,
                candidate_finalized,
                newest_capture,
            );
            return Ok(None);
        }
        let exact_steps = (0..measurement.step_count).collect::<BTreeSet<_>>();
        if unique_steps != exact_steps {
            candidate_samples.retain(|sample| sample.procedure_id != measurement.procedure_id);
            remember_finalized(&mut candidate_finalized, procedure_key);
            self.install_cs_state(
                scope,
                candidate_replay,
                candidate_samples,
                candidate_finalized,
                newest_capture,
            );
            return Ok(None);
        }

        let mut grouped: BTreeMap<u32, Vec<&ChannelSoundingMeasurement>> = BTreeMap::new();
        for sample in &candidate_samples {
            grouped.entry(sample.procedure_id).or_default().push(sample);
        }
        let mut complete_groups: Vec<_> = grouped
            .into_iter()
            .filter(|(_, samples)| valid_complete_procedure(samples))
            .map(|(procedure_id, samples)| {
                let first_capture = samples
                    .iter()
                    .map(|sample| sample.capture_at_gateway_boot_us)
                    .min()
                    .unwrap_or(0);
                (first_capture, procedure_id)
            })
            .collect();
        complete_groups.sort_unstable();
        let window_procedure_ids: Vec<_> = complete_groups
            .iter()
            .map(|(_, procedure_id)| *procedure_id)
            .collect();
        let admitted_ids: BTreeSet<_> = window_procedure_ids.iter().copied().collect();
        let samples: Vec<_> = candidate_samples
            .iter()
            .filter(|sample| admitted_ids.contains(&sample.procedure_id))
            .cloned()
            .collect();
        let respiration = estimate_channel_sounding_respiration(
            &samples,
            RespirationEstimatorConfig {
                evidence: RadioEvidenceLabel::Claimed,
                ..RespirationEstimatorConfig::default()
            },
        );
        let unique_channels = u16::try_from(unique_channels.len()).unwrap_or(u16::MAX);
        let window_sample_count = u16::try_from(samples.len()).unwrap_or(u16::MAX);
        let window_start_gateway_boot_us = samples
            .iter()
            .map(|sample| sample.capture_at_gateway_boot_us)
            .min()
            .unwrap_or(measurement.capture_at_gateway_boot_us);
        let window_end_gateway_boot_us = samples
            .iter()
            .map(|sample| sample.capture_at_gateway_boot_us)
            .max()
            .unwrap_or(measurement.capture_at_gateway_boot_us);
        let expires_at_unix_us = measurement
            .host_received_at_unix_us
            .checked_add(RADIO_EVIDENCE_TTL_US)
            .ok_or(RadioIngressError::Bounds)?;
        let update = RadioIngressUpdate::ChannelSoundingRespiration {
            evidence: RadioEvidenceLabel::Claimed,
            gateway_node_id: measurement.gateway_node_id,
            source_id: measurement.source_id,
            source_session_id: measurement.source_session_id,
            trigger_procedure_id: measurement.procedure_id,
            complete_steps: measurement.step_count,
            unique_channels,
            window_procedure_ids,
            window_sample_count,
            window_start_gateway_boot_us,
            window_end_gateway_boot_us,
            publish_gateway_sequence: measurement.gateway_sequence,
            valid_from_unix_us: measurement.host_received_at_unix_us,
            expires_at_unix_us,
            respiration,
        };

        remember_finalized(&mut candidate_finalized, procedure_key);
        self.install_cs_state(
            scope,
            candidate_replay,
            candidate_samples,
            candidate_finalized,
            newest_capture,
        );
        Ok(Some(update))
    }

    fn install_cs_state(
        &mut self,
        scope: CsScope,
        replay: RadioReplayGuard,
        samples: VecDeque<ChannelSoundingMeasurement>,
        finalized_procedures: VecDeque<(u64, u32, u32)>,
        last_capture_at_gateway_boot_us: u64,
    ) {
        self.replay = replay;
        self.cs_states.insert(
            scope,
            CsState {
                samples,
                finalized_procedures,
                last_capture_at_gateway_boot_us,
            },
        );
        while self.cs_states.len() > MAX_CS_SCOPES {
            let Some(oldest) = self
                .cs_states
                .iter()
                .min_by_key(|(_, state)| state.last_capture_at_gateway_boot_us)
                .map(|(scope, _)| *scope)
            else {
                break;
            };
            self.cs_states.remove(&oldest);
        }
    }

    fn persist_replay(&self, replay: &RadioReplayGuard) -> Result<(), RadioRuntimeError> {
        let snapshot = replay.snapshot()?;
        let bytes = serde_json::to_vec(&snapshot)
            .map_err(|error| RadioRuntimeError::ReplayState(error.to_string()))?;
        atomic_write_private(&self.replay_state_file, &bytes)
    }
}

fn valid_complete_procedure(samples: &[&ChannelSoundingMeasurement]) -> bool {
    let Some(first) = samples.first() else {
        return false;
    };
    if !(4..=79).contains(&first.step_count) || samples.len() != usize::from(first.step_count) {
        return false;
    }
    let steps: BTreeSet<_> = samples.iter().map(|sample| sample.step_index).collect();
    let channels: BTreeSet<_> = samples.iter().map(|sample| sample.channel_index).collect();
    samples.iter().all(|sample| {
        sample.step_count == first.step_count
            && sample.source_id == first.source_id
            && sample.source_session_id == first.source_session_id
            && sample.gateway_node_id == first.gateway_node_id
            && sample.gateway_key_id == first.gateway_key_id
            && sample.gateway_boot_nonce == first.gateway_boot_nonce
    }) && steps == (0..first.step_count).collect()
        && channels.len() == samples.len()
}

fn remember_finalized(finalized: &mut VecDeque<(u64, u32, u32)>, procedure: (u64, u32, u32)) {
    if !finalized.contains(&procedure) {
        finalized.push_back(procedure);
    }
    while finalized.len() > MAX_FINALIZED_CS_PROCEDURES {
        finalized.pop_front();
    }
}

fn ble_update(
    evidence: BleIdentityEvidence,
    host_pseudonym_secret: &[u8; 32],
) -> RadioIngressUpdate {
    let mut mac = HmacSha256::new_from_slice(host_pseudonym_secret)
        .expect("HMAC-SHA256 accepts a 32-byte key");
    mac.update(b"rufield.ble.identity.v1\0");
    mac.update(&evidence.pseudonym);
    mac.update(&u64::from(evidence.token_epoch_min).to_le_bytes());
    let digest = mac.finalize().into_bytes();
    let pseudonymous_token = format!(
        "blep:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    RadioIngressUpdate::BleIdentity {
        evidence: RadioEvidenceLabel::Claimed,
        gateway_node_id: evidence.node_id,
        pseudonymous_token,
        rssi_dbm: evidence.rssi_dbm,
        confidence_permille: evidence.confidence_permille,
        expires_at_unix_ms: evidence.expires_at_unix_ms,
        scanner_time_verified: evidence.scanner_time_verified,
        gateway_timing_uncertainty_us: evidence.gateway_timing_uncertainty_us,
    }
}

fn read_secret(path: &Path) -> Result<([u8; SECRET_SIZE], FileIdentity), RadioRuntimeError> {
    let mut file = match open_read_nofollow(path) {
        Ok(file) => file,
        Err(error) if is_symlink_open_error(&error) => {
            return Err(RadioRuntimeError::SecretFileType)
        }
        Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if !metadata.is_file() {
        return Err(RadioRuntimeError::SecretFileType);
    }
    let identity = file_identity(&file, path)?;
    check_private_permissions(&metadata, RadioRuntimeError::SecretPermissions)?;
    if metadata.len() != SECRET_SIZE as u64 {
        return Err(RadioRuntimeError::SecretLength);
    }
    let mut secret = [0u8; SECRET_SIZE];
    file.read_exact(&mut secret)
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if secret.iter().all(|byte| *byte == 0) {
        return Err(RadioRuntimeError::ZeroSecret);
    }
    Ok((secret, identity))
}

fn read_replay_guard(
    path: &Path,
    now_unix_ms: i64,
) -> Result<Option<(RadioReplayGuard, FileIdentity)>, RadioRuntimeError> {
    let file = match open_read_nofollow(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) if is_symlink_open_error(&error) => {
            return Err(RadioRuntimeError::ReplayFileType)
        }
        Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if !metadata.is_file() {
        return Err(RadioRuntimeError::ReplayFileType);
    }
    let identity = file_identity(&file, path)?;
    check_private_permissions(&metadata, RadioRuntimeError::ReplayPermissions)?;
    if metadata.len() > MAX_REPLAY_FILE_SIZE {
        return Err(RadioRuntimeError::ReplayFileSize);
    }
    let mut bytes = Vec::new();
    file.take(MAX_REPLAY_FILE_SIZE + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if bytes.len() as u64 > MAX_REPLAY_FILE_SIZE {
        return Err(RadioRuntimeError::ReplayFileSize);
    }
    let snapshot: RadioReplaySnapshot = serde_json::from_slice(&bytes)
        .map_err(|error| RadioRuntimeError::ReplayState(error.to_string()))?;
    RadioReplayGuard::from_snapshot(snapshot, now_unix_ms)
        .map(|replay| Some((replay, identity)))
        .map_err(RadioRuntimeError::from)
}

fn replay_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn acquire_replay_lock(path: &Path) -> Result<(File, FileIdentity, PathBuf), RadioRuntimeError> {
    let mut lock_name = OsString::from(path.as_os_str());
    lock_name.push(".lock");
    let lock_path = PathBuf::from(lock_name);
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = match options.open(&lock_path) {
        Ok(file) => file,
        Err(error) if is_symlink_open_error(&error) => {
            return Err(RadioRuntimeError::ReplayFileType)
        }
        Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if !metadata.is_file() {
        return Err(RadioRuntimeError::ReplayFileType);
    }
    let identity = file_identity(&file, &lock_path)?;
    check_private_permissions(&metadata, RadioRuntimeError::ReplayPermissions)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok((file, identity, lock_path)),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            Err(RadioRuntimeError::ReplayLocked)
        }
        Err(error) => Err(RadioRuntimeError::Storage(error.to_string())),
    }
}

#[cfg(unix)]
fn file_identity(file: &File, _path: &Path) -> Result<FileIdentity, RadioRuntimeError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn file_identity(_file: &File, path: &Path) -> Result<FileIdentity, RadioRuntimeError> {
    Ok(FileIdentity {
        canonical_path: fs::canonicalize(path)
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?,
    })
}

fn open_read_nofollow(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(not(unix))]
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "symbolic links are not accepted",
        ));
    }
    options.open(path)
}

fn is_symlink_open_error(error: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(libc::ELOOP)
    }
    #[cfg(not(unix))]
    {
        error.kind() == std::io::ErrorKind::InvalidInput
    }
}

fn atomic_write_private(path: &Path, bytes: &[u8]) -> Result<(), RadioRuntimeError> {
    if bytes.len() as u64 > MAX_REPLAY_FILE_SIZE {
        return Err(RadioRuntimeError::ReplayFileSize);
    }
    let parent = replay_parent(path);
    fs::create_dir_all(parent).map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            RadioRuntimeError::Storage("replay state path has no UTF-8 filename".to_string())
        })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?
        .as_nanos();
    let temporary = parent.join(format!(".{name}.tmp.{}.{nonce}", std::process::id()));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let mut file = options
            .open(&temporary)
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        fs::rename(&temporary, path)
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// An authenticated-radio datagram copied out of the shared UDP receive loop.
/// The bounded worker owns all cryptography, estimation, and storage work.
#[derive(Debug)]
pub struct RadioDatagram {
    pub frame: Vec<u8>,
    pub host_received_at_unix_us: i64,
    pub source: SocketAddr,
}

/// Start the single-owner radio worker. At most 256 datagrams can wait in
/// memory. Accepted records are group committed at most once per second or per
/// 256 records, and updates are released only after the snapshot and audit
/// record are durable.
pub fn spawn_radio_ingress_worker(
    runtime: RadioIngressRuntime,
    export_policy: RadioExportPolicy,
    export_audit_log: Option<PathBuf>,
    broadcast: tokio::sync::broadcast::Sender<String>,
) -> Result<SyncSender<RadioDatagram>, RadioRuntimeError> {
    let audit = match (export_policy.any(), export_audit_log) {
        (true, Some(path)) => Some(open_export_audit(&path, &runtime.protected_files)?),
        (true, None) => return Err(RadioRuntimeError::ExportAuditRequired),
        (false, _) => None,
    };
    let (sender, receiver) = mpsc::sync_channel(RADIO_WORKER_QUEUE_CAPACITY);
    thread::Builder::new()
        .name("ruview-radio-ingress".to_string())
        .spawn(move || {
            if let Err(error) =
                run_radio_worker(runtime, export_policy, audit, broadcast, receiver)
            {
                tracing::error!(
                    %error,
                    "radio ingress worker stopped; its queue is disconnected and radio ingress is unhealthy"
                );
            }
        })
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    Ok(sender)
}

fn run_radio_worker(
    mut runtime: RadioIngressRuntime,
    export_policy: RadioExportPolicy,
    mut audit: Option<File>,
    broadcast: tokio::sync::broadcast::Sender<String>,
    receiver: Receiver<RadioDatagram>,
) -> Result<(), RadioRuntimeError> {
    let mut rejected_since_log = 0u64;
    let mut first_rejection = None::<String>;
    let mut last_rejection_log = Instant::now();
    while let Ok(first) = receiver.recv() {
        let started = Instant::now();
        let mut batch = vec![first];
        let mut disconnected = false;
        while batch.len() < RADIO_COMMIT_BATCH_LIMIT {
            let Some(remaining) = RADIO_COMMIT_INTERVAL.checked_sub(started.elapsed()) else {
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(datagram) => batch.push(datagram),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        while batch.len() < RADIO_COMMIT_BATCH_LIMIT {
            match receiver.try_recv() {
                Ok(datagram) => batch.push(datagram),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        let mut accepted = 0usize;
        let mut pending_updates = Vec::new();
        for datagram in batch {
            match runtime.stage(&datagram.frame, datagram.host_received_at_unix_us) {
                Ok(update) => {
                    accepted += 1;
                    if let Some(update) = update {
                        pending_updates.push(update);
                    }
                }
                Err(error) => {
                    rejected_since_log = rejected_since_log.saturating_add(1);
                    first_rejection.get_or_insert_with(|| {
                        format!("source={}, error={error}", datagram.source)
                    });
                }
            }
        }

        if accepted > 0 {
            if let Err(error) = runtime.commit() {
                tracing::error!(
                    %error,
                    accepted,
                    "radio replay group commit failed; worker stopped and no pending evidence was published"
                );
                return Err(error);
            }
            publish_committed_updates(&pending_updates, export_policy, audit.as_mut(), &broadcast)?;
        }

        if rejected_since_log > 0 && last_rejection_log.elapsed() >= Duration::from_secs(1) {
            tracing::warn!(
                rejected = rejected_since_log,
                first = first_rejection.as_deref().unwrap_or("unknown"),
                "authenticated-radio frames rejected during the last reporting interval"
            );
            rejected_since_log = 0;
            first_rejection = None;
            last_rejection_log = Instant::now();
        }
        if disconnected {
            break;
        }
    }
    Ok(())
}

fn publish_committed_updates(
    updates: &[RadioIngressUpdate],
    export_policy: RadioExportPolicy,
    audit: Option<&mut File>,
    broadcast: &tokio::sync::broadcast::Sender<String>,
) -> Result<(), RadioRuntimeError> {
    let now_unix_us = unix_time_us();
    let mut messages = Vec::new();
    let mut p4_count = 0usize;
    let mut p5_count = 0usize;
    for update in updates {
        if !export_policy.allows(update) || !update_is_live(update, now_unix_us) {
            continue;
        }
        match update.privacy_class() {
            RadioPrivacyClass::P4 => p4_count += 1,
            RadioPrivacyClass::P5 => p5_count += 1,
        }
        let message = serde_json::json!({
            "type": "radio_evidence",
            "provenance": "CLAIMED",
            "privacy_class": update.privacy_class(),
            "export_gate": "audited_local_override",
            "data": update,
        });
        messages.push(
            serde_json::to_string(&message)
                .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?,
        );
    }
    if messages.is_empty() {
        return Ok(());
    }
    let audit = audit.ok_or(RadioRuntimeError::ExportAuditRequired)?;
    let audit_record = serde_json::json!({
        "event": "radio_export_batch",
        "at_unix_us": now_unix_us,
        "p4_count": p4_count,
        "p5_count": p5_count,
    });
    serde_json::to_vec(&audit_record)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        .and_then(|mut bytes| {
            bytes.push(b'\n');
            audit.write_all(&bytes).and_then(|()| audit.sync_data())
        })
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    for message in messages {
        let _ = broadcast.send(message);
    }
    Ok(())
}

fn update_is_live(update: &RadioIngressUpdate, now_unix_us: i64) -> bool {
    match update {
        RadioIngressUpdate::BleIdentity {
            expires_at_unix_ms, ..
        } => now_unix_us.div_euclid(1_000) < *expires_at_unix_ms,
        RadioIngressUpdate::ChannelSoundingRespiration {
            valid_from_unix_us,
            expires_at_unix_us,
            ..
        } => now_unix_us >= *valid_from_unix_us && now_unix_us < *expires_at_unix_us,
    }
}

fn unix_time_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_micros()).ok())
        .unwrap_or(i64::MAX)
}

fn audit_open_options(create_new: bool) -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).append(true).create_new(create_new);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    options
}

fn open_export_audit(
    path: &Path,
    protected_files: &ProtectedFiles,
) -> Result<File, RadioRuntimeError> {
    let parent = replay_parent(path);
    if !parent.exists() {
        return Err(RadioRuntimeError::Storage(format!(
            "audit directory does not exist: {}",
            parent.display()
        )));
    }
    let (mut file, created) = match audit_open_options(true).open(path) {
        Ok(file) => (file, true),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            match audit_open_options(false).open(path) {
                Ok(file) => (file, false),
                Err(error) if is_symlink_open_error(&error) => {
                    return Err(RadioRuntimeError::AuditFileType)
                }
                Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
            }
        }
        Err(error) if is_symlink_open_error(&error) => {
            return Err(RadioRuntimeError::AuditFileType)
        }
        Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    if !metadata.is_file() {
        return Err(RadioRuntimeError::AuditFileType);
    }
    check_private_permissions(&metadata, RadioRuntimeError::AuditPermissions)?;
    if protected_files.contains(&file, path)? {
        return Err(RadioRuntimeError::AuditAliasesProtectedFile);
    }
    if metadata.len() > 0 {
        file.seek(SeekFrom::End(-1))
            .and_then(|_| {
                let mut tail = [0u8; 1];
                file.read_exact(&mut tail).map(|()| tail)
            })
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))
            .and_then(|tail| {
                if tail == [b'\n'] {
                    Ok(())
                } else {
                    Err(RadioRuntimeError::AuditMalformed)
                }
            })?;
    }
    match file.try_lock_exclusive() {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            return Err(RadioRuntimeError::AuditLocked)
        }
        Err(error) => return Err(RadioRuntimeError::Storage(error.to_string())),
    }
    if created {
        file.sync_all()
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| RadioRuntimeError::Storage(error.to_string()))?;
    }
    Ok(file)
}

#[cfg(unix)]
fn check_private_permissions(
    metadata: &fs::Metadata,
    error: RadioRuntimeError,
) -> Result<(), RadioRuntimeError> {
    use std::os::unix::fs::PermissionsExt;
    if metadata.permissions().mode() & 0o077 != 0 {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
fn check_private_permissions(
    _metadata: &fs::Metadata,
    _error: RadioRuntimeError,
) -> Result<(), RadioRuntimeError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    use tempfile::TempDir;

    type HmacSha256 = Hmac<Sha256>;
    const GW_SECRET: [u8; 32] = [0x31; 32];
    const GW_SECRET_2: [u8; 32] = [0x53; 32];
    const HOST_SECRET: [u8; 32] = [0x42; 32];
    const CS_SECRET: [u8; 32] = [0x64; 32];

    fn write_secret(path: &Path, secret: &[u8; 32]) {
        fs::write(path, secret).expect("write secret");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).expect("chmod secret");
        }
    }

    fn options(temp: &TempDir) -> RadioIngressOptions {
        let secret = temp.path().join("gateway.bin");
        let host_secret = temp.path().join("host-pseudonym.bin");
        write_secret(&secret, &GW_SECRET);
        write_secret(&host_secret, &HOST_SECRET);
        RadioIngressOptions {
            gateways: vec![GatewayRuntimeOptions {
                node_id: 7,
                key_id: 3,
                secret_file: secret,
            }],
            host_pseudonym_secret_file: host_secret,
            replay_state_file: temp.path().join("replay.json"),
            initialize_replay_state: true,
            channel_sounding: None,
        }
    }

    fn ble_envelope(sequence: u32, inner_sequence: u32, epoch_min: u32) -> Vec<u8> {
        ble_envelope_for(sequence, inner_sequence, epoch_min, 7, 3, &GW_SECRET)
    }

    fn ble_envelope_for(
        sequence: u32,
        inner_sequence: u32,
        epoch_min: u32,
        node_id: u8,
        gateway_key_id: u8,
        gateway_secret: &[u8; 32],
    ) -> Vec<u8> {
        let mut payload = vec![0u8; 36];
        payload[0..4]
            .copy_from_slice(&ruview_fusion::radio_fusion::BLE_IDENTITY_MAGIC.to_le_bytes());
        payload[4] = 1;
        payload[5] = node_id;
        payload[6] = 0b11;
        payload[7] = 9;
        payload[8..12].copy_from_slice(&inner_sequence.to_le_bytes());
        payload[12..16].copy_from_slice(&1_000u32.to_le_bytes());
        payload[16..18].copy_from_slice(&5_000u16.to_le_bytes());
        payload[18..20].copy_from_slice(&800u16.to_le_bytes());
        payload[20] = (-55i8) as u8;
        payload[21] = (-8i8) as u8;
        payload[24..32].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        payload[32..36].copy_from_slice(&epoch_min.to_le_bytes());

        let total = 40 + payload.len() + 16;
        let mut frame = vec![0u8; total];
        frame[0..4]
            .copy_from_slice(&ruview_fusion::radio_fusion::GATEWAY_ENVELOPE_MAGIC.to_le_bytes());
        frame[4] = 1;
        frame[5] = 1;
        frame[6] = 1;
        frame[7] = gateway_key_id;
        frame[8..10].copy_from_slice(&(total as u16).to_le_bytes());
        frame[10..12].copy_from_slice(&(payload.len() as u16).to_le_bytes());
        frame[12] = node_id;
        frame[16..20].copy_from_slice(&sequence.to_le_bytes());
        frame[20..28].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        frame[28..36].copy_from_slice(&2_000_000u64.to_le_bytes());
        frame[36..40].copy_from_slice(&2_000u32.to_le_bytes());
        frame[40..40 + payload.len()].copy_from_slice(&payload);
        let signed_len = 40 + payload.len();
        let mut mac = HmacSha256::new_from_slice(gateway_secret).expect("HMAC key");
        mac.update(b"RuView/GW/v1");
        mac.update(&frame[..signed_len]);
        let tag = mac.finalize().into_bytes();
        frame[signed_len..].copy_from_slice(&tag[..16]);
        frame
    }

    #[test]
    fn missing_replay_requires_explicit_initialization_and_runtime_is_exclusive() {
        let temp = TempDir::new().expect("tempdir");
        let mut uninitialized = options(&temp);
        uninitialized.initialize_replay_state = false;
        assert!(matches!(
            RadioIngressRuntime::open(uninitialized, 1_900_000_000_000),
            Err(RadioRuntimeError::ReplayInitializationRequired)
        ));

        let first =
            RadioIngressRuntime::open(options(&temp), 1_900_000_000_000).expect("first runtime");
        assert!(matches!(
            RadioIngressRuntime::open(options(&temp), 1_900_000_000_000),
            Err(RadioRuntimeError::ReplayLocked)
        ));
        drop(first);
        assert!(matches!(
            RadioIngressRuntime::open(options(&temp), 1_900_000_000_000),
            Err(RadioRuntimeError::ReplayAlreadyInitialized)
        ));
        let mut restart = options(&temp);
        restart.initialize_replay_state = false;
        RadioIngressRuntime::open(restart, 1_900_000_000_000)
            .expect("lock released and one-shot initialization removed");
    }

    #[test]
    fn admits_only_exact_supported_rvae_envelope_lengths() {
        assert!(is_supported_rvae_datagram_len(92));
        assert!(is_supported_rvae_datagram_len(128));
        for length in [0, 4, 91, 93, 127, 129, 65_507] {
            assert!(!is_supported_rvae_datagram_len(length), "length={length}");
        }
    }

    #[test]
    fn independently_authenticates_two_enrolled_gateways() {
        let temp = TempDir::new().expect("tempdir");
        let mut options = options(&temp);
        let second_secret = temp.path().join("gateway-2.bin");
        write_secret(&second_secret, &GW_SECRET_2);
        options.gateways.push(GatewayRuntimeOptions {
            node_id: 8,
            key_id: 4,
            secret_file: second_secret,
        });
        let now_ms = 1_900_000_000_000i64;
        let epoch = (now_ms / 60_000) as u32;
        let mut runtime = RadioIngressRuntime::open(options, now_ms).expect("runtime");
        runtime
            .ingest_durable(&ble_envelope(1, 1, epoch), now_ms * 1_000)
            .expect("primary gateway");
        let second = ble_envelope_for(1, 1, epoch, 8, 4, &GW_SECRET_2);
        let update = runtime
            .ingest_durable(&second, now_ms * 1_000)
            .expect("second gateway")
            .expect("BLE update");
        assert!(matches!(
            update,
            RadioIngressUpdate::BleIdentity {
                gateway_node_id: 8,
                ..
            }
        ));
    }

    #[test]
    fn worker_commits_and_audits_before_p5_publication() {
        let temp = TempDir::new().expect("tempdir");
        let now_ms = chrono::Utc::now().timestamp_millis();
        let runtime = RadioIngressRuntime::open(options(&temp), now_ms).expect("runtime");
        let audit_path = temp.path().join("radio-audit.jsonl");
        let (broadcast, mut receiver) = tokio::sync::broadcast::channel(8);
        let sender = spawn_radio_ingress_worker(
            runtime,
            RadioExportPolicy {
                allow_biological_p4: false,
                allow_identity_p5: true,
            },
            Some(audit_path.clone()),
            broadcast,
        )
        .expect("worker");
        sender
            .send(RadioDatagram {
                frame: ble_envelope(1, 1, (now_ms / 60_000) as u32),
                host_received_at_unix_us: now_ms * 1_000,
                source: "127.0.0.1:5005".parse().expect("socket"),
            })
            .expect("queue");
        std::thread::sleep(Duration::from_millis(100));
        assert!(matches!(
            receiver.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
        std::thread::sleep(Duration::from_millis(1_050));
        let message = receiver.try_recv().expect("publication after group commit");
        assert!(message.contains("audited_local_override"));
        assert!(fs::read_to_string(&audit_path)
            .expect("audit")
            .contains("radio_export_batch"));
        assert!(fs::read_to_string(temp.path().join("replay.json"))
            .expect("replay")
            .contains("gateway"));
    }

    #[test]
    fn persists_replay_before_return_and_rejects_after_restart() {
        let temp = TempDir::new().expect("tempdir");
        let now_ms = 1_900_000_000_000i64;
        let frame = ble_envelope(1, 1, (now_ms / 60_000) as u32);
        let mut runtime = RadioIngressRuntime::open(options(&temp), now_ms).expect("open runtime");
        let update = runtime
            .ingest_durable(&frame, now_ms * 1_000)
            .expect("ingest")
            .expect("BLE emits");
        assert!(matches!(update, RadioIngressUpdate::BleIdentity { .. }));
        let update_json = serde_json::to_string(&update).expect("serialize update");
        assert!(update_json.contains("\"pseudonymous_token\":\"blep:"));
        assert!(!update_json.contains("[1,2,3,4,5,6,7,8]"));
        assert!(runtime.replay_state_file().exists());
        let replay_json = fs::read_to_string(runtime.replay_state_file()).expect("read replay");
        assert!(!replay_json.contains("\"pseudonym\""));
        assert!(!replay_json.contains("[1,2,3,4,5,6,7,8]"));

        drop(runtime);
        let mut restart_options = options(&temp);
        restart_options.initialize_replay_state = false;
        let mut restarted = RadioIngressRuntime::open(restart_options, now_ms).expect("restart");
        assert!(matches!(
            restarted.ingest_durable(&frame, now_ms * 1_000),
            Err(RadioRuntimeError::Ingress(RadioIngressError::Replay))
        ));
    }

    #[test]
    fn creates_private_exclusively_locked_audit_file() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("radio-audit.jsonl");
        let protected = ProtectedFiles::default();
        let first = open_export_audit(&path, &protected).expect("create audit");
        assert!(path.is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                first.metadata().expect("metadata").permissions().mode() & 0o077,
                0
            );
        }
        assert!(matches!(
            open_export_audit(&path, &protected),
            Err(RadioRuntimeError::AuditLocked)
        ));
    }

    #[test]
    fn rejects_audit_paths_aliasing_every_runtime_protected_file() {
        for protected_name in ["gateway", "host", "replay", "lock", "companion"] {
            let temp = TempDir::new().expect("tempdir");
            let mut runtime_options = options(&temp);
            let companion_path = temp.path().join("companion.bin");
            write_secret(&companion_path, &CS_SECRET);
            runtime_options.channel_sounding = Some(ChannelSoundingRuntimeOptions {
                key_id: 9,
                source_id: 99,
                secret_file: companion_path.clone(),
            });
            let replay_path = runtime_options.replay_state_file.clone();
            let gateway_path = runtime_options.gateways[0].secret_file.clone();
            let host_path = runtime_options.host_pseudonym_secret_file.clone();
            let runtime =
                RadioIngressRuntime::open(runtime_options, 1_900_000_000_000).expect("runtime");
            let audit_path = match protected_name {
                "gateway" => gateway_path,
                "host" => host_path,
                "replay" => replay_path.clone(),
                "lock" => {
                    let mut name = OsString::from(replay_path.as_os_str());
                    name.push(".lock");
                    PathBuf::from(name)
                }
                "companion" => companion_path,
                _ => unreachable!(),
            };
            let (broadcast, _) = tokio::sync::broadcast::channel(1);
            assert!(
                matches!(
                    spawn_radio_ingress_worker(
                        runtime,
                        RadioExportPolicy {
                            allow_biological_p4: false,
                            allow_identity_p5: true,
                        },
                        Some(audit_path),
                        broadcast,
                    ),
                    Err(RadioRuntimeError::AuditAliasesProtectedFile)
                ),
                "protected_name={protected_name}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_hard_linked_audit_alias_by_opened_file_identity() {
        let temp = TempDir::new().expect("tempdir");
        let runtime_options = options(&temp);
        let secret_path = runtime_options.gateways[0].secret_file.clone();
        let audit_path = temp.path().join("radio-audit.jsonl");
        fs::hard_link(&secret_path, &audit_path).expect("hard link");
        let runtime =
            RadioIngressRuntime::open(runtime_options, 1_900_000_000_000).expect("runtime");
        let (broadcast, _) = tokio::sync::broadcast::channel(1);
        assert!(matches!(
            spawn_radio_ingress_worker(
                runtime,
                RadioExportPolicy {
                    allow_biological_p4: false,
                    allow_identity_p5: true,
                },
                Some(audit_path),
                broadcast,
            ),
            Err(RadioRuntimeError::AuditAliasesProtectedFile)
        ));
    }

    #[test]
    fn malformed_audit_tail_fails_closed() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("radio-audit.jsonl");
        fs::write(&path, b"{\"incomplete\":true}").expect("write malformed audit");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("chmod audit");
        }
        assert!(matches!(
            open_export_audit(&path, &ProtectedFiles::default()),
            Err(RadioRuntimeError::AuditMalformed)
        ));
    }

    #[test]
    fn audit_write_failure_stops_worker_before_publication() {
        let temp = TempDir::new().expect("tempdir");
        let now_ms = chrono::Utc::now().timestamp_millis();
        let runtime = RadioIngressRuntime::open(options(&temp), now_ms).expect("runtime");
        let audit_path = temp.path().join("read-only-audit.jsonl");
        fs::write(&audit_path, b"").expect("audit file");
        let read_only_audit = File::open(&audit_path).expect("read-only audit descriptor");
        let (broadcast, mut published) = tokio::sync::broadcast::channel(1);
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(RadioDatagram {
                frame: ble_envelope(1, 1, (now_ms / 60_000) as u32),
                host_received_at_unix_us: now_ms * 1_000,
                source: "127.0.0.1:5005".parse().expect("socket"),
            })
            .expect("queue");
        drop(sender);

        assert!(matches!(
            run_radio_worker(
                runtime,
                RadioExportPolicy {
                    allow_biological_p4: false,
                    allow_identity_p5: true,
                },
                Some(read_only_audit),
                broadcast,
                receiver,
            ),
            Err(RadioRuntimeError::Storage(_))
        ));
        assert!(matches!(
            published.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Closed)
        ));
    }

    #[test]
    fn rejects_zero_and_reused_secrets() {
        let temp = TempDir::new().expect("tempdir");
        let zero_options = options(&temp);
        write_secret(&zero_options.gateways[0].secret_file, &[0; 32]);
        assert!(matches!(
            RadioIngressRuntime::open(zero_options, 1),
            Err(RadioRuntimeError::ZeroSecret)
        ));

        let temp = TempDir::new().expect("tempdir");
        let mut reused_options = options(&temp);
        let companion = temp.path().join("companion.bin");
        write_secret(&companion, &GW_SECRET);
        reused_options.channel_sounding = Some(ChannelSoundingRuntimeOptions {
            key_id: 4,
            source_id: 99,
            secret_file: companion,
        });
        assert!(matches!(
            RadioIngressRuntime::open(reused_options, 1),
            Err(RadioRuntimeError::ReusedSecret)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_world_readable_secret() {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new().expect("tempdir");
        let options = options(&temp);
        fs::set_permissions(
            &options.gateways[0].secret_file,
            fs::Permissions::from_mode(0o644),
        )
        .expect("chmod");
        assert!(matches!(
            RadioIngressRuntime::open(options, 1),
            Err(RadioRuntimeError::SecretPermissions)
        ));
    }
}
