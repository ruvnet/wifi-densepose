//! Durable controller pairing records.
//!
//! This module persists only long-term controller identities and Ed25519 public
//! keys. It does not implement HAP Pair-Setup or Pair-Verify. Those protocol
//! phases must populate this store only after their cryptographic transcript
//! has been authenticated.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use tempfile::Builder;

use crate::error::HapError;

const STORE_VERSION: u32 = 1;
const MAX_STORE_BYTES: u64 = 1024 * 1024;
const MAX_CONTROLLER_ID_BYTES: usize = 64;

/// A controller authorized by a completed HAP pairing ceremony.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerPairing {
    /// HAP controller pairing identifier.
    pub controller_id: String,
    /// Controller Ed25519 long-term public key.
    pub public_key: [u8; 32],
    /// Whether this controller may manage other pairings.
    pub admin: bool,
}

impl ControllerPairing {
    /// Validate bounded identifiers and the encoded Ed25519 point.
    pub fn validate(&self) -> Result<(), HapError> {
        let len = self.controller_id.len();
        if len == 0
            || len > MAX_CONTROLLER_ID_BYTES
            || self.controller_id.chars().any(char::is_control)
        {
            return Err(HapError::InvalidPairingRecord(
                "controller_id must contain 1..=64 bytes and no control characters".into(),
            ));
        }
        VerifyingKey::from_bytes(&self.public_key).map_err(|_| {
            HapError::InvalidPairingRecord("controller public key is not valid Ed25519".into())
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairingFile {
    version: u32,
    controllers: Vec<ControllerPairing>,
}

/// Thread-safe, atomically persisted controller pairing store.
#[derive(Debug)]
pub struct PairingStore {
    path: PathBuf,
    controllers: RwLock<BTreeMap<String, ControllerPairing>>,
}

impl PairingStore {
    /// Open an existing store or create an empty in-memory store.
    ///
    /// Existing files with group/other permission bits on Unix are rejected
    /// instead of silently accepting exposed controller keys.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, HapError> {
        let path = path.into();
        let controllers = if path.exists() {
            load_file(&path)?
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            path,
            controllers: RwLock::new(controllers),
        })
    }

    /// Path used for durable records.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return a deterministic snapshot ordered by controller identifier.
    pub fn list(&self) -> Result<Vec<ControllerPairing>, HapError> {
        Ok(self
            .controllers
            .read()
            .map_err(|_| HapError::PairingStore("pairing store lock poisoned".into()))?
            .values()
            .cloned()
            .collect())
    }

    /// Look up one controller.
    pub fn get(&self, controller_id: &str) -> Result<Option<ControllerPairing>, HapError> {
        Ok(self
            .controllers
            .read()
            .map_err(|_| HapError::PairingStore("pairing store lock poisoned".into()))?
            .get(controller_id)
            .cloned())
    }

    /// Whether at least one controller has completed pairing.
    pub fn is_paired(&self) -> Result<bool, HapError> {
        Ok(!self
            .controllers
            .read()
            .map_err(|_| HapError::PairingStore("pairing store lock poisoned".into()))?
            .is_empty())
    }

    /// Add a controller and durably commit it before exposing it in memory.
    pub fn add(&self, pairing: ControllerPairing) -> Result<(), HapError> {
        pairing.validate()?;
        let mut guard = self
            .controllers
            .write()
            .map_err(|_| HapError::PairingStore("pairing store lock poisoned".into()))?;
        if guard.contains_key(&pairing.controller_id) {
            return Err(HapError::PairingAlreadyExists(pairing.controller_id));
        }
        let mut next = guard.clone();
        next.insert(pairing.controller_id.clone(), pairing);
        persist_file(&self.path, &next)?;
        *guard = next;
        Ok(())
    }

    /// Remove a controller, refusing to orphan remaining non-admin pairings.
    pub fn remove(&self, controller_id: &str) -> Result<(), HapError> {
        let mut guard = self
            .controllers
            .write()
            .map_err(|_| HapError::PairingStore("pairing store lock poisoned".into()))?;
        if !guard.contains_key(controller_id) {
            return Err(HapError::PairingNotFound(controller_id.to_owned()));
        }
        let mut next = guard.clone();
        next.remove(controller_id);
        if !next.is_empty() && !next.values().any(|pairing| pairing.admin) {
            return Err(HapError::InvalidPairingRecord(
                "cannot remove the last administrator while pairings remain".into(),
            ));
        }
        persist_file(&self.path, &next)?;
        *guard = next;
        Ok(())
    }
}

fn load_file(path: &Path) -> Result<BTreeMap<String, ControllerPairing>, HapError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| HapError::PairingStore(format!("metadata {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(HapError::PairingStore(format!(
            "{} must be a regular, non-symlink file",
            path.display()
        )));
    }
    if metadata.len() > MAX_STORE_BYTES {
        return Err(HapError::PairingStore(format!(
            "{} exceeds the {MAX_STORE_BYTES}-byte limit",
            path.display()
        )));
    }
    validate_permissions(path, &metadata)?;

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|file| file.take(MAX_STORE_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| HapError::PairingStore(format!("read {}: {error}", path.display())))?;
    if bytes.len() as u64 > MAX_STORE_BYTES {
        return Err(HapError::PairingStore(
            "pairing store exceeds size limit".into(),
        ));
    }
    let file: PairingFile = serde_json::from_slice(&bytes)
        .map_err(|error| HapError::PairingStore(format!("parse {}: {error}", path.display())))?;
    if file.version != STORE_VERSION {
        return Err(HapError::PairingStore(format!(
            "unsupported pairing store version {}",
            file.version
        )));
    }

    let mut controllers = BTreeMap::new();
    for pairing in file.controllers {
        pairing.validate()?;
        let id = pairing.controller_id.clone();
        if controllers.insert(id.clone(), pairing).is_some() {
            return Err(HapError::InvalidPairingRecord(format!(
                "duplicate controller_id {id}"
            )));
        }
    }
    if !controllers.is_empty() && !controllers.values().any(|pairing| pairing.admin) {
        return Err(HapError::InvalidPairingRecord(
            "persisted pairings have no administrator".into(),
        ));
    }
    Ok(controllers)
}

fn persist_file(
    path: &Path,
    controllers: &BTreeMap<String, ControllerPairing>,
) -> Result<(), HapError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    create_private_dir(parent)?;
    let payload = serde_json::to_vec_pretty(&PairingFile {
        version: STORE_VERSION,
        controllers: controllers.values().cloned().collect(),
    })
    .map_err(|error| HapError::PairingStore(format!("serialize pairings: {error}")))?;

    let mut temp = Builder::new()
        .prefix(".homecore-hap-pairings-")
        .tempfile_in(parent)
        .map_err(|error| HapError::PairingStore(format!("create temporary store: {error}")))?;
    set_private_file_permissions(temp.as_file())?;
    temp.write_all(&payload)
        .and_then(|_| temp.flush())
        .and_then(|_| temp.as_file().sync_all())
        .map_err(|error| HapError::PairingStore(format!("write temporary store: {error}")))?;
    temp.persist(path).map_err(|error| {
        HapError::PairingStore(format!("replace {}: {}", path.display(), error.error))
    })?;

    #[cfg(unix)]
    {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                HapError::PairingStore(format!("sync {}: {error}", parent.display()))
            })?;
    }
    Ok(())
}

fn create_private_dir(path: &Path) -> Result<(), HapError> {
    let created = !path.exists();
    if created {
        fs::create_dir_all(path).map_err(|error| {
            HapError::PairingStore(format!("create {}: {error}", path.display()))
        })?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if created {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
                HapError::PairingStore(format!("chmod {}: {error}", path.display()))
            })?;
        }
    }
    Ok(())
}

fn set_private_file_permissions(_file: &File) -> Result<(), HapError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        _file
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| HapError::PairingStore(format!("chmod temporary store: {error}")))?;
    }
    Ok(())
}

fn validate_permissions(path: &Path, metadata: &fs::Metadata) -> Result<(), HapError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        if mode & 0o077 != 0 {
            return Err(HapError::InsecurePermissions {
                path: path.to_path_buf(),
                mode: mode & 0o777,
            });
        }
    }
    let _ = (path, metadata);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn pairing(id: &str, byte: u8, admin: bool) -> ControllerPairing {
        let public_key = SigningKey::from_bytes(&[byte; 32])
            .verifying_key()
            .to_bytes();
        ControllerPairing {
            controller_id: id.into(),
            public_key,
            admin,
        }
    }

    #[test]
    fn restart_loads_atomically_persisted_pairings() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pairings.json");
        let store = PairingStore::open(&path).unwrap();
        store.add(pairing("controller-1", 7, true)).unwrap();
        drop(store);

        let reopened = PairingStore::open(&path).unwrap();
        assert_eq!(
            reopened.list().unwrap(),
            vec![pairing("controller-1", 7, true)]
        );
    }

    #[test]
    fn malformed_and_duplicate_records_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pairings.json");
        fs::write(&path, br#"{"version":1,"controllers":[{"controller_id":"","public_key":[0,1],"admin":true}]}"#).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(PairingStore::open(path).is_err());
    }

    #[test]
    fn cannot_remove_last_admin_while_members_remain() {
        let directory = tempfile::tempdir().unwrap();
        let store = PairingStore::open(directory.path().join("pairings.json")).unwrap();
        store.add(pairing("admin", 1, true)).unwrap();
        store.add(pairing("member", 2, false)).unwrap();
        assert!(store.remove("admin").is_err());
        assert_eq!(store.list().unwrap().len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn permissive_existing_file_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pairings.json");
        fs::write(&path, br#"{"version":1,"controllers":[]}"#).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            PairingStore::open(path),
            Err(HapError::InsecurePermissions { .. })
        ));
    }
}
