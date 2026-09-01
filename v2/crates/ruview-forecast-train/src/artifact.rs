//! Hash-addressed dataset and artifact file handling.
//!
//! Security-sensitive opens are relative to an already-open directory and use
//! `O_NOFOLLOW`. The checked object is therefore the object consumed by the
//! trainer; there is no check-then-reopen window.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
#[cfg(all(test, feature = "cpu"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rustix::fs::{self, AtFlags, Mode, OFlags};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::{DatasetInput, JobId, RelativeDataPath, Sha256Digest};

/// Maximum size of any emitted model record.
pub const MAX_MODEL_ARTIFACT_BYTES: u64 = ruview_forecast_model::MAX_ARTIFACT_BYTES as u64;
/// Maximum size of a manifest or receipt.
pub const MAX_METADATA_ARTIFACT_BYTES: u64 = 1024 * 1024;
/// Fixed provider-side prefix. Descriptors never carry arbitrary filesystem
/// paths; the fal client combines this trusted prefix with an [`ArtifactId`].
pub const FAL_ARTIFACT_ROOT: &str = "/data/ruview-forecast/artifacts";

/// I/O and integrity failures at the dataset/artifact boundary.
#[derive(Debug, Error)]
pub enum ArtifactError {
    /// A filesystem operation failed.
    #[error("artifact I/O failed: {0}")]
    Io(#[from] io::Error),
    /// A file was not a regular file or had additional hard links.
    #[error("expected a regular, single-link file")]
    UnsafeFileType,
    /// A file length did not match its declaration.
    #[error("size mismatch: expected {expected} bytes, received {actual} bytes")]
    SizeMismatch {
        /// Declared byte count.
        expected: u64,
        /// Observed byte count.
        actual: u64,
    },
    /// A file digest did not match its declaration.
    #[error("SHA-256 mismatch: expected {expected}, received {actual}")]
    DigestMismatch {
        /// Declared digest.
        expected: Sha256Digest,
        /// Observed digest.
        actual: Sha256Digest,
    },
    /// The same idempotency key was reused for different artifact bytes.
    #[error("artifact conflict for existing job output")]
    Conflict,
    /// An artifact exceeded its kind-specific cap.
    #[error("{kind:?} artifact exceeds its {maximum} byte cap")]
    ArtifactTooLarge {
        /// Artifact kind.
        kind: ArtifactKind,
        /// Maximum accepted size.
        maximum: u64,
    },
}

/// The fixed set of files a training run may publish.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Burn NamedMpk model record.
    Model,
    /// Canonical model artifact manifest.
    Manifest,
    /// Training inputs, metrics, and environment receipt.
    Receipt,
    /// Latest cooperative-cancellation checkpoint.
    Checkpoint,
}

impl ArtifactKind {
    /// Fixed filename for the kind.
    pub fn filename(self) -> &'static str {
        match self {
            Self::Model => "model.mpk",
            Self::Manifest => "artifact-manifest.json",
            Self::Receipt => "training-receipt.json",
            Self::Checkpoint => "checkpoint.mpk",
        }
    }

    fn maximum_bytes(self) -> u64 {
        match self {
            Self::Model | Self::Checkpoint => MAX_MODEL_ARTIFACT_BYTES,
            Self::Manifest | Self::Receipt => MAX_METADATA_ARTIFACT_BYTES,
        }
    }
}

/// Opaque provider artifact identity.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactId {
    /// Validated job namespace.
    pub job_id: JobId,
    /// Fixed artifact kind.
    pub kind: ArtifactKind,
}

impl ArtifactId {
    /// Returns a provider-relative path assembled only from validated values.
    pub fn relative_path(&self) -> String {
        format!("{}/{}", self.job_id.as_str(), self.kind.filename())
    }
}

/// Immutable artifact identity returned by local and hosted runs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDescriptor {
    /// Opaque, root-relative object identity.
    pub id: ArtifactId,
    /// Exact byte count.
    pub size_bytes: u64,
    /// Exact SHA-256 digest.
    pub sha256: Sha256Digest,
}

impl ArtifactDescriptor {
    /// Validates descriptor bounds independently of a remote response.
    pub fn validate(&self) -> Result<(), ArtifactError> {
        let maximum = self.id.kind.maximum_bytes();
        if self.size_bytes == 0 || self.size_bytes > maximum {
            return Err(ArtifactError::ArtifactTooLarge {
                kind: self.id.kind,
                maximum,
            });
        }
        Ok(())
    }
}

/// A verified dataset held open against path replacement.
#[derive(Debug)]
pub struct VerifiedDataset {
    logical_path: RelativeDataPath,
    file: File,
    expected_size_bytes: u64,
    expected_sha256: Sha256Digest,
}

impl VerifiedDataset {
    /// Returns the non-sensitive, root-relative identifier for receipts.
    pub fn logical_path(&self) -> &RelativeDataPath {
        &self.logical_path
    }

    /// Returns the already-verified handle. Consumers deserialize from this
    /// handle rather than reopening a path.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Returns the byte count that every epoch must independently observe.
    pub const fn expected_size_bytes(&self) -> u64 {
        self.expected_size_bytes
    }

    /// Returns the digest that every epoch must independently reproduce.
    pub const fn expected_sha256(&self) -> Sha256Digest {
        self.expected_sha256
    }
}

/// Resolves and verifies a hash-addressed dataset beneath `root` using
/// handle-relative, no-follow opens for every component.
pub fn open_verified_dataset(
    root: &Path,
    input: &DatasetInput,
) -> Result<VerifiedDataset, ArtifactError> {
    input
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let root = open_directory(root)?;
    let mut file = open_relative_file(&root, &input.path)?;
    verify_regular_single_link(&file)?;
    let actual_size = file.metadata()?.len();
    if actual_size != input.size_bytes {
        return Err(ArtifactError::SizeMismatch {
            expected: input.size_bytes,
            actual: actual_size,
        });
    }
    let actual = digest_reader(&mut file, input.size_bytes)?;
    if actual != input.sha256 {
        return Err(ArtifactError::DigestMismatch {
            expected: input.sha256,
            actual,
        });
    }
    file.rewind()?;
    Ok(VerifiedDataset {
        logical_path: input.path.clone(),
        file,
        expected_size_bytes: input.size_bytes,
        expected_sha256: input.sha256,
    })
}

/// A root-confined, atomic artifact publisher.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root_path: PathBuf,
    root: Arc<File>,
    #[cfg(all(test, feature = "cpu"))]
    commits_before_failure: Arc<AtomicUsize>,
}

/// Exclusive per-job execution lease. Closing the private file releases the
/// operating-system lock, including during unwinding.
pub struct JobLease {
    _lock: File,
}

impl ArtifactStore {
    /// Creates or opens an artifact root with owner-only permissions.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        std::fs::create_dir_all(root.as_ref())?;
        let root_file = open_directory(root.as_ref())?;
        fs::fchmod(&root_file, Mode::RUSR | Mode::WUSR | Mode::XUSR).map_err(io::Error::from)?;
        let root_path = root.as_ref().canonicalize()?;
        Ok(Self {
            root_path,
            root: Arc::new(root_file),
            #[cfg(all(test, feature = "cpu"))]
            commits_before_failure: Arc::new(AtomicUsize::new(usize::MAX)),
        })
    }

    /// Returns the canonical artifact root for local operator output only.
    pub fn root(&self) -> &Path {
        &self.root_path
    }

    /// Returns a local path assembled from a validated descriptor.
    pub fn local_path(&self, descriptor: &ArtifactDescriptor) -> PathBuf {
        self.root_path
            .join(descriptor.id.job_id.as_str())
            .join(descriptor.id.kind.filename())
    }

    /// Serializes expensive training and restart recovery for one job across
    /// processes sharing this artifact root.
    pub fn lock_job(&self, job_id: &JobId) -> Result<JobLease, ArtifactError> {
        let job_dir = open_or_create_job_dir(&self.root, job_id)?;
        let lock = open_named_lock_file(&job_dir, ".run.lock")?;
        fs::flock(&lock, fs::FlockOperation::LockExclusive).map_err(io::Error::from)?;
        Ok(JobLease { _lock: lock })
    }

    /// Atomically publishes fixed-kind bytes. Repeating identical bytes is
    /// idempotent; reusing the id for different bytes fails closed.
    pub fn commit_bytes(
        &self,
        job_id: &JobId,
        kind: ArtifactKind,
        bytes: &[u8],
    ) -> Result<ArtifactDescriptor, ArtifactError> {
        #[cfg(all(test, feature = "cpu"))]
        self.maybe_fail_commit()?;
        let maximum = kind.maximum_bytes();
        if bytes.is_empty() || bytes.len() as u64 > maximum {
            return Err(ArtifactError::ArtifactTooLarge { kind, maximum });
        }
        let size_bytes = bytes.len() as u64;
        let digest = Sha256Digest::of_bytes(bytes);
        let job_dir = open_or_create_job_dir(&self.root, job_id)?;
        let final_name = kind.filename();
        let lock = open_lock_file(&job_dir)?;
        fs::flock(&lock, fs::FlockOperation::LockExclusive).map_err(io::Error::from)?;

        match open_file_at(&job_dir, final_name, OFlags::RDONLY) {
            Ok(mut existing) => {
                return verify_existing(job_id, kind, &mut existing, size_bytes, digest)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        // A fixed temporary name bounds crash residue to one file per kind.
        // The exclusive per-job lock proves no live writer owns it.
        let temporary_name = format!(".{final_name}.partial");
        match fs::unlinkat(&job_dir, temporary_name.as_str(), AtFlags::empty()) {
            Ok(()) => {}
            Err(error) if error_kind(error) == io::ErrorKind::NotFound => {}
            Err(error) => return Err(io::Error::from(error).into()),
        }
        let temporary_fd = fs::openat(
            &job_dir,
            temporary_name.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(io::Error::from)?;
        let mut temporary = File::from(temporary_fd);
        let write_result = (|| -> io::Result<()> {
            temporary.write_all(bytes)?;
            temporary.sync_all()
        })();
        drop(temporary);
        if let Err(error) = write_result {
            let _ = fs::unlinkat(&job_dir, temporary_name.as_str(), AtFlags::empty());
            return Err(error.into());
        }

        let link_result = fs::linkat(
            &job_dir,
            temporary_name.as_str(),
            &job_dir,
            final_name,
            AtFlags::empty(),
        );
        let _ = fs::unlinkat(&job_dir, temporary_name.as_str(), AtFlags::empty());
        match link_result {
            Ok(()) => {
                job_dir.sync_all()?;
                Ok(descriptor(job_id, kind, size_bytes, digest))
            }
            Err(error) if error_kind(error) == io::ErrorKind::AlreadyExists => {
                let mut existing = open_file_at(&job_dir, final_name, OFlags::RDONLY)?;
                verify_existing(job_id, kind, &mut existing, size_bytes, digest)
            }
            Err(error) => Err(io::Error::from(error).into()),
        }
    }

    /// Re-hashes a descriptor using a no-follow open relative to this store.
    pub fn verify(&self, descriptor: &ArtifactDescriptor) -> Result<(), ArtifactError> {
        descriptor.validate()?;
        let job_dir = open_job_dir(&self.root, &descriptor.id.job_id)?;
        let mut file = open_file_at(&job_dir, descriptor.id.kind.filename(), OFlags::RDONLY)?;
        verify_existing(
            &descriptor.id.job_id,
            descriptor.id.kind,
            &mut file,
            descriptor.size_bytes,
            descriptor.sha256,
        )?;
        Ok(())
    }

    /// Returns the current verified descriptor for a fixed job/kind, if it
    /// exists. Unsafe directories or files fail closed instead of appearing
    /// absent.
    pub fn existing_descriptor(
        &self,
        job_id: &JobId,
        kind: ArtifactKind,
    ) -> Result<Option<ArtifactDescriptor>, ArtifactError> {
        let job_dir = match open_job_dir(&self.root, job_id) {
            Ok(directory) => directory,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut file = match open_file_at(&job_dir, kind.filename(), OFlags::RDONLY) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        verify_regular_single_link(&file)?;
        let size_bytes = file.metadata()?.len();
        let maximum = kind.maximum_bytes();
        if size_bytes == 0 || size_bytes > maximum {
            return Err(ArtifactError::ArtifactTooLarge { kind, maximum });
        }
        let sha256 = digest_reader(&mut file, maximum)?;
        Ok(Some(descriptor(job_id, kind, size_bytes, sha256)))
    }

    /// Reads exact verified bytes through the same confined no-follow
    /// capability used by [`Self::verify`].
    pub fn read_bytes(&self, descriptor: &ArtifactDescriptor) -> Result<Vec<u8>, ArtifactError> {
        descriptor.validate()?;
        let job_dir = open_job_dir(&self.root, &descriptor.id.job_id)?;
        let mut file = open_file_at(&job_dir, descriptor.id.kind.filename(), OFlags::RDONLY)?;
        verify_regular_single_link(&file)?;
        let actual_size = file.metadata()?.len();
        if actual_size != descriptor.size_bytes {
            return Err(ArtifactError::SizeMismatch {
                expected: descriptor.size_bytes,
                actual: actual_size,
            });
        }
        let capacity = usize::try_from(descriptor.size_bytes).map_err(|_| {
            ArtifactError::ArtifactTooLarge {
                kind: descriptor.id.kind,
                maximum: descriptor.id.kind.maximum_bytes(),
            }
        })?;
        let mut bytes = Vec::with_capacity(capacity);
        Read::by_ref(&mut file)
            .take(descriptor.size_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() != capacity {
            return Err(ArtifactError::SizeMismatch {
                expected: descriptor.size_bytes,
                actual: bytes.len() as u64,
            });
        }
        let actual_digest = Sha256Digest::of_bytes(&bytes);
        if actual_digest != descriptor.sha256 {
            return Err(ArtifactError::DigestMismatch {
                expected: descriptor.sha256,
                actual: actual_digest,
            });
        }
        Ok(bytes)
    }

    /// Removes only the fixed candidate outputs for one validated job.
    ///
    /// This is used to roll back a publication that crossed a wall-time or
    /// retention boundary between individual durable writes. Lock and job
    /// directories remain so concurrent or later executions keep the same
    /// serialization boundary.
    #[cfg(any(feature = "training", feature = "fal-client", test))]
    pub(crate) fn remove_job_outputs(&self, job_id: &JobId) -> Result<(), ArtifactError> {
        let job_dir = open_job_dir(&self.root, job_id)?;
        let lock = open_lock_file(&job_dir)?;
        fs::flock(&lock, fs::FlockOperation::LockExclusive).map_err(io::Error::from)?;
        for kind in [
            ArtifactKind::Model,
            ArtifactKind::Manifest,
            ArtifactKind::Receipt,
            ArtifactKind::Checkpoint,
        ] {
            unlink_if_present(&job_dir, kind.filename())?;
            unlink_if_present(&job_dir, &format!(".{}.partial", kind.filename()))?;
        }
        job_dir.sync_all()?;
        Ok(())
    }

    #[cfg(all(test, feature = "cpu"))]
    pub(crate) fn fail_after_successful_commits(&self, successful_commits: usize) {
        self.commits_before_failure
            .store(successful_commits, Ordering::SeqCst);
    }

    #[cfg(all(test, feature = "cpu"))]
    fn maybe_fail_commit(&self) -> Result<(), ArtifactError> {
        loop {
            let remaining = self.commits_before_failure.load(Ordering::SeqCst);
            if remaining == usize::MAX {
                return Ok(());
            }
            if remaining == 0 {
                return Err(io::Error::other("injected artifact commit failure").into());
            }
            if self
                .commits_before_failure
                .compare_exchange(remaining, remaining - 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(());
            }
        }
    }
}

#[cfg(any(feature = "training", feature = "fal-client", test))]
fn unlink_if_present(directory: &File, name: &str) -> Result<(), ArtifactError> {
    match fs::unlinkat(directory, name, AtFlags::empty()) {
        Ok(()) => Ok(()),
        Err(error) if error_kind(error) == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io::Error::from(error).into()),
    }
}

fn descriptor(
    job_id: &JobId,
    kind: ArtifactKind,
    size_bytes: u64,
    sha256: Sha256Digest,
) -> ArtifactDescriptor {
    ArtifactDescriptor {
        id: ArtifactId {
            job_id: job_id.clone(),
            kind,
        },
        size_bytes,
        sha256,
    }
}

fn verify_existing(
    job_id: &JobId,
    kind: ArtifactKind,
    file: &mut File,
    expected_size: u64,
    expected_digest: Sha256Digest,
) -> Result<ArtifactDescriptor, ArtifactError> {
    verify_regular_single_link(file)?;
    if file.metadata()?.len() != expected_size {
        return Err(ArtifactError::Conflict);
    }
    if digest_reader(file, expected_size)? != expected_digest {
        return Err(ArtifactError::Conflict);
    }
    Ok(descriptor(job_id, kind, expected_size, expected_digest))
}

fn open_directory(path: &Path) -> io::Result<File> {
    let fd = fs::open(
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(fd))
}

fn open_relative_file(root: &File, path: &RelativeDataPath) -> io::Result<File> {
    let segments: Vec<&str> = path.as_str().split('/').collect();
    let (filename, directories) = segments
        .split_last()
        .expect("RelativeDataPath is never empty");
    let mut directory = root.try_clone()?;
    for segment in directories {
        let fd = fs::openat(
            &directory,
            *segment,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(io::Error::from)?;
        directory = File::from(fd);
    }
    open_file_at(&directory, filename, OFlags::RDONLY)
}

fn open_or_create_job_dir(root: &File, job_id: &JobId) -> io::Result<File> {
    match fs::mkdirat(root, job_id.as_str(), Mode::RUSR | Mode::WUSR | Mode::XUSR) {
        Ok(()) => {}
        Err(error) if error_kind(error) == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io::Error::from(error)),
    }
    let directory = open_job_dir(root, job_id)?;
    fs::fchmod(&directory, Mode::RUSR | Mode::WUSR | Mode::XUSR).map_err(io::Error::from)?;
    Ok(directory)
}

fn open_job_dir(root: &File, job_id: &JobId) -> io::Result<File> {
    let fd = fs::openat(
        root,
        job_id.as_str(),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(fd))
}

fn open_file_at(directory: &File, name: &str, access: OFlags) -> io::Result<File> {
    let fd = fs::openat(
        directory,
        name,
        access | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(fd))
}

fn open_lock_file(directory: &File) -> io::Result<File> {
    open_named_lock_file(directory, ".job.lock")
}

fn open_named_lock_file(directory: &File, name: &str) -> io::Result<File> {
    let fd = fs::openat(
        directory,
        name,
        OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    let file = File::from(fd);
    verify_regular_single_link(&file)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(file)
}

fn verify_regular_single_link(file: &File) -> Result<(), ArtifactError> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(ArtifactError::UnsafeFileType);
    }
    #[cfg(unix)]
    if metadata.nlink() != 1 {
        return Err(ArtifactError::UnsafeFileType);
    }
    Ok(())
}

fn error_kind(error: rustix::io::Errno) -> io::ErrorKind {
    io::Error::from_raw_os_error(error.raw_os_error()).kind()
}

fn digest_reader(file: &mut File, maximum: u64) -> Result<Sha256Digest, ArtifactError> {
    file.rewind()?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or(ArtifactError::SizeMismatch {
                expected: maximum,
                actual: u64::MAX,
            })?;
        if total > maximum {
            return Err(ArtifactError::SizeMismatch {
                expected: maximum,
                actual: total,
            });
        }
        hasher.update(&buffer[..read]);
    }
    Sha256Digest::from_hex(&hex::encode(hasher.finalize()))
        .map_err(|_| ArtifactError::UnsafeFileType)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_open_verifies_size_and_digest() {
        let root = tempfile::tempdir().expect("temp root");
        let bytes = b"governed training bytes";
        std::fs::write(root.path().join("corpus.json"), bytes).expect("fixture");
        let input = DatasetInput {
            path: RelativeDataPath::new("corpus.json").expect("path"),
            size_bytes: bytes.len() as u64,
            sha256: Sha256Digest::of_bytes(bytes),
            window_count: 1,
            variates: 1,
            feature_schema_digest: ruview_forecast_core::CanonicalDigest::of_bytes(
                b"test-feature-schema-v1",
                b"value",
            ),
        };
        let verified = open_verified_dataset(root.path(), &input).expect("verified dataset");
        assert_eq!(verified.logical_path().as_str(), "corpus.json");

        let mut wrong = input.clone();
        wrong.sha256 = Sha256Digest::of_bytes(b"other");
        assert!(matches!(
            open_verified_dataset(root.path(), &wrong),
            Err(ArtifactError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn artifact_commit_is_atomic_and_idempotent() {
        let root = tempfile::tempdir().expect("temp root");
        let store = ArtifactStore::new(root.path()).expect("store");
        let job = JobId::new("job-1").expect("job id");
        let first = store
            .commit_bytes(&job, ArtifactKind::Model, b"model")
            .expect("first commit");
        let second = store
            .commit_bytes(&job, ArtifactKind::Model, b"model")
            .expect("idempotent commit");
        assert_eq!(first, second);
        store.verify(&first).expect("verify committed artifact");
        assert!(matches!(
            store.commit_bytes(&job, ArtifactKind::Model, b"different"),
            Err(ArtifactError::Conflict)
        ));

        let recovered = store
            .existing_descriptor(&job, ArtifactKind::Model)
            .expect("lookup")
            .expect("existing model");
        assert_eq!(recovered, first);
        assert_eq!(store.read_bytes(&recovered).expect("read"), b"model");
        assert!(store
            .existing_descriptor(&job, ArtifactKind::Receipt)
            .expect("missing lookup")
            .is_none());

        store
            .remove_job_outputs(&job)
            .expect("rollback fixed outputs");
        assert!(store
            .existing_descriptor(&job, ArtifactKind::Model)
            .expect("rolled-back lookup")
            .is_none());
        store
            .commit_bytes(&job, ArtifactKind::Model, b"replacement")
            .expect("job remains reusable after rollback");
    }

    #[cfg(unix)]
    #[test]
    fn dataset_symlink_and_hardlink_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        let outside_file = outside.path().join("source");
        std::fs::write(&outside_file, b"not training data").expect("fixture");
        symlink(&outside_file, root.path().join("symlink")).expect("symlink");
        std::fs::hard_link(&outside_file, root.path().join("hardlink")).expect("hardlink");
        for name in ["symlink", "hardlink"] {
            let input = DatasetInput {
                path: RelativeDataPath::new(name).expect("path"),
                size_bytes: 17,
                sha256: Sha256Digest::of_bytes(b"not training data"),
                window_count: 1,
                variates: 1,
                feature_schema_digest: ruview_forecast_core::CanonicalDigest::of_bytes(
                    b"test-feature-schema-v1",
                    b"value",
                ),
            };
            assert!(open_verified_dataset(root.path(), &input).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn preexisting_artifact_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("root");
        let store = ArtifactStore::new(root.path()).expect("store");
        let job = JobId::new("job-2").expect("job id");
        let job_dir = root.path().join(job.as_str());
        std::fs::create_dir(&job_dir).expect("job dir");
        let outside = root.path().join("outside");
        std::fs::write(&outside, b"model").expect("outside");
        symlink(&outside, job_dir.join("model.mpk")).expect("symlink");
        assert!(store
            .commit_bytes(&job, ArtifactKind::Model, b"model")
            .is_err());
    }
}
