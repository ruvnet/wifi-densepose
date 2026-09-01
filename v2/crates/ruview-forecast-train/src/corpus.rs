//! Streaming, bounded JSONL window shards for local Linux training.

use std::io::{self, BufRead, BufReader, Read, Seek};

use ruview_forecast_core::{SeriesKey, TrainSpec};
use ruview_forecast_model::ForecastModelConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    artifact::VerifiedDataset,
    config::{MAX_JSONL_LINE_BYTES, MAX_WINDOW_CELLS},
};

/// Maximum in-memory shuffle window count.
pub const MAX_SHUFFLE_WINDOWS: usize = 64;

/// Bounded corpus decode error.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// Shard I/O failed.
    #[error("JSONL shard I/O failed: {0}")]
    Io(#[from] io::Error),
    /// A line exceeded the cap.
    #[error("JSONL line exceeds {MAX_JSONL_LINE_BYTES} bytes")]
    LineTooLarge,
    /// JSON syntax or unknown fields were invalid.
    #[error("invalid JSONL window")]
    InvalidJson,
    /// Shape, mask, split, or finite-value validation failed.
    #[error("invalid training window: {0}")]
    InvalidWindow(&'static str),
    /// Bytes read during an epoch no longer match the manifest that was
    /// verified before training.
    #[error("JSONL shard changed after initial verification")]
    DatasetChanged,
}

/// One pre-windowed local training example. It contains no arbitrary tensor
/// dimensions: context and horizon are taken from the trusted TrainSpec/model.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonlWindow {
    /// Format version; must be one.
    pub version: u16,
    /// Exact split membership key. This stays on the local Linux box.
    pub series_key: SeriesKey,
    /// Timestamp of the first context row.
    pub context_start_ms: u64,
    /// Number of coupled variates.
    pub variates: u16,
    /// Row-major context values `[context, variates]`.
    pub values: Vec<f32>,
    /// Binary context mask with the same shape.
    pub observed_mask: Vec<u8>,
    /// Row-major future targets `[variates, horizon]`.
    pub targets: Vec<f32>,
    /// Binary target mask.
    pub target_mask: Vec<u8>,
}

impl JsonlWindow {
    /// Validates bounds, numeric values, and exact inclusion in a training
    /// split member before tensors are allocated.
    pub fn validate(
        &self,
        train: &TrainSpec,
        model: &ForecastModelConfig,
    ) -> Result<(), CorpusError> {
        if self.version != 1
            || self.variates == 0
            || usize::from(self.variates) > model.max_variates
        {
            return Err(CorpusError::InvalidWindow("version or variates"));
        }
        let variates = usize::from(self.variates);
        let context_cells = model
            .context_len
            .checked_mul(variates)
            .ok_or(CorpusError::InvalidWindow("context overflow"))?;
        let target_cells = model
            .horizon
            .checked_mul(variates)
            .ok_or(CorpusError::InvalidWindow("target overflow"))?;
        if context_cells
            .checked_add(target_cells)
            .is_none_or(|cells| cells > MAX_WINDOW_CELLS)
            || self.values.len() != context_cells
            || self.observed_mask.len() != context_cells
            || self.targets.len() != target_cells
            || self.target_mask.len() != target_cells
        {
            return Err(CorpusError::InvalidWindow("shape or cell cap"));
        }
        if self
            .values
            .iter()
            .chain(self.targets.iter())
            .any(|v| !v.is_finite())
            || self
                .observed_mask
                .iter()
                .chain(self.target_mask.iter())
                .any(|v| *v > 1)
            || !self.target_mask.contains(&1)
        {
            return Err(CorpusError::InvalidWindow(
                "non-finite value, mask, or empty target",
            ));
        }
        let rows = u64::try_from(model.context_len + model.horizon)
            .map_err(|_| CorpusError::InvalidWindow("row overflow"))?;
        let end = rows
            .checked_mul(train.step_ms())
            .and_then(|span| self.context_start_ms.checked_add(span))
            .ok_or(CorpusError::InvalidWindow("timestamp overflow"))?;
        let allowed = train.split_plan().train().iter().any(|member| {
            member.key() == &self.series_key
                && self.context_start_ms >= member.range().start_ms()
                && end <= member.range().end_ms()
        });
        if !allowed {
            return Err(CorpusError::InvalidWindow(
                "window is outside training partition",
            ));
        }
        Ok(())
    }
}

/// Reader over one already-open and hash-verified shard.
pub struct JsonlWindowReader {
    reader: BufReader<EpochDigestReader>,
    line_number: u64,
    expected_size_bytes: u64,
    expected_sha256: crate::config::Sha256Digest,
    verified_eof: bool,
}

impl JsonlWindowReader {
    /// Clones the verified file capability and rewinds it for one epoch.
    pub fn new(dataset: &VerifiedDataset) -> Result<Self, CorpusError> {
        let mut file = dataset.file().try_clone()?;
        file.rewind()?;
        Ok(Self {
            reader: BufReader::new(EpochDigestReader::new(file)),
            line_number: 0,
            expected_size_bytes: dataset.expected_size_bytes(),
            expected_sha256: dataset.expected_sha256(),
            verified_eof: false,
        })
    }

    /// Reads and validates one bounded line.
    pub fn next_window(
        &mut self,
        train: &TrainSpec,
        model: &ForecastModelConfig,
    ) -> Result<Option<JsonlWindow>, CorpusError> {
        let Some(bytes) = read_line_bounded(&mut self.reader)? else {
            self.verify_epoch_digest()?;
            return Ok(None);
        };
        self.line_number = self.line_number.saturating_add(1);
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Err(CorpusError::InvalidJson);
        }
        let window: JsonlWindow =
            serde_json::from_slice(&bytes).map_err(|_| CorpusError::InvalidJson)?;
        window.validate(train, model)?;
        Ok(Some(window))
    }

    fn verify_epoch_digest(&mut self) -> Result<(), CorpusError> {
        if self.verified_eof {
            return Ok(());
        }
        let digest_reader = self.reader.get_ref();
        let actual_digest: [u8; 32] = digest_reader.hasher.clone().finalize().into();
        if digest_reader.bytes_read != self.expected_size_bytes
            || &actual_digest != self.expected_sha256.as_bytes()
        {
            return Err(CorpusError::DatasetChanged);
        }
        self.verified_eof = true;
        Ok(())
    }
}

struct EpochDigestReader {
    file: std::fs::File,
    hasher: Sha256,
    bytes_read: u64,
}

impl EpochDigestReader {
    fn new(file: std::fs::File) -> Self {
        Self {
            file,
            hasher: Sha256::new(),
            bytes_read: 0,
        }
    }
}

impl Read for EpochDigestReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.file.read(buffer)?;
        let read_u64 = u64::try_from(read)
            .map_err(|_| io::Error::other("JSONL byte count conversion overflow"))?;
        self.bytes_read = self
            .bytes_read
            .checked_add(read_u64)
            .ok_or_else(|| io::Error::other("JSONL byte count overflow"))?;
        self.hasher.update(&buffer[..read]);
        Ok(read)
    }
}

/// Deterministic finite shuffle buffer. It never indexes or loads the entire
/// shard and its memory is bounded by `capacity * MAX_JSONL_LINE_BYTES`.
pub struct ShuffledWindows<'a> {
    reader: JsonlWindowReader,
    train: &'a TrainSpec,
    model: &'a ForecastModelConfig,
    buffer: Vec<JsonlWindow>,
    capacity: usize,
    state: u64,
    exhausted: bool,
}

impl<'a> ShuffledWindows<'a> {
    /// Creates a bounded deterministic stream.
    pub fn new(
        dataset: &VerifiedDataset,
        train: &'a TrainSpec,
        model: &'a ForecastModelConfig,
        capacity: usize,
        seed: u64,
    ) -> Result<Self, CorpusError> {
        if capacity == 0 || capacity > MAX_SHUFFLE_WINDOWS {
            return Err(CorpusError::InvalidWindow("shuffle capacity"));
        }
        Ok(Self {
            reader: JsonlWindowReader::new(dataset)?,
            train,
            model,
            buffer: Vec::with_capacity(capacity),
            capacity,
            state: seed,
            exhausted: false,
        })
    }

    /// Next validated window.
    pub fn next_window(&mut self) -> Result<Option<JsonlWindow>, CorpusError> {
        while !self.exhausted && self.buffer.len() < self.capacity {
            match self.reader.next_window(self.train, self.model)? {
                Some(v) => self.buffer.push(v),
                None => self.exhausted = true,
            }
        }
        if self.buffer.is_empty() {
            return Ok(None);
        }
        self.state = splitmix64(self.state);
        let index = (self.state as usize) % self.buffer.len();
        let output = self.buffer.swap_remove(index);
        if !self.exhausted {
            match self.reader.next_window(self.train, self.model)? {
                Some(v) => self.buffer.push(v),
                None => self.exhausted = true,
            }
        }
        Ok(Some(output))
    }
}

fn read_line_bounded<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>, CorpusError> {
    let mut output = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if output.is_empty() {
                Ok(None)
            } else {
                Ok(Some(output))
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |index| index + 1);
        if output
            .len()
            .checked_add(take)
            .is_none_or(|size| size > MAX_JSONL_LINE_BYTES)
        {
            return Err(CorpusError::LineTooLarge);
        }
        output.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            return Ok(Some(output));
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_reader_rejects_oversized_line() {
        let bytes = vec![b'x'; MAX_JSONL_LINE_BYTES + 1];
        let mut reader = io::Cursor::new(bytes);
        assert!(matches!(
            read_line_bounded(&mut reader),
            Err(CorpusError::LineTooLarge)
        ))
    }

    #[test]
    fn epoch_digest_rejects_in_place_dataset_mutation() {
        let original = b"same-size-original";
        let replacement = b"same-size-mutated!";
        assert_eq!(original.len(), replacement.len());

        let mut temporary = tempfile::NamedTempFile::new().expect("temporary shard");
        std::io::Write::write_all(&mut temporary, original).expect("write original");
        temporary.as_file().sync_all().expect("sync original");
        let read_handle = std::fs::File::open(temporary.path()).expect("open verified inode");

        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temporary.path())
            .expect("open same inode for mutation");
        std::io::Write::write_all(&mut writer, replacement).expect("replace bytes");
        writer.sync_all().expect("sync replacement");

        let mut reader = JsonlWindowReader {
            reader: BufReader::new(EpochDigestReader::new(read_handle)),
            line_number: 0,
            expected_size_bytes: u64::try_from(original.len()).expect("bounded fixture length"),
            expected_sha256: crate::config::Sha256Digest::of_bytes(original),
            verified_eof: false,
        };
        let mut observed = Vec::new();
        reader
            .reader
            .read_to_end(&mut observed)
            .expect("read mutated inode");
        assert_eq!(observed, replacement);
        assert!(matches!(
            reader.verify_epoch_digest(),
            Err(CorpusError::DatasetChanged)
        ));
    }
}
