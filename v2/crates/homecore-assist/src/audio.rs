//! Bounded audio types shared by STT, TTS, and satellite transports.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum audio accepted in one chunk (256 KiB).
pub const MAX_AUDIO_CHUNK_BYTES: usize = 256 * 1024;
/// Maximum audio accepted in a single utterance (16 MiB).
pub const MAX_UTTERANCE_AUDIO_BYTES: usize = 16 * 1024 * 1024;

/// Audio encodings supported by the native voice pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioCodec {
    /// Signed, little-endian, 16-bit PCM.
    PcmS16Le,
}

/// A validated audio stream format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u8,
}

impl AudioFormat {
    pub fn validate(self) -> Result<Self, AudioError> {
        if !(8_000..=48_000).contains(&self.sample_rate) {
            return Err(AudioError::InvalidSampleRate(self.sample_rate));
        }
        if !(1..=2).contains(&self.channels) {
            return Err(AudioError::InvalidChannels(self.channels));
        }
        Ok(self)
    }
}

/// One bounded audio packet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioChunk {
    bytes: Vec<u8>,
}

impl AudioChunk {
    pub fn new(bytes: Vec<u8>) -> Result<Self, AudioError> {
        if bytes.is_empty() {
            return Err(AudioError::Empty);
        }
        if bytes.len() > MAX_AUDIO_CHUNK_BYTES {
            return Err(AudioError::ChunkTooLarge(bytes.len()));
        }
        if bytes.len() % 2 != 0 {
            return Err(AudioError::UnalignedPcm);
        }
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AudioError {
    #[error("audio chunk is empty")]
    Empty,
    #[error("audio chunk is {0} bytes; maximum is {MAX_AUDIO_CHUNK_BYTES}")]
    ChunkTooLarge(usize),
    #[error("16-bit PCM data must contain an even number of bytes")]
    UnalignedPcm,
    #[error("sample rate {0} Hz is outside 8000..=48000")]
    InvalidSampleRate(u32),
    #[error("channel count {0} is outside 1..=2")]
    InvalidChannels(u8),
    #[error("utterance audio exceeds {MAX_UTTERANCE_AUDIO_BYTES} bytes")]
    UtteranceTooLarge,
}
