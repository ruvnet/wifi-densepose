//! Provider-neutral speech-to-text and text-to-speech contracts.

use async_trait::async_trait;
use thiserror::Error;

use crate::audio::{AudioFormat, MAX_UTTERANCE_AUDIO_BYTES};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub language: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthesizedSpeech {
    pub audio: Vec<u8>,
    pub format: AudioFormat,
}

#[derive(Debug, Error)]
pub enum SpeechError {
    #[error("{0} provider is not configured")]
    NotConfigured(&'static str),
    #[error("speech provider rejected the request: {0}")]
    Provider(String),
    #[error("speech provider returned invalid data: {0}")]
    InvalidOutput(String),
    #[error("speech operation timed out")]
    Timeout,
}

#[async_trait]
pub trait SpeechToText: Send + Sync {
    async fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: &str,
    ) -> Result<Transcript, SpeechError>;
}

#[async_trait]
pub trait TextToSpeech: Send + Sync {
    async fn synthesize(
        &self,
        text: &str,
        language: &str,
    ) -> Result<SynthesizedSpeech, SpeechError>;
}

/// Fail-closed provider used until an STT integration is configured.
pub struct DisabledStt;

#[async_trait]
impl SpeechToText for DisabledStt {
    async fn transcribe(
        &self,
        _audio: &[u8],
        _format: AudioFormat,
        _language: &str,
    ) -> Result<Transcript, SpeechError> {
        Err(SpeechError::NotConfigured("STT"))
    }
}

/// Fail-closed provider used until a TTS integration is configured.
pub struct DisabledTts;

#[async_trait]
impl TextToSpeech for DisabledTts {
    async fn synthesize(
        &self,
        _text: &str,
        _language: &str,
    ) -> Result<SynthesizedSpeech, SpeechError> {
        Err(SpeechError::NotConfigured("TTS"))
    }
}

pub(crate) fn validate_provider_audio(audio: &[u8]) -> Result<(), SpeechError> {
    if audio.is_empty() || audio.len() > MAX_UTTERANCE_AUDIO_BYTES || audio.len() % 2 != 0 {
        return Err(SpeechError::InvalidOutput(
            "audio must be non-empty, bounded, aligned PCM".into(),
        ));
    }
    Ok(())
}
