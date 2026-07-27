//! Transport-independent satellite voice session protocol.
//!
//! Text frames use [`SatelliteClientMessage`] / [`SatelliteServerMessage`].
//! Binary frames are accepted only while a stream is active.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::audio::{AudioChunk, AudioError, AudioFormat, MAX_UTTERANCE_AUDIO_BYTES};

pub const SATELLITE_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SatelliteClientMessage {
    Hello {
        version: u16,
        token: String,
    },
    Start {
        language: String,
        format: AudioFormat,
    },
    End,
    Cancel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SatelliteServerMessage {
    Ready { version: u16 },
    Started,
    Transcript { text: String, language: String },
    Intent { response: crate::IntentResponse },
    Audio { format: AudioFormat, bytes: usize },
    Finished,
    Error { code: String, message: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SatelliteState {
    AwaitingHello,
    Idle,
    Streaming,
    Closed,
}

/// Strict state machine used by WebSocket or native satellite transports.
pub struct SatelliteSession {
    state: SatelliteState,
    authenticated: bool,
    audio: Vec<u8>,
    format: Option<AudioFormat>,
    language: Option<String>,
}

impl SatelliteSession {
    pub fn new() -> Self {
        Self {
            state: SatelliteState::AwaitingHello,
            authenticated: false,
            audio: Vec::new(),
            format: None,
            language: None,
        }
    }

    pub fn state(&self) -> SatelliteState {
        self.state
    }

    /// Handle a control message. `authenticate` must perform constant-time
    /// credential comparison; credentials are never retained by this session.
    pub fn control(
        &mut self,
        message: SatelliteClientMessage,
        authenticate: impl FnOnce(&str) -> bool,
    ) -> Result<SatelliteServerMessage, SatelliteError> {
        match (self.state, message) {
            (SatelliteState::AwaitingHello, SatelliteClientMessage::Hello { version, token }) => {
                if version != SATELLITE_PROTOCOL_VERSION {
                    self.state = SatelliteState::Closed;
                    return Err(SatelliteError::UnsupportedVersion(version));
                }
                if !authenticate(&token) {
                    self.state = SatelliteState::Closed;
                    return Err(SatelliteError::Unauthorized);
                }
                self.authenticated = true;
                self.state = SatelliteState::Idle;
                Ok(SatelliteServerMessage::Ready { version })
            }
            (SatelliteState::Idle, SatelliteClientMessage::Start { language, format })
                if self.authenticated =>
            {
                if language.is_empty() || language.len() > 35 {
                    return Err(SatelliteError::InvalidLanguage);
                }
                self.format = Some(format.validate()?);
                self.language = Some(language);
                self.audio.clear();
                self.state = SatelliteState::Streaming;
                Ok(SatelliteServerMessage::Started)
            }
            (SatelliteState::Streaming, SatelliteClientMessage::End) => {
                if self.audio.is_empty() {
                    return Err(SatelliteError::EmptyStream);
                }
                self.state = SatelliteState::Idle;
                Ok(SatelliteServerMessage::Finished)
            }
            (SatelliteState::Streaming, SatelliteClientMessage::Cancel) => {
                self.reset_stream();
                Ok(SatelliteServerMessage::Finished)
            }
            (_, SatelliteClientMessage::Cancel) => {
                self.reset_stream();
                Ok(SatelliteServerMessage::Finished)
            }
            _ => Err(SatelliteError::InvalidSequence),
        }
    }

    pub fn audio(&mut self, chunk: AudioChunk) -> Result<(), SatelliteError> {
        if self.state != SatelliteState::Streaming {
            return Err(SatelliteError::InvalidSequence);
        }
        if self.audio.len().saturating_add(chunk.as_bytes().len()) > MAX_UTTERANCE_AUDIO_BYTES {
            self.reset_stream();
            return Err(SatelliteError::AudioLimit);
        }
        self.audio.extend_from_slice(chunk.as_bytes());
        Ok(())
    }

    pub fn take_utterance(&mut self) -> Option<(Vec<u8>, AudioFormat, String)> {
        if self.state != SatelliteState::Idle || self.audio.is_empty() {
            return None;
        }
        let audio = std::mem::take(&mut self.audio);
        Some((audio, self.format.take()?, self.language.take()?))
    }

    fn reset_stream(&mut self) {
        self.audio.clear();
        self.format = None;
        self.language = None;
        self.state = if self.authenticated {
            SatelliteState::Idle
        } else {
            SatelliteState::AwaitingHello
        };
    }
}

impl Default for SatelliteSession {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum SatelliteError {
    #[error("satellite message is invalid in the current session state")]
    InvalidSequence,
    #[error("satellite protocol version {0} is unsupported")]
    UnsupportedVersion(u16),
    #[error("satellite authentication failed")]
    Unauthorized,
    #[error("invalid language tag")]
    InvalidLanguage,
    #[error("audio stream is empty")]
    EmptyStream,
    #[error("audio stream exceeded its size limit")]
    AudioLimit,
    #[error(transparent)]
    Audio(#[from] AudioError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioCodec;

    fn format() -> AudioFormat {
        AudioFormat {
            codec: AudioCodec::PcmS16Le,
            sample_rate: 16_000,
            channels: 1,
        }
    }

    #[test]
    fn happy_path_preserves_audio_and_metadata() {
        let mut session = SatelliteSession::new();
        session
            .control(
                SatelliteClientMessage::Hello {
                    version: 1,
                    token: "secret".into(),
                },
                |token| token == "secret",
            )
            .unwrap();
        session
            .control(
                SatelliteClientMessage::Start {
                    language: "en-CA".into(),
                    format: format(),
                },
                |_| false,
            )
            .unwrap();
        session
            .audio(AudioChunk::new(vec![1, 0, 2, 0]).unwrap())
            .unwrap();
        session
            .control(SatelliteClientMessage::End, |_| false)
            .unwrap();
        let (audio, stored_format, language) = session.take_utterance().unwrap();
        assert_eq!(audio, vec![1, 0, 2, 0]);
        assert_eq!(stored_format, format());
        assert_eq!(language, "en-CA");
    }

    #[test]
    fn unauthenticated_stream_is_rejected_and_closed() {
        let mut session = SatelliteSession::new();
        let error = session
            .control(
                SatelliteClientMessage::Hello {
                    version: 1,
                    token: "wrong".into(),
                },
                |_| false,
            )
            .unwrap_err();
        assert!(matches!(error, SatelliteError::Unauthorized));
        assert_eq!(session.state(), SatelliteState::Closed);
    }

    #[test]
    fn binary_before_start_is_rejected() {
        let mut session = SatelliteSession::new();
        let error = session
            .audio(AudioChunk::new(vec![0, 0]).unwrap())
            .unwrap_err();
        assert!(matches!(error, SatelliteError::InvalidSequence));
    }
}
