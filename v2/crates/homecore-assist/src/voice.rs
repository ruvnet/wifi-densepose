//! End-to-end STT → intent → TTS pipeline.

use std::time::Duration;

use homecore::HomeCore;
use thiserror::Error;
use tokio::time::timeout;

use crate::audio::{AudioFormat, MAX_UTTERANCE_AUDIO_BYTES};
use crate::pipeline::AssistPipeline;
use crate::recognizer::IntentRecognizer;
use crate::speech::{
    validate_provider_audio, SpeechError, SpeechToText, SynthesizedSpeech, TextToSpeech, Transcript,
};
use crate::{AssistError, IntentResponse};

pub const DEFAULT_VOICE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub struct VoiceResponse {
    pub transcript: Transcript,
    pub intent: IntentResponse,
    pub speech: SynthesizedSpeech,
}

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("input audio is empty or exceeds the utterance limit")]
    InvalidAudio,
    #[error(transparent)]
    Speech(#[from] SpeechError),
    #[error(transparent)]
    Assist(#[from] AssistError),
    #[error("voice pipeline timed out")]
    Timeout,
}

pub struct VoicePipeline<S, T, R: IntentRecognizer> {
    stt: S,
    tts: T,
    assist: AssistPipeline<R>,
    timeout: Duration,
}

impl<S, T, R> VoicePipeline<S, T, R>
where
    S: SpeechToText,
    T: TextToSpeech,
    R: IntentRecognizer,
{
    pub fn new(stt: S, tts: T, assist: AssistPipeline<R>) -> Self {
        Self {
            stt,
            tts,
            assist,
            timeout: DEFAULT_VOICE_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, value: Duration) -> Self {
        self.timeout = value;
        self
    }

    pub async fn process(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: &str,
        hc: &HomeCore,
    ) -> Result<VoiceResponse, VoiceError> {
        if audio.is_empty() || audio.len() > MAX_UTTERANCE_AUDIO_BYTES || audio.len() % 2 != 0 {
            return Err(VoiceError::InvalidAudio);
        }
        let format = format.validate().map_err(|_| VoiceError::InvalidAudio)?;
        timeout(self.timeout, async {
            let transcript = self.stt.transcribe(audio, format, language).await?;
            let intent = self
                .assist
                .process(&transcript.text, &transcript.language, hc)
                .await?;
            let speech = self
                .tts
                .synthesize(&intent.speech, &transcript.language)
                .await?;
            speech
                .format
                .validate()
                .map_err(|error| SpeechError::InvalidOutput(error.to_string()))?;
            validate_provider_audio(&speech.audio)?;
            Ok(VoiceResponse {
                transcript,
                intent,
                speech,
            })
        })
        .await
        .map_err(|_| VoiceError::Timeout)?
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::audio::AudioCodec;
    use crate::recognizer::RegexIntentRecognizer;
    use crate::speech::{SpeechToText, TextToSpeech};

    struct FixedStt;

    #[async_trait]
    impl SpeechToText for FixedStt {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _format: AudioFormat,
            language: &str,
        ) -> Result<Transcript, SpeechError> {
            Ok(Transcript {
                text: "never mind".into(),
                language: language.into(),
            })
        }
    }

    struct FixedTts;

    #[async_trait]
    impl TextToSpeech for FixedTts {
        async fn synthesize(
            &self,
            text: &str,
            _language: &str,
        ) -> Result<SynthesizedSpeech, SpeechError> {
            assert!(!text.is_empty());
            Ok(SynthesizedSpeech {
                audio: vec![0, 0, 1, 0],
                format: format(),
            })
        }
    }

    fn format() -> AudioFormat {
        AudioFormat {
            codec: AudioCodec::PcmS16Le,
            sample_rate: 16_000,
            channels: 1,
        }
    }

    #[tokio::test]
    async fn runs_stt_intent_and_tts() {
        let recognizer = RegexIntentRecognizer::new();
        recognizer
            .register("HassNevermind", r"never ?mind", "*")
            .await
            .unwrap();
        let pipeline = crate::pipeline::default_pipeline(recognizer);
        let voice = VoicePipeline::new(FixedStt, FixedTts, pipeline);
        let result = voice
            .process(&[0, 0, 1, 0], format(), "en-CA", &HomeCore::new())
            .await
            .unwrap();
        assert_eq!(result.transcript.text, "never mind");
        assert_eq!(result.speech.audio.len(), 4);
    }

    #[tokio::test]
    async fn rejects_unaligned_audio_before_provider_call() {
        let voice = VoicePipeline::new(
            FixedStt,
            FixedTts,
            crate::pipeline::default_pipeline(RegexIntentRecognizer::new()),
        );
        let error = voice
            .process(&[0], format(), "en", &HomeCore::new())
            .await
            .unwrap_err();
        assert!(matches!(error, VoiceError::InvalidAudio));
    }
}
