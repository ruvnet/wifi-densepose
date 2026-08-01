//! MiniMax text-to-audio provider for the HOMECORE TTS contract.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::audio::{AudioCodec, AudioFormat};
use crate::speech::{SpeechError, SynthesizedSpeech, TextToSpeech};

pub const MINIMAX_TTS_MODELS: &[&str] = &[
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
];

pub const MINIMAX_TTS_DEFAULT_MODEL: &str = "speech-2.8-hd";

pub const MINIMAX_TTS_ENDPOINTS: &[(&str, &str)] = &[
    ("global_en", "https://api.minimax.io/v1/t2a_v2"),
    ("cn_zh", "https://api.minimaxi.com/v1/t2a_v2"),
];

pub const MINIMAX_TTS_DOCS_URLS: &[&str] = &[
    "https://platform.minimax.io/docs/api-reference/speech-t2a-http",
    "https://platform.minimax.io/docs/api-reference/speech-t2a-async-create",
    "https://platform.minimax.io/docs/api-reference/speech-t2a-websocket",
    "https://platform.minimaxi.com/docs/api-reference/speech-t2a-http",
    "https://platform.minimaxi.com/docs/api-reference/speech-t2a-async-create",
    "https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket",
];

pub const MINIMAX_TTS_REQUEST_FIELDS: &[&str] = &[
    "model",
    "text",
    "stream",
    "language_boost",
    "output_format",
    "voice_setting",
    "pronunciation_dict",
    "audio_setting",
    "voice_modify",
    "subtitle_enable",
];

pub const MINIMAX_TTS_AUDIO_FORMATS: &[MiniMaxAudioOutputFormat] = &[
    MiniMaxAudioOutputFormat::Mp3,
    MiniMaxAudioOutputFormat::Wav,
    MiniMaxAudioOutputFormat::Flac,
    MiniMaxAudioOutputFormat::Pcm,
];

pub const MINIMAX_TTS_RESPONSE_FIELDS: &[&str] =
    &["data.audio", "data.status", "base_resp.status_code"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxRegion {
    GlobalEn,
    CnZh,
}

impl MiniMaxRegion {
    pub fn endpoint(self) -> &'static str {
        match self {
            Self::GlobalEn => "https://api.minimax.io/v1/t2a_v2",
            Self::CnZh => "https://api.minimaxi.com/v1/t2a_v2",
        }
    }

    pub fn docs_root(self) -> &'static str {
        match self {
            Self::GlobalEn => "https://platform.minimax.io/docs",
            Self::CnZh => "https://platform.minimaxi.com/docs",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxAudioOutputFormat {
    Mp3,
    Wav,
    Flac,
    Pcm,
}

impl MiniMaxAudioOutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxTtsConfig {
    pub api_key: String,
    pub region: MiniMaxRegion,
    pub model: String,
    pub voice_id: Option<String>,
    pub output_format: MiniMaxAudioOutputFormat,
    pub sample_rate: u32,
    pub channels: u8,
    pub bitrate: u32,
    pub timeout: Duration,
}

impl MiniMaxTtsConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            region: MiniMaxRegion::GlobalEn,
            model: MINIMAX_TTS_DEFAULT_MODEL.into(),
            voice_id: None,
            output_format: MiniMaxAudioOutputFormat::Pcm,
            sample_rate: 16_000,
            channels: 1,
            bitrate: 128_000,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_region(mut self, region: MiniMaxRegion) -> Self {
        self.region = region;
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = Some(voice_id.into());
        self
    }

    pub fn with_output_format(mut self, output_format: MiniMaxAudioOutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    pub fn validate(&self) -> Result<(), SpeechError> {
        if self.api_key.trim().is_empty() {
            return Err(SpeechError::NotConfigured("MiniMax TTS"));
        }
        if !MINIMAX_TTS_MODELS.contains(&self.model.as_str()) {
            return Err(SpeechError::Provider(format!(
                "unsupported MiniMax speech model: {}",
                self.model
            )));
        }
        AudioFormat {
            codec: AudioCodec::PcmS16Le,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
        .validate()
        .map_err(|err| SpeechError::Provider(err.to_string()))?;
        Ok(())
    }
}

pub struct MiniMaxTts {
    config: MiniMaxTtsConfig,
}

impl MiniMaxTts {
    pub fn new(config: MiniMaxTtsConfig) -> Result<Self, SpeechError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &MiniMaxTtsConfig {
        &self.config
    }

    fn synthesize_blocking(
        &self,
        text: &str,
        language: &str,
    ) -> Result<SynthesizedSpeech, SpeechError> {
        let request = build_tts_request(&self.config, text, language);
        let body = serde_json::to_string(&request)
            .map_err(|err| SpeechError::Provider(err.to_string()))?;
        let response = ureq::AgentBuilder::new()
            .timeout(self.config.timeout)
            .build()
            .post(self.config.region.endpoint())
            .set("Authorization", &format!("Bearer {}", self.config.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body);

        let response = match response {
            Ok(response) => response,
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                return Err(SpeechError::Provider(format!(
                    "MiniMax TTS returned HTTP {code}: {body}"
                )));
            }
            Err(ureq::Error::Transport(err)) => {
                return Err(SpeechError::Provider(err.to_string()));
            }
        };

        let body = response
            .into_string()
            .map_err(|err| SpeechError::InvalidOutput(err.to_string()))?;
        parse_tts_response(&body, self.config.sample_rate, self.config.channels)
    }
}

#[async_trait]
impl TextToSpeech for MiniMaxTts {
    async fn synthesize(
        &self,
        text: &str,
        language: &str,
    ) -> Result<SynthesizedSpeech, SpeechError> {
        if text.trim().is_empty() {
            return Err(SpeechError::Provider("text is required".into()));
        }
        self.synthesize_blocking(text, language)
    }
}

#[derive(Debug, Serialize)]
struct MiniMaxTtsRequest<'a> {
    model: &'a str,
    text: &'a str,
    stream: bool,
    language_boost: &'a str,
    output_format: &'a str,
    voice_setting: Option<MiniMaxVoiceSetting<'a>>,
    audio_setting: MiniMaxAudioSetting,
}

#[derive(Debug, Serialize)]
struct MiniMaxVoiceSetting<'a> {
    voice_id: &'a str,
}

#[derive(Debug, Serialize)]
struct MiniMaxAudioSetting {
    sample_rate: u32,
    bitrate: u32,
    channel: u8,
    format: &'static str,
}

fn build_tts_request<'a>(
    config: &'a MiniMaxTtsConfig,
    text: &'a str,
    language: &'a str,
) -> MiniMaxTtsRequest<'a> {
    MiniMaxTtsRequest {
        model: &config.model,
        text,
        stream: false,
        language_boost: language,
        output_format: config.output_format.as_str(),
        voice_setting: config
            .voice_id
            .as_deref()
            .map(|voice_id| MiniMaxVoiceSetting { voice_id }),
        audio_setting: MiniMaxAudioSetting {
            sample_rate: config.sample_rate,
            bitrate: config.bitrate,
            channel: config.channels,
            format: config.output_format.as_str(),
        },
    }
}

#[derive(Debug, Deserialize)]
struct MiniMaxTtsResponse {
    data: Option<MiniMaxTtsData>,
    base_resp: Option<MiniMaxBaseResponse>,
}

#[derive(Debug, Deserialize)]
struct MiniMaxTtsData {
    audio: Option<String>,
    status: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct MiniMaxBaseResponse {
    status_code: Option<i64>,
    status_msg: Option<String>,
}

fn parse_tts_response(
    body: &str,
    sample_rate: u32,
    channels: u8,
) -> Result<SynthesizedSpeech, SpeechError> {
    let response: MiniMaxTtsResponse =
        serde_json::from_str(body).map_err(|err| SpeechError::InvalidOutput(err.to_string()))?;
    if let Some(base_resp) = response.base_resp {
        if base_resp.status_code.unwrap_or(0) != 0 {
            return Err(SpeechError::Provider(
                base_resp
                    .status_msg
                    .unwrap_or_else(|| "MiniMax TTS request failed".into()),
            ));
        }
    }

    let audio_hex = response
        .data
        .and_then(|data| {
            let _ = data.status;
            data.audio
        })
        .ok_or_else(|| SpeechError::InvalidOutput("missing data.audio".into()))?;
    let audio = decode_hex_audio(&audio_hex)?;
    crate::speech::validate_provider_audio(&audio)?;
    Ok(SynthesizedSpeech {
        audio,
        format: AudioFormat {
            codec: AudioCodec::PcmS16Le,
            sample_rate,
            channels,
        },
    })
}

fn decode_hex_audio(input: &str) -> Result<Vec<u8>, SpeechError> {
    let hex = input.trim();
    if hex.len() % 2 != 0 {
        return Err(SpeechError::InvalidOutput(
            "data.audio hex length must be even".into(),
        ));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Result<u8, SpeechError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(SpeechError::InvalidOutput(
            "data.audio must be hex encoded".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_current_minimax_tts_models_regions_and_fields() {
        assert_eq!(MINIMAX_TTS_DEFAULT_MODEL, "speech-2.8-hd");
        assert!(MINIMAX_TTS_MODELS.contains(&"speech-2.8-turbo"));
        assert!(MINIMAX_TTS_ENDPOINTS.contains(&("global_en", "https://api.minimax.io/v1/t2a_v2")));
        assert!(MINIMAX_TTS_ENDPOINTS.contains(&("cn_zh", "https://api.minimaxi.com/v1/t2a_v2")));
        assert!(MINIMAX_TTS_REQUEST_FIELDS.contains(&"voice_setting"));
        assert!(MINIMAX_TTS_REQUEST_FIELDS.contains(&"audio_setting"));
        assert!(MINIMAX_TTS_AUDIO_FORMATS.contains(&MiniMaxAudioOutputFormat::Pcm));
        assert!(MINIMAX_TTS_AUDIO_FORMATS.contains(&MiniMaxAudioOutputFormat::Flac));
        assert!(MINIMAX_TTS_RESPONSE_FIELDS.contains(&"data.audio"));
    }

    #[test]
    fn builds_http_request_with_region_model_voice_audio_and_language_fields() {
        let config = MiniMaxTtsConfig::new("test-key")
            .with_region(MiniMaxRegion::CnZh)
            .with_model("speech-2.8-turbo")
            .with_voice_id("voice-123")
            .with_output_format(MiniMaxAudioOutputFormat::Pcm);

        let request = build_tts_request(&config, "hello", "zh");
        let json = serde_json::to_value(request).unwrap();

        assert_eq!(
            config.region.endpoint(),
            "https://api.minimaxi.com/v1/t2a_v2"
        );
        assert_eq!(json["model"], "speech-2.8-turbo");
        assert_eq!(json["text"], "hello");
        assert_eq!(json["stream"], false);
        assert_eq!(json["language_boost"], "zh");
        assert_eq!(json["output_format"], "pcm");
        assert_eq!(json["voice_setting"]["voice_id"], "voice-123");
        assert_eq!(json["audio_setting"]["format"], "pcm");
        assert_eq!(json["audio_setting"]["sample_rate"], 16000);
    }

    #[test]
    fn parses_hex_audio_response_and_validates_base_response() {
        let speech = parse_tts_response(
            r#"{"data":{"audio":"00000100","status":2},"base_resp":{"status_code":0}}"#,
            16_000,
            1,
        )
        .unwrap();

        assert_eq!(speech.audio, vec![0, 0, 1, 0]);
        assert_eq!(speech.format.sample_rate, 16_000);
        assert_eq!(speech.format.channels, 1);
    }

    #[test]
    fn rejects_failed_or_malformed_responses() {
        assert!(parse_tts_response(
            r#"{"data":{"audio":"0000"},"base_resp":{"status_code":1001,"status_msg":"bad request"}}"#,
            16_000,
            1,
        )
        .is_err());
        assert!(parse_tts_response(r#"{"data":{"audio":"xyz"}}"#, 16_000, 1).is_err());
        assert!(parse_tts_response(r#"{"data":{}}"#, 16_000, 1).is_err());
    }
}
