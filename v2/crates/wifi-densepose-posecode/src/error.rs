//! Error types for parser and adapter boundaries.

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors are structured and line anchored where input text is involved.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("invalid scene: {0}")]
    Validation(String),
    #[error("invalid observation: {0}")]
    Observation(String),
}

impl Error {
    pub(crate) fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }
}
