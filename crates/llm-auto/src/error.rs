//! Error types for the autonomous orchestration system.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AutoError {
    #[error("inference error: {0}")]
    Inference(String),

    #[error("task decomposition failed: {0}")]
    Decomposition(String),

    #[error("agent error: {0}")]
    Agent(String),

    #[error("training error: {0}")]
    Training(String),

    #[error("paper generation error: {0}")]
    Paper(String),

    #[error("IO error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("timeout: {0}")]
    Timeout(String),
}

impl From<reqwest::Error> for AutoError {
    fn from(e: reqwest::Error) -> Self {
        AutoError::Http(e.to_string())
    }
}
