//! Error types for the `llm-serve` inference server (railway programming).

use std::path::PathBuf;

/// Errors surfaced by `llm-serve` (M5 private inference server).
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// Errors originating from candle (tensor ops, weight loading, model build).
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Errors loading or using the tokenizer.
    #[error("tokenizer error loading `{path}`: {source}")]
    Tokenizer {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },

    /// Invalid or unusable configuration.
    #[error("configuration error: {0}")]
    Config(String),

    /// Model directory is missing weights, config, or is otherwise unusable.
    #[error("model error: {0}")]
    Model(String),

    /// HTTP request/response handling errors.
    #[error("http error: {0}")]
    Http(String),

    /// Filesystem errors.
    #[error("io error on `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl ServeError {
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn http(msg: impl Into<String>) -> Self {
        Self::Http(msg.into())
    }
}
