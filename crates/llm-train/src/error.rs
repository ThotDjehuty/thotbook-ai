use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrainError {
    #[error("Candle error: {0}")]
    Candle(String),

    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Dataset error: {0}")]
    Dataset(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),
}

impl From<candle_core::Error> for TrainError {
    fn from(e: candle_core::Error) -> Self {
        TrainError::Candle(e.to_string())
    }
}
