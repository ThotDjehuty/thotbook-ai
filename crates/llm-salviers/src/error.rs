use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SalviersError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Agent parse error for {path}: {reason}")]
    Parse { path: PathBuf, reason: String },

    #[error("No agents loaded")]
    NoAgents,

    #[error("Embedding error: {0}")]
    Embedding(String),
}
