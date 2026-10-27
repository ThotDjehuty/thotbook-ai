use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("PDF extraction failed for {path}: {source}")]
    Pdf {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Markdown parsing failed for {path}: {reason}")]
    Markdown { path: PathBuf, reason: String },

    #[error("Agent file parse error for {path}: {reason}")]
    Agent { path: PathBuf, reason: String },

    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Chunking error: {0}")]
    Chunking(String),
}
