use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("Embedding generation failed: {0}")]
    Embedding(String),

    #[error("HNSW index error: {0}")]
    Hnsw(String),

    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Index not loaded: {0}")]
    NotLoaded(String),
}
