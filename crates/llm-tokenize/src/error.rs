//! Error types for the tokenization pipeline (railway programming).

use std::path::PathBuf;

/// Errors surfaced by `llm-tokenize` (M2 sharded tokenization).
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("Delta read error for table `{table}` at `{path}`: {detail}")]
    Delta {
        table: String,
        path: PathBuf,
        detail: String,
    },

    #[error("tokenizer error: {source}")]
    Tokenizer {
        #[source]
        source: tokenizers::Error,
    },

    #[error("failed to download tokenizer from `{url}`: {source}")]
    Download {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("IO error on `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("content column `{0}` not found in table schema")]
    MissingColumn(String),

    #[error("content column `{col}` has unexpected type `{ty}` — expected string")]
    UnexpectedColumnType { col: String, ty: String },

    #[error("invalid tokens-per-shard: {0} (must be > 0)")]
    InvalidShardSize(usize),
}
