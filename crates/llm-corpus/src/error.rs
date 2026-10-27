//! Error types for the corpus ingestion pipeline (railway programming).

use std::path::PathBuf;

/// Errors surfaced by `llm-corpus` (M1 ingest).
#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("SQLite error in {db}: {source}")]
    Sqlite {
        db: PathBuf,
        #[source]
        source: rusqlite::Error,
    },

    #[error("Delta/Lakehouse error on table `{table}`: {source}")]
    Delta {
        table: String,
        #[source]
        source: polarway_lakehouse::error::LakehouseError,
    },

    #[error("Arrow error: {0}")]
    Arrow(#[from] deltalake::arrow::error::ArrowError),

    #[error("redaction error: {0}")]
    Redaction(String),

    #[error("IO error on `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("missing source database: {0}")]
    MissingDb(PathBuf),

    #[error("invalid table selection `{0}` — expected one of: sessions, corpus, equations, citations")]
    UnknownTable(String),
}
