//! `llm-ingest` — Universal document ingestion for thotbook-ai.
//!
//! Handles PDF extraction, Markdown parsing, and agent file ingestion.
//! Outputs chunked documents ready for RAG indexing and embedding.

pub mod agent;
pub mod chunker;
pub mod error;
pub mod latex;
pub mod markdown;
pub mod pdf;
pub mod pipeline;

pub use error::IngestError;
pub use pipeline::{IngestConfig, IngestPipeline, IngestReport};
