//! `llm-corpus` — thotbook-ai M1.
//!
//! Ingests the opencode session corpus (`historia.db`) and the research index
//! (`thotbook.db`) into Polarway Delta tables (`datasets.sessions`,
//! `datasets.corpus`, `datasets.equations`, `datasets.citations`) with secret
//! redaction applied at write time.

pub mod delta;
pub mod error;
pub mod redact;
pub mod schema;
pub mod sources;

pub use delta::{WriteReport, PreparedBatch};
pub use error::CorpusError;
