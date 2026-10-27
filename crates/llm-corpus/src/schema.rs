//! Delta table schemas (Arrow + Delta `StructField`), mirroring the
//! polarway-lakehouse `schema.rs` convention.

use deltalake::arrow::datatypes::{DataType, Field, Schema};
use deltalake::kernel::{DataType as DeltaDataType, PrimitiveType, StructField};

// ─── Table names ───

pub const TABLE_SESSIONS: &str = "datasets.sessions";
pub const TABLE_CORPUS: &str = "datasets.corpus";
pub const TABLE_EQUATIONS: &str = "datasets.equations";
pub const TABLE_CITATIONS: &str = "datasets.citations";
pub const TABLE_ARXIV_TEXT: &str = "datasets.arxiv_text";

/// All tables managed by `llm-corpus`.
pub const ALL_TABLES: [&str; 5] = [
    TABLE_SESSIONS,
    TABLE_CORPUS,
    TABLE_EQUATIONS,
    TABLE_CITATIONS,
    TABLE_ARXIV_TEXT,
];

fn s(name: &str, nullable: bool) -> StructField {
    StructField::new(name, DeltaDataType::Primitive(PrimitiveType::String), nullable)
}

fn i64(name: &str, nullable: bool) -> StructField {
    StructField::new(name, DeltaDataType::Primitive(PrimitiveType::Long), nullable)
}

fn f(name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable)
}

/// `datasets.sessions` — one row per persisted opencode session document.
pub fn sessions_schema() -> Schema {
    Schema::new(vec![
        f("id", DataType::Utf8, false),
        f("filename", DataType::Utf8, false),
        f("date", DataType::Utf8, false),
        f("title", DataType::Utf8, false),
        f("content", DataType::Utf8, false),
        f("tags", DataType::Utf8, false),
        f("category", DataType::Utf8, false),
        f("word_count", DataType::Int64, false),
        f("indexed_at", DataType::Utf8, false),
        f("file_modified", DataType::Utf8, true),
    ])
}

pub fn sessions_delta_fields() -> Vec<StructField> {
    vec![
        s("id", false),
        s("filename", false),
        s("date", false),
        s("title", false),
        s("content", false),
        s("tags", false),
        s("category", false),
        i64("word_count", false),
        s("indexed_at", false),
        s("file_modified", true),
    ]
}

/// `datasets.corpus` — unified research corpus (papers + notebooks + arxiv).
pub fn corpus_schema() -> Schema {
    Schema::new(vec![
        f("source_type", DataType::Utf8, false),
        f("source_id", DataType::Utf8, false),
        f("title", DataType::Utf8, false),
        f("authors", DataType::Utf8, false),
        f("content", DataType::Utf8, false),
        f("tags", DataType::Utf8, false),
        f("category", DataType::Utf8, false),
        f("year", DataType::Int64, true),
        f("word_count", DataType::Int64, false),
        f("equation_count", DataType::Int64, false),
        f("indexed_at", DataType::Utf8, false),
    ])
}

pub fn corpus_delta_fields() -> Vec<StructField> {
    vec![
        s("source_type", false),
        s("source_id", false),
        s("title", false),
        s("authors", false),
        s("content", false),
        s("tags", false),
        s("category", false),
        i64("year", true),
        i64("word_count", false),
        i64("equation_count", false),
        s("indexed_at", false),
    ]
}

/// `datasets.equations` — LaTeX equations extracted from papers/notebooks.
pub fn equations_schema() -> Schema {
    Schema::new(vec![
        f("id", DataType::Int64, false),
        f("latex", DataType::Utf8, false),
        f("label", DataType::Utf8, false),
        f("context", DataType::Utf8, false),
        f("source_type", DataType::Utf8, false),
        f("source_id", DataType::Utf8, false),
        f("section", DataType::Utf8, false),
        f("indexed_at", DataType::Utf8, false),
    ])
}

pub fn equations_delta_fields() -> Vec<StructField> {
    vec![
        i64("id", false),
        s("latex", false),
        s("label", false),
        s("context", false),
        s("source_type", false),
        s("source_id", false),
        s("section", false),
        s("indexed_at", false),
    ]
}

/// `datasets.citations` — citation graph nodes (M6 crawler populates edges).
pub fn citations_schema() -> Schema {
    Schema::new(vec![
        f("arxiv_id", DataType::Utf8, false),
        f("title", DataType::Utf8, false),
        f("authors", DataType::Utf8, false),
        f("categories", DataType::Utf8, false),
        f("primary_category", DataType::Utf8, false),
        f("published_date", DataType::Utf8, false),
        f("abstract", DataType::Utf8, false),
        f("cited_ids", DataType::Utf8, false),
        f("fetch_status", DataType::Utf8, false),
        f("indexed_at", DataType::Utf8, false),
    ])
}

pub fn citations_delta_fields() -> Vec<StructField> {
    vec![
        s("arxiv_id", false),
        s("title", false),
        s("authors", false),
        s("categories", false),
        s("primary_category", false),
        s("published_date", false),
        s("abstract", false),
        s("cited_ids", false),
        s("fetch_status", false),
        s("indexed_at", false),
    ]
}

/// `datasets.arxiv_text` — M6 crawler output: one row per arXiv paper with
/// metadata + full text (ar5iv) + reference-closure citation edges.
pub fn arxiv_text_schema() -> Schema {
    Schema::new(vec![
        f("arxiv_id", DataType::Utf8, false),
        f("title", DataType::Utf8, false),
        f("authors", DataType::Utf8, false),
        f("categories", DataType::Utf8, false),
        f("primary_category", DataType::Utf8, false),
        f("published_date", DataType::Utf8, false),
        f("abstract", DataType::Utf8, false),
        f("full_text", DataType::Utf8, false),
        f("cited_ids", DataType::Utf8, false),
        f("fetch_status", DataType::Utf8, false),
        f("indexed_at", DataType::Utf8, false),
    ])
}

pub fn arxiv_text_delta_fields() -> Vec<StructField> {
    vec![
        s("arxiv_id", false),
        s("title", false),
        s("authors", false),
        s("categories", false),
        s("primary_category", false),
        s("published_date", false),
        s("abstract", false),
        s("full_text", false),
        s("cited_ids", false),
        s("fetch_status", false),
        s("indexed_at", false),
    ]
}
