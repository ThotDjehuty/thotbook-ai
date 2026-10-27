//! SQLite source readers.
//!
//! Reads the three source databases:
//! - `historia.db` — persisted opencode session documents (`documents` table)
//! - `thotbook.db` — research index (`papers`, `notebooks`, `equations`, `arxiv_papers`)
//!
//! All reads are intentionally row-shaped (`Vec<Row>`) so downstream steps
//! (redaction → Arrow → Delta) are pure data transformations.

use std::path::Path;

use rusqlite::Connection;

use crate::error::CorpusError;

/// A persisted opencode session document from `historia.db`.
#[derive(Debug, Clone)]
pub struct SessionDoc {
    pub id: String,
    pub filename: String,
    pub date: String,
    pub title: String,
    pub content: String,
    pub tags: String,
    pub category: String,
    pub word_count: i64,
    pub indexed_at: String,
    pub file_modified: Option<String>,
}

/// A paper (markdown or LaTeX dossier) from `thotbook.db`.
#[derive(Debug, Clone)]
pub struct Paper {
    pub id: String,
    pub filename: String,
    pub title: String,
    pub authors: String,
    pub abstract_: String,
    pub content: String,
    pub tags: String,
    pub category: String,
    pub year: Option<i64>,
    pub word_count: i64,
    pub equation_count: i64,
    pub indexed_at: String,
}

/// A Jupyter notebook from `thotbook.db`.
#[derive(Debug, Clone)]
pub struct Notebook {
    pub id: String,
    pub filename: String,
    pub title: String,
    pub description: String,
    pub content: String,
    pub tags: String,
    pub category: String,
    pub cell_count: i64,
    pub code_cell_count: i64,
    pub markdown_cell_count: i64,
    pub has_latex: i64,
    pub has_rust_bindings: i64,
    pub indexed_at: String,
}

/// A LaTeX equation from `thotbook.db`.
#[derive(Debug, Clone)]
pub struct Equation {
    pub id: i64,
    pub latex: String,
    pub label: String,
    pub context: String,
    pub source_type: String,
    pub source_id: String,
    pub section: String,
    pub indexed_at: String,
}

/// An arXiv paper from `thotbook.db`.
#[derive(Debug, Clone)]
pub struct ArxivPaper {
    pub arxiv_id: String,
    pub title: String,
    pub authors: String,
    pub abstract_: String,
    pub categories: String,
    pub primary_category: String,
    pub published_date: String,
    pub updated_date: String,
    pub pdf_url: String,
    pub abs_url: String,
    pub html_url: String,
    pub comment: String,
    pub journal_ref: String,
    pub fetch_status: String,
    pub full_text: String,
    pub indexed_at: String,
}

fn open(path: &Path) -> Result<Connection, CorpusError> {
    Connection::open(path).map_err(|source| CorpusError::Sqlite {
        db: path.to_path_buf(),
        source,
    })
}

/// Read every session document from `historia.db`.
pub fn read_sessions(path: &Path) -> Result<Vec<SessionDoc>, CorpusError> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, filename, date, title, content, tags, category, word_count, indexed_at, file_modified
             FROM documents ORDER BY date",
        )
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(SessionDoc {
                id: row.get(0)?,
                filename: row.get(1)?,
                date: row.get(2)?,
                title: row.get(3)?,
                content: row.get(4)?,
                tags: row.get(5)?,
                category: row.get(6)?,
                word_count: row.get(7)?,
                indexed_at: row.get(8)?,
                file_modified: row.get(9)?,
            })
        })
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;
    Ok(rows)
}

/// Read every paper from `thotbook.db`.
pub fn read_papers(path: &Path) -> Result<Vec<Paper>, CorpusError> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, filename, title, authors, abstract, content, tags, category, year, word_count, equation_count, indexed_at
             FROM papers ORDER BY indexed_at",
        )
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(Paper {
                id: row.get(0)?,
                filename: row.get(1)?,
                title: row.get(2)?,
                authors: row.get(3)?,
                abstract_: row.get(4)?,
                content: row.get(5)?,
                tags: row.get(6)?,
                category: row.get(7)?,
                year: row.get(8)?,
                word_count: row.get(9)?,
                equation_count: row.get(10)?,
                indexed_at: row.get(11)?,
            })
        })
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;
    Ok(rows)
}

/// Read every notebook from `thotbook.db`.
pub fn read_notebooks(path: &Path) -> Result<Vec<Notebook>, CorpusError> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, filename, title, description, content, tags, category,
                    cell_count, code_cell_count, markdown_cell_count, has_latex, has_rust_bindings, indexed_at
             FROM notebooks ORDER BY indexed_at",
        )
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(Notebook {
                id: row.get(0)?,
                filename: row.get(1)?,
                title: row.get(2)?,
                description: row.get(3)?,
                content: row.get(4)?,
                tags: row.get(5)?,
                category: row.get(6)?,
                cell_count: row.get(7)?,
                code_cell_count: row.get(8)?,
                markdown_cell_count: row.get(9)?,
                has_latex: row.get(10)?,
                has_rust_bindings: row.get(11)?,
                indexed_at: row.get(12)?,
            })
        })
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;
    Ok(rows)
}

/// Read every equation from `thotbook.db`.
pub fn read_equations(path: &Path) -> Result<Vec<Equation>, CorpusError> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, latex, label, context, source_type, source_id, section, indexed_at
             FROM equations ORDER BY id",
        )
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(Equation {
                id: row.get(0)?,
                latex: row.get(1)?,
                label: row.get(2)?,
                context: row.get(3)?,
                source_type: row.get(4)?,
                source_id: row.get(5)?,
                section: row.get(6)?,
                indexed_at: row.get(7)?,
            })
        })
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;
    Ok(rows)
}

/// Read every arXiv paper from `thotbook.db`.
pub fn read_arxiv(path: &Path) -> Result<Vec<ArxivPaper>, CorpusError> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT arxiv_id, title, authors, abstract, categories, primary_category,
                    published_date, updated_date, pdf_url, abs_url, html_url,
                    comment, journal_ref, fetch_status, full_text, indexed_at
             FROM arxiv_papers ORDER BY indexed_at",
        )
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(ArxivPaper {
                arxiv_id: row.get(0)?,
                title: row.get(1)?,
                authors: row.get(2)?,
                abstract_: row.get(3)?,
                categories: row.get(4)?,
                primary_category: row.get(5)?,
                published_date: row.get(6)?,
                updated_date: row.get(7)?,
                pdf_url: row.get(8)?,
                abs_url: row.get(9)?,
                html_url: row.get(10)?,
                comment: row.get(11)?,
                journal_ref: row.get(12)?,
                fetch_status: row.get(13)?,
                full_text: row.get(14)?,
                indexed_at: row.get(15)?,
            })
        })
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CorpusError::Sqlite { db: path.to_path_buf(), source })?;
    Ok(rows)
}
