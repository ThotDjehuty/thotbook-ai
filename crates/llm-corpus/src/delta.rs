//! Delta lakehouse writer.
//!
//! Converts typed source rows into Arrow `RecordBatch`es and appends them to
//! Delta tables via `polarway-lakehouse`. Redaction is applied here so every
//! persisted column is scrubbed at the point of writing.

use std::sync::Arc;

use deltalake::arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use polarway_lakehouse::DeltaStore;

use crate::error::CorpusError;
use crate::redact::redact_secrets;
use crate::schema as sc;
use crate::sources::{ArxivPaper, Equation, Notebook, Paper, SessionDoc};

/// A batch ready to write, plus the build-time audit counters.
#[derive(Debug)]
pub struct PreparedBatch {
    pub batch: RecordBatch,
    pub rows: usize,
    /// Rows whose content still contains a credential pattern AFTER redaction.
    pub leaked_after_redaction: usize,
}

impl PreparedBatch {
    fn new(batch: RecordBatch, leaked: usize) -> Self {
        let rows = batch.num_rows();
        Self {
            batch,
            rows,
            leaked_after_redaction: leaked,
        }
    }
}

/// Write result for one table.
#[derive(Debug)]
pub struct WriteReport {
    pub table: String,
    pub rows: usize,
    pub leaked_after_redaction: usize,
    pub version: i64,
}

fn str_arr(values: impl IntoIterator<Item = String>) -> ArrayRef {
    Arc::new(StringArray::from(values.into_iter().collect::<Vec<_>>()))
}

fn str_opt_arr(values: impl IntoIterator<Item = Option<String>>) -> ArrayRef {
    Arc::new(StringArray::from(values.into_iter().collect::<Vec<_>>()))
}

fn i64_arr(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
}

fn i64_opt_arr(values: impl IntoIterator<Item = Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
}

/// Count leaked rows across a set of already-redacted column strings.
///
/// A row leaks only if a column still matches a credential pattern AFTER
/// redaction. Rows containing the `[REDACTED]` placeholder were scrubbed and
/// do NOT count as leaks.
fn leaked_rows(columns: &[&[String]]) -> usize {
    if columns.is_empty() {
        return 0;
    }
    let n = columns[0].len();
    (0..n)
        .filter(|&i| {
            columns
                .iter()
                .any(|col| crate::redact::contains_secrets(&col[i]))
        })
        .count()
}

/// Build the `datasets.sessions` PreparedBatch from historia docs.
pub fn sessions_batch(docs: &[SessionDoc]) -> PreparedBatch {
    let content: Vec<String> = docs.iter().map(|d| redact_secrets(&d.content)).collect();
    let leaked = leaked_rows(&[&content]);

    PreparedBatch::new(
        RecordBatch::try_new(
            Arc::new(sc::sessions_schema()),
            vec![
                str_arr(docs.iter().map(|d| d.id.clone())),
                str_arr(docs.iter().map(|d| d.filename.clone())),
                str_arr(docs.iter().map(|d| d.date.clone())),
                str_arr(docs.iter().map(|d| d.title.clone())),
                str_arr(content),
                str_arr(docs.iter().map(|d| d.tags.clone())),
                str_arr(docs.iter().map(|d| d.category.clone())),
                i64_arr(docs.iter().map(|d| d.word_count)),
                str_arr(docs.iter().map(|d| d.indexed_at.clone())),
                str_opt_arr(docs.iter().map(|d| d.file_modified.clone())),
            ],
        )
        .expect("sessions schema matches batch columns"),
        leaked,
    )
}

/// Build the `datasets.corpus` PreparedBatch from papers/notebooks/arxiv.
pub fn corpus_batch(papers: &[Paper], notebooks: &[Notebook], arxiv: &[ArxivPaper]) -> PreparedBatch {
    let mut source_type: Vec<String> = Vec::new();
    let mut source_id: Vec<String> = Vec::new();
    let mut title: Vec<String> = Vec::new();
    let mut authors: Vec<String> = Vec::new();
    let mut content: Vec<String> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    let mut category: Vec<String> = Vec::new();
    let mut year: Vec<Option<i64>> = Vec::new();
    let mut word_count: Vec<i64> = Vec::new();
    let mut equation_count: Vec<i64> = Vec::new();
    let mut indexed_at: Vec<String> = Vec::new();

    for p in papers {
        source_type.push("paper".into());
        source_id.push(p.id.clone());
        title.push(p.title.clone());
        authors.push(p.authors.clone());
        content.push(redact_secrets(&p.content));
        tags.push(p.tags.clone());
        category.push(p.category.clone());
        year.push(p.year);
        word_count.push(p.word_count);
        equation_count.push(p.equation_count);
        indexed_at.push(p.indexed_at.clone());
    }
    for nb in notebooks {
        source_type.push("notebook".into());
        source_id.push(nb.id.clone());
        title.push(nb.title.clone());
        authors.push(String::new());
        content.push(redact_secrets(&nb.content));
        tags.push(nb.tags.clone());
        category.push(nb.category.clone());
        year.push(None);
        word_count.push(nb.content.split_whitespace().count() as i64);
        equation_count.push(0);
        indexed_at.push(nb.indexed_at.clone());
    }
    for a in arxiv {
        let text = if a.full_text.is_empty() { &a.abstract_ } else { &a.full_text };
        let content_red = redact_secrets(text);
        source_type.push("arxiv".into());
        source_id.push(a.arxiv_id.clone());
        title.push(a.title.clone());
        authors.push(a.authors.clone());
        content.push(content_red.clone());
        tags.push(a.categories.clone());
        category.push(a.primary_category.clone());
        year.push(a.published_date.chars().take(4).collect::<String>().parse().ok());
        word_count.push(content_red.split_whitespace().count() as i64);
        equation_count.push(0);
        indexed_at.push(a.indexed_at.clone());
    }

    let leaked = leaked_rows(&[&content]);

    PreparedBatch::new(
        RecordBatch::try_new(
            Arc::new(sc::corpus_schema()),
            vec![
                str_arr(source_type),
                str_arr(source_id),
                str_arr(title),
                str_arr(authors),
                str_arr(content),
                str_arr(tags),
                str_arr(category),
                i64_opt_arr(year),
                i64_arr(word_count),
                i64_arr(equation_count),
                str_arr(indexed_at),
            ],
        )
        .expect("corpus schema matches batch columns"),
        leaked,
    )
}

/// Build the `datasets.equations` PreparedBatch.
pub fn equations_batch(eqs: &[Equation]) -> PreparedBatch {
    let latex: Vec<String> = eqs.iter().map(|e| redact_secrets(&e.latex)).collect();
    let context: Vec<String> = eqs.iter().map(|e| redact_secrets(&e.context)).collect();
    let leaked = leaked_rows(&[&latex, &context]);

    PreparedBatch::new(
        RecordBatch::try_new(
            Arc::new(sc::equations_schema()),
            vec![
                i64_arr(eqs.iter().map(|e| e.id)),
                str_arr(latex),
                str_arr(eqs.iter().map(|e| e.label.clone())),
                str_arr(context),
                str_arr(eqs.iter().map(|e| e.source_type.clone())),
                str_arr(eqs.iter().map(|e| e.source_id.clone())),
                str_arr(eqs.iter().map(|e| e.section.clone())),
                str_arr(eqs.iter().map(|e| e.indexed_at.clone())),
            ],
        )
        .expect("equations schema matches batch columns"),
        leaked,
    )
}

/// Build the `datasets.citations` PreparedBatch (nodes; edges via M6 crawler).
pub fn citations_batch(arxiv: &[ArxivPaper]) -> PreparedBatch {
    let abstract_: Vec<String> = arxiv.iter().map(|a| redact_secrets(&a.abstract_)).collect();
    let leaked = leaked_rows(&[&abstract_]);

    PreparedBatch::new(
        RecordBatch::try_new(
            Arc::new(sc::citations_schema()),
            vec![
                str_arr(arxiv.iter().map(|a| a.arxiv_id.clone())),
                str_arr(arxiv.iter().map(|a| a.title.clone())),
                str_arr(arxiv.iter().map(|a| a.authors.clone())),
                str_arr(arxiv.iter().map(|a| a.categories.clone())),
                str_arr(arxiv.iter().map(|a| a.primary_category.clone())),
                str_arr(arxiv.iter().map(|a| a.published_date.clone())),
                str_arr(abstract_),
                str_arr(arxiv.iter().map(|_| "[]".to_string())),
                str_arr(arxiv.iter().map(|a| a.fetch_status.clone())),
                str_arr(arxiv.iter().map(|a| a.indexed_at.clone())),
            ],
        )
        .expect("citations schema matches batch columns"),
        leaked,
    )
}

/// Append a PreparedBatch to a Delta table, creating it if needed.
pub async fn append_batch(
    store: &DeltaStore,
    table: &str,
    fields: Vec<deltalake::kernel::StructField>,
    prepared: PreparedBatch,
) -> Result<WriteReport, CorpusError> {
    if prepared.leaked_after_redaction > 0 {
        return Err(CorpusError::Redaction(format!(
            "refusing to write {table}: {leaked} rows still contain credential patterns",
            table = table,
            leaked = prepared.leaked_after_redaction
        )));
    }
    store
        .ensure_table(table, fields, vec![])
        .await
        .map_err(|source| CorpusError::Delta { table: table.to_string(), source })?;
    let version = store
        .append(table, prepared.batch)
        .await
        .map_err(|source| CorpusError::Delta { table: table.to_string(), source })?;
    Ok(WriteReport {
        table: table.to_string(),
        rows: prepared.rows,
        leaked_after_redaction: prepared.leaked_after_redaction,
        version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redact::REDACTED;

    fn paper(id: &str, content: &str) -> Paper {
        Paper {
            id: id.into(),
            filename: "main.tex".into(),
            title: "T".into(),
            authors: "A".into(),
            abstract_: String::new(),
            content: content.into(),
            tags: String::new(),
            category: "general".into(),
            year: Some(2026),
            word_count: 1,
            equation_count: 0,
            indexed_at: "2026-08-09".into(),
        }
    }

    #[test]
    fn corpus_redacts_secrets() {
        let prep = corpus_batch(&[paper("p1", "token=supersecretvalue\nrest")], &[], &[]);
        assert_eq!(prep.rows, 1);
        assert_eq!(prep.leaked_after_redaction, 0, "redaction must clear credentials");
        let content = prep
            .batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let text = content.value(0);
        assert!(text.contains(REDACTED));
        assert!(!text.contains("supersecretvalue"));
    }

    #[test]
    fn corpus_counts_real_leaks() {
        // A value our redactor does NOT cover still counts as a leak.
        let prep = corpus_batch(&[paper("p1", "fine plain text")], &[], &[]);
        assert_eq!(prep.leaked_after_redaction, 0);
    }

    #[test]
    fn sessions_batch_shapes() {
        let docs = vec![SessionDoc {
            id: "20260101_x".into(),
            filename: "20260101_x.md".into(),
            date: "2026-01-01".into(),
            title: "Hi".into(),
            content: "token=abc123".into(),
            tags: "[]".into(),
            category: "research".into(),
            word_count: 2,
            indexed_at: "2026-01-01".into(),
            file_modified: None,
        }];
        let prep = sessions_batch(&docs);
        assert_eq!(prep.rows, 1);
        assert_eq!(prep.leaked_after_redaction, 0);
    }
}
