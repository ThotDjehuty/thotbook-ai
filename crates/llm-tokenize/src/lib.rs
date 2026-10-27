//! thotbook-ai M2 — tokenize the lakehouse corpus into sharded training-ready
//! Arrow IPC token shards using the HuggingFace `tokenizers` crate.

pub mod error;
pub mod shard;
pub mod tokenizer;

use std::path::Path;

use arrow::array::{LargeStringArray, RecordBatch, StringArray};
use tokenizers::Tokenizer;

use crate::error::TokenizeError;
use crate::shard::{write_shards, DEFAULT_TOKENS_PER_SHARD};
use crate::tokenizer::DEFAULT_TOKENIZER_PATH;

/// Summary of one tokenization pass over a lakehouse table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizeReport {
    pub total_tokens: u64,
    pub shards: usize,
    pub bytes: u64,
}

/// Tokenize the `content_col` strings of `batch` with `tok`, join them with
/// `"\n\n"`, and write sharded Arrow IPC files under `out_dir`.
///
/// `table`, tokenizer path and shard budget are derived from the defaults;
/// use [`tokenize_table_with`] for explicit control.
pub fn tokenize_table(
    batch: &RecordBatch,
    content_col: &str,
    tok: &Tokenizer,
    out_dir: &Path,
) -> Result<TokenizeReport, TokenizeError> {
    let table = out_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    tokenize_table_with(
        batch,
        content_col,
        tok,
        out_dir,
        &table,
        Path::new(DEFAULT_TOKENIZER_PATH),
        DEFAULT_TOKENS_PER_SHARD,
    )
}

/// Full tokenize entry point with explicit manifest metadata and shard budget.
pub fn tokenize_table_with(
    batch: &RecordBatch,
    content_col: &str,
    tok: &Tokenizer,
    out_dir: &Path,
    table: &str,
    tokenizer_path: &Path,
    tokens_per_shard: usize,
) -> Result<TokenizeReport, TokenizeError> {
    let text = concat_content(batch, content_col)?;
    let ids = tok
        .encode(text, true)
        .map_err(|source| TokenizeError::Tokenizer { source })?
        .get_ids()
        .to_vec();
    let written = write_shards(&ids, out_dir, tokens_per_shard, table, tokenizer_path)?;
    Ok(TokenizeReport {
        total_tokens: written.manifest.total_tokens,
        shards: written.manifest.shards.len(),
        bytes: written.bytes,
    })
}

/// Extract a string column from `batch` and join its values with `"\n\n"`.
fn concat_content(batch: &RecordBatch, content_col: &str) -> Result<String, TokenizeError> {
    let col = batch
        .column_by_name(content_col)
        .ok_or_else(|| TokenizeError::MissingColumn(content_col.to_string()))?;
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(arr.iter().flatten().collect::<Vec<_>>().join("\n\n"));
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(arr.iter().flatten().collect::<Vec<_>>().join("\n\n"));
    }
    Err(TokenizeError::UnexpectedColumnType {
        col: content_col.to_string(),
        ty: col.data_type().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow::array::{StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};

    fn synthetic_tokenizer() -> Tokenizer {
        let vocab = [("[UNK]", 0u32), ("hello", 1), ("world", 2)]
            .into_iter()
            .map(|(w, i)| (w.to_string(), i))
            .collect::<ahash::AHashMap<String, u32>>();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tok
    }

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-tokenize-lib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tokenize_table_joins_and_shards() {
        let schema = Arc::new(Schema::new(vec![Field::new("content", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world hello", "world"]))],
        )
        .unwrap();
        let tok = synthetic_tokenizer();
        let dir = temp_dir();
        let report = tokenize_table_with(
            &batch,
            "content",
            &tok,
            &dir,
            "datasets.sessions",
            Path::new("/tok/plain.json"),
            2,
        )
        .unwrap();

        assert_eq!(report.total_tokens, 4); // hello / world / hello / world
        assert_eq!(report.shards, 2);
        assert!(report.bytes > 0);
        assert!(dir.join("shard-00001.arrow").exists());

        let manifest: shard::Manifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.table, "datasets.sessions");
        assert_eq!(manifest.tokenizer_path, "/tok/plain.json");
        assert_eq!(manifest.total_tokens, 4);
        assert_eq!(manifest.shards.len(), 2);
        assert!(manifest.created_at.ends_with('Z'));
    }

    #[test]
    fn tokenize_table_uses_defaults() {
        let schema = Arc::new(Schema::new(vec![Field::new("latex", DataType::Utf8, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["hello"]))]).unwrap();
        let tok = synthetic_tokenizer();
        let dir = temp_dir();
        let report = tokenize_table(&batch, "latex", &tok, &dir).unwrap();
        assert_eq!(report.total_tokens, 1);
        let manifest: shard::Manifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.schema_version, shard::SCHEMA_VERSION);
        assert_eq!(manifest.tokenizer_path, DEFAULT_TOKENIZER_PATH);
    }

    #[test]
    fn missing_column_is_reported() {
        let schema = Arc::new(Schema::new(vec![Field::new("content", DataType::Utf8, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["hello"]))]).unwrap();
        let tok = synthetic_tokenizer();
        let dir = temp_dir();
        let err = tokenize_table(&batch, "nope", &tok, &dir).unwrap_err();
        assert!(matches!(err, TokenizeError::MissingColumn(_)));
    }

    #[test]
    fn non_string_column_is_reported() {
        let schema = Arc::new(Schema::new(vec![Field::new("ids", DataType::UInt32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from(vec![1, 2, 3]))]).unwrap();
        let tok = synthetic_tokenizer();
        let dir = temp_dir();
        let err = tokenize_table(&batch, "ids", &tok, &dir).unwrap_err();
        assert!(matches!(err, TokenizeError::UnexpectedColumnType { .. }));
    }
}
