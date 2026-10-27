//! Sharded Arrow IPC writing for token ID streams.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use serde::{Deserialize, Serialize};

use crate::error::TokenizeError;

/// Default token budget per shard.
pub const DEFAULT_TOKENS_PER_SHARD: usize = 1_000_000;

/// Version of the on-disk manifest/shards layout.
pub const SCHEMA_VERSION: u32 = 1;

/// One shard entry inside a [`Manifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardEntry {
    pub path: String,
    pub tokens: u64,
}

/// Sidecar manifest describing a tokenized table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub tokenizer_path: String,
    pub table: String,
    pub shards: Vec<ShardEntry>,
    pub total_tokens: u64,
    pub created_at: String,
}

/// Result of a shard write pass.
#[derive(Debug, Clone)]
pub struct WrittenShards {
    pub manifest: Manifest,
    /// Total bytes of the shard files (excluding the manifest).
    pub bytes: u64,
}

fn ids_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("ids", DataType::Int64, false)]))
}

fn shard_batch(ids: &[u32]) -> Result<RecordBatch, TokenizeError> {
    let values: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    RecordBatch::try_new(ids_schema(), vec![Arc::new(Int64Array::from(values))])
        .map_err(TokenizeError::Arrow)
}

fn write_shard(ids: &[u32], path: &Path) -> Result<u64, TokenizeError> {
    let file = std::fs::File::create(path).map_err(|source| TokenizeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = FileWriter::try_new_buffered(file, ids_schema().as_ref())?;
    writer.write(&shard_batch(ids)?)?;
    writer.finish()?;
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|source| TokenizeError::Io {
            path: path.to_path_buf(),
            source,
        })
}

/// Split `ids` into `tokens_per_shard`-bounded Arrow IPC files under `out_dir`
/// and write a sidecar `manifest.json`. Pure and iterator-driven.
pub fn write_shards(
    ids: &[u32],
    out_dir: &Path,
    tokens_per_shard: usize,
    table: &str,
    tokenizer_path: &Path,
) -> Result<WrittenShards, TokenizeError> {
    if tokens_per_shard == 0 {
        return Err(TokenizeError::InvalidShardSize(tokens_per_shard));
    }
    std::fs::create_dir_all(out_dir).map_err(|source| TokenizeError::Io {
        path: out_dir.to_path_buf(),
        source,
    })?;

    let shards: Vec<(String, u64, u64)> = ids
        .chunks(tokens_per_shard)
        .enumerate()
        .map(|(i, chunk)| {
            let name = format!("shard-{i:05}.arrow");
            let bytes = write_shard(chunk, &out_dir.join(&name))?;
            Ok((name, chunk.len() as u64, bytes))
        })
        .collect::<Result<_, TokenizeError>>()?;

    let bytes: u64 = shards.iter().map(|(_, _, b)| b).sum();
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        tokenizer_path: tokenizer_path.to_string_lossy().into_owned(),
        table: table.to_string(),
        shards: shards
            .into_iter()
            .map(|(path, tokens, _)| ShardEntry { path, tokens })
            .collect(),
        total_tokens: ids.len() as u64,
        created_at: rfc3339_now(),
    };

    let manifest_path = out_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json).map_err(|source| TokenizeError::Io {
        path: manifest_path,
        source,
    })?;

    Ok(WrittenShards { manifest, bytes })
}

/// Current UTC time as an RFC 3339 timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hh, mm, ss) = (secs_of_day / 3600, (secs_of_day % 3600) / 60, secs_of_day % 60);
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Convert days since epoch to a proleptic Gregorian `(year, month, day)`
/// using Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-tokenize-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn shards_split_by_token_budget() {
        let dir = temp_dir();
        let ids: Vec<u32> = (0..2500).collect();
        let written = write_shards(&ids, &dir, 1000, "datasets.sessions", Path::new("/tok/plain.json"))
            .unwrap();

        assert_eq!(written.manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(written.manifest.table, "datasets.sessions");
        assert_eq!(written.manifest.tokenizer_path, "/tok/plain.json");
        assert_eq!(written.manifest.total_tokens, 2500);
        assert_eq!(written.manifest.shards.len(), 3);
        assert_eq!(
            written
                .manifest
                .shards
                .iter()
                .map(|s| s.tokens)
                .collect::<Vec<_>>(),
            vec![1000, 1000, 500]
        );
        assert!(written.bytes > 0);
        assert!(dir.join("manifest.json").exists());

        for (i, entry) in written.manifest.shards.iter().enumerate() {
            assert_eq!(entry.path, format!("shard-{i:05}.arrow"));
            let file = std::fs::File::open(dir.join(&entry.path)).unwrap();
            let reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
            let batch = reader.into_iter().next().unwrap().unwrap();
            let col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.len(), entry.tokens as usize);
        }
    }

    #[test]
    fn empty_ids_produce_empty_manifest() {
        let dir = temp_dir();
        let written = write_shards(&[], &dir, 1000, "datasets.corpus", Path::new("/tok/plain.json"))
            .unwrap();
        assert_eq!(written.manifest.shards.len(), 0);
        assert_eq!(written.manifest.total_tokens, 0);
        assert_eq!(written.bytes, 0);
    }

    #[test]
    fn manifest_json_round_trips() {
        let m = Manifest {
            schema_version: 1,
            tokenizer_path: "/data/tokenizers/qwen3-tokenizer.json".into(),
            table: "datasets.sessions".into(),
            shards: vec![ShardEntry {
                path: "shard-00000.arrow".into(),
                tokens: 10,
            }],
            total_tokens: 10,
            created_at: "2026-08-09T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn zero_shard_size_is_rejected() {
        let dir = temp_dir();
        let err = write_shards(&[1, 2, 3], &dir, 0, "t", Path::new("/tok")).unwrap_err();
        assert!(matches!(err, TokenizeError::InvalidShardSize(0)));
    }
}
