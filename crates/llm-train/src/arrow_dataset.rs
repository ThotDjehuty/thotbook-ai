//! Arrow IPC dataset reader for training data.
//!
//! Reads tokenized shards produced by llm-tokenize (M2).

use std::path::{Path, PathBuf};

use arrow::array::{Array, UInt32Array};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;

use crate::dataset::TrainingExample;
use crate::error::TrainError;

/// Manifest for Arrow IPC shards (written by llm-tokenize).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ShardManifest {
    pub table: String,
    pub shard_count: usize,
    pub total_tokens: usize,
    pub tokens_per_shard: usize,
    pub schema: Vec<String>,
}

/// Dataset that streams from Arrow IPC shard files.
pub struct ArrowDataset {
    shard_paths: Vec<PathBuf>,
    current_shard: usize,
    current_batch_idx: usize,
    current_reader: Option<FileReader<std::fs::File>>,
    batch_size: usize,
    buffer: Vec<TrainingExample>,
    buffer_pos: usize,
}

impl ArrowDataset {
    /// Load from a directory containing manifest.json and shard-*.arrow files.
    pub fn from_dir(dir: &Path, batch_size: usize) -> Result<Self, TrainError> {
        let manifest_path = dir.join("manifest.json");
        let manifest_data = std::fs::read_to_string(&manifest_path).map_err(|e| TrainError::Io {
            path: manifest_path.clone(),
            source: e,
        })?;
        let manifest: ShardManifest =
            serde_json::from_str(&manifest_data).map_err(|e| TrainError::Dataset(e.to_string()))?;

        let mut shard_paths = Vec::new();
        for i in 0..manifest.shard_count {
            let shard_path = dir.join(format!("shard-{i:05}.arrow"));
            if shard_path.exists() {
                shard_paths.push(shard_path);
            }
        }

        if shard_paths.is_empty() {
            return Err(TrainError::Dataset(format!(
                "no shard files found in {}",
                dir.display()
            )));
        }

        tracing::info!(
            dir = %dir.display(),
            shards = shard_paths.len(),
            total_tokens = manifest.total_tokens,
            "loaded Arrow dataset"
        );

        Ok(Self {
            shard_paths,
            current_shard: 0,
            current_batch_idx: 0,
            current_reader: None,
            batch_size,
            buffer: Vec::new(),
            buffer_pos: 0,
        })
    }

    /// Open the next shard file.
    fn open_next_shard(&mut self) -> Result<bool, TrainError> {
        if self.current_shard >= self.shard_paths.len() {
            return Ok(false);
        }

        let path = &self.shard_paths[self.current_shard];
        let file = std::fs::File::open(path).map_err(|e| TrainError::Io {
            path: path.clone(),
            source: e,
        })?;

        let reader = FileReader::try_new(file, None)
            .map_err(|e| TrainError::Dataset(e.to_string()))?;

        self.current_reader = Some(reader);
        self.current_batch_idx = 0;
        self.current_shard += 1;
        Ok(true)
    }

    /// Read examples from current shard into buffer.
    fn fill_buffer(&mut self) -> Result<bool, TrainError> {
        // Open first shard if needed
        if self.current_reader.is_none()
            && !self.open_next_shard()? {
                return Ok(false);
            }

        loop {
            let reader = match &mut self.current_reader {
                Some(r) => r,
                None => return Ok(false),
            };

            // Try to read next batch
            match reader.next() {
                Some(Ok(batch)) => {
                    self.extract_examples(&batch)?;
                    if !self.buffer.is_empty() {
                        return Ok(true);
                    }
                }
                Some(Err(e)) => return Err(TrainError::Dataset(e.to_string())),
                None => {
                    // End of current shard, try next
                    self.current_reader = None;
                    if !self.open_next_shard()? {
                        return Ok(false);
                    }
                }
            }
        }
    }

    /// Extract training examples from an Arrow RecordBatch.
    fn extract_examples(&mut self, batch: &RecordBatch) -> Result<(), TrainError> {
        // Look for "input_ids" or "tokens" column
        let col_idx = batch
            .schema()
            .index_of("input_ids")
            .or_else(|_| batch.schema().index_of("tokens"))
            .map_err(|_| {
                TrainError::Dataset("no input_ids or tokens column in Arrow batch".to_string())
            })?;

        let array = batch.column(col_idx);

        // Handle list<u32> or direct u32 array
        if let Some(list_array) = array.as_any().downcast_ref::<arrow::array::ListArray>() {
            for i in 0..list_array.len() {
                if list_array.is_null(i) {
                    continue;
                }
                let values = list_array.value(i);
                if let Some(u32_array) = values.as_any().downcast_ref::<UInt32Array>() {
                    let tokens: Vec<u32> = u32_array.iter().flatten().collect();
                    if !tokens.is_empty() {
                        // For causal LM, labels = input_ids shifted by 1
                        let labels = tokens.clone();
                        self.buffer.push(TrainingExample {
                            input_ids: tokens,
                            labels,
                        });
                    }
                }
            }
        } else if let Some(u32_array) = array.as_any().downcast_ref::<UInt32Array>() {
            // Single flat array - treat as one sequence
            let tokens: Vec<u32> = u32_array.iter().flatten().collect();
            if !tokens.is_empty() {
                let labels = tokens.clone();
                self.buffer.push(TrainingExample {
                    input_ids: tokens,
                    labels,
                });
            }
        }

        Ok(())
    }

    /// Get the next batch of examples.
    pub fn next_batch(&mut self) -> Option<&[TrainingExample]> {
        // Refill buffer if needed
        while self.buffer_pos >= self.buffer.len() {
            self.buffer.clear();
            self.buffer_pos = 0;
            match self.fill_buffer() {
                Ok(true) => {}
                _ => return None,
            }
        }

        let start = self.buffer_pos;
        let end = (start + self.batch_size).min(self.buffer.len());
        self.buffer_pos = end;

        if start >= end {
            None
        } else {
            Some(&self.buffer[start..end])
        }
    }

    /// Reset for another epoch.
    pub fn reset(&mut self) {
        self.current_shard = 0;
        self.current_batch_idx = 0;
        self.current_reader = None;
        self.buffer.clear();
        self.buffer_pos = 0;
    }

    /// Estimated total examples (approximate).
    pub fn len(&self) -> usize {
        // Rough estimate based on shard count
        self.shard_paths.len() * 1000
    }

    pub fn is_empty(&self) -> bool {
        self.shard_paths.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parse() {
        let json = r#"{"table":"test","shard_count":2,"total_tokens":10000,"tokens_per_shard":5000,"schema":["input_ids"]}"#;
        let manifest: ShardManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.shard_count, 2);
        assert_eq!(manifest.total_tokens, 10000);
    }
}
