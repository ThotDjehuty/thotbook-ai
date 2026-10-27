use std::path::Path;

use crate::error::TrainError;

/// Label marking a position the loss must ignore — padding, and the prompt
/// half of a prompt/completion example.
///
/// `labels` is `Vec<u32>`, so the usual `-100` sentinel cannot be stored
/// directly; `trainer.rs` maps this value to `-100i64` when it builds the
/// label tensor, and `loss.rs` masks every negative label. Writing
/// `-100i64 as u32` instead would produce `4294967196`, which is positive
/// once widened back to `i64`: the mask would pass it through and the gather
/// would index far past the vocabulary.
pub const IGNORE_LABEL: u32 = u32::MAX;

/// A single training example.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input_ids: Vec<u32>,
    pub labels: Vec<u32>,
}

/// Streaming dataset that reads training examples in chunks.
/// Never loads all data into RAM.
pub struct StreamingDataset {
    examples: Vec<TrainingExample>,
    batch_size: usize,
    position: usize,
}

impl StreamingDataset {
    /// Load examples from a JSONL file (one JSON object per line).
    pub fn from_jsonl(path: &Path, batch_size: usize) -> Result<Self, TrainError> {
        let file = std::fs::File::open(path).map_err(|e| TrainError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let reader = std::io::BufReader::new(file);
        let mut examples = Vec::new();

        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(|e| TrainError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| TrainError::Dataset(e.to_string()))?;

            let line_no = examples.len() + 1;
            let input_ids: Vec<u32> = record["input_ids"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default();
            // A negative label is the "ignore this position" sentinel. Parsing
            // labels as u64 dropped them instead, which silently shortened the
            // vector and shifted every label after it onto the wrong token.
            let labels: Vec<u32> = record["labels"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_i64())
                        .map(|n| if n < 0 { IGNORE_LABEL } else { n as u32 })
                        .collect()
                })
                .unwrap_or_default();

            // A line that parses but carries no input_ids is a mistake in
            // whatever produced the file — most often prompt/completion JSONL
            // fed straight to a loader that wants token ids. Skipping it
            // quietly made that look like an empty dataset and a training run
            // that did nothing.
            if input_ids.is_empty() {
                return Err(TrainError::Dataset(format!(
                    "{}: record {line_no} has no \"input_ids\"; this loader wants tokenised \
                     records. Prompt/completion shards from polarway-learn go through \
                     llm_train::samples::load_owner_dataset.",
                    path.display()
                )));
            }
            examples.push(TrainingExample { input_ids, labels });
        }

        tracing::info!(count = examples.len(), "loaded training examples");
        Ok(Self {
            examples,
            batch_size,
            position: 0,
        })
    }

    /// Load examples from in-memory (for small datasets).
    pub fn from_examples(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            position: 0,
        }
    }

    /// Get the next batch of examples.
    pub fn next_batch(&mut self) -> Option<&[TrainingExample]> {
        if self.position >= self.examples.len() {
            self.position = 0; // Reset for next epoch
            return None;
        }
        let end = (self.position + self.batch_size).min(self.examples.len());
        let batch = &self.examples[self.position..end];
        self.position = end;
        Some(batch)
    }

    /// Total number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_jsonl() {
        let dir = std::env::temp_dir().join(format!("ds-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{\"input_ids\":[1,2,3],\"labels\":[1,2,3]}}").unwrap();
        writeln!(f, "{{\"input_ids\":[4,5,6],\"labels\":[4,5,6]}}").unwrap();

        let mut ds = StreamingDataset::from_jsonl(&path, 10).unwrap();
        assert_eq!(ds.len(), 2);
        let batch = ds.next_batch().unwrap();
        assert_eq!(batch.len(), 2);

        std::fs::remove_dir_all(dir).unwrap();
    }

    fn write(tag: &str, lines: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ds-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        path
    }

    #[test]
    fn a_negative_label_becomes_the_ignore_sentinel() {
        // Parsing labels as u64 silently dropped -100, which shortened the
        // vector and shifted every later label onto the wrong token.
        let path = write(
            "neg",
            &[r#"{"input_ids":[1,2,3],"labels":[-100,-100,7]}"#],
        );
        let mut ds = StreamingDataset::from_jsonl(&path, 1).unwrap();
        let ex = &ds.next_batch().unwrap()[0];
        assert_eq!(ex.labels.len(), 3, "no label may be dropped");
        assert_eq!(ex.labels[0], IGNORE_LABEL);
        assert_eq!(ex.labels[1], IGNORE_LABEL);
        assert_eq!(ex.labels[2], 7);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn prompt_completion_records_are_refused_not_skipped() {
        // polarway-learn shards look like this. Skipping them in silence made
        // an empty dataset look like a successful load.
        let path = write(
            "pc",
            &[r#"{"tenant":"me","prompt":"q","completion":"a"}"#],
        );
        let msg = match StreamingDataset::from_jsonl(&path, 1) {
            Err(e) => e.to_string(),
            Ok(ds) => panic!("expected a refusal, loaded {} example(s)", ds.len()),
        };
        assert!(msg.contains("input_ids"), "{msg}");
        assert!(msg.contains("load_owner_dataset"), "{msg}");
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
