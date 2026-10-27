//! Training orchestration for the self-improvement loop.

use std::path::PathBuf;
use std::fs::OpenOptions;
use std::io::Write;

use crate::error::AutoError;

/// Orchestrates the training process.
pub struct TrainingOrchestrator {
    session_corpus_dir: PathBuf,
    lakehouse_dir: PathBuf,
    training_data_path: PathBuf,
}

impl TrainingOrchestrator {
    pub fn new(session_corpus_dir: PathBuf, lakehouse_dir: PathBuf) -> Self {
        let training_data_path = lakehouse_dir.join("training_queue.jsonl");
        Self {
            session_corpus_dir,
            lakehouse_dir,
            training_data_path,
        }
    }

    /// Add training examples to the queue.
    pub fn add_examples(&self, examples: Vec<(String, String)>) -> Result<(), AutoError> {
        std::fs::create_dir_all(&self.lakehouse_dir).map_err(|e| AutoError::Io {
            path: self.lakehouse_dir.clone(),
            source: e,
        })?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.training_data_path)
            .map_err(|e| AutoError::Io {
                path: self.training_data_path.clone(),
                source: e,
            })?;

        for (input, output) in examples {
            // Format as chat-style training example
            let example = serde_json::json!({
                "messages": [
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": output}
                ]
            });
            writeln!(file, "{}", example).map_err(|e| AutoError::Io {
                path: self.training_data_path.clone(),
                source: e,
            })?;
        }

        tracing::info!(
            path = %self.training_data_path.display(),
            "appended training examples"
        );
        Ok(())
    }

    /// Get the path to the training data file.
    pub fn training_data_path(&self) -> &PathBuf {
        &self.training_data_path
    }

    /// Count pending training examples.
    pub fn count_examples(&self) -> Result<usize, AutoError> {
        if !self.training_data_path.exists() {
            return Ok(0);
        }

        let content = std::fs::read_to_string(&self.training_data_path).map_err(|e| AutoError::Io {
            path: self.training_data_path.clone(),
            source: e,
        })?;

        Ok(content.lines().filter(|l| !l.trim().is_empty()).count())
    }

    /// Get session-corpus statistics.
    pub fn session_corpus_stats(&self) -> Result<SessionCorpusStats, AutoError> {
        let mut stats = SessionCorpusStats::default();

        if !self.session_corpus_dir.exists() {
            return Ok(stats);
        }

        for entry in std::fs::read_dir(&self.session_corpus_dir).map_err(|e| AutoError::Io {
            path: self.session_corpus_dir.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| AutoError::Io {
                path: self.session_corpus_dir.clone(),
                source: e,
            })?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "md") {
                stats.documents += 1;

                if let Ok(content) = std::fs::read_to_string(&path) {
                    stats.total_words += content.split_whitespace().count();
                    stats.total_bytes += content.len();
                }
            }
        }

        Ok(stats)
    }

    /// Alias for prepare_from_session_corpus.
    pub fn extract_from_session_corpus(&self) -> Result<usize, AutoError> {
        self.prepare_from_session_corpus()
    }

    /// Prepare training data from the session-log corpus.
    pub fn prepare_from_session_corpus(&self) -> Result<usize, AutoError> {
        let mut count = 0;

        if !self.session_corpus_dir.exists() {
            return Ok(0);
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.training_data_path)
            .map_err(|e| AutoError::Io {
                path: self.training_data_path.clone(),
                source: e,
            })?;

        for entry in std::fs::read_dir(&self.session_corpus_dir).map_err(|e| AutoError::Io {
            path: self.session_corpus_dir.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| AutoError::Io {
                path: self.session_corpus_dir.clone(),
                source: e,
            })?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "md") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Extract goal/task and response from session-log documents
                    if let Some((task, response)) = extract_task_response(&content) {
                        let example = serde_json::json!({
                            "messages": [
                                {"role": "user", "content": task},
                                {"role": "assistant", "content": response}
                            ]
                        });
                        writeln!(file, "{}", example).map_err(|e| AutoError::Io {
                            path: self.training_data_path.clone(),
                            source: e,
                        })?;
                        count += 1;
                    }
                }
            }
        }

        tracing::info!(
            count,
            path = %self.training_data_path.display(),
            "prepared training data from session corpus"
        );
        Ok(count)
    }
}

/// Statistics about the session-log corpus.
#[derive(Debug, Default)]
pub struct SessionCorpusStats {
    pub documents: usize,
    pub total_words: usize,
    pub total_bytes: usize,
}

/// Extract task and response from a session-log document.
fn extract_task_response(content: &str) -> Option<(String, String)> {
    // Look for "## Goal" section as the task
    let goal_start = content.find("## Goal")?;
    let goal_content = &content[goal_start..];
    let goal_end = goal_content[7..]
        .find("##")
        .map(|i| i + 7)
        .unwrap_or(goal_content.len().min(500));
    let task = goal_content[7..goal_end].trim().to_string();

    if task.is_empty() {
        return None;
    }

    // Use the rest of the document (or a summary section) as the response
    let response_start = content.find("## Summary")
        .or_else(|| content.find("## Solution"))
        .or_else(|| content.find("## Result"))
        .unwrap_or(goal_end + goal_start);

    let response = content[response_start..].trim();
    if response.len() < 100 {
        return None;
    }

    // Truncate very long responses
    let response = if response.len() > 8000 {
        &response[..8000]
    } else {
        response
    };

    Some((task, response.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_task_response_basic() {
        // Create content with at least 100 chars in response
        let content = r#"
# Session Title

## Goal

Fix the deployment bug in the API.

## Summary

The bug was caused by a missing environment variable. Fixed by adding DEFAULT_PORT to the configuration file. This ensures the server starts correctly on port 8080 even when the PORT env var is not set. Additional logging was added to help diagnose similar issues in the future.
"#;
        let result = extract_task_response(content);
        assert!(result.is_some());
        let (task, _response) = result.unwrap();
        assert!(task.contains("Fix the deployment bug"));
    }
}
