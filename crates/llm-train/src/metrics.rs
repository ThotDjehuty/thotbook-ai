use serde::{Deserialize, Serialize};

/// Training metrics logged during a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub step: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub perplexity: f32,
    pub tokens_per_second: f32,
}

/// Metrics logger that writes to a JSONL file.
pub struct MetricsLogger {
    path: std::path::PathBuf,
}

impl MetricsLogger {
    pub fn new(path: &std::path::Path) -> Self {
        Self { path: path.to_path_buf() }
    }

    pub fn log(&self, metrics: &TrainingMetrics) {
        let line = serde_json::to_string(metrics).unwrap_or_default();
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Compute perplexity from cross-entropy loss.
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}
