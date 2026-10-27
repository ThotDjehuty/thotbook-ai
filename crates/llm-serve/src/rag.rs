//! RAG middleware: retrieves relevant context before generation.

use std::path::{Path, PathBuf};

use crate::error::ServeError;

/// Configuration for RAG integration.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Path to the RAG index directory.
    pub index_dir: PathBuf,
    /// Number of context chunks to retrieve.
    pub top_k: usize,
    /// Maximum context tokens to inject.
    pub max_context_tokens: usize,
    /// Whether RAG is enabled.
    pub enabled: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            index_dir: PathBuf::from("data/rag"),
            top_k: 5,
            max_context_tokens: 2048,
            enabled: false,
        }
    }
}

/// A retrieved context chunk.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    pub text: String,
    pub source: String,
    pub score: f32,
}

/// RAG context retrieval engine.
pub struct RagEngine {
    chunks: Vec<ContextChunk>,
    config: RagConfig,
}

impl RagEngine {
    pub fn new(config: RagConfig) -> Self {
        Self {
            chunks: Vec::new(),
            config,
        }
    }

    /// Load context from a directory of text files.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<(), ServeError> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                if let Ok(text) = std::fs::read_to_string(path) {
                    self.chunks.push(ContextChunk {
                        text,
                        source: path.to_string_lossy().to_string(),
                        score: 1.0,
                    });
                }
            }
        }

        tracing::info!(count = self.chunks.len(), "loaded RAG context");
        Ok(())
    }

    /// Retrieve relevant context for a query (simple keyword matching).
    pub fn retrieve(&self, query: &str) -> Vec<ContextChunk> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let text_lower = chunk.text.to_lowercase();
                let score = query_words
                    .iter()
                    .filter(|w| text_lower.contains(*w))
                    .count() as f32
                    / query_words.len().max(1) as f32;
                (i, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(self.config.top_k)
            .map(|(i, score)| ContextChunk {
                text: self.chunks[i].text.clone(),
                source: self.chunks[i].source.clone(),
                score,
            })
            .collect()
    }

    /// Assemble context string for prompt augmentation.
    pub fn assemble_context(&self, query: &str) -> String {
        let chunks = self.retrieve(query);
        if chunks.is_empty() {
            return String::new();
        }

        let mut context = String::from("## Retrieved Context\n\n");
        for (i, chunk) in chunks.iter().enumerate() {
            context.push_str(&format!(
                "### Source {} ({})\n{}\n\n",
                i + 1,
                chunk.source,
                &chunk.text[..chunk.text.len().min(500)]
            ));
        }
        context
    }
}

/// Middleware that augments prompts with RAG context.
pub fn augment_prompt_with_context(
    system_prompt: &str,
    user_query: &str,
    rag_engine: &RagEngine,
) -> String {
    let context = rag_engine.assemble_context(user_query);

    if context.is_empty() {
        format!("{system_prompt}\n\n{user_query}")
    } else {
        format!("{system_prompt}\n\n{context}\n\n## User Query\n{user_query}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rag_retrieve_basic() {
        let mut engine = RagEngine::new(RagConfig::default());
        engine.chunks.push(ContextChunk {
            text: "quantum mechanics wave function".to_string(),
            source: "test.txt".to_string(),
            score: 1.0,
        });
        engine.chunks.push(ContextChunk {
            text: "classical mechanics Newton".to_string(),
            source: "test2.txt".to_string(),
            score: 1.0,
        });

        let results = engine.retrieve("quantum");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("quantum"));
    }
}
