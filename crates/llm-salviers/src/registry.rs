use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::SalviersError;

/// A loaded agent definition with its system prompt and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    pub tools: Vec<String>,
    pub system_prompt: String,
    pub source_path: String,
    /// Pre-computed description embedding for semantic matching (384-dim).
    pub description_embedding: Vec<f32>,
}

/// Registry of all available agents.
pub struct AgentRegistry {
    pub agents: Vec<AgentDef>,
}

impl AgentRegistry {
    /// Load all `.agent.md` files from a directory.
    pub fn load_from_dir(dir: &Path) -> Result<Self, SalviersError> {
        let mut agents = Vec::new();

        if !dir.is_dir() {
            return Err(SalviersError::Io {
                path: dir.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "agent directory not found",
                ),
            });
        }

        for entry in std::fs::read_dir(dir).map_err(|e| SalviersError::Io {
            path: dir.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| SalviersError::Io {
                path: dir.to_path_buf(),
                source: e,
            })?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("agent"))
            {
                match Self::parse_agent_file(&path) {
                    Ok(agent) => agents.push(agent),
                    Err(e) => {
                        tracing::warn!("failed to parse {}: {}", path.display(), e);
                    }
                }
            }
        }

        tracing::info!(count = agents.len(), "loaded agents from {}", dir.display());
        Ok(Self { agents })
    }

    /// Parse a single agent `.agent.md` file.
    fn parse_agent_file(path: &Path) -> Result<AgentDef, SalviersError> {
        let raw = std::fs::read_to_string(path).map_err(|e| SalviersError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let (frontmatter, body) = strip_frontmatter(&raw);
        let fm = frontmatter.ok_or_else(|| SalviersError::Parse {
            path: path.to_path_buf(),
            reason: "missing YAML frontmatter".to_string(),
        })?;

        let name = extract_yaml_field(&fm, "name").unwrap_or_else(|| {
            path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .replace(".agent", "")
                .replace(".md", "")
        });

        let description = extract_yaml_field(&fm, "description").unwrap_or_default();
        let tools_str = extract_yaml_field(&fm, "tools").unwrap_or_default();
        let tools: Vec<String> = tools_str
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Simple TF-IDF embedding for the description
        let embedding = embed_description(&description);

        Ok(AgentDef {
            name,
            description,
            tools,
            system_prompt: body.to_string(),
            source_path: path.to_string_lossy().to_string(),
            description_embedding: embedding,
        })
    }

    /// Find the best-matching agent for a query using cosine similarity.
    pub fn dispatch(&self, query_embedding: &[f32]) -> Option<&AgentDef> {
        if self.agents.is_empty() {
            return None;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, agent) in self.agents.iter().enumerate() {
            let sim = cosine_similarity(query_embedding, &agent.description_embedding);
            if sim > best_score {
                best_score = sim;
                best_idx = i;
            }
        }

        Some(&self.agents[best_idx])
    }

    /// Find the best-matching agent by simple keyword overlap (fallback).
    pub fn dispatch_keywords(&self, query: &str) -> Option<&AgentDef> {
        if self.agents.is_empty() {
            return None;
        }

        let query_words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let mut best_score = 0;
        let mut best_idx = 0;

        for (i, agent) in self.agents.iter().enumerate() {
            let desc_words: Vec<String> = agent
                .description
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let score = query_words
                .iter()
                .filter(|w| desc_words.contains(w))
                .count();
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        if best_score > 0 {
            Some(&self.agents[best_idx])
        } else {
            None
        }
    }
}

/// Simple TF-based embedding for agent descriptions (384-dim).
fn embed_description(text: &str) -> Vec<f32> {
    let mut embedding = vec![0.0f32; 384];
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Hash each word to a dimension index
        let hash = word.bytes().fold(0usize, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as usize)
        });
        let dim = hash % 384;
        let sign = if hash % 2 == 0 { 1.0 } else { -1.0 };
        embedding[dim] += sign / (i as f32 + 1.0);
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Cosine similarity between two L2-normalized vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Strip YAML frontmatter, returning (frontmatter, body).
fn strip_frontmatter(content: &str) -> (Option<String>, &str) {
    if let Some(rest) = content.strip_prefix("---") {
        if let Some(end) = rest.find("---") {
            let fm = &rest[..end];
            let body = &rest[end + 3..];
            return (Some(fm.trim().to_string()), body.trim());
        }
    }
    (None, content.trim())
}

/// Extract a simple YAML field value.
fn extract_yaml_field(yaml: &str, field: &str) -> Option<String> {
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&format!("{field}:")) {
            let rest = &trimmed[field.len()..];
            let val = rest.trim_start_matches(':').trim();
            let val = val.trim_matches('"').trim_matches('\'');
            return Some(val.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_returns_closest() {
        let dir = std::env::temp_dir().join(format!(
            "salviers-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // Write two agent files
        std::fs::write(
            dir.join("writer.agent.md"),
            "---\ndescription: \"Research paper writer for LaTeX\"\nname: \"Writer\"\ntools: [read]\n---\nWrite papers.",
        )
        .unwrap();
        std::fs::write(
            dir.join("coder.agent.md"),
            "---\ndescription: \"Rust code developer\"\nname: \"Coder\"\ntools: [edit]\n---\nWrite code.",
        )
        .unwrap();

        let registry = AgentRegistry::load_from_dir(&dir).unwrap();
        assert_eq!(registry.agents.len(), 2);

        // Keyword dispatch
        let agent = registry.dispatch_keywords("write a research paper in LaTeX").unwrap();
        assert_eq!(agent.name, "Writer");

        let agent = registry.dispatch_keywords("implement a Rust function").unwrap();
        assert_eq!(agent.name, "Coder");

        std::fs::remove_dir_all(dir).unwrap();
    }
}
