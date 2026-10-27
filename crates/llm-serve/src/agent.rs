//! Agent context: loads agent system prompts and dispatches to the right agent.

use crate::error::ServeError;

/// Parsed agent definition.
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
}

/// Agent registry: loads and manages agent definitions.
pub struct AgentRegistry {
    agents: Vec<AgentContext>,
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Load all `.agent.md` files from a directory.
    pub fn load_from_dir(&mut self, dir: &std::path::Path) -> Result<(), ServeError> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir).map_err(|e| ServeError::Io {
            path: dir.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| ServeError::Io {
                path: dir.to_path_buf(),
                source: e,
            })?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.contains("agent") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            let (fm, body) = strip_frontmatter(&content);
                            if let Some(fm) = fm {
                                let agent_name = extract_field(&fm, "name")
                                    .unwrap_or_else(|| name.replace(".agent.md", ""));
                                let description =
                                    extract_field(&fm, "description").unwrap_or_default();

                                self.agents.push(AgentContext {
                                    name: agent_name,
                                    description,
                                    system_prompt: body.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        tracing::info!(count = self.agents.len(), "loaded agent contexts");
        Ok(())
    }

    /// Find the best matching agent for a query using keyword overlap.
    pub fn dispatch(&self, query: &str) -> Option<&AgentContext> {
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

    /// Get all loaded agents.
    pub fn agents(&self) -> &[AgentContext] {
        &self.agents
    }
}

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

fn extract_field(yaml: &str, field: &str) -> Option<String> {
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
    fn dispatch_by_keywords() {
        let mut registry = AgentRegistry::new();
        registry.agents.push(AgentContext {
            name: "Writer".to_string(),
            description: "Research paper writer for LaTeX".to_string(),
            system_prompt: "Write papers".to_string(),
        });
        registry.agents.push(AgentContext {
            name: "Coder".to_string(),
            description: "Rust code developer".to_string(),
            system_prompt: "Write code".to_string(),
        });

        let agent = registry.dispatch("write a research paper").unwrap();
        assert_eq!(agent.name, "Writer");

        let agent = registry.dispatch("implement a Rust function").unwrap();
        assert_eq!(agent.name, "Coder");
    }
}
