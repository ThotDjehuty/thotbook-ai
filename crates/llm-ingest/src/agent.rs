use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::IngestError;
use crate::markdown::strip_frontmatter;

/// Parsed agent definition from a `.agent.md` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDoc {
    pub name: String,
    pub description: String,
    pub tools: Vec<String>,
    pub system_prompt: String,
    pub source_path: String,
}

/// Parse an agent `.agent.md` file into a structured definition.
pub fn parse_agent_file(path: &Path) -> Result<AgentDoc, IngestError> {
    let raw = std::fs::read_to_string(path).map_err(|e| IngestError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let (frontmatter, body) = strip_frontmatter(&raw);
    let fm = frontmatter.ok_or_else(|| IngestError::Agent {
        path: path.to_path_buf(),
        reason: "missing YAML frontmatter".to_string(),
    })?;

    let name = extract_yaml_field(&fm, "name").unwrap_or_else(|| {
        path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .replace(".agent", "")
    });
    let description = extract_yaml_field(&fm, "description").unwrap_or_default();
    let tools_str = extract_yaml_field(&fm, "tools").unwrap_or_default();
    let tools: Vec<String> = tools_str
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(AgentDoc {
        name,
        description,
        tools,
        system_prompt: body.to_string(),
        source_path: path.to_string_lossy().to_string(),
    })
}

/// Extract a simple YAML field value (key: "value" or key: value).
fn extract_yaml_field(yaml: &str, field: &str) -> Option<String> {
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&format!("{field}:")) {
            let rest = &trimmed[field.len()..];
            // Skip the colon and any whitespace
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
    use std::io::Write;

    #[test]
    fn parse_agent_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "agent-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.agent.md");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            "---\ndescription: \"Test agent\"\nname: \"TestAgent\"\ntools: [read, edit]\n---\n\nYou are a test agent."
        )
        .unwrap();

        let agent = parse_agent_file(&path).unwrap();
        assert_eq!(agent.name, "TestAgent");
        assert_eq!(agent.tools, vec!["read", "edit"]);
        assert!(agent.system_prompt.contains("test agent"));

        std::fs::remove_dir_all(dir).unwrap();
    }
}
