use std::path::Path;

use crate::error::IngestError;

/// Parse a Markdown file and extract structured text content.
pub fn extract_markdown(path: &Path) -> Result<String, IngestError> {
    let raw = std::fs::read_to_string(path).map_err(|e| IngestError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let parser = pulldown_cmark::Parser::new(&raw);
    let mut text = String::new();

    for event in parser {
        match event {
            pulldown_cmark::Event::Text(t) => text.push_str(&t),
            pulldown_cmark::Event::Code(c) => {
                text.push(' ');
                text.push_str(&c);
                text.push(' ');
            }
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Heading { .. }) => text.push('\n'),
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Heading(_)) => text.push('\n'),
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph) => text.push('\n'),
            _ => {}
        }
    }

    Ok(text)
}

/// Strip YAML frontmatter from markdown content, returning (frontmatter, body).
pub fn strip_frontmatter(content: &str) -> (Option<String>, &str) {
    if let Some(rest) = content.strip_prefix("---") {
        if let Some(end) = rest.find("---") {
            let fm = &rest[..end];
            let body = &rest[end + 3..];
            return (Some(fm.trim().to_string()), body.trim());
        }
    }
    (None, content.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_frontmatter_present() {
        let input = "---\nname: test\n---\n# Body";
        let (fm, body) = strip_frontmatter(input);
        assert!(fm.is_some());
        assert_eq!(body, "# Body");
    }

    #[test]
    fn strip_frontmatter_absent() {
        let input = "# No frontmatter";
        let (fm, body) = strip_frontmatter(input);
        assert!(fm.is_none());
        assert_eq!(body, "# No frontmatter");
    }
}
