//! nLab corpus scanner.
//!
//! Reads the official nLab file mirror (`ncatlab/nlab-content`, git branch
//! `master`) whose page tree is sharded as
//! `pages/<a>/<b>/<c>/<d>/<page-id>/content.md` plus a sibling `name` file
//! holding the human-readable page title.
//!
//! Content is Markdown + itex2MML math (`$...$`, `\[...\]`) — kept verbatim so
//! the downstream LLM sees both the prose and the LaTeX-ish mathematics.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use walkdir::WalkDir;

use crate::error::CorpusError;

/// A single nLab page from the mirror.
#[derive(Debug, Clone)]
pub struct NLabPage {
    /// Human-readable page name (from the `name` file).
    pub name: String,
    /// Raw Markdown + itex2MML page body (`content.md`).
    pub content: String,
    /// Unique arXiv identifiers referenced by this page (new + old style).
    pub arxiv_ids: Vec<String>,
}

/// Matches new-style arXiv IDs (`1601.05956` / `2301.12345`) on word
/// boundaries and old-style IDs (`hep-th/0512197`, `math/0101023`).
fn new_style_re() -> fancy_regex::Regex {
    fancy_regex::RegexBuilder::new(r"(?<![0-9])(\d{4}\.\d{4,5})(?![0-9])")
        .build()
        .expect("static arxiv new-style regex")
}

fn old_style_re() -> fancy_regex::Regex {
    fancy_regex::RegexBuilder::new(r"(?i)\b((?:[a-z]+(?:-[a-z]+)*|math|nlin|cond-mat|physics)/\d{7})\b")
        .build()
        .expect("static arxiv old-style regex")
}

/// Extract unique arXiv identifiers from arbitrary text (both ID styles).
pub fn extract_arxiv_ids(text: &str) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for caps in new_style_re().captures_iter(text) {
        if let Ok(c) = caps {
            if let Some(m) = c.get(1) {
                set.insert(m.as_str().to_string());
            }
        }
    }
    for caps in old_style_re().captures_iter(text) {
        if let Ok(c) = caps {
            if let Some(m) = c.get(1) {
                set.insert(m.as_str().to_string());
            }
        }
    }
    set.into_iter().collect()
}

/// Scan the nLab mirror and return every page found under `pages/`.
///
/// Pages whose `content.md` or `name` file is missing/empty are skipped; a
/// `(path, error)` is logged by the caller, never fatal.
pub fn scan_pages(root: &Path) -> Result<Vec<NLabPage>, CorpusError> {
    if !root.is_dir() {
        return Err(CorpusError::Io {
            path: root.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "nLab mirror directory not found",
            ),
        });
    }

    let mut pages = Vec::new();
    for entry in WalkDir::new(root)
        .min_depth(6)
        .max_depth(6)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() || entry.file_name() != "content.md" {
            continue;
        }
        let dir = entry.path().parent().expect("content.md always has a parent dir");
        let name_file = dir.join("name");
        let name = match fs::read_to_string(&name_file) {
            Ok(n) => n.trim().to_string(),
            Err(_) => continue, // no name file → not a real page
        };
        if name.is_empty() {
            continue;
        }
        let content = match fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let arxiv_ids = extract_arxiv_ids(&content);
        pages.push(NLabPage {
            name,
            content,
            arxiv_ids,
        });
    }

    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_new_style_arxiv_ids() {
        let ids = extract_arxiv_ids("see https://arxiv.org/abs/1601.05956 and 2301.12345 v2");
        assert!(ids.contains(&"1601.05956".to_string()));
        assert!(ids.contains(&"2301.12345".to_string()));
        // no false positives on surrounding digits
        assert!(!extract_arxiv_ids("value 12345.678 is not an id").contains(&"12345.678".to_string()));
    }

    #[test]
    fn extracts_old_style_arxiv_ids() {
        let ids = extract_arxiv_ids("cf. hep-th/0512197 and math/0101023, plus pdf link");
        assert!(ids.contains(&"hep-th/0512197".to_string()));
        assert!(ids.contains(&"math/0101023".to_string()));
    }

    #[test]
    fn dedups_ids() {
        let ids = extract_arxiv_ids("1601.05956 and 1601.05956 again");
        assert_eq!(ids, vec!["1601.05956".to_string()]);
    }

    #[test]
    fn scans_mirror_layout() {
        let dir = std::env::temp_dir().join("nlab-test");
        let page = dir.join("pages/0/0/0/0/1000");
        fs::create_dir_all(&page).unwrap();
        fs::write(page.join("name"), "supermanifold\n").unwrap();
        fs::write(
            page.join("content.md"),
            "# Supermanifold\n\nSee arXiv:1601.05956.\n",
        )
        .unwrap();
        // a stub dir without a name file must be skipped
        let stub = dir.join("pages/0/0/0/1/9999");
        fs::create_dir_all(&stub).unwrap();
        fs::write(stub.join("content.md"), "no name here").unwrap();

        let pages = scan_pages(&dir).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].name, "supermanifold");
        assert!(pages[0].content.contains("1601.05956"));
        assert!(pages[0].arxiv_ids.contains(&"1601.05956".to_string()));
        fs::remove_dir_all(&dir).unwrap();
    }
}
