//! Research paper generation pipeline.
//!
//! Generates publication-ready LaTeX papers using only local thotbook-ai weights.

use std::path::PathBuf;

use crate::agent::AgentRegistry;
use crate::error::AutoError;
use crate::task::{Task, TaskType, OutputFormat};

/// Configuration for paper generation.
#[derive(Debug, Clone)]
pub struct PaperConfig {
    /// Inference server URL.
    pub inference_url: String,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Maximum sections.
    pub max_sections: usize,
    /// Include abstract.
    pub include_abstract: bool,
    /// Include bibliography.
    pub include_bibliography: bool,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            inference_url: "http://127.0.0.1:8100".to_string(),
            output_dir: PathBuf::from("papers"),
            max_sections: 10,
            include_abstract: true,
            include_bibliography: true,
        }
    }
}

/// Generates research papers.
pub struct PaperGenerator {
    config: PaperConfig,
    agents: AgentRegistry,
    /// Current paper being generated.
    title: String,
    topic: String,
}

impl PaperGenerator {
    pub fn new(config: PaperConfig) -> Self {
        let agents = AgentRegistry::new(&config.inference_url);
        Self {
            config,
            agents,
            title: String::new(),
            topic: String::new(),
        }
    }

    /// Generate a complete research paper on a topic.
    pub async fn generate(&mut self, topic: &str) -> Result<Paper, AutoError> {
        self.topic = topic.to_string();
        self.title = format!("A Study of {}", topic);
        tracing::info!(title = %self.title, topic = %self.topic, "generating paper");

        // Step 1: Research the topic
        let research = self.research_topic().await?;

        // Step 2: Generate outline
        let outline = self.generate_outline(&research).await?;

        // Step 3: Generate each section
        let mut sections = Vec::new();
        for section_title in &outline {
            let section = self.generate_section(section_title, &research).await?;
            sections.push(section);
        }

        // Step 4: Generate abstract
        let abstract_text = if self.config.include_abstract {
            Some(self.generate_abstract(&sections).await?)
        } else {
            None
        };

        // Step 5: Generate bibliography
        let bibliography = if self.config.include_bibliography {
            Some(self.generate_bibliography(&research).await?)
        } else {
            None
        };

        // Step 6: Compile to LaTeX
        let latex = self.compile_latex(abstract_text.as_deref(), &sections, bibliography.as_deref())?;

        // Step 7: Write output
        std::fs::create_dir_all(&self.config.output_dir).map_err(|e| AutoError::Io {
            path: self.config.output_dir.clone(),
            source: e,
        })?;

        let filename = sanitize_filename(&self.title);
        let latex_path = self.config.output_dir.join(format!("{}.tex", filename));
        std::fs::write(&latex_path, &latex).map_err(|e| AutoError::Io {
            path: latex_path.clone(),
            source: e,
        })?;

        tracing::info!(path = %latex_path.display(), "paper generated");

        Ok(Paper {
            title: self.title.clone(),
            abstract_text,
            sections,
            bibliography,
            content: latex,
            output_path: latex_path,
        })
    }

    /// Research the topic using the researcher agent.
    async fn research_topic(&self) -> Result<String, AutoError> {
        let agent = self
            .agents
            .find_agent(TaskType::Research)
            .ok_or_else(|| AutoError::Agent("no researcher agent".to_string()))?;

        let task = Task::new(
            format!("Research the topic: {}", self.topic),
            TaskType::Research,
            format!(
                "Conduct a comprehensive literature review on: {}\n\n\
                Focus on:\n\
                1. Key papers and results\n\
                2. Open problems\n\
                3. Recent developments\n\
                4. Mathematical foundations\n\n\
                Output should be well-structured with citations.",
                self.topic
            ),
        ).with_output_format(OutputFormat::Markdown);

        agent.execute(&task).await
    }

    /// Generate the paper outline.
    async fn generate_outline(&self, research: &str) -> Result<Vec<String>, AutoError> {
        let agent = self
            .agents
            .find_agent(TaskType::Writing)
            .ok_or_else(|| AutoError::Agent("no writer agent".to_string()))?;

        let task = Task::new(
            "Generate paper outline",
            TaskType::Writing,
            format!(
                "Based on this research:\n\n{}\n\n\
                Generate an outline for a research paper titled '{}' with up to {} sections.\n\n\
                Output a JSON array of section titles, e.g.:\n\
                [\"Introduction\", \"Background\", \"Main Results\", \"Applications\", \"Conclusion\"]",
                research, self.title, self.config.max_sections
            ),
        ).with_output_format(OutputFormat::Json);

        let response = agent.execute(&task).await?;

        // Parse section titles from JSON
        let sections: Vec<String> = serde_json::from_str(&response)
            .unwrap_or_else(|_| {
                // Fallback: extract from text
                vec![
                    "Introduction".to_string(),
                    "Background".to_string(),
                    "Main Results".to_string(),
                    "Discussion".to_string(),
                    "Conclusion".to_string(),
                ]
            });

        Ok(sections)
    }

    /// Generate a single section.
    async fn generate_section(&self, title: &str, research: &str) -> Result<Section, AutoError> {
        // Use mathematician for math-heavy sections, writer for others
        let task_type = if title.contains("Result") || title.contains("Theorem") || title.contains("Proof") {
            TaskType::Mathematics
        } else {
            TaskType::Writing
        };

        let agent = self
            .agents
            .find_agent(task_type)
            .ok_or_else(|| AutoError::Agent(format!("no agent for {:?}", task_type)))?;

        let task = Task::new(
            format!("Write section: {}", title),
            task_type,
            format!(
                "Write the '{}' section for a paper titled '{}'.\n\n\
                Research context:\n{}\n\n\
                Requirements:\n\
                - Write in LaTeX format\n\
                - Include equations where appropriate\n\
                - Be rigorous and precise\n\
                - Use \\label{{}} for cross-references\n\
                - Aim for 500-1500 words",
                title, self.title, &research[..research.len().min(3000)]
            ),
        ).with_output_format(OutputFormat::LaTeX);

        let content = agent.execute(&task).await?;

        Ok(Section {
            title: title.to_string(),
            content,
        })
    }

    /// Generate the abstract.
    async fn generate_abstract(&self, sections: &[Section]) -> Result<String, AutoError> {
        let agent = self
            .agents
            .find_agent(TaskType::Writing)
            .ok_or_else(|| AutoError::Agent("no writer agent".to_string()))?;

        let section_summaries: String = sections
            .iter()
            .map(|s| format!("- {}: {}", s.title, &s.content[..s.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let task = Task::new(
            "Write abstract",
            TaskType::Writing,
            format!(
                "Write an abstract for the paper '{}' with these sections:\n\n{}\n\n\
                The abstract should be 150-250 words and highlight:\n\
                1. The problem addressed\n\
                2. The approach\n\
                3. Key results\n\
                4. Significance",
                self.title, section_summaries
            ),
        ).with_output_format(OutputFormat::LaTeX);

        agent.execute(&task).await
    }

    /// Generate the bibliography.
    async fn generate_bibliography(&self, research: &str) -> Result<String, AutoError> {
        let agent = self
            .agents
            .find_agent(TaskType::Research)
            .ok_or_else(|| AutoError::Agent("no researcher agent".to_string()))?;

        let task = Task::new(
            "Generate bibliography",
            TaskType::Research,
            format!(
                "Based on this research:\n\n{}\n\n\
                Generate a BibTeX bibliography with the key references.\n\
                Include author, title, journal/arxiv, year for each entry.\n\
                Output should be valid BibTeX format.",
                &research[..research.len().min(4000)]
            ),
        ).with_output_format(OutputFormat::LaTeX);

        agent.execute(&task).await
    }

    /// Compile all parts into a complete LaTeX document.
    fn compile_latex(
        &self,
        abstract_text: Option<&str>,
        sections: &[Section],
        bibliography: Option<&str>,
    ) -> Result<String, AutoError> {
        let mut latex = String::new();

        // Preamble
        latex.push_str(LATEX_PREAMBLE);
        latex.push_str(&format!("\\title{{{}}}\n", escape_latex(&self.title)));
        latex.push_str("\\author{ThotBook-AI Autonomous Research System}\n");
        latex.push_str("\\date{\\today}\n\n");
        latex.push_str("\\begin{document}\n");
        latex.push_str("\\maketitle\n\n");

        // Abstract
        if let Some(abstract_text) = abstract_text {
            latex.push_str("\\begin{abstract}\n");
            latex.push_str(abstract_text);
            latex.push_str("\n\\end{abstract}\n\n");
        }

        // Sections
        for section in sections {
            latex.push_str(&format!("\\section{{{}}}\n", escape_latex(&section.title)));
            latex.push_str(&section.content);
            latex.push_str("\n\n");
        }

        // Bibliography
        if let Some(bib) = bibliography {
            latex.push_str("\\bibliographystyle{plain}\n");
            latex.push_str("\\begin{thebibliography}{99}\n");
            // Convert BibTeX to simple bibliography if needed
            latex.push_str(bib);
            latex.push_str("\n\\end{thebibliography}\n");
        }

        latex.push_str("\\end{document}\n");

        Ok(latex)
    }
}

/// A generated research paper.
#[derive(Debug)]
pub struct Paper {
    pub title: String,
    pub abstract_text: Option<String>,
    pub sections: Vec<Section>,
    pub bibliography: Option<String>,
    pub content: String,
    pub output_path: PathBuf,
}

/// A section of the paper.
#[derive(Debug)]
pub struct Section {
    pub title: String,
    pub content: String,
}

/// LaTeX preamble for research papers.
const LATEX_PREAMBLE: &str = r#"\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\newtheorem{remark}[theorem]{Remark}

"#;

/// Escape special LaTeX characters.
fn escape_latex(s: &str) -> String {
    s.replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('~', "\\textasciitilde{}")
        .replace('^', "\\textasciicircum{}")
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>()
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_latex_special() {
        assert_eq!(escape_latex("a & b"), "a \\& b");
        assert_eq!(escape_latex("100%"), "100\\%");
    }

    #[test]
    fn sanitize_filename_basic() {
        assert_eq!(sanitize_filename("Hello World!"), "hello_world_");
        assert_eq!(sanitize_filename("paper-2024"), "paper-2024");
    }
}
