//! LaTeX source extraction.
//!
//! Courses and papers in this workspace are written directly in LaTeX, so
//! ingesting only the compiled PDFs loses the structure that makes them useful
//! as training material: theorem and proof boundaries, exercise/solution pairs,
//! and the equations themselves.
//!
//! This extractor keeps the semantic text and discards the typesetting. It is
//! deliberately not a full TeX parser — that would require expanding macros —
//! but a pragmatic stripper that preserves what a reader would read aloud.

use std::path::Path;

use crate::error::IngestError;

/// Environments whose body is presentation rather than content.
const DROP_ENVIRONMENTS: &[&str] = &[
    "tikzpicture", "pgfplots", "axis", "loglogaxis", "semilogyaxis",
    "lstlisting", "verbatim", "figure", "table", "tabular", "thebibliography",
];

/// Commands whose *argument* should be kept, brace stripped.
const UNWRAP_COMMANDS: &[&str] = &[
    "section", "subsection", "subsubsection", "chapter", "paragraph",
    "textbf", "textit", "emph", "texttt", "title", "author", "caption",
    "item", "label", "text",
];

/// Parse a LaTeX file and extract its readable text content.
pub fn extract_latex(path: &Path) -> Result<String, IngestError> {
    let raw = std::fs::read_to_string(path).map_err(|e| IngestError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(strip_latex(&raw))
}

/// Strip LaTeX markup, keeping prose, theorem statements and inline maths.
pub fn strip_latex(src: &str) -> String {
    let mut out = String::with_capacity(src.len() / 2);

    for line in src.lines() {
        let line = strip_comment(line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        // Preamble noise.
        if trimmed.starts_with("\\usepackage")
            || trimmed.starts_with("\\documentclass")
            || trimmed.starts_with("\\newcommand")
            || trimmed.starts_with("\\renewcommand")
            || trimmed.starts_with("\\newtcolorbox")
            || trimmed.starts_with("\\definecolor")
            || trimmed.starts_with("\\geometry")
            || trimmed.starts_with("\\pgfplotsset")
            || trimmed.starts_with("\\lstset")
            || trimmed.starts_with("\\hypersetup")
            || trimmed.starts_with("\\addplot")
            || trimmed.starts_with("\\input")
        {
            continue;
        }
        out.push_str(&clean_line(trimmed));
        out.push('\n');
    }

    drop_environments(&out)
}

/// Remove `%` comments, honouring the `\%` escape.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && (i == 0 || bytes[i - 1] != b'\\') {
            return &line[..i];
        }
        i += 1;
    }
    line
}

/// Unwrap known commands and drop the rest, keeping their brace contents.
fn clean_line(line: &str) -> String {
    let mut s = line.to_string();

    // \begin{env} / \end{env} become markers the environment pass can see.
    if let Some(env) = s.strip_prefix("\\begin{") {
        if let Some(end) = env.find('}') {
            return format!("<<BEGIN:{}>>", &env[..end]);
        }
    }
    if let Some(env) = s.strip_prefix("\\end{") {
        if let Some(end) = env.find('}') {
            return format!("<<END:{}>>", &env[..end]);
        }
    }

    for cmd in UNWRAP_COMMANDS {
        let pat = format!("\\{}{{", cmd);
        while let Some(pos) = s.find(&pat) {
            let after = pos + pat.len();
            if let Some(close) = matching_brace(&s, after) {
                let inner = s[after..close].to_string();
                s.replace_range(pos..=close, &inner);
            } else {
                break;
            }
        }
    }

    // Remaining control sequences: keep maths operators readable, drop the rest.
    let mut cleaned = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let mut name = String::new();
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphabetic() {
                    name.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if name.is_empty() {
                chars.next(); // escaped punctuation such as \& or \%
            } else {
                // Greek letters and operators read as words; keep the name.
                cleaned.push(' ');
                cleaned.push_str(&name);
                cleaned.push(' ');
            }
        } else if c == '{' || c == '}' || c == '$' {
            cleaned.push(' ');
        } else {
            cleaned.push(c);
        }
    }

    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Index of the `}` matching an opening brace whose content starts at `from`.
fn matching_brace(s: &str, from: usize) -> Option<usize> {
    let mut depth = 1usize;
    for (i, c) in s[from..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(from + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Drop the bodies of presentation environments, keep everything else.
fn drop_environments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skip_depth = 0usize;

    for line in text.lines() {
        if let Some(env) = line.strip_prefix("<<BEGIN:").and_then(|l| l.strip_suffix(">>")) {
            if DROP_ENVIRONMENTS.contains(&env) || skip_depth > 0 {
                skip_depth += 1;
                continue;
            }
            continue;
        }
        if let Some(_env) = line.strip_prefix("<<END:").and_then(|l| l.strip_suffix(">>")) {
            skip_depth = skip_depth.saturating_sub(1);
            continue;
        }
        if skip_depth == 0 {
            out.push_str(line);
            out.push('\n');
        }
    }

    // Collapse runs of blank lines.
    let mut result = String::with_capacity(out.len());
    let mut blank = false;
    for line in out.lines() {
        if line.trim().is_empty() {
            if !blank {
                result.push('\n');
            }
            blank = true;
        } else {
            result.push_str(line);
            result.push('\n');
            blank = false;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_prose_and_structure() {
        let src = r"\section{Analyse}
Le théorème de Stokes s'écrit $\int_M d\omega = \int_{\partial M}\omega$.
% un commentaire ignoré
\textbf{Important} : la forme doit être lisse.";
        let out = strip_latex(src);
        assert!(out.contains("Analyse"), "heading lost: {out}");
        assert!(out.contains("Stokes"), "prose lost: {out}");
        assert!(out.contains("Important"), "bold text lost: {out}");
        assert!(!out.contains("commentaire"), "comment kept: {out}");
        assert!(!out.contains("textbf"), "command name kept: {out}");
    }

    #[test]
    fn drops_presentation_environments() {
        let src = r"Avant.
\begin{tikzpicture}
\draw (0,0) -- (1,1);
\end{tikzpicture}
Après.";
        let out = strip_latex(src);
        assert!(out.contains("Avant"), "{out}");
        assert!(out.contains("Après"), "{out}");
        assert!(!out.contains("draw"), "tikz body kept: {out}");
    }

    #[test]
    fn keeps_exercise_and_solution_text() {
        let src = r"\begin{exobox}[ --- calcul]
Calculer $\Gamma(1/2)$.
\end{exobox}
\begin{corrbox}
Par réflexion, $\Gamma(1/2)=\sqrt{\pi}$.
\end{corrbox}";
        let out = strip_latex(src);
        assert!(out.contains("Calculer"), "exercise lost: {out}");
        assert!(out.contains("réflexion"), "solution lost: {out}");
    }

    #[test]
    fn strips_preamble_noise() {
        let src = "\\documentclass{article}\n\\usepackage{amsmath}\nTexte réel.";
        let out = strip_latex(src);
        assert!(out.contains("Texte"), "{out}");
        assert!(!out.contains("amsmath"), "{out}");
    }

    #[test]
    fn handles_escaped_percent() {
        let out = strip_latex(r"Une erreur de 160\% sur la mesure.");
        assert!(out.contains("160"), "{out}");
        assert!(out.contains("mesure"), "escape truncated the line: {out}");
    }
}
