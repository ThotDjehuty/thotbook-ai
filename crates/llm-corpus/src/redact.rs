//! Secret redaction for corpus ingestion.
//!
//! Faithful Rust port of `ThotCloud/services/thotbook-research-mcp/src/redact.ts`
//! so that session/corpus content persisted to the lakehouse is never
//! exfiltratable through the search layer. Idempotent: the `[REDACTED]`
//! placeholder never re-triggers a pattern (mirrors the TS lookahead).

use std::sync::OnceLock;

use fancy_regex::{Regex, RegexBuilder};

/// Placeholder substituted for every detected credential.
pub const REDACTED: &str = "[REDACTED]";

/// Backtracking budget for lazy block patterns on very large documents
/// (a single session transcript can exceed the default 1M limit).
const BACKTRACK_LIMIT: usize = 50_000_000;

struct Pattern {
    re: Regex,
    replacement: &'static str,
}

fn build(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .backtrack_limit(BACKTRACK_LIMIT)
        .build()
        .expect("static redaction regex is valid")
}

fn patterns() -> &'static Vec<Pattern> {
    static PATTERNS: OnceLock<Vec<Pattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // API keys (sk-…, pk-…, acct-…)
            Pattern {
                re: build(r"\b(sk|pk|rk|acct|ak)-[A-Za-z0-9_-]{16,}\b"),
                replacement: REDACTED,
            },
            // AWS access keys
            Pattern {
                re: build(r"\b(AKIA|ASIA|AGPA|AROA)[0-9A-Z]{16}\b"),
                replacement: REDACTED,
            },
            // GitHub tokens
            Pattern {
                re: build(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
                replacement: REDACTED,
            },
            // Slack tokens
            Pattern {
                re: build(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
                replacement: REDACTED,
            },
            // PEM / SSH / PGP private keys (block forms)
            Pattern {
                re: build(
                    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                ),
                replacement: REDACTED,
            },
            Pattern {
                re: build(
                    r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----",
                ),
                replacement: REDACTED,
            },
            Pattern {
                re: build(
                    r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----",
                ),
                replacement: REDACTED,
            },
            // JWT / bearer tokens
            Pattern {
                re: build(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
                replacement: REDACTED,
            },
            Pattern {
                re: build(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=]{16,}\b"),
                replacement: REDACTED,
            },
            // Generic name=secret assignments (skip already-redacted placeholders)
            Pattern {
                re: build(
                    r#"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|session[_-]?key|auth[_-]?token)\b\s*[:=]\s*["']?(?!\[REDACTED\]|$)[^\s"',;}{]{6,}"#,
                ),
                replacement: "$1=[REDACTED]",
            },
            // OpenAI / Anthropic / other provider keys
            Pattern {
                re: build(r"\b(sk-ant|sk-proj|sk-svcacct|sk-gcp)[A-Za-z0-9_-]{10,}\b"),
                replacement: REDACTED,
            },
            // Hex seeds / long hex
            Pattern {
                re: build(r"\b[0-9a-fA-F]{32,}\b"),
                replacement: REDACTED,
            },
        ]
    })
}

/// Mask known credential patterns in arbitrary text (idempotent).
pub fn redact_secrets(text: &str) -> String {
    patterns()
        .iter()
        .fold(text.to_string(), |out, p| {
            // fancy-regex: replacement strings here are plain literal
            // placeholders; a backreference error cannot occur.
            p.re.replace_all(&out, p.replacement).into_owned()
        })
}

/// True if the given text still contains a credential-like pattern.
pub fn contains_secrets(text: &str) -> bool {
    patterns()
        .iter()
        .any(|p| p.re.is_match(text).unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_api_key() {
        assert_eq!(redact_secrets("key sk-abc12345678901234567890 end"), "key [REDACTED] end");
    }

    #[test]
    fn masks_assignment_and_is_idempotent() {
        let once = redact_secrets("token=abc123");
        assert_eq!(once, "token=[REDACTED]");
        assert_eq!(redact_secrets(&once), once, "idempotent");
    }

    #[test]
    fn masks_jwt() {
        let s = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        assert!(!contains_secrets(&redact_secrets(s)));
    }

    #[test]
    fn keeps_french_false_positive() {
        let s = "le contexte risk-off et le signal";
        assert!(!contains_secrets(s));
    }

    #[test]
    fn masks_pem_block() {
        let s = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----";
        assert!(!contains_secrets(&redact_secrets(s)));
    }

    #[test]
    fn masks_long_hex() {
        let s = "0x9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        assert!(!contains_secrets(&redact_secrets(s)));
    }
}
