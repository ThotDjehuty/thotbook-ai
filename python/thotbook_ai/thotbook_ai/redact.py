"""Secret redaction for corpus ingestion.

Faithful Python port of ``ThotCloud/services/thotbook-research-mcp/src/redact.ts``
(and the Rust mirror ``crates/llm-corpus/src/redact.rs``) so parity checks can
run across all three implementations. Idempotent: the ``[REDACTED]`` placeholder
never re-triggers a pattern (mirrors the TS lookahead).
"""

from __future__ import annotations

import re

# Placeholder substituted for every detected credential.
REDACTED = "[REDACTED]"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API keys (sk-…, pk-…, acct-…)
    (re.compile(r"\b(sk|pk|rk|acct|ak)-[A-Za-z0-9_-]{16,}\b"), REDACTED),
    # AWS access keys
    (re.compile(r"\b(AKIA|ASIA|AGPA|AROA)[0-9A-Z]{16}\b"), REDACTED),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), REDACTED),
    # Slack tokens
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), REDACTED),
    # PEM / SSH / PGP private keys (block forms)
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
        REDACTED,
    ),
    (
        re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----"),
        REDACTED,
    ),
    (
        re.compile(
            r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----"
        ),
        REDACTED,
    ),
    # JWT / bearer tokens
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    (re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=]{16,}\b"), REDACTED),
    # Generic name=secret assignments (skip already-redacted placeholders)
    (
        re.compile(
            r"\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|"
            r"client[_-]?secret|session[_-]?key|auth[_-]?token)\b\s*[:=]\s*[\"']?"
            r"(?!\[REDACTED\]|$)[^\s\"',;}{]{6,}",
            re.IGNORECASE,
        ),
        r"\1=[REDACTED]",
    ),
    # OpenAI / Anthropic / other provider keys
    (re.compile(r"\b(sk-ant|sk-proj|sk-svcacct|sk-gcp)[A-Za-z0-9_-]{10,}\b"), REDACTED),
    # Hex seeds / long hex
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), REDACTED),
]


def redact_secrets(text: str) -> str:
    """Mask known credential patterns in arbitrary text (idempotent)."""
    out = text
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def contains_secrets(text: str) -> bool:
    """True if the given text still contains a credential-like pattern."""
    return any(pattern.search(text) is not None for pattern, _ in _PATTERNS)
