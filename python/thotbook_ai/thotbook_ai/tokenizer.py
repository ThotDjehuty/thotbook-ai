"""Tokenizer helpers.

Mirrors ``crates/llm-tokenize``: loads a HuggingFace-style ``tokenizer.json``
(``data/tokenizers/qwen3-tokenizer.json`` by default) via the ``tokenizers``
Python package.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from .corpus import project_root

DEFAULT_TOKENIZER_PATH = project_root() / "data" / "tokenizers" / "qwen3-tokenizer.json"


def load_tokenizer(path: str | Path | None = None) -> Tokenizer:
    """Load a ``tokenizers.Tokenizer`` from ``path``.

    Defaults to ``data/tokenizers/qwen3-tokenizer.json`` in the workspace.
    """
    tok_path = Path(path) if path else DEFAULT_TOKENIZER_PATH
    if not tok_path.is_file():
        raise FileNotFoundError(
            f"tokenizer file not found: {tok_path} "
            f"(expected data/tokenizers/qwen3-tokenizer.json)"
        )
    return Tokenizer.from_file(str(tok_path))


def tokenize_doc(text: str, tokenizer: Tokenizer) -> list[int]:
    """Tokenize a document into a list of token ids."""
    return tokenizer.encode(text).ids
