# thotbook-ai (Python connectors)

Thin, pragmatic Python connectors for the `thotbook-ai` Rust pipeline
(`crates/llm-corpus`, `crates/llm-tokenize`). No pyo3 build step: the
connectors shell out to the compiled Rust binaries via `subprocess` and read
the resulting lakehouse / tokenized data with **polars** (parquet / Arrow IPC).

The package also ships a `ChatClient` that talks to the **local** inference
server (`crates/llm-serve`) over its OpenAI-compatible API. Everything runs
on your own machine — an open-source (Apache 2.0) Qwen3-8B, CPU-only, with
**no API key and no billing**. Same maths, same tools, free by design.

## Install

```sh
pip install polars pyarrow requests tokenizers
# or, from this directory:
pip install -e .
```

Requires Python >= 3.10.

## CLI

From `python/thotbook_ai`:

```sh
# List the datasets.* tables present in the lakehouse (mode RÉEL).
python -m thotbook_ai tables

# Rows / columns / bytes for a table (parquet footers, no full scan).
python -m thotbook_ai stats --table datasets.sessions

# Run the compiled llm-corpus binary against the SQLite source DBs.
# (Binary auto-detected: target/release/llm-corpus, falls back to debug.
#  Build it first with: cd thotbook-ai && cargo build --release)
python -m thotbook_ai ingest --historia /path/historia.db --thotbook /path/thotbook.db

# Ask the local open-source model (llm-serve on 127.0.0.1:8100).
python -m thotbook_ai chat "Give one line of Kähler geometry intuition."
python -m thotbook_ai chat --system "You are a maths teacher." "Explain a fibration."
python -m thotbook_ai chat "..." --max-tokens 128 --temperature 0.7

# Inspect the local server.
python -m thotbook_ai health
python -m thotbook_ai models
```

All commands accept `--lakehouse <dir>` to point at a different lakehouse.
The `chat` / `health` / `models` commands accept `--base-url`, `--model`,
`--timeout` and `--verbose`.

> The server must be running first: `cd thotbook-ai && ./target/release/llm-serve`
> (base URL configurable, default `http://127.0.0.1:8100/v1` — see
> `docs/source/getting_started.rst`). If the server is offline, `chat` exits
> with code 1 and a clear message instead of hanging.

## Library

```python
from thotbook_ai import ChatClient, CorpusClient, load_tokenizer, tokenize_doc, redact_secrets

# --- Local open-source chat (no API key, no billing) ---
with ChatClient() as client:                 # base http://127.0.0.1:8100/v1, model Qwen/Qwen3-8B
    client.is_online()                       # True
    client.models()                          # ['Qwen/Qwen3-8B']
    response = client.ask(
        "Recover the metric from a Hermitian structure.",
        system="Answer as a differential geometer.",
        temperature=0.2,
    )
    response.text                           # assistant reply
    response.model                          # 'Qwen/Qwen3-8B'
    response.completion_tokens              # tokens generated

# Multi-turn conversation — pass the full history each time:
history = [
    ChatMessage(role="user", content="Define a Kähler form."),
    ChatMessage(role="assistant", content=first_round.text),
    ChatMessage(role="user", content="Now give a compact example on CP^n."),
]
second_round = client.complete(history, max_tokens=128)

# Lakehouse reads (parquet via polars)
client = CorpusClient()                      # defaults to data/lakehouse
client.tables()                              # ['datasets.citations', 'datasets.corpus', ...]
df = client.read("datasets.sessions")        # polars.DataFrame
client.stats("datasets.corpus")              # {'rows': ..., 'columns': ..., 'bytes': ...}

# Ingest via the Rust binary (returns subprocess.CompletedProcess)
res = client.ingest("historia.db", "thotbook.db")
assert res.returncode == 0, res.stderr

# Tokenization (data/tokenizers/qwen3-tokenizer.json by default)
tok = load_tokenizer()
ids = tokenize_doc("bonjour", tok)           # list[int]

# Secret redaction (parity with redact.ts / redact.rs)
redact_secrets("token=abc123")               # 'token=[REDACTED]'
contains_secrets("token=[REDACTED]")         # False
```

> `ChatMessage`/`ChatResponse` are simple dataclasses — see `tests/test_chat.py`
> for the complete request/response contract.

## References

- `optimiz-rs`: the companion Rust scientific-computing library downstream of
  this pipeline — https://github.com/ThotDjehuty/optimiz-rs
- HFThot Research Lab site (WP1–WP5 mission statement):
  https://hfthot-lab.eu/research.html (source: `hfthot-lab-strategies/website/`)
- Full docs (Sphinx / ReadTheDocs): `docs/source/` in the repository root.

## Layout

| File | Purpose |
| --- | --- |
| `thotbook_ai/corpus.py` | `CorpusClient` — discover/read/stats lakehouse tables, `ingest` via `llm-corpus` |
| `thotbook_ai/tokenizer.py` | `load_tokenizer`, `tokenize_doc` (HuggingFace `tokenizers`) |
| `thotbook_ai/redact.py` | `redact_secrets`, `contains_secrets` (Python port of `redact.ts`/`redact.rs`) |
| `thotbook_ai/chat.py` | `ChatClient`, `ChatMessage`, `ChatResponse` — local OpenAI-compatible chat |
| `thotbook_ai/cli.py` | `python -m thotbook_ai` entry point (`tables`, `stats`, `ingest`, `chat`, `health`, `models`) |
| `tests/test_chat.py` | pytest suite — mock transport + live integration (auto-skipped when server offline) |
