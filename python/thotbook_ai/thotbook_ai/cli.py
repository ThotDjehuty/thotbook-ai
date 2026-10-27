"""``python -m thotbook_ai`` entry point.

Subcommands:
    tables                          list lakehouse datasets.* tables
    stats  --table <name>           show rows/columns/bytes for a table
    ingest --historia <db> --thotbook <db>   run llm-corpus ingest
    chat   <prompt>                  ask the local open-source model
    health                           check the local inference server
    models                           list served models
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .chat import DEFAULT_BASE_URL, DEFAULT_MODEL, ChatMessage, ChatClient
from .corpus import CorpusClient


def _print(label: str, value: object) -> None:
    print(f"{label:<18}: {value}")


def cmd_tables(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    tables = client.tables()
    _print("mode", "RÉEL (lakehouse parquet present)")
    _print("lakehouse", client.lakehouse_dir)
    _print("tables", len(tables))
    for table in tables:
        print(f"  - {table}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    stats = client.stats(args.table)
    _print("mode", "RÉEL (lakehouse parquet present)")
    _print("table", args.table)
    _print("rows", stats["rows"])
    _print("columns", stats["columns"])
    _print("bytes", stats["bytes"])
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    result = client.ingest(args.historia, args.thotbook, lakehouse_dir=args.lakehouse)
    _print("mode", "RÉEL (SQLite sources present)")
    _print("returncode", result.returncode)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def cmd_health(args: argparse.Namespace) -> int:
    with ChatClient(args.base_url, model=args.model, timeout=args.timeout) as client:
        _print("mode", "RÉEL (local llm-serve inference)")
        _print("status", "ok" if client.is_online() else "offline")
        _print("base_url", client.base_url)
        _print("models", ", ".join(client.models()) if client.is_online() else "(server offline)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    with ChatClient(args.base_url, model=args.model, timeout=args.timeout) as client:
        if not client.is_online():
            print(f"server offline: {client.root_url}")
            return 1
        models = client.models()
        _print("mode", "RÉEL (local llm-serve inference)")
        _print("models", len(models))
        for model in models:
            print(f"  - {model}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    with ChatClient(args.base_url, model=args.model, timeout=args.timeout) as client:
        if not client.is_online():
            print(f"server offline: {client.root_url} (start it with `llm-serve` or `make up-llm`)")
            return 1

        messages = [ChatMessage(role="user", content=args.prompt)]
        if args.system:
            messages.insert(0, ChatMessage(role="system", content=args.system))

        if args.verbose:
            _print("mode", "RÉEL (local llm-serve inference)")
            _print("model", args.model)
            _print("messages", len(messages))
            _print("max_tokens", args.max_tokens)

        response = client.complete(
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(response.text)
        if args.verbose:
            print()
            _print("generated_tokens", response.completion_tokens)
            _print("prompt_tokens", response.prompt_tokens)
            _print("created_unix", response.created)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thotbook_ai",
        description="thotbook-ai Python connectors (Rust pipeline).",
    )
    parser.add_argument("--version", action="version", version=f"thotbook-ai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tables = sub.add_parser("tables", help="list lakehouse datasets.* tables")
    p_tables.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_tables.set_defaults(func=cmd_tables)

    p_stats = sub.add_parser("stats", help="show rows/columns/bytes for a lakehouse table")
    p_stats.add_argument("--table", default="datasets.sessions", help="table name (default: datasets.sessions)")
    p_stats.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_stats.set_defaults(func=cmd_stats)

    p_ingest = sub.add_parser("ingest", help="run llm-corpus ingest from SQLite sources into the lakehouse")
    p_ingest.add_argument("--historia", required=True, help="path to historia.db (opencode session documents)")
    p_ingest.add_argument("--thotbook", required=True, help="path to thotbook.db (research index)")
    p_ingest.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_health = sub.add_parser("health", help="check the local inference server health")
    _add_llm_args(p_health)
    p_health.set_defaults(func=cmd_health)

    p_models = sub.add_parser("models", help="list models served by the local inference server")
    _add_llm_args(p_models)
    p_models.set_defaults(func=cmd_models)

    p_chat = sub.add_parser("chat", help="ask the local open-source model (llm-serve)")
    p_chat.add_argument("prompt", help="user prompt")
    p_chat.add_argument("--system", default=None, help="optional system prompt")
    p_chat.add_argument("--max-tokens", type=int, default=None, help="generation budget (server default)")
    p_chat.add_argument("--temperature", type=float, default=None, help="sampling temperature (server default)")
    p_chat.add_argument("--top-p", type=float, default=None, help="nucleus sampling (server default)")
    _add_llm_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    return parser


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"llm-serve base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=float, default=600.0, help="request timeout in seconds (default: 600)")
    parser.add_argument("--verbose", action="store_true", help="print request/response metadata")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
