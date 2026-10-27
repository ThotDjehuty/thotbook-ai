#!/usr/bin/env python3
"""Build a clean historia.db from the historia session directory.

Why this exists: the historia-mcp copy at
  ThotCloud/services/historia-mcp/data/historia.db
is truncated at exactly 1,000,000,000 bytes (its .part/000000 sibling is the
same size), so `pragma integrity_check` reports "database disk image is
malformed" and llm-corpus cannot read a single row from it. The session files
themselves are intact in HISTORIA_DIR, so we rebuild the index from those.

The schema is exactly what crates/llm-corpus/src/sources.rs::read_sessions
selects:
    SELECT id, filename, date, title, content, tags, category, word_count,
           indexed_at, file_modified FROM documents ORDER BY date

Usage:
    build_historia_db.py [--src DIR] [--out PATH] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path("historia")
DEFAULT_OUT = REPO / "data" / "ingest" / "historia.db"

TEXT_SUFFIXES = {".md", ".markdown", ".log", ".txt", ".org", ".rst"}
# Session filenames usually lead with a date: 20260120_..., 2026-01-04_...
DATE_RX = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    date          TEXT NOT NULL,
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT 'session',
    word_count    INTEGER NOT NULL DEFAULT 0,
    indexed_at    TEXT NOT NULL,
    file_modified TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_date ON documents(date);
"""


def guess_date(name: str, mtime: float) -> str:
    m = DATE_RX.search(name)
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc) \
                .date().isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(mtime, timezone.utc).date().isoformat()


def guess_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:40]:
        s = line.strip()
        if s.startswith("#"):
            t = s.lstrip("#").strip()
            if t:
                return t[:300]
    return path.stem.replace("_", " ").replace("-", " ")[:300]


def category_for(path: Path) -> str:
    if path.suffix == ".log":
        return "log"
    low = path.name.lower()
    for key in ("deployment", "release", "fix", "test", "session", "mirror"):
        if key in low:
            return key
    return "session"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help=f"session directory (default: {DEFAULT_SRC})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"database to write (default: {DEFAULT_OUT})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing database")
    ap.add_argument("--min-bytes", type=int, default=64,
                    help="skip files smaller than this (default 64)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"source directory not found: {src}")
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists; pass --force to rebuild it.")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)

    kept = skipped = 0
    total_words = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []

    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            skipped += 1
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        if len(raw) < args.min_bytes:
            skipped += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        st = path.stat()
        words = len(text.split())
        total_words += words
        rel = str(path.relative_to(src))
        rows.append((
            hashlib.sha256(rel.encode()).hexdigest()[:32],
            rel,
            guess_date(path.name, st.st_mtime),
            guess_title(path, text),
            text,
            "",
            category_for(path),
            words,
            now,
            datetime.fromtimestamp(st.st_mtime, timezone.utc)
            .isoformat(timespec="seconds"),
        ))
        kept += 1

    conn.executemany(
        "INSERT OR REPLACE INTO documents (id, filename, date, title, content,"
        " tags, category, word_count, indexed_at, file_modified)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

    check = conn.execute("pragma integrity_check").fetchone()[0]
    n = conn.execute("select count(*) from documents").fetchone()[0]
    conn.execute("vacuum")
    conn.close()

    if check != "ok":
        tmp.unlink(missing_ok=True)
        sys.exit(f"refusing to publish a bad database: integrity_check = {check}")

    tmp.replace(out)
    size = out.stat().st_size

    print(f"\n  source      {src}")
    print(f"  indexed     {kept} documents ({skipped} non-text skipped)")
    print(f"  words       {total_words:,}")
    print(f"  integrity   {check}")
    print(f"  rows        {n}")
    print(f"  written     {out}  ({size / 1e6:.1f} MB)")
    print("\n  next:  make ingest-historia\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
