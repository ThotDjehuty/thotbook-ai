"""Lakehouse connector for the thotbook-ai Rust pipeline.

Wraps the compiled ``llm-corpus`` binary (subprocess) and reads the resulting
Polarway/Delta tables (stored as parquet under ``data/lakehouse/datasets.*``)
directly with polars.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pyarrow.parquet as pa_parquet

# Tables managed by the Rust ``llm-corpus`` crate (crates/llm-corpus/src/schema.rs).
KNOWN_TABLES = [
    "datasets.sessions",
    "datasets.corpus",
    "datasets.equations",
    "datasets.citations",
]


def project_root() -> Path:
    """Absolute path of the thotbook-ai workspace (walk up to Cargo.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").is_file():
            return parent
    return here.parents[3]


DEFAULT_LAKEHOUSE_DIR = project_root() / "data" / "lakehouse"


class CorpusClient:
    """Read/ingest access to the thotbook-ai lakehouse."""

    def __init__(self, lakehouse_dir: str | Path | None = None) -> None:
        self.lakehouse_dir = Path(lakehouse_dir) if lakehouse_dir else DEFAULT_LAKEHOUSE_DIR
        if not self.lakehouse_dir.is_dir():
            raise FileNotFoundError(f"lakehouse dir not found: {self.lakehouse_dir}")

    # ─── discovery ───

    def tables(self) -> list[str]:
        """List the ``datasets.*`` Delta tables present in the lakehouse."""
        return sorted(
            p.name
            for p in self.lakehouse_dir.iterdir()
            if p.is_dir() and p.name.startswith("datasets.")
        )

    # ─── reads ───

    def _table_dir(self, table: str) -> Path:
        table_dir = self.lakehouse_dir / table
        if not table_dir.is_dir():
            raise FileNotFoundError(
                f"no lakehouse table at {table_dir} (tables: {', '.join(self.tables())})"
            )
        return table_dir

    def _parts(self, table: str) -> list[Path]:
        table_dir = self._table_dir(table)
        parts = sorted(table_dir.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no part-*.parquet files under {table_dir}")
        return parts

    def read(self, table: str) -> pl.DataFrame:
        """Read all ``part-*.parquet`` files of a table into a polars DataFrame.

        Uses ``pl.scan_parquet`` + ``collect`` for lazy, multithreaded reads.
        """
        parts = [str(p) for p in self._parts(table)]
        return pl.scan_parquet(parts).collect()

    def stats(self, table: str) -> dict[str, int]:
        """Return ``{rows, columns, bytes}`` for a table (from parquet footers)."""
        parts = self._parts(table)
        rows = 0
        columns = 0
        total_bytes = 0
        for part in parts:
            total_bytes += part.stat().st_size
            pf = pa_parquet.ParquetFile(part)
            rows += pf.metadata.num_rows
            if columns == 0:
                columns = len(pf.schema_arrow.names)
        return {"rows": rows, "columns": columns, "bytes": total_bytes}

    # ─── ingest ───

    def _find_binary(self) -> Path:
        """Locate the compiled ``llm-corpus`` binary (release preferred)."""
        root = project_root()
        candidates = [
            root / "target" / "release" / "llm-corpus",
            root / "target" / "debug" / "llm-corpus",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "llm-corpus binary not found (looked in target/release and target/debug). "
            "Build it first: cd thotbook-ai && cargo build --release"
        )

    def ingest(
        self,
        historia_db: str | Path,
        thotbook_db: str | Path,
        lakehouse_dir: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the compiled ``llm-corpus`` ingest against the source SQLite DBs.

        Returns the ``subprocess.CompletedProcess``; inspect ``.returncode`` /
        ``.stdout`` / ``.stderr``. Raises ``RuntimeError`` if the binary is
        missing (hint: ``cargo build --release``).
        """
        binary = self._find_binary()
        target = Path(lakehouse_dir) if lakehouse_dir else self.lakehouse_dir
        cmd = [
            str(binary),
            "ingest",
            "--historia",
            str(historia_db),
            "--thotbook",
            str(thotbook_db),
            "--lakehouse",
            str(target),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
