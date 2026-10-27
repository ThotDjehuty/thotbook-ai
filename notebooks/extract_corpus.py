"""Extract text from the HFThot-Research paper corpus into JSONL chunks.

Produces a real-document corpus for the membership-inference experiments,
replacing the toy templated sentences of the first audit notebook.

Output: notebooks/data/corpus_chunks.jsonl with one record per chunk:
    {"doc": <pdf stem>, "domain": "physics"|"finance", "chunk_id": int,
     "text": str, "n_chars": int}
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# Corpus location is environment-specific and intentionally not hard-coded:
# the source PDFs are private/third-party material that is never committed.
PAPERS = Path(os.environ.get("THOTBOOK_PAPERS_DIR", "./papers-input")).expanduser()
OUT = Path(__file__).parent / "data" / "corpus_chunks.jsonl"

CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 400
MAX_CHUNKS_PER_DOC = 40   # keep the corpus balanced across documents
MAX_PAGES = 30            # skip the tail of very long books


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_doc(text: str) -> list[str]:
    out, buf = [], []
    size = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        buf.append(sent)
        size += len(sent)
        if size >= CHUNK_CHARS:
            out.append(" ".join(buf))
            buf, size = [], 0
    if size >= MIN_CHUNK_CHARS:
        out.append(" ".join(buf))
    return out[:MAX_CHUNKS_PER_DOC]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not PAPERS.is_dir():
        print(f"corpus dir not found: {PAPERS}\n"
              f"Set THOTBOOK_PAPERS_DIR to a directory of PDFs, e.g.:\n"
              f"  THOTBOOK_PAPERS_DIR=~/my-papers python3 {Path(__file__).name}")
        return 1
    pdfs = sorted(PAPERS.rglob("*.pdf"))
    print(f"found {len(pdfs)} PDFs")

    n_written = 0
    n_docs = 0
    with open(OUT, "w") as fh:
        for pdf in pdfs:
            domain = "physics" if "physics" in str(pdf.parent).lower() else "finance"
            try:
                reader = PdfReader(str(pdf))
                pages = reader.pages[:MAX_PAGES]
                text = clean(" ".join((p.extract_text() or "") for p in pages))
            except Exception as exc:                      # corrupt / encrypted
                print(f"  SKIP {pdf.name}: {type(exc).__name__}: {exc}")
                continue
            if len(text) < MIN_CHUNK_CHARS:
                print(f"  SKIP {pdf.name}: too little extractable text "
                      f"({len(text)} chars — likely a scan)")
                continue
            chunks = chunk_doc(text)
            if not chunks:
                continue
            n_docs += 1
            for i, ch in enumerate(chunks):
                fh.write(json.dumps({
                    "doc": pdf.stem, "domain": domain, "chunk_id": i,
                    "text": ch, "n_chars": len(ch),
                }) + "\n")
                n_written += 1
            print(f"  {pdf.stem[:58]:60s} {len(chunks):3d} chunks [{domain}]")

    print(f"\nwrote {n_written} chunks from {n_docs} documents -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
