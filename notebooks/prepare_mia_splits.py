"""Build member / non-member splits from the real paper corpus.

CRITICAL METHODOLOGY: the split is by *document*, never by chunk. Two chunks
from the same paper share vocabulary, notation and phrasing, so splitting at
chunk level would leak membership through style alone and inflate every
attack's AUC. Splitting by document makes "was this text trained on?" the
only signal available.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260925
DATA = Path(__file__).parent / "data"
MAX_CHUNKS_PER_DOC = 12       # cap so long books don't dominate the split


def main() -> None:
    chunks = [json.loads(l) for l in open(DATA / "corpus_chunks.jsonl")]
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc"], []).append(c)

    docs = sorted(by_doc)
    rng = random.Random(SEED)
    rng.shuffle(docs)
    half = len(docs) // 2
    member_docs, nonmember_docs = set(docs[:half]), set(docs[half:2 * half])

    def emit(doc_set: set[str], path: Path, tag: str) -> int:
        n = 0
        with open(path, "w") as fh:
            for doc in sorted(doc_set):
                for c in by_doc[doc][:MAX_CHUNKS_PER_DOC]:
                    fh.write(json.dumps({
                        "id": f"{tag}_{doc[:40]}_{c['chunk_id']}",
                        "text": c["text"],
                        "doc": doc,
                        "domain": c["domain"],
                    }) + "\n")
                    n += 1
        return n

    n_m = emit(member_docs, DATA / "mia_member.jsonl", "member")
    n_n = emit(nonmember_docs, DATA / "mia_nonmember.jsonl", "nonmember")

    meta = {
        "seed": SEED,
        "n_documents_total": len(docs),
        "n_member_docs": len(member_docs),
        "n_nonmember_docs": len(nonmember_docs),
        "n_member_chunks": n_m,
        "n_nonmember_chunks": n_n,
        "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        "split_level": "document",
        "member_docs": sorted(member_docs),
        "nonmember_docs": sorted(nonmember_docs),
    }
    json.dump(meta, open(DATA / "mia_split_meta.json", "w"), indent=2)
    print(f"documents: {len(docs)}  ->  {len(member_docs)} member / "
          f"{len(nonmember_docs)} non-member (disjoint at document level)")
    print(f"chunks:    {n_m} member / {n_n} non-member")


if __name__ == "__main__":
    main()
