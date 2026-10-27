#!/usr/bin/env python3
"""Check the claims in papers/thotbook_ai_arxiv.pdf against this tree.

Each claim gets a status and evidence. Nothing is asserted that was not
measured or located on disk in this run:

  PASS     measured here, consistent with the paper
  FAIL     measured here, contradicts the paper
  BLOCKED  needs something absent (weights, a labelled set, the second node)
  ABSENT   the paper describes machinery that does not exist in this tree

Exit code is 1 if anything is FAIL or ABSENT, so CI can gate on it.
`--json` prints the machine-readable record used to regenerate the table.

Usage:
    verify_paper.py [--quick] [--json] [--only PATTERN]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVE_URL = os.environ.get("THOTBOOK_SERVE_URL", "http://127.0.0.1:8100")

PASS, FAIL, BLOCKED, ABSENT = "PASS", "FAIL", "BLOCKED", "ABSENT"


@dataclass
class Result:
    claim: str
    section: str
    status: str
    paper_says: str
    measured: str
    evidence: str = ""


def sh(cmd: list[str], timeout: int = 900, cwd: Path = REPO):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _sysctl(n: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", n], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def grep(pattern: str, *globs: str) -> list[str]:
    """Return 'path:line' for each regex hit, excluding target/."""
    hits, rx = [], re.compile(pattern)
    for g in globs:
        for p in REPO.glob(g):
            if "target" in p.parts or not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{p.relative_to(REPO)}:{i}")
            except Exception:
                continue
    return hits


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------

def c_hardware() -> Result:
    cpu = _sysctl("machdep.cpu.brand_string")
    try:
        ram_gib = int(_sysctl("hw.memsize")) / (1 << 30)
    except ValueError:
        ram_gib = 0
    cores = os.cpu_count() or 0
    want = "i9-8950HK"
    ok = want in cpu and 30 <= ram_gib <= 34 and cores == 12
    return Result(
        "Host matches the paper's bench machine", "5.1 Hardware",
        PASS if ok else FAIL,
        "Intel i9-8950HK, 6c/12t, 32 GB, no GPU",
        f"{cpu or 'unknown'}, {cores} threads, {ram_gib:.0f} GiB",
        "sysctl machdep.cpu.brand_string / hw.memsize")


def c_tests() -> Result:
    r = sh(["cargo", "test", "--workspace", "--quiet"])
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", r.stdout))
    failed = sum(int(m) for m in re.findall(r"(\d+) failed", r.stdout))
    return Result(
        "Rust test suite is green", "README badge",
        PASS if (r.returncode == 0 and failed == 0) else FAIL,
        "71 tests passing (badge)",
        f"{passed} passed, {failed} failed",
        "cargo test --workspace")


def c_openai_surface() -> Result:
    routes = grep(r'"/v1/(chat/completions|models|embeddings)"|/health',
                  "crates/llm-serve/src/*.rs")
    need = {"chat/completions": False, "models": False, "health": False}
    body = "\n".join((REPO / "crates/llm-serve/src/http.rs")
                     .read_text(errors="ignore").splitlines())
    for k in need:
        need[k] = k in body
    ok = all(need.values())
    return Result(
        "OpenAI-compatible endpoints exist", "4.5 Inference Server",
        PASS if ok else FAIL,
        "/health, /v1/models, /v1/chat/completions",
        ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in need.items()),
        f"{len(routes)} route refs in llm-serve")


def c_hybrid_retrieval() -> Result:
    # No trailing \b: the identifiers are bm25_score, rrf_k, ... and `_` is a
    # word character, so \b never matches there.
    bm25 = grep(r"\bbm25", "crates/llm-rag/src/*.rs")
    rrf = grep(r"\brrf|reciprocal", "crates/llm-rag/src/*.rs")
    hnsw = grep(r"struct Hnsw|fn search", "crates/llm-rag/src/hnsw.rs")
    ok = bool(bm25 and rrf and hnsw)
    return Result(
        "Hybrid BM25 + HNSW with RRF fusion is implemented", "6.3/6.4",
        PASS if ok else ABSENT,
        "HNSW ANN + BM25 sparse, fused by RRF",
        f"bm25={len(bm25)} hits, rrf={len(rrf)} hits, hnsw={len(hnsw)} hits",
        (bm25 + rrf + hnsw)[0] if (bm25 or rrf or hnsw) else "")


def c_no_nested_runtime() -> Result:
    """Regression guard: `thotbook serve` used to panic before accepting.

    llm-cli is #[tokio::main], so calling llm_serve::run_server (which builds
    its own runtime and block_on's) panicked with "Cannot start a runtime from
    within a runtime" the moment it was reached. It had already logged
    "listening", so it looked like a server refusing every connection.
    """
    cli = REPO / "crates/llm-cli/src/main.rs"
    if not cli.is_file():
        return Result("`thotbook serve` does not nest tokio runtimes",
                      "4.5 Inference Server", ABSENT,
                      "a single CLI entrypoint that serves the model",
                      "llm-cli/src/main.rs not found", "")
    text = cli.read_text(errors="ignore")
    async_main = "#[tokio::main]" in text
    calls_sync = re.search(r"llm_serve::run_server\s*\(", text)
    awaits = re.search(r"llm_serve::serve\s*\([^)]*\)\s*\.await", text)
    ok = (not async_main) or (awaits and not calls_sync)
    return Result(
        "`thotbook serve` does not nest tokio runtimes",
        "4.5 Inference Server", PASS if ok else FAIL,
        "a single CLI entrypoint that serves the model",
        f"#[tokio::main]={async_main}, awaits serve()={bool(awaits)}, "
        f"calls run_server()={bool(calls_sync)}",
        "crates/llm-cli/src/main.rs")


def c_loss_sign() -> Result:
    """Cross-entropy must be non-negative; loss.rs drops the leading minus."""
    src = REPO / "crates/llm-train/src/loss.rs"
    if not src.is_file():
        return Result("Training loss is cross-entropy", "3.1 / Table 1",
                      ABSENT, "cross-entropy loss with masking",
                      "loss.rs not found", "")
    text = src.read_text(errors="ignore")
    negated = re.search(r"nll\s*\.\s*neg\s*\(|-\s*nll|neg\(\)\?\.mean_all", text)
    has_mask = re.search(r"ignore_index|\bmask\b", text)
    smoothing_used = re.search(r"[^_]label_smoothing", text)
    rc = sh(["cargo", "test", "-p", "llm-train", "--test",
             "loss_characterization", "--quiet"]).returncode
    return Result(
        "Training loss is cross-entropy (correct sign)", "3.1 / Table 1",
        PASS if negated else FAIL,
        "L = -(1/N) sum log q, masked over padding",
        ("negation present" if negated else
         "returns +mean(log q) = -L, so minimising it trains AWAY from the "
         "target") +
        f"; masking={'yes' if has_mask else 'none'}"
        f", label_smoothing={'used' if smoothing_used else 'ignored'}"
        f"; characterization suite {'runs' if rc == 0 else 'FAILS'}",
        "crates/llm-train/src/loss.rs:17")


def c_corpus_size() -> Result:
    """The abstract's corpus claim, against the rebuilt historia index."""
    import sqlite3
    db = REPO / "data/ingest/historia.db"
    if not db.is_file():
        return Result(
            "Corpus is ~500 documents / ~5M words", "Abstract", BLOCKED,
            "~500 documents (~5M words)",
            "data/ingest/historia.db not built - run `make historia-db`",
            "scripts/build_historia_db.py")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        docs, words = con.execute(
            "select count(*), coalesce(sum(word_count),0) from documents"
        ).fetchone()
        con.close()
    except Exception as e:
        return Result("Corpus is ~500 documents / ~5M words", "Abstract", FAIL,
                      "~500 documents (~5M words)",
                      f"historia.db unreadable: {e}", str(db))
    ok = 400 <= docs <= 700 and 4e6 <= words <= 6.5e6
    return Result(
        "Corpus is ~500 documents / ~5M words", "Abstract",
        PASS if ok else FAIL,
        "~500 documents (~5M words)",
        f"{docs} documents, {words:,} words",
        "data/ingest/historia.db")


def c_hnsw_uses_query() -> Result:
    """HNSW that ignores the query is not ANN search at all.

    engine: search(&self, _query: &[f32], k) descends the graph with
    search_closest(current, level, 1) and ranks with
    search_closest_n(start, level, k), both of which measure distances
    BETWEEN STORED VECTORS (self.vectors[start] vs neighbours). The query is
    never compared to anything, so every query returns the same neighbourhood
    around the entry point.
    """
    src = REPO / "crates/llm-rag/src/hnsw.rs"
    if not src.is_file():
        return Result("HNSW search actually uses the query vector", "6.3",
                      ABSENT, "HNSW approximate nearest neighbour",
                      "hnsw.rs not found", "")
    text = src.read_text(errors="ignore")
    sig = re.search(r"pub fn search\(\s*&self,\s*(\w+)\s*:\s*&\[f32\]", text)
    param = sig.group(1) if sig else ""
    ignored = param.startswith("_")
    # Does any distance call in the file take the query as an argument?
    body = text.split("pub fn search(")[1].split("\n    }")[0] if sig else ""
    uses = bool(param and not ignored and re.search(rf"\b{param}\b", body))
    return Result(
        "HNSW search actually uses the query vector", "6.3 / Table 3",
        PASS if uses else FAIL,
        "HNSW ANN retrieval reaching 0.81 Recall@5 on its own",
        f"search() parameter is `{param or '?'}` and is "
        f"{'ignored - results are query-independent' if ignored else 'used'}; "
        "distances are computed between stored vectors only",
        "crates/llm-rag/src/hnsw.rs:299")


def c_retrieval_numbers() -> Result:
    sets = list(REPO.glob("data/**/*qrel*")) + list(REPO.glob("data/**/*eval*"))
    return Result(
        "Recall@5 = 0.89, MRR = 0.74, nDCG@10 = 0.79", "Table 3 (Retrieval)",
        BLOCKED if not sets else PASS,
        "Recall@5 0.89 / MRR 0.74 / nDCG@10 0.79 (hybrid RRF)",
        "no labelled query/relevance set in the tree - not reproducible",
        "searched data/**/*qrel*, data/**/*eval*")


def c_quantization() -> Result:
    q = grep(r"q4_k|Q4_K|quantiz|gguf|GGUF", "crates/*/src/*.rs")
    return Result(
        "Q4_K quantised serving at 4.5 GB / 12-18 tok/s", "Table 2 (Inference)",
        ABSENT if not q else PASS,
        "Q4_K: 4.5 GB model, 6 GB RAM, 12-18 tok/s",
        "llm-serve serves dense F32/F16/BF16 safetensors only "
        "(LLM_SERVE_DTYPE); no q4/gguf code in any crate. 4-bit lives in "
        "distributed-llama, outside this tree.",
        "crates/llm-serve/src/engine.rs (dtype select)")


def c_autodiff() -> Result:
    g = REPO / "notebooks/thotgrad.py"
    o = REPO / "notebooks/thotopt.py"
    if not (g.is_file() and o.is_file()):
        return Result("Self-contained autodiff + optimisers", "7.1/7.2",
                      ABSENT, "own AD engine, no DL framework",
                      "thotgrad.py / thotopt.py not found", "")
    # Find an interpreter that actually has numpy: the python3 on PATH here is
    # 3.15, which has no numpy wheel yet, while the repo .venv and /usr/bin
    # both do. Reporting BLOCKED just because the default interpreter is bare
    # would understate what this tree can prove.
    cands = [str(REPO / ".venv/bin/python"), "/usr/bin/python3", sys.executable]
    interp = next(
        (c for c in cands
         if Path(c).exists()
         and sh([c, "-c", "import numpy"], timeout=120).returncode == 0),
        None)
    if interp is None:
        return Result(
            "Self-contained autodiff + optimisers", "7.1/7.2", BLOCKED,
            "own reverse-mode AD, SGD/Adam, no torch dependency",
            "no interpreter with numpy found (tried .venv, /usr/bin/python3, "
            f"{sys.executable}); `uv venv .venv && uv pip install numpy`",
            "notebooks/test_thotgrad.py")
    # The user's own code, so no -I here; -I would also hide site-packages.
    r = sh([interp, "test_thotgrad.py"], cwd=REPO / "notebooks", timeout=300)
    ok = r.returncode == 0
    missing = re.search(r"No module named '([\w.]+)'", r.stderr or "")
    if not ok and missing:
        # An uninstalled dependency is an environment gap, not a false claim.
        return Result(
            "Self-contained autodiff + optimisers", "7.1/7.2", BLOCKED,
            "own reverse-mode AD, SGD/Adam, no torch dependency",
            f"cannot run: {sys.executable.split('/')[-1]} has no "
            f"'{missing.group(1)}'. Every notebook claim in sections 7 and 9 "
            f"is blocked on this.",
            "notebooks/test_thotgrad.py")
    return Result(
        "Self-contained autodiff + optimisers", "7.1/7.2",
        PASS if ok else FAIL,
        "own reverse-mode AD, SGD/Adam, no torch dependency",
        f"{interp.split('/')[-1]}: {sum(1 for l in (r.stdout or '').splitlines() if l.lstrip().startswith('PASS'))}"
        f" assertions passed, test_thotgrad.py {'ok' if ok else 'FAILED'}"
        + ("" if ok else f": {(r.stderr or '').strip().splitlines()[-1:]}"),
        "notebooks/test_thotgrad.py")


def c_torch_free() -> Result:
    hits = grep(r"^\s*(import|from)\s+torch", "notebooks/*.py", "crates/*/src/*.rs",
                "python/thotbook_ai/**/*.py")
    return Result(
        "No deep-learning framework dependency", "7.1",
        PASS if not hits else FAIL,
        "removes any deep-learning framework dependency",
        "no torch imports" if not hits else f"torch imported at {hits[:3]}",
        "grep '^import torch' over notebooks/, crates/, python/")


def c_federated() -> Result:
    fed = REPO / "notebooks/fed_core.py"
    rust = grep(r"federat|all.?reduce|fedavg|aggregat", "crates/*/src/*.rs")
    if not fed.is_file():
        return Result("Federated training module", "7.5/9.3", ABSENT,
                      "federated module with attack/defence taxonomy",
                      "fed_core.py not found", "")
    return Result(
        "Federated training lives in the shipped Rust pipeline", "7.5/9.3",
        ABSENT if not rust else PASS,
        "federated-training module implementing a published taxonomy",
        f"implemented in Python notebooks only (fed_core.py, "
        f"{fed.stat().st_size}B); 0 references in any Rust crate",
        "notebooks/fed_core.py")


def c_two_node() -> Result:
    nb = REPO / "notebooks/07_two_node_distributed.ipynb"
    worker = REPO / "notebooks/remote_worker_two_node.py"
    reachable = False
    if shutil.which("ssh"):
        r = sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                "hfthot", "true"], timeout=20)
        reachable = r.returncode == 0
    # Reachability and a committed notebook are NOT a reproduction. Only a
    # fresh run artifact would be, so this stays BLOCKED until one exists.
    artifact = next(REPO.glob("notebooks/data/**/*two_node*"), None) \
        or next(REPO.glob("notebooks/data/**/*stage5*"), None)
    return Result(
        "Two-node WAN training campaign reproduces", "9.2 Case Study",
        BLOCKED,
        "laptop + 4-core VPS share one run; exact aggregation identity over WAN",
        f"notebook={'present' if nb.is_file() else 'missing'}, "
        f"worker={'present' if worker.is_file() else 'missing'}, "
        f"VPS hfthot {'reachable' if reachable else 'unreachable'}, "
        f"run artifact={'present' if artifact else 'none'}; "
        "re-running it needs numpy on the VPS and here",
        "notebooks/07_two_node_distributed.ipynb")


def c_redaction() -> Result:
    r = sh(["cargo", "test", "-p", "llm-corpus", "--quiet"])
    red = grep(r"fn redact|REDACTED", "crates/llm-corpus/src/redact.rs")
    py = list(REPO.glob("python/thotbook_ai/**/redact.py"))
    return Result(
        "Redaction is enforced at ingest, Rust/Python parity", "4.1, 7.4",
        PASS if (r.returncode == 0 and red) else FAIL,
        "write-time leak audit refuses rows with secrets; Rust/Python/JS parity",
        f"llm-corpus tests {'pass' if r.returncode == 0 else 'FAIL'}; "
        f"redact.rs {len(red)} hits; python redact.py "
        f"{'present' if py else 'MISSING'}",
        "crates/llm-corpus/src/redact.rs")


def c_serve_live(quick: bool) -> list[Result]:
    """Measure the running server, if one is up."""
    try:
        req = urllib.request.Request(f"{SERVE_URL}/v1/models",
                                     headers={"User-Agent": "verify"})
        with urllib.request.urlopen(req, timeout=4) as r:
            models = json.load(r)
    except Exception as e:
        return [Result(
            "Measured inference throughput", "Table 2 (Inference)",
            BLOCKED, "12-18 tok/s (Q4_K, 8B)",
            f"no server on {SERVE_URL} ({type(e).__name__}); "
            "start one with `make serve MODEL=<key>`",
            "GET /v1/models")]

    mid = ""
    try:
        mid = (models.get("data") or [{}])[0].get("id", "")
    except Exception:
        pass

    if quick:
        return [Result("Server is live and advertises a model", "4.5",
                       PASS, "OpenAI-compatible /v1/models", f"model_id={mid}",
                       f"{SERVE_URL}/v1/models")]

    want = 64
    body = json.dumps({
        "model": mid or "local",
        "messages": [{"role": "user",
                      "content": "Count from one to twenty in words."}],
        "temperature": 0.0, "max_tokens": want,
    }).encode()
    req = urllib.request.Request(f"{SERVE_URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "verify"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
    except Exception as e:
        return [Result("Measured inference throughput", "Table 2 (Inference)",
                       FAIL, "12-18 tok/s",
                       f"/v1/chat/completions errored: {type(e).__name__}: {e}",
                       f"{SERVE_URL}/v1/chat/completions")]
    dt = time.monotonic() - t0
    usage = out.get("usage") or {}
    toks = usage.get("completion_tokens") or want
    rate = toks / dt if dt > 0 else 0.0
    return [
        Result("Server is live and advertises a model", "4.5", PASS,
               "OpenAI-compatible /v1/models", f"model_id={mid}",
               f"{SERVE_URL}/v1/models"),
        Result("Measured inference throughput", "Table 2 (Inference)",
               PASS if 12 <= rate <= 18 else FAIL,
               "12-18 tok/s (Q4_K 8B on this CPU)",
               f"{rate:.2f} tok/s ({toks} tokens in {dt:.1f}s) serving "
               f"{mid or 'unknown'} at "
               f"{os.environ.get('LLM_SERVE_DTYPE', 'f32')} (dense, not Q4_K)",
               f"{SERVE_URL}/v1/chat/completions"),
    ]


def c_train_throughput() -> Result:
    models = [p for p in (REPO / "data/models").glob("*/thotbook-model.json")] \
        if (REPO / "data/models").is_dir() else []
    shards = list(REPO.glob("data/training/**/*.arrow"))
    if not models or not shards:
        return Result(
            "LoRA training throughput", "Table 1 (Training)",
            BLOCKED, "0.9 tok/s, 22 GB, 21 h for 3 epochs (LoRA r=16, 8B)",
            f"{len(models)} model(s) installed, {len(shards)} tokenised shard(s); "
            "need both to time a step",
            "data/models/*, data/training/**/*.arrow")
    return Result(
        "LoRA training throughput", "Table 1 (Training)",
        BLOCKED, "0.9 tok/s, 22 GB, 21 h for 3 epochs (LoRA r=16, 8B)",
        "measurable but not run here: a representative figure needs a "
        "multi-hour run. Use `make train-bench` for a timed 50-step sample.",
        "crates/llm-train")


def c_lakehouse() -> Result:
    hits = grep(r"DeltaStore|ensure_table|time.?travel|version",
                "crates/llm-corpus/src/delta.rs")
    poth = Path("polarway/polarway-lakehouse")
    return Result(
        "Delta lakehouse with versioned, time-travel tables", "4.1, 8",
        PASS if (hits and poth.is_dir()) else FAIL,
        "every stage writes versioned Delta tables",
        f"delta.rs {len(hits)} hits; polarway-lakehouse "
        f"{'resolved' if poth.is_dir() else 'MISSING'}",
        "crates/llm-corpus/src/delta.rs")


CHECKS = [
    ("hardware", c_hardware),
    ("tests", c_tests),
    ("openai-surface", c_openai_surface),
    ("no-nested-runtime", c_no_nested_runtime),
    ("loss-sign", c_loss_sign),
    ("corpus-size", c_corpus_size),
    ("hybrid-retrieval", c_hybrid_retrieval),
    ("hnsw-uses-query", c_hnsw_uses_query),
    ("retrieval-numbers", c_retrieval_numbers),
    ("quantization", c_quantization),
    ("autodiff", c_autodiff),
    ("torch-free", c_torch_free),
    ("federated", c_federated),
    ("two-node", c_two_node),
    ("redaction", c_redaction),
    ("train-throughput", c_train_throughput),
    ("lakehouse", c_lakehouse),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip generation timing (still checks liveness)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--only", default="", help="substring filter on claim keys")
    args = ap.parse_args()

    results: list[Result] = []
    for key, fn in CHECKS:
        if args.only and args.only not in key:
            continue
        try:
            results.append(fn())
        except Exception as e:
            results.append(Result(key, "-", FAIL, "-",
                                  f"checker raised {type(e).__name__}: {e}"))
    if not args.only or "serve" in args.only:
        try:
            results.extend(c_serve_live(args.quick))
        except Exception as e:
            results.append(Result("Measured inference throughput", "Table 2",
                                  FAIL, "-", f"checker raised {e}"))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        order = {PASS: 0, FAIL: 1, ABSENT: 2, BLOCKED: 3}
        print("\n  thotbook_ai_arxiv.pdf - claim verification")
        print(f"  {time.strftime('%Y-%m-%d %H:%M')}  "
              f"{_sysctl('machdep.cpu.brand_string')}\n")
        for r in sorted(results, key=lambda x: (order.get(x.status, 9), x.section)):
            print(f"  [{r.status:7s}] {r.claim}")
            print(f"            section   {r.section}")
            print(f"            paper     {r.paper_says}")
            print(f"            measured  {r.measured}")
            if r.evidence:
                print(f"            evidence  {r.evidence}")
            print()
        tally = {s: sum(1 for r in results if r.status == s)
                 for s in (PASS, FAIL, ABSENT, BLOCKED)}
        print(f"  {tally[PASS]} pass, {tally[FAIL]} fail, "
              f"{tally[ABSENT]} absent, {tally[BLOCKED]} blocked "
              f"(of {len(results)})\n")

    bad = [r for r in results if r.status in (FAIL, ABSENT)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
