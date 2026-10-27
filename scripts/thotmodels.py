#!/usr/bin/env python3
"""Model catalogue, feasibility check and downloader for thotbook-ai.

Stdlib only: no huggingface-cli, no git-lfs, no pip install. Sizes come from
the live HuggingFace API, so the table never drifts from reality, and the
download ETA comes from a measured bandwidth probe rather than a guess.

Two serving backends, because they are genuinely different machines:

  candle  safetensors -> crates/llm-serve (F16, CPU, OpenAI-compatible).
          Qwen3 architecture ONLY: engine.rs hardcodes
          candle_transformers::models::qwen3::ModelForCausalLM, and loads
          every shard at F16 with no quantisation path.
  dllama  q40 .m files -> ../distributed-llama-cloudnostic (4-bit,
          tensor-parallel over 2^n nodes). Fast, and the only 8B path that
          fits this laptop. Inference only - it has no backward pass.

All sizes are decimal GB (10^9 bytes), matching HuggingFace and `df`.

Usage:
    thotmodels.py list [--backend candle|dllama|all] [--refresh]
    thotmodels.py plan  <key>
    thotmodels.py setup <key> [--yes] [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / ".model-catalogue.json"
CACHE_TTL = 24 * 3600
HF_API = "https://huggingface.co/api/models/"
UA = {"User-Agent": "thotbook-ai/0.3 (+local)"}

# Peak RSS while llm-serve loads weights, from reading engine.rs load():
#   load_shards() reads each shard into a Vec<u8>, load_buffer() materialises
#   BF16 tensors from it, then to_dtype() allocates a full second copy before
#   the originals drop.
# The serving dtype is F32, not F16: llm-serve links accelerate-src, and the
# Accelerate backend has no f16 matmul (every completion 500s with
# "the accelerate backend does not support f16 matmul"). F32 is 4 bytes
# against 2 on disk, so the copy is twice the weight bytes:
#       1 x (BF16 weights) + 2 x (F32 copy) + largest shard, transient
# Set LLM_SERVE_DTYPE=f16 (with accelerate removed) to get the 2.0 factor back.
CANDLE_WEIGHT_COPIES = 3.0
# dllama mmaps the q40 file and keeps a modest KV cache beside it.
DLLAMA_PEAK_FACTOR = 1.15

HEADROOM_GB = 3.0  # leave this much RAM for the OS and everything else


@dataclass
class Model:
    key: str
    repo: str
    backend: str  # "candle" | "dllama"
    params: str
    note: str
    # candle: which files we actually need out of the HF repo
    patterns: tuple[str, ...] = (".safetensors", "config.json", "generation_config.json")
    # dllama: explicit URLs (its repos are flat .m / .t blobs)
    urls: tuple[str, ...] = ()
    tokenizer_url: str = ""
    # dllama multi-part weights are concatenated into this single file
    merge_name: str = ""
    dest: str = ""
    size_gb: float = 0.0
    size_known: bool = True
    files: list[dict] = field(default_factory=list)


def _dl(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


_B4 = "https://huggingface.co/b4rtaz"


def _b4(repo: str, name: str) -> str:
    return f"{_B4}/{repo}/resolve/main/{name}"


def catalogue() -> list[Model]:
    """Free, non-gated models worth offering on this host."""
    return [
        # ---- candle / llm-serve (safetensors, F16) -- Qwen3 architecture only
        Model("qwen3-0.6b", "Qwen/Qwen3-0.6B", "candle", "0.6B",
              "smallest useful; instant load, good for wiring tests",
              dest="data/models/qwen3-0.6b"),
        Model("qwen3-1.7b", "Qwen/Qwen3-1.7B", "candle", "1.7B",
              "best speed/quality trade-off for F32 on this CPU",
              dest="data/models/qwen3-1.7b"),
        Model("qwen3-4b", "Qwen/Qwen3-4B", "candle", "4B",
              "largest that fits at F32, but tight - prefer dllama-qwen3-8b",
              dest="data/models/qwen3-4b"),
        Model("qwen3-8b", "Qwen/Qwen3-8B", "candle", "8B",
              "the README/paper target; does NOT fit at F32 - use dllama-qwen3-8b",
              dest="data/models/qwen3-8b"),

        # ---- dllama / distributed-llama (q40, tensor-parallel)
        Model("dllama-qwen3-0.6b", "b4rtaz/Qwen3-0.6B-Q40-Distributed-Llama",
              "dllama", "0.6B", "4-bit, single node, trivial smoke test",
              urls=(_b4("Qwen3-0.6B-Q40-Distributed-Llama",
                        "dllama_model_qwen3_0.6b_q40.m"),),
              tokenizer_url=_b4("Qwen3-0.6B-Q40-Distributed-Llama",
                                "dllama_tokenizer_qwen3_0.6b.t"),
              merge_name="dllama_model_qwen3_0.6b_q40.m",
              dest="data/models/dllama-qwen3-0.6b"),
        Model("dllama-qwen3-1.7b", "b4rtaz/Qwen3-1.7B-Q40-Distributed-Llama",
              "dllama", "1.7B", "4-bit, comfortable on one node",
              urls=(_b4("Qwen3-1.7B-Q40-Distributed-Llama",
                        "dllama_model_qwen3_1.7b_q40.m"),),
              tokenizer_url=_b4("Qwen3-1.7B-Q40-Distributed-Llama",
                                "dllama_tokenizer_qwen3_1.7b.t"),
              merge_name="dllama_model_qwen3_1.7b_q40.m",
              dest="data/models/dllama-qwen3-1.7b"),
        Model("dllama-qwen3-8b", "b4rtaz/Qwen3-8B-Q40-Distributed-Llama",
              "dllama", "8B",
              "RECOMMENDED: a real 8B on this laptop, 4-bit, splits across nodes",
              urls=(_b4("Qwen3-8B-Q40-Distributed-Llama",
                        "dllama_model_qwen3_8b_q40.m"),),
              tokenizer_url=_b4("Qwen3-8B-Q40-Distributed-Llama",
                                "dllama_tokenizer_qwen3_8b.t"),
              merge_name="dllama_model_qwen3_8b_q40.m",
              dest="data/models/dllama-qwen3-8b"),
        Model("dllama-qwen3-14b", "b4rtaz/Qwen3-14B-Q40-Distributed-Llama",
              "dllama", "14B", "4-bit, two split parts joined on download",
              urls=(_b4("Qwen3-14B-Q40-Distributed-Llama",
                        "dllama_model_qwen3_14b_q40_aa"),
                    _b4("Qwen3-14B-Q40-Distributed-Llama",
                        "dllama_model_qwen3_14b_q40_ab")),
              tokenizer_url=_b4("Qwen3-14B-Q40-Distributed-Llama",
                                "dllama_tokenizer_qwen3_14b.t"),
              merge_name="dllama_model_qwen3_14b_q40.m",
              dest="data/models/dllama-qwen3-14b"),
        Model("dllama-llama3.2-3b", "b4rtaz/Llama-3_2-3B-Q40-Instruct-Distributed-Llama",
              "dllama", "3B", "non-Qwen alternative (Llama 3.2 community licence)",
              urls=(_b4("Llama-3_2-3B-Q40-Instruct-Distributed-Llama",
                        "dllama_model_llama3.2-3b-instruct_q40.m"),),
              tokenizer_url=_b4("Llama-3_2-3B-Q40-Instruct-Distributed-Llama",
                                "dllama_tokenizer_llama3_2.t"),
              merge_name="dllama_model_llama3.2-3b-instruct_q40.m",
              dest="data/models/dllama-llama3.2-3b"),
        Model("dllama-deepseek-r1-8b",
              "b4rtaz/DeepSeek-R1-Distill-Llama-8B-Distributed-Llama",
              "dllama", "8B", "reasoning-distilled alternative, MIT weights",
              urls=(_b4("DeepSeek-R1-Distill-Llama-8B-Distributed-Llama",
                        "dllama_model_deepseek-r1-distill-llama-8b_q40.m"),),
              tokenizer_url=_b4("DeepSeek-R1-Distill-Llama-8B-Distributed-Llama",
                                "dllama_tokenizer_deepseek-r1-distill-llama-8b.t"),
              merge_name="dllama_model_deepseek-r1-distill-llama-8b_q40.m",
              dest="data/models/dllama-deepseek-r1-8b"),
    ]


# --------------------------------------------------------------------------
# host facts
# --------------------------------------------------------------------------

def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def host() -> dict:
    try:
        ram = int(_sysctl("hw.memsize"))
    except ValueError:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            ram = 0
    return {
        "cpu": _sysctl("machdep.cpu.brand_string") or "unknown CPU",
        "cores": os.cpu_count() or 0,
        "ram_gb": ram / 1e9,
        "ram_gib": ram / (1 << 30),
        "disk_free_gb": shutil.disk_usage(REPO).free / 1e9,
    }


# --------------------------------------------------------------------------
# HF metadata + bandwidth
# --------------------------------------------------------------------------

TIMEOUT = 12
RETRIES = 3


def _get_json(url: str, timeout: int = TIMEOUT) -> dict:
    """GET with bounded retries. HF rate-limits bursts, so back off and retry."""
    last: Exception = RuntimeError("no attempt made")
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # URLError, timeout, HTTP 429, truncated JSON
            last = e
            if attempt < RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last


def _repo_blobs(repo: str) -> dict[str, int]:
    """{filename: size} for a whole repo in ONE API call.

    Used for both backends: a ranged GET per file would be N round trips and
    is exactly the burst that gets us throttled.
    """
    meta = _get_json(f"{HF_API}{repo}?blobs=true")
    return {f["rfilename"]: (f.get("size") or 0)
            for f in meta.get("siblings", [])}


def _wanted(name: str, patterns: tuple[str, ...]) -> bool:
    if name == "model.safetensors.index.json":
        return True
    return any(name.endswith(p) or name == p for p in patterns)


def _load_cache() -> tuple[dict, bool]:
    """(models, fresh). A stale cache is still far better than nothing."""
    if not CACHE.is_file():
        return {}, False
    try:
        blob = json.loads(CACHE.read_text())
        return blob.get("models", {}), (time.time() - blob.get("fetched", 0)) < CACHE_TTL
    except Exception:
        return {}, False


def _describe(m: Model) -> None:
    """Fill m.files / m.size_gb from one HF API call. Raises on failure."""
    blobs = _repo_blobs(m.repo)
    if m.backend == "candle":
        m.files = [{"name": n, "url": _dl(m.repo, n), "size": sz}
                   for n, sz in sorted(blobs.items()) if _wanted(n, m.patterns)]
    else:
        m.files = []
        for u in m.urls:
            n = u.rsplit("/", 1)[1]
            m.files.append({"name": n, "url": u, "size": blobs.get(n, 0),
                            "part": True})
        if m.tokenizer_url:
            n = m.tokenizer_url.rsplit("/", 1)[1]
            m.files.append({"name": n, "url": m.tokenizer_url,
                            "size": blobs.get(n, 0)})
    m.size_gb = sum(f["size"] for f in m.files) / 1e9
    # A zero-byte weight file means the URL is wrong or HF withheld it: report
    # the total as unknown rather than quietly understating the download.
    m.size_known = bool(m.files) and all(
        f["size"] > 0 for f in m.files
        if f.get("part") or f["name"].endswith(".safetensors"))


def resolve_sizes(models: list[Model], refresh: bool = False) -> list[Model]:
    cache, fresh = _load_cache()
    use_cache = cache if (fresh and not refresh) else {}

    todo = []
    for m in models:
        hit = use_cache.get(m.key)
        if hit:
            m.size_gb, m.files = hit["size_gb"], hit["files"]
            m.size_known = hit.get("size_known", True)
        else:
            todo.append(m)

    if todo:
        # One call per repo, a few at a time: fast enough, gentle enough.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_describe, m): m for m in todo}
            for fut, m in futures.items():
                try:
                    fut.result()
                except Exception as e:
                    stale = cache.get(m.key)
                    if stale:  # network down: stale numbers beat no numbers
                        m.size_gb, m.files = stale["size_gb"], stale["files"]
                        m.size_known = stale.get("size_known", True)
                        m.note = f"{m.note}  [cached]"
                    else:
                        m.note = f"{m.note}  [{type(e).__name__}]"
                        m.size_gb, m.files, m.size_known = 0.0, [], False

        merged = dict(cache)
        merged.update({m.key: {"size_gb": m.size_gb, "files": m.files,
                               "size_known": m.size_known}
                       for m in models if m.files})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"fetched": time.time(), "models": merged},
                                    indent=1))
    return models


def probe_bandwidth(sample_bytes: int = 24 << 20, warmup: int = 6 << 20) -> float:
    """MB/s against the CDN we are about to pull from.

    The clock starts only after `warmup` bytes: TCP slow-start and TLS setup
    dominate the first few MB and made the old 8 MB probe read ~4x low against
    the rate real downloads then sustained.
    """
    url = _dl("Qwen/Qwen3-0.6B", "model.safetensors")
    req = urllib.request.Request(
        url, headers={**UA, "Range": f"bytes=0-{sample_bytes - 1}"})
    try:
        got, t0, measured = 0, None, 0
        with urllib.request.urlopen(req, timeout=40) as r:
            while chunk := r.read(1 << 18):
                got += len(chunk)
                if t0 is None and got >= warmup:
                    t0, measured = time.monotonic(), got
        if t0 is None:
            return 0.0
        dt = time.monotonic() - t0
        return ((got - measured) / 1e6) / dt if dt > 0 else 0.0
    except Exception:
        return 0.0


def human_eta(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# --------------------------------------------------------------------------
# feasibility
# --------------------------------------------------------------------------

def peak_gb(m: Model) -> float:
    if m.backend == "dllama":
        return m.size_gb * DLLAMA_PEAK_FACTOR
    weights = sum(f["size"] for f in m.files
                  if f["name"].endswith(".safetensors")) / 1e9
    largest = max((f["size"] for f in m.files
                   if f["name"].endswith(".safetensors")), default=0) / 1e9
    return weights * CANDLE_WEIGHT_COPIES + largest


def verdict(m: Model, h: dict) -> tuple[str, str]:
    """(status, why) with status in OK / TIGHT / NO / UNKNOWN."""
    if not m.size_known or m.size_gb <= 0:
        return "UNKNOWN", "could not read sizes from HuggingFace"
    if m.size_gb > h["disk_free_gb"]:
        return "NO", f"needs {m.size_gb:.1f} GB disk, {h['disk_free_gb']:.0f} GB free"
    peak = peak_gb(m)
    usable = h["ram_gb"] - HEADROOM_GB
    if peak > h["ram_gb"]:
        return "NO", f"peak ~{peak:.0f} GB > {h['ram_gb']:.0f} GB RAM"
    if peak > usable:
        return "TIGHT", f"peak ~{peak:.0f} GB vs {usable:.0f} GB usable"
    return "OK", f"peak ~{peak:.0f} GB of {usable:.0f} GB usable"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_list(args) -> int:
    h = host()
    models = resolve_sizes(catalogue(), refresh=args.refresh)
    if args.backend != "all":
        models = [m for m in models if m.backend == args.backend]

    bw = 0.0 if args.no_probe else probe_bandwidth()
    print(f"\nhost: {h['cpu']}  |  {h['cores']} threads  |  "
          f"{h['ram_gb']:.0f} GB ({h['ram_gib']:.0f} GiB) RAM  |  "
          f"{h['disk_free_gb']:.0f} GB free")
    if bw:
        print(f"measured bandwidth: {bw:.1f} MB/s ({bw * 8:.0f} Mbit/s)")
    print()
    hdr = (f"{'KEY':22s} {'BACKEND':8s} {'PAR':5s} {'SIZE':>9s} {'ETA':>7s} "
           f"{'FITS':8s} NOTE")
    print(hdr)
    print("-" * 104)
    for m in models:
        st, why = verdict(m, h)
        eta = human_eta(m.size_gb * 1000 / bw) if (bw and m.size_known) else "-"
        size = f"{m.size_gb:.2f} GB" if m.size_known else "?"
        print(f"{m.key:22s} {m.backend:8s} {m.params:5s} {size:>9s} {eta:>7s} "
              f"{st:8s} {m.note}")
        if st != "OK":
            print(f"{'':54s} {'':8s} -> {why}")
    print("\nnext:  make plan-model  MODEL=<key>     # show the plan only")
    print("       make setup-model MODEL=<key>     # asks yes/no, then downloads\n")
    return 0


def _print_plan(m: Model, h: dict, bw: float) -> None:
    st, why = verdict(m, h)
    dest = REPO / m.dest
    size = f"{m.size_gb:.2f} GB" if m.size_known else "unknown"
    print(f"\n  model      {m.key}   ({m.params}, {m.backend} backend)")
    print(f"  source     https://huggingface.co/{m.repo}")
    print(f"  files      {len(m.files)}"
          + (f"  (joined into {m.merge_name})" if len(m.urls) > 1 else ""))
    print(f"  download   {size}")
    if bw and m.size_known:
        print(f"  est. time  {human_eta(m.size_gb * 1000 / bw)}   "
              f"at measured {bw:.1f} MB/s")
    else:
        print("  est. time  unknown")
    print(f"  dest       {dest}")
    if m.size_known:
        print(f"  disk after {h['disk_free_gb'] - m.size_gb:.0f} GB free")
    print(f"  memory     {st}: {why}")
    print(f"  serve with make {'serve' if m.backend == 'candle' else 'dllama-serve'}"
          f" MODEL={m.key}")
    if st == "NO":
        print("\n  !! This model will not run on this host. See the line above.")
    print()


def cmd_plan(args) -> int:
    m = _find(args.key)
    resolve_sizes([m], refresh=args.refresh)
    _print_plan(m, host(), 0.0 if args.no_probe else probe_bandwidth())
    return 0


def _find(key: str) -> Model:
    for m in catalogue():
        if m.key == key:
            return m
    sys.exit(f"unknown model '{key}'.\navailable: "
             + ", ".join(m.key for m in catalogue()))


def download(url: str, path: Path, expect: int = 0, append_to=None) -> None:
    """Resumable single-file download, or append-into-open-handle for parts."""
    if append_to is not None:
        _stream(url, append_to, expect, path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if expect and path.is_file() and path.stat().st_size == expect:
        print(f"    {path.name}: already complete")
        return
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = expect or (int(r.headers.get("Content-Length") or 0) + have)
        resumed = r.status == 206 and have > 0
        with open(part, "ab" if resumed else "wb") as fh:
            if not resumed:
                fh.truncate(0)
                have = 0
            _pump(r, fh, have, total, path.name)
    got = part.stat().st_size
    if total and got != total:
        sys.exit(f"\n  size mismatch on {path.name}: got {got}, expected {total}."
                 f"\n  Re-run the same command to resume.")
    part.replace(path)


def _stream(url: str, fh, expect: int, label: str) -> None:
    start = fh.tell()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = expect or int(r.headers.get("Content-Length") or 0)
        _pump(r, fh, 0, total, label, offset=start)
    if expect and fh.tell() - start != expect:
        sys.exit(f"\n  short read on {label}: "
                 f"{fh.tell() - start} of {expect} bytes")


def _pump(r, fh, have: int, total: int, label: str, offset: int = 0) -> None:
    t0 = time.monotonic()
    done, last = have, 0.0
    while chunk := r.read(1 << 20):
        fh.write(chunk)
        done += len(chunk)
        now = time.monotonic()
        if now - last > 0.5:
            last = now
            rate = (done - have) / 1e6 / max(now - t0, 1e-9)
            pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
            left = human_eta((total - done) / 1e6 / rate) if (total and rate) else "?"
            print(f"\r    {label[:36]:36s} {pct} {done / 1e9:6.2f}/"
                  f"{total / 1e9:5.2f} GB {rate:6.1f} MB/s  ETA {left:>7s}",
                  end="", flush=True)
    print()


def cmd_setup(args) -> int:
    m = _find(args.key)
    h = host()
    resolve_sizes([m], refresh=args.refresh)
    if not m.files:
        sys.exit("could not list files for this model (offline?).\n"
                 "Re-run with --refresh once back online.")
    bw = 0.0 if args.no_probe else probe_bandwidth()
    _print_plan(m, h, bw)

    st, _ = verdict(m, h)
    if not args.yes:
        prompt = ("  This model will NOT run on this host. Download anyway? [y/N] "
                  if st == "NO" else "  Download now? [y/N] ")
        try:
            if input(prompt).strip().lower() not in ("y", "yes"):
                print("  aborted.\n")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\n  aborted (no tty: pass --yes to skip the prompt).\n")
            return 1

    dest = Path(args.dest) if args.dest else REPO / m.dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n  -> {dest}")

    parts = [f for f in m.files if f.get("part")]
    others = [f for f in m.files if not f.get("part")]

    if len(parts) > 1:
        target = dest / m.merge_name
        expect = sum(f["size"] for f in parts)
        if target.is_file() and target.stat().st_size == expect:
            print(f"    {target.name}: already complete")
        else:
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as fh:
                for f in parts:
                    download(f["url"], Path(f["name"]), f["size"], append_to=fh)
            tmp.replace(target)
            print(f"    joined {len(parts)} parts -> {target.name}")
    else:
        for f in parts:
            download(f["url"], dest / (m.merge_name or f["name"]), f["size"])

    for f in others:
        download(f["url"], dest / Path(f["name"]).name, f.get("size") or 0)

    if m.backend == "candle":
        tok = dest / "tokenizer.json"
        if not tok.is_file():
            print("    fetching tokenizer.json")
            download(_dl(m.repo, "tokenizer.json"), tok)
        shared = REPO / "data" / "tokenizers" / "qwen3-tokenizer.json"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tok, shared)
        print(f"    tokenizer -> {shared.relative_to(REPO)}")

    (dest / "thotbook-model.json").write_text(json.dumps({
        "key": m.key, "repo": m.repo, "backend": m.backend,
        "params": m.params, "model_id": m.repo,
        "size_gb": round(m.size_gb, 3),
        "weights": m.merge_name or None,
        "installed": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n")

    print(f"\n  done: {m.key} installed in {dest}")
    print(f"  serve:  make {'serve' if m.backend == 'candle' else 'dllama-serve'}"
          f" MODEL={m.key}")
    print(f"  wire:   make opencode-register MODEL={m.key}\n")
    return 0


def main() -> int:
    # Shared flags live on a parent parser so they work on either side of the
    # subcommand: `thotmodels.py --refresh list` and `... list --refresh`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-probe", action="store_true",
                        help="skip the bandwidth probe")
    common.add_argument("--refresh", action="store_true",
                        help="ignore cached HF sizes")

    p = argparse.ArgumentParser(
        parents=[common], description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", parents=[common], help="show the catalogue")
    pl.add_argument("--backend", choices=["candle", "dllama", "all"], default="all")
    pl.set_defaults(fn=cmd_list)

    pp = sub.add_parser("plan", parents=[common], help="show what setup would do")
    pp.add_argument("key")
    pp.set_defaults(fn=cmd_plan)

    ps = sub.add_parser("setup", parents=[common],
                        help="download and install a model")
    ps.add_argument("key")
    ps.add_argument("--yes", action="store_true", help="skip the confirmation")
    ps.add_argument("--dest", default="", help="override the destination dir")
    ps.set_defaults(fn=cmd_setup)

    args = p.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
