#!/usr/bin/env python3
"""Register a locally-served thotbook-ai model with opencode.

Both backends expose an OpenAI-compatible /v1, so opencode talks to them
through @ai-sdk/openai-compatible:

  candle  crates/llm-serve            http://127.0.0.1:8100/v1
  dllama  ../distributed-llama...     http://127.0.0.1:9990/v1   (dllama-api)

This edits opencode.json in place, additively: it touches only
provider.<id>.models.<model> and leaves every other provider, MCP server and
plugin exactly as it was. A timestamped backup is written first, and the
result is re-parsed before it replaces the original -- a malformed write would
take opencode down with it.

Usage:
    opencode_register.py --model qwen3-0.6b [--config PATH] [--dry-run]
    opencode_register.py --list
    opencode_register.py --remove qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO.parent / "opencode.json"

BACKENDS = {
    "candle": {
        "provider_id": "thotbook",
        "provider_name": "ThotBook-AI (local, candle)",
        "base_url": "http://127.0.0.1:8100/v1",
        "api_key": "thotbook-local",
        "context": 16384,
        "output": 2048,
    },
    "dllama": {
        "provider_id": "thotbook-dllama",
        "provider_name": "ThotBook-AI (local, distributed-llama)",
        "base_url": "http://127.0.0.1:9990/v1",
        "api_key": "dllama-local",
        "context": 4096,   # launch.py pins --max-seq-len 4096
        "output": 2048,
    },
}


def installed() -> dict[str, dict]:
    """Every model with a thotbook-model.json under data/models."""
    out = {}
    root = REPO / "data" / "models"
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        meta = d / "thotbook-model.json"
        if meta.is_file():
            try:
                blob = json.loads(meta.read_text())
                blob["dir"] = str(d)
                out[blob.get("key", d.name)] = blob
            except Exception:
                continue
    return out


def probe(url: str, timeout: float = 2.0) -> str:
    """'up' / 'down' -- purely informational, never blocks registration."""
    try:
        req = urllib.request.Request(url.rstrip("/") + "/models",
                                     headers={"User-Agent": "thotbook-ai"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return "up" if r.status == 200 else f"http {r.status}"
    except urllib.error.HTTPError as e:
        return f"http {e.code}"   # answering, just not that route
    except Exception:
        return "down"


def load_config(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"opencode config not found: {path}\n"
                 "Pass --config with its real location.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON ({e}).\n"
                 "Fix it by hand first - refusing to overwrite a broken config.")


def save_config(path: Path, cfg: dict) -> Path:
    body = json.dumps(cfg, indent=2) + "\n"
    json.loads(body)  # never write something we cannot read back
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copyfile(path, backup)
    path.write_text(body)
    return backup


def cmd_list(args) -> int:
    cfg = load_config(Path(args.config))
    inst = installed()
    print(f"\nconfig: {args.config}")
    print(f"models dir: {REPO / 'data' / 'models'}\n")

    print("installed locally:")
    if not inst:
        print("  (none yet - run `make setup-model MODEL=<key>`)")
    for k, m in inst.items():
        print(f"  {k:24s} {m.get('params', '?'):5s} {m.get('backend', '?'):7s} "
              f"{m.get('size_gb', '?')} GB")

    print("\nregistered in opencode:")
    provs = cfg.get("provider", {})
    any_reg = False
    for b in BACKENDS.values():
        p = provs.get(b["provider_id"])
        if not p:
            continue
        state = probe(b["base_url"])
        for mid, mm in (p.get("models") or {}).items():
            any_reg = True
            print(f"  {b['provider_id']}/{mid:22s} {b['base_url']}  [{state}]")
    if not any_reg:
        print("  (none)")
    print()
    return 0


def cmd_register(args) -> int:
    path = Path(args.config)
    cfg = load_config(path)
    inst = installed()

    key = args.model
    meta = inst.get(key)
    if meta is None:
        print(f"'{key}' is not installed under data/models.", file=sys.stderr)
        if inst:
            print("installed: " + ", ".join(inst), file=sys.stderr)
        print(f"run:  make setup-model MODEL={key}", file=sys.stderr)
        return 1

    backend = meta.get("backend", "candle")
    b = BACKENDS.get(backend)
    if b is None:
        sys.exit(f"unknown backend '{backend}' in {meta['dir']}/thotbook-model.json")

    prov = cfg.setdefault("provider", {}).setdefault(b["provider_id"], {})
    prov["npm"] = "@ai-sdk/openai-compatible"
    prov["name"] = b["provider_name"]
    opts = prov.setdefault("options", {})
    opts["baseURL"] = b["base_url"]
    opts.setdefault("apiKey", b["api_key"])

    models = prov.setdefault("models", {})
    models[key] = {
        "name": f"{meta.get('repo', key)} ({meta.get('params', '?')}, local)",
        "limit": {"context": b["context"], "output": b["output"]},
        # llm-serve has no function-calling surface; dllama-api has none either.
        "tool_call": False,
        "temperature": True,
    }

    if args.dry_run:
        print(json.dumps({"provider": {b["provider_id"]: prov}}, indent=2))
        print("\n(dry run: nothing written)")
        return 0

    backup = save_config(path, cfg)
    state = probe(b["base_url"])
    print(f"\n  registered  {b['provider_id']}/{key}")
    print(f"  endpoint    {b['base_url']}  [{state}]")
    print(f"  config      {path}")
    print(f"  backup      {backup.name}")
    if state != "up":
        serve = "make serve" if backend == "candle" else "make dllama-serve"
        print(f"\n  server is {state}; start it with:  {serve} MODEL={key}")
    print(f"\n  use in opencode:  /model {b['provider_id']}/{key}\n")
    return 0


def cmd_remove(args) -> int:
    path = Path(args.config)
    cfg = load_config(path)
    removed = []
    for b in BACKENDS.values():
        prov = (cfg.get("provider") or {}).get(b["provider_id"])
        if not prov:
            continue
        if args.remove in (prov.get("models") or {}):
            del prov["models"][args.remove]
            removed.append(f"{b['provider_id']}/{args.remove}")
            if not prov["models"]:
                del cfg["provider"][b["provider_id"]]
                removed.append(f"(empty provider {b['provider_id']} dropped)")
    if not removed:
        print(f"'{args.remove}' was not registered.")
        return 1
    backup = save_config(path, cfg)
    print("removed: " + ", ".join(removed))
    print(f"backup:  {backup.name}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"opencode.json path (default: {DEFAULT_CONFIG})")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="installed model key to register")
    g.add_argument("--list", action="store_true", help="show installed vs registered")
    g.add_argument("--remove", metavar="KEY", help="unregister a model")
    p.add_argument("--dry-run", action="store_true",
                   help="print the patch without writing")
    args = p.parse_args()

    if args.list:
        return cmd_list(args)
    if args.remove:
        return cmd_remove(args)
    return cmd_register(args)


if __name__ == "__main__":
    sys.exit(main())
