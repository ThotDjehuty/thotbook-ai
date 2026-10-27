#!/usr/bin/env bash
# End-to-end proof of the owner-only learning chain, from one command.
#
#   MCP writes a query  →  lakehouse row carries a tenant
#                       →  polarway-learn admits only the owner's
#                       →  JSONL shards, audited
#                       →  llm-train loads them as supervised examples
#
# Everything happens in a scratch directory. The real lakehouse, the real
# shards and the real model are never written to. Run it as often as you like.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
MCP="$WORKSPACE/ThotCloud/services/thotbook-research-mcp"
POLARWAY="$WORKSPACE/polarway"
OWNER="${OWNER_TENANT:-owner@example.org}"
INTRUDER="rival@othercorp.example"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/thot-chain.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
LAKE="$WORK/lakehouse"; SHARDS="$WORK/shards"
mkdir -p "$LAKE"

pass=0; fail=0
ok()   { printf "  \033[32m✅\033[0m %s\n" "$1"; pass=$((pass+1)); }
no()   { printf "  \033[31m❌\033[0m %s\n" "$1"; fail=$((fail+1)); }
step() { printf "\n\033[1m%s\033[0m\n" "$1"; }

# ── 1. the MCP writes, and stamps identity without being asked ──────────────
step "1 · the MCP stamps a tenant on every row it writes"
[ -d "$MCP/dist" ] || (cd "$MCP" && npx --no-install tsc >/dev/null 2>&1 || node node_modules/typescript/bin/tsc >/dev/null)
cat > "$WORK/write.mjs" <<'JS'
import { LakehouseBridge } from process.env.BRIDGE;
import { loadConfig } from process.env.CFG;
const b = new LakehouseBridge(loadConfig());
for (const q of JSON.parse(process.argv[2])) await b.logQuery(q);
JS
# (import() needs literals; generate the file with the paths baked in)
cat > "$WORK/write.mjs" <<JS
import { LakehouseBridge } from '$MCP/dist/lakehouse-bridge.js';
import { loadConfig } from '$MCP/dist/config.js';
const b = new LakehouseBridge(loadConfig());
for (const q of JSON.parse(process.argv[2])) await b.logQuery(q);
JS

OWNER_ROWS='[
 {"query_text":"hermitian yang-mills stability on the torus","query_type":"arxiv_search","result_count":2,"result_ids":["1906.00992v3","1907.10651v2"]},
 {"query_text":"hermitian yang-mills and the donaldson-uhlenbeck-yau theorem","query_type":"arxiv_search","result_count":0,"result_ids":[]},
 {"query_text":"wheeler-dewitt factor ordering in minisuperspace","query_type":"arxiv_search","result_count":0,"result_ids":[]}
]'
THOTBOOK_LAKEHOUSE_PATH="$LAKE" THOTBOOK_TENANT_ID="$OWNER" \
  node "$WORK/write.mjs" "$OWNER_ROWS" >/dev/null
THOTBOOK_LAKEHOUSE_PATH="$LAKE" THOTBOOK_TENANT_ID="$INTRUDER" \
  node "$WORK/write.mjs" '[{"query_text":"their-private-question","query_type":"arxiv_search","result_count":5,"result_ids":["x"]}]' >/dev/null

stamped=$(python3 - "$LAKE" <<'PY'
import glob, sys, collections
try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow missing")
c = collections.Counter()
for f in glob.glob(f"{sys.argv[1]}/research_queries/**/*.parquet", recursive=True):
    for v in pq.read_table(f).to_pydict().get("user_context", []):
        c[v] += 1
print(f"{c.get(sys.argv[2] if len(sys.argv)>2 else '', 0)}|{sum(c.values())}|{len([k for k in c if k is None])}")
PY
) || { no "could not read the lakehouse back (is pyarrow installed?)"; exit 1; }
total=${stamped#*|}; nulls=${total#*|}; total=${total%%|*}
[ "$total" = "4" ] && ok "4 rows written" || no "expected 4 rows, got $total"
[ "$nulls" = "0" ] && ok "none has a NULL user_context" || no "$nulls row(s) unattributed"

# ── 2. polarway-learn admits the owner's rows and only those ────────────────
step "2 · polarway-learn admits the owner, never the intruder"
REPORT="$WORK/report.json"
cargo run --quiet --release --manifest-path "$POLARWAY/polarway-learn/Cargo.toml" \
    --features loop --bin thotbook-learn -- \
    --lakehouse "$LAKE" --out "$SHARDS" --db "$MCP/data/thotbook.db" \
    --tenant "$OWNER" --batch 4 --json > "$REPORT" 2>/dev/null

python3 - "$REPORT" "$SHARDS" "$OWNER" "$INTRUDER" <<'PY' && ok "report consistent" || no "report inconsistent"
import json, sys, glob, pathlib
rep = json.load(open(sys.argv[1])); shards, owner, intruder = sys.argv[2], sys.argv[3], sys.argv[4]
assert rep["admitted"] == 3, f'admitted {rep["admitted"]}, expected 3'
assert rep["census"]["research_queries"]["mine"] == 3
assert rep["census"]["research_queries"]["other_tenants"] == 1
assert rep["audit_clean"] is True, "shard audit failed"
blob = "".join(pathlib.Path(f).read_text() for f in glob.glob(f"{shards}/*.jsonl"))
assert intruder not in blob, "the intruder's tenant reached a shard"
assert "their-private-question" not in blob, "the intruder's text reached a shard"
assert owner in blob
PY
admitted=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['admitted'])" "$REPORT")
[ "$admitted" = "3" ] && ok "3 admitted, 1 refused (never read)" || no "admitted=$admitted"

# ── 3. the shards are trainable, and only the owner's ───────────────────────
step "3 · llm-train accepts the shards, and refuses a foreign one"
TOK="$WORKSPACE/thotbook-ai/data/tokenizers/qwen3-tokenizer.json"
if [ ! -f "$TOK" ]; then
  printf "  \033[33m–\033[0m no tokenizer at %s; skipping the training link\n" "$TOK"
else
  # NOT `| grep -q`: grep closes the pipe on its first match, cargo dies of
  # SIGPIPE, and `set -o pipefail` then reports the whole pipeline as failed.
  # That produced a red line for a suite that was passing.
  cargo test --quiet --manifest-path "$WORKSPACE/thotbook-ai/Cargo.toml" \
      -p llm-train samples:: > "$WORK/samples.log" 2>&1 || true
  if grep -q "test result: ok" "$WORK/samples.log"; then
      n=$(grep -oE '[0-9]+ passed' "$WORK/samples.log" | head -1)
      ok "llm-train sample tests pass ($n — tenant refusal, prompt masking)"
  else
      no "llm-train sample tests failed (see $WORK/samples.log)"
  fi
  # a planted foreign sample must stop the load
  cp -R "$SHARDS" "$WORK/tainted"
  printf '{"tenant":"%s","kind":"supervised","prompt":"x","completion":"y"}\n' "$INTRUDER" \
      >> "$WORK/tainted/samples-00000.jsonl"
  if cargo run --quiet --release --manifest-path "$WORKSPACE/thotbook-ai/Cargo.toml" \
        -p llm-cli -- train --owner-shards "$WORK/tainted" --tenant "$OWNER" \
        --model-dir "$WORKSPACE/thotbook-ai/data/models/qwen3-0.6b" --epochs 1 >/dev/null 2>&1; then
      no "a tainted shard was accepted"
  else
      ok "a tainted shard stops the run"
  fi
fi

# ── 4. reinforcement turns a gap into real training material ───────────────
step "4 · a concept the owner keeps missing is reinforced from the local corpus"
if [ ! -f "$MCP/data/thotbook.db" ]; then
  printf "  \033[33m–\033[0m no MCP reference store; skipping reinforcement\n"
else
  LAKE2="$WORK/lakehouse2"; SHARDS2="$WORK/shards2"; mkdir -p "$LAKE2"
  # One paper found, then asked twice more with nothing: confidence 0.20.
  # The id is UNVERSIONED on purpose — the store holds a vN suffix, and an
  # exact match used to report "no local material" for a paper sitting right
  # there in the same database.
  REF_ID="${REF_ID:-2608.10133}"
  cat > "$WORK/weak.mjs" <<JS
import { LakehouseBridge } from '$MCP/dist/lakehouse-bridge.js';
import { loadConfig } from '$MCP/dist/config.js';
const b = new LakehouseBridge(loadConfig());
for (let i = 0; i < 3; i++) {
  await b.logQuery({ query_text: 'yang-mills mass gap confinement', query_type: 'arxiv_search',
                     result_count: i === 0 ? 1 : 0, result_ids: i === 0 ? ['$REF_ID'] : [] });
}
JS
  THOTBOOK_LAKEHOUSE_PATH="$LAKE2" THOTBOOK_TENANT_ID="$OWNER" node "$WORK/weak.mjs" >/dev/null 2>&1
  cargo run --quiet --release --manifest-path "$POLARWAY/polarway-learn/Cargo.toml" \
      --features loop --bin thotbook-learn -- \
      --lakehouse "$LAKE2" --out "$SHARDS2" --db "$MCP/data/thotbook.db" \
      --tenant "$OWNER" --batch 4 --json > "$WORK/report2.json" 2>/dev/null

  python3 - "$WORK/report2.json" "$SHARDS2" > "$WORK/reinforce.log" 2>&1 <<'CHECK'
import json, sys, glob
rep = json.load(open(sys.argv[1]))
assert rep["reinforcement"]["samples"] >= 1, "reinforcement produced nothing"
sup = [json.loads(l) for f in glob.glob(f"{sys.argv[2]}/*.jsonl") for l in open(f)]
sup = [s for s in sup if s["kind"] == "supervised"]
assert sup, "no supervised sample"
assert len(sup[0]["completion"]) > 120, "the abstract did not come through"
assert sup[0]["references"], "the sample cites no reference"
print(sup[0]["references"][0])
CHECK
  if [ $? = 0 ]; then
      ok "a weak concept pulled a real abstract in, citing $(cat "$WORK/reinforce.log")"
  else
      no "reinforcement produced no usable material ($(tail -1 "$WORK/reinforce.log"))"
  fi
fi

if [ "$fail" = "0" ]; then
    printf "\n\033[1;32mCHAIN OK — %d checks passed\033[0m\n" "$pass"
else
    printf "\n\033[1;31mCHAIN BROKEN — %d passed, %d failed\033[0m\n" "$pass" "$fail"
fi
[ "$fail" = "0" ]
