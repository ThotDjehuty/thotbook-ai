# ============================================================================
# thotbook-AmentI Makefile
# ============================================================================
# A private, local-first LLM training-to-serving pipeline in Rust.
# CPU-only, lakehouse-native, OpenAI-compatible.
#
#   make help            every target, grouped
#   make models          what you can download, with size + ETA + "does it fit"
#   make setup-model MODEL=dllama-qwen3-8b
#   make doctor          what is installed, running and reachable
#   make verify-paper    check the arXiv claims against this tree
# ============================================================================

SHELL := /bin/bash
PY    := python3 -I

# ---- paths ----------------------------------------------------------------
REPO            := $(CURDIR)
WORKSPACE       := $(abspath $(CURDIR)/..)
DLLAMA          ?= $(WORKSPACE)/distributed-llama-cloudnostic
POLARWAY        ?= $(WORKSPACE)/polarway
OPENCODE_CONFIG ?= $(WORKSPACE)/opencode.json

LAKEHOUSE   ?= data/lakehouse
HISTORIA_DIR?= $(WORKSPACE)/historia
HISTORIA_DB ?= data/ingest/historia.db
THOTBOOK_DB ?= $(WORKSPACE)/ThotCloud/services/thotbook-research-mcp/data/thotbook.db

# ---- defaults you will want to override -----------------------------------
MODEL     ?= qwen3-0.6b
TABLE     ?= datasets.sessions
TOKENIZED ?= data/tokenized/$(TABLE)
EPOCHS    ?= 3
LR        ?= 0.0002
MODE      ?= lora
RANK      ?= 16
HOST      ?= 127.0.0.1
PORT      ?= 8100
NTHREADS  ?= 10
WORKERS   ?=
STEPS     ?= 50

MODEL_DIR  = data/models/$(MODEL)
DLLAMA_BIN = $(DLLAMA)/dllama
DLLAMA_API = $(DLLAMA)/dllama-api

.DEFAULT_GOAL := help

.PHONY: all build release test check fmt clippy clean \
        models list-models plan-model setup-model setup-model-yes \
        serve serve-bg stop-serve dllama-build dllama-serve dllama-chat \
        opencode-register opencode-list opencode-remove \
        historia-db ingest-historia ingest tokenize \
        train train-historia train-bench train-custom \
        rag-build rag-search auto-research auto-extract auto-improve paper \
        status doctor agents smoke-test verify-paper verify-paper-json \
        docs docs-serve docs-clean install setup help

# ============================================================================
# Build
# ============================================================================

all: build

## Build all crates (debug)
build:
	cargo build --workspace

## Build all crates (release, optimised)
release:
	cargo build --workspace --release

## Run all tests
test:
	cargo test --workspace

## Type-check without building
check:
	cargo check --workspace

## Format all code
fmt:
	cargo fmt --all

## Run clippy with warnings as errors
clippy:
	cargo clippy --workspace -- -D warnings

## Remove build artifacts
clean:
	cargo clean

# ============================================================================
# Models  --  download and install weights
# ============================================================================
# `models` reads live sizes from the HuggingFace API, measures your actual
# bandwidth, and tells you whether each model fits THIS machine before you
# commit to a download. `setup-model` shows the same plan and waits for y/N.
#
# Backends differ in kind, not just degree:
#   candle  -> crates/llm-serve, F16 safetensors, Qwen3 architecture only
#   dllama  -> distributed-llama, 4-bit q40, tensor-parallel, inference only

## List every available model with size, ETA and whether it fits
models list-models:
	@$(PY) scripts/thotmodels.py list

## Show only models that llm-serve can load
models-candle:
	@$(PY) scripts/thotmodels.py list --backend candle

## Show only distributed-llama models
models-dllama:
	@$(PY) scripts/thotmodels.py list --backend dllama

## Show what downloading MODEL would cost, without downloading
plan-model:
	@$(PY) scripts/thotmodels.py plan $(MODEL)

## Download and install MODEL (prints the plan, then asks y/N)
setup-model:
	@$(PY) scripts/thotmodels.py setup $(MODEL)

## Same, without the confirmation prompt (for scripts and CI)
setup-model-yes:
	@$(PY) scripts/thotmodels.py setup $(MODEL) --yes

# ============================================================================
# Serving  --  candle (llm-serve)
# ============================================================================

## Start the OpenAI-compatible server on MODEL (foreground)
serve:
	@test -d "$(MODEL_DIR)" || { \
	  echo "no weights at $(MODEL_DIR)"; \
	  echo "run: make setup-model MODEL=$(MODEL)"; exit 1; }
	LLM_SERVE_MODEL_DIR=$(MODEL_DIR) \
	LLM_SERVE_MODEL_ID=$$($(PY) -c "import json;print(json.load(open('$(MODEL_DIR)/thotbook-model.json'))['model_id'])") \
	LLM_SERVE_HOST=$(HOST) LLM_SERVE_PORT=$(PORT) RUST_LOG=info \
	cargo run --release -p llm-cli -- serve --model-dir $(MODEL_DIR) \
	    --host $(HOST) --port $(PORT)

## Start the server in the background, logging to data/serve.log
serve-bg:
	@$(MAKE) --no-print-directory serve >data/serve.log 2>&1 & \
	echo "serving $(MODEL) on http://$(HOST):$(PORT)  (log: data/serve.log)"

## Stop a background server
stop-serve:
	@pkill -f "llm-cli -- serve" && echo "stopped" || echo "nothing to stop"

# ============================================================================
# Serving  --  distributed-llama (fast 4-bit, multi-node)
# ============================================================================
# This is the only path that runs a real 8B on a 32 GB laptop, and the one to
# use when you want speed. It does inference only: there is no backward pass
# anywhere in distributed-llama, so it cannot train or fine-tune.

## Compile the distributed-llama binaries
dllama-build:
	@test -d "$(DLLAMA)" || { echo "not found: $(DLLAMA)"; exit 1; }
	$(MAKE) -C $(DLLAMA) dllama dllama-api

## Serve MODEL through dllama-api on port 9990 (OpenAI-compatible)
dllama-serve: $(DLLAMA_API)
	@m=$$(ls $(MODEL_DIR)/*.m 2>/dev/null | head -1); \
	t=$$(ls $(MODEL_DIR)/*.t 2>/dev/null | head -1); \
	test -n "$$m" -a -n "$$t" || { \
	  echo "no dllama weights in $(MODEL_DIR)"; \
	  echo "run: make setup-model MODEL=dllama-qwen3-8b"; exit 1; }; \
	echo "serving $$m on http://127.0.0.1:9990/v1"; \
	$(DLLAMA_API) --model "$$m" --tokenizer "$$t" \
	    --buffer-float-type q80 --nthreads $(NTHREADS) --port 9990 \
	    $(if $(WORKERS),--workers $(WORKERS),)

## One-shot chat against MODEL through dllama (no server)
dllama-chat: $(DLLAMA_BIN)
	@m=$$(ls $(MODEL_DIR)/*.m 2>/dev/null | head -1); \
	t=$$(ls $(MODEL_DIR)/*.t 2>/dev/null | head -1); \
	$(DLLAMA_BIN) chat --model "$$m" --tokenizer "$$t" \
	    --buffer-float-type q80 --nthreads $(NTHREADS) \
	    $(if $(WORKERS),--workers $(WORKERS),)

$(DLLAMA_BIN) $(DLLAMA_API):
	@$(MAKE) --no-print-directory dllama-build

# ============================================================================
# opencode integration
# ============================================================================

## Add MODEL to opencode.json as a local provider (backs the file up first)
opencode-register:
	@$(PY) scripts/opencode_register.py --model $(MODEL) --config $(OPENCODE_CONFIG)

## Show installed models next to what opencode has registered
opencode-list:
	@$(PY) scripts/opencode_register.py --list --config $(OPENCODE_CONFIG)

## Print the opencode.json patch without writing it
opencode-dry-run:
	@$(PY) scripts/opencode_register.py --model $(MODEL) --dry-run \
	    --config $(OPENCODE_CONFIG)

## Unregister MODEL from opencode.json
opencode-remove:
	@$(PY) scripts/opencode_register.py --remove $(MODEL) --config $(OPENCODE_CONFIG)

# ============================================================================
# Corpus -> tokens -> training
# ============================================================================

## Rebuild a clean historia.db from $(HISTORIA_DIR)
# The historia-mcp copy is truncated at exactly 1e9 bytes and fails
# integrity_check, so we index the intact session files instead.
historia-db:
	@$(PY) scripts/build_historia_db.py --src $(HISTORIA_DIR) \
	    --out $(HISTORIA_DB) --force

## Ingest historia + thotbook into the Delta lakehouse (redacts, refuses leaks)
ingest-historia: $(HISTORIA_DB)
	@test -f "$(THOTBOOK_DB)" || { echo "missing $(THOTBOOK_DB)"; exit 1; }
	cargo run --release -p llm-corpus -- ingest \
	    --historia  $(HISTORIA_DB) \
	    --thotbook  $(THOTBOOK_DB) \
	    --lakehouse $(LAKEHOUSE)

$(HISTORIA_DB):
	@$(MAKE) --no-print-directory historia-db

## Tokenize TABLE from the lakehouse into sharded Arrow IPC
tokenize:
	cargo run --release -p llm-tokenize -- tokenize \
	    --lakehouse $(LAKEHOUSE) --table $(TABLE) --out $(TOKENIZED)

## Ingest arbitrary document directories (SOURCES=dir1 dir2 ...)
ingest:
	cargo run --release -p llm-cli -- ingest $(SOURCES)

## historia -> lakehouse -> tokens -> LoRA, end to end
train-historia: ingest-historia tokenize
	@$(MAKE) --no-print-directory train DATA=$(TOKENIZED)

## Train on DATA (LoRA on CPU by default)
train:
	@test -n "$(DATA)" || { echo "set DATA=<path>  (or use make train-historia)"; exit 1; }
	RUST_LOG=info cargo run --release -p llm-cli -- train \
	    --data $(DATA) --mode $(MODE) --epochs $(EPOCHS) --lr $(LR)

## Time STEPS training steps and extrapolate - the honest way to check
## the paper's "0.9 tok/s, 21 h for 3 epochs" without waiting 21 hours
train-bench:
	@test -n "$(DATA)" || { echo "set DATA=$(TOKENIZED) first"; exit 1; }
	RUST_LOG=info timeout_steps=$(STEPS) cargo run --release -p llm-cli -- train \
	    --data $(DATA) --mode $(MODE) --epochs 1 --lr $(LR)

## Train with every knob explicit
train-custom:
	cargo run --release -p llm-cli -- train \
	    --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --mode $(MODE)

# ============================================================================
# Owner-only learning loop  (thotbook-mcp -> polarway-learn -> thotbook-ai)
# ============================================================================
# The MCP stamps every lakehouse row with the owner's tenant id; polarway-learn
# admits only rows carrying that id, accumulates them as training shards, and
# reinforces the concepts the owner keeps asking about from references the MCP
# already holds. No other user's questions or references are ever read.

MCP_DIR       ?= $(WORKSPACE)/ThotCloud/services/thotbook-research-mcp
MCP_LAKEHOUSE ?= $(MCP_DIR)/data/lakehouse
LEARN_BIN     := cargo run --release --manifest-path $(POLARWAY)/polarway-learn/Cargo.toml \
                   --features loop --bin thotbook-learn --
LEARN_OUT     ?= data/learn/shards
BATCH         ?= 32

## End-to-end proof of the owner-only chain, in a scratch directory:
## MCP writes -> tenant stamped -> polarway-learn admits only yours ->
## shards audited -> llm-train refuses a tainted one. Repeatable, writes
## nothing to the real lakehouse, shards or model.
chain-verify:
	@./scripts/verify_chain.sh

## What the lakehouse holds and how much of it is attributable to you
learn-census:
	@$(LEARN_BIN) --lakehouse $(MCP_LAKEHOUSE) --out $(LEARN_OUT) --db $(THOTBOOK_DB) \
	    --batch $(BATCH) --json

## Collect your rows into training shards (batch mode)
learn:
	@$(LEARN_BIN) --lakehouse $(MCP_LAKEHOUSE) --out $(LEARN_OUT) --db $(THOTBOOK_DB) \
	    --batch $(BATCH)

## Same, flushing a shard per sample - online learning
learn-online:
	@$(LEARN_BIN) --lakehouse $(MCP_LAKEHOUSE) --out $(LEARN_OUT) --db $(THOTBOOK_DB) --batch 1

## Also search the shared corpus for weak concepts (off by default: the
## question is yours, but which documents come back depends on what others
## have fetched)
learn-wide:
	@$(LEARN_BIN) --lakehouse $(MCP_LAKEHOUSE) --out $(LEARN_OUT) --db $(THOTBOOK_DB) \
	    --batch $(BATCH) --corpus-search

## Fine-tune on your shards only. Refuses the run if a shard contains a
## sample belonging to anyone else.
train-owner:
	@tenant="$(THOTBOOK_TENANT_ID)"; \
	if [ -z "$$tenant" ] && [ -f "$(MCP_DIR)/data/tenant.json" ]; then \
	    tenant=$$($(PY) -c 'import json,sys;print(json.load(open(sys.argv[1]))["tenant_id"])' \
	             "$(MCP_DIR)/data/tenant.json"); \
	fi; \
	test -n "$$tenant" || { \
	    echo "no tenant: set THOTBOOK_TENANT_ID, or run the MCP once so it writes"; \
	    echo "$(MCP_DIR)/data/tenant.json"; exit 1; }; \
	echo "training as $$tenant"; \
	RUST_LOG=info cargo run --release -p llm-cli -- train \
	    --owner-shards $(LEARN_OUT) --tenant "$$tenant" \
	    --model-dir data/models/$(MODEL) \
	    --mode $(MODE) --epochs $(EPOCHS) --lr $(LR)

## learn + train in one go
learn-and-train: learn train-owner

# ============================================================================
# RAG
# ============================================================================

## Build the RAG index from INPUT
rag-build:
	cargo run --release -p llm-cli -- rag build --input $(INPUT)

## Search the index (QUERY=... TOP_K=5)
rag-search:
	cargo run --release -p llm-cli -- rag search "$(QUERY)" --top-k $(or $(TOP_K),5)

# ============================================================================
# Autonomous research
# ============================================================================

## Run autonomous research on TOPIC
auto-research:
	cargo run --release -p llm-cli -- auto research "$(TOPIC)"

## Extract training data from the session corpus
auto-extract:
	cargo run -p llm-cli -- auto extract --session-corpus $(or $(SESSION_CORPUS),../session-corpus)

## Check the self-improvement loop
auto-improve:
	cargo run -p llm-cli -- auto improve --data-dir $(or $(DATA_DIR),data)

## Generate a research paper on TOPIC
paper:
	cargo run --release -p llm-cli -- paper "$(TOPIC)"

# ============================================================================
# Status and verification
# ============================================================================

## CLI status
status:
	cargo run -p llm-cli -- status

## List available agents
agents:
	cargo run -p llm-cli -- agent

## What is installed, what is running, what is reachable
doctor:
	@echo ""
	@echo "  host"
	@printf "    %-22s %s\n" cpu "$$(sysctl -n machdep.cpu.brand_string)"
	@printf "    %-22s %s threads, %s GiB RAM\n" resources \
	    "$$(sysctl -n hw.ncpu)" "$$(($$(sysctl -n hw.memsize)/1073741824))"
	@printf "    %-22s %s free\n" disk "$$(df -h . | tail -1 | awk '{print $$4}')"
	@echo ""
	@echo "  toolchain"
	@printf "    %-22s %s\n" cargo "$$(cargo --version 2>/dev/null || echo MISSING)"
	@printf "    %-22s %s\n" python "$$(python3 -V 2>&1)"
	@printf "    %-22s %s\n" dllama \
	    "$$(test -x $(DLLAMA_BIN) && echo built || echo 'not built (make dllama-build)')"
	@printf "    %-22s %s\n" polarway \
	    "$$(test -d $(POLARWAY)/polarway-lakehouse && echo resolved || echo MISSING)"
	@echo ""
	@echo "  corpus"
	@printf "    %-22s %s\n" "historia dir" \
	    "$$(test -d $(HISTORIA_DIR) && echo "$$(ls $(HISTORIA_DIR) | wc -l | tr -d ' ') files" || echo MISSING)"
	@printf "    %-22s %s\n" "historia.db" \
	    "$$(test -f $(HISTORIA_DB) && echo "$$(sqlite3 $(HISTORIA_DB) 'select count(*) from documents' 2>/dev/null) docs, integrity $$(sqlite3 $(HISTORIA_DB) 'pragma integrity_check' 2>/dev/null)" || echo 'not built (make historia-db)')"
	@printf "    %-22s %s\n" "thotbook.db" \
	    "$$(test -f $(THOTBOOK_DB) && sqlite3 $(THOTBOOK_DB) 'pragma integrity_check' 2>/dev/null || echo MISSING)"
	@printf "    %-22s %s\n" lakehouse \
	    "$$(test -d $(LAKEHOUSE) && echo "$$(ls $(LAKEHOUSE) 2>/dev/null | wc -l | tr -d ' ') tables" || echo 'not built (make ingest-historia)')"
	@echo ""
	@echo "  models installed"
	@found=0; for d in data/models/*/; do \
	  [ -f "$$d/thotbook-model.json" ] || continue; \
	  found=1; m=$$(basename $$d); \
	  printf "    %-22s %-7s %s\n" "$$m" \
	    "$$($(PY) -c "import json;print(json.load(open('$$d/thotbook-model.json'))['backend'])" 2>/dev/null)" \
	    "$$(du -sh $$d 2>/dev/null | cut -f1)"; \
	done; [ $$found = 1 ] || echo "    (none yet - run: make models)"
	@for d in data/models/*/; do \
	  [ -d "$$d" ] && [ ! -f "$$d/thotbook-model.json" ] && \
	    echo "    (incomplete: $$(basename $$d) - no thotbook-model.json)" || true; \
	done
	@echo ""
	@echo "  endpoints"
	@printf "    %-22s %s\n" "llm-serve :$(PORT)" \
	    "$$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://$(HOST):$(PORT)/health 2>/dev/null | grep -q 200 && echo up || echo down)"
	@printf "    %-22s %s\n" "dllama-api :9990" \
	    "$$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:9990/v1/models 2>/dev/null | grep -qE '200|40' && echo up || echo down)"
	@echo ""

## Check every falsifiable claim in papers/thotbook_ai_arxiv.pdf
verify-paper:
	@$(PY) scripts/verify_paper.py

## Same, as JSON
verify-paper-json:
	@$(PY) scripts/verify_paper.py --json

## Fast subset (skips generation timing)
verify-paper-quick:
	@$(PY) scripts/verify_paper.py --quick

## Build then check the CLI answers
smoke-test: build
	cargo run -p llm-cli -- status
	@echo "smoke test passed."

# ============================================================================
# Documentation
# ============================================================================

## Build Sphinx docs
docs:
	cd docs && $(MAKE) html

## Serve docs on :8000
docs-serve:
	cd docs/build/html && python3 -m http.server 8000

## Clean docs
docs-clean:
	cd docs && $(MAKE) clean

# ============================================================================
# Install / setup
# ============================================================================

## Install the CLI globally
install:
	cargo install --path crates/llm-cli

## Create the lakehouse directory skeleton
setup-lakehouse:
	@mkdir -p data/lakehouse data/training data/ingest data/models data/tokenizers
	@echo "lakehouse directories ready."

## Build, create directories, then show the model menu
setup: setup-lakehouse build
	@echo ""
	@echo "Built. Now pick a model:"
	@$(PY) scripts/thotmodels.py list

# ============================================================================
# Help
# ============================================================================

## Show this help
help:
	@echo ""
	@echo "  thotbook-AmentI  --  private CPU LLM pipeline"
	@echo ""
	@echo "  FIRST RUN"
	@echo "    make setup                     build + show the model menu"
	@echo "    make models                    sizes, ETA, and what fits 32 GB"
	@echo "    make setup-model MODEL=<key>   plan + y/N, then download"
	@echo "    make doctor                    what is installed and running"
	@echo ""
	@echo "  SERVE (pick one)"
	@echo "    make serve MODEL=qwen3-1.7b           candle, F16, :8100"
	@echo "    make dllama-serve MODEL=dllama-qwen3-8b   4-bit, fast, :9990"
	@echo "    make serve-bg / make stop-serve       background the candle one"
	@echo ""
	@echo "  USE FROM OPENCODE"
	@echo "    make opencode-register MODEL=<key>    add to opencode.json"
	@echo "    make opencode-list                    installed vs registered"
	@echo ""
	@echo "  TRAIN ON YOUR OWN HISTORY"
	@echo "    make historia-db               rebuild the session index"
	@echo "    make ingest-historia           -> Delta lakehouse (redacted)"
	@echo "    make tokenize                  -> Arrow shards"
	@echo "    make train-historia            all three, then LoRA"
	@echo "    make train-bench DATA=...      time STEPS=50 steps, extrapolate"
	@echo ""
	@echo "  BUILD / QA"
	@echo "    make build release test check fmt clippy clean"
	@echo "    make verify-paper              arXiv claims vs this tree"
	@echo ""
	@echo "  RAG / RESEARCH"
	@echo "    make rag-build INPUT=...       make rag-search QUERY='...'"
	@echo "    make auto-research TOPIC='...' make paper TOPIC='...'"
	@echo ""
	@echo "  Variables: MODEL TABLE DATA EPOCHS LR MODE RANK HOST PORT"
	@echo "             NTHREADS WORKERS STEPS LAKEHOUSE HISTORIA_DIR"
	@echo ""
	@echo "  Multi-node dllama (worker on the VPS, root here):"
	@echo "    on the VPS:  ./dllama worker --port 9999 --nthreads 4"
	@echo "    here:        make dllama-serve MODEL=dllama-qwen3-8b \\"
	@echo "                     WORKERS=VPS_HOST:9999"
	@echo ""
