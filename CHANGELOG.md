# Changelog

All notable changes to thotbook-ai are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-10-20

### Fixed — `thotbook serve` never worked

- **Nested tokio runtime.** `llm-cli` is `#[tokio::main]` but called
  `llm_serve::run_server`, which builds its own runtime and `block_on`s it, so
  the process panicked with *"Cannot start a runtime from within a runtime"*
  before the socket was ever bound. It had already logged `listening`, so it
  looked like a healthy server refusing every connection. Split into an async
  `serve()` the CLI awaits and a sync `run_server()` wrapper for the
  standalone binary, and moved the log to after the bind.
- **Every completion returned HTTP 500.** `llm-serve` depends on
  `accelerate-src`, which enables candle's Accelerate backend, and Accelerate
  has no f16 matmul while `engine.rs` pinned `DType::F16`:
  *"the accelerate backend does not support f16 matmul"*. `/health` and
  `/v1/models` answered fine, so the break only showed on real traffic. Now
  F32 by default, overridable with `LLM_SERVE_DTYPE=f16|bf16|f32`.
- **`/v1/models` reported the wrong model.** The CLI built `ServeConfig`
  literally, pinning `model_id` to the `Qwen/Qwen3-8B` default whatever was
  loaded. It now starts from `from_env()`.

### Changed — the training loss comes from optimiz-rs

- `llm-train`'s `cross_entropy_loss` returned `+mean(log q)` instead of
  `-mean(log q)`: the negation was missing, so minimising it trained the model
  *away* from its targets, and its unit test asserted the wrong sign. It also
  ignored its `label_smoothing` argument and offered no masking despite the
  paper's §3.1 being titled *Cross-Entropy Loss with Masking*.
- `crates/llm-train/src/loss.rs` is now a thin adapter over
  `optimizr::losses`, so there is one audited definition of cross-entropy in
  the stack. It returns `d(loss)/d(logits)`, which the trainer needed and
  lacked. A parity test asserts it matches the trainer's candle path.

### Added

- **`notebooks/09_pinn_physics_optimizr.ipynb`** — physics-informed neural
  networks for the Wheeler--DeWitt equation, Hermitian Yang--Mills, the Dirac
  equation, generalized Dyson Brownian motion and the McKean--Vlasov equation
  (the last two following [arXiv:1303.1240](https://arxiv.org/abs/1303.1240)),
  trained on CPU with `thotgrad` for AD, `thotopt` L-BFGS, and **every**
  objective from `optimiz-rs`. Each PINN is validated against an independent
  reference, not against itself.
- **`notebooks/08_thotbook_ai_guide.ipynb`** — the runnable reference: which
  tests pass, the equations with the line implementing each, five benchmarks
  measured on the host, and the ordered steps to train and serve.
- **`scripts/thotmodels.py`** and `make models` / `setup-model` — model
  catalogue with live HuggingFace sizes, a measured-bandwidth ETA and a
  per-host memory verdict, confirmed with y/N before downloading. Stdlib only.
- **`scripts/opencode_register.py`** and `make opencode-register` — additive
  registration of a served model in `opencode.json`, with a backup first.
- **`scripts/build_historia_db.py`** and `make historia-db` — rebuilds the
  session index; the `historia-mcp` copy is truncated at exactly 1e9 bytes and
  fails `pragma integrity_check`.
- **`scripts/verify_paper.py`** and `make verify-paper` — checks each
  falsifiable claim in the arXiv paper, reporting PASS / FAIL / BLOCKED /
  ABSENT with evidence, and exits non-zero on FAIL or ABSENT.
- Characterization tests pinning a defect that is **not** yet fixed:
  `HnswIndex::search` discards its query vector, so every query returns the
  same neighbourhood and Table 3's HNSW-only Recall@5 cannot stand.
- `nblite.py` now captures matplotlib figures as image outputs, emits valid
  `markdown` cell types rather than `md`, adds cell ids, and names the failing
  cell on error.

### Notes

- 97 tests pass, up from 81.
- Known open defect: `crates/llm-rag/src/hnsw.rs` — see the characterization
  suite. The trainer also never calls `update_param`, so training currently
  evaluates a loss without updating parameters.

## [0.3.0] - 2026-09-25

### Added

- **`thotgrad`** — array-level reverse-mode automatic differentiation in
  NumPy (~250 lines), replacing PyTorch entirely. Input derivatives are
  propagated analytically through the tanh recursion and stay on the same
  tape as the value, so one reverse sweep yields parameter gradients of a
  loss containing `u`, `u_x` and `u_xx` (the PINN double-backward).
- **`thotopt`** — AdamW (decoupled decay, mirroring the Rust optimiser) and
  L-BFGS (Nocedal two-loop recursion + Armijo line search), filling the
  gradient-based gap left by optimiz-rs's gradient-free optimisers.
- **`test_thotgrad.py`** — every gradient pinned against central finite
  differences: elementary ops ~1e-10, `u_x` 1e-12, `u_xx` 1e-6, full PINN
  parameter gradient 1.8e-11 max-norm with cosine similarity 1.000000;
  optimisers checked against closed-form optima.

- **Physics-Informed Neural Networks** (`notebooks/04_pinn_physics.ipynb`,
  `pinn_core.py`, `pinn_problems.py`) — solves the Dirac, Brans-Dicke and
  Wheeler-DeWitt equations on CPU, each graded against an independent
  reference (exact closed form / exact Nariai power law / Radau at
  `rtol=1e-12` cross-checked by an Airy asymptotic). Relative L2 errors
  1.7e-3, 4.3e-3 and 8.3e-6 respectively.
- **Membership inference at scale** (`notebooks/02_mia_real_corpus.ipynb`) —
  the privacy audit run over a real research corpus with document-level
  splits, calibrated membership probabilities (Platt scaling + reliability
  diagram + ECE) and block-bootstrap CIs.
- **Federated learning** (`notebooks/03_federated_ensemble.ipynb`,
  `fed_core.py`) — FedAvg with an honest-but-curious server running the
  update-based attack, plus the four defence families of Bai et al.'s survey
  (partial sharing, secure aggregation, noise perturbation, anomaly
  detection) and the measured privacy/utility trade-off.
- `docs/source/algorithms/variational_calculus.rst` — Euler-Lagrange,
  gradient flow, the ELBO, and the mapping from each functional-minimisation
  problem to its optimiz-rs primitive.
- `docs/source/algorithms/membership_inference.rst` — attack/defence theory
  with citations (Shokri 2017, Yeom 2018, Carlini 2022).
- Explicit AdamW update equations in `docs/source/algorithms/training.rst`.
- `crates/llm-train/examples/membership_probe.rs` — real LoRA training probe
  (candle autograd + `AdamWOptimizer`) with `--hidden` / `--limit` controls.

### Fixed

- **`AdamWOptimizer` never applied weight decay.** The `weight_decay` field
  was stored but unused, making the optimiser plain Adam despite the name and
  the documented equation. Decay is now applied decoupled (Loshchilov &
  Hutter 2019), with a regression test that fails if it regresses.
- All 33 `cargo clippy --workspace --all-targets -- -D warnings` errors
  (manual prefix stripping, private-type-in-public-field, `&PathBuf` over
  `&Path`, `loop`/`match` over `while let`).

### Changed

- Workspace version `0.2.0` → `0.3.0`.
- `notebooks/data/` is gitignored: it holds verbatim text extracted from
  private and third-party copyrighted PDFs and must never reach a public
  remote. Regenerate locally via `THOTBOOK_PAPERS_DIR=... python3
  notebooks/extract_corpus.py`.
- `papers/thotbook_ai_arxiv.tex`: added a System Capabilities section
  covering the autodiff engine, optimisers, document verification by equation
  solving, the privacy audit and federated training, plus an explicit
  negative-results subsection. Corrected the hardware section, which
  claimed an Apple M1 Max; all measurements in this repo were taken on an
  Intel Core i9-8950HK with no GPU. Added a verified-benchmarks section and
  an explicit threats-to-validity subsection marking the inherited throughput
  and retrieval tables as provisional and not re-measured.

### Notes

- 71/71 Rust tests pass; `cargo clippy -D warnings` is clean.
- PyTorch is no longer a dependency anywhere in the project.

## [0.2.0] - 2026-09-25

### Added

- **`llm-auto`** — autonomous orchestration crate (M7): task decomposition,
  sub-agent spawning, self-training loop, and paper generation from the
  research pipeline. Training data is sourced from a locally configured
  session-log corpus directory (`--session-corpus`, generic Markdown
  `## Goal` / `## Summary` documents — no specific dataset is bundled or
  disclosed).
- **`llm-ingest`** — universal document ingestion crate: PDF, Markdown, and
  agent-definition file parsing into the corpus lakehouse.
- **`llm-salviers`** — agent dispatch system: semantic (keyword/BM25) matching
  of incoming queries to registered agent definitions.
- `llm-rag`: dedicated `embed`, `error`, `hnsw`, `search`, and `store` modules,
  splitting retrieval concerns out of `lib.rs`.
- `llm-serve`: `agent.rs` and `rag.rs` modules wiring the RAG pipeline and
  autonomous agents into the OpenAI-compatible server.
- `llm-train`: full training-loop modules — `arrow_dataset`, `checkpoint`,
  `dataset`, `error`, `lora`, `loss`, `metrics`, `model`, `optimizer`,
  `scheduler`, `trainer`.
- `llm-cli`: standalone `main.rs` entry point.
- Sphinx documentation: `docs/source/algorithms/` (embeddings, inference,
  orchestration, rag, training pages) and top-level `algorithms.rst`.
- Sphinx theming: Thot transparent logo, furo light/dark brand colors, and
  GitHub footer/source links, matching the optimiz-rs documentation template.
- `docs/AUTONOMOUS_ARCHITECTURE.md` — design notes for the M7 autonomous pipeline.
- Root `Makefile` with build/test/docs targets.

### Changed

- Workspace version bumped `0.1.0` → `0.2.0` to match the documentation
  release version.

### Status

- `cargo build --workspace` and `cargo test --workspace` are green
  (70/70 tests passing across all 9 crates).
- `cargo clippy --workspace -- -D warnings` is not yet clean (pending
  follow-up cleanup pass).

[0.3.0]: https://github.com/ThotDjehuty/thotbook-ai/releases/tag/v0.3.0
[0.2.0]: https://github.com/ThotDjehuty/thotbook-ai/releases/tag/v0.2.0
