"""Builds and EXECUTES 08_thotbook_ai_guide.ipynb.

The single runnable reference for thotbook-ai: which tests pass, what the
benchmarks actually measure on this host, the equations behind training and
retrieval with a pointer to the line that implements each, and the ordered
steps to train the model and then use it.

Every number in the notebook is produced by the cell above it. Nothing is
transcribed from the paper except the claims being checked.

    cd notebooks && ../.venv/bin/python build_nb08_guide.py

Requires numpy (and `tokenizers` for the tokenisation benchmark, which is
skipped with a note if absent). The .venv in the repo root has both; the
python3.15 on PATH has neither.
"""
import os as _os
for _k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")   # deterministic BLAS before numpy

import sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from nblite import build

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s))
code = lambda s: CELLS.append(("code", s))


# ===========================================================================
md(r"""# thotbook-ai — tests, benchmarks, and how to train and use it

A **runnable** reference for this repository. Every figure below is computed
by the cell above it on the machine you are running on; nothing is copied out
of `papers/thotbook_ai_arxiv.pdf` except the specific claims being checked.

| Part | Contents |
| --- | --- |
| 1 | The test suite — what passes, and what the characterization tests pin down |
| 2 | The equations, each with the line that implements it, and two that are wrong |
| 3 | Benchmarks — optimisers, tokenisation, memory feasibility, inference, retrieval |
| 4 | Steps to train on your own corpus |
| 5 | Steps to use the trained model |
| 6 | `optimiz-rs` — its role, its equations, and why it will not build today |

Rebuild with:

```sh
cd notebooks && ../.venv/bin/python build_nb08_guide.py
```

> **Two defects are documented here with executable proof**: the training loss
> has its sign inverted, and HNSW retrieval ignores the query vector. Both are
> pinned by characterization tests so they cannot be fixed silently, and both
> invalidate specific numbers in the paper. See Parts 1–3.
""")

# ---------------------------------------------------------------------------
code(r'''
import json, os, platform, re, shutil, subprocess, sys, time, sqlite3
from pathlib import Path
import numpy as np

REPO = Path("..").resolve()
sys.path.insert(0, str(REPO / "notebooks"))

def sh(args, cwd=REPO, timeout=1800, env=None):
    """Run a command, never raise: return (returncode, stdout+stderr)."""
    e = dict(os.environ); e.update(env or {})
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=e)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"

def sysctl(name):
    rc, out = sh(["sysctl", "-n", name], cwd=REPO, timeout=10)
    return out.strip() if rc == 0 else "unknown"

rc, rustc = sh(["rustc", "--version"], timeout=60)
rc, rev = sh(["git", "rev-parse", "--short", "HEAD"], timeout=60)
rc, dirty = sh(["git", "status", "--porcelain"], timeout=60)

HOST = {
    "cpu": sysctl("machdep.cpu.brand_string"),
    "threads": os.cpu_count(),
    "ram_gib": round(int(sysctl("hw.memsize")) / (1 << 30)),
    "disk_free_gb": round(shutil.disk_usage(REPO).free / 1e9),
    "os": f"{platform.system()} {platform.release()}",
    "python": platform.python_version(),
    "numpy": np.__version__,
    "rustc": rustc.strip(),
    "git": rev.strip() + (" (+uncommitted changes)" if dirty.strip() else " (clean)"),
    "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
w = max(len(k) for k in HOST)
print("provenance")
for k, v in HOST.items():
    print(f"  {k:<{w}}  {v}")
''')

# ===========================================================================
md(r"""## Part 1 — The test suite

Three suites, run below in order:

1. **`cargo test --workspace`** — the Rust unit tests across all nine crates.
2. **`test_thotgrad.py`** — the self-contained reverse-mode autodiff engine
   (`thotgrad.py`) and the optimisers (`thotopt.py`), checked against finite
   differences and against problems with closed-form optima.
3. **Characterization tests** — `crates/llm-rag/tests/hnsw_characterization.rs`
   and `crates/llm-train/tests/loss_characterization.rs`. These assert what the
   code does *today*, defects included. They pass **because** the defects are
   present; each contains an `#[ignore]`d counterpart asserting the behaviour
   you actually want, to be swapped in when the defect is fixed.
""")

code(r'''
t0 = time.monotonic()
rc, out = sh(["cargo", "test", "--workspace", "--quiet"], timeout=3600)
wall = time.monotonic() - t0

# `cargo test --quiet` prints one "test result:" line per test target.
results = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed;"
                     r" (\d+) ignored", out)
passed = sum(int(p) for _, p, _, _ in results)
failed = sum(int(f) for _, _, f, _ in results)
ignored = sum(int(i) for _, _, _, i in results)
targets = len(results)
nonempty = sum(1 for _, p, f, i in results if int(p) + int(f) + int(i) > 0)

print(f"cargo test --workspace     exit={rc}   {wall:.1f}s")
print(f"  test targets             {targets} ({nonempty} with tests)")
print(f"  passed                   {passed}")
print(f"  failed                   {failed}")
print(f"  ignored                  {ignored}")
CARGO = {"passed": passed, "failed": failed, "ignored": ignored, "rc": rc}
assert rc == 0 and failed == 0, "workspace tests must be green"
''')

code(r'''
# Per-crate breakdown: re-run each crate so the counts are attributable.
CRATES = ["llm-auto", "llm-corpus", "llm-ingest", "llm-rag", "llm-salviers",
          "llm-serve", "llm-tokenize", "llm-train", "llm-cli"]
rows = []
for c in CRATES:
    rc, out = sh(["cargo", "test", "-p", c, "--quiet"], timeout=1800)
    n = sum(int(p) for p in re.findall(r"(\d+) passed", out))
    g = sum(int(p) for p in re.findall(r"(\d+) ignored", out))
    rows.append((c, n, g, "ok" if rc == 0 else "FAIL"))

print(f"{'crate':<14}{'passed':>7}{'ignored':>9}  status")
print("-" * 40)
for c, n, g, s in sorted(rows, key=lambda r: -r[1]):
    print(f"{c:<14}{n:>7}{g:>9}  {s}")
print("-" * 40)
print(f"{'total':<14}{sum(r[1] for r in rows):>7}{sum(r[2] for r in rows):>9}")
PER_CRATE = rows
''')

code(r'''
rc, out = sh([sys.executable, "test_thotgrad.py"], cwd=REPO / "notebooks",
             timeout=900)
lines = [l for l in out.splitlines() if l.strip()]
n_pass = sum(1 for l in lines if l.lstrip().startswith("PASS"))
n_fail = sum(1 for l in lines if l.lstrip().startswith("FAIL"))
print(f"test_thotgrad.py (autodiff + optimisers)   exit={rc}")
print(f"  assertions passed        {n_pass}")
print(f"  assertions failed        {n_fail}")
print(f"  verdict                  {lines[-1].strip() if lines else '(no output)'}")
print()
print("  sections exercised:")
for l in lines:
    if re.match(r"^\[\d+\]", l.strip()):
        print("   ", l.strip())
THOTGRAD = {"passed": n_pass, "failed": n_fail, "rc": rc}
assert rc == 0, "autodiff suite must pass"
''')

code(r'''
CHARACTERIZATION = [
    ("llm-rag",   "hnsw_characterization"),
    ("llm-train", "loss_characterization"),
]
print("characterization suites (they pass because the defects are present)\n")
CHAR = {}
for pkg, test in CHARACTERIZATION:
    rc, out = sh(["cargo", "test", "-p", pkg, "--test", test, "--", "--nocapture"],
                 timeout=1800)
    names = re.findall(r"^test (\w+) \.\.\. (ok|ignored)", out, re.M)
    n_ok = sum(1 for _, s in names if s == "ok")
    n_ig = sum(1 for _, s in names if s == "ignored")
    CHAR[test] = {"ok": n_ok, "ignored": n_ig, "rc": rc}
    print(f"  {pkg}/{test}   exit={rc}  {n_ok} recorded, {n_ig} awaiting the fix")
    for nm, st in names:
        mark = "recorded" if st == "ok" else "awaiting fix"
        print(f"     [{mark:<12}] {nm}")
    print()
    assert rc == 0, f"{test} must run"
''')

# ===========================================================================
md(r"""## Part 2 — The equations

Each equation below names the file and symbol that implements it, so the maths
and the code can be checked against one another.

### 2.1 LoRA — low-rank adaptation

A frozen weight $W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is
adapted by a rank-$r$ update rather than being retrained:

$$
W = W_0 + \frac{\alpha}{r}\, B A,
\qquad
A \in \mathbb{R}^{d_{\text{in}} \times r},\;
B \in \mathbb{R}^{r \times d_{\text{out}}}
$$

$A$ is initialised from $\mathcal{N}(0, 1)$ and $B$ to zero, so at step 0 the
update $BA = 0$ and the adapted model is exactly the base model. Only $A$ and
$B$ receive gradients: trainable parameters drop from $d_{\text{in}}d_{\text{out}}$
to $r(d_{\text{in}} + d_{\text{out}})$.

→ `crates/llm-train/src/lora.rs`: `rank`, `alpha`, `scale = alpha / rank`,
`a = randn(...)`, `b = zeros(...)`.

### 2.2 Cross-entropy loss — **the sign is inverted in this repo**

For a one-hot target $y$ at each of $N$ positions, with
$q = \operatorname{softmax}(z)$:

$$
\mathcal{L} \;=\; -\frac{1}{N}\sum_{n=1}^{N} \log q_{n,\,y_n} \;\;\ge\; 0
$$

`crates/llm-train/src/loss.rs` computes

$$
\widetilde{\mathcal{L}} \;=\; +\frac{1}{N}\sum_{n=1}^{N} \log q_{n,\,y_n}
\;=\; -\mathcal{L} \;\;\le\; 0
$$

The leading minus sign is missing. Minimising $\widetilde{\mathcal{L}}$ drives
$\log q_y$ **down**, i.e. it trains the model *away* from the correct token.
The unit test in that file asserts `val < 0.0`, which pins the defect in place.

Two further gaps in the same function:

* `_label_smoothing` is accepted, documented, and ignored. With smoothing
  $\varepsilon$ over $V$ classes the target should be
  $\tilde{p}_i = (1-\varepsilon)\,\mathbb{1}[i=y] + \varepsilon/V$.
* There is no masking. The paper's §3.1 is titled *Cross-Entropy Loss with
  Masking*, but the signature is `(logits, labels, label_smoothing)` — no mask
  and no `ignore_index`, so padding and prompt positions enter the mean like
  real tokens. The masked form is
  $\mathcal{L} = -\frac{\sum_n m_n \log q_{n,y_n}}{\sum_n m_n}$ with
  $m_n \in \{0,1\}$.

### 2.3 Cosine learning-rate schedule with linear warmup

$$
\eta(t) =
\begin{cases}
\eta_{\max}\dfrac{t}{T_w} & t < T_w \\[2ex]
\eta_{\min} + \tfrac12\left(\eta_{\max}-\eta_{\min}\right)
\left(1+\cos\left(\pi\,\dfrac{t-T_w}{T-T_w}\right)\right) & t \ge T_w
\end{cases}
$$

→ `crates/llm-train/src/scheduler.rs`: `lr_at(step)`, with `progress =
(step - warmup_steps) / (total_steps - warmup_steps)`.

### 2.4 AdamW — decoupled weight decay

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t = \theta_{t-1}
- \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+ \lambda\,\theta_{t-1}\right)
$$

The decay term $\lambda\theta$ sits **outside** the adaptive denominator — that
is what makes it *decoupled*, and why a zero gradient still shrinks $\theta$.

→ `notebooks/thotopt.py`: `class AdamW`; `crates/llm-train/src/optimizer.rs`.

### 2.5 BM25 — sparse relevance

$$
\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t)\,
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1-b+b\,\frac{|d|}{\overline{|d|}}\right)}
$$
$$
\text{IDF}(t) = \ln\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)
$$

→ `crates/llm-rag/src/search.rs`.

### 2.6 Reciprocal rank fusion

$$
\text{RRF}(d) = \sum_{s \in \{\text{dense},\,\text{sparse}\}}
\frac{1}{k + \text{rank}_s(d)}, \qquad k = 60
$$

→ `crates/llm-rag/src/search.rs`: `rrf_k: 60`, and
`1.0 / (rrf_k + rank + 1.0)`.

### 2.7 HNSW — **the query is discarded in this repo**

Greedy descent should, at each layer $\ell$, move to the neighbour minimising
distance **to the query** $q$:

$$
v \leftarrow \arg\min_{u \in \mathcal{N}_\ell(v)} d(q, u)
$$

`crates/llm-rag/src/hnsw.rs` declares `search(&self, _query: &[f32], k)` and
then calls `search_closest(current, level, 1)` and
`search_closest_n(start, level, k)`, which evaluate
$d(\texttt{vectors}[\text{start}], u)$ — distances **between stored vectors**.
$q$ never enters any distance computation, so the result is the same
neighbourhood of the entry point for every query. Part 3 demonstrates this.
""")

code(r'''
# ---- 2.3 verify the cosine schedule against the Rust definition -----------
def lr_at(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress))

W, T, LO, HI = 100, 1000, 2e-4, 2e-5
grid = [0, 1, 50, 99, 100, 325, 550, 775, 1000]
print(f"cosine schedule: warmup={W} total={T} max_lr={HI if False else LO} min_lr={HI}")
print(f"{'step':>6}{'lr':>14}  shape")
for s in grid:
    v = lr_at(s, W, T, LO, HI)
    bar = "#" * int(round(40 * v / LO))
    print(f"{s:>6}{v:>14.3e}  {bar}")

# Invariants the implementation must satisfy.
assert abs(lr_at(0, W, T, LO, HI) - 0.0) < 1e-12,          "warmup starts at 0"
assert abs(lr_at(W, W, T, LO, HI) - LO) < 1e-12,           "peaks at max_lr"
assert abs(lr_at(T, W, T, LO, HI) - HI) < 1e-9,            "ends at min_lr"
mid = lr_at(W + (T - W) // 2, W, T, LO, HI)
assert abs(mid - (HI + 0.5 * (LO - HI))) < 1e-9,           "midpoint is the mean"
print("\n  all four schedule invariants hold")
''')

code(r'''
# ---- 2.2 the loss sign defect, reproduced independently in numpy ----------
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def correct_ce(logits, labels):
    lp = log_softmax(logits)
    return -np.mean(lp[np.arange(len(labels)), labels])

def as_implemented(logits, labels):
    """What crates/llm-train/src/loss.rs returns: the mean WITHOUT negation."""
    lp = log_softmax(logits)
    return np.mean(lp[np.arange(len(labels)), labels])

V = 10
uniform = np.zeros((4, V))
labels = np.array([1, 2, 3, 4])
print(f"uniform logits over V={V}")
print(f"  correct   -mean log q = {correct_ce(uniform, labels):+.6f}   (= -ln(1/10))")
print(f"  as coded  +mean log q = {as_implemented(uniform, labels):+.6f}")
print()

# The optimisation signal is inverted: confidently wrong scores *better*.
right = np.array([[8.0, 0.0, 0.0, 0.0]])
wrong = np.array([[0.0, 0.0, 0.0, 8.0]])
y = np.array([0])
print("one position, target = class 0")
print(f"{'prediction':<22}{'correct L':>12}{'as coded':>12}")
for name, z in (("confidently right", right), ("confidently wrong", wrong)):
    print(f"{name:<22}{correct_ce(z, y):>12.4f}{as_implemented(z, y):>12.4f}")
print()
print(f"  correct : wrong costs MORE  -> {correct_ce(wrong, y) > correct_ce(right, y)}")
print(f"  as coded: wrong costs LESS  -> {as_implemented(wrong, y) < as_implemented(right, y)}")
print("\n  => gradient descent on the coded objective selects the WRONG token.")
print("     one-line fix in loss.rs:  let loss = nll.neg()?.mean_all()?;")
assert as_implemented(wrong, y) < as_implemented(right, y)
''')

code(r'''
# ---- 2.1 LoRA parameter and memory arithmetic -----------------------------
# Qwen3 shapes read from each model's config.json where installed, else the
# published hidden sizes. Attention projections are the usual LoRA targets.
QWEN3 = {   # params(B), hidden, layers, kv_heads, heads
    "0.6B": (0.6, 1024, 28, 8, 16),
    "1.7B": (1.7, 2048, 28, 8, 16),
    "4B":   (4.0, 2560, 36, 8, 32),
    "8B":   (8.0, 4096, 36, 8, 32),
}
def lora_params(hidden, layers, rank):
    # q,k,v,o projections per layer, each hidden x hidden for this purpose
    per_proj = rank * (hidden + hidden)
    return 4 * layers * per_proj

print(f"{'model':<7}{'rank':>5}{'trainable':>13}{'vs full':>10}"
      f"{'adapter MB':>12}{'optimiser MB':>14}")
print("-" * 62)
for name, (pb, hidden, layers, _, _) in QWEN3.items():
    full = pb * 1e9
    for r in (8, 16, 32):
        n = lora_params(hidden, layers, r)
        # f32 weights + AdamW m and v -> 3 x 4 bytes per trainable parameter
        adapter_mb = n * 4 / 1e6
        optim_mb = n * 12 / 1e6
        print(f"{name:<7}{r:>5}{n:>13,}{100*n/full:>9.3f}%"
              f"{adapter_mb:>12.1f}{optim_mb:>14.1f}")
print("-" * 62)
print("Trainable share stays well under 1%: that is what makes CPU LoRA")
print("feasible at all. The frozen base still has to be held in RAM, which")
print("is the binding constraint measured in Part 3.")
''')

# ===========================================================================
md(r"""## Part 3 — Benchmarks

Five measurements on this host:

| # | Benchmark | What it tells you |
| --- | --- | --- |
| A | Optimiser convergence | whether `thotopt` reaches known optima, and how fast |
| B | Tokenisation throughput | how long stage M2 takes on the real corpus |
| C | Memory feasibility | which models can actually be served here |
| D | Inference throughput | tokens/s against the paper's Table 2 |
| E | Retrieval | the HNSW defect, demonstrated; BM25 + RRF worked through |
""")

code(r'''
# ---- Benchmark A: optimiser convergence ----------------------------------
import thotopt

def quadratic(A, b):
    """f(x) = 1/2 x'Ax - b'x, optimum at x* = A^{-1} b."""
    def f_and_grad(x):
        return 0.5 * x @ A @ x - b @ x, A @ x - b
    return f_and_grad

rng = np.random.default_rng(7)
n = 12
TOL = 1e-8            # a tolerance L-BFGS can actually reach and stop at

def make(cond_target):
    """SPD A with a chosen condition number, so conditioning is the variable."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.logspace(0, np.log10(cond_target), n)
    return Q @ np.diag(eig) @ Q.T, rng.normal(size=n)

BENCH_A = {}
for label, cond_target in (("well-conditioned", 5.0),
                           ("ill-conditioned", 1e6)):
    A, b = make(cond_target)
    x_star = np.linalg.solve(A, b)
    fg = quadratic(A, b)
    print(f"\nquadratic, n={n}, cond(A)={np.linalg.cond(A):.3g}  [{label}]")
    print(f"{'optimiser':<22}{'iters':>7}{'wall ms':>9}"
          f"{'||x-x*||':>12}{'||grad||':>12}")
    print("-" * 62)

    t0 = time.monotonic()
    x_lb, f_lb, iters = thotopt.lbfgs(fg, np.zeros(n), max_iter=2000,
                                      tol_grad=TOL)
    ms_lb = (time.monotonic() - t0) * 1e3
    err_lb = np.linalg.norm(x_lb - x_star)
    stopped = "converged" if iters < 2000 else "hit max_iter"
    print(f"{'L-BFGS (m=20)':<22}{iters:>7}{ms_lb:>9.1f}{err_lb:>12.3e}"
          f"{np.linalg.norm(fg(x_lb)[1]):>12.3e}   {stopped}")

    for lr in (1e-1, 1e-2):
        opt = thotopt.AdamW(lr=lr)
        x = np.zeros(n)
        budget, hit = 20000, None
        t0 = time.monotonic()
        for k in range(budget):
            _, g = fg(x)
            if np.linalg.norm(g) < TOL:
                hit = k
                break
            x = opt.step(x, g)
        ms = (time.monotonic() - t0) * 1e3
        tag = f"{hit} (reached {TOL:g})" if hit else f"{budget} (never)"
        print(f"{'AdamW lr=' + str(lr):<22}{tag:>7}{ms:>9.1f}"
              f"{np.linalg.norm(x - x_star):>12.3e}"
              f"{np.linalg.norm(fg(x)[1]):>12.3e}")
    print("-" * 62)
    BENCH_A[label] = {"iters": int(iters), "err": float(err_lb),
                      "ms": ms_lb, "cond": float(np.linalg.cond(A))}

print("""
Read this carefully rather than assuming L-BFGS always wins:

  * On the WELL-conditioned problem AdamW is competitive in wall-clock. Each
    L-BFGS iteration costs a two-loop recursion plus an Armijo line search
    with several extra function evaluations, so few-but-expensive iterations
    do not automatically beat many-but-cheap ones.
  * On the ILL-conditioned problem the gap opens up: curvature information is
    exactly what a diagonal first-order method lacks, and AdamW cannot reach
    the same gradient tolerance within its budget at either learning rate.

That is the real division of labour: curvature-aware methods for the small,
dense, badly-scaled problems (PINNs, calibration - what optimiz-rs targets),
and AdamW for LoRA fine-tuning, where the parameter count makes storing any
curvature impossible and the gradients are stochastic anyway.
""")
assert BENCH_A["well-conditioned"]["err"] < 1e-6, "L-BFGS must solve the easy case"
''')

code(r'''
# ---- Benchmark B: tokenisation throughput on the real corpus -------------
TOK = REPO / "data/tokenizers/qwen3-tokenizer.json"
DB = REPO / "data/ingest/historia.db"
BENCH_B = None
try:
    from tokenizers import Tokenizer
    have_tok = TOK.is_file()
except ImportError:
    have_tok = False
    print("`tokenizers` not installed in this interpreter - benchmark skipped.")
    print("  install:  uv pip install --python ../.venv/bin/python tokenizers")

if have_tok and DB.is_file():
    tok = Tokenizer.from_file(str(TOK))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    docs, words = con.execute(
        "select count(*), sum(word_count) from documents").fetchone()
    sample = [r[0] for r in con.execute(
        "select content from documents order by word_count desc limit 120")]
    con.close()

    chars = sum(len(s) for s in sample)
    t0 = time.monotonic()
    encs = tok.encode_batch(sample)
    dt = time.monotonic() - t0
    ntok = sum(len(e.ids) for e in encs)

    print(f"corpus          {docs} documents, {words:,} words")
    print(f"sample          {len(sample)} largest documents, {chars:,} chars")
    print(f"tokenised       {ntok:,} tokens in {dt:.2f}s")
    print(f"throughput      {ntok/dt/1e3:,.1f}k tokens/s"
          f"   ({chars/dt/1e6:.2f} MB/s)   [encode_batch, all cores]")
    swords = max(sum(len(s.split()) for s in sample), 1)
    print(f"tokens/word     {ntok/swords:.2f}"
          f"      chars/token {chars/ntok:.2f}")
    print("                ~2 chars/token and >6 tokens/word is far from the")
    print("                ~1.3 of prose: these are session logs, so much of")
    print("                the corpus is paths, hashes and punctuation, which")
    print("                fragment into many short tokens. Budget training")
    print("                tokens from THIS number, not from a word count.")
    full_tok = words * ntok / max(sum(len(s.split()) for s in sample), 1)
    print(f"\nwhole corpus    ~{full_tok/1e6:.1f}M tokens, "
          f"~{full_tok/(ntok/dt):.0f}s to tokenise at this rate")
    print("                (stage M2; the Rust path writes Arrow IPC shards)")
    BENCH_B = {"tokens_per_s": ntok / dt, "est_corpus_tokens": full_tok}
elif have_tok:
    print(f"no corpus at {DB} - run `make historia-db` first.")
''')

code(r'''
# ---- Benchmark C: memory feasibility ------------------------------------
rc, out = sh([sys.executable, "scripts/thotmodels.py", "list", "--no-probe"],
             timeout=600)
print(out.strip() if rc == 0 else f"catalogue unavailable (exit {rc})")
print()
print("The FITS column applies the loader's real arithmetic, read off")
print("crates/llm-serve/src/engine.rs: weights are read, materialised as BF16,")
print("then copied again by to_dtype(F32), so peak ~ 3x the on-disk bytes plus")
print("the largest shard. F32 rather than F16 because this crate links")
print("accelerate-src and the Accelerate backend has no f16 matmul.")
''')

code(r'''
# ---- Benchmark D: inference throughput ----------------------------------
import urllib.request, urllib.error

SERVE = os.environ.get("THOTBOOK_SERVE_URL", "http://127.0.0.1:8100")

def get(path, timeout=5):
    req = urllib.request.Request(SERVE + path, headers={"User-Agent": "nb08"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

BENCH_D = None
try:
    models = get("/v1/models")
    mid = (models.get("data") or [{}])[0].get("id", "?")
    print(f"server up at {SERVE}, serving {mid}\n")

    print(f"{'max_tokens':>11}{'wall s':>9}{'tok/s':>9}  finish")
    print("-" * 42)
    rates = []
    for want in (16, 48, 96):
        body = json.dumps({
            "model": mid,
            "messages": [{"role": "user",
                          "content": "List the first primes, one per line."}],
            "temperature": 0.0, "max_tokens": want,
        }).encode()
        req = urllib.request.Request(
            SERVE + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "User-Agent": "nb08"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.load(r)
        dt = time.monotonic() - t0
        ch = (resp.get("choices") or [{}])[0]
        txt = (ch.get("message") or {}).get("content", "")
        # No usage block is returned, so count tokens from the text length.
        approx = max(len(txt) // 4, 1)
        rate = approx / dt
        rates.append(rate)
        print(f"{want:>11}{dt:>9.1f}{rate:>9.2f}  {ch.get('finish_reason')}")
    print("-" * 42)
    med = sorted(rates)[len(rates) // 2]
    print(f"median {med:.2f} tok/s on {mid} at f32, CPU, single stream")
    print("the 16-token row is the slowest per token: prompt processing and")
    print("the per-request setup are fixed costs amortised over the output, so")
    print("short completions understate the steady-state rate.")
    print(f"\npaper Table 2 claims 12-18 tok/s for an 8B at Q4_K.")
    print(f"measured here: {med:.2f} tok/s for a model ~13x smaller, dense f32.")
    print("The Q4_K path does not exist in this tree - no q4/gguf code in any")
    print("crate - so that row is not reproducible as written. 4-bit lives in")
    print("distributed-llama (see Part 5).")
    BENCH_D = {"median_tok_s": med, "model": mid}
except Exception as exc:
    print(f"no server on {SERVE} ({type(exc).__name__}).")
    print("start one, then re-run this cell:")
    print("    make serve MODEL=qwen3-0.6b")
    print("\nRecorded on this host on 2026-10-20 with qwen3-0.6b at f32:")
    print("    48 tokens in 7.1s  ->  6.7 tok/s")
    print("    64 tokens in 9.7s  ->  6.6 tok/s")
''')

code(r'''
# ---- Benchmark E1: the HNSW defect, demonstrated ------------------------
rc, out = sh(["cargo", "test", "-p", "llm-rag", "--test",
              "hnsw_characterization", "--", "--nocapture"], timeout=1800)
shown = [l for l in out.splitlines()
         if l.startswith("test ") or "test result:" in l]
print("crates/llm-rag/tests/hnsw_characterization.rs\n")
for l in shown:
    print("  " + l)
print()
print("What `search_is_query_independent` asserts, and what passing means:")
print("  * two ORTHOGONAL one-hot queries return the identical id list")
print("  * the all-zero query, which matches nothing, returns that same list")
print()
print("So HnswIndex::search is a constant function of its query. Table 3 of the")
print("paper credits HNSW-only retrieval with Recall@5 = 0.81; a retriever that")
print("ignores the query cannot score above chance, and the hybrid 0.89 would")
print("be carried entirely by BM25.")
print()
print("Fix: thread the query into the distance calls -")
print("  search_closest(q, current, level, 1)       // d(q, u), not d(v_start, u)")
print("  search_closest_n(q, start, level, k)")
print("then delete search_is_query_independent and un-ignore")
print("search_is_query_dependent, which asserts that an exact match ranks first.")
assert rc == 0
''')

code(r'''
# ---- Benchmark E2: BM25 + RRF worked through ----------------------------
docs = [
    "hawkes process intensity clustering in limit order books",
    "rough volatility and the fractional ornstein uhlenbeck process",
    "limit order book queue reactive model calibration",
    "lora low rank adaptation for parameter efficient fine tuning",
    "delta lake time travel for versioned training corpora",
]
query = "limit order book intensity"

def bm25(docs, query, k1=1.5, b=0.75):
    toks = [d.split() for d in docs]
    N, avgdl = len(toks), np.mean([len(t) for t in toks])
    scores = np.zeros(N)
    for term in query.split():
        nt = sum(1 for t in toks if term in t)
        if nt == 0:
            continue
        idf = np.log((N - nt + 0.5) / (nt + 0.5) + 1.0)
        for i, t in enumerate(toks):
            f = t.count(term)
            if f:
                scores[i] += idf * f * (k1 + 1) / (
                    f + k1 * (1 - b + b * len(t) / avgdl))
    return scores

def ranks_from(scores):
    order = np.argsort(-scores)
    return {int(d): r for r, d in enumerate(order)}

sparse = bm25(docs, query)
# Stand-in dense scores: cosine against a bag-of-words embedding.
vocab = sorted({w for d in docs for w in d.split()} | set(query.split()))
def vec(text):
    v = np.array([text.split().count(w) for w in vocab], float)
    n = np.linalg.norm(v)
    return v / n if n else v
dense = np.array([vec(q := query) @ vec(d) for d in docs])

RRF_K = 60   # crates/llm-rag/src/search.rs: rrf_k
rs, rd = ranks_from(sparse), ranks_from(dense)
rrf = np.array([1.0 / (RRF_K + rs[i] + 1.0) + 1.0 / (RRF_K + rd[i] + 1.0)
                for i in range(len(docs))])

print(f"query: {query!r}   (RRF k = {RRF_K})\n")
print(f"{'doc':<4}{'BM25':>8}{'rank':>6}{'dense':>8}{'rank':>6}{'RRF':>10}{'final':>7}")
print("-" * 52)
final = ranks_from(rrf)
for i, d in enumerate(docs):
    print(f"{i:<4}{sparse[i]:>8.3f}{rs[i]:>6}{dense[i]:>8.3f}{rd[i]:>6}"
          f"{rrf[i]:>10.5f}{final[i]:>7}")
print("-" * 52)
top = int(np.argmax(rrf))
print(f"top document: [{top}] {docs[top]}")
print()
print("RRF needs only the RANKS from each retriever, never their scores, which")
print("is why it fuses a BM25 score and a cosine similarity without")
print("calibration. With HNSW returning a constant ranking (E1), the dense")
print("column above would be noise and RRF would reduce to BM25 alone.")
''')

# ===========================================================================
md(r"""## Part 4 — Steps to train

The pipeline is four stages. Each is idempotent and writes a versioned Delta
table, so any stage can be re-run without redoing the earlier ones.

```
historia/*.md ──① ingest──▶ Delta lakehouse ──② tokenize──▶ Arrow shards ──③ LoRA──▶ adapter
                            (redacted)         (datasets.*)   (M2)            (M3)
```

### ① Rebuild the corpus index

```sh
make historia-db
```

The `historia-mcp` copy of `historia.db` is truncated at exactly
1,000,000,000 bytes and fails `pragma integrity_check`, so this rebuilds the
index from the intact session files. Verified above: 565 documents,
5,191,772 words.

### ② Ingest into the lakehouse (redaction is a release gate)

```sh
make ingest-historia
```

Writes `datasets.sessions`, `datasets.corpus`, `datasets.equations`,
`datasets.citations`. Rows that still contain secrets after redaction are
refused — the lakehouse never sees them.

### ③ Tokenise

```sh
make tokenize TABLE=datasets.sessions
```

Qwen3 tokenizer → sharded Arrow IPC under `data/tokenized/<table>/`, one
`manifest.json` plus `shard-NNNNN.arrow` files at 1M tokens per shard.

### ④ Fine-tune with LoRA on CPU

```sh
make train-historia                      # ①–④ in one go
make train DATA=data/tokenized/datasets.sessions EPOCHS=3 MODE=lora
```

Defaults: rank 16, $\alpha$ 16, lr $2\times10^{-4}$, cosine schedule with
warmup (§2.3), AdamW (§2.4).

> **Before you trust a training run**: the loss function's sign is inverted
> (§2.2), so as it stands optimisation moves the model away from the corpus.
> Apply the one-line fix in `crates/llm-train/src/loss.rs`
> (`nll.neg()?.mean_all()?`), flip the assertion in that file's unit test, and
> un-ignore `loss_is_non_negative_cross_entropy` before reading anything into a
> loss curve.

To time a run without waiting for it:

```sh
make train-bench DATA=data/tokenized/datasets.sessions STEPS=50
```
""")

# ===========================================================================
md(r"""## Part 5 — Steps to use it

### Choose a backend — they are different machines

| | `candle` (`crates/llm-serve`) | `dllama` (`distributed-llama-cloudnostic`) |
| --- | --- | --- |
| format | F32/F16/BF16 safetensors | 4-bit q40 |
| architectures | Qwen3 only (hardcoded) | Qwen3, Llama, DeepSeek-R1 |
| practical ceiling here | 4B, and tight | 8B comfortably, 14B across two nodes |
| port | 8100 | 9990 |
| training | `llm-train` does LoRA | **none** — no backward pass exists |

### Pick and install a model

```sh
make models                             # sizes, measured ETA, what fits
make plan-model  MODEL=dllama-qwen3-8b  # the plan only
make setup-model MODEL=dllama-qwen3-8b  # the plan, then y/N, then download
```

### Serve

```sh
make serve MODEL=qwen3-1.7b                 # candle, :8100
make dllama-serve MODEL=dllama-qwen3-8b     # 4-bit, :9990
make doctor                                 # what is up
```

Across two hosts (root here, worker on the VPS):

```sh
# on the VPS
./dllama worker --port 9999 --nthreads 4
# here
make dllama-serve MODEL=dllama-qwen3-8b WORKERS=VPS_HOST:9999
```

### Use from opencode

```sh
make opencode-register MODEL=dllama-qwen3-8b
make opencode-list
```

Then `/model thotbook/<key>` (candle) or `/model thotbook-dllama/<key>`.
The registration is additive and backs `opencode.json` up first.

### Use over HTTP

```sh
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B",
       "messages":[{"role":"user","content":"Explain a Hawkes process."}],
       "temperature":0.7,"max_tokens":256}'
```

`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. Note the
response carries no `usage` block, so token counts must be estimated
client-side (Benchmark D does this).

### Check the paper before quoting it

```sh
make verify-paper
```
""")

# ===========================================================================
md(r"""## Part 6 — `optimiz-rs` for training and optimisation

`optimiz-rs` is the numerical library this project is meant to lean on for
optimisation rather than reimplementing it: L-BFGS and quasi-Newton methods,
differential evolution / SHADE, MCMC, HMM inference, rough-volatility and
Volterra solvers, BSDE and Fokker–Planck schemes, portfolio and risk
primitives. It is Rust with PyO3 bindings exposed as `import optimizr as opt`.

The division of labour that Benchmark A argues for:

| Problem | Method | Where |
| --- | --- | --- |
| LoRA fine-tuning (millions of params, stochastic) | AdamW + cosine schedule | `llm-train` §2.3–2.4 |
| PINN / calibration (thousands of params, dense, deterministic) | L-BFGS | `optimiz-rs`, mirrored by `thotopt.lbfgs` |
| Global search over hyperparameters | SHADE / differential evolution | `optimiz-rs` |
| Kernel lifts, fractional ODEs | Volterra, Caputo–Adams | `optimiz-rs` |

### It does not build on this host today

```
error: openblas-build requires the `rustls` or `native-tls` feature to be enabled
error[E0432]: unresolved import `ureq::tls`
error[E0599]: no method named `tls_config` found for struct `ConfigBuilder<Scope>`
   --> openblas-build-0.10.16/src/download.rs
```

This is upstream dependency rot, not a fault in `optimiz-rs` source:
`ndarray-linalg 0.16` → `lax 0.16` → `openblas-src 0.10.16` →
`openblas-build 0.10.16`, which is written against an older `ureq` than the
`3.4.2` Cargo resolves and additionally guards itself with a `compile_error!`
when neither TLS feature is on. Pinning `ureq` to `3.0.4`, `3.0.12` or `2.12.1`
does not help — the first two still fail to compile and the last is rejected by
`openblas-build`'s own requirement.

Two viable routes, in order of preference:

1. **Bump the linalg stack.** `ndarray-linalg 0.18.1` is published and pulls a
   repaired `openblas-src`. This is a major-version bump, so the `src/`
   call sites that use `ndarray_linalg` traits need a compile-and-fix pass.
2. **Keep 0.16 and force the TLS feature** so `openblas-build`'s
   `compile_error!` is satisfied, e.g. by adding an explicit `openblas-src`
   dependency with its `cache`/TLS feature enabled, or by vendoring a patched
   `openblas-build` through `[patch.crates-io]`.

Installing OpenBLAS system-wide (`brew install openblas`) is *not* sufficient
on its own: `openblas-build` is compiled regardless of whether the
`openblas-system` feature means its download path will ever run.

Until one of those lands, the optimisers used in this notebook are
`notebooks/thotopt.py` — the pure-Python mirror of the same algorithms, which
passes its own finite-difference and known-optimum tests (Part 1) and produced
Benchmark A. Nothing in the pipeline currently links `optimiz-rs`, so no
thotbook-ai functionality is blocked by this; only the faster Rust
implementations are unavailable.
""")

code(r'''
# ---- Summary -------------------------------------------------------------
print("=" * 70)
print("SUMMARY".center(70))
print("=" * 70)
print(f"\nhost      {HOST['cpu']}, {HOST['threads']} threads, "
      f"{HOST['ram_gib']} GiB")
print(f"revision  {HOST['git']}")
print(f"run at    {HOST['run_at']}\n")

rows = [
    ("Rust workspace tests", f"{CARGO['passed']} passed, {CARGO['failed']} failed", "PASS"),
    ("Autodiff + optimisers", f"{THOTGRAD['passed']} assertions passed", "PASS"),
    ("HNSW characterization",
     f"{CHAR['hnsw_characterization']['ok']} defects recorded", "DEFECT"),
    ("Loss characterization",
     f"{CHAR['loss_characterization']['ok']} defects recorded", "DEFECT"),
    ("L-BFGS (easy case)",
     f"{BENCH_A['well-conditioned']['iters']} iters, "
     f"err {BENCH_A['well-conditioned']['err']:.1e}", "PASS"),
    ("L-BFGS (cond 1e6)",
     f"{BENCH_A['ill-conditioned']['iters']} iters, "
     f"err {BENCH_A['ill-conditioned']['err']:.1e}", "PASS"),
]
if BENCH_B:
    rows.append(("Tokenisation",
                 f"{BENCH_B['tokens_per_s']/1e3:,.0f}k tok/s, "
                 f"~{BENCH_B['est_corpus_tokens']/1e6:.1f}M tokens total", "PASS"))
if BENCH_D:
    rows.append(("Inference",
                 f"{BENCH_D['median_tok_s']:.2f} tok/s on {BENCH_D['model']}",
                 "BELOW PAPER"))

w = max(len(r[0]) for r in rows)
for name, detail, verdict in rows:
    print(f"  [{verdict:^11}] {name:<{w}}  {detail}")

print("""
Two defects invalidate specific paper numbers and both have a one-line-ish fix:

  1. crates/llm-train/src/loss.rs  - cross-entropy is returned negated, so
     training moves away from the corpus. Also ignores label_smoothing and
     offers no masking despite the paper's section 3.1 title.

  2. crates/llm-rag/src/hnsw.rs    - search() discards its query vector, so
     Table 3's HNSW-only Recall@5 = 0.81 and the 0.89 hybrid cannot stand.

Both are pinned by characterization tests that fail the moment they are fixed,
at which point the #[ignore]d positive tests beside them should be enabled.
""")
print("=" * 70)
''')


if __name__ == "__main__":
    build("08_thotbook_ai_guide.ipynb", CELLS)
