# thotgrad

Railway-style reverse-mode automatic differentiation and gradient-based
optimisers in Rust, exposed to Python through PyO3. This is the Rust rewrite of
`notebooks/thotgrad.py` (the tape) and `notebooks/thotopt.py` (AdamW + L-BFGS).

The algorithm is identical to the NumPy originals — an array-level reverse-mode
tape, the analytic first/second input-derivative recursion for a tanh MLP, the
decoupled-weight-decay AdamW update, and the Nocedal two-loop L-BFGS with an
Armijo line search — but it runs as native `ndarray` code with no Python
dispatch per operation.

## Railway style

Every fallible step in [`core`](src/core.rs) returns `Result<T, ThotError>` and
the pipelines compose with `?` / `and_then`, exactly like the
`Result<DataFrame, Error>` channel in the Polarway engine. A shape mismatch or a
non-finite value short-circuits onto the error track instead of panicking or
silently producing a `NaN`. At the PyO3 boundary a `ThotError` becomes a clean
Python `ValueError`.

## Python API

```python
import thotgrad_rs as tg

net = tg.Mlp(n_in=1, n_out=1, width=32, depth=3, seed=0)
u, u_x, u_xx = net.forward_with_derivs(x)          # exact input-derivatives

pinn = tg.PoissonPinn(width=32, depth=3, seed=0, n_coll=256)
loss, grad = pinn.loss_and_grad(theta)             # one forward+backward
theta, hist = pinn.train_adamw(steps=2000, lr=3e-3)
theta, loss, n_iter = pinn.train_lbfgs(max_iter=500)
print(pinn.l2_error())                             # vs exact sin(pi x)

opt = tg.AdamW(lr=1e-3, weight_decay=0.0)
theta = opt.step(theta, grad)
```

## Build

```bash
# into the active venv/conda env
maturin develop --release
# or a wheel
maturin build --release
```

Pure Rust, no deep-learning framework: `ndarray`, `rand`, `pyo3`, `numpy`.
Keeping it framework-free is deliberate — it is what lets the companion
CPU-PINN paper claim "no deep-learning framework."

## Tests

```bash
cargo test --no-default-features     # pure core, no Python needed
```

Correctness is pinned against central finite differences, mirroring
`test_thotgrad.py`: the input-derivatives, the PINN parameter gradient (the
double-backward), the AdamW decay at zero gradient, and L-BFGS convergence on a
problem with a known optimum.

## Benchmark

`notebooks/10_thotgrad_rust_benchmark.ipynb` compares this crate against the
pure-Python tape and shows the PyTorch / JAX equivalents, on the 1-D Poisson
PINN. Honest finding: Rust is fastest in the small-network PINN regime (per step
and full-solve); for large dense networks NumPy's BLAS tape catches up, which is
the regime you would move to a GPU / JAX for anyway.
