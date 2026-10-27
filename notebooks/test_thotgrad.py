"""Correctness tests for thotgrad / thotopt.

Every gradient is checked against central finite differences, and the
optimisers against closed-form optima. Run: python3 test_thotgrad.py
"""
from __future__ import annotations

import numpy as np

from thotgrad import MLP, Tensor
from thotopt import AdamW, lbfgs

rng = np.random.default_rng(0)
FAILS = []


def check(name: str, got: float, want: float, tol: float = 1e-6) -> None:
    err = abs(got - want) / max(1.0, abs(want))
    ok = err < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} got={got: .10e} want={want: .10e} rel={err:.2e}")
    if not ok:
        FAILS.append(name)


def fd_grad(fn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient."""
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += h
        xm.flat[i] -= h
        g.flat[i] = (fn(xp) - fn(xm)) / (2 * h)
    return g


# ---------------------------------------------------------------------------
print("\n[1] elementary operations vs finite differences")
for op_name, op in [
    ("add/mul/sum",  lambda a, b: ((a * b) + a).sum()),
    ("matmul",       lambda a, b: (a @ b.T if False else a @ b).sum()),
    ("tanh chain",   lambda a, b: ((a @ b).tanh() * a.sum()).sum()),
    ("pow/div",      lambda a, b: (((a ** 2.0) / (b ** 2.0 + Tensor(1.0)))).mean()),
]:
    A0 = rng.normal(size=(3, 4))
    B0 = rng.normal(size=(4, 3)) if "matmul" in op_name or "tanh" in op_name else rng.normal(size=(3, 4))

    def scalar(flat):
        a = Tensor(flat[:A0.size].reshape(A0.shape), requires_grad=True)
        b = Tensor(flat[A0.size:].reshape(B0.shape), requires_grad=True)
        return float(op(a, b).data)

    a_t = Tensor(A0, requires_grad=True)
    b_t = Tensor(B0, requires_grad=True)
    out = op(a_t, b_t)
    out.backward()
    ana = np.concatenate([a_t.grad.reshape(-1), b_t.grad.reshape(-1)])
    num = fd_grad(scalar, np.concatenate([A0.reshape(-1), B0.reshape(-1)]))
    check(op_name, float(np.abs(ana - num).max()), 0.0, tol=1e-5)

# ---------------------------------------------------------------------------
print("\n[2] network input-derivatives u_x, u_xx vs finite differences")
net = MLP(1, 1, width=12, depth=2, seed=1)
x0 = np.linspace(-1.2, 1.4, 7).reshape(-1, 1)
u, ux, uxx = net.forward_with_derivs(Tensor(x0))


def u_at(xv):
    return net(Tensor(np.array([[xv]]))).data.item()


h = 1e-5
for k in range(x0.size):
    xv = x0[k, 0]
    num_1 = (u_at(xv + h) - u_at(xv - h)) / (2 * h)
    num_2 = (u_at(xv + h) - 2 * u_at(xv) + u_at(xv - h)) / (h * h)
    check(f"u_x  at x={xv:+.3f}", float(ux.data[k, 0]), num_1, tol=1e-5)
    check(f"u_xx at x={xv:+.3f}", float(uxx.data[k, 0]), num_2, tol=1e-3)

# ---------------------------------------------------------------------------
print("\n[3] parameter gradients of a PINN-style loss (double-backward)")
net2 = MLP(1, 1, width=8, depth=2, seed=2)
xc = Tensor(np.linspace(0, 1, 11).reshape(-1, 1))


def pinn_loss_value(flat):
    net2.set_flat(flat)
    u_, ux_, uxx_ = net2.forward_with_derivs(xc)
    r = uxx_ + u_                      # harmonic-oscillator residual u'' + u
    return float((r * r).mean().data)


theta0 = net2.get_flat().copy()
net2.set_flat(theta0)
u_, ux_, uxx_ = net2.forward_with_derivs(xc)
r = uxx_ + u_
loss = (r * r).mean()
loss.backward()
ana = net2.grad_flat()
num = fd_grad(pinn_loss_value, theta0, h=1e-6)
net2.set_flat(theta0)
check("max |analytic - FD| over all params", float(np.abs(ana - num).max()), 0.0, tol=1e-4)
check("cosine similarity(analytic, FD)",
      float(ana @ num / (np.linalg.norm(ana) * np.linalg.norm(num))), 1.0, tol=1e-8)

# ---------------------------------------------------------------------------
print("\n[4] optimisers on a quadratic with known optimum")
A = np.array([[3.0, 0.4], [0.4, 1.0]])
b = np.array([1.0, -2.0])
x_star = np.linalg.solve(A, b)


def quad(x):
    return 0.5 * x @ A @ x - b @ x, A @ x - b


x_lb, f_lb, n_it = lbfgs(quad, np.zeros(2), max_iter=100)
check("L-BFGS x[0]", float(x_lb[0]), float(x_star[0]), tol=1e-8)
check("L-BFGS x[1]", float(x_lb[1]), float(x_star[1]), tol=1e-8)
print(f"        (converged in {n_it} iterations)")

opt = AdamW(lr=0.05, weight_decay=0.0)
xa = np.zeros(2)
for _ in range(4000):
    xa = opt.step(xa, A @ xa - b)
check("AdamW x[0]", float(xa[0]), float(x_star[0]), tol=1e-4)
check("AdamW x[1]", float(xa[1]), float(x_star[1]), tol=1e-4)

print("\n[5] AdamW decoupled decay (zero gradient must still shrink)")
o = AdamW(lr=0.1, weight_decay=0.5)
p = o.step(np.ones(3), np.zeros(3))
check("theta after 1 step, g=0", float(p[0]), 0.95, tol=1e-9)

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILS:
    print(f"FAILED ({len(FAILS)}): {FAILS}")
    raise SystemExit(1)
print("ALL THOTGRAD / THOTOPT TESTS PASSED")
