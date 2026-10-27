"""thotopt — gradient-based optimisers owned by this project.

`optimiz-rs` ships gradient-*free* global optimisers (differential evolution,
grid search, MCMC). Those are the right tool for non-differentiable objectives
such as the attack-threshold calibration in the privacy-audit notebook, but
they are the wrong tool for a PINN, where autodiff hands us exact gradients.

This module fills that gap with the two optimisers a PINN actually needs,
implemented directly rather than imported from SciPy or PyTorch:

* :class:`AdamW` — the decoupled-weight-decay update of Loshchilov & Hutter
  (2019). Deliberately mirrors `llm_train::optimizer::AdamWOptimizer` in the
  Rust crate, including the decay term that was missing there until v0.3.0.
* :func:`lbfgs` — limited-memory BFGS via the Nocedal two-loop recursion with
  a backtracking Armijo line search.

Both are validated in `test_thotgrad.py` against closed-form optima.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AdamW:
    r"""Adam with **decoupled** weight decay.

    .. math::
        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
        \theta_t &= \theta_{t-1} - \eta\Bigl(
            \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_{t-1}\Bigr)

    The decay multiplies the parameter, not the gradient, so it is never
    rescaled by the adaptive denominator — the distinction between AdamW and
    Adam+L2.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * theta)


def lbfgs(f_and_grad, x0: np.ndarray, max_iter: int = 500, m: int = 20,
          tol_grad: float = 1e-12, c1: float = 1e-4, max_ls: int = 25,
          callback=None) -> tuple[np.ndarray, float, int]:
    r"""Limited-memory BFGS with backtracking Armijo line search.

    Implements the Nocedal & Wright two-loop recursion: the inverse-Hessian
    action :math:`H_k \nabla f_k` is reconstructed from the last `m` secant
    pairs :math:`(s_k, y_k) = (\theta_{k+1}-\theta_k,\ g_{k+1}-g_k)` without
    ever forming a matrix. Curvature pairs failing the condition
    :math:`s^\top y > 0` are skipped, keeping the implicit Hessian positive
    definite.

    Parameters
    ----------
    f_and_grad : callable
        ``x -> (value, gradient)``.

    Returns
    -------
    (x, f, n_iter)
    """
    x = x0.copy()
    f, g = f_and_grad(x)
    S: deque = deque(maxlen=m)
    Y: deque = deque(maxlen=m)
    it = 0

    for it in range(1, max_iter + 1):
        gnorm = np.linalg.norm(g)
        if gnorm < tol_grad or not np.isfinite(f):
            break

        # --- two-loop recursion: q <- H_k g ---------------------------------
        q = g.copy()
        alphas = []
        for s_i, y_i in zip(reversed(S), reversed(Y)):
            rho_i = 1.0 / (y_i @ s_i)
            a_i = rho_i * (s_i @ q)
            alphas.append((rho_i, a_i, s_i, y_i))
            q = q - a_i * y_i
        if S:                       # Nocedal's H0 scaling
            s_last, y_last = S[-1], Y[-1]
            q *= (s_last @ y_last) / (y_last @ y_last)
        for rho_i, a_i, s_i, y_i in reversed(alphas):
            beta = rho_i * (y_i @ q)
            q = q + s_i * (a_i - beta)
        p = -q
        if p @ g >= 0:              # not a descent direction -> reset memory
            p = -g
            S.clear(); Y.clear()

        # --- backtracking Armijo -------------------------------------------
        step, gp = 1.0, p @ g
        f_new, g_new, x_new = f, g, x
        ok = False
        for _ in range(max_ls):
            x_new = x + step * p
            f_new, g_new = f_and_grad(x_new)
            if np.isfinite(f_new) and f_new <= f + c1 * step * gp:
                ok = True
                break
            step *= 0.5
        if not ok:
            break

        s, y = x_new - x, g_new - g
        if s @ y > 1e-12:           # curvature condition
            S.append(s); Y.append(y)
        x, f, g = x_new, f_new, g_new
        if callback is not None:
            callback(f)

    return x, f, it
