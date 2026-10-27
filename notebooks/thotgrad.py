"""thotgrad — a minimal reverse-mode automatic differentiation engine.

Written to remove the PyTorch dependency from the PINN notebooks. The only
requirement is NumPy, which is already a core dependency of `optimizr` and of
every notebook in this repo.

Why this exists rather than `import torch`:

* PyTorch publishes no macOS x86_64 wheel past 2.2.2, and that build links
  against NumPy 1.x while this environment runs NumPy 2.x — the `.numpy()`
  bridge is permanently broken here.
* A PINN needs exactly three things: forward evaluation, derivatives of the
  network *with respect to its input* (up to second order), and gradients of
  the loss *with respect to the parameters*. That is a few hundred lines of
  well-understood calculus, not a reason to vendor a deep-learning framework.

Design: array-level (not scalar-level) reverse mode. Each `Tensor` records the
operation that produced it; `backward()` walks the tape in reverse topological
order accumulating gradients. Broadcasting is undone explicitly in
`_unbroadcast` so that gradients always match parameter shapes.

Correctness is pinned by `test_thotgrad.py`, which checks every operation and
the full PINN derivative stack against central finite differences.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the autodiff tape."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, requires_grad: bool = False, _children=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad or any(c.requires_grad for c in _children)
        self.grad: np.ndarray | None = None
        self._prev = tuple(_children)
        self._backward = _backward or (lambda: None)

    # -- construction -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    # -- binary ops ---------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _bw():
            if self.requires_grad:
                self._ensure_grad(); self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad(); other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _bw
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _bw():
            if self.requires_grad:
                self._ensure_grad(); self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad(); other.grad += self.data.T @ out.grad
        out._backward = _bw
        return out

    def __pow__(self, p: float):
        out = Tensor(self.data ** p, _children=(self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (p * self.data ** (p - 1))
        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return (Tensor(other) if not isinstance(other, Tensor) else other) + (-self)

    # -- unary ops ----------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad(); self.grad += out.grad * (1.0 - t * t)
        out._backward = _bw
        return out

    def sum(self):
        out = Tensor(self.data.sum(), _children=(self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad(); self.grad += out.grad * np.ones_like(self.data)
        out._backward = _bw
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), _children=(self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * np.ones_like(self.data) / n
        out._backward = _bw
        return out

    # -- tape ---------------------------------------------------------------
    def backward(self):
        """Reverse-mode sweep from this (scalar) node."""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)

        build(self)
        for v in topo:
            v.grad = None
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ---------------------------------------------------------------------------
# A tanh MLP whose input-derivatives are propagated analytically
# ---------------------------------------------------------------------------
class MLP:
    r"""Fully-connected tanh network with exact input derivatives.

    For a 1-D input the derivatives are propagated layer by layer alongside
    the activations. With :math:`h=\tanh(a)`:

    .. math::
        h' = (1-h^2)\,a', \qquad
        h'' = (1-h^2)\bigl(a'' - 2h\,(a')^2\bigr)

    Both recursions are written in `Tensor` operations, so the whole extended
    computation (value *and* derivatives) sits on the same tape. A single
    `backward()` therefore yields parameter gradients of a loss containing
    :math:`u`, :math:`u_x` and :math:`u_{xx}` — the double-backward a PINN
    needs, obtained without ever differentiating a derivative twice.
    """

    def __init__(self, n_in=1, n_out=1, width=32, depth=3, seed=0):
        rng = np.random.default_rng(seed)
        self.params: list[Tensor] = []
        self.layers: list[tuple[Tensor, Tensor]] = []
        dims = [n_in] + [width] * depth + [n_out]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            # Xavier/Glorot normal — the standard PINN initialisation
            W = Tensor(rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out)),
                       requires_grad=True)
            b = Tensor(np.zeros((1, fan_out)), requires_grad=True)
            self.layers.append((W, b))
            self.params += [W, b]

    def n_params(self) -> int:
        return sum(p.data.size for p in self.params)

    def __call__(self, x: Tensor) -> Tensor:
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = h.tanh()
        return h

    def forward_with_derivs(self, x: Tensor, order: int = 2):
        """Return (u, u_x, u_xx) for a column-vector input `x` of shape [N,1]."""
        one = Tensor(np.ones_like(x.data))
        zero = Tensor(np.zeros_like(x.data))
        h, dh, d2h = x, one, zero
        for i, (W, b) in enumerate(self.layers):
            a = h @ W + b
            da = dh @ W
            d2a = d2h @ W
            if i < len(self.layers) - 1:
                h = a.tanh()
                t = Tensor(1.0) - h * h              # sech^2
                dh = t * da
                d2h = t * (d2a - Tensor(2.0) * h * da * da)
            else:
                h, dh, d2h = a, da, d2a
        return (h, dh, d2h) if order == 2 else (h, dh)

    # -- flat parameter vector helpers (for L-BFGS) -------------------------
    def get_flat(self) -> np.ndarray:
        return np.concatenate([p.data.reshape(-1) for p in self.params])

    def set_flat(self, vec: np.ndarray) -> None:
        i = 0
        for p in self.params:
            n = p.data.size
            p.data = vec[i:i + n].reshape(p.data.shape).copy()
            i += n

    def grad_flat(self) -> np.ndarray:
        out = []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.data)
            out.append(g.reshape(-1))
        return np.concatenate(out)
