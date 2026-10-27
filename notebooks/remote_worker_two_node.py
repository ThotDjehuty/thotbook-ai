#!/usr/bin/env python3
"""Slab worker for the two-node distributed PINN campaign (paper Stage 5).

Self-contained: only numpy (>=1.20) and thotgrad.py in the same directory.
The math is IDENTICAL to the notebook's `loss_and_grad_given` (Dirac 1+1D
stationary pair, mean of squared residuals over the slab, plus the weighted
boundary terms at x=0), evaluated on one collocation slab from a fixed
parameter vector loaded from disk.

Prints EXACTLY ONE stdout line: a JSON object with float64 timings and the
gradient base64-encoded little-endian (dtype '<f8', cross-platform exact).
"""
import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one task, one core

import argparse
import base64
import json
import sys
import time

import numpy as np

from thotgrad import MLP, Tensor


def evaluate(E, m, x0, xs, theta):
    mdl = MLP(n_in=1, n_out=2, width=64, depth=4, seed=0)
    mdl.set_flat(theta)
    x = Tensor(xs.reshape(-1, 1))
    s1 = np.array([[1.0], [0.0]])
    s2 = np.array([[0.0], [1.0]])

    def closure():
        out, dout = mdl.forward_with_derivs(x, order=1)
        u, v = out @ Tensor(s1), out @ Tensor(s2)
        du, dv = dout @ Tensor(s1), dout @ Tensor(s2)
        r1 = du - Tensor(E + m) * v
        r2 = dv + Tensor(E - m) * u
        o0 = mdl(Tensor(x0))
        b_u = o0 @ Tensor(s1) - Tensor(1.0)
        b_v = o0 @ Tensor(s2)
        return ((r1 * r1).mean() + (r2 * r2).mean()
                + Tensor(10.0) * ((b_u * b_u).mean() + (b_v * b_v).mean()))

    t0 = time.perf_counter()
    L = closure()
    L.backward()
    compute_ms = (time.perf_counter() - t0) * 1e3
    return float(L.data), mdl.grad_flat(), compute_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--x0", required=True)
    ap.add_argument("--name", default="w")
    a = ap.parse_args()

    E, m = 2.0, 1.0

    t0 = time.perf_counter()
    theta = np.load(a.theta)
    xs = np.load(a.x)
    x0 = np.load(a.x0)
    load_ms = (time.perf_counter() - t0) * 1e3

    loss, grad, compute_ms = evaluate(E, m, x0, xs, theta)

    sys.stdout.write(json.dumps({
        "worker": a.name,
        "n": int(xs.size),
        "loss": loss,
        "load_ms": round(load_ms, 2),
        "compute_ms": round(compute_ms, 2),
        "g_maxabs": float(np.max(np.abs(grad))),
        "grad_b64": base64.b64encode(np.ascontiguousarray(grad, dtype="<f8").tobytes()).decode(),
    }))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
