"""Agent-agnostic VAE membership-inference scorer.

Real (analytic, hand-derived) backprop for a small encoder/decoder VAE —
no PyTorch dependency, CPU-only, matching the ELBO derivation in
docs/source/algorithms/variational_calculus.rst and the attack design in
docs/source/algorithms/membership_inference.rst.

Architecture (all shapes are (batch, dim)):
    x -> tanh(W1 x + b1) = h1
    h1 -> mu = Wmu h1 + bmu,  logvar = Wlv h1 + blv
    z = mu + exp(0.5*logvar) * eps                      (reparameterisation)
    z -> tanh(W2 z + b2) = h2
    h2 -> x_hat = W3 h2 + b3

Loss (per example, Gaussian decoder with unit variance):
    recon = 0.5 * ||x - x_hat||^2
    kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    elbo_loss = recon + kl              (minimised; -elbo_loss = ELBO)
"""
from __future__ import annotations

import numpy as np


class TinyVAE:
    def __init__(self, dim: int, hidden: int = 16, latent: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = lambda *shape: rng.normal(0, 0.1, size=shape)  # noqa: E731
        self.dim, self.hidden, self.latent = dim, hidden, latent
        self.params = {
            "W1": s(hidden, dim), "b1": np.zeros(hidden),
            "Wmu": s(latent, hidden), "bmu": np.zeros(latent),
            "Wlv": s(latent, hidden), "blv": np.zeros(latent),
            "W2": s(hidden, latent), "b2": np.zeros(hidden),
            "W3": s(dim, hidden), "b3": np.zeros(dim),
        }
        # Adam moment buffers — same update rule as
        # docs/source/algorithms/training.rst (AdamW section), weight_decay=0
        # here since the VAE is tiny and regularisation is dominated by KL.
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def forward(self, X: np.ndarray, rng: np.random.Generator, sample: bool = True):
        p = self.params
        a1 = X @ p["W1"].T + p["b1"]
        h1 = np.tanh(a1)
        mu = h1 @ p["Wmu"].T + p["bmu"]
        logvar = h1 @ p["Wlv"].T + p["blv"]
        std = np.exp(0.5 * logvar)
        eps = rng.normal(size=mu.shape) if sample else np.zeros_like(mu)
        z = mu + std * eps

        a2 = z @ p["W2"].T + p["b2"]
        h2 = np.tanh(a2)
        x_hat = h2 @ p["W3"].T + p["b3"]

        cache = dict(X=X, h1=h1, mu=mu, logvar=logvar, std=std, eps=eps, z=z,
                     h2=h2, x_hat=x_hat)
        return x_hat, mu, logvar, cache

    def elbo_terms(self, X: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (recon_loss, kl) per example — both >= 0, both minimised."""
        recon = 0.5 * np.sum((X - cache["x_hat"]) ** 2, axis=1)
        mu, logvar = cache["mu"], cache["logvar"]
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=1)
        return recon, kl

    def backward(self, cache: dict, lr: float = 1e-2, beta1=0.9, beta2=0.999, eps_adam=1e-8):
        p = self.params
        X, h1, mu, logvar, std, e, z, h2, x_hat = (
            cache["X"], cache["h1"], cache["mu"], cache["logvar"], cache["std"],
            cache["eps"], cache["z"], cache["h2"], cache["x_hat"],
        )
        n = X.shape[0]

        dxhat = (x_hat - X) / n
        dW3 = dxhat.T @ h2
        db3 = dxhat.sum(0)
        dh2 = dxhat @ p["W3"]
        da2 = dh2 * (1 - h2 ** 2)
        dW2 = da2.T @ z
        db2 = da2.sum(0)
        dz = da2 @ p["W2"]

        dmu = dz + mu / n
        dlogvar = dz * (0.5 * std * e) + 0.5 * (np.exp(logvar) - 1) / n

        dWmu = dmu.T @ h1
        dbmu = dmu.sum(0)
        dWlv = dlogvar.T @ h1
        dblv = dlogvar.sum(0)

        dh1 = dmu @ p["Wmu"] + dlogvar @ p["Wlv"]
        da1 = dh1 * (1 - h1 ** 2)
        dW1 = da1.T @ X
        db1 = da1.sum(0)

        grads = dict(W1=dW1, b1=db1, Wmu=dWmu, bmu=dbmu, Wlv=dWlv, blv=dblv,
                     W2=dW2, b2=db2, W3=dW3, b3=db3)

        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    def fit(self, X: np.ndarray, epochs: int = 400, lr: float = 5e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            x_hat, mu, logvar, cache = self.forward(X, rng, sample=True)
            recon, kl = self.elbo_terms(X, cache)
            self.backward(cache, lr=lr)
            history.append(float(np.mean(recon + kl)))
        return np.array(history)

    def elbo_score(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        """Higher = better reconstruction under the fitted member distribution
        = the attack score s_VAE(x) from membership_inference.rst
        (ELBO = -(recon + kl), evaluated with the deterministic mean z=mu,
        i.e. no sampling noise, for a reproducible score)."""
        rng = np.random.default_rng(seed)
        _, _, _, cache = self.forward(X, rng, sample=False)
        recon, kl = self.elbo_terms(X, cache)
        return -(recon + kl)
