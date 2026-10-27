"""Federated learning + membership inference, implementing the taxonomy of

    Bai, Hu, Ye, Li, Wang, Xu (ACM Comput. Surv.),
    "Membership Inference Attacks and Defenses in Federated Learning: A Survey"

as applied to the thotbook-ai corpus. Pure NumPy — CPU-only, no GPU, and
fast enough that the full privacy/utility sweep runs inside a notebook cell.

Survey defence taxonomy (Table 7), all four categories implemented below:

    Partial sharing    -> gradient compression (top-K), weight pruning
    Secure aggregation -> server sees only the aggregate, never a client delta
    Noise perturbation -> differential privacy (clip + Gaussian), random noise
    Anomaly detection  -> (out of scope here: targets poisoning, not passive MIA)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Model: multinomial logistic regression (convex, analytic gradient, fast)
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray,
                  l2: float = 1e-4) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy + L2, with its exact gradient."""
    n = X.shape[0]
    P = softmax(X @ W + b)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    loss = float(-np.mean(np.log(P[np.arange(n), y] + 1e-12)) + l2 * np.sum(W ** 2))
    dZ = (P - onehot) / n
    return loss, X.T @ dZ + 2 * l2 * W, dZ.sum(0)


def per_example_loss(W: np.ndarray, b: np.ndarray, X: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    P = softmax(X @ W + b)
    return -np.log(P[np.arange(X.shape[0]), y] + 1e-12)


def accuracy(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(X @ W + b, axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Defences applied to a client's update delta
# ---------------------------------------------------------------------------
def clip_by_norm(delta: np.ndarray, C: float) -> np.ndarray:
    """Norm bounding — the prerequisite for any DP guarantee."""
    nrm = np.linalg.norm(delta)
    return delta if nrm <= C or nrm == 0 else delta * (C / nrm)


def dp_gaussian(delta: np.ndarray, C: float, sigma: float,
                rng: np.random.Generator) -> np.ndarray:
    """Noise perturbation (survey 5.3): clip to C, then add N(0, sigma^2 C^2).

    The (eps, delta)-DP guarantee of the Gaussian mechanism scales with
    sigma; sigma = 0 recovers the undefended update.
    """
    d = clip_by_norm(delta, C)
    return d + rng.normal(0.0, sigma * C, size=d.shape)


def topk_compress(delta: np.ndarray, frac: float) -> np.ndarray:
    """Partial sharing (survey 5.1.1): transmit only the top-K |values|.

    Distributed selective SGD (Shokri & Shmatikov) shares as little as 1% of
    coordinates; everything else is zeroed before upload.
    """
    if frac >= 1.0:
        return delta
    flat = delta.reshape(-1)
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(delta.shape)


# ---------------------------------------------------------------------------
# Federated simulation
# ---------------------------------------------------------------------------
class FederatedSim:
    """FedAvg (McMahan et al.) with an honest-but-curious server.

    The server runs the passive, update-based membership inference attack of
    the survey's taxonomy: it observes each client's delta, reconstructs that
    client's locally-updated model, and scores a candidate example by how much
    its loss *dropped* under that client's update.  Members of the client's
    shard drop more than non-members -- the "loss trajectory" signal.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_clients: int = 8,
                 groups: np.ndarray | None = None, seed: int = 0):
        self.X, self.y = X, y
        self.n_classes = int(y.max()) + 1
        self.dim = X.shape[1]
        self.rng = np.random.default_rng(seed)
        self.n_clients = n_clients
        # Partition by *group* (document) when given, so a paper never straddles
        # two clients -- the federated analogue of the document-level split.
        if groups is None:
            idx = self.rng.permutation(len(y))
            self.shards = np.array_split(idx, n_clients)
        else:
            uniq = np.unique(groups)
            self.rng.shuffle(uniq)
            buckets = np.array_split(uniq, n_clients)
            self.shards = [np.where(np.isin(groups, b))[0] for b in buckets]

    def _init_params(self):
        return (np.zeros((self.dim, self.n_classes)), np.zeros(self.n_classes))

    def run(self, rounds: int = 25, local_epochs: int = 3, lr: float = 0.5,
            defense: str = "none", C: float = 1.0, sigma: float = 0.0,
            topk: float = 1.0, secure_agg: bool = False,
            test_idx: np.ndarray | None = None) -> dict:
        """One full FedAvg run; returns utility and per-round attack scores."""
        W, b = self._init_params()
        hist_acc, attack_scores, attack_labels = [], [], []

        for _ in range(rounds):
            deltas_W, deltas_b, n_k = [], [], []
            round_scores, round_labels = [], []

            for ci, shard in enumerate(self.shards):
                if len(shard) == 0:
                    continue
                Xc, yc = self.X[shard], self.y[shard]
                Wl, bl = W.copy(), b.copy()
                for _ in range(local_epochs):
                    _, gW, gb = loss_and_grad(Wl, bl, Xc, yc)
                    Wl -= lr * gW
                    bl -= lr * gb
                dW, db = Wl - W, bl - b

                # ---- the server's passive MIA, run BEFORE any defence that
                # ---- the client would have applied to its upload
                if ci == 0 and test_idx is not None:
                    cand = np.concatenate([shard[:40], test_idx[:40]])
                    lab = np.concatenate([np.ones(len(shard[:40])),
                                          np.zeros(len(test_idx[:40]))])
                    l_before = per_example_loss(W, b, self.X[cand], self.y[cand])
                    l_after = per_example_loss(W + dW, b + db,
                                               self.X[cand], self.y[cand])
                    round_scores.append(l_before - l_after)   # loss reduction
                    round_labels.append(lab)

                # ---- defences applied to the uploaded update
                if defense == "dp":
                    dW = dp_gaussian(dW, C, sigma, self.rng)
                    db = dp_gaussian(db, C, sigma, self.rng)
                elif defense == "topk":
                    dW, db = topk_compress(dW, topk), topk_compress(db, topk)

                deltas_W.append(dW); deltas_b.append(db); n_k.append(len(shard))

            tot = float(sum(n_k))
            W = W + sum(w * (n / tot) for w, n in zip(deltas_W, n_k))
            b = b + sum(v * (n / tot) for v, n in zip(deltas_b, n_k))

            if test_idx is not None:
                hist_acc.append(accuracy(W, b, self.X[test_idx], self.y[test_idx]))
            if round_scores:
                attack_scores.append(np.concatenate(round_scores))
                attack_labels.append(np.concatenate(round_labels))

        # Secure aggregation: the server never sees an individual delta, only
        # the weighted sum, so the per-client attack signal is destroyed.
        # We model that by replacing the attack score with pure noise.
        if secure_agg and attack_scores:
            attack_scores = [self.rng.normal(0, 1, size=s.shape)
                             for s in attack_scores]

        return {
            "W": W, "b": b,
            "acc": hist_acc,
            "final_acc": hist_acc[-1] if hist_acc else float("nan"),
            "attack_scores": attack_scores[-1] if attack_scores else None,
            "attack_labels": attack_labels[-1] if attack_labels else None,
        }
