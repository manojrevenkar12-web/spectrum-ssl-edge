"""Accuracy, calibration, selective prediction and open-set metrics. Pure numpy, unit-tested."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax

SNR_BINS = [-4, 0, 4, 8, 12, 16, 20]


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(logits.argmax(1) == y)) if len(y) else float("nan")


def accuracy_by_snr(logits: np.ndarray, y: np.ndarray, snr: np.ndarray, bins: list[float] = SNR_BINS) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (snr >= lo) & (snr < hi) if hi != bins[-1] else (snr >= lo) & (snr <= hi)
        out.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "accuracy": accuracy(logits[sel], y[sel])})
    return out


def confusion(logits: np.ndarray, y: np.ndarray, k: int) -> list[list[int]]:
    c = np.zeros((k, k), dtype=int)
    np.add.at(c, (y, logits.argmax(1)), 1)
    return c.tolist()


# --- calibration -------------------------------------------------------------------------------------


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(log_softmax(logits, axis=1)[np.arange(len(y)), y]))


def brier(logits: np.ndarray, y: np.ndarray) -> float:
    p = softmax(logits, axis=1)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width confidence bins (Guo et al., 2017)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(sel.sum()),
                "confidence": float(conf[sel].mean()) if sel.any() else None,
                "accuracy": float(correct[sel].mean()) if sel.any() else None,
            }
        )
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar T > 0 minimising validation NLL of logits / T."""
    res = minimize_scalar(lambda lt: nll(logits / np.exp(lt), y), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def calibration_summary(probs: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-12
    return {
        "ece": ece(probs, y),
        "nll": float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps))),
        "brier": float(np.mean(np.sum((probs - np.eye(probs.shape[1])[y]) ** 2, axis=1))),
        "accuracy": float(np.mean(probs.argmax(1) == y)),
    }


# --- selective prediction ----------------------------------------------------------------------------


def risk_coverage(probs: np.ndarray, y: np.ndarray, points: int = 20) -> dict:
    """Abstain on the least confident windows. AURC = area under the risk-coverage curve (lower is better)."""
    conf = probs.max(1)
    order = np.argsort(-conf, kind="stable")
    wrong = (probs.argmax(1) != y)[order].astype(float)
    n = len(y)
    cum_risk = np.cumsum(wrong) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    curve = []
    for c in np.linspace(1 / points, 1, points):
        k = max(1, int(round(c * n)))
        curve.append({"coverage": float(k / n), "risk": float(cum_risk[k - 1])})
    return {
        "aurc": float(np.trapezoid(cum_risk, cov)),
        "curve": curve,
        "risk_at_80": float(cum_risk[max(0, int(0.8 * n) - 1)]),
    }


# --- open set ----------------------------------------------------------------------------------------


def auroc(score_in: np.ndarray, score_out: np.ndarray) -> float:
    """P(score_out > score_in): higher score = more likely novel. Ties count half (Mann-Whitney)."""
    s = np.concatenate([score_in, score_out])
    ranks = _rankdata(s)
    r_out = ranks[len(score_in) :].sum()
    n_in, n_out = len(score_in), len(score_out)
    return float((r_out - n_out * (n_out + 1) / 2) / (n_in * n_out))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def fpr_at_tpr(score_in: np.ndarray, score_out: np.ndarray, tpr: float = 0.95) -> float:
    """Share of known windows flagged novel when the threshold catches `tpr` of the novel ones."""
    thr = np.quantile(score_out, 1 - tpr)
    return float(np.mean(score_in >= thr))


def msp_score(logits: np.ndarray) -> np.ndarray:
    return -softmax(logits, axis=1).max(1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Liu et al., 2020: negative free energy of known classes; higher = more novel."""
    return -logsumexp(logits, axis=1)


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance on embeddings (Lee et al., 2018)."""

    def __init__(self, z: np.ndarray, y: np.ndarray, k: int, shrink: float = 1e-3) -> None:
        self.means = np.stack([z[y == c].mean(0) for c in range(k)])
        centred = z - self.means[y]
        cov = centred.T @ centred / len(z)
        cov += shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.prec = np.linalg.inv(cov)

    def score(self, z: np.ndarray) -> np.ndarray:
        d = z[:, None, :] - self.means[None, :, :]
        m = np.einsum("nkd,de,nke->nk", d, self.prec, d)
        return m.min(1)


class KNN:
    """Cosine distance to the k-th nearest training embedding (Sun et al., 2022)."""

    def __init__(self, z: np.ndarray, k: int = 10) -> None:
        self.bank = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
        self.k = k

    def score(self, z: np.ndarray, batch: int = 2048) -> np.ndarray:
        q = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
        out = []
        for i in range(0, len(q), batch):
            sim = q[i : i + batch] @ self.bank.T
            kth = -np.partition(-sim, self.k - 1, axis=1)[:, self.k - 1]
            out.append(1 - kth)
        return np.concatenate(out)
