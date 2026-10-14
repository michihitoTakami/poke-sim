"""Scores for probabilities and win rates, with battles (not turns) as the unit.

A battle's turns are not independent trials: every per-turn score here can weight each battle equally, and the
bootstrap resamples whole battles (or any other group: day, opponent).
"""
import math
from collections.abc import Callable, Sequence

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n independent trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def group_weights(groups: Sequence) -> np.ndarray:
    """Weights giving every group (battle) the same total weight 1."""
    groups = list(groups)
    counts: dict = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    return np.array([1.0 / counts[g] for g in groups])


def brier(p, y, w=None) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(p, y, w=None, eps: float = 1e-6) -> float:
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def reliability(p, y, bins: int = 5, w=None) -> list[dict]:
    """Per equal-width probability bin: weight, mean prediction, observed rate."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if w[m].sum() > 0:
            out.append({"bin": (round(lo, 2), round(hi, 2)), "weight": float(w[m].sum()),
                        "predicted": float(np.average(p[m], weights=w[m])),
                        "observed": float(np.average(y[m], weights=w[m]))})
    return out


def bootstrap(stat: Callable[[np.ndarray], float], groups: Sequence, n: int = 2000, seed: int = 0,
              level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of stat(row indices) over resamples of whole groups (battles, days, opponents)."""
    groups = list(groups)
    rows: dict = {}
    for i, g in enumerate(groups):
        rows.setdefault(g, []).append(i)
    keys = list(rows)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        values.append(stat(np.concatenate([rows[keys[k]] for k in pick])))
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
