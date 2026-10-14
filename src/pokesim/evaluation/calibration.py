"""Turn the policy's value into a win probability only where data supports it.

The advisor's raw estimate is the PPO critic mapped to [0, 1]: its rewards include HP / faint shaping and
discounting, so it is not a win probability (estimates of 60-80% went with a 32% win rate in our battles). A
calibration maps it to one with two parameters, fitted per source (Switch / Showdown) on the battles of the
calibration split (pokesim.data.split_manifest), one estimate per battle at a fixed point (the first decision):

  P(win) = sigmoid(a * logit(raw) + b)      (L2 on a: with little data it shrinks to the base rate)

It is enabled only with at least --min-battles battles whose result is known and when, held out battle by battle
(leave-one-out), it beats the source's base rate on log loss. Otherwise the advisor shows no win probability.
A calibration is for one policy file (its hash): another policy's values mean something else.

  uv run python -m pokesim.evaluation.calibration fit --source showdown --policy checkpoints/ppo_baton_items.zip
  uv run python -m pokesim.evaluation.calibration report
"""
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pokesim.config import PROCESSED_DIR
from pokesim.evaluation.metrics import brier, log_loss, reliability

CALIBRATION = PROCESSED_DIR / "calibration.json"
MIN_BATTLES = 40
L2 = 1.0


def logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(raw, y, l2: float = L2, iters: int = 50) -> tuple[float, float]:
    """(a, b) of sigmoid(a * logit(raw) + b) by Newton's method, with L2 on a only."""
    x, y = logit(raw), np.asarray(y, float)
    X = np.stack([x, np.ones_like(x)], axis=1)
    w = np.zeros(2)
    reg = np.array([l2, 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + reg * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


@dataclass
class Calibration:
    source: str
    policy_sha: str | None
    a: float = 0.0
    b: float = 0.0
    battles: int = 0
    enabled: bool = False
    reason: str = ""
    scores: dict = field(default_factory=dict)
    fitted: str = ""

    def apply(self, raw: float | None) -> float | None:
        if raw is None or not self.enabled:
            return None
        return float(1 / (1 + math.exp(-(self.a * float(logit(np.array([raw]))[0]) + self.b))))


def evaluate_fit(raw: np.ndarray, y: np.ndarray, l2: float = L2) -> dict:
    """Leave-one-battle-out log loss / Brier of the calibration and of the base rate (one estimate per battle)."""
    n = len(y)
    cal, base = np.zeros(n), np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        a, b = fit_platt(raw[keep], y[keep], l2)
        cal[i] = 1 / (1 + np.exp(-(a * logit(raw[i:i + 1])[0] + b)))
        base[i] = (y[keep].sum() + 1) / (keep.sum() + 2)
    return {"log_loss": log_loss(cal, y), "log_loss_base": log_loss(base, y), "log_loss_raw": log_loss(raw, y),
            "brier": brier(cal, y), "brier_base": brier(base, y), "brier_raw": brier(raw, y),
            "reliability_raw": reliability(raw, y)}


def fit(raw, y, source: str, policy_sha: str | None, min_battles: int = MIN_BATTLES, l2: float = L2) -> Calibration:
    raw, y = np.asarray(raw, float), np.asarray(y, float)
    cal = Calibration(source, policy_sha, battles=len(y), fitted=time.strftime("%Y-%m-%d %H:%M"))
    if len(y) < max(min_battles, 3):
        cal.reason = f"{len(y)} battles, need {min_battles}"
        return cal
    cal.a, cal.b = fit_platt(raw, y, l2)
    cal.scores = evaluate_fit(raw, y, l2)
    if cal.scores["log_loss"] >= cal.scores["log_loss_base"]:
        cal.reason = "does not beat the base rate held out"
        return cal
    cal.enabled, cal.reason = True, "beats the base rate held out"
    return cal


def battle_estimates(ledger, decisions: list[dict], battle_ids: set[str], source: str,
                     policy_sha: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One raw estimate per battle (its first saved decision with one) and the result, for known results only."""
    first: dict[str, float] = {}
    for d in decisions:
        raw = d.get("win", d.get("win_prob"))  # live records before 2026-10-15 saved it as win_prob
        if d["battle_id"] in battle_ids and d["battle_id"] not in first and raw is not None:
            first[d["battle_id"]] = float(raw)
    raw, y, ids = [], [], []
    for b in ledger:
        if b.battle_id not in first or b.source != source or b.result not in ("win", "loss"):
            continue
        sha = ((b.versions or {}).get("policy") or {}).get("sha")
        if policy_sha is not None and sha != policy_sha:
            continue
        raw.append(first[b.battle_id])
        y.append(b.result == "win")
        ids.append(b.battle_id)
    return np.array(raw), np.array(y, float), ids


class Calibrations:
    """The saved calibrations (one per source and policy)."""

    def __init__(self, path: Path = CALIBRATION):
        data = json.loads(Path(path).read_text()) if Path(path).exists() else {"calibrations": []}
        self.items = [Calibration(**c) for c in data["calibrations"]]

    def get(self, source: str | None, policy_sha: str | None) -> Calibration | None:
        return next((c for c in self.items if c.source == source and c.policy_sha == policy_sha), None)

    def save(self, path: Path = CALIBRATION) -> None:
        Path(path).write_text(json.dumps({"calibrations": [asdict(c) for c in self.items]}, indent=1))


def main() -> None:
    from pokesim.data.curate_battles import CURATED, load
    from pokesim.data.split_manifest import MANIFEST, Splits
    from pokesim.runtime.bundle import file_hash

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--source", choices=["switch", "showdown"], required=True)
    f.add_argument("--policy", help="the policy file the estimates came from (records saved with its hash)")
    f.add_argument("--split", default="calibration")
    f.add_argument("--min-battles", type=int, default=MIN_BATTLES)
    sub.add_parser("report")
    args = ap.parse_args()
    saved = Calibrations()
    if args.cmd == "report":
        for c in saved.items:
            scores = {k: round(v, 3) for k, v in c.scores.items() if isinstance(v, float)}
            print(f"{c.source:8s} policy {c.policy_sha}  {'ON ' if c.enabled else 'off'}  {c.battles} battles  "
                  f"a={c.a:.2f} b={c.b:.2f}  {c.reason}  {json.dumps(scores)}")
        return
    ledger = load()
    decisions = [json.loads(line) for line in (CURATED / "decisions.jsonl").read_text().splitlines()]
    ids = Splits(MANIFEST).ids(args.split)
    sha = file_hash(args.policy) if args.policy else None
    raw, y, used = battle_estimates(ledger, decisions, ids, args.source, sha)
    cal = fit(raw, y, args.source, sha, args.min_battles)
    saved.items = [c for c in saved.items if not (c.source == cal.source and c.policy_sha == cal.policy_sha)] + [cal]
    saved.save()
    print(f"{args.source} / policy {sha}: {len(used)} battles from the {args.split} split -> "
          f"{'enabled' if cal.enabled else 'not enabled'} ({cal.reason}) -> {CALIBRATION}")
    for k, v in cal.scores.items():
        if isinstance(v, float):
            print(f"  {k:14s} {v:.3f}")


if __name__ == "__main__":
    main()
