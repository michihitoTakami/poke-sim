"""E1: fixed tactical positions with an expected answer, run through the lookahead worker.

Each fixture (src/pokesim/evaluation/fixtures/*.json) is a position as the worker takes it (one or more sampled
worlds) and what a correct decision must satisfy:
  {"name", "category", "note", "samples": [SAMPLE, ...],
   "expect": [{"better": label, "than": label} | {"best": label} | {"best_in": [labels]} | {"not_best": label}],
   "opponent": {label: p}  (optional: a fixed opponent mix; otherwise the planner's prediction),
   "known_noisy": true     (optional: the answer is within the rollouts' noise; reported, not used as a test)}
"better" compares values against the predicted opponent; "best" / "best_in" / "not_best" look at the top of our mix.

A method is the worker settings plus how the worlds are solved:
  legacy   greedy continuation, HP-balance scoring, per-world equilibria averaged (pokesim.search.lookahead)
  planner  the advisor's planner settings (pokesim.live.advisor.PLANNER_SEARCH) and one common mix

The fixtures check known failure patterns; passing them does not show a higher win rate against people.

  uv run python -m pokesim.evaluation.fixtures [--method legacy planner] [-v]
"""
import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.search.lookahead import Advice, aggregate
from pokesim.search.planner import plan

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = {"seeds": 2, "rollout": 3}


def methods() -> dict[str, tuple[dict, str]]:
    from pokesim.live.advisor import PLANNER, PLANNER_SEARCH

    return {"legacy": (LEGACY, "aggregate"),
            "planner": (dict(PLANNER_SEARCH, deadlineMs=None), f"plan:{PLANNER['rho']}")}


@dataclass
class Outcome:
    fixture: str
    method: str
    passed: bool
    checks: list[tuple[dict, bool]] = field(default_factory=list)
    top: str = ""
    values: dict[str, float] = field(default_factory=dict)


def load(folder: Path = FIXTURES) -> list[dict]:
    return [dict(json.loads(p.read_text()), file=p.name) for p in sorted(folder.glob("*.json"))]


class Worker:
    def __init__(self):
        self.proc = subprocess.Popen(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def query(self, samples: list[dict], settings: dict) -> list[dict]:
        req = {"format": FORMAT, "samples": samples, **{k: v for k, v in settings.items() if v is not None}}
        self.proc.stdin.write(json.dumps(req) + "\n")
        reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["results"]

    def close(self) -> None:
        self.proc.kill()


def solve(results: list[dict], how: str, opponent: dict | None) -> Advice:
    if how == "aggregate":
        adv = aggregate(results)
        if opponent:  # values against the fixed mix instead of each world's equilibrium
            value = np.zeros(len(adv.labels))
            for res in results:
                idx = [res["p1"].index(lab) for lab in adv.labels]
                q = np.array([opponent.get(lab, 0.0) for lab in res["p2"]])
                if q.sum() > 0:
                    value += np.array(res["payoff"], dtype=float)[idx] @ (q / q.sum()) / len(results)
            adv.value = value
        return adv
    rho = float(how.split(":")[1])
    return plan(results, rho=rho, cloned=opponent, cloned_weight=1.0 if opponent else 0.5)


def check(expect: dict, adv: Advice) -> bool:
    value = dict(zip(adv.labels, adv.value))
    top = adv.labels[int(np.argmax(adv.prob))]
    if "better" in expect:
        return value[expect["better"]] > value[expect["than"]]
    if "best" in expect:
        return top == expect["best"]
    if "best_in" in expect:
        return top in expect["best_in"]
    if "not_best" in expect:
        return top != expect["not_best"]
    raise ValueError(f"unknown expectation {expect}")


def run(fixtures: list[dict], names: list[str], worker: Worker | None = None) -> list[Outcome]:
    own = worker is None
    worker = worker or Worker()
    out = []
    try:
        for fx in fixtures:
            for name in names:
                settings, how = methods()[name]
                adv = solve(worker.query(fx["samples"], settings), how, fx.get("opponent"))
                checks = [(e, check(e, adv)) for e in fx["expect"]]
                out.append(Outcome(fx["name"], name, all(ok for _, ok in checks), checks,
                                   adv.labels[int(np.argmax(adv.prob))],
                                   {lab: round(float(v), 3) for lab, v in zip(adv.labels, adv.value)}))
    finally:
        if own:
            worker.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs="+", default=["legacy", "planner"], choices=["legacy", "planner"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    fixtures = load()
    results = run(fixtures, args.method)
    by = {(r.fixture, r.method): r for r in results}
    print(f"{'fixture':38s} {'category':14s} " + " ".join(f"{m:>8s}" for m in args.method))
    for fx in fixtures:
        cells = " ".join(f"{'pass' if by[(fx['name'], m)].passed else 'FAIL':>8s}" for m in args.method)
        print(f"{fx['name']:38s} {fx['category']:14s} {cells}" + ("  (noisy)" if fx.get("known_noisy") else ""))
        if args.verbose:
            for m in args.method:
                r = by[(fx["name"], m)]
                print(f"    {m}: top {r.top}  values {r.values}")
    for m in args.method:
        n = sum(by[(fx["name"], m)].passed for fx in fixtures)
        print(f"{m}: {n}/{len(fixtures)} passed")


if __name__ == "__main__":
    main()
