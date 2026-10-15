"""One strategy for every sampled world: the lookahead's root decision when we do not know the opponent's sets.

The legacy aggregate (pokesim.search.lookahead.aggregate) solves each sampled world's matrix game on its own and
averages the solutions. That mixes strategies that each assume we know the world: "attack if it is set A, set up if
it is set B" averages to half attack / half set up, which can be bad in both worlds while a third action is safe in
both. Here our mix x is shared by all worlds (we see the same thing in each), and solved as one linear program:

  maximize over x:  (1 - rho) * sum_g w_g x^T Q_g q_g  +  rho * sum_g w_g min_b x^T Q_g[:, b]

Q_g is world g's payoff matrix (our options x the opponent's), w_g its weight, q_g the predicted opponent mix in it
(fixed before solving: the opponent model's prediction, optionally mixed with a policy cloned from people). rho is
how much we guard against the prediction being wrong: each world's worst case is taken over the opponent's options
in that world (the opponent knows its own set, so its options differ by world). rho = 0 trusts the prediction,
rho = 1 is the worst case in every world.

Each world is its own group for the worst case: our sets are the same in all of them, so worlds only differ in what
the opponent knows anyway.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from pokesim.search.lookahead import Advice, clean, opponent_mix, solve_zero_sum

RHO = 0.3  # initial guard against a wrong prediction; to be set on development data, not on test results


@dataclass
class World:
    payoff: np.ndarray  # our options (common order) x this world's opponent options
    opponent: list[str]  # this world's opponent labels
    weight: float = 1.0
    predicted: np.ndarray | None = None  # opponent mix in this world (fixed before solving)


@dataclass
class Solution:
    prob: np.ndarray  # our common mix
    expected: np.ndarray  # per action: sum_g w_g Q_g q_g (against the prediction)
    worst: np.ndarray  # per action: sum_g w_g min_b Q_g[a, b]
    status: dict = field(default_factory=dict)


def solve_common(worlds: list[World], rho: float = RHO, secondary: np.ndarray | None = None) -> Solution:
    """The common mix maximizing the objective above (scipy's HiGHS LP). Falls back to the best action against the
    prediction when the solver fails."""
    n = worlds[0].payoff.shape[0]
    w = np.array([wd.weight for wd in worlds], dtype=float)
    w = w / w.sum()
    expected = sum(wg * wd.payoff @ wd.predicted for wg, wd in zip(w, worlds))
    worst = sum(wg * wd.payoff.min(axis=1) for wg, wd in zip(w, worlds))
    g = len(worlds)
    # variables: x (n), t (g); linprog minimizes
    c = -np.concatenate([(1 - rho) * expected, rho * w])
    rows = []
    for k, wd in enumerate(worlds):
        for b in range(wd.payoff.shape[1]):  # t_k - x^T Q_k[:, b] <= 0
            row = np.zeros(n + g)
            row[:n] = -wd.payoff[:, b]
            row[n + k] = 1.0
            rows.append(row)
    a_eq = np.concatenate([np.ones(n), np.zeros(g)])[None, :]
    bounds = [(0, 1)] * n + [(None, None)] * g
    res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=a_eq, b_eq=[1.0], bounds=bounds,
                  method="highs")
    if res.success:
        if secondary is not None:
            # Lexicographic: preserve the optimal win objective, then prefer earlier terminal wins.
            # No HP reward, discount, or speed bonus can trade away meaningful winning probability.
            tie = linprog(-np.r_[secondary, np.zeros(g)],
                          A_ub=np.vstack([rows, c]), b_ub=np.r_[np.zeros(len(rows)), res.fun + 1e-9],
                          A_eq=a_eq, b_eq=[1.0], bounds=bounds, method='highs')
            if tie.success:
                res = tie
        x = np.clip(res.x[:n], 0, None)
        x[x < 1e-8] = 0
        x = x / x.sum()
        status = {"solver": "highs", "objective": float(-res.fun)}
    else:
        x = np.zeros(n)
        x[int(np.argmax((1 - rho) * expected + rho * worst))] = 1.0
        status = {"solver": "fallback", "reason": res.message}
    # Recompute the realized objective, including the risk term, for comparing switch candidates and fallback.
    objective = (1 - rho) * float(x @ expected) + rho * sum(
        wg * float((x @ wd.payoff).min()) for wg, wd in zip(w, worlds))
    status["objective"] = objective
    return Solution(x, expected, worst, status)


def base_label(label: str) -> str:
    return ":".join(label.split(":")[:2])


def predict(payoff: np.ndarray, labels: list[str], opponent_model: dict, last_move: str | None,
            cloned: dict[str, float] | None, cloned_weight: float) -> np.ndarray:
    """The opponent's mix in one world, fixed before planning: the search's prediction (equilibrium / softmax best
    response to our equilibrium, see opponent_mix), a share for repeating its last move, then a share for the
    cloned human policy (its labels have no Mega variants: split over them)."""
    x, y = solve_zero_sum(payoff)
    q = opponent_mix(payoff, x, y, opponent_model)
    repeat = opponent_model.get("repeat", 0.0)
    if repeat > 0 and last_move in labels:
        q = (1 - repeat) * q
        q[labels.index(last_move)] += repeat
    if cloned:
        bases = [base_label(lab) for lab in labels]
        bc = np.array([cloned.get(b, 0.0) / bases.count(b) for b in bases])
        if bc.sum() > 0:
            q = (1 - cloned_weight) * q + cloned_weight * bc / bc.sum()
    return q / q.sum()


def plan(results: list[dict], opponent_model: dict | None = None, last_move: str | None = None,
         rho: float = RHO, cloned: dict[str, float] | None = None, cloned_weight: float = 0.5,
         pure: bool = False) -> Advice:
    """Advice from the lookahead's sampled worlds with one common mix (see the module docstring).
    value[a] is the expected score against the predicted opponent; the opponent dict the predicted mix averaged
    over worlds. Worlds where every branch failed are left out and counted in status."""
    labels = results[0]["p1"]
    tree = results[0].get('terminal_search')
    if tree and tree.get('immediate_wins'):
        # A proven immediate win in every sampled world/reply/chance path outranks all bootstrap/rollout guesses.
        candidates = set.intersection(*(set(r['terminal_search']['immediate_wins']) for r in results))
        if candidates:
            chosen = min(candidates)
            prob = np.array([float(lab == chosen) for lab in labels])
            return Advice(labels, prob, prob.copy(), {},
                          status={'solver': 'sampled-world-immediate-win', 'objective': 1.,
                                  'terminal_search': tree, 'worlds': len(results)}, worst=prob.copy())
    worlds, failed_cells, failed_worlds = [], 0, 0
    speed_terms = []
    extras: list[dict[str, np.ndarray]] = []  # per world: payoff per continuation policy, share resolved
    for res in results:
        try:
            payoff, opp, failed = clean(res)
        except ValueError:
            failed_worlds += 1
            continue
        idx = [res["p1"].index(lab) for lab in labels]  # our options are the same in every world
        payoff = payoff[idx]
        q = predict(payoff, opp, opponent_model or {}, last_move, cloned, cloned_weight)
        worlds.append(World(payoff, opp, 1.0, q))
        cols = [res["p2"].index(lab) for lab in opp]
        if tree:
            speed_terms.append(matrix(res['winSpeed'], idx, cols, np.zeros_like(payoff)) @ q)
        extras.append({name: matrix(m, idx, cols, payoff) for name, m in res.get("byContinuation", {}).items()}
                      | ({"_resolved": matrix(res["resolved"], idx, cols, np.zeros_like(payoff))}
                         if "resolved" in res else {}))
        failed_cells += failed
    if not worlds:
        raise ValueError("every sampled world failed")
    secondary = np.mean(speed_terms, axis=0) if speed_terms else None
    sol = solve_common(worlds, rho, secondary)
    if pure:
        # The caller plays one top move. Rank actual pure-action utilities, not the largest probability of a
        # mixed equilibrium whose protection requires randomizing between ALL of its actions.
        objective = (1 - rho) * sol.expected + rho * sol.worst
        candidates = np.flatnonzero(objective >= objective.max() - 1e-9)
        chosen = int(candidates[np.argmax(secondary[candidates])]) if secondary is not None else int(candidates[0])
        sol.prob = np.eye(len(labels))[chosen]
        sol.status.update(solver='pure-robust', objective=float(objective[chosen]))
    opponent: dict[str, float] = {}
    for wd in worlds:
        for lab, p in zip(wd.opponent, wd.predicted):
            opponent[lab] = opponent.get(lab, 0.0) + float(p) / len(worlds)
    status = dict(sol.status, rho=rho, worlds=len(worlds), failed_worlds=failed_worlds, failed_cells=failed_cells)
    status.update(diagnose(sol.prob, worlds, extras))
    if tree:
        status['terminal_search'] = tree
        status['root_evidence'] = [{'label': lab, 'expected': float(sol.expected[i]),
                                    'worst': float(sol.worst[i]),
                                    'trials': sum(sum(r['visits'][r['p1'].index(lab)]) for r in results)}
                                   for i, lab in enumerate(labels)]
    return Advice(labels, sol.prob, sol.expected, opponent, status=status, worst=sol.worst)


def matrix(rows: list[list], idx: list[int], cols: list[int], fallback: np.ndarray) -> np.ndarray:
    """A worker matrix restricted to the world's kept options; missing cells (no run) take the payoff's value."""
    m = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)[np.ix_(idx, cols)]
    return np.where(np.isnan(m), fallback, m)


def diagnose(x: np.ndarray, worlds: list[World], extras: list[dict[str, np.ndarray]]) -> dict:
    """How far to trust the chosen mix's value: under each continuation policy (their spread is a disagreement of
    the rollouts, not a confidence interval) and the share of rollouts that reached a real end."""
    out: dict = {}
    names = sorted({k for e in extras for k in e if not k.startswith("_")})
    if names:
        by = {n: float(np.mean([x @ e.get(n, wd.payoff) @ wd.predicted for wd, e in zip(worlds, extras)]))
              for n in names}
        out["by_continuation"] = by
        out["continuation_spread"] = max(by.values()) - min(by.values())
    if extras and all("_resolved" in e for e in extras):
        out["resolved"] = float(np.mean([x @ e["_resolved"] @ wd.predicted for wd, e in zip(worlds, extras)]))
    return out
