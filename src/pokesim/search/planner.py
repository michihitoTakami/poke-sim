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
TREE_TIE_MARGIN = .02  # a tiny difference between sampled utilities is not a proven winning-probability gain


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


def solve_common(worlds: list[World], rho: float = RHO, secondary: np.ndarray | None = None,
                 tie_margin: float = 0.) -> Solution:
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
        if secondary is not None and np.ptp(secondary) > 1e-12:
            # Exact matrices preserve the primary objective. Sampled trees may supply a bounded tolerance
            # for noisy utilities; only inside that band do we prefer earlier terminal wins.
            tie = linprog(-np.r_[secondary, np.zeros(g)],
                          A_ub=np.vstack([rows, c]), b_ub=np.r_[np.zeros(len(rows)), res.fun + max(1e-9, tie_margin)],
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
            cloned: dict[str, float] | None, cloned_weight: float, prior: dict[str, float] | None = None) -> np.ndarray:
    """The opponent's mix in one world, fixed before planning: the search's prediction (equilibrium / softmax best
    response to our equilibrium, see opponent_mix), a share for repeating its last move, then a share for the
    cloned human policy (its labels have no Mega variants: split over them). prior: a learned opponent policy's
    probabilities over this world's labels (the worker's p2Prior); when given it takes the cloned policy's share."""
    x, y = solve_zero_sum(payoff)
    q = opponent_mix(payoff, x, y, opponent_model)
    repeat = opponent_model.get("repeat", 0.0)
    if repeat > 0 and last_move in labels:
        q = (1 - repeat) * q
        q[labels.index(last_move)] += repeat
    if prior:
        learned = np.array([prior.get(lab, 0.0) for lab in labels])
        if learned.sum() > 0:
            q = (1 - cloned_weight) * q + cloned_weight * learned / learned.sum()
    elif cloned:
        bases = [base_label(lab) for lab in labels]
        bc = np.array([cloned.get(b, 0.0) / bases.count(b) for b in bases])
        if bc.sum() > 0:
            q = (1 - cloned_weight) * q + cloned_weight * bc / bc.sum()
    return q / q.sum()


def plan(results: list[dict], opponent_model: dict | None = None, last_move: str | None = None,
         rho: float = RHO, cloned: dict[str, float] | None = None, cloned_weight: float = 0.5,
         pure: bool = False, classes: dict[str, str] | None = None, rho_by_class: dict[str, float] | None = None,
         near: float | None = None, near_tau: float = .02) -> Advice:
    """Advice from the lookahead's sampled worlds with one common mix (see the module docstring).
    classes / rho_by_class: a class per label (e.g. 'setup') and the worst-case share to use for it in the pure
    ranking (a set-up move is judged with more weight on the opponent's best reply). near: with pure, the
    probability is a softmax (temperature near_tau) over the actions within `near` of the best objective instead
    of one-hot, so a caller that samples does not repeat the same move against the same opponent.
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
    error_terms = []
    extras: list[dict[str, np.ndarray]] = []  # per world: payoff per continuation policy, share resolved
    for res in results:
        try:
            payoff, opp, failed = clean(res)
        except ValueError:
            failed_worlds += 1
            continue
        idx = [res["p1"].index(lab) for lab in labels]  # our options are the same in every world
        payoff = payoff[idx]
        prior = dict(zip(res["p2"], res["p2Prior"], strict=True)) if res.get("p2Prior") else None
        q = predict(payoff, opp, opponent_model or {}, last_move, cloned, cloned_weight, prior)
        worlds.append(World(payoff, opp, 1.0, q))
        cols = [res["p2"].index(lab) for lab in opp]
        if tree:
            # Secondary progress across legal replies, rather than the fitted opponent mix: a prediction
            # concentrated on Protect must not erase the immediate finishing opportunities of an attack.
            speed_terms.append(matrix(res['winSpeed'], idx, cols, np.zeros_like(payoff)).mean(axis=1))
            if 'uncertainty' in res:
                error_terms.append(matrix(res['uncertainty'], idx, cols, np.zeros_like(payoff)).max(axis=1))
        extras.append({name: matrix(m, idx, cols, payoff) for name, m in res.get("byContinuation", {}).items()}
                      | ({"_resolved": matrix(res["resolved"], idx, cols, np.zeros_like(payoff))}
                         if "resolved" in res else {}))
        failed_cells += failed
    if not worlds:
        raise ValueError("every sampled world failed")
    secondary = np.mean(speed_terms, axis=0) if speed_terms else None
    errors = np.mean(error_terms, axis=0) if error_terms else np.zeros(len(labels))
    tie_margin = min(.1, max(TREE_TIE_MARGIN, 2 * float(errors.max()))) if tree else 0.
    sol = solve_common(worlds, rho, secondary, tie_margin=tie_margin)
    if pure:
        # The caller plays one top move. Rank actual pure-action utilities, not the largest probability of a
        # mixed equilibrium whose protection requires randomizing between ALL of its actions.
        rhos = np.array([(rho_by_class or {}).get((classes or {}).get(lab, ''), rho) for lab in labels])
        objective = (1 - rhos) * sol.expected + rhos * sol.worst
        candidates = np.flatnonzero(objective >= objective.max() - max(1e-9, tie_margin))
        chosen = int(candidates[np.argmax(secondary[candidates])]) if secondary is not None else int(candidates[0])
        sol.prob = np.eye(len(labels))[chosen]
        sol.status.update(solver='pure-robust', objective=float(objective[chosen]))
        if near is not None and near > 0:
            close = np.flatnonzero(objective >= objective[chosen] - near)
            weights = np.exp((objective[close] - objective[chosen]) / max(1e-6, near_tau))
            sol.prob = np.zeros(len(labels))
            sol.prob[close] = weights / weights.sum()
            sol.status.update(solver='pure-near', near=near, near_tau=near_tau, near_actions=int(len(close)))
        if rho_by_class:
            sol.status['rho_by_class'] = {lab: float(r) for lab, r in zip(labels, rhos) if r != rho}
    opponent: dict[str, float] = {}
    for wd in worlds:
        for lab, p in zip(wd.opponent, wd.predicted):
            opponent[lab] = opponent.get(lab, 0.0) + float(p) / len(worlds)
    status = dict(sol.status, rho=rho, worlds=len(worlds), failed_worlds=failed_worlds, failed_cells=failed_cells)
    status.update(diagnose(sol.prob, worlds, extras))
    if tree:
        status['tie_margin'] = tie_margin
        status['terminal_search'] = tree
        status['root_evidence'] = [{'label': lab, 'expected': float(sol.expected[i]),
                                    'worst': float(sol.worst[i]),
                                    'utility': float((1 - rho) * sol.expected[i] + rho * sol.worst[i]),
                                    'sampling_error': float(errors[i]),
                                    'trials': sum(sum(r['visits'][r['p1'].index(lab)]) for r in results)}
                                   for i, lab in enumerate(labels)]
        status['score_kind'] = tree.get('value_kind', 'uncalibrated-search-utility')
        status['calibrated_win_probability'] = False
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
