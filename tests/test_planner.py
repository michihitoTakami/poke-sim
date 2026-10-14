import numpy as np
import pytest

from pokesim.search.lookahead import aggregate, clean
from pokesim.search.planner import World, plan, predict, solve_common

LABELS = ["move:attack", "move:setup", "switch:wall"]


def test_common_mix_differs_from_the_average_of_per_world_solutions():
    # set A: attacking wins, set B: setting up wins, the switch is fine against both
    a = np.array([[1.0], [-1.0], [0.5]])
    b = np.array([[-1.0], [1.0], [0.5]])
    results = [{"p1": LABELS, "p2": ["move:x"], "payoff": a.tolist()},
               {"p1": LABELS, "p2": ["move:x"], "payoff": b.tolist()}]
    legacy = aggregate(results)
    assert legacy.prob[2] < 0.05  # each world's own best action, averaged: half attack, half set up
    common = plan(results, rho=0.0)
    assert common.prob[2] > 0.99  # one mix for both worlds: the switch
    assert common.value[2] == pytest.approx(0.5)


def test_worst_case_guard_is_taken_per_world():
    # the prediction says the opponent attacks; protecting is only good if it does not
    q = np.array([1.0, 0.0])
    world = World(np.array([[2.0, -3.0], [0.5, 0.4]]), ["move:hit", "move:setup"], 1.0, q)
    assert solve_common([world], rho=0.0).prob[0] > 0.99  # trusts the prediction
    assert solve_common([world], rho=1.0).prob[1] > 0.99  # guards against the other option


def test_rho_one_on_rock_paper_scissors_is_the_equilibrium():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    sol = solve_common([World(rps, ["r", "p", "s"], 1.0, np.array([1.0, 0, 0]))], rho=1.0)
    assert np.allclose(sol.prob, 1 / 3, atol=1e-6)


def test_failed_branches_are_not_scored_as_results():
    res = {"p1": ["a", "b"], "p2": ["x", "y", "z"], "payoff": [[1.0, None, None], [-1.0, 0.5, None]]}
    payoff, labels, failed = clean(res)
    assert labels == ["x", "y"]  # z failed with every option of ours: not possible in this world
    assert failed == 3
    assert payoff[0, 1] == pytest.approx(0.5)  # a failed cell takes its column's worst value for us


def test_a_world_where_everything_failed_is_left_out():
    ok = {"p1": ["a", "b"], "p2": ["x"], "payoff": [[1.0], [0.0]]}
    broken = {"p1": ["a", "b"], "p2": ["x"], "payoff": [[None], [None]]}
    adv = plan([ok, broken], rho=0.0)
    assert adv.status["worlds"] == 1 and adv.status["failed_worlds"] == 1
    assert adv.prob[0] > 0.99
    assert aggregate([ok, broken]).status["failed_worlds"] == 1


def test_cloned_prediction_and_repeat_enter_the_world_prediction():
    payoff = np.array([[0.0, 0.0], [0.0, 0.0]])
    labels = ["move:yawn", "move:earthquake"]
    q = predict(payoff, labels, {}, None, {"move:earthquake": 1.0}, 0.5)
    assert q[1] == pytest.approx(0.75)  # half the uniform equilibrium, half the cloned policy
    q = predict(payoff, labels, {"repeat": 0.3}, "move:yawn", None, 0.5)
    assert q[0] == pytest.approx(0.3 + 0.7 * 0.5)
    q = predict(payoff, ["move:fire", "move:fire:mega"], {}, None, {"move:fire": 1.0}, 1.0)
    assert np.allclose(q, 0.5)  # the cloned policy has no Mega variants: split over them


def test_prediction_changes_what_we_play():
    # we beat an attack with Protect and a set-up move with an attack: what we expect them to do decides
    payoff = [[1.0, -1.0], [-1.0, 1.0]]
    results = [{"p1": ["move:attack", "move:protect"], "p2": ["move:setup", "move:attack"], "payoff": payoff}]
    attack_expected = plan(results, rho=0.0, cloned={"move:attack": 1.0}, cloned_weight=1.0)
    setup_expected = plan(results, rho=0.0, cloned={"move:setup": 1.0}, cloned_weight=1.0)
    assert attack_expected.prob[1] > 0.99 and setup_expected.prob[0] > 0.99


@pytest.mark.parametrize("rho, expected", [(0.0, "risky"), (1.0, "safe")])
def test_switch_candidates_are_compared_using_the_same_risk_objective(rho, expected):
    from types import SimpleNamespace

    from pokesim.search.lookahead import Lookahead

    engine = Lookahead.__new__(Lookahead)
    battle = SimpleNamespace(available_switches=[SimpleNamespace(species=s) for s in ("risky", "safe")])

    def solve(battle, own, last, incoming):
        payoff = [[2.0, -3.0]] if incoming.species == "risky" else [[0.5, 0.4]]
        return plan([{"p1": ["move:hit"], "p2": ["move:hit", "move:setup"], "payoff": payoff}],
                    rho=rho, cloned={"move:hit": 1.0}, cloned_weight=1.0)

    advice = engine.advise_switch(battle, {}, solve=solve)
    assert advice.labels[np.argmax(advice.prob)] == f"switch:{expected}"


def test_solver_fallback_keeps_the_risk_objective(monkeypatch):
    from types import SimpleNamespace

    import pokesim.search.planner as planner

    monkeypatch.setattr(planner, "linprog", lambda *a, **kw: SimpleNamespace(success=False, message="failed"))
    world = World(np.array([[2.0, -3.0], [0.5, 0.4]]), ["hit", "setup"], predicted=np.array([1.0, 0.0]))
    sol = solve_common([world], rho=1.0)
    assert sol.prob.tolist() == [0.0, 1.0]
    assert sol.status["objective"] == pytest.approx(0.4)
