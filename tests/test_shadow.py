import json

import numpy as np
import pytest

from pokesim.evaluation.shadow import (
    Decision,
    Verdict,
    decisions,
    disagreement,
    judge,
    load_verdicts,
    records,
    referee_candidates,
    referee_summary,
    score_worlds,
    summary,
)


def advice(turn, chosen, shadow, force=False, secs=3.0):
    return {"turn": turn, "force": force, "chosen": chosen, "shadow": shadow, "secs": secs, "status": {}}


def record(result="win", advice_list=()):
    return {"result": result, "policy": "ppo_a.zip", "versions": {"code": {"commit": "abc"}},
            "advice": list(advice_list)}


def test_disagreement_kinds():
    assert disagreement("move:dragondance", "move:dragondance") == "agree"
    assert disagreement("move:doubleedge", "move:doubleedge:mega") == "mega timing"
    assert disagreement("move:doubleedge:mega", "move:dragondance:mega") == "attack -> status"
    assert disagreement("move:irondefense", "switch:gholdengo") == "status -> switch"
    assert disagreement("switch:umbreon", "move:earthquake") == "switch -> attack"


def test_decisions_take_the_planner_top_and_its_own_gap():
    rec = record(advice_list=[
        advice(1, "move:doubleedge", [["move:dragondance", 0.8, 0.3], ["move:doubleedge", 0.2, 0.1]]),
        advice(2, "move:doubleedge", [["move:doubleedge", 1.0, 0.2]]),
        advice(3, "move:roost", [["move:dragondance", 1.0, 0.2]]),  # played action not in the saved top 5
        advice(4, "switch:umbreon", [], force=True),  # no shadow
    ])
    ds = decisions([("b1", rec)])
    assert [d.planned for d in ds] == ["move:dragondance", "move:doubleedge", "move:dragondance", None]
    assert ds[0].planner_gap == pytest.approx(0.2)
    assert ds[1].kind == "agree" and ds[2].planner_gap is None and ds[3].kind == "no shadow"
    s = summary(ds)
    assert s["with_shadow"] == 3 and s["disagree (normal)"].startswith("2/3")


def test_records_need_shadow_advice_and_respect_the_time_range(tmp_path):
    shadowed = record(advice_list=[advice(1, "move:a", [["move:a", 1.0, 0.0]])])
    for name, rec in (("20261015-010000-x", shadowed), ("20261015-020000-y", record(advice_list=[])),
                      ("20261015-030000-z", shadowed)):
        (tmp_path / f"{name}.json").write_text(json.dumps(rec))
    assert [s for s, _ in records(root=tmp_path)] == ["20261015-010000-x", "20261015-030000-z"]
    assert [s for s, _ in records("20261015-020000", root=tmp_path)] == ["20261015-030000-z"]
    assert [s for s, _ in records(until="20261015-020000", root=tmp_path)] == ["20261015-010000-x"]


def test_referee_candidates_are_normal_disagreements_once_per_turn():
    d = dict(result="win", policy=None, commit=None, secs=1.0)
    ds = [Decision("b", 1, False, played="move:a", planned="move:b", kind="x", **d),
          Decision("b", 1, False, played="move:a", planned="move:c", kind="x", **d),  # same turn again
          Decision("b", 2, False, played="move:a", planned="move:a", kind="agree", **d),
          Decision("b", 2, True, played="switch:a", planned="switch:b", kind="x", **d),
          Decision("b", 3, False, played="move:a", planned=None, kind="no shadow", **d)]
    assert [(c.turn, c.planned) for c in referee_candidates(ds)] == [(1, "move:b")]


def test_judge_needs_a_margin_and_agreeing_continuations():
    assert judge(0.2, 0.05, {"greedy": 0.1, "tactical": 0.3}) == "planner"
    assert judge(-0.2, 0.05, {"greedy": -0.1, "tactical": -0.3}) == "played"
    assert judge(0.2, 0.15, {"greedy": 0.1, "tactical": 0.3}) == "unclear"  # within two standard errors
    assert judge(0.2, 0.05, {"greedy": -0.1, "tactical": 0.5}) == "unclear"  # the continuations disagree


def test_score_worlds_compares_both_actions_against_the_prediction():
    labels = ["move:hit", "move:setup", "switch:wall"]
    # the opponent only has one option, so the prediction is certain; setup is better by 0.5 in both worlds
    worlds = [{"p1": labels, "p2": ["move:x"], "payoff": [[0.0], [0.5], [0.1]],
               "byContinuation": {"greedy": [[0.0], [0.4], [0.1]], "tactical": [[0.0], [0.6], [0.1]]},
               "resolved": [[0.0], [1.0], [0.0]]},
              {"p1": labels, "p2": ["move:x"], "payoff": [[0.2], [0.7], [0.1]]}]
    s = score_worlds(worlds, "move:hit", "move:setup", {}, None, None, 0.5)
    assert s["delta"] == pytest.approx(0.5) and s["se"] == pytest.approx(0.0)
    assert s["value"] == {"move:hit": pytest.approx(0.1), "move:setup": pytest.approx(0.6)}
    # a world without byContinuation counts its payoff for every continuation
    assert s["by_continuation"]["greedy"] == pytest.approx(0.45)
    assert s["by_continuation"]["tactical"] == pytest.approx(0.55)
    assert s["worlds"] == 2 and s["resolved"] == pytest.approx(0.5)


def test_score_worlds_skips_failed_worlds():
    labels = ["move:a", "move:b"]
    ok = {"p1": labels, "p2": ["move:x", "move:y"], "payoff": [[1.0, -1.0], [0.0, 0.0]]}
    failed = {"p1": labels, "p2": ["move:x"], "payoff": [[None], [None]]}
    s = score_worlds([failed, ok], "move:a", "move:b", {}, None, None, 0.5)
    assert s["worlds"] == 1
    assert s["worst"] == {"move:a": -1.0, "move:b": 0.0}
    with pytest.raises(ValueError):
        score_worlds([failed], "move:a", "move:b", {}, None, None, 0.5)


def test_referee_summary_and_cache_round_trip(tmp_path):
    vs = [Verdict("b1", 1, "attack -> status", "move:a", "move:b", delta=0.3, se=0.05, worst_delta=0.1,
                  by_continuation={"greedy": 0.2, "tactical": 0.4}, verdict="planner"),
          Verdict("b2", 4, "status -> status", "move:c", "move:d", delta=-0.1, se=0.2, worst_delta=0.0,
                  by_continuation={"greedy": -0.1, "tactical": -0.1}, verdict="unclear"),
          Verdict("b3", 2, "status -> status", "move:c", "move:d", status="skipped: no active Pokemon")]
    path = tmp_path / "cache.jsonl"
    from dataclasses import asdict

    path.write_text("".join(json.dumps(asdict(v)) + "\n" for v in vs))
    loaded = load_verdicts(path)
    assert set(loaded) == {("b1", 1), ("b2", 4), ("b3", 2)}
    s = referee_summary(list(loaded.values()))
    assert s["scored"] == 2 and s["skipped / failed"] == {"skipped": 1}
    assert s["mean delta (planner - played)"] == pytest.approx(0.1)
    assert s["verdicts"] == {"planner": 1, "unclear": 1}
    assert np.isclose(s["mean delta by continuation"]["tactical"], 0.15)
