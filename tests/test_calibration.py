import numpy as np
import pytest

from pokesim.evaluation.calibration import (
    DECISION_POINT,
    Calibration,
    Calibrations,
    battle_estimates,
    continuation_context,
    fit,
    fit_platt,
)
from pokesim.evaluation.metrics import bootstrap, brier, group_weights, wilson


def test_overconfident_estimates_are_pulled_to_what_happened():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.6, 0.8, 200)  # "60-80%" ...
    y = rng.uniform(size=200) < 0.32  # ... with a 32% win rate
    cal = fit(raw, y, "showdown", "abc", min_battles=40)
    assert cal.enabled is False or cal.apply(0.7) < 0.45
    a, b = fit_platt(raw, y)
    assert 1 / (1 + np.exp(-(a * np.log(0.7 / 0.3) + b))) == pytest.approx(y.mean(), abs=0.08)


def test_informative_estimates_get_enabled():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.95, 120)
    y = rng.uniform(size=120) < raw
    cal = fit(raw, y, "showdown", "abc", min_battles=40)
    assert cal.enabled and cal.apply(0.9) > cal.apply(0.1)
    assert cal.scores["log_loss"] < cal.scores["log_loss_base"]


def test_too_few_battles_show_no_probability():
    cal = fit([0.7] * 10, [1, 0] * 5, "switch", "abc", min_battles=40)
    assert not cal.enabled and cal.apply(0.7) is None and "10 battles" in cal.reason
    assert Calibration("switch", None).apply(0.5) is None


def test_battles_not_turns_are_the_unit():
    groups = ["a"] * 9 + ["b"]
    w = group_weights(groups)
    assert w[:9].sum() == pytest.approx(1.0) and w[9] == pytest.approx(1.0)
    p, y = np.array([0.9] * 9 + [0.1]), np.array([1] * 9 + [0])
    assert brier(p, y, w) == pytest.approx(0.01)
    lo, hi = bootstrap(lambda idx: float(y[idx].mean()), groups, n=500)
    assert 0.0 <= lo <= hi <= 1.0
    assert wilson(33, 79) == pytest.approx((0.32, 0.53), abs=0.01)


@pytest.mark.parametrize("raw,y", [([np.nan] * 40, [1, 0] * 20), ([0.7] * 40, [1] * 40),
                                  ([1.1] * 40, [1, 0] * 20), ([0.7] * 40, [np.nan] * 40)])
def test_invalid_or_one_class_data_cannot_enable_a_calibration(raw, y):
    assert not fit(raw, y, "switch", "abc").enabled


def test_rare_outcomes_and_extreme_values_stay_finite():
    raw, y = np.array([0.0, 1.0] * 20), np.array([1] + [0] * 39)
    a, b = fit_platt(raw, y)
    assert np.isfinite([a, b]).all()
    cal = fit(raw, y, "switch", "abc")
    assert np.isfinite([cal.a, cal.b, cal.scores["log_loss"]]).all()
    extreme = Calibration("switch", "abc", a=10000, enabled=True)
    assert extreme.apply(0) == 0 and extreme.apply(1) == 1
    assert extreme.apply(np.nan) is None


def test_extraction_uses_only_the_validated_point_and_exact_identity():
    from types import SimpleNamespace

    ledger = [SimpleNamespace(battle_id=s, source="switch", result="win",
                             versions={"policy": {"sha": sha}, "calibration_context": context})
              for s, sha, context in [("ok", "A", "legacy"), ("late", "A", "legacy"),
                                      ("forced", "A", "legacy"), ("other", "B", "legacy"),
                                      ("new_mode", "A", "planner")]]
    decisions = [dict(battle_id=b.battle_id, turn=2 if b.battle_id == "late" else 1, win=0.7,
                      force=b.battle_id == "forced") for b in ledger]
    raw, y, ids = battle_estimates(ledger, decisions, {b.battle_id for b in ledger}, "switch", "A", "legacy")
    assert ids == ["ok"] and raw.tolist() == [0.7] and y.tolist() == [1.0]


def test_saved_calibration_requires_policy_source_context_and_point(tmp_path):
    saved = Calibrations(tmp_path / "missing.json")
    cal = Calibration("switch", "A", enabled=True, context="legacy", decision_point=DECISION_POINT)
    saved.items = [cal]
    assert saved.get("switch", "A", "legacy") is cal
    assert saved.get("switch", "B", "legacy") is None
    assert saved.get("showdown", "A", "legacy") is None
    assert saved.get("switch", "A", "planner") is None
    cal.decision_point = "unknown"
    assert saved.get("switch", "A", "legacy") is None


def test_continuation_identity_separates_sampling_and_planner_modes():
    args = ([], None, {}, "sparse", True)
    baseline = continuation_context(*args)
    assert baseline != continuation_context(*args, sample=True)
    assert baseline != continuation_context([], {"mode": "on"}, {}, "sparse", True)


def test_ab_policy_switch_selects_or_clears_its_own_calibration():
    from types import SimpleNamespace

    from scripts.ladder import AdvisorPlayer

    player = AdvisorPlayer.__new__(AdvisorPlayer)
    player.advisor = SimpleNamespace()
    player.policies = {"A": ("model-a", "features-a"), "B": ("model-b", "features-b")}
    player.calibrations = {"A": "calibration-a", "B": None}
    player.activate_policy("A")
    assert player.advisor.calibration == "calibration-a"
    player.activate_policy("B")
    assert player.advisor.model == "model-b" and player.advisor.calibration is None
    player.activate_policy("A")
    assert player.advisor.model == "model-a" and player.advisor.calibration == "calibration-a"
