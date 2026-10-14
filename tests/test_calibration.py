import numpy as np
import pytest

from pokesim.evaluation.calibration import Calibration, fit, fit_platt
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
