import numpy as np

from pokesim.rl.threats import expected_best


def test_expected_best():
    assert expected_best(np.array([0.5, 1.0]), np.array([1.0, 0.0])) == 0.5
    assert abs(expected_best(np.array([0.5, 1.0]), np.array([1.0, 0.5])) - 0.75) < 1e-9
