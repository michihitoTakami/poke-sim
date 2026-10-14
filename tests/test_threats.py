import numpy as np

from pokesim.rl.threats import expected_best


def test_expected_best():
    assert expected_best(np.array([0.5, 1.0]), np.array([1.0, 0.0])) == 0.5
    assert abs(expected_best(np.array([0.5, 1.0]), np.array([1.0, 0.5])) - 0.75) < 1e-9


def test_effects_block():
    from poke_env.battle import Effect, Pokemon

    from pokesim.rl.threats import N_OWN, N_OPP, SIZE, ThreatFeatures

    mon = Pokemon(gen=9, species="umbreon")
    mon._effects = {Effect.CURSE: 1, Effect.PERISH2: 1}
    own = ThreatFeatures.effects(mon, True)
    assert len(own) == N_OWN and own[0] == 1.0 and own[-3:] == [1.0, 2 / 3, 0.0]
    assert len(ThreatFeatures.effects(None, False)) == N_OPP and SIZE == 10 + N_OWN + N_OPP
