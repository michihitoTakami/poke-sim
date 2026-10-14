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
    assert len(ThreatFeatures.effects(None, False)) == N_OPP and SIZE == 10 + N_OWN + N_OPP + 6 + 4 + 15


def test_good_as_gold_blocks_status_counters():
    from poke_env.battle import Pokemon

    from pokesim.rl.threats import ThreatFeatures

    gholdengo, scolipede = Pokemon(gen=9, species="gholdengo"), Pokemon(gen=9, species="scolipede")
    gholdengo._ability, scolipede._ability = "goodasgold", "speedboost"
    foe = Pokemon(gen=9, species="hippowdon")
    priors = {"hippowdon": {"counter": {"whirlwind": 0.6, "yawn": 0.5, "dragontail": 0.0}}}
    f = ThreatFeatures(priors)
    on_scolipede, on_gholdengo = f.counters(foe, scolipede), f.counters(foe, gholdengo)
    assert abs(on_scolipede[1] - 0.6) < 1e-6 and abs(on_scolipede[4] - 0.5) < 1e-6
    assert on_gholdengo[1] == 0.0 and on_gholdengo[4] == 0.0 and on_gholdengo[-1] == 1.0


def test_disguise_shield():
    from poke_env.battle import Pokemon

    from pokesim.rl.threats import ThreatFeatures

    f = ThreatFeatures({})
    assert f.shields(Pokemon(gen=9, species="mimikyu"))[0] == 1.0
    assert f.shields(Pokemon(gen=9, species="mimikyubusted"))[0] == 0.0


def test_scrappy_hits_ghosts():
    from poke_env.battle import Move, Pokemon

    from pokesim.sim.effectiveness import effectiveness

    gholdengo = Pokemon(gen=9, species="gholdengo")
    mega = Pokemon(gen=9, species="lopunnymega")  # Scrappy
    assert effectiveness(Move("closecombat", gen=9), mega, gholdengo).multiplier == 2.0
    assert effectiveness(Move("closecombat", gen=9), Pokemon(gen=9, species="lopunny"), gholdengo).multiplier == 0.0


def test_aerilate_changes_the_type():
    from poke_env.battle import Move, Pokemon

    from pokesim.sim.effectiveness import effectiveness

    mega = Pokemon(gen=9, species="salamencemega")  # Aerilate: Double-Edge is Flying
    assert effectiveness(Move("doubleedge", gen=9), mega, Pokemon(gen=9, species="skeledirge")).multiplier == 1.0
