from poke_env.battle import Move, Pokemon

from pokesim.sim.effectiveness import effectiveness, type_vs_defender


def mon(species: str) -> Pokemon:
    return Pokemon(gen=9, species=species)


def test_levitate_rotom_is_immune_to_ground():
    eff = effectiveness(Move("earthpower", gen=9), mon("glimmora"), mon("rotomheat"))
    assert eff.multiplier == 0 and eff.p_immune == 1


def test_air_balloon_blocks_ground_until_popped():
    gholdengo = mon("gholdengo")
    assert effectiveness(Move("earthquake", gen=9), mon("garchomp"), gholdengo).multiplier == 2
    gholdengo.item = "airballoon"
    assert effectiveness(Move("earthquake", gen=9), mon("garchomp"), gholdengo).p_immune == 1


def test_revealed_ability_overrides_prior():
    gastrodon = mon("gastrodon")
    gastrodon.ability = "stormdrain"
    assert effectiveness(Move("surf", gen=9), mon("primarina"), gastrodon).multiplier == 0


def test_mold_breaker_ignores_levitate():
    attacker = mon("excadrill")
    attacker.ability = "moldbreaker"
    assert effectiveness(Move("earthquake", gen=9), attacker, mon("rotomheat")).multiplier == 4  # Electric/Fire


def test_type_vs_defender_levitate():
    assert type_vs_defender("ground", mon("rotomwash")) == 0
