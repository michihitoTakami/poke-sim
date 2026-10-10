from types import SimpleNamespace

import pytest
from poke_env.battle import Move, Pokemon

from pokesim.rl.features import best_damage, foe_boosts, move_features, self_boosts, threat


def mon(species: str, moves: list[str], hp: float = 1.0) -> Pokemon:
    p = Pokemon(gen=9, species=species)
    for m in moves:
        p._add_move(m)
    p._max_hp, p._current_hp = 100, int(100 * hp)
    return p


BATTLE = SimpleNamespace(weather={}, fields={}, opponent_side_conditions={}, side_conditions={})
NAMES = ["bp", "eff", "p_immune", "stab", "dmg", "ko_min", "ko_mean", "phys", "spec", "status", "prio",
         *[f"up_{s}" for s in "atk def spa spd spe".split()], *[f"down_{s}" for s in "atk def spa spd spe".split()],
         "setup_off", "setup_def", "heal", "cost", "sub", "protect", "pivot", "baton", "phaze", "inflict",
         "hazard", "screen", "volatile"]


def features(move: str, me: Pokemon, opp: Pokemon) -> dict[str, float]:
    ctx = {"my_best": best_damage(me, opp, True, {}), "threat": threat(opp, me, {})}
    return dict(zip(NAMES, move_features(Move(move, gen=9), me, opp, BATTLE, ctx)))


def test_boost_parsing():
    mimikyu = mon("mimikyu", [])
    assert self_boosts(Move("swordsdance", gen=9), mimikyu) == {"atk": 2}
    assert self_boosts(Move("dracometeor", gen=9), mimikyu) == {"spa": -2}
    assert foe_boosts(Move("charm", gen=9)) == {"atk": -2}
    assert self_boosts(Move("curse", gen=9), mimikyu) == {}  # Ghost-type Curse is not a boost


def test_swords_dance_doubles_damage():
    f = features("swordsdance", mon("mimikyu", ["swordsdance", "playrough", "shadowsneak"]),
                 mon("garchomp", ["earthquake"]))
    assert f["up_atk"] == 1.0 and f["setup_off"] == pytest.approx(1.0, abs=0.05)  # log2(2x damage)


def test_iron_defense_cuts_physical_threat():
    f = features("irondefense", mon("corviknight", ["irondefense", "bravebird"]), mon("garchomp", ["earthquake"]))
    assert f["setup_def"] > 0.9  # Garchomp is physical: +2 Def roughly halves its damage


def test_charm_cuts_physical_threat():
    f = features("charm", mon("primarina", ["charm", "moonblast"]), mon("garchomp", ["earthquake"]))
    assert f["down_atk"] == -1.0 and f["setup_def"] > 0.9


def test_recovery_and_substitute_flags():
    hurt = mon("corviknight", ["roost", "substitute", "uturn"], hp=0.4)
    opp = mon("garchomp", ["earthquake"])
    assert features("roost", hurt, opp)["heal"] == pytest.approx(0.3)  # 50% heal, 60% missing
    sub = features("substitute", hurt, opp)
    assert sub["sub"] == 1 and sub["cost"] == 0.25
    assert features("uturn", hurt, opp)["pivot"] == 1


def test_baton_pass_and_hazards():
    me, opp = mon("garchomp", ["batonpass", "stealthrock"]), mon("hippowdon", ["earthquake"])
    assert features("batonpass", me, opp)["baton"] == 1
    assert features("stealthrock", me, opp)["hazard"] == 1


def test_damage_and_ko_flags():
    f = features("moonblast", mon("primarina", ["moonblast"]), mon("garchomp", ["earthquake"], hp=0.5))
    assert f["ko_min"] == 1 and f["dmg"] == 2.0  # clipped at 2x remaining HP
