import json

from pokesim.config import ROOT
from pokesim.live.events import EventTranslator
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import load_my_team
from pokesim.team.roles import hits, roles, win_path

TEAM = ROOT / "data/my_teams/baton.json"


def test_roles_from_the_team_file():
    r = roles(json.loads(TEAM.read_text()))
    assert r["scolipede"] == {"passer", "setup", "protect"}  # no damaging move
    assert {"attacker", "setup", "recovery"} <= r["gholdengo"]
    assert "passer" in r["umbreon"] and "attacker" in r["umbreon"]


def battle_with(lineup, events):
    s = BattleSession(load_my_team(TEAM, lineup))
    s.preview(["Garchomp", "Primarina", "Ninetales-Alola", "Hippowdon", "Gyarados", "Kingambit"])
    s.select([0, 1, 2])
    EventTranslator(s).apply(events)
    return s.decision()


def test_scolipede_alone_has_no_damage_source():
    b = battle_with(["Scolipede", "Floette-Eternal", "Gholdengo"],
                    [{"type": "switch", "side": "me", "species": "Floette-Eternal"},
                     {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
                     {"type": "faint", "side": "me"}, {"type": "switch", "side": "me", "species": "Gholdengo"},
                     {"type": "faint", "side": "me"}, {"type": "switch", "side": "me", "species": "Scolipede"},
                     {"type": "turn"}])
    path = win_path(b)
    assert path.no_source and path.sources == []
    assert path.blocked == {"garchomp": ["scolipede"]}


def test_one_attacker_left_is_flagged():
    b = battle_with(["Scolipede", "Floette-Eternal", "Gholdengo"],
                    [{"type": "switch", "side": "me", "species": "Floette-Eternal"},
                     {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
                     {"type": "faint", "side": "me"}, {"type": "switch", "side": "me", "species": "Gholdengo"},
                     {"type": "turn"}])
    path = win_path(b)
    assert path.last_source and path.sources == ["gholdengo"]


def test_hits_uses_attackers_type_changes_and_immunity_bypass():
    from poke_env.battle import Pokemon

    attacker = Pokemon(species="salamencemega", gen=9)
    attacker.temporary_ability = "aerilate"
    ghost = Pokemon(species="gengar", gen=9)
    ghost._ability = "cursedbody"
    assert hits("doubleedge", ghost, attacker)
    attacker.temporary_ability = "intimidate"
    assert not hits("doubleedge", ghost, attacker)
    attacker.temporary_ability = "scrappy"
    assert hits("doubleedge", ghost, attacker)
    assert hits("doubleedge", Pokemon(species="dusclops", gen=9), attacker)
    ground = Pokemon(species="garchomp", gen=9)
    ground._ability = "roughskin"
    ghost._ability = "levitate"
    assert not hits("earthquake", ghost, ground)
    ground._ability = "moldbreaker"
    assert hits("earthquake", ghost, ground)


def test_trapping_does_not_hide_selected_bench_attackers():
    from poke_env.battle import Pokemon

    b = battle_with(["Scolipede", "Floette-Eternal", "Gholdengo"],
                    [{"type": "switch", "side": "me", "species": "Scolipede"},
                     {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"}])
    # Preview objects may still be present; the request identifies the actual three, regardless of switch legality.
    b.team["p1: Umbreon"] = Pokemon(species="umbreon", gen=9)
    b._available_switches = []
    b._trapped = True
    path = win_path(b)
    assert not path.no_source
    assert set(path.sources) == {"floetteeternal", "gholdengo"}
