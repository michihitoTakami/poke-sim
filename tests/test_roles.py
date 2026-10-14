import json

from pokesim.config import ROOT
from pokesim.live.events import EventTranslator
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import load_my_team
from pokesim.team.roles import roles, win_path

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
