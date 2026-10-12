import json

import pytest

from pokesim.config import ROOT
from pokesim.live.events import EventTranslator
from pokesim.live.selection import PickModel, scale_to
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import load_my_team

TEAM = ROOT / "data/my_teams/baton.json"


@pytest.fixture
def session():
    sets = load_my_team(TEAM, ["Scolipede", "Floette-Eternal", "Gholdengo"])
    s = BattleSession(sets)
    s.preview(["Garchomp", "Primarina", "Ninetales-Alola", "Hippowdon", "Gyarados", "Kingambit"])
    s.select([0, 1, 2])
    return s


def test_events_rebuild_state(session):
    t = EventTranslator(session)
    t.apply([{"type": "switch", "side": "me", "species": "Scolipede"},
             {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
             {"type": "move", "side": "me", "move": "irondefense"}, {"type": "boost", "side": "me", "stat": "def",
                                                                      "amount": 2},
             {"type": "move", "side": "opp", "move": "earthquake"}, {"type": "hp", "side": "me", "hp": 140},
             {"type": "turn"}])
    b = session.decision()
    assert b.active_pokemon.species == "scolipede"
    assert b.active_pokemon.boosts["def"] == 2
    assert round(b.active_pokemon.current_hp_fraction, 2) == round(140 / 165, 2)
    assert "earthquake" in b.opponent_active_pokemon.moves
    assert sorted(m.id for m in b.available_moves) == ["batonpass", "irondefense", "protect", "substitute"]
    assert sorted(m.species for m in b.available_switches) == ["floetteeternal", "gholdengo"]
    assert b.turn == 2
    t.apply([{"type": "move", "side": "opp", "move": "earthquake"}, {"type": "faint", "side": "me"}])
    b = session.decision(force_switch=True)
    assert b.force_switch and not b.available_moves


def test_opponent_switch_and_damage(session):
    t = EventTranslator(session)
    t.apply([{"type": "switch", "side": "me", "species": "Scolipede"},
             {"type": "switch", "side": "opp", "species": "Ninetales-Alola"}, {"type": "turn"},
             {"type": "hp", "side": "opp", "hp": 37}, {"type": "status", "side": "opp", "status": "brn"}])
    opp = session.decision().opponent_active_pokemon
    assert opp.species == "ninetalesalola" and opp.current_hp_fraction == pytest.approx(0.37)
    assert opp.status.name == "BRN"


def test_pick_model_scaling():
    p = scale_to([0.9, 0.9, 0.5, 0.2, 0.2, 0.1], 3.0)
    assert p.sum() == pytest.approx(3.0) and p.max() <= 1.0
    picks = PickModel().predict(["Garchomp", "Primarina", "Ninetales-Alola", "Hippowdon", "Gyarados", "Kingambit"])
    assert picks.brought.sum() == pytest.approx(3.0) and picks.lead.sum() == pytest.approx(1.0)
    json.dumps(picks.brought.tolist())
