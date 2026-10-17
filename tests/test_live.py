import json

import pytest

from pokesim.config import ROOT
from pokesim.live.events import EventTranslator
from pokesim.live.selection import PickModel, scale_to
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import load_my_team
from pokesim.search.lookahead import mega_evolved

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


def test_mega_floette_substitute_and_yawn_reach_the_battle(session):
    t = EventTranslator(session)
    t.apply([{"type": "switch", "side": "me", "species": "Floette-Eternal"},
             {"type": "switch", "side": "opp", "species": "Hippowdon"}, {"type": "turn"},
             {"type": "mega", "side": "me"}, {"type": "move", "side": "me", "move": "substitute"},
             {"type": "start", "side": "me", "effect": "Substitute"},
             {"type": "move", "side": "opp", "move": "yawn"}, {"type": "start", "side": "me", "effect": "move: Yawn"}])
    me = session.decision().active_pokemon  # our request is rebuilt as at a decision
    assert mega_evolved(me) and me.ability == "fairyaura" and me.base_stats["spa"] == 155
    assert not mega_evolved(session.battle.opponent_active_pokemon)
    assert {e.name for e in me.effects} >= {"SUBSTITUTE", "YAWN"}
    req = session.request()
    assert "canMegaEvo" not in req["active"][0]


def test_last_move_and_repeat_share(session):
    from pokesim.search.lookahead import aggregate

    t = EventTranslator(session)
    t.apply([{"type": "switch", "side": "me", "species": "Scolipede"},
             {"type": "switch", "side": "opp", "species": "Hippowdon"}, {"type": "turn"},
             {"type": "move", "side": "opp", "move": "yawn"}, {"type": "turn"}])
    assert session.last_move() == "move:yawn"
    results = [{"p1": ["move:a", "move:b"], "p2": ["move:yawn", "move:earthquake"],
                "payoff": [[0.1, -0.5], [0.2, -0.4]]}]
    plain = aggregate(results, {}).opponent
    rep = aggregate(results, {"repeat": 0.3}, "move:yawn").opponent
    assert rep["move:yawn"] == pytest.approx(0.7 * plain["move:yawn"] + 0.3)
    t.apply([{"type": "switch", "side": "opp", "species": "Garchomp"}])
    assert session.last_move() is None


def test_choice_lock_ends_when_it_switches_out():
    sets = load_my_team(TEAM, ["Scolipede", "Floette-Eternal", "Gholdengo"])
    sets = [dict(s, item="Choice Specs") if s["species"] == "Gholdengo" else s for s in sets]
    s = BattleSession(sets)
    s.preview(["Garchomp", "Primarina", "Ninetales-Alola", "Hippowdon", "Gyarados", "Kingambit"])
    s.select([0, 1, 2])
    t = EventTranslator(s)
    t.apply([{"type": "switch", "side": "me", "species": "Gholdengo"},
             {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
             {"type": "move", "side": "me", "move": "shadowball"}, {"type": "turn"}])
    assert [m.id for m in s.decision().available_moves] == ["shadowball"]
    t.apply([{"type": "switch", "side": "me", "species": "Scolipede"}, {"type": "turn"},
             {"type": "switch", "side": "me", "species": "Gholdengo"}, {"type": "turn"}])
    assert len(s.decision().available_moves) == 4  # a new stint: no lock until it moves again


def test_a_failed_search_falls_back_instead_of_losing_the_turn(session):
    from pokesim.live.advisor import Advisor

    class Broken:
        opponent_model: dict = {}

        def query(self, *args, **kwargs):
            raise RuntimeError("node worker died")

    t = EventTranslator(session)
    t.apply([{"type": "switch", "side": "me", "species": "Scolipede"},
             {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"}])
    advisor = Advisor(session.sets, policy=None, lookahead=False, opponent_policy=None, planner={"mode": "on"})
    advisor.engine = Broken()
    sug = advisor.advise(session.decision())
    assert "node worker died" in sug.status["fallback"]
    assert sug.actions == []  # no policy loaded either: the caller picks a legal move


def test_shadow_failure_preserves_the_legacy_recommendation(session, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from pokesim.live.advisor import Advisor
    from pokesim.search.lookahead import Advice

    EventTranslator(session).apply([{"type": "switch", "side": "me", "species": "Scolipede"},
                                   {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"}])
    advisor = Advisor(session.sets, policy=None, lookahead=False, opponent_policy=None, planner={"mode": "shadow"})
    advisor.engine = SimpleNamespace()

    def search(*args, planned=False, **kwargs):
        if planned:
            raise RuntimeError("shadow failed")
        return Advice(["move:protect"], np.array([1.0]), np.array([1.0]))

    monkeypatch.setattr(advisor, "_search", search)
    sug = advisor.advise(session.decision())
    assert sug.actions == [("move:protect", 1.0)]
    assert "shadow failed" in sug.status["shadow"]["error"]
    assert "fallback" not in sug.status


def test_planner_does_not_play_argmax_of_a_mix_in_fixed_depth_mode(session):
    from types import SimpleNamespace

    from pokesim.live.advisor import Advisor

    EventTranslator(session).apply([{'type': 'switch', 'side': 'me', 'species': 'Scolipede'},
                                   {'type': 'switch', 'side': 'opp', 'species': 'Garchomp'}, {'type': 'turn'}])
    row = {'p1': ['move:batonpass', 'move:irondefense'], 'p2': ['move:earthquake', 'move:swordsdance'],
           'payoff': [[1, -.9], [-1, 0]]}
    advisor = Advisor(session.sets, lookahead=False, opponent_policy=None,
                      planner={'mode': 'on', 'rho': 1, 'search': {'plans': None}})
    advisor.engine = SimpleNamespace(opponent_model={}, query=lambda *a, **k: [row])
    try:
        suggestion = advisor.advise(session.decision())
        assert suggestion.actions[0] == ('move:batonpass', 1.)
        assert suggestion.status['search']['solver'] == 'pure-robust'
        advisor.sample = True
        mixed = advisor.advise(session.decision())
        assert mixed.actions[0][0] == 'move:irondefense'
        assert 0 < mixed.actions[0][1] < 1
    finally:
        advisor.engine = None


def test_switch_request_keeps_the_outgoing_state_for_the_simulator(session):
    from types import SimpleNamespace

    from pokesim.rl.myteam import to_id
    from pokesim.search.lookahead import Lookahead

    EventTranslator(session).apply([{"type": "switch", "side": "me", "species": "Scolipede"},
                                   {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
                                   {"type": "boost", "side": "me", "stat": "def", "amount": 2},
                                   {"type": "move", "side": "me", "move": "batonpass"}])
    battle = session.decision(force_switch=True)
    incoming = battle.available_switches[0]
    engine = Lookahead.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda sid: sid)
    own = engine.own_side(battle, {to_id(s["species"]): s for s in session.sets}, incoming)
    assert own["team"][0]["species"] == "Scolipede"
    assert to_id(own["team"][own["incoming"]]["species"]) == incoming.species
    assert own["mons"][0]["boosts"]["def"] == 2
    assert own["switchMove"] == "batonpass"


def test_first_decision_calibration_is_not_applied_to_late_or_forced_positions(session, monkeypatch):
    from pokesim.evaluation.calibration import DECISION_POINT, Calibration
    from pokesim.live.advisor import Advisor

    EventTranslator(session).apply([{"type": "switch", "side": "me", "species": "Scolipede"},
                                   {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"}])
    advisor = Advisor(session.sets, policy=None, lookahead=False, opponent_policy=None)
    advisor.calibration = Calibration("switch", "abc", a=1, enabled=True, decision_point=DECISION_POINT)
    monkeypatch.setattr(advisor, "_policy", lambda b: ({"move:protect": 1.0}, 0.7))
    assert advisor.advise(session.decision()).win_prob == pytest.approx(0.7)
    assert advisor.advise(session.decision(force_switch=True)).win_prob is None
    EventTranslator(session).apply([{"type": "turn"}])
    assert advisor.advise(session.decision()).win_prob is None


def test_double_faint_forced_switch_is_searched(session, monkeypatch):
    """Both actives fainted the same turn: the opponent's active is fainted while we pick a replacement. The
    simulator rebuilds that as two forced switches, so the search runs instead of leaving no advice (which, with
    --policy none, became a random switch and a quality-gate stop of ladder collection)."""
    from types import SimpleNamespace

    import numpy as np

    from pokesim.live.advisor import Advisor
    from pokesim.search.lookahead import Advice

    EventTranslator(session).apply([{"type": "switch", "side": "me", "species": "Scolipede"},
                                   {"type": "switch", "side": "opp", "species": "Garchomp"}, {"type": "turn"},
                                   {"type": "faint", "side": "opp"}, {"type": "faint", "side": "me"}])
    battle = session.decision(force_switch=True)
    assert battle.force_switch and battle.opponent_active_pokemon.fainted
    advisor = Advisor(session.sets, policy=None, lookahead=False, opponent_policy=None, planner={"mode": "on"})
    advisor.engine = SimpleNamespace()
    searched = []

    def search(b, *args, **kwargs):
        searched.append(b.force_switch)
        return Advice(["switch:gholdengo", "switch:floetteeternal"], np.array([1.0, 0.0]), np.array([0.6, 0.4]))

    monkeypatch.setattr(advisor, "_search", search)
    sug = advisor.advise(battle)
    assert searched == [True]
    assert sug.actions[0] == ("switch:gholdengo", 1.0)
    assert "fallback" not in sug.status
