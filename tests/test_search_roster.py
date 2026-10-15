"""The six preview members must never become extra teammates in a 3v3 search."""
import json
from types import SimpleNamespace

import pytest
from test_worker_transitions import CHOMP, worker

from pokesim.config import ROOT
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import Lookahead, to_id


@pytest.fixture
def roster():
    sets = json.loads((ROOT / "data/my_teams/baton.json").read_text())
    session = BattleSession(sets)
    session.preview(["Garchomp", "Gyarados", "Kingambit"])
    session.select([1, 0, 3])  # Gholdengo, Scolipede, Floette; the other three stay in battle.team.
    session.apply(["|switch|p1a: Scolipede|Scolipede, L50, F|165/165",
                   "|switch|p2a: Garchomp|Garchomp, L50, M|100/100", "|turn|1"])
    engine = Lookahead.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=to_id)
    return session, engine, {to_id(s["species"]): s for s in sets}


@pytest.mark.parametrize("reverse_memory", [False, True])
def test_last_survivor_cannot_pass_to_an_unselected_teammate(roster, reverse_memory):
    session, engine, sets = roster
    session.apply(["|faint|p1: Gholdengo", "|faint|p1: Floette"])
    battle = session.decision()
    assert len(battle.team) == 6 and not battle.available_switches
    if reverse_memory:
        battle.team = dict(reversed(list(battle.team.items())))
    ours = engine.own_side(battle, sets)
    assert [s["species"] for s in ours["team"]] == ["Scolipede", "Gholdengo", "Floette-Eternal"]
    assert [m["hp"] for m in ours["mons"]] == [1, 0, 0]
    restored = worker("""
const b = setUp(sample,format,1);
step(b,'move 4','move 1'); // Baton Pass fails: there is no living teammate to receive it.
console.log(JSON.stringify({left:b.p1.pokemonLeft, active:b.p1.active[0].species.id,
  members:b.p1.pokemon.map(m=>m.species.baseSpecies),
  failed:b.log.some(line=>line.startsWith('|-fail|p1a: Scolipede')),
  switches:b.p1.pokemon.filter(m=>!m.isActive&&!m.fainted).length}));
""", {"p1": ours, "p2": {"team": [CHOMP], "mons": [{"hp": 1}]}})
    assert restored == {"left": 1, "active": "scolipede", "members": ["Scolipede", "Gholdengo", "Floette"],
                        "failed": True, "switches": 0}


def test_trapped_active_keeps_its_selected_living_bench(roster):
    session, engine, sets = roster
    request = session.request()
    request["active"][0]["trapped"] = True
    session.battle.parse_request(request)
    battle = session.battle
    assert battle.trapped and not battle.available_switches
    ours = engine.own_side(battle, sets)
    assert [s["species"] for s in ours["team"]] == ["Scolipede", "Gholdengo", "Floette-Eternal"]
    assert [m["hp"] for m in ours["mons"]] == [1, 1, 1]


def test_forced_switch_only_enters_a_selected_receiver(roster):
    session, engine, sets = roster
    session.apply("|faint|p1a: Scolipede")
    battle = session.decision(force_switch=True)
    incoming = battle.available_switches[0]
    ours = engine.own_side(battle, sets, incoming)
    assert ours["mons"][0]["hp"] == 0
    assert to_id(ours["team"][ours["incoming"]]["species"]) == incoming.species
    restored = worker("""
const b = setUp(sample,format,1);
console.log(JSON.stringify({left:b.p1.pokemonLeft, active:b.p1.active[0].species.id}));
""", {"p1": ours, "p2": {"team": [CHOMP], "mons": [{"hp": 1}]}})
    assert restored == {"left": 2, "active": incoming.species}
    with pytest.raises(ValueError, match="selected"):
        engine.own_side(battle, sets, battle.team["p1: Salamence"])


@pytest.mark.parametrize("invalid", ["missing", "preview", "duplicate", "unknown", "no_active", "six"])
def test_ambiguous_selected_rosters_fail_instead_of_guessing(roster, invalid):
    session, engine, sets = roster
    battle = session.decision()
    request = battle.last_request
    members = request["side"]["pokemon"]
    if invalid == "missing":
        request.pop("side")
    elif invalid == "preview":
        request["teamPreview"] = True
    elif invalid == "duplicate":
        members[2] = members[1]
    elif invalid == "unknown":
        members[2] = dict(members[2], ident="p1: missing")
    elif invalid == "no_active":
        request["side"]["pokemon"] = members[1:]
    else:
        request["side"]["pokemon"] = members * 2
    with pytest.raises(ValueError, match="selected"):
        engine.own_side(battle, sets)
