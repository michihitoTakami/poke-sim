"""A worker's evolved-form switch label must remain executable in poke-env's base-species roster."""
import json
from types import SimpleNamespace

import pytest
from poke_env.battle import Pokemon
from poke_env.environment import SinglesEnv

from pokesim.config import ROOT
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import label_to_action


@pytest.mark.parametrize("species,forme,label", [
    ("floetteeternal", "Floette-Mega", "floettemega"),
    ("salamence", "Salamence-Mega", "salamencemega"),
    ("charizard", "Charizard-Mega-X", "charizardmegax"),
    ("charizard", "Charizard-Mega-Y", "charizardmegay"),
    ("mimikyu", "Mimikyu-Busted", "mimikyubusted"),
])
def test_changed_form_maps_to_its_original_team_slot(species, forme, label):
    mon = Pokemon(gen=9, species=species)
    mon.forme_change(forme)
    battle = SimpleNamespace(team={"p1: active": Pokemon(gen=9, species="gholdengo"), "p1: target": mon})
    assert label_to_action(f"switch:{label}", battle) == 1
    assert label_to_action(f"switch:{species}", battle) == 1


@pytest.mark.parametrize("species,label", [
    ("floetteeternal", "floettemega"), ("floette", "floettemega"),
    ("rotomwash", "rotomheat"), ("charizard", "charizardmegax"),
])
def test_an_unobserved_or_unrelated_form_does_not_match(species, label):
    battle = SimpleNamespace(team={"p1: target": Pokemon(gen=9, species=species)})
    assert label_to_action(f"switch:{label}", battle) is None


def test_other_mega_branch_is_not_the_same_current_form():
    mon = Pokemon(gen=9, species="charizard")
    mon.forme_change("Charizard-Mega-X")
    assert label_to_action("switch:charizardmegay", SimpleNamespace(team={"p1: target": mon})) is None


def test_benched_mega_floette_is_a_legal_switch_in_a_real_request():
    sets = json.loads((ROOT / "data/my_teams/baton.json").read_text())
    session = BattleSession(sets)
    session.preview(["Samurott-Hisui", "Metagross", "Hippowdon"])
    session.select([1, 0, 3])
    session.apply(["|switch|p1a: Floette|Floette-Eternal, L50, F|173/173",
                   "|switch|p2a: Samurott|Samurott-Hisui, L50, F|100/100",
                   "|detailschange|p1a: Floette|Floette-Mega, L50, F",
                   "|switch|p1a: Gholdengo|Gholdengo, L50|175/175", "|turn|7"])
    battle = session.decision()
    action = label_to_action("switch:floettemega", battle)
    assert action is not None
    assert SinglesEnv.get_action_mask(battle)[action]
    order = SinglesEnv.action_to_order(action, battle, strict=True)
    assert "switch" in order.message
