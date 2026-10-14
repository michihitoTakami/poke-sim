import json

import numpy as np

from pokesim.config import ROOT
from pokesim.rl.human import canonical, decisions, known, split


def test_split_keeps_replays_on_one_side_as_data_grows():
    small = np.array([f"r{i}:{side}" for i in range(200) for side in ("p1", "p2")])
    big = np.array([f"r{i}:{side}" for i in range(400) for side in ("p1", "p2")])
    held_small = {small[i].split(":")[0] for i in split(small)[1]}
    held_big = {big[i].split(":")[0] for i in split(big)[1]}
    assert held_small == {r for r in held_big if int(r[1:]) < 200}
    tr, te = split(small)
    assert not {small[i].split(":")[0] for i in tr} & held_small
    assert 0.05 < len(te) / len(small) < 0.3


def test_canonical_renames_only_the_deciding_side():
    lines = ["|switch|p1a: Fluffy|Ninetales-Alola, L50, M|100/100", "|switch|p2a: Bob|Garchomp, L50|100/100",
             "|move|p2a: Bob|Earthquake|p1a: Fluffy", "|-damage|p1a: Fluffy|40/100"]
    out = canonical(lines, "p1", {"ninetalesalola": "Ninetales-Alola"}, {})
    assert out[0].startswith("|switch|p1a: Ninetales-Alola|")
    assert out[2] == "|move|p2a: Bob|Earthquake|p1a: Ninetales-Alola"
    assert out[3] == "|-damage|p1a: Ninetales-Alola|40/100"


TEAM = json.loads((ROOT / "data/my_teams/baton.json").read_text())


class StubSides:
    """Our real sets for p1, no database: what decisions() needs from Sides."""
    species: dict = {}

    def sets(self, log, role):
        return TEAM

    def revealed(self, log, role):
        return {"scolipede": {"irondefense", "protect", "substitute", "batonpass"},
                "umbreon": {"snarl", "batonpass", "calmmind", "moonlight"}}


def battle_log(turn1: list[str], turn2: list[str] | None = None) -> str:
    head = ["|player|p1|me|", "|player|p2|foe|", "|gen|9", "|gametype|singles"]
    head += [f"|poke|p1|{s['species']}, L50|" for s in TEAM]
    head += [f"|poke|p2|{s}, L50|" for s in ("Garchomp", "Hippowdon", "Rotom-Wash", "Gyarados", "Kingambit",
                                               "Primarina")]
    head += ["|teampreview|3", "|start", "|switch|p1a: Scolipede|Scolipede, L50|165/165",
             "|switch|p2a: Garchomp|Garchomp, L50|100/100", "|turn|1"]
    return "\n".join(head + turn1 + ["|turn|2"] + (turn2 or ["|move|p1a: Scolipede|Protect|p1a: Scolipede"]))


def turn_obs(log: str) -> list:
    ds, _ = decisions(log, "p1", StubSides())
    return [d for d in ds if d.kind == "turn"]


def test_turn_input_is_fixed_before_either_side_acts():
    ours = ["|move|p1a: Scolipede|Iron Defense|p1a: Scolipede", "|-boost|p1a: Scolipede|def|2"]
    theirs = ["|-mega|p2a: Garchomp|Garchomp|Garchompite", "|move|p2a: Garchomp|Earthquake|p1a: Scolipede",
              "|-damage|p1a: Scolipede|40/165"]
    first = turn_obs(battle_log(ours + theirs))
    second = turn_obs(battle_log(theirs + ours))  # the opponent moved first: its move, Mega and damage come before
    assert [d.label for d in first][:1] == [d.label for d in second][:1] == ["move:irondefense"]
    assert np.array_equal(first[0].obs, second[0].obs)
    assert np.array_equal(first[0].mask, second[0].mask)


def test_later_lines_do_not_change_earlier_inputs():
    base = turn_obs(battle_log(["|move|p1a: Scolipede|Iron Defense|p1a: Scolipede"]))
    longer = turn_obs(battle_log(["|move|p1a: Scolipede|Iron Defense|p1a: Scolipede"],
                                 ["|move|p1a: Scolipede|Protect|p1a: Scolipede", "|-mega|p2a: Garchomp|Garchomp|x",
                                  "|move|p2a: Garchomp|Earthquake|p1a: Scolipede", "|faint|p1a: Scolipede",
                                  "|win|foe"]))
    assert np.array_equal(base[0].obs, longer[0].obs)


def test_switch_after_our_faint_is_a_forced_decision():
    turn1 = ["|move|p2a: Garchomp|Earthquake|p1a: Scolipede", "|-damage|p1a: Scolipede|0 fnt",
             "|faint|p1a: Scolipede", "|switch|p1a: Umbreon|Umbreon, L50|201/201"]
    ds, _ = decisions(battle_log(turn1, ["|move|p1a: Umbreon|Calm Mind|p1a: Umbreon"]), "p1", StubSides())
    assert [(d.kind, d.turn, d.label) for d in ds] == [("force", 1, "switch:umbreon"),
                                                         ("turn", 2, "move:calmmind")]


def test_unknown_results_have_no_value_target():
    assert known({"returns": np.array([0.97, 0.0, -1.0])}).tolist() == [True, False, True]
    assert known({"returns": np.zeros(2), "known": np.array([True, False])}).tolist() == [True, False]
