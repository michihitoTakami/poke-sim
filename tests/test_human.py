import numpy as np

from pokesim.rl.human import canonical, split


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
