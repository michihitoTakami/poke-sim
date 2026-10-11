from pokesim.data.parse_replays import parse_log
from pokesim.db.models import Species


def sp(sid: str, mega_of: str | None = None) -> Species:
    return Species(id=sid, is_mega=mega_of is not None, changes_from=mega_of)


SPECIES = {s.id: s for s in [sp("charizard"), sp("charizardmegay", "Charizard"), sp("primarina"), sp("salamence"),
                             sp("garchomp"), sp("rotomwash"), sp("hippowdon"), sp("corviknight")]}
LOG = """|player|p1|Alice|1|
|player|p2|Bob|2|
|poke|p1|Primarina, L50, F|
|poke|p1|Salamence, L50, F|
|poke|p1|Garchomp, L50, M|
|poke|p2|Charizard, L50, F|
|poke|p2|Rotom-Wash, L50|
|poke|p2|Hippowdon, L50, M|
|switch|p1a: Prim|Primarina, L50, F|100/100
|switch|p2a: Charizard|Charizard, L50, F|100/100
|detailschange|p2a: Charizard|Charizard-Mega-Y, L50, F
|-mega|p2a: Charizard|Charizard|Charizardite Y
|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Charizard
|move|p1a: Prim|Moonblast|p2a: Charizard
|-heal|p1a: Prim|79/100|[from] item: Leftovers
|move|p2a: Charizard|Solar Beam|p1a: Prim
|switch|p1a: Salamence|Salamence, L50, F|100/100
|-unboost|p2a: Charizard|atk|1|[from] ability: Intimidate|[of] p1a: Salamence
|move|p1a: Salamence|Sleep Talk|p1a: Salamence
|move|p1a: Salamence|Earthquake|p2a: Charizard|[from] move: Sleep Talk
|win|Alice"""


def test_reads_preview_moves_items_and_abilities():
    sides = parse_log(LOG, SPECIES)
    p1, p2 = sides["p1"], sides["p2"]
    assert p1.player == "Alice" and p1.won and not p2.won
    assert p1.preview == ["primarina", "salamence", "garchomp"]
    assert p1.mons["primarina"].moves == ["moonblast"] and p1.mons["primarina"].item == "leftovers"
    assert p1.mons["salamence"].ability == "intimidate"
    assert p1.mons["salamence"].moves == ["sleeptalk"]  # moves called by Sleep Talk are not the set's
    zard = p2.mons["charizard"]
    assert zard.mega and zard.item == "charizarditey" and zard.ability is None  # Drought is the mega's ability
    assert zard.moves == ["solarbeam"]
