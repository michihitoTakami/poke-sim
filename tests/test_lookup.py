import pytest
from sqlalchemy import text

from pokesim.lookup.names import NameIndex, normalize
from pokesim.lookup.tags import derive_tags


def test_normalize_width_and_kana():
    assert normalize("１０まんボルト") == normalize("10まんぼると")
    assert normalize("イカク") == normalize("いかく")
    assert normalize("メガリザードンＸ") == normalize("メガリザードンx")


def test_derive_tags():
    assert derive_tags(["onStart"], "On switch-in, lowers the Attack of opponents by 1 stage.") == \
        ["stat_drop", "switch_in"]
    assert derive_tags([], "", "levitate") == ["immunity"]


def test_name_index_fuzzy():
    idx = NameIndex([("species", "garchomp", "ja", "ガブリアス"), ("species", "garchomp", "en", "Garchomp"),
                     ("species", "gengar", "ja", "ゲンガー")])
    assert idx.resolve("ガブリアヌ", "species") == "garchomp"
    assert idx.resolve("garchomp", "species") == "garchomp"
    assert idx.resolve("全然違う", "species") is None


def _db_available() -> bool:
    try:
        from pokesim.db.session import engine
        with engine.connect() as c:
            return c.execute(text("select count(*) from species")).scalar() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="database not ingested")
def test_lookup_ability_ja_en_equivalent():
    from pokesim.lookup.effects import lookup
    ja, en = lookup("いかく", "ability"), lookup("Intimidate", "ability")
    assert ja.id == en.id == "intimidate"
    assert "switch_in" in ja.tags


@pytest.mark.skipif(not _db_available(), reason="database not ingested")
def test_every_legal_species_has_japanese_name():
    from pokesim.db.session import engine
    with engine.connect() as c:
        missing = c.execute(text("select count(*) from species s where not exists (select 1 from names_i18n n "
                                 "where n.kind='species' and n.ref_id=s.id and n.lang='ja')")).scalar()
    assert missing == 0
