import pytest
from test_lookup import _db_available

from pokesim.data.parse_articles import parse_stat_points


def test_stat_points_parenthesized_sequence():
    assert parse_stat_points("215(32)-132(0)-\n170(17)\n-\nx\n-107(17)-67(0)") == [32, 0, 17, 0, 17, 0]
    assert parse_stat_points("183-198(30)-119(4)-x-105-154(32)") == [0, 30, 4, 0, 0, 32]


def test_stat_points_letters():
    assert parse_stat_points("臆病　B2 C32 S32") == [0, 0, 2, 32, 0, 32]


def test_stat_points_rejects_false_matches():
    assert parse_stat_points("HB 11n") is None  # an adjustment note, not points
    assert parse_stat_points("2026-10-10") is None


ARTICLE = """<div class="entry-content">
<p>個体解説</p><p>カバルドン</p><p>ルカリオ</p>
<h3>カバルドン@食べ残し</h3>
<p>特性　砂おこし</p><p>性格　腕白</p>
<p>努力値 215(32)-132(0)-170(17)-x-107(17)-67(0)</p>
<p>技　地震　欠伸　ステルスロック　怠ける</p>
<h3>ルカリオ</h3>
<p>特性　精神力→波動の防護</p><p>もちもの　ルカリオナイトZ</p><p>性格　控えめ</p>
<p>技　ラスターカノン　真空波　悪の波動　悪巧み</p>
</div>"""


@pytest.mark.skipif(not _db_available(), reason="database not ingested")
def test_parse_article_extracts_sets():
    from pokesim.data.parse_articles import parse_article
    from pokesim.lookup.names import default_index
    hippo, lucario = parse_article(default_index(), ARTICLE)
    assert (hippo.species_id, hippo.item_id, hippo.ability_id, hippo.nature) == \
        ("hippowdon", "leftovers", "sandstream", "impish")
    assert hippo.moves == ["earthquake", "yawn", "stealthrock", "slackoff"]
    assert hippo.stat_points == [32, 0, 17, 0, 17, 0]
    assert (lucario.item_id, lucario.ability_id) == ("lucarionitez", "auraguard")
    assert lucario.moves == ["flashcannon", "vacuumwave", "darkpulse", "nastyplot"]
