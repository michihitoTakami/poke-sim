from pokesim.rl.items import ITEM_CLASSES, SIZE, item_class


def test_item_class():
    assert item_class("choicescarf", set()) == "choicescarf"
    assert item_class("salamencite", set()) == "megastone"
    assert item_class("sitrusberry", set()) == "recovery"
    assert item_class("shucaberry", set()) == "resistberry"
    assert item_class("lumberry", set()) == "statusberry"
    assert item_class("blackglasses", set()) == "typeboost"
    assert item_class("lightclay", set()) == "other"
    assert SIZE == len(ITEM_CLASSES) + 12


def test_combine_product_needs_both():
    from pokesim.live.advisor import combine

    policy = {"move:calmmind": 0.997, "move:batonpass": 0.003}
    ranked = [("move:batonpass", 0.66, -1.6), ("move:moonlight", 0.23, -1.8), ("move:calmmind", 0.02, -2.9)]
    assert combine(policy, ranked, mode="linear")[0][0] == "move:calmmind"
    out = dict(combine(policy, ranked, temperature=2.0, margin=1.0))
    assert max(out, key=out.get) == "move:batonpass" and out["move:calmmind"] == 0.0
    assert abs(sum(out.values()) - 1) < 1e-6


def test_combine_defers_to_a_confident_lookahead():
    from pokesim.live.advisor import combine

    policy = {"move:substitute": 0.66, "move:protect": 0.29, "switch:umbreon": 0.0}
    ranked = [("switch:umbreon", 0.976, -0.63), ("switch:gholdengo", 0.024, -0.82), ("move:substitute", 0.0, -1.2)]
    assert combine(policy, ranked, temperature=2.0, margin=1.0, defer=0.9)[0][0] == "switch:umbreon"


def test_combine_without_lookahead_mass_uses_values():
    from pokesim.live.advisor import combine

    policy = {"move:slackoff": 1.0, "switch:scolipede": 0.0}
    ranked = [("switch:scolipede", 0.0, -3.16), ("move:slackoff", 0.0, -1.28)]
    assert combine(policy, ranked, temperature=2.0, margin=1.0, defer=0.9)[0][0] == "move:slackoff"
