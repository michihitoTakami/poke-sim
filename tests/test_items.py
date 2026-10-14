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
