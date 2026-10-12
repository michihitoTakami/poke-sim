"""Battle-screen text -> events (MessageParser, ScreenTracker). Texts are real OCR outputs from captures."""
from pokesim.live.screen import MessageParser, ScreenTracker, Vocab
from pokesim.vision.battle import Hud, Reading

SPECIES = {"umbreon": "ブラッキー", "scolipede": "ペンドラー", "salamence": "ボーマンダ", "gholdengo": "サーフゴー",
           "floetteeternal": "フラエッテ", "skeledirge": "ラウドボーン", "glimmora": "キラフロル",
           "mimikyu": "ミミッキュ", "lucario": "ルカリオ", "golisopod": "グソクムシャ", "gengar": "ゲンガー",
           "charizard": "リザードン", "charizardmegax": "メガリザードンＸ", "charizardmegay": "メガリザードンＹ"}
MOVES = {"calmmind": "めいそう", "snarl": "バークアウト", "moonlight": "つきのひかり", "batonpass": "バトンタッチ",
         "dazzlinggleam": "マジカルシャイン", "mortalspin": "キラースピン", "powergem": "パワージェム",
         "meteorbeam": "メテオビーム", "irondefense": "てっぺき", "substitute": "みがわり", "protect": "まもる",
         "flamethrower": "かえんほうしゃ", "earthquake": "じしん"}
MY = ["Umbreon", "Scolipede", "Salamence", "Gholdengo", "Floette-Eternal", "Skeledirge"]
MY_MOVES = {"umbreon": ["snarl", "batonpass", "calmmind", "moonlight"],
            "scolipede": ["irondefense", "protect", "substitute", "batonpass"]}
OPP = ["Glimmora", "Mimikyu", "Lucario", "Golisopod", "Gengar", "Charizard"]


def vocab() -> Vocab:
    return Vocab(me=MY, my_moves=MY_MOVES, opp=OPP, species_ja=SPECIES, moves_ja=MOVES)


def test_parse_ocr_messages():
    p = MessageParser(vocab())
    assert p.parse("相手の キラフロルの マジカルシヤイン」") == [
        {"type": "move", "side": "opp", "move": "dazzlinggleam", "species": "Glimmora"}]
    assert p.parse("ブラツキーの めいそう!")[0]["move"] == "calmmind"
    boosts = p.parse("ブラツキーの 特攻 特防が 上がつた!")
    assert [(b["stat"], b["amount"]) for b in boosts] == [("spa", 1), ("spd", 1)]
    assert p.parse("相手の キラフロルの 特攻が 下がった!")[0] | {"species": None} == {
        "type": "boost", "side": "opp", "stat": "spa", "amount": -1, "species": None}
    assert p.parse("%つ:んは 三ミツキユを 繰り出した!")[0]["species"] == "Mimikyu"     # trainer name in any script
    assert p.parse("きつねは た アシレーヌを 繰ゆ出したり") == []                        # not in the opponent's 6
    assert p.parse("ゆけつ! フラエッテ!")[0] == {"type": "switch", "side": "me", "species": "Floette-Eternal"}
    assert p.parse("ブラツキーは とく 毒を あびた!")[0]["status"] == "psn"
    assert p.parse("ペンドラーは たおれた」")[0]["type"] == "faint"
    assert p.parse("味方の 周りに とがつた岩が ただよい始めた!") == [
        {"type": "side_start", "side": "me", "condition": "Stealth Rock"}]
    assert p.parse("砂あらしが 吹き始めたり") == [{"type": "weather", "weather": "Sandstorm"}]
    assert p.parse("相手の リザードンは メガリザードンＹに メガシンカした!")[0]["mega"] == "charizardmegay"
    assert p.parse("~ん』ゆゆ」との 勝負に 負けたり") == [{"type": "win", "winner": "opp"}]
    assert p.parse("効果は バツグンだ!") == []
    assert p.parse("ブラツキーは とく 毒の ダメージを受けた」") == []


def test_tracker_turns_and_hp():
    t = ScreenTracker(vocab(), maxhp={"Umbreon": 201, "Scolipede": 165})
    hud_me = lambda f: Hud("ブラツキ一", f, "")  # noqa: E731
    hud_opp = lambda f: Hud("キラフロル", f, f"{round(f * 100)}%")  # noqa: E731
    seq = [Reading(message="yoshiは キラフロルを 繰り出した!"), Reading(message="yoshiは キラフロルを 繰り出した!"),
           Reading(message="ゆけつ! ブラツキー!"),
           Reading(opp=hud_opp(1.0), me=hud_me(1.0), decision=True),        # turn 1
           Reading(opp=hud_opp(1.0), me=hud_me(1.0), decision=True),
           Reading(message="相手の キラフロルの マジカルシヤイン」"),
           Reading(message="相手の キラフロルの マジカルシャイン!"),
           Reading(me=hud_me(0.68)), Reading(me=hud_me(0.68)),
           Reading(message="ブラツキーの バトンタツチ」"),
           Reading(decision=True),                                         # pivot: choose who comes in
           Reading(message="ゆけつ! ペンドラー!"),
           Reading(message="ペンドラーは たおれた!"),
           Reading(decision=True)]                                         # fainted: forced switch
    events = t.run(seq)
    kinds = [(e["type"], e.get("side")) for e in events]
    assert kinds == [("switch", "opp"), ("switch", "me"), ("turn", None), ("move", "opp"), ("hp", "me"),
                     ("move", "me"), ("force_switch", None), ("switch", "me"), ("faint", "me"),
                     ("force_switch", None)]
    assert events[4]["hp"] == round(0.68 * 201)


def test_tracker_infers_missed_switch_and_turn():
    t = ScreenTracker(vocab(), maxhp={"Umbreon": 201})
    events = t.run([Reading(message="相手の キラフロルの パワージエム!"), Reading(message="ブラツキーの めいそう!"),
                    Reading(), Reading(), Reading(), Reading(), Reading(message="ブラツキーの めいそう!")])
    kinds = [(e["type"], e.get("side")) for e in events]
    # the switch-ins were never shown; a side moving twice means a turn prompt was missed
    assert kinds == [("switch", "opp"), ("move", "opp"), ("switch", "me"), ("move", "me"), ("turn", None),
                     ("move", "me")]


def test_parse_volatile_effects():
    p = MessageParser(vocab())
    assert p.parse("ペンドラーの みか あらわ 身代わりが 現れた!")[0] | {"species": None} == {
        "type": "start", "side": "me", "effect": "Substitute", "species": None}
    assert p.parse("ペンドラーの みか 身代わりは 消えてしまつた")[0]["type"] == "end"
    assert p.parse("ペンドラーの 眠気を 誘った!")[0]["effect"] == "move: Yawn"
    assert p.parse("相手の ゲンガーは 混乱した!")[0]["side"] == "opp"


def test_opponent_outside_the_preview():
    v = vocab()
    v.species_ja["blastoise"] = "カメックス"
    v.__post_init__()
    p = MessageParser(v)
    assert p.parse("トレーナーは カメックスを 繰り出した!")[0]["species"] == "blastoise"
