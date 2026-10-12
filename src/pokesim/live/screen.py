"""Battle screen readings -> structured events for EventTranslator (replaces the manual turn input).

MessageParser  one message-box text -> events (move, switch, faint, status, boosts, weather, hazards, mega, items,
               win). Names are matched fuzzily against the two teams (OCR misreads ミ as 三, ッ as ツ, ...), moves
               against our known moves or every legal move for the opponent.
ScreenTracker  a stream of per-frame readings -> events in battle order:
               - a message is emitted once (the same text on consecutive frames is one message);
               - HP from the HUD bars once a value is stable over two frames;
               - a switch the messages missed is caught from the HUD name;
               - the COMMAND prompt appearing marks our decision: "turn" (or a forced switch after our Pokémon
                 fainted / used a pivot move).
"""
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from pokesim.rl.myteam import to_id
from pokesim.vision.battle import Hud, Reading, norm

DEX = Path(__file__).resolve().parents[3] / "data" / "processed" / "dex.json"
STATS = {"攻撃": "atk", "防御": "def", "特攻": "spa", "特防": "spd", "素早さ": "spe", "命中率": "accuracy",
         "回避率": "evasion"}
STAGES = [("ぐぐーんと上がった", 3), ("ぐーんと上がった", 2), ("上がった", 1), ("がくーんと下がった", -3),
          ("がくっと下がった", -2), ("下がった", -1), ("もう上がらない", 0), ("もう下がらない", 0)]
STATUS_START = {"猛毒をあびた": "tox", "毒をあびた": "psn", "やけどを負った": "brn", "まひして": "par",
                "眠ってしまった": "slp", "凍りついた": "frz", "こおりついた": "frz"}
STATUS_END = {"目を覚ました": "slp", "やけどが治った": "brn", "しびれがとれた": "par", "毒はきれいさっぱり": "psn",
              "こおりが溶けた": "frz", "こおりがとけた": "frz"}
WEATHER = {"砂あらしが吹き始めた": "Sandstorm", "日差しが強くなった": "SunnyDay", "雨が降り始めた": "RainDance",
           "雪が降り始めた": "Snowscape", "砂あらしがおさまった": "none", "日差しが元に戻った": "none",
           "雨が上がった": "none", "雨がやんだ": "none", "雪が止んだ": "none", "雪がやんだ": "none"}
SIDE_START = {"周りにとがった岩がただよい始めた": "Stealth Rock", "足下にまきびしが散らばった": "Spikes",
              "足下にどくびしが散らばった": "Toxic Spikes", "足下にねばねばネットが広がった": "Sticky Web",
              "リフレクターで物理に強くなった": "Reflect", "ひかりのかべで特殊に強くなった": "Light Screen",
              "オーロラベールで": "Aurora Veil", "追い風が吹き始めた": "Tailwind"}
SIDE_END = {"ステルスロックが消え去った": "Stealth Rock", "まきびしが消え去った": "Spikes",
            "どくびしが消え去った": "Toxic Spikes", "ねばねばネットが消え去った": "Sticky Web",
            "リフレクターがなくなった": "Reflect", "ひかりのかべがなくなった": "Light Screen",
            "オーロラベールがなくなった": "Aurora Veil", "追い風が止んだ": "Tailwind"}
FIELD = {"電気がかけめぐる": "Electric Terrain", "草がおいしげった": "Grassy Terrain",
         "霧が立ち込めた": "Misty Terrain",
         "不思議な感じになった": "Psychic Terrain", "時空をゆがめた": "Trick Room"}
ITEM_MSG = {"ふうせんで浮いている": ("airballoon", False), "ふうせんが割れた": ("airballoon", True),
            "たべのこしで少し回復": ("leftovers", False), "命が少し削られた": ("lifeorb", False),
            "きあいのタスキでもちこたえた": ("focussash", True), "くろいヘドロで": ("blacksludge", False)}
# volatile effects on a Pokémon: phrase -> (start/end, effect as Showdown writes it in -start / -end)
VOLATILE = {"身代わりが現れた": ("start", "Substitute"), "身代わりは消えてしまった": ("end", "Substitute"),
            "眠気を誘": ("start", "move: Yawn"), "ねむけをさそ": ("start", "move: Yawn"),
            "混乱した": ("start", "confusion"), "混乱が解けた": ("end", "confusion"),
            "挑発に乗ってしまった": ("start", "move: Taunt"), "挑発の効果が解けた": ("end", "move: Taunt"),
            "アンコールを受けた": ("start", "Encore"), "アンコール状態が解けた": ("end", "Encore"),
            "種を植えつけた": ("start", "move: Leech Seed"), "やどりぎのタネから解放": ("end", "move: Leech Seed"),
            "回復動作を封じられた": ("start", "move: Heal Block"), "いちゃもんをつけられた": ("start", "move: Torment"),
            "しおづけになった": ("start", "Salt Cure"), "のろいをかけた": ("start", "Curse"),
            "メロメロになった": ("start", "Attract"), "張り切っている": ("start", "move: Focus Energy"),
            "道連れにしようとしている": ("start", "Destiny Bond")}
ABILITY_MSG = {"ばけのかわがはがれた": "disguise", "ばけのかわが": "disguise"}
MY_SWITCH = ("ゆけっ", "いけっ", "行け", "ゆけ", "いけ", "がんばれ", "まかせた", "たのんだ", "いってこい", "頼んだ")
PIVOTS = {"batonpass", "uturn", "voltswitch", "flipturn", "partingshot", "teleport", "shedtail", "chillyreception"}
IGNORE = ("効果はバツグン", "効果はいまひとつ", "急所に当たった", "当たらなかった", "うまく決まらなかった")


def _n(*parts: str) -> list[str]:
    return [norm(p) for p in parts]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    return max(r, 0.9) if (len(a) >= 3 and a in b) or (len(b) >= 3 and b in a) else r


def best(text: str, options: dict[str, str], cutoff: float) -> tuple[str | None, float]:
    """options: normalized display name -> id."""
    scored = [(similarity(text, k), v) for k, v in options.items()]
    if not scored:
        return None, 0.0
    score, ident = max(scored)
    return (ident, score) if score >= cutoff else (None, score)


def load_names() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """id -> Japanese name for species, moves, items, abilities (Showdown dex export)."""
    dex = json.loads(DEX.read_text())
    out = []
    for key in ("species", "moves", "items", "abilities"):
        out.append({e["id"]: e["nameJa"] for e in dex[key] if e.get("nameJa")})
    return tuple(out)


@dataclass
class Vocab:
    me: list[str]                         # our team, species ids
    my_moves: dict[str, list[str]]        # species id -> move ids
    opp: list[str]                        # opponent's 6, species ids
    species_ja: dict[str, str]
    moves_ja: dict[str, str]
    items_ja: dict[str, str] = field(default_factory=dict)
    abilities_ja: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        def names(ids):
            out = {}
            for sid in ids:
                ja = self.species_ja.get(to_id(sid)) or self.species_ja.get(to_id(sid.split("-")[0]))
                if ja:
                    out[norm(ja)] = sid
            return out

        self.me_names, self.opp_names = names(self.me), names(self.opp)
        self.all_moves = {norm(v): k for k, v in self.moves_ja.items()}
        self.items = {norm(v): k for k, v in self.items_ja.items()}
        self.abilities = {norm(v): k for k, v in self.abilities_ja.items()}

    @classmethod
    def build(cls, sets: list[dict], opp: list[str], ja: dict[str, str] | None = None) -> "Vocab":
        """ja: id -> Japanese name from the DB (covers Champions formes the dex export lacks, e.g. Floette-Eternal)."""
        species, moves, items, abilities = load_names()
        if ja:
            species = {**species, **{k: v for k, v in ja.items() if k not in moves}}
        return cls(me=[s["species"] for s in sets], my_moves={to_id(s["species"]): [to_id(m) for m in s["moves"]]
                                                             for s in sets},
                   opp=list(opp), species_ja=species, moves_ja=moves, items_ja=items, abilities_ja=abilities)

    def moves_for(self, side: str, species: str | None) -> dict[str, str]:
        if side == "me" and species and to_id(species) in self.my_moves:
            return {norm(self.moves_ja.get(m, m)): m for m in self.my_moves[to_id(species)]}
        return self.all_moves


class MessageParser:
    def __init__(self, vocab: Vocab):
        self.v = vocab

    def who(self, text: str, side: str) -> tuple[str | None, str]:
        """A Pokémon name -> (species, side). Falls back to the other team when the name clearly belongs to it
        (the "相手の" prefix can be lost by OCR)."""
        own, other = (self.v.opp_names, self.v.me_names) if side == "opp" else (self.v.me_names, self.v.opp_names)
        sid, score = best(text, own, 0.5)
        alt, alt_score = best(text, other, 0.75)
        if alt and alt_score > score + 0.15:
            return alt, "me" if side == "opp" else "opp"
        return sid, side

    def parse(self, raw: str) -> list[dict]:
        t = norm(raw)
        if not t or any(k in t for k in _n(*IGNORE)):
            return []
        side = "me"
        m = re.match("^(?:あいて)?相手の", t)
        if m:
            side, t = "opp", t[m.end():]
        t = re.sub("^みかた(?=味方)", "", t)

        if norm("勝負に勝った") in t:
            return [{"type": "win", "winner": "me"}]
        if norm("勝負に負けた") in t:
            return [{"type": "win", "winner": "opp"}]
        for phrase, weather in WEATHER.items():
            if norm(phrase) in t:
                return [{"type": "weather", "weather": weather}]
        for phrase, cond in FIELD.items():
            if norm(phrase) in t:
                return [{"type": "field", "field": cond}]
        for table, kind in ((SIDE_START, "side_start"), (SIDE_END, "side_end")):
            for phrase, cond in table.items():
                if norm(phrase) in t:
                    team = "opp" if side == "opp" or t.startswith("相手") else "me"
                    return [{"type": kind, "side": team, "condition": cond}]

        sent_out = norm("は") + "(.+?)" + norm("を") + ".{0,3}" + norm("出し")  # 繰り出した, often misread
        m = re.search(sent_out + "|" + "(.+?)" + norm("は戦闘に引きずりだされた"), t)
        if m:  # the opponent's trainer sends out (their name is any script, so only the tail is used)
            name = m.group(1) or m.group(2)
            sid, s = self.who(name, "opp" if m.group(1) else side)
            return [{"type": "switch", "side": s, "species": sid}] if sid else []
        for word in _n(*MY_SWITCH):
            if t.startswith(word):
                sid, s = self.who(t[len(word):], "me")
                return [{"type": "switch", "side": s, "species": sid}] if sid else []
        if norm("を引っこめた") in t or t.endswith(norm("戻れ")):
            return []

        m = re.match("(.+?)" + norm("はたおれた"), t)
        if m:
            sid, s = self.who(m.group(1), side)
            return [{"type": "faint", "side": s, "species": sid}] if sid else []
        m = re.match("(.+?)" + norm("はメガ") + "(.+?)" + norm("にメガシンカ"), t)
        if m:
            sid, s = self.who(m.group(1), side)
            if not sid:
                return []
            forms = {norm(v): k for k, v in self.v.species_ja.items() if k.startswith(to_id(sid) + "mega")}
            mega, _ = best(norm("メガ") + m.group(2), forms, 0.0)  # Mega X / Y: the letter decides when it is read
            return [{"type": "mega", "side": s, "species": sid, "mega": mega}]
        for phrase, status in STATUS_START.items():
            m = re.match("(.+?)" + norm("は") + "(?:.{0,4})" + norm(phrase), t)
            if m:
                sid, s = self.who(m.group(1), side)
                return [{"type": "status", "side": s, "status": status, "species": sid}] if sid else []
        for phrase, status in STATUS_END.items():
            m = re.match("(.+?)" + "[はの]" + "(?:.{0,4})" + norm(phrase), t)
            if m:
                sid, s = self.who(m.group(1), side)
                return [{"type": "cure", "side": s, "status": status, "species": sid}] if sid else []
        for phrase, (kind, effect) in VOLATILE.items():
            if norm(phrase) not in t:
                continue
            # "{SOURCE}は ... {POKEMON}に のろいを かけた": the cursed one is named last
            names = re.findall("(.+?)" + "[のはに]", t[:t.find(norm(phrase))])
            name = names[-1] if phrase == "のろいをかけた" and names else (names[0] if names else "")
            sid, s = self.who(name, side)
            return [{"type": kind, "side": s, "effect": effect, "species": sid}] if sid else []
        for phrase, (item, consumed) in ITEM_MSG.items():
            m = re.match("(.+?)" + "[はの]" + norm(phrase), t)
            if m:
                sid, s = self.who(m.group(1), side)
                return [{"type": "item", "side": s, "item": item, "consumed": consumed, "species": sid}] if sid else []
        for phrase, ability in ABILITY_MSG.items():
            m = re.match("(.+?)" + norm("の") + norm(phrase), t)
            if m:
                sid, s = self.who(m.group(1), side)
                return [{"type": "ability", "side": s, "ability": ability, "species": sid}] if sid else []

        boost = self.boosts(t, side)
        if boost is not None:
            return boost
        m = re.match("(.+?)" + norm("の") + "(.+)$", t)
        if m:
            sid, s = self.who(m.group(1), side)
            if sid:
                move, _ = best(m.group(2), self.v.moves_for(s, sid), 0.72)
                if move:
                    return [{"type": "move", "side": s, "move": move, "species": sid}]
        return []

    def boosts(self, t: str, side: str) -> list[dict] | None:
        for phrase, amount in STAGES:
            p = norm(phrase)
            if not t.endswith(p) and p + "!" not in t and not re.search(re.escape(p) + ".?$", t):
                continue
            head = t[:t.rfind(p)]
            m = re.match("(.+?)" + norm("の") + "(.+?)" + "[がは]$", head)
            if not m:
                return None
            stats = [sid for word, sid in STATS.items() if norm(word) in m.group(2)]
            if not stats:
                return None
            sid, s = self.who(m.group(1), side)
            if not sid:
                return []
            if not amount:
                return []
            return [{"type": "boost", "side": s, "stat": st, "amount": amount, "species": sid} for st in stats]
        return None


@dataclass
class SideState:
    active: str | None = None
    hp_emitted: float | None = None   # percent (opp) or HP points (us)
    hp_pending: float | None = None
    name_pending: str | None = None
    name_streak: int = 0
    fainted: set = field(default_factory=set)


class ScreenTracker:
    def __init__(self, vocab: Vocab, maxhp: dict[str, int]):
        self.v, self.parser = vocab, MessageParser(vocab)
        self.maxhp = {to_id(k): v for k, v in maxhp.items()}
        self.sides = {"me": SideState(), "opp": SideState()}
        self.last_msg: str | None = None
        self.msg_gap = 99
        self.in_decision = False
        self.decision_gap = 99
        self.since_decision = 0      # messages since the last decision prompt
        self.force = False           # the next prompt is a switch-in choice (our Pokémon fainted / pivot move)
        self.finished = False
        self.moved: set[str] = set()  # sides that used a move since the last turn
        self.last_events: list[dict] = []

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _switch(self, side: str, species: str) -> list[dict]:
        st = self.sides[side]
        if st.active and to_id(st.active) == to_id(species):
            return []
        st.active, st.hp_emitted, st.hp_pending, st.name_pending, st.name_streak = species, None, None, None, 0
        return [{"type": "switch", "side": side, "species": species}]

    def _message(self, text: str | None) -> list[dict]:
        if not text:
            self.msg_gap += 1
            return []
        key = norm(text)
        same = self.last_msg is not None and (key.startswith(self.last_msg) or self.last_msg.startswith(key)
                                              or similarity(key, self.last_msg) >= 0.8)
        if same and self.msg_gap <= 2:  # the same message on consecutive frames (OCR may vary a little)
            self.msg_gap = 0
            if len(key) > len(self.last_msg):
                self.last_msg = key
            return []
        events = self.parser.parse(text)
        if not events:
            return []
        gap, self.last_msg, self.msg_gap = self.msg_gap, key, 0
        if events == self.last_events and gap <= 2:
            return []
        self.last_events = events
        out = []
        for ev in events:
            if ev["type"] == "switch":
                out += self._switch(ev["side"], ev["species"])
                continue
            if ev.get("species") and ev["type"] != "faint":
                out += self._switch(ev["side"], ev["species"])  # its switch-in message was missed
            if ev["type"] == "faint":
                if not self._faint(ev["side"]):
                    continue
            if ev["type"] == "move" and ev["side"] == "me" and ev["move"] in PIVOTS:
                self.force = True
            if ev["type"] == "move":
                if ev["side"] in self.moved:  # each side moves once a turn: the turn prompt was missed
                    out.append({"type": "turn", "missed": True})
                    self.moved.clear()
                self.moved.add(ev["side"])
            if ev["type"] == "win":
                self.finished = True
            if self.sides[ev.get("side", "me")].active is None and ev["type"] not in ("weather", "field", "win",
                                                                                     "side_start", "side_end"):
                continue
            out.append({k: v for k, v in ev.items() if k != "species"})
        self.since_decision += len(out)
        return out

    def _faint(self, side: str) -> bool:
        """Mark the active Pokémon fainted; False when it already was (message and HP bar both report it)."""
        st = self.sides[side]
        if st.active is None or st.active in st.fainted:
            return False
        st.fainted.add(st.active)
        st.hp_emitted = 0
        if side == "me":
            self.force = True
        return True

    def _hud(self, side: str, hud: Hud | None) -> list[dict]:
        if hud is None:
            return []
        st = self.sides[side]
        names = self.v.opp_names if side == "opp" else self.v.me_names
        sid, _ = best(norm(hud.name), names, 0.5)
        out = []
        if sid and (st.active is None or to_id(sid) != to_id(st.active)):
            # a switch the messages missed: trust the name plate once it is stable
            st.name_streak = st.name_streak + 1 if st.name_pending == sid else 1
            st.name_pending = sid
            if st.name_streak < 3:
                return []
            out += self._switch(side, sid)
        st.name_streak = 0
        if not sid or st.active is None or st.active in st.fainted:
            return out
        value = self._hp_value(side, st.active, hud)
        if value is None:
            return out
        tol = 2 if side == "opp" else max(2, 0.02 * self.maxhp.get(to_id(st.active), 100))
        if st.hp_pending is not None and abs(value - st.hp_pending) <= tol:
            ref = st.hp_emitted
            if ref is None or abs(value - ref) > tol:
                if value == 0:  # an empty bar: fainted (its message may be missed)
                    if self._faint(side):
                        out.append({"type": "faint", "side": side})
                elif not (ref is None and value >= self._full(side) - tol):
                    out.append({"type": "hp", "side": side, "hp": value})
                st.hp_emitted = value
        st.hp_pending = value
        return out

    def _full(self, side: str) -> float:
        return 100 if side == "opp" else self.maxhp.get(to_id(self.sides[side].active), 0)

    def _flush_hp(self, side: str, hud: Hud | None) -> list[dict]:
        st = self.sides[side]
        if hud is None or st.active is None or st.active in st.fainted:
            return []
        sid, _ = best(norm(hud.name), self.v.opp_names if side == "opp" else self.v.me_names, 0.5)
        if not sid or to_id(sid) != to_id(st.active):
            return []
        value = self._hp_value(side, st.active, hud)
        tol = 2 if side == "opp" else max(2, 0.02 * self.maxhp.get(to_id(st.active), 100))
        if not value or (st.hp_emitted is not None and abs(value - st.hp_emitted) <= tol):
            return []
        if st.hp_emitted is None and value >= self._full(side) - tol:
            st.hp_emitted = st.hp_pending = value  # full HP is what the switch-in already said
            return []
        st.hp_emitted = st.hp_pending = value
        return [{"type": "hp", "side": side, "hp": value}]

    def _hp_value(self, side: str, species: str, hud: Hud) -> float | None:
        if side == "opp":
            bar = round(hud.frac * 100 - 0.5)
            m = re.search(r"(\d{1,3})\s*%", hud.value)
            if m and abs(int(m.group(1)) - bar) <= 6 and int(m.group(1)) <= 100:
                return int(m.group(1))
            return max(0, bar)
        maxhp = self.maxhp.get(to_id(species))
        if not maxhp:
            return None
        m = re.search(r"(\d{1,3})\D{1,2}(\d{2,3})", hud.value)
        # the digits are exact when they read cleanly; the denominator (our max HP) checks the read
        if m and int(m.group(1)) <= maxhp and abs(int(m.group(1)) - hud.frac * maxhp) <= 0.06 * maxhp \
                and (int(m.group(2)) == maxhp or SequenceMatcher(None, m.group(2), str(maxhp)).ratio() >= 0.66):
            return int(m.group(1))
        return round(hud.frac * maxhp)

    # -- main ------------------------------------------------------------------------------------------------------
    def feed(self, r: Reading) -> list[dict]:
        """Events for one frame, in order. A decision is returned as {"type": "turn"} or {"type": "force_switch"}."""
        if self.finished:
            return []
        if r.message and norm("降参") in norm(r.message):  # forfeit dialog / notice: not a turn
            self.since_decision = 0
        out = self._message(r.message)
        out += self._hud("opp", r.opp)
        out += self._hud("me", r.me)
        dialog = bool(r.message) and norm("いいえ") in norm(r.message) and norm("はい") in norm(r.message)
        if r.decision and dialog:  # a yes/no dialog over the prompt (forfeit): no new decision
            return out
        if r.decision:
            self.decision_gap = 0
            both_out = self.sides["me"].active and self.sides["opp"].active
            if not self.in_decision and self.since_decision > 0 and both_out:
                out += self._flush_hp("opp", r.opp) + self._flush_hp("me", r.me)  # the prompt shows settled HP
                if self.force:
                    out.append({"type": "force_switch"})
                else:
                    out.append({"type": "turn"})
                    self.moved.clear()
                self.force = False
                self.since_decision = 0
            self.in_decision = True
        else:
            self.decision_gap += 1
            if self.decision_gap >= 2:
                self.in_decision = False
        return out

    def run(self, readings: Iterable[Reading]) -> list[dict]:
        return [ev for r in readings for ev in self.feed(r)]
