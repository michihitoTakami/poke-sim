"""Held items: per-species item priors from the set data, and observation features that account for them.

The base observation (pokesim.rl.features.embed) only knows items once they are revealed, so an unrevealed Choice
Scarf or Focus Sash is treated as no item at all. ItemFeatures adds, for the opposing active Pokémon:
- the chance of each kind of item that changes the decision (ITEM_CLASSES), from the set data (published team
  write-ups, Smogon usage, the opponents we faced; pokesim.search.lookahead.SetSampler) or 1/0 once revealed;
- the chance we move first (Choice Scarf), that our best move knocks it out (Focus Sash at full HP) and that its
  best move knocks us out (Life Orb, type-boosting items);
plus the worst case of Scarf / Sash over its other members, and whether each of our 6 still holds its item
(Air Balloon popped, Mental Herb / Sitrus Berry used).

  uv run python -m pokesim.rl.items          # build data/processed/item_priors.json and print the item tendencies
"""
import json
from collections import Counter

import numpy as np
from poke_env.battle import AbstractBattle, Field, Pokemon

from pokesim.config import PROCESSED_DIR
from pokesim.rl.features import best_damage, threat
from pokesim.rl.myteam import to_id
from pokesim.sim import damage

PRIORS_PATH = PROCESSED_DIR / "item_priors.json"
UNKNOWN = "unknown_item"
RECOVERY = {"leftovers", "sitrusberry", "blacksludge", "shellbell", "figyberry", "wikiberry", "magoberry",
            "aguavberry", "iapapaberry"}
RESIST_BERRIES = {"occaberry", "passhoberry", "wacanberry", "rindoberry", "yacheberry", "chopleberry", "kebiaberry",
                  "shucaberry", "cobaberry", "payapaberry", "tangaberry", "chartiberry", "kasibberry", "habanberry",
                  "colburberry", "babiriberry", "roseliberry", "chilanberry"}
TYPE_BOOST = {"blackglasses", "spelltag", "mysticwater", "miracleseed", "charcoal", "magnet", "nevermeltice",
              "softsand", "sharpbeak", "poisonbarb", "silkscarf", "silverpowder", "hardstone", "twistedspoon",
              "dragonfang", "blackbelt", "metalcoat", "fairyfeather", "expertbelt"}
STATUS_BERRIES = {"lumberry", "chestoberry", "persimberry", "rawstberry", "cheriberry", "pechaberry", "aspearberry"}
HERBS = {"mentalherb", "whiteherb", "powerherb", "mirrorherb"}
# no Choice Band / Specs or Assault Vest in the Champions item list
ITEM_CLASSES = ["choicescarf", "focussash", "lifeorb", "typeboost", "megastone", "recovery", "resistberry",
                "statusberry", "herb", "airballoon", "rockyhelmet", "other"]
SIZE = len(ITEM_CLASSES) + 1 + 3 + 2 + 6
DAMAGE_MULT = {"lifeorb": 1.3, "typeboost": 1.2}  # on the opponent's best move (type boost: assumed to match)


def item_class(item: str, mega_stones: set[str]) -> str:
    if item in ITEM_CLASSES:
        return item
    if item in mega_stones or item.endswith("ite") or item.endswith("itex") or item.endswith("itey"):
        return "megastone"
    for cls, members in (("recovery", RECOVERY), ("resistberry", RESIST_BERRIES), ("typeboost", TYPE_BOOST),
                         ("statusberry", STATUS_BERRIES), ("herb", HERBS)):
        if item in members:
            return cls
    return "other"


def build_priors(sampler=None) -> tuple[dict[str, dict[str, float]], Counter]:
    """Per species: share of each item class (species with any item data), and the overall class shares."""
    from pokesim.search.lookahead import SetSampler

    sampler = SetSampler(["M-6"]) if sampler is None else sampler
    stones = {i for i, it in sampler.items.items() if it.mega_stone}
    out, overall = {}, Counter()
    for sid, sm in sampler.summary.items():
        total = sum(v for k, v in sm.items.items() if k)
        if not total:
            continue
        shares = Counter()
        for k, v in sm.items.items():
            if k:
                shares[item_class(k, stones)] += v / total
        out[sid] = {k: round(v, 4) for k, v in shares.items()}
        for k, v in shares.items():
            overall[k] += v * max(sm.teams, 1)
    return out, overall


def load_priors() -> dict[str, dict[str, float]]:
    return json.loads(PRIORS_PATH.read_text()) if PRIORS_PATH.exists() else {}


def is_mega(mon: Pokemon) -> bool:
    return "mega" in to_id(mon.species)[len(to_id(mon.base_species)):] or to_id(mon.species).endswith("mega")


class ItemFeatures:
    size = SIZE

    def __init__(self, priors: dict[str, dict[str, float]] | None = None):
        self.priors = load_priors() if priors is None else priors
        self.default = self._overall()

    def _overall(self) -> dict[str, float]:
        acc = Counter()
        for shares in self.priors.values():
            acc.update(shares)
        total = sum(acc.values()) or 1.0
        return {k: v / total for k, v in acc.items()}

    def probs(self, mon: Pokemon, battle: AbstractBattle) -> tuple[np.ndarray, bool]:
        """Item-class distribution for an opposing Pokémon and whether its item is known."""
        out = np.zeros(len(ITEM_CLASSES), dtype=np.float32)
        item = mon.item
        if is_mega(mon):
            out[ITEM_CLASSES.index("megastone")] = 1.0
            return out, True
        if item is None or item == "":  # consumed or knocked off
            return out, True
        if item != UNKNOWN:
            out[ITEM_CLASSES.index(item_class(item, set()))] = 1.0
            return out, True
        shares = dict(self.priors.get(to_id(mon.base_species)) or self.priors.get(to_id(mon.species))
                      or self.default)
        if any(is_mega(m) for m in battle.opponent_team.values()):
            shares.pop("megastone", None)  # one Mega Evolution per battle: its stone no longer matters
        total = sum(shares.values())
        for k, v in shares.items():
            out[ITEM_CLASSES.index(k)] = v / total if total else 0.0
        return out, False

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        feats = np.zeros(self.size, dtype=np.float32)
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        n = len(ITEM_CLASSES)
        if opp is not None:
            p, known = self.probs(opp, battle)
            feats[:n], feats[n] = p, float(known)
            if me is not None:
                feats[n + 1:n + 4] = self._chances(battle, me, opp, dict(zip(ITEM_CLASSES, p)))
        others = [m for m in battle.opponent_team.values() if not m.fainted and m is not opp]
        if others:
            ps = [self.probs(m, battle)[0] for m in others]
            feats[n + 4] = max(q[ITEM_CLASSES.index("choicescarf")] for q in ps)
            feats[n + 5] = max(q[ITEM_CLASSES.index("focussash")] * float(m.current_hp_fraction >= 1.0)
                               for q, m in zip(ps, others))
        for slot, mon in enumerate(list(battle.team.values())[:6]):
            feats[n + 6 + slot] = float(bool(mon.item) and mon.item != UNKNOWN and not mon.fainted)
        return feats

    @staticmethod
    def _with_item(mon: Pokemon, item: str | None, fn):
        saved = mon._item
        mon._item = item
        try:
            return fn()
        finally:
            mon._item = saved

    def _chances(self, battle: AbstractBattle, me: Pokemon, opp: Pokemon, p: dict[str, float]) -> list[float]:
        """P(we move first) over a Choice Scarf, P(our best move knocks it out) over a Focus Sash at full HP, and
        P(its best move knocks us out) over Life Orb / type-boosting items."""
        none = None if opp.item == UNKNOWN else opp.item
        trick_room = Field.TRICK_ROOM in battle.fields
        my_speed = damage.speed(me, True)

        def faster(item):
            return float((my_speed > self._with_item(opp, item, lambda: damage.speed(opp, False))) != trick_room)

        p_scarf = p["choicescarf"]
        p_first = (1 - p_scarf) * faster(none) + p_scarf * faster("choicescarf")
        p_kill = float(best_damage(me, opp, True, battle.weather) >= max(opp.current_hp_fraction, 1e-3))
        if opp.current_hp_fraction >= 1.0:
            p_kill *= 1 - p["focussash"]
        dmg, hp = threat(opp, me, battle.weather), max(me.current_hp_fraction, 1e-3)
        p_ko = (1 - sum(p[c] for c in DAMAGE_MULT)) * float(dmg >= hp)
        p_ko += sum(p[c] * float(dmg * m >= hp) for c, m in DAMAGE_MULT.items())
        return [p_first, p_kill, p_ko]


def main() -> None:
    priors, overall = build_priors()
    PRIORS_PATH.write_text(json.dumps(priors, ensure_ascii=False, indent=0))
    total = sum(overall.values())
    print(f"wrote {len(priors)} species to {PRIORS_PATH}")
    print("item classes, weighted by team count:")
    for k, v in overall.most_common():
        print(f"  {k:12s} {v / total:6.1%}")
    for cls in ("choicescarf", "focussash", "lifeorb", "typeboost", "statusberry", "herb"):
        top = sorted(((sid, s.get(cls, 0)) for sid, s in priors.items()), key=lambda kv: -kv[1])[:8]
        print(f"{cls}: " + ", ".join(f"{sid} {v:.0%}" for sid, v in top if v > 0))


if __name__ == "__main__":
    main()
