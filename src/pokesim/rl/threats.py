"""Threats the opponent has not shown yet: its likely moves (usage data) and its Mega Evolution.

The base observation (pokesim.rl.features.embed) scores the opponent's threat from revealed moves plus assumed
STAB moves of its current forme, so a Delphox that will Mega Evolve and use Flamethrower on turn 1 looks harmless.
ThreatFeatures adds, for the opposing active Pokémon over its possible sets (revealed moves for sure, other
damaging moves with their usage share, the Mega forme with its stone's share while it can still Mega Evolve):
- the chance one of its moves knocks out our active Pokémon, and its expected best damage to it;
- the chance it moves first, and the chance it still Mega Evolves;
- for each of our 6: the chance it knocks that member out (e.g. on the switch-in);
plus, per kind of set-up counter (Encore, phazing, Taunt, Haze, Yawn), the chance the opponent has one that works on
our active Pokémon (Good as Gold / Magic Bounce block status moves aimed at it, a Substitute blocks Yawn and Clear
Smog; Haze and Dragon Tail still work), whether its first hit is absorbed (Disguise still up; Sturdy, Multiscale,
Shadow Shield at full HP), and volatile effects the base observation leaves out, for both active Pokémon (Curse and Perish Song, which Baton
Pass hands on, Aqua Ring, Ingrain, Focus Energy, Magnet Rise, No Retreat, trapping; ours also Heal Block, Torment,
Disable).

  uv run python -m pokesim.rl.threats      # build data/processed/threat_priors.json from the set data
"""
import json
from functools import lru_cache

import numpy as np
from poke_env.battle import AbstractBattle, Field, Move, MoveCategory, Pokemon

from pokesim.config import PROCESSED_DIR
from pokesim.rl.items import UNKNOWN, is_mega
from pokesim.rl.myteam import COUNTER_GROUPS, to_id
from pokesim.sim import damage
from pokesim.sim.formes import current_forme

PRIORS_PATH = PROCESSED_DIR / "threat_priors.json"
SHARED_EFFECTS = ["CURSE", "AQUA_RING", "INGRAIN", "FOCUS_ENERGY", "MAGNET_RISE", "NO_RETREAT"]
OWN_EFFECTS = ["HEAL_BLOCK", "TORMENT", "DISABLE"]
PERISH = {"PERISH0": 0, "PERISH1": 1, "PERISH2": 2, "PERISH3": 3}
TRAPPED = {"TRAPPED", "PARTIALLY_TRAPPED", "OCTOLOCK"}
N_OWN = len(SHARED_EFFECTS) + len(OWN_EFFECTS) + 3
N_OPP = len(SHARED_EFFECTS) + 3
STATUS_BLOCKERS = {"goodasgold", "magicbounce"}
SUB_BLOCKED = {"yawn", "clearsmog"}
N_COUNTER = len(COUNTER_GROUPS) + 1
SHIELD_ABILITIES = ["sturdy", "multiscale", "shadowshield"]  # survive / halve a hit at full HP
N_SHIELD = 1 + len(SHIELD_ABILITIES)
SIZE = 4 + 6 + N_OWN + N_OPP + N_COUNTER + N_SHIELD
MAX_MOVES = 10  # likely damaging moves kept per species


@lru_cache(maxsize=None)
def any_move(mid: str) -> Move:
    return Move(mid, gen=9)


@lru_cache(maxsize=None)
def move(mid: str) -> Move | None:
    try:
        m = Move(mid, gen=9)
        return m if m.category != MoveCategory.STATUS and (m.base_power or m.damage) else None
    except Exception:  # noqa: BLE001 - unknown move id
        return None


def build_priors() -> dict[str, dict]:
    """species -> {"moves": {move: share of sets carrying it}, "mega": {mega species id: share}}."""
    from pokesim.search.lookahead import SetSampler

    sampler = SetSampler(["M-6"])
    out = {}
    for sid, sm in sampler.summary.items():
        total = sum(v for k, v in sm.moves.items() if k)
        moves = {k: min(1.0, 4 * v / total) for k, v in sm.moves.items() if k and total and move(k)}
        moves = dict(sorted(moves.items(), key=lambda kv: -kv[1] * (move(kv[0]).base_power or 60))[:MAX_MOVES])
        items = sum(v for k, v in sm.items.items() if k) or 1.0
        mega = {}
        for k, v in sm.items.items():
            it = sampler.items.get(k)
            if it is not None and it.mega_stone:
                mega[to_id(it.mega_stone)] = mega.get(to_id(it.mega_stone), 0.0) + v / items
        counters = {k: min(1.0, 4 * v / total) for k, v in sm.moves.items()
                    if total and any(k in g for g in COUNTER_GROUPS.values())}
        ab_total = sum(v for k, v in sm.abilities.items() if k) or 1.0
        shields = {k: sm.abilities.get(k, 0) / ab_total for k in SHIELD_ABILITIES}
        out[sid] = {"moves": {k: round(v, 3) for k, v in moves.items()},
                    "counter": {k: round(v, 3) for k, v in counters.items()},
                    "shield": {k: round(v, 3) for k, v in shields.items() if v > 0},
                    "mega": {k: round(v, 3) for k, v in mega.items() if v >= 0.01}}
    return out


def load_priors() -> dict[str, dict]:
    return json.loads(PRIORS_PATH.read_text()) if PRIORS_PATH.exists() else {}


@lru_cache(maxsize=None)
def forme(species_id: str) -> Pokemon | None:
    try:
        return Pokemon(gen=9, species=species_id)
    except Exception:  # noqa: BLE001 - not in the pokedex
        return None


def expected_best(dmg: np.ndarray, q: np.ndarray) -> float:
    """E[max damage] when move i is carried independently with probability q[i]."""
    order = np.argsort(-dmg)
    out, none_better = 0.0, 1.0
    for i in order:
        out += dmg[i] * q[i] * none_better
        none_better *= 1 - q[i]
    return out


class ThreatFeatures:
    size = SIZE

    def __init__(self, priors: dict[str, dict] | None = None):
        self.priors = load_priors() if priors is None else priors

    def candidates(self, mon: Pokemon) -> tuple[list[Move], np.ndarray]:
        revealed = [m for m in mon.moves.values() if m.category != MoveCategory.STATUS]
        moves, q = list(revealed), [1.0] * len(revealed)
        if len(mon.moves) < 4:
            prior = self.priors.get(to_id(mon.base_species), {}) or self.priors.get(to_id(mon.species), {})
            known = {m.id for m in mon.moves.values()}
            for mid, share in prior.get("moves", {}).items():
                if mid not in known and (m := move(mid)) is not None:
                    moves.append(m)
                    q.append(share)
        return moves, np.array(q)

    def formes(self, mon: Pokemon, battle: AbstractBattle) -> list[tuple[Pokemon, float]]:
        """The opponent's current forme and, while it can still Mega Evolve, its Mega forme(s) with their chance."""
        if is_mega(mon) or any(is_mega(m) for m in battle.opponent_team.values()):
            return [(mon, 1.0)]
        if mon.item not in (UNKNOWN,) and not (mon.item or "").endswith(("ite", "itex", "itey")):
            return [(mon, 1.0)]  # known non-stone item (or none left)
        prior = self.priors.get(to_id(mon.base_species), {}) or self.priors.get(to_id(mon.species), {})
        out, rest = [], 1.0
        for sid, p in prior.get("mega", {}).items():
            mega = forme(sid)
            if mega is not None and p > 0:
                mega._boosts = dict(mon.boosts)
                mega._status = mon.status
                mega._item = None
                out.append((mega, p))
                rest -= p
        return [(mon, max(rest, 0.0))] + out

    @staticmethod
    def effects(mon: Pokemon | None, own: bool) -> list[float]:
        names = {e.name for e in mon.effects} if mon is not None and not mon.fainted else set()
        perish = [PERISH[n] for n in names if n in PERISH]
        out = [float(n in names) for n in SHARED_EFFECTS + (OWN_EFFECTS if own else [])]
        return out + [float(bool(perish)), (min(perish) / 3.0) if perish else 0.0, float(bool(names & TRAPPED))]

    @staticmethod
    def blocked(mid: str, target: Pokemon) -> bool:
        """Whether the move cannot affect `target` (status move into Good as Gold / Magic Bounce, Yawn into a
        Substitute)."""
        m = any_move(mid)
        aimed = m.target is not None and m.target.name in ("NORMAL", "ANY", "ADJACENT_FOE", "RANDOM_NORMAL")
        if m.category == MoveCategory.STATUS and aimed and to_id(target.ability or "") in STATUS_BLOCKERS:
            return True
        return mid in SUB_BLOCKED and any(e.name == "SUBSTITUTE" for e in target.effects)

    def counters(self, opp: Pokemon, me: Pokemon) -> list[float]:
        """Per counter group: the chance the opponent has a move of it that works on `me`; then whether `me`
        blocks status moves."""
        prior = self.priors.get(to_id(opp.base_species), {}) or self.priors.get(to_id(opp.species), {})
        known = set(opp.moves)
        out = []
        for moves in COUNTER_GROUPS.values():
            usable = [m for m in moves if not self.blocked(m, me)]
            if known & set(usable):
                out.append(1.0)
            elif len(known) >= 4:
                out.append(0.0)
            else:
                share = prior.get("counter", {})
                out.append(1.0 - float(np.prod([1 - share.get(m, 0.0) for m in usable])))
        out.append(float(to_id(me.ability or "") in STATUS_BLOCKERS))
        return out

    def shields(self, opp: Pokemon) -> list[float]:
        """Disguise still up; chance of Sturdy / Multiscale / Shadow Shield working now (at full HP)."""
        disguise = float(current_forme(opp) in ("mimikyu", "mimikyutotem"))
        full = opp.current_hp_fraction >= 1.0
        if opp.ability:
            ab = to_id(opp.ability)
            return [disguise] + [float(full and ab == k) for k in SHIELD_ABILITIES]
        prior = self.priors.get(to_id(opp.base_species), {}).get("shield", {})
        return [disguise] + [float(full) * prior.get(k, 0.0) for k in SHIELD_ABILITIES]

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        feats = np.zeros(self.size, dtype=np.float32)
        if battle.opponent_active_pokemon is not None:
            feats[-N_SHIELD:] = self.shields(battle.opponent_active_pokemon)
        n = 10 + N_OWN + N_OPP
        feats[10:10 + N_OWN] = self.effects(battle.active_pokemon, True)
        feats[10 + N_OWN:n] = self.effects(battle.opponent_active_pokemon, False)
        if battle.active_pokemon is not None and battle.opponent_active_pokemon is not None \
                and not battle.active_pokemon.fainted:
            feats[n:n + N_COUNTER] = self.counters(battle.opponent_active_pokemon, battle.active_pokemon)
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is not None and me.fainted:  # forced switch: only the switch-in chances matter
            me = None
        if opp is None:
            return feats
        moves, q = self.candidates(opp)
        formes = self.formes(opp, battle)
        feats[3] = sum(p for f, p in formes if f is not opp)
        team = list(battle.team.values())[:6]
        trick_room = Field.TRICK_ROOM in battle.fields
        for f, p in formes:
            if p <= 0:
                continue
            if me is not None and moves:
                dmg = np.array([damage.calc(m, f, me, attacker_own=False, weather=battle.weather).mean
                                for m in moves])
                hp = max(me.current_hp_fraction, 1e-3)
                feats[0] += p * (1 - np.prod(1 - q * (dmg >= hp)))
                feats[1] += p * min(expected_best(dmg, q) / hp, 2.0)
            if me is not None:
                faster = damage.speed(f, False) > damage.speed(me, True)
                feats[2] += p * float(faster != trick_room)
            for slot, mon in enumerate(team):
                if mon.fainted or not moves:
                    continue
                if mon is me:
                    feats[4 + slot] = feats[0]
                    continue
                saved = dict(mon.boosts)
                mon._boosts.clear()  # boosts do not carry over a switch
                dmg = np.array([damage.calc(m, f, mon, attacker_own=False, weather=battle.weather).mean
                                for m in moves])
                mon._boosts.update(saved)
                hp = max(mon.current_hp_fraction, 1e-3)
                feats[4 + slot] += p * (1 - np.prod(1 - q * (dmg >= hp)))
        return feats


def main() -> None:
    priors = build_priors()
    PRIORS_PATH.write_text(json.dumps(priors, ensure_ascii=False, indent=0))
    megas = sum(bool(v["mega"]) for v in priors.values())
    print(f"wrote {len(priors)} species ({megas} with a Mega Evolution) to {PRIORS_PATH}")
    for sid in ("delphox", "garchomp", "kleavor", "pelipper", "primarina", "sneasler"):
        if sid in priors:
            print(sid, priors[sid])


if __name__ == "__main__":
    main()
