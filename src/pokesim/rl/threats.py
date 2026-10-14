"""Threats the opponent has not shown yet: its likely moves (usage data) and its Mega Evolution.

The base observation (pokesim.rl.features.embed) scores the opponent's threat from revealed moves plus assumed
STAB moves of its current forme, so a Delphox that will Mega Evolve and use Flamethrower on turn 1 looks harmless.
ThreatFeatures adds, for the opposing active Pokémon over its possible sets (revealed moves for sure, other
damaging moves with their usage share, the Mega forme with its stone's share while it can still Mega Evolve):
- the chance one of its moves knocks out our active Pokémon, and its expected best damage to it;
- the chance it moves first, and the chance it still Mega Evolves;
- for each of our 6: the chance it knocks that member out (e.g. on the switch-in).

  uv run python -m pokesim.rl.threats      # build data/processed/threat_priors.json from the set data
"""
import json
from functools import lru_cache

import numpy as np
from poke_env.battle import AbstractBattle, Field, Move, MoveCategory, Pokemon

from pokesim.config import PROCESSED_DIR
from pokesim.rl.items import UNKNOWN, is_mega
from pokesim.rl.myteam import to_id
from pokesim.sim import damage

PRIORS_PATH = PROCESSED_DIR / "threat_priors.json"
SIZE = 4 + 6
MAX_MOVES = 10  # likely damaging moves kept per species


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
        out[sid] = {"moves": {k: round(v, 3) for k, v in moves.items()},
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

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        feats = np.zeros(self.size, dtype=np.float32)
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
