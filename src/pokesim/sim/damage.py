"""Fast approximate damage calculation for battle features (Champions, level 50).

Own Pokémon use their real stats (from the battle request). Opponents' stats are estimated
from the season's common spread/nature (article sets) or an offensive default.
Champions stats: HP = base + points + 75, others = (base + points + 20) * nature.
"""
import json
from dataclasses import dataclass
from functools import lru_cache

from poke_env.battle import Move, MoveCategory, Pokemon, PokemonType, Status, Weather

from pokesim.config import PROCESSED_DIR
from pokesim.sim.effectiveness import effectiveness, move_type, type_power

LEVEL = 50
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
NATURES = {  # nature -> (plus, minus)
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}
USES_TARGET_DEF = {"psyshock", "psystrike", "secretsword"}
ROLL_MEAN, ROLL_MIN = 0.925, 0.85


@lru_cache(maxsize=4)
def spread_prior(season: str = "M-6") -> dict[str, dict]:
    """species -> {"points": [6 ints], "nature": str} (most common in article sets)."""
    path = PROCESSED_DIR / f"spread_prior_{season}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def boost_multiplier(stage: int) -> float:
    stage = max(-6, min(6, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def estimate_stats(mon: Pokemon, own: bool, season: str = "M-6") -> dict[str, float]:
    if own and mon.stats and all(mon.stats.get(s) for s in STATS[1:]):
        return {"hp": float(mon.max_hp or 1), **{s: float(mon.stats[s]) for s in STATS[1:]}}
    base = mon.base_stats
    prior = spread_prior(season).get(mon.species)
    if prior:
        points, nature = prior["points"], prior["nature"]
    else:  # default: max the better attacking stat and speed
        offense = "atk" if base["atk"] >= base["spa"] else "spa"
        points = [2, 32 * (offense == "atk"), 0, 32 * (offense == "spa"), 0, 32]
        nature = "adamant" if offense == "atk" else "modest"
    plus, minus = NATURES.get(nature, (None, None))
    out = {"hp": float(base["hp"] + points[0] + 75)}
    for stat, pts in zip(STATS[1:], points[1:]):
        value = base[stat] + pts + 20
        out[stat] = float(int(value * (1.1 if stat == plus else 0.9 if stat == minus else 1.0)))
    return out


@dataclass
class Damage:
    mean: float  # expected damage as a fraction of the defender's max HP
    minimum: float  # min roll, fraction of max HP


def calc(move: Move, attacker: Pokemon, defender: Pokemon, *, attacker_own: bool, weather: dict | None = None,
         attacker_boosts: dict[str, int] | None = None, season: str = "M-6") -> Damage:
    if move.category == MoveCategory.STATUS:
        return Damage(0.0, 0.0)
    a_stats = estimate_stats(attacker, attacker_own, season)
    d_stats = estimate_stats(defender, not attacker_own, season)
    if move.damage:  # fixed damage (Seismic Toss = level, Night Shade, ...)
        fixed = LEVEL if move.damage == "level" else float(move.damage) if isinstance(move.damage, int) else 0.0
        frac = fixed / d_stats["hp"] * (effectiveness(move, attacker, defender).multiplier > 0)
        return Damage(frac, frac)
    boosts = attacker_boosts if attacker_boosts is not None else attacker.boosts
    physical = move.category == MoveCategory.PHYSICAL
    if move.id == "bodypress":
        atk_stat, atk_boost = a_stats["def"], boosts.get("def", 0)
    elif move.id == "foulplay":
        atk_stat, atk_boost = estimate_stats(defender, not attacker_own, season)["atk"], defender.boosts.get("atk", 0)
    else:
        atk_key = "atk" if physical else "spa"
        atk_stat, atk_boost = a_stats[atk_key], boosts.get(atk_key, 0)
    def_key = "def" if physical or move.id in USES_TARGET_DEF else "spd"
    attack = atk_stat * boost_multiplier(atk_boost)
    defense = d_stats[def_key] * boost_multiplier(defender.boosts.get(def_key, 0))

    base = ((2 * LEVEL / 5 + 2) * move.base_power * attack / max(defense, 1)) / 50 + 2
    mod = effectiveness(move, attacker, defender, season).multiplier
    mtype = move_type(move, attacker)
    mod *= type_power(move, attacker)
    if mtype in attacker.types:
        mod *= 2.0 if attacker.ability == "adaptability" else 1.5
    if physical and attacker.status == Status.BRN and attacker.ability != "guts":
        mod *= 0.5
    weather = weather or {}
    if Weather.SUNNYDAY in weather:
        mod *= {PokemonType.FIRE: 1.5, PokemonType.WATER: 0.5}.get(mtype, 1.0)
    elif Weather.RAINDANCE in weather:
        mod *= {PokemonType.WATER: 1.5, PokemonType.FIRE: 0.5}.get(mtype, 1.0)
    item = attacker.item or ""
    mod *= {"choiceband": 1.5 if physical else 1.0, "choicespecs": 1.0 if physical else 1.5,
            "lifeorb": 1.3}.get(item, 1.0)
    if item == "expertbelt" and defender.damage_multiplier(move) > 1:
        mod *= 1.2
    if defender.item == "assaultvest" and not physical:
        mod *= 2 / 3
    mod *= move.expected_hits
    dmg = base * mod
    return Damage(dmg * ROLL_MEAN / d_stats["hp"], dmg * ROLL_MIN / d_stats["hp"])


def speed(mon: Pokemon, own: bool, season: str = "M-6") -> float:
    spe = estimate_stats(mon, own, season)["spe"] * boost_multiplier(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        spe *= 0.5
    if mon.item == "choicescarf":
        spe *= 1.5
    return spe
