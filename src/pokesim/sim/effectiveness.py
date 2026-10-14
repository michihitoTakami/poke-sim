"""Ability/item-aware move effectiveness.

poke-env's `damage_multiplier` only considers types. Here the defender's ability and item
are taken into account: known when revealed, otherwise as an expectation over the species'
possible abilities weighted by the season's usage (article sets).
"""
import json
from dataclasses import dataclass
from functools import lru_cache

from poke_env.battle import Move, MoveCategory, Pokemon

from pokesim.config import PROCESSED_DIR

# ability -> (move type that is nullified)
TYPE_IMMUNITY_ABILITIES: dict[str, str] = {
    "levitate": "ground", "eartheater": "ground",
    "flashfire": "fire", "wellbakedbody": "fire",
    "waterabsorb": "water", "stormdrain": "water", "dryskin": "water",
    "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
    "sapsipper": "grass",
}
FLAG_IMMUNITY_ABILITIES: dict[str, str] = {"bulletproof": "bullet", "soundproof": "sound", "windrider": "wind"}
# ability -> {move type: multiplier}
TYPE_MODIFIER_ABILITIES: dict[str, dict[str, float]] = {
    "thickfat": {"fire": 0.5, "ice": 0.5}, "heatproof": {"fire": 0.5}, "waterbubble": {"fire": 0.5},
    "purifyingsalt": {"ghost": 0.5}, "dryskin": {"fire": 1.25}, "fluffy": {"fire": 2.0},
}
SUPER_EFFECTIVE_REDUCTION = {"filter": 0.75, "solidrock": 0.75, "prismarmor": 0.75}
FULL_HP_REDUCTION = {"multiscale": 0.5, "shadowshield": 0.5}
CATEGORY_REDUCTION = {"icescales": MoveCategory.SPECIAL, "furcoat": MoveCategory.PHYSICAL}
WONDER_GUARD = "wonderguard"
ATTACKER_IGNORES_ABILITIES = {"moldbreaker", "teravolt", "turboblaze", "myceliummight"}


@lru_cache(maxsize=1)
def _species_abilities() -> dict[str, list[str]]:
    dex = json.loads((PROCESSED_DIR / "dex.json").read_text())
    by_name = {a["name"]: a["id"] for a in dex["abilities"]}
    return {s["id"]: [by_name.get(a, a.lower().replace(" ", "")) for a in s["abilities"]] for s in dex["species"]}


@lru_cache(maxsize=4)
def ability_prior(season: str = "M-6") -> dict[str, dict[str, float]]:
    """species -> {ability: probability}, from article sets (falls back to uniform)."""
    path = PROCESSED_DIR / f"ability_prior_{season}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def ability_distribution(mon: Pokemon, season: str = "M-6") -> dict[str, float]:
    if mon.ability:
        return {mon.ability: 1.0}
    possible = _species_abilities().get(mon.species) or list(mon.possible_abilities or [])
    if not possible:
        return {}
    prior = ability_prior(season).get(mon.species, {})
    weights = {a: prior.get(a, 0.0) + 0.05 for a in possible}  # small floor: rare abilities stay possible
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


@dataclass
class Effectiveness:
    multiplier: float  # expected multiplier over the defender's possible abilities
    p_immune: float  # probability the move does nothing


# attacker abilities that turn Normal moves into another type (and power them up by 1.2)
TYPE_CHANGE = {"aerilate": "FLYING", "pixilate": "FAIRY", "refrigerate": "ICE", "galvanize": "ELECTRIC"}


def move_type(move: Move, attacker: Pokemon | None):
    """The type the move hits with: -ate abilities and Normalize change it, Liquid Voice makes sound moves Water."""
    from poke_env.battle import PokemonType

    ability = attacker.ability if attacker is not None else None
    if ability in TYPE_CHANGE and move.type == PokemonType.NORMAL:
        return PokemonType[TYPE_CHANGE[ability]]
    if ability == "normalize":
        return PokemonType.NORMAL
    if ability == "liquidvoice" and "sound" in move.flags:
        return PokemonType.WATER
    return move.type


def type_power(move: Move, attacker: Pokemon | None) -> float:
    """1.2 when an -ate ability or Normalize changed the move's type."""
    ability = attacker.ability if attacker is not None else None
    changed = (ability in TYPE_CHANGE and move.type.name == "NORMAL") or ability == "normalize"
    return 1.2 if changed else 1.0


def _ability_multiplier(ability: str, move: Move, defender: Pokemon, type_mult: float, mtype=None) -> float:
    move_type = (mtype or move.type).name.lower()
    if TYPE_IMMUNITY_ABILITIES.get(ability) == move_type:
        return 0.0
    if (flag := FLAG_IMMUNITY_ABILITIES.get(ability)) and flag in move.flags:
        return 0.0
    if ability == WONDER_GUARD and type_mult <= 1:
        return 0.0
    mult = type_mult * TYPE_MODIFIER_ABILITIES.get(ability, {}).get(move_type, 1.0)
    if type_mult > 1:
        mult *= SUPER_EFFECTIVE_REDUCTION.get(ability, 1.0)
    if defender.current_hp_fraction >= 1.0 or not defender.max_hp:  # unseen Pokémon are at full HP
        mult *= FULL_HP_REDUCTION.get(ability, 1.0)
    if CATEGORY_REDUCTION.get(ability) == move.category:
        mult *= 0.5
    return mult


# attacker abilities that let Normal / Fighting moves hit Ghost types (Mega Lopunny, Ursaluna-Bloodmoon)
GHOST_HITTERS = {"scrappy", "mindseye"}


def _scrappy_share(attacker: Pokemon | None, season: str) -> float:
    if attacker is None:
        return 0.0
    if attacker.ability:
        return float(attacker.ability in GHOST_HITTERS)
    return sum(p for a, p in ability_distribution(attacker, season).items() if a in GHOST_HITTERS)


def _ghost_hit_multiplier(mtype, defender: Pokemon) -> float:
    """The type multiplier with the Ghost type's immunity to Normal / Fighting ignored."""
    from poke_env.battle import PokemonType
    from poke_env.data import GenData

    types = [t if t != PokemonType.GHOST else None for t in defender.types]
    types += [None] * (2 - len(types))
    return mtype.damage_multiplier(types[0], types[1], type_chart=GenData.from_gen(9).type_chart)


def effectiveness(move: Move, attacker: Pokemon | None, defender: Pokemon, season: str = "M-6") -> Effectiveness:
    mtype = move_type(move, attacker)
    type_mult = defender.damage_multiplier(mtype if move.category != MoveCategory.STATUS else move)
    if type_mult == 0 and mtype.name in ("NORMAL", "FIGHTING") and move.category != MoveCategory.STATUS \
            and any(t is not None and t.name == "GHOST" for t in defender.types):
        share = _scrappy_share(attacker, season)
        if share > 0:  # Scrappy / Mind's Eye: the Ghost immunity does not apply
            hit = _ghost_hit_multiplier(mtype, defender)
            if share >= 1:
                type_mult = hit
            else:
                return Effectiveness(share * hit, 1 - share)
    if move.category == MoveCategory.STATUS:
        return Effectiveness(type_mult, float(type_mult == 0))
    if mtype.name.lower() == "ground" and defender.item == "airballoon":
        return Effectiveness(0.0, 1.0)
    if attacker is not None and attacker.ability in ATTACKER_IGNORES_ABILITIES:
        return Effectiveness(type_mult, float(type_mult == 0))
    dist = ability_distribution(defender, season)
    if not dist:
        return Effectiveness(type_mult, float(type_mult == 0))
    expected = p_immune = 0.0
    for ability, p in dist.items():
        m = _ability_multiplier(ability, move, defender, type_mult, mtype)
        expected += p * m
        p_immune += p * (m == 0)
    return Effectiveness(expected, p_immune)


def type_vs_defender(move_type: str, defender: Pokemon, season: str = "M-6") -> float:
    """Expected multiplier of an attacking type (no specific move) against the defender."""
    from poke_env.battle import PokemonType
    ptype = PokemonType.from_name(move_type)
    mult = defender.damage_multiplier(ptype)
    if move_type == "ground" and defender.item == "airballoon":
        return 0.0
    dist = ability_distribution(defender, season)
    if not dist:
        return mult
    return sum(p * (0.0 if TYPE_IMMUNITY_ABILITIES.get(a) == move_type else
                    mult * TYPE_MODIFIER_ABILITIES.get(a, {}).get(move_type, 1.0)) for a, p in dist.items())
