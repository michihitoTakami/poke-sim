"""Battle state -> fixed-size feature vector for the policy.

Per move: expected damage / KO chance (pokesim.sim.damage), ability-aware effectiveness, and
what a status move does (stat changes, healing, Substitute/Protect, pivoting, status, hazards)
together with how much setting up would raise our damage or cut the opponent's.
Board: weather/terrain, hazards and screens, volatiles (Substitute, Yawn, Taunt, ...), speed order,
and the estimated threat each side poses.
"""
import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

import pokesim.sim.formes  # noqa: F401 - remember forme changes
from pokesim.sim import damage
from pokesim.sim.effectiveness import effectiveness, type_vs_defender

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
STATUSES = list(Status)
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT_BOOSTS = ["atk", "def", "spa", "spd", "spe"]
WEATHERS = [Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOWSCAPE]
FIELDS = [Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.PSYCHIC_TERRAIN, Field.MISTY_TERRAIN,
          Field.TRICK_ROOM]
SIDE_CONDITIONS = [SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES,
                   SideCondition.STICKY_WEB, SideCondition.REFLECT, SideCondition.LIGHT_SCREEN,
                   SideCondition.AURORA_VEIL, SideCondition.TAILWIND]
SIDE_MAX = {SideCondition.SPIKES: 3, SideCondition.TOXIC_SPIKES: 2}
VOLATILES = [Effect.SUBSTITUTE, Effect.YAWN, Effect.TAUNT, Effect.ENCORE, Effect.LEECH_SEED, Effect.CONFUSION,
             Effect.SALT_CURE]
HAZARDS = {SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB}
SCREENS = {SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL}
# Effects poke-env's move data does not express directly.
SPECIAL_MOVES: dict[str, dict] = {
    "wish": {"heal": 0.5}, "rest": {"heal": 1.0}, "painsplit": {"heal": 0.25},
    "bellydrum": {"self_boosts": {"atk": 6}, "cost": 0.5},
    "curse": {"self_boosts": {"atk": 1, "def": 1, "spe": -1}},  # non-Ghost Curse
    "filletaway": {"self_boosts": {"atk": 2, "spa": 2, "spe": 2}, "cost": 0.5},
}
ASSUMED_BP = 90  # unrevealed opponent moves: a STAB move of this power
SELF_TARGETS = {"SELF", "ADJACENT_ALLY_OR_SELF", "ALLIES", "ALLY_SIDE", "ALLY_TEAM"}


class AssumedMove(Move):
    """A generic STAB attack of a given type/category, standing in for unrevealed opponent moves."""

    def __init__(self, ptype: PokemonType, physical: bool):
        super().__init__("tackle" if physical else "swift", gen=9)
        self._base_power_override = ASSUMED_BP
        self._assumed_type = ptype

    @property
    def type(self) -> PokemonType:
        return self._assumed_type

    @property
    def expected_hits(self) -> float:
        return 1.0


_ASSUMED: dict[tuple, AssumedMove] = {}


def assumed_move(ptype: PokemonType, physical: bool) -> AssumedMove:
    key = (ptype, physical)
    if key not in _ASSUMED:
        _ASSUMED[key] = AssumedMove(ptype, physical)
    return _ASSUMED[key]

MOVE_FEATURES = 34


def _types(mon: Pokemon | None) -> list[float]:
    present = set(mon.types) if mon else set()
    return [float(t in present) for t in TYPES]


def _status(mon: Pokemon | None) -> list[float]:
    return [float(mon is not None and mon.status == s) for s in STATUSES]


def _boosts(mon: Pokemon | None) -> list[float]:
    return [(mon.boosts.get(b, 0) / 6.0) if mon else 0.0 for b in BOOSTS]


def _log_mult(mult: float) -> float:
    return np.log2(mult) / 2.0 if mult > 0 else -1.5


def _log_ratio(new: float, old: float) -> float:
    return float(np.clip(np.log2((new + 1e-3) / (old + 1e-3)), -2, 2))


def self_boosts(move: Move, user: Pokemon) -> dict[str, int]:
    if move.id == "curse" and PokemonType.GHOST in user.types:
        return {}
    if move.id in SPECIAL_MOVES and "self_boosts" in SPECIAL_MOVES[move.id]:
        return SPECIAL_MOVES[move.id]["self_boosts"]
    boosts = dict(move.self_boost or {})
    if move.boosts and move.target and move.target.name in SELF_TARGETS:
        for k, v in move.boosts.items():
            boosts[k] = boosts.get(k, 0) + v
    return boosts


def foe_boosts(move: Move) -> dict[str, int]:
    if move.boosts and move.target and move.target.name not in SELF_TARGETS:
        return dict(move.boosts)
    return {}


def best_damage(attacker: Pokemon, defender: Pokemon, attacker_own: bool, weather: dict,
                boosts: dict[str, int] | None = None) -> float:
    """Best expected damage (fraction of defender's max HP) among the attacker's known moves."""
    moves = [m for m in attacker.moves.values() if m.category != MoveCategory.STATUS]
    return max((damage.calc(m, attacker, defender, attacker_own=attacker_own, weather=weather,
                            attacker_boosts=boosts).mean for m in moves), default=0.0)


def threat(opp: Pokemon, me: Pokemon, weather: dict, my_boosts: dict[str, int] | None = None) -> float:
    """Opponent's best expected damage to `me` (fraction of my max HP): revealed moves + assumed STABs."""
    if my_boosts is not None:
        saved = dict(me.boosts)
        me._boosts.update(my_boosts)
    best = best_damage(opp, me, attacker_own=False, weather=weather)
    stats = damage.estimate_stats(opp, own=False)
    physical = stats["atk"] >= stats["spa"]
    for ptype in opp.types:
        if ptype is not None:
            best = max(best, damage.calc(assumed_move(ptype, physical), opp, me, attacker_own=False,
                                         weather=weather).mean)
    if my_boosts is not None:
        me._boosts.clear()
        me._boosts.update(saved)
    return best


def move_features(move: Move | None, me: Pokemon | None, opp: Pokemon | None, battle: AbstractBattle,
                  context: dict) -> list[float]:
    if move is None or me is None or opp is None:
        return [0.0] * MOVE_FEATURES
    eff = effectiveness(move, me, opp)
    dmg = damage.calc(move, me, opp, attacker_own=True, weather=battle.weather)
    opp_hp = max(opp.current_hp_fraction, 1e-3)
    special = SPECIAL_MOVES.get(move.id, {})
    mine_up = self_boosts(move, me)
    foe_down = foe_boosts(move)
    heal = special.get("heal", move.heal or 0.0)
    # Setting up: how much our best damage rises / the opponent's threat falls after this move's boosts.
    setup_offense = setup_defense = 0.0
    if any(v > 0 for v in mine_up.values()):
        after = {k: min(6, me.boosts.get(k, 0) + mine_up.get(k, 0)) for k in STAT_BOOSTS}
        setup_offense = _log_ratio(best_damage(me, opp, True, battle.weather, after), context["my_best"])
        setup_defense = _log_ratio(context["threat"], threat(opp, me, battle.weather, after))
    if foe_down:
        saved = dict(opp.boosts)
        opp._boosts.update({k: max(-6, opp.boosts.get(k, 0) + v) for k, v in foe_down.items()})
        setup_defense = _log_ratio(context["threat"], threat(opp, me, battle.weather))
        opp._boosts.clear()
        opp._boosts.update(saved)
    volatile = move.volatile_status
    opp_side = battle.opponent_side_conditions
    return [
        move.base_power / 150.0,
        _log_mult(eff.multiplier),
        eff.p_immune,
        float(move.type in me.types),
        float(np.clip(dmg.mean / opp_hp, 0, 2)),  # expected damage relative to the target's remaining HP
        float(dmg.minimum >= opp_hp),  # guaranteed KO (ignoring Sash/Sturdy and crits)
        float(dmg.mean >= opp_hp),
        float(move.category == MoveCategory.PHYSICAL),
        float(move.category == MoveCategory.SPECIAL),
        float(move.category == MoveCategory.STATUS),
        move.priority / 3.0,
        *[mine_up.get(k, 0) / 2.0 for k in STAT_BOOSTS],
        *[foe_down.get(k, 0) / 2.0 for k in STAT_BOOSTS],
        setup_offense,
        setup_defense,
        heal * (1.0 - me.current_hp_fraction),  # useful healing: capped by missing HP
        special.get("cost", 0.0) + (0.25 if volatile == Effect.SUBSTITUTE else 0.0),
        float(volatile == Effect.SUBSTITUTE),
        float(move.is_protect_move),
        float(bool(move.self_switch)),
        float(move.self_switch == "copyvolatile"),  # Baton Pass
        float(bool(move.force_switch)),
        float(move.status is not None and opp.status is None),  # inflicts a status that can still land
        float(move.side_condition in HAZARDS and move.side_condition not in opp_side),
        float(move.side_condition in SCREENS),
        float(volatile is not None and volatile not in (Effect.SUBSTITUTE, Effect.PROTECT)),  # Yawn, Taunt, ...
    ]


def _matchup(mon: Pokemon, opp: Pokemon | None) -> list[float]:
    """Ability/item-aware type matchup of a team member against the opposing active Pokémon."""
    if opp is None:
        return [0.0, 0.0, 0.0]
    offense = max((effectiveness(m, mon, opp).multiplier for m in mon.moves.values() if m.base_power > 0),
                  default=1.0)
    defense = max((type_vs_defender(t.name.lower(), mon) for t in opp.types if t), default=1.0)
    revealed = max((effectiveness(m, opp, mon).multiplier for m in opp.moves.values() if m.base_power > 0),
                   default=1.0)
    return [_log_mult(offense), _log_mult(defense), _log_mult(revealed)]


def _side(conditions: dict) -> list[float]:
    return [min(conditions.get(c, 0), SIDE_MAX.get(c, 1)) / SIDE_MAX.get(c, 1) if c in conditions else 0.0
            for c in SIDE_CONDITIONS]


def _volatiles(mon: Pokemon | None) -> list[float]:
    effects = mon.effects if mon else {}
    return [float(e in effects) for e in VOLATILES] + [float(bool(mon and mon.protect_counter))]


def embed(battle: AbstractBattle) -> np.ndarray:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    moves = list(me.moves.values())[:4] if me else []
    moves += [None] * (4 - len(moves))
    feats: list[float] = []
    for mon in (me, opp):
        feats += [mon.current_hp_fraction if mon else 0.0] + _types(mon) + _status(mon) + _boosts(mon)
        feats += _volatiles(mon)

    context = {"my_best": 0.0, "threat": 0.0}
    if me and opp:
        context = {"my_best": best_damage(me, opp, True, battle.weather), "threat": threat(opp, me, battle.weather)}
        my_speed, opp_speed = damage.speed(me, True), damage.speed(opp, False)
        trick_room = Field.TRICK_ROOM in battle.fields
        faster = (my_speed > opp_speed) != trick_room
        my_hp, opp_hp = max(me.current_hp_fraction, 1e-3), max(opp.current_hp_fraction, 1e-3)
        feats += [float(faster), _log_ratio(my_speed, opp_speed),
                  float(np.clip(context["my_best"] / opp_hp, 0, 2)), float(context["my_best"] >= opp_hp),
                  float(np.clip(context["threat"] / my_hp, 0, 2)), float(context["threat"] >= my_hp)]
    else:
        feats += [0.0] * 6
    feats += [float(battle.can_mega_evolve)]
    feats += [float(w in battle.weather) for w in WEATHERS] + [float(f in battle.fields) for f in FIELDS]
    feats += _side(battle.side_conditions) + _side(battle.opponent_side_conditions)
    for move in moves:
        feats += move_features(move, me, opp, battle, context)

    team = list(battle.team.values())[:6]
    for mon in team + [None] * (6 - len(team)):
        if mon is None:
            feats += [0.0] * 8
            continue
        bench_threat = threat(opp, mon, battle.weather) if opp and not mon.fainted else 0.0
        feats += [mon.current_hp_fraction, float(mon.fainted), float(mon.active), float(mon.status is not None),
                  float(np.clip(bench_threat / max(mon.current_hp_fraction, 1e-3), 0, 2))] + _matchup(mon, opp)
    opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
    my_fainted = sum(m.fainted for m in battle.team.values())
    feats += [opp_fainted / 3.0, my_fainted / 3.0, min(battle.turn, 30) / 30.0]
    return np.asarray(feats, dtype=np.float32)


OBS_SIZE = (2 * (1 + len(TYPES) + len(STATUSES) + len(BOOSTS) + len(VOLATILES) + 1) + 6 + 1
            + len(WEATHERS) + len(FIELDS) + 2 * len(SIDE_CONDITIONS) + 4 * MOVE_FEATURES + 6 * 8 + 3)
