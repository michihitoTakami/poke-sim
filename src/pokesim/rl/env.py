"""Gymnasium environment for Champions singles (3v3) on the local Showdown server.

Observation: a fixed-size feature vector (active matchup, the 4 moves' expected power,
the team's HP/matchups). Actions follow poke-env's SinglesEnv mapping
(0-5 switch, 6-9 move, 10-13 move + mega); illegal actions are masked.
"""
import uuid

import numpy as np
from gymnasium import Env, spaces
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, Battle, Move, MoveCategory, Pokemon, PokemonType, Status
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player

from pokesim.config import FORMAT

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
STATUSES = list(Status)
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
N_ACTIONS = SinglesEnv.get_action_space_size(9)


def _types(mon: Pokemon | None) -> list[float]:
    present = set(mon.types) if mon else set()
    return [float(t in present) for t in TYPES]


def _status(mon: Pokemon | None) -> list[float]:
    return [float(mon is not None and mon.status == s) for s in STATUSES]


def _boosts(mon: Pokemon | None) -> list[float]:
    return [(mon.boosts.get(b, 0) / 6.0) if mon else 0.0 for b in BOOSTS]


def _speed_ratio(a: Pokemon | None, b: Pokemon | None) -> float:
    if not a or not b:
        return 0.0
    return float(np.clip((a.base_stats["spe"] - b.base_stats["spe"]) / 100.0, -1, 1))


def _move_features(move: Move | None, user: Pokemon | None, target: Pokemon | None) -> list[float]:
    if move is None or user is None:
        return [0.0] * 8
    mult = target.damage_multiplier(move) if target else 1.0
    stab = 1.5 if move.type in user.types else 1.0
    power = move.base_power * stab * mult * (move.accuracy if isinstance(move.accuracy, float) else 1.0)
    return [
        move.base_power / 150.0,
        np.log2(mult + 1e-3) / 2.0 if mult > 0 else -1.5,
        float(stab > 1),
        power / 300.0,
        float(move.category == MoveCategory.PHYSICAL),
        float(move.category == MoveCategory.SPECIAL),
        float(move.category == MoveCategory.STATUS),
        move.priority / 3.0,
    ]


def _matchup(mon: Pokemon, opp: Pokemon | None) -> list[float]:
    """Offense: best multiplier of mon's known moves vs opp; defense: worst multiplier of opp's types vs mon."""
    if opp is None:
        return [0.0, 0.0]
    offense = max((opp.damage_multiplier(m) for m in mon.moves.values() if m.base_power > 0), default=1.0)
    defense = max((mon.damage_multiplier(t) for t in opp.types if t), default=1.0)
    return [np.log2(offense + 1e-3) / 2.0, np.log2(defense + 1e-3) / 2.0]


def embed(battle: AbstractBattle) -> np.ndarray:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    moves = list(me.moves.values())[:4] if me else []
    moves += [None] * (4 - len(moves))
    feats: list[float] = []
    for mon in (me, opp):
        feats += [mon.current_hp_fraction if mon else 0.0] + _types(mon) + _status(mon) + _boosts(mon)
    feats += [_speed_ratio(me, opp), float(battle.can_mega_evolve)]
    for move in moves:
        feats += _move_features(move, me, opp)
    team = list(battle.team.values())[:6]
    for mon in team + [None] * (6 - len(team)):
        if mon is None:
            feats += [0.0] * 6
            continue
        feats += [mon.current_hp_fraction, float(mon.fainted), float(mon.active),
                  _speed_ratio(mon, opp)] + _matchup(mon, opp)
    opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
    my_fainted = sum(m.fainted for m in battle.team.values())
    feats += [opp_fainted / 3.0, my_fainted / 3.0, min(battle.turn, 30) / 30.0]
    return np.asarray(feats, dtype=np.float32)


OBS_SIZE = 2 * (1 + len(TYPES) + len(STATUSES) + len(BOOSTS)) + 2 + 4 * 8 + 6 * 6 + 3


class ChampionsSinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        box = spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.observation_spaces = {agent: spaces.Dict({
            "observation": box, "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8)})
            for agent in self.possible_agents}

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, number_of_pokemons=3,
                                            victory_value=15.0)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed(battle)


class MaskedSinglesEnv(Env):
    """Single-agent view (vs a fixed opponent) exposing `action_masks()` for MaskablePPO."""

    def __init__(self, opponent_cls: type[Player], team, opponent_team, format_id: str = FORMAT):
        tag = uuid.uuid4().hex[:6]
        self.poke_env = ChampionsSinglesEnv(
            battle_format=format_id, server_configuration=LocalhostServerConfiguration,
            account_configuration1=AccountConfiguration(f"agent_{tag}", None),
            account_configuration2=AccountConfiguration(f"envopp_{tag}", None),
            team=team, choose_on_teampreview=False, strict=False)
        opponent = opponent_cls(battle_format=format_id, team=opponent_team, start_listening=False,
                                account_configuration=AccountConfiguration(f"oppbot_{tag}", None))
        self.env = SingleAgentWrapper(self.poke_env, opponent)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._mask = np.ones(N_ACTIONS, dtype=bool)

    def _unpack(self, obs) -> np.ndarray:
        self._mask = obs["action_mask"].astype(bool)
        if not self._mask.any():
            self._mask[0] = True
        return obs["observation"]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._unpack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.int64(action))
        if terminated or truncated:
            battle = self.poke_env.battle1
            info = {**info, "won": bool(battle and battle.won)}
        return self._unpack(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._mask

    def close(self):
        self.env.close()


def battle_from(env: MaskedSinglesEnv) -> Battle | None:
    return env.poke_env.battle1
