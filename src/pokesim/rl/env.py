"""Gymnasium environment for Champions singles (3v3) on the local Showdown server.

Observation: a fixed-size feature vector (see pokesim.rl.features). Actions follow poke-env's SinglesEnv mapping
(0-5 switch, 6-9 move, 10-13 move + mega); illegal actions are masked.
"""
import uuid
from collections.abc import Callable

import numpy as np
from gymnasium import Env, spaces
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player

from pokesim.config import FORMAT
from pokesim.rl.features import OBS_SIZE, embed

N_ACTIONS = SinglesEnv.get_action_space_size(9)

# Reward presets: "dense" pays every HP point traded each turn; "sparse" mostly pays for winning, so turns spent
# setting up or recovering are not judged by the immediate HP swing.
REWARDS = {
    "dense": {"fainted_value": 2.0, "hp_value": 1.0, "victory_value": 15.0},
    "sparse": {"fainted_value": 1.0, "hp_value": 0.25, "victory_value": 20.0},
    "win": {"fainted_value": 0.05, "hp_value": 0.0125, "victory_value": 1.0},  # sparse / 20 (value head in ±1)
}


class ChampionsSinglesEnv(SinglesEnv):
    """`extra` appends features (e.g. pokesim.rl.myteam.TeamFeatures); `lineup` fixes our team preview order
    (e.g. "123"); `opponent_preview(battle) -> "/team ..."` picks the opponent's 3 (default: random)."""

    def __init__(self, reward: str = "dense", extra: Callable[[AbstractBattle], np.ndarray] | None = None,
                 lineup: str | Callable[[AbstractBattle], str] | None = None,
                 opponent_preview: Callable[[AbstractBattle], str] | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reward_weights = REWARDS[reward]
        self.extra, self.lineup, self.opponent_preview = extra, lineup, opponent_preview
        self.obs_size = obs_size(extra)
        box = spaces.Box(-np.inf, np.inf, shape=(self.obs_size,), dtype=np.float32)
        # team preview is decided by the inner env players, so override theirs
        if callable(lineup):
            self.agent1.teampreview = lineup
        elif lineup:
            self.agent1.teampreview = lambda battle: "/team " + lineup
        if opponent_preview:
            self.agent2.teampreview = opponent_preview
        self.observation_spaces = {agent: spaces.Dict({
            "observation": box, "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8)})
            for agent in self.possible_agents}

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, number_of_pokemons=3, **self.reward_weights)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        obs = embed(battle)
        return np.concatenate([obs, self.extra(battle)]) if self.extra else obs


def obs_size(extra=None) -> int:
    return OBS_SIZE + (extra.size if extra is not None else 0)


class MaskedSinglesEnv(Env):
    """Single-agent view (vs a fixed opponent) exposing `action_masks()` for MaskablePPO."""

    def __init__(self, opponent_cls: Callable[..., Player], team, opponent_team, format_id: str = FORMAT,
                 server_configuration: ServerConfiguration = LocalhostServerConfiguration, reward: str = "dense",
                 **env_kwargs):
        tag = uuid.uuid4().hex[:6]
        self.poke_env = ChampionsSinglesEnv(
            reward=reward, battle_format=format_id, server_configuration=server_configuration,
            account_configuration1=AccountConfiguration(f"agent_{tag}", None),
            account_configuration2=AccountConfiguration(f"envopp_{tag}", None),
            team=team, choose_on_teampreview=False, strict=False, **env_kwargs)
        # poke-env gives `team` to both inner players; the opponent brings its own team instead
        self.poke_env.agent2._team = opponent_team
        opponent = opponent_cls(battle_format=format_id, team=opponent_team, start_listening=False,
                                server_configuration=server_configuration,
                                account_configuration=AccountConfiguration(f"oppbot_{tag}", None))
        self.env = SingleAgentWrapper(self.poke_env, opponent)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.poke_env.obs_size,), dtype=np.float32)
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
