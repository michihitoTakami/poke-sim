"""Gymnasium environment for Champions singles (3v3) on the local Showdown server.

Observation: a fixed-size feature vector (see pokesim.rl.features). Actions follow poke-env's SinglesEnv mapping
(0-5 switch, 6-9 move, 10-13 move + mega); illegal actions are masked.
"""
import uuid

import numpy as np
from gymnasium import Env, spaces
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player

from pokesim.config import FORMAT
from pokesim.rl.features import OBS_SIZE, embed

N_ACTIONS = SinglesEnv.get_action_space_size(9)


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

    def __init__(self, opponent_cls: type[Player], team, opponent_team, format_id: str = FORMAT,
                 server_configuration: ServerConfiguration = LocalhostServerConfiguration):
        tag = uuid.uuid4().hex[:6]
        self.poke_env = ChampionsSinglesEnv(
            battle_format=format_id, server_configuration=server_configuration,
            account_configuration1=AccountConfiguration(f"agent_{tag}", None),
            account_configuration2=AccountConfiguration(f"envopp_{tag}", None),
            team=team, choose_on_teampreview=False, strict=False)
        opponent = opponent_cls(battle_format=format_id, team=opponent_team, start_listening=False,
                                server_configuration=server_configuration,
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
