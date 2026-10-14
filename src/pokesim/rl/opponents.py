"""Opponents for training: frozen policy snapshots and a league mixing them with baseline bots."""
import random
from pathlib import Path

import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player, SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO

from pokesim.rl.features import embed


class PolicyStrategy:
    """Plays a saved MaskablePPO policy (sampled, not greedy, so the opponent stays varied)."""

    def __init__(self, path: str | Path, deterministic: bool = False):
        self.model = MaskablePPO.load(path, device="cpu")
        self.deterministic = deterministic

    def choose_move(self, battle):
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if not mask.any():
            return Player.choose_random_move(battle)
        action, _ = self.model.predict(embed(battle), action_masks=mask, deterministic=self.deterministic)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


class LeaguePlayer(Player):
    """Picks one strategy per battle: the lookahead player (pokesim.search) with probability `p_lookahead`, else
    the heuristic bot with probability `p_heuristic`, else a random snapshot."""

    def __init__(self, *args, snapshots: list[str | Path] = (), p_heuristic: float = 0.5, seed: int = 0,
                 p_lookahead: float = 0.0, teams: list[dict] | None = None, prior_bundle: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        fmt = kwargs.get("battle_format")
        self.heuristic = SimpleHeuristicsPlayer(battle_format=fmt, start_listening=False)
        self.policies = [PolicyStrategy(p) for p in snapshots]
        self.p_heuristic = p_heuristic if self.policies else 1.0
        self.p_lookahead, self.lookahead = p_lookahead, None
        if p_lookahead > 0:
            from pokesim.data.prior_bundle import sampler
            from pokesim.search.player import LookaheadPlayer

            self.lookahead = LookaheadPlayer(battle_format=fmt, teams=teams, seasons=["M-6"], start_listening=False,
                                            sampler=sampler(prior_bundle) if prior_bundle is not None else None)
        self.rng = random.Random(seed)
        self.assigned: dict[str, object] = {}

    def _strategy(self, battle):
        if battle.battle_tag not in self.assigned and self.lookahead and self.rng.random() < self.p_lookahead:
            self.assigned[battle.battle_tag] = self.lookahead
        if battle.battle_tag not in self.assigned:
            use_bot = self.rng.random() < self.p_heuristic
            self.assigned[battle.battle_tag] = self.heuristic if use_bot else self.rng.choice(self.policies)
        return self.assigned[battle.battle_tag]

    def choose_move(self, battle):
        return self._strategy(battle).choose_move(battle)
