"""A poke-env player that picks moves by lookahead (for evaluation against the baseline bots)."""
import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player, SimpleHeuristicsPlayer

from pokesim.search.lookahead import Lookahead, SetSampler, label_to_action, to_id


class LookaheadPlayer(Player):
    """Plays the lookahead's top action; falls back to the heuristic bot for forced switches or on errors."""

    def __init__(self, *args, teams: list[dict], seasons: list[str], samples: int = 6, seeds: int = 2,
                 rollout: int = 3, **kwargs):
        # set up before Player.__init__ starts listening, since requests can arrive right away
        self.teams = teams
        self.engine = Lookahead(SetSampler(seasons), samples, seeds, rollout)
        self.fallback = SimpleHeuristicsPlayer(battle_format=kwargs.get("battle_format"), start_listening=False)
        self.decisions = self.fallbacks = 0
        super().__init__(*args, **kwargs)

    def own_sets(self, battle) -> dict[str, dict]:
        mine = {self.engine.sampler.base_id(to_id(m.species)) for m in battle.team.values()}
        for team in self.teams:
            sets = {to_id(s["species"]): s for s in team["sets"]}
            if mine <= set(sets):
                return sets
        raise KeyError(f"team not found for {sorted(mine)}")

    def choose_move(self, battle):
        if battle.force_switch or not battle.active_pokemon or not battle.opponent_active_pokemon:
            return self.fallback.choose_move(battle)
        try:
            advice = self.engine.advise(battle, self.own_sets(battle))
        except Exception as e:  # noqa: BLE001 - keep playing; count it
            self.fallbacks += 1
            print("lookahead failed:", repr(e)[:200], flush=True)
            return self.fallback.choose_move(battle)
        self.decisions += 1
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        for label, _, _ in advice.ranked():
            action = label_to_action(label, battle)
            if action is not None and mask[action]:
                return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        self.fallbacks += 1
        return self.fallback.choose_move(battle)
