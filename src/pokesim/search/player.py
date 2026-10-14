"""A poke-env player that picks moves by lookahead (for evaluation against the baseline bots)."""
import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player, SimpleHeuristicsPlayer

from pokesim.rl.advise import ja_names
from pokesim.search.lookahead import Lookahead, SetSampler, label_to_action, to_id


def ja_label(label: str, ja: dict[str, str]) -> str:
    kind, _, rest = label.partition(":")
    if kind == "switch":
        return f"交代→{ja.get(rest, rest)}"
    move_id, _, mega = rest.partition(":")
    return ja.get(move_id, move_id) + (" (メガシンカ)" if mega else "")


def describe(battle, advice, ja: dict[str, str], top: int = 4) -> str:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    head = (f"T{battle.turn:>2} {ja.get(me.species, me.species)}({me.current_hp_fraction:.0%}) vs "
            f"{ja.get(opp.species, opp.species)}({opp.current_hp_fraction:.0%}): ")
    return head + " / ".join(f"{ja_label(lab, ja)} {p:.0%} ({v:+.2f})" for lab, p, v in advice.ranked()[:top])


class LookaheadPlayer(Player):
    """Plays the lookahead's top action; falls back to the heuristic bot for forced switches or on errors."""

    def __init__(self, *args, teams: list[dict], seasons: list[str], samples: int = 6, seeds: int = 2,
                 rollout: int = 3, show: bool = False, sampler=None, **kwargs):
        # set up before Player.__init__ starts listening, since requests can arrive right away
        self.teams = teams
        self.engine = Lookahead(SetSampler(seasons) if sampler is None else sampler, samples, seeds, rollout)
        self.fallback = SimpleHeuristicsPlayer(battle_format=kwargs.get("battle_format"), start_listening=False)
        self.decisions = self.fallbacks = 0
        self.ja = ja_names() if show else None
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
        if self.ja is not None:
            print(describe(battle, advice, self.ja), flush=True)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        for label, _, _ in advice.ranked():
            action = label_to_action(label, battle)
            if action is not None and mask[action]:
                return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        self.fallbacks += 1
        return self.fallback.choose_move(battle)
