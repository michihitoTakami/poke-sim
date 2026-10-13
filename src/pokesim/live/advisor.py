"""Per-decision advice for a live battle: what to do (with probabilities), estimated win chance, and what the
opponent is likely to do next.

Sources, combined when both are available:
- the fixed-team policy (pokesim.rl.train --my-team): action probabilities and its value head (as a rough win chance);
- the lookahead (pokesim.search.lookahead): equilibrium mix over our options and the opponent's options, averaged
  over sets sampled for the opponent from usage data and everything revealed so far.
The opponent's next action mixes the search's prediction with a policy cloned from human players
(pokesim.live.opponent), which needs the battle's protocol lines (`log`).
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from sb3_contrib import MaskablePPO

from pokesim.rl.advise import action_probs
from pokesim.rl.env import REWARDS
from pokesim.rl.features import embed
from pokesim.rl.myteam import TeamFeatures, to_id
from pokesim.search.lookahead import Lookahead, SetSampler, label_to_action

POLICY_WEIGHT = 0.5  # share of the policy in the combined recommendation


@dataclass
class Suggestion:
    actions: list[tuple[str, float]]  # (label, probability), best first; labels as in the lookahead
    win_prob: float | None = None  # policy value head mapped to [0, 1]; a rough guide
    opponent: list[tuple[str, float]] = field(default_factory=list)  # opponent's next action, most likely first
    policy: dict[str, float] = field(default_factory=dict)
    lookahead: list[tuple[str, float, float]] = field(default_factory=list)  # (label, prob, value)
    elapsed: float = 0.0


def action_label(action: int, battle: Battle) -> str | None:
    """poke-env SinglesEnv action index -> lookahead-style label."""
    if action < 6:
        team = list(battle.team.values())
        return f"switch:{to_id(team[action].species)}" if action < len(team) else None
    moves = list(battle.active_pokemon.moves.keys())[:4] if battle.active_pokemon else []
    slot = (action - 6) % 4
    if slot >= len(moves):
        return None
    return f"move:{moves[slot]}" + (":mega" if action >= 10 else "")


# how the opponent's next action is predicted (pokesim.search.lookahead.opponent_mix); chosen on our own records
# with scripts/eval_records.py
# 166 decisions in 14 battles (2026-10-13): top-1 32% -> 42%, top-3 63% -> 69%, mean log-lik -2.65 -> -2.09
OPPONENT_MODEL = {"br_weight": 0.75, "tau": 0.7, "repeat": 0.3}
# share of the cloned human policy in the opponent prediction (scripts/eval_records.py score --bc-weight)
OPPONENT_POLICY_WEIGHT = 0.5


class Advisor:
    def __init__(self, sets: list[dict], policy: str | None = None, lookahead: bool = True,
                 seasons: tuple[str, ...] = ("M-6",), samples: int = 6, seeds: int = 2, rollout: int = 3,
                 reward: str = "sparse", opponent_model: dict | None = None,
                 opponent_policy: str | Path | None = "default"):
        self.extra = TeamFeatures([s["species"] for s in sets])
        self.model = MaskablePPO.load(policy, device="cpu") if policy else None
        self.victory = REWARDS[reward]["victory_value"]
        model = OPPONENT_MODEL if opponent_model is None else opponent_model
        self.engine = Lookahead(SetSampler(list(seasons)), samples, seeds, rollout, model) if lookahead else None
        base = self.engine.sampler.base_id if self.engine else (lambda s: s)
        self.own_sets = {base(to_id(s["species"])): s for s in sets}
        self.opponent_policy = None
        if opponent_policy:
            from pokesim.live.opponent import DEFAULT_POLICY, OpponentPolicy

            path = DEFAULT_POLICY if opponent_policy == "default" else Path(opponent_policy)
            if Path(path).exists():
                self.opponent_policy = OpponentPolicy(path)

    def close(self) -> None:
        if self.engine:
            self.engine.close()

    def _policy(self, battle: Battle) -> tuple[dict[str, float], float | None]:
        if self.model is None:
            return {}, None
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if not mask.any():
            return {}, None
        obs = np.concatenate([embed(battle), self.extra(battle)])
        probs = action_probs(self.model, obs, mask)
        out = {}
        for a in np.flatnonzero(mask):
            label = action_label(int(a), battle)
            if label:
                out[label] = out.get(label, 0.0) + float(probs[a])
        with torch.no_grad():
            value = float(self.model.policy.predict_values(self.model.policy.obs_to_tensor(obs)[0])[0, 0])
        return out, float(np.clip((value / self.victory + 1) / 2, 0, 1))

    def advise(self, battle: Battle, last_move: str | None = None, log: list[str] | None = None,
               our_role: str = "p1") -> Suggestion:
        """last_move: the opponent active's previous move ("move:<id>"), see BattleSession.last_move.
        log: the battle's protocol lines so far (BattleSession.log), for the cloned opponent policy."""
        start = time.perf_counter()
        policy, win = self._policy(battle)
        ranked, opponent = [], []
        can_search = self.engine and not battle.force_switch and battle.active_pokemon \
            and battle.opponent_active_pokemon
        if can_search:
            advice = self.engine.advise(battle, self.own_sets, last_move)
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            ranked = [(lab, p, v) for lab, p, v in advice.ranked()
                      if (a := label_to_action(lab, battle)) is not None and mask[a]]
            opponent = sorted(advice.opponent.items(), key=lambda kv: -kv[1])
        combined: dict[str, float] = {}
        look_total = sum(p for _, p, _ in ranked) or 1.0
        w = POLICY_WEIGHT if policy and ranked else (1.0 if policy else 0.0)
        for lab, p, _ in ranked:
            combined[lab] = combined.get(lab, 0.0) + (1 - w) * p / look_total
        for lab, p in policy.items():
            combined[lab] = combined.get(lab, 0.0) + w * p
        actions = sorted(combined.items(), key=lambda kv: -kv[1])
        if self.opponent_policy is not None and log and battle.opponent_active_pokemon is not None:
            opponent = mix_opponent(opponent, self.opponent_policy.predict(log, our_role))
        return Suggestion(actions, win, opponent, policy, ranked, time.perf_counter() - start)


def mix_opponent(search: list[tuple[str, float]], cloned: list[tuple[str, float]],
                 w: float = OPPONENT_POLICY_WEIGHT) -> list[tuple[str, float]]:
    """Share w for the cloned policy (all of it when the search has nothing); Mega variants merged."""
    if not cloned:
        return search
    w = w if search else 1.0
    out: dict[str, float] = {}
    for lab, p in search:
        key = ":".join(lab.split(":")[:2])
        out[key] = out.get(key, 0.0) + (1 - w) * p
    for lab, p in cloned:
        out[lab] = out.get(lab, 0.0) + w * p
    return sorted(out.items(), key=lambda kv: -kv[1])
