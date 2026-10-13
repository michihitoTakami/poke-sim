"""The opponent's next action predicted by a policy cloned from human players (pokesim.rl.human).

The battle so far (protocol lines, as in BattleSession.log) is replayed from the opponent's side: its own team is
what it has revealed, the rest filled in from the set data (a few samples, averaged); our side is only what it
has seen. The policy's action probabilities on that view are the prediction.
Scored on our records with scripts/eval_records.py bc (2026-10-14, 166 decisions, only what was known at each
turn): top-1 44%, top-3 69% alone; mixed half and half with the search's prediction 46% / 72% (search alone 39% / 68%).
"""
import random

import numpy as np
import torch
from poke_env.environment import SinglesEnv
from sb3_contrib import MaskablePPO

from pokesim.config import ROOT
from pokesim.rl.features import embed
from pokesim.rl.human import Sides, open_session
from pokesim.rl.myteam import to_id

DEFAULT_POLICY = ROOT / "checkpoints/bc_human.zip"


def label_of(action: int, battle) -> str | None:
    if action < 6:
        team = list(battle.team.values())
        return f"switch:{to_id(team[action].species)}" if action < len(team) else None
    moves = list(battle.active_pokemon.moves.keys())[:4] if battle.active_pokemon else []
    slot = (action - 6) % 4
    return f"move:{moves[slot]}" if slot < len(moves) else None


class OpponentPolicy:
    def __init__(self, path=DEFAULT_POLICY, samples: int = 4):
        self.model = MaskablePPO.load(str(path), device="cpu")
        self.sides = Sides(seed=0)
        self.samples = samples

    def predict(self, log: list[str], our_role: str = "p1", force_switch: bool = False) -> list[tuple[str, float]]:
        """Most likely first. force_switch: the opponent is choosing a replacement (its active fainted)."""
        role = "p2" if our_role == "p1" else "p1"
        lines = [ln for ln in log if ln.startswith("|")]
        text = "\n".join(lines)
        out: dict[str, float] = {}
        done = 0
        for k in range(self.samples):
            self.sides.sampler.rng = random.Random(k)  # the same fill-ins each call: cached stats, stable output
            try:
                sets = self.sides.sets(text, role)
                session, body = open_session(lines, role, sets, self.sides.species)
                session.apply(body)
                battle = session.decision(force_switch=force_switch)
            except Exception as e:  # noqa: BLE001 - a log the session cannot follow: no prediction
                print("opponent prediction failed:", repr(e)[:160], flush=True)
                continue
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            if not mask.any():
                continue
            with torch.no_grad():
                obs = torch.tensor(embed(battle)[None])
                probs = self.model.policy.get_distribution(obs, action_masks=mask[None]).distribution.probs[0].numpy()
            for a in np.flatnonzero(mask):
                label = label_of(int(a), battle)
                if label:
                    out[label] = out.get(label, 0.0) + float(probs[a])
            done += 1
        if not done:
            return []
        return sorted(((lab, p / done) for lab, p in out.items()), key=lambda kv: -kv[1])
