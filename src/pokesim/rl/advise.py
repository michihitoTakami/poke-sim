"""Evaluate a trained policy and show its per-turn action probabilities ("what to do next").

Usage:
  uv run python -m pokesim.rl.advise --model checkpoints/ppo_m6.zip --eval 100
  uv run python -m pokesim.rl.advise --model checkpoints/ppo_m6.zip --show 1
"""
import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sqlalchemy import select

from pokesim.db.models import NameI18n
from pokesim.db.session import Session
from pokesim.rl.env import MaskedSinglesEnv, battle_from
from pokesim.rl.train import OPPONENTS
from pokesim.sim.pool import TeamPool, load_teams


def ja_names() -> dict[str, str]:
    with Session() as s:
        return dict(s.execute(select(NameI18n.ref_id, NameI18n.name).where(NameI18n.lang == "ja")).all())


def action_probs(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None])
        return dist.distribution.probs.cpu().numpy()[0]


def describe_action(action: int, battle, ja: dict[str, str]) -> str:
    if action < 6:
        mon = list(battle.team.values())[action]
        return f"交代→{ja.get(mon.species, mon.species)}"
    moves = list(battle.active_pokemon.moves.values())[:4]
    move = moves[(action - 6) % 4] if (action - 6) % 4 < len(moves) else None
    name = ja.get(move.id, move.id) if move else "?"
    return f"{name}{' (メガシンカ)' if 10 <= action < 14 else ''}"


def run(model_path: str, opponent: str, episodes: int, show: int, season: str) -> None:
    teams = load_teams(season)
    model = MaskablePPO.load(model_path, device="cuda")
    env = MaskedSinglesEnv(OPPONENTS[opponent], TeamPool(teams, 11), TeamPool(teams, 12))
    ja = ja_names() if show else {}
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done, verbose = False, ep < show
        while not done:
            mask = env.action_masks()
            probs = action_probs(model, obs, mask)
            battle = battle_from(env)
            decision = battle is not None and not battle._wait and mask.sum() > 1
            if verbose and decision and battle.active_pokemon and battle.opponent_active_pokemon:
                me, opp = battle.active_pokemon, battle.opponent_active_pokemon
                ranked = sorted(np.flatnonzero(mask), key=lambda a: -probs[a])[:4]
                print(f"T{battle.turn:>2} {ja.get(me.species, me.species)}({me.current_hp_fraction:.0%}) vs "
                      f"{ja.get(opp.species, opp.species)}({opp.current_hp_fraction:.0%}): " +
                      " / ".join(f"{describe_action(a, battle, ja)} {probs[a]:.0%}" for a in ranked))
            action = int(np.argmax(np.where(mask, probs, -1)))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        wins += info.get("won", False)
        if verbose:
            print(f"--- battle {ep + 1}: {'WIN' if info.get('won') else 'LOSE'}\n")
    print(f"vs {opponent}: {wins}/{episodes} ({wins / episodes:.0%})")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponent", default="heuristic", choices=list(OPPONENTS))
    parser.add_argument("--eval", type=int, default=0, help="battles to evaluate (greedy policy)")
    parser.add_argument("--show", type=int, default=0, help="battles to print turn by turn")
    parser.add_argument("--season", default="M-6")
    args = parser.parse_args()
    run(args.model, args.opponent, max(args.eval, args.show), args.show, args.season)


if __name__ == "__main__":
    main()
