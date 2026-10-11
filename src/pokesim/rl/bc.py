"""Behaviour cloning for the fixed team: record the scripted Baton Pass plans, then fit a MaskablePPO policy to them.

The recorded games use the same observation (embed + TeamFeatures), team order and lineup as training
(pokesim.rl.train --my-team), so the cloned model can be passed to it with --init and improved by self-play.
The value head is fitted to the discounted return of the training reward, so PPO starts with a sensible critic.

Usage:
  uv run python -m pokesim.rl.bc record --battles 3000 --out data/processed/bc_baton.npz
  uv run python -m pokesim.rl.bc fit --data data/processed/bc_baton.npz --out checkpoints/bc_baton
"""
import argparse
import asyncio
import random
import uuid

import numpy as np
import torch
from gymnasium import Env, spaces
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer

from pokesim.config import FORMAT, ROOT
from pokesim.rl.baton import BatonScript
from pokesim.rl.env import N_ACTIONS, REWARDS
from pokesim.rl.features import embed
from pokesim.rl.myteam import FixedTeam, TeamFeatures, counter_preview, load_my_team
from pokesim.sim.pool import TeamPool, load_teams

DEFAULT_TEAM = ROOT / "data/my_teams/baton.json"
DEFAULT_LINEUP = ["Scolipede", "Floette-Eternal", "Gholdengo"]


def state_value(battle, w: dict, n: int = 3) -> float:
    """The state value behind poke-env's reward_computing_helper (no status term), so returns match training."""
    v = sum(m.current_hp_fraction * w["hp_value"] - m.fainted * w["fainted_value"] for m in battle.team.values())
    v += (n - len(battle.team)) * w["hp_value"]
    v -= sum(m.current_hp_fraction * w["hp_value"] - m.fainted * w["fainted_value"]
             for m in battle.opponent_team.values())
    v -= (n - len(battle.opponent_team)) * w["hp_value"]
    if battle.won:
        v += w["victory_value"]
    elif battle.lost:
        v -= w["victory_value"]
    return v


class CounterPickHeuristic(SimpleHeuristicsPlayer):
    """The heuristic bot, but it brings Encore / phazing / Haze users after seeing our team."""

    def __init__(self, *args, p_counter: float = 0.7, seed: int = 0, **kwargs):
        self.rng, self.p_counter = random.Random(seed), p_counter
        super().__init__(*args, **kwargs)

    def teampreview(self, battle):
        return counter_preview(battle, self.rng, self.p_counter)


class Recorder(BatonScript):
    """Plays a scripted plan and records (observation, mask, action, state value) for every decision."""

    def __init__(self, *args, extra: TeamFeatures, weights: dict, **kwargs):
        self.extra, self.weights = extra, weights
        self.steps: dict[str, list[tuple]] = {}
        super().__init__(*args, **kwargs)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        try:
            action = int(SinglesEnv.order_to_action(order, battle, strict=False))
        except (ValueError, AssertionError, IndexError):
            return order
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if 0 <= action < N_ACTIONS and mask[action]:
            obs = np.concatenate([embed(battle), self.extra(battle)])
            value = state_value(battle, self.weights)
            self.steps.setdefault(battle.battle_tag, []).append((obs, mask, action, value))
        return order


def returns(battle, steps: list[tuple], w: dict, gamma: float) -> list[float]:
    """Discounted return after each recorded decision (rewards are differences of the state values taken at
    decision time, ending with the final state)."""
    values = [s[3] for s in steps] + [state_value(battle, w)]
    rewards = np.diff(values)
    out, g = [], 0.0
    for r in rewards[::-1]:
        g = r + gamma * g
        out.append(g)
    return out[::-1]


async def record(args) -> None:
    w = REWARDS[args.reward]
    sets = load_my_team(args.team, args.lineup)
    extra = TeamFeatures([s["species"] for s in sets])
    teams = load_teams(args.pool)
    server = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket", None)
    pairs = []
    for i, (plan, pass_to) in enumerate([("pass6", None), ("sub", "Floette-Eternal")]):
        tag = uuid.uuid4().hex[:6]
        me = Recorder(battle_format=FORMAT, team=FixedTeam(sets), plan=plan, pass_to=pass_to, extra=extra, weights=w,
                      server_configuration=server, max_concurrent_battles=args.concurrency,
                      account_configuration=AccountConfiguration(f"bc{i}_{tag}", None))
        opp = CounterPickHeuristic(battle_format=FORMAT, team=TeamPool(teams, 11 + i), seed=i,
                                   p_counter=args.p_counter, server_configuration=server,
                                   max_concurrent_battles=args.concurrency,
                                   account_configuration=AccountConfiguration(f"bco{i}_{tag}", None))
        pairs.append((me, opp))
    await asyncio.gather(*(me.battle_against(opp, n_battles=args.battles // 2) for me, opp in pairs))

    obs, masks, actions, rets, wins = [], [], [], [], []
    for me, _ in pairs:
        for tag, battle in me.battles.items():
            steps = me.steps.get(tag, [])
            if not steps:
                continue
            rets += returns(battle, steps, w, args.gamma)
            obs += [s[0] for s in steps]
            masks += [s[1] for s in steps]
            actions += [s[2] for s in steps]
            wins.append(battle.won)
        print(f"[{me.plan}] {me.n_won_battles}/{len(me.battles)} won", flush=True)
    np.savez_compressed(args.out, obs=np.array(obs, dtype=np.float32), masks=np.array(masks),
                        actions=np.array(actions), returns=np.array(rets, dtype=np.float32))
    print(f"saved {len(actions)} decisions from {len(wins)} battles to {args.out}")


class SpacesOnly(Env):
    """Just the spaces, to build a MaskablePPO model outside a live environment."""

    def __init__(self, obs_size: int):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)


def fit(args) -> None:
    from pokesim.rl.train import new_model

    data = np.load(args.data)
    obs, masks = torch.tensor(data["obs"]), torch.tensor(data["masks"])
    actions, rets = torch.tensor(data["actions"]), torch.tensor(data["returns"])
    n = len(actions)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val, train = perm[: n // 10], perm[n // 10:]
    model = new_model(SpacesOnly(obs.shape[1]))
    policy = model.policy
    device = policy.device
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def evaluate(idx):
        with torch.no_grad():
            o, m = obs[idx].to(device), masks[idx].numpy()
            dist = policy.get_distribution(o, action_masks=m)
            acc = (dist.distribution.probs.argmax(1).cpu() == actions[idx]).float().mean().item()
            v = policy.predict_values(o).squeeze(1).cpu()
            return acc, torch.nn.functional.mse_loss(v, rets[idx]).item()

    for epoch in range(args.epochs):
        order = train[torch.randperm(len(train))]
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            o = obs[idx].to(device)
            dist = policy.get_distribution(o, action_masks=masks[idx].numpy())
            logp = dist.log_prob(actions[idx].to(device))
            v = policy.predict_values(o).squeeze(1)
            loss = -logp.mean() + args.vf_coef * torch.nn.functional.mse_loss(v, rets[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, vloss = evaluate(val)
        print(f"epoch {epoch + 1}: held-out action accuracy {acc:.1%}, value MSE {vloss:.2f}", flush=True)
    model.save(args.out)
    print(f"saved {args.out}.zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--battles", type=int, default=2000)
    rec.add_argument("--out", default=str(ROOT / "data/processed/bc_baton.npz"))
    rec.add_argument("--team", default=str(DEFAULT_TEAM))
    rec.add_argument("--lineup", nargs=3, default=DEFAULT_LINEUP)
    rec.add_argument("--pool", default="all")
    rec.add_argument("--port", type=int, default=8000)
    rec.add_argument("--concurrency", type=int, default=8)
    rec.add_argument("--p-counter", type=float, default=0.7)
    rec.add_argument("--reward", default="sparse", choices=list(REWARDS))
    rec.add_argument("--gamma", type=float, default=0.995)
    ft = sub.add_parser("fit")
    ft.add_argument("--data", default=str(ROOT / "data/processed/bc_baton.npz"))
    ft.add_argument("--out", default=str(ROOT / "checkpoints/bc_baton"))
    ft.add_argument("--epochs", type=int, default=20)
    ft.add_argument("--batch", type=int, default=256)
    ft.add_argument("--lr", type=float, default=1e-3)
    ft.add_argument("--vf-coef", type=float, default=0.05)
    args = parser.parse_args()
    if args.cmd == "record":
        asyncio.run(record(args))
    else:
        fit(args)


if __name__ == "__main__":
    main()
