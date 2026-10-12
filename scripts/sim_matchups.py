"""Simulate games of my team (trained any-lineup policy, random lineup each game) against the team pool, and fit
the per-lineup matchup model used for team preview suggestions (pokesim.live.selection.MatchupModel).

  record: appends one JSON line per game to data/processed/matchups.jsonl:
          {"lineup": "lead,second,third", "opponent": [species brought and seen], "won": bool}
  fit:    per lineup, L2-regularised logistic regression of the result on which opponent species were brought.

Usage:
  uv run python scripts/sim_matchups.py record --policy checkpoints/ppo_baton_any.zip --battles 6000
  uv run python scripts/sim_matchups.py fit
"""
import argparse
import asyncio
import json
import random
import re
import uuid
from collections import defaultdict

import numpy as np
from poke_env import AccountConfiguration, ServerConfiguration

from pokesim.config import FORMAT, PROCESSED_DIR, ROOT
from pokesim.live.selection import MATCHUP_MODEL
from pokesim.rl.baton import PolicyPilot
from pokesim.rl.bc import CounterPickHeuristic
from pokesim.rl.myteam import FixedTeam, load_my_team, to_id
from pokesim.sim.pool import TeamPool, load_teams

DATA = PROCESSED_DIR / "matchups.jsonl"


def base_id(species: str) -> str:
    return re.sub(r"mega[xyz]?$", "", to_id(species))


class LineupPilot(PolicyPilot):
    """The trained policy, bringing a random 3 (random lead) each game and remembering which."""

    def __init__(self, *args, seed: int = 0, **kwargs):
        self.rng, self.lineups = random.Random(seed), {}
        super().__init__(*args, **kwargs)

    def teampreview(self, battle):
        lead, *back = self.rng.sample(range(6), 3)
        picks = [lead, *sorted(back)]  # the back two in team order, so each lineup has one key
        team = list(battle.team.values())
        self.lineups[battle.battle_tag] = [base_id(team[i].species) for i in picks]
        return "/team " + "".join(str(i + 1) for i in picks + [i for i in range(6) if i not in picks])


async def record(args) -> None:
    sets = load_my_team(args.team, args.lineup)
    roster = [s["species"] for s in sets]
    teams = load_teams(args.pool)
    server = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket", None)
    pairs = []
    for i in range(args.pairs):
        tag = uuid.uuid4().hex[:6]
        me = LineupPilot(battle_format=FORMAT, team=FixedTeam(sets), policy=args.policy, roster=roster,
                         seed=args.seed * 100 + i, server_configuration=server,
                         max_concurrent_battles=args.concurrency,
                         account_configuration=AccountConfiguration(f"mu{i}_{tag}", None))
        opp = CounterPickHeuristic(battle_format=FORMAT, team=TeamPool(teams, args.seed * 100 + 50 + i),
                                   seed=args.seed * 100 + i, p_counter=args.p_counter, server_configuration=server,
                                   max_concurrent_battles=args.concurrency,
                                   account_configuration=AccountConfiguration(f"muo{i}_{tag}", None))
        pairs.append((me, opp))
    await asyncio.gather(*(me.battle_against(opp, n_battles=args.battles // args.pairs) for me, opp in pairs))
    n = 0
    with DATA.open("a") as f:
        for me, _ in pairs:
            for tag, battle in me.battles.items():
                if tag not in me.lineups or not battle.finished:
                    continue
                opponent = sorted({base_id(m.species) for m in battle.opponent_team.values()})
                f.write(json.dumps({"lineup": ",".join(me.lineups[tag]), "opponent": opponent,
                                    "won": bool(battle.won)}) + "\n")
                n += 1
    wins = sum(me.n_won_battles for me, _ in pairs)
    print(f"recorded {n} games ({wins} won) to {DATA}")


def logistic(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 50) -> np.ndarray:
    """Newton's method; the bias (column 0) is not regularised."""
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + reg * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def fit(args) -> None:
    games = defaultdict(list)
    for line in DATA.read_text().splitlines():
        g = json.loads(line)
        games[g["lineup"]].append(g)
    model = {"lineups": {}}
    for lineup, rows in sorted(games.items()):
        species = sorted({s for g in rows for s in g["opponent"]})
        index = {s: i + 1 for i, s in enumerate(species)}
        X = np.zeros((len(rows), len(species) + 1))
        X[:, 0] = 1.0
        for r, g in enumerate(rows):
            for s in g["opponent"]:
                X[r, index[s]] = 1.0
        y = np.array([g["won"] for g in rows], dtype=float)
        w = logistic(X, y, args.l2)
        model["lineups"][lineup] = {"bias": float(w[0]), "games": len(rows), "win_rate": float(y.mean()),
                                    "effects": {s: round(float(w[index[s]]), 4) for s in species}}
    MATCHUP_MODEL.write_text(json.dumps(model))
    best = sorted(model["lineups"].items(), key=lambda kv: -kv[1]["win_rate"])
    print(f"fitted {len(best)} lineups from {sum(len(r) for r in games.values())} games -> {MATCHUP_MODEL}")
    for lineup, m in best[:8] + [("...", None)] + best[-3:]:
        if m:
            print(f"  {lineup:40s} {m['win_rate']:.0%} over {m['games']} games")
        else:
            print("  ...")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--policy", required=True)
    rec.add_argument("--battles", type=int, default=6000)
    rec.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    rec.add_argument("--lineup", nargs=3, default=["Scolipede", "Floette-Eternal", "Gholdengo"],
                     help="team order used in training (the roster order of the policy)")
    rec.add_argument("--pool", default="all")
    rec.add_argument("--port", type=int, default=8000)
    rec.add_argument("--pairs", type=int, default=4)
    rec.add_argument("--concurrency", type=int, default=8)
    rec.add_argument("--p-counter", type=float, default=0.7)
    rec.add_argument("--seed", type=int, default=0)
    ft = sub.add_parser("fit")
    ft.add_argument("--l2", type=float, default=2.0)
    args = parser.parse_args()
    if args.cmd == "record":
        asyncio.run(record(args))
    else:
        fit(args)


if __name__ == "__main__":
    main()
