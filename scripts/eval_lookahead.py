"""Win rate of the lookahead player against the heuristic bot on the local Showdown server (port 8000).

Usage: uv run python scripts/eval_lookahead.py [--battles 50] [--samples 6] [--seeds 2] [--rollout 3]
"""
import argparse
import asyncio
import time
import uuid

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from pokesim.config import FORMAT
from pokesim.search.player import LookaheadPlayer
from pokesim.sim.pool import TeamPool, load_teams


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--teams", default="M-6")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rollout", type=int, default=3)
    args = parser.parse_args()
    teams = load_teams(args.teams)
    tag = uuid.uuid4().hex[:6]
    player = LookaheadPlayer(battle_format=FORMAT, team=TeamPool(teams, 51), teams=teams, seasons=["M-6"],
                             samples=args.samples, seeds=args.seeds, rollout=args.rollout,
                             account_configuration=AccountConfiguration(f"lookahead_{tag}", None))
    bot = SimpleHeuristicsPlayer(battle_format=FORMAT, team=TeamPool(teams, 52),
                                 account_configuration=AccountConfiguration(f"heurbot_{tag}", None))
    start = time.time()
    await player.battle_against(bot, n_battles=args.battles)
    elapsed = time.time() - start
    print(f"lookahead vs heuristic: {player.n_won_battles}/{args.battles} ({player.n_won_battles / args.battles:.0%}), "
          f"{player.decisions} decisions ({elapsed / max(player.decisions, 1):.2f}s each), "
          f"{player.fallbacks} fallbacks")
    player.engine.close()


if __name__ == "__main__":
    asyncio.run(main())
