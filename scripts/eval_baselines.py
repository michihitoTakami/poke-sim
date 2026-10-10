"""Round-robin of poke-env baseline bots on M-6 article teams (gen9championsbssregmc)."""
import asyncio
import itertools
import sys

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokesim.config import FORMAT
from pokesim.sim.pool import TeamPool, load_teams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50


async def main() -> None:
    teams = load_teams("M-6")
    players = {}
    for i, (name, cls) in enumerate([("random", RandomPlayer), ("maxbp", MaxBasePowerPlayer),
                                     ("heuristic", SimpleHeuristicsPlayer)]):
        players[name] = cls(battle_format=FORMAT, server_configuration=LocalhostServerConfiguration,
                            account_configuration=AccountConfiguration(f"eval_{name}", None),
                            team=TeamPool(teams, seed=i), max_concurrent_battles=10)
    for a, b in itertools.combinations(players, 2):
        pa, pb = players[a], players[b]
        before = pa.n_won_battles, pa.n_finished_battles
        await pa.battle_against(pb, n_battles=N)
        won, total = pa.n_won_battles - before[0], pa.n_finished_battles - before[1]
        print(f"{a:>9} vs {b:<9}: {won}/{total} ({won / total:.0%})", flush=True)


asyncio.run(main())
