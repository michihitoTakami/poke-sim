"""P0 smoke test: two random players on the local Showdown server."""
import asyncio
import sys

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer

FORMAT = sys.argv[1] if len(sys.argv) > 1 else "gen9championsrandombattle"


async def main() -> None:
    p1 = RandomPlayer(battle_format=FORMAT, server_configuration=LocalhostServerConfiguration,
                      account_configuration=AccountConfiguration("smoke_a", None))
    p2 = RandomPlayer(battle_format=FORMAT, server_configuration=LocalhostServerConfiguration,
                      account_configuration=AccountConfiguration("smoke_b", None))
    await p1.battle_against(p2, n_battles=10)
    print(f"{FORMAT}: p1 won {p1.n_won_battles}/{p1.n_finished_battles}")


asyncio.run(main())
