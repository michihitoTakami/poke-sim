"""Team pool: each battle draws a random team from the built teams (teams_<season>.json)."""
import json
import random

from poke_env.teambuilder import Teambuilder

from pokesim.config import PROCESSED_DIR


def load_teams(season: str = "M-6") -> list[dict]:
    return json.loads((PROCESSED_DIR / f"teams_{season}.json").read_text())


class TeamPool(Teambuilder):
    def __init__(self, teams: list[dict], seed: int | None = None):
        self.packed = [t["packed"] for t in teams]
        self.rng = random.Random(seed)

    def yield_team(self) -> str:
        return self.rng.choice(self.packed)
