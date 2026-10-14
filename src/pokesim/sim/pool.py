"""Team pool: each battle draws a random team from the built teams (teams_<season>.json)."""
import json
import random

from poke_env.teambuilder import Teambuilder

from pokesim.config import PROCESSED_DIR


def load_teams(season: str = "M-6", scope=None) -> list[dict]:
    path = PROCESSED_DIR / f"teams_{season}.json"
    if scope is not None and season in {"observed", "mix"}:
        from pokesim.data.dataset import validate_artifact

        path = PROCESSED_DIR / "scoped" / f"teams_{season}.json"
        validate_artifact(path, scope)
    return json.loads(path.read_text())


class TeamPool(Teambuilder):
    def __init__(self, teams: list[dict], seed: int | None = None):
        self.packed = [t["packed"] for t in teams]
        self.rng = random.Random(seed)

    def yield_team(self) -> str:
        return self.rng.choice(self.packed)
