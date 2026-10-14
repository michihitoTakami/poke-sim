"""Team preview: which 3 the opponent is likely to bring (and lead with), and which 3 we should bring.

Opponent: per species, how often it is brought / leads when shown, from rated Showdown replays (6 shown -> the
Pokémon that appeared). A game can end before the third member appears, so the rates are relative; for a given 6
they are scaled so the expected number brought is 3 and lead probabilities sum to 1.

Us: `MatchupModel` (fitted offline from simulated games by scripts/sim_matchups.py) gives the win chance of each of
our lineups (3 of 6, lead first) against each opponent member being brought; at preview time it is averaged over
the opponent's predicted picks.

Usage: uv run python -m pokesim.live.selection pick-rates   # rebuild the replay statistics
"""
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokesim.config import PROCESSED_DIR, RAW_DIR
from pokesim.rl.myteam import to_id

PICK_RATES = PROCESSED_DIR / "pick_rates.json"
MATCHUP_MODEL = PROCESSED_DIR / "matchup_model.json"
PRIOR = 2.0  # pseudo-observations at the average rate, for rarely seen species


def build_pick_rates() -> dict:
    from sqlalchemy import select

    from pokesim.data.parse_replays import base_species, parse_log
    from pokesim.db.models import Species
    from pokesim.db.session import Session

    with Session() as s:
        species = {sp.id: sp for sp in s.scalars(select(Species))}
    shown, brought, led = Counter(), Counter(), Counter()
    for path in sorted((RAW_DIR / "replays").glob("*.json")):
        log = json.loads(path.read_text())["log"]
        sides = parse_log(log, species)
        for role, side in sides.items():
            if len(side.preview) != 6:
                continue
            first = next((line.split("|")[3] for line in log.split("\n")
                          if line.startswith(f"|switch|{role}a:")), None)
            lead = base_species(first, species) if first else None
            for sid in side.preview:
                shown[sid] += 1
                brought[sid] += sid in side.nick.values()
            if lead:
                led[lead] += 1
    total_shown = sum(shown.values())
    mean_brought = sum(brought.values()) / total_shown
    mean_led = sum(led.values()) / total_shown
    rates = {sid: {"shown": n,
                   "brought": (brought[sid] + PRIOR * mean_brought) / (n + PRIOR),
                   "lead": (led[sid] + PRIOR * mean_led) / (n + PRIOR)} for sid, n in shown.items()}
    out = {"mean": {"brought": mean_brought, "lead": mean_led}, "species": rates}
    PICK_RATES.write_text(json.dumps(out, indent=0))
    return out


def scale_to(rates: np.ndarray, total: float) -> np.ndarray:
    """Scale rates so they sum to `total` with each at most 1 (iteratively capping)."""
    p = np.asarray(rates, dtype=float).copy()
    for _ in range(20):
        free = p < 1.0
        rest = total - (~free).sum()
        if free.sum() == 0 or p[free].sum() == 0:
            break
        p[free] = p[free] * rest / p[free].sum()
        if (p <= 1.0 + 1e-9).all():
            break
        p = np.minimum(p, 1.0)
    return np.minimum(p, 1.0)


@dataclass
class OpponentPicks:
    species: list[str]
    brought: np.ndarray  # P(brought), sums to 3
    lead: np.ndarray  # P(lead), sums to 1


class PickModel:
    def __init__(self, path: Path = PICK_RATES):
        data = json.loads(path.read_text()) if path.exists() else {"mean": {"brought": 0.5, "lead": 1 / 6},
                                                                      "species": {}}
        self.mean, self.rates = data["mean"], data["species"]

    def predict(self, species: list[str]) -> OpponentPicks:
        ids = [to_id(s) for s in species]
        b = np.array([self.rates.get(s, self.mean)["brought"] for s in ids])
        lead = np.array([self.rates.get(s, self.mean)["lead"] for s in ids])
        return OpponentPicks(list(species), scale_to(b, 3.0), lead / lead.sum())


class MatchupModel:
    """logit P(win) = bias[lineup] + sum over brought opponent members of effect[lineup][species] (fitted offline).

    A lineup key is "lead,second,third" (our species ids). Unknown species fall back to the lineup's mean effect.
    `meta` says what the table was fitted from (scripts/sim_matchups.py fit): the policy that played the simulated
    games (policy_sha), the data file and when. A table fitted for another policy is a reference only."""

    def __init__(self, path: Path = MATCHUP_MODEL):
        data = json.loads(path.read_text()) if path.exists() else {"lineups": {}}
        self.lineups: dict[str, dict] = data["lineups"]
        self.meta: dict = data.get("meta", {})

    def check(self, policy: str | Path | None) -> str | None:
        """Why the table may not fit the policy that will play (None when it does, or there is no policy)."""
        from pokesim.runtime.bundle import file_hash

        if not policy or not self.lineups:
            return None
        sha = file_hash(policy)
        fitted = self.meta.get("policy_sha")
        if fitted is None:
            return "fitted before policy versions were recorded: its win chances are for an unknown policy"
        if fitted != sha:
            return f"fitted for policy {fitted}, playing {sha}: its win chances are a reference only"
        return None

    def win_prob(self, lineup: str, picks: OpponentPicks) -> float | None:
        m = self.lineups.get(lineup)
        if m is None:
            return None
        effects = m["effects"]
        default = float(np.mean(list(effects.values()))) if effects else 0.0
        logit = m["bias"] + sum(p * effects.get(to_id(s), default) for s, p in zip(picks.species, picks.brought))
        return float(1 / (1 + np.exp(-logit)))

    def rank(self, roster: list[str], picks: OpponentPicks) -> list[tuple[list[str], float]]:
        out = []
        for trio in itertools.permutations(range(len(roster)), 3):
            if list(trio[1:]) != sorted(trio[1:]):  # order of the back two does not matter
                continue
            names = [roster[i] for i in trio]
            p = self.win_prob(",".join(to_id(n) for n in names), picks)
            if p is not None:
                out.append((names, p))
        return sorted(out, key=lambda kv: -kv[1])


def main() -> None:
    if sys.argv[1:] == ["pick-rates"]:
        out = build_pick_rates()
        top = sorted(out["species"].items(), key=lambda kv: -kv[1]["shown"])[:15]
        mean = out["mean"]
        print(f"{len(out['species'])} species; mean brought {mean['brought']:.2f}, lead {mean['lead']:.2f}")
        for sid, r in top:
            print(f"  {sid:18s} shown {r['shown']:4d}  brought {r['brought']:.2f}  lead {r['lead']:.2f}")


if __name__ == "__main__":
    main()
